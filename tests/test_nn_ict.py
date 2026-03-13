"""
Tests for the NN-ICT pipeline.
No actual training is done — tests are fast and offline.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from backtesting.data_feed import Bar, DataFeed
from backtesting.ml.dataset import SequenceDataset, make_labels, walk_forward_splits
from backtesting.ml.features import (
    ICTFeatureEngineer, N_FEATURES, FEATURE_NAMES,
    MultiTimeframeFeatureEngineer, N_MTF_FEATURES, aggregate_bars,
)
from backtesting.ml.model import LSTMSignalModel
from backtesting.ml.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n=200, interval_mins=5, seed=42) -> list[Bar]:
    rng = np.random.default_rng(seed)
    base  = datetime(2023, 6, 1, 9, 30)
    price = 17_500.0
    bars  = []
    for i in range(n):
        chg   = rng.normal(0, 20)
        open_ = price
        close = open_ + chg
        high  = max(open_, close) + abs(rng.normal(0, 5))
        low   = min(open_, close) - abs(rng.normal(0, 5))
        bars.append(Bar(
            timestamp=base + timedelta(minutes=i * interval_mins),
            symbol="NQ=F",
            open=round(open_, 2), high=round(high, 2),
            low=round(low, 2),   close=round(close, 2),
            volume=float(rng.integers(500, 5000)),
            contract_multiplier=20.0,
        ))
        price = close
    return bars


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestAggregateBars:

    def test_basic_grouping(self):
        bars = _make_bars(120)
        agg  = aggregate_bars(bars, 3)
        assert len(agg) == 40

    def test_ohlcv_correctness(self):
        bars = _make_bars(12)
        agg  = aggregate_bars(bars, 3)
        # First aggregated bar covers bars[0..2]
        assert agg[0].open  == bars[0].open
        assert agg[0].close == bars[2].close
        assert agg[0].high  == max(b.high for b in bars[:3])
        assert agg[0].low   == min(b.low  for b in bars[:3])

    def test_trailing_bars_dropped(self):
        bars = _make_bars(100)
        agg  = aggregate_bars(bars, 7)
        assert len(agg) == 14   # 100 // 7 = 14


class TestMultiTimeframeFeatureEngineer:

    def test_output_shape(self):
        bars = _make_bars(200)
        eng  = MultiTimeframeFeatureEngineer()
        feat = eng.transform(bars)
        assert feat.shape == (200, N_MTF_FEATURES)

    def test_n_features_property(self):
        eng = MultiTimeframeFeatureEngineer()
        assert eng.n_features == N_MTF_FEATURES

    def test_no_nans(self):
        bars = _make_bars(200)
        feat = MultiTimeframeFeatureEngineer().transform(bars)
        assert not np.isnan(feat).any()

    def test_no_lookahead_early_bars(self):
        # Base is 1m; first complete 4h candle needs 240 bars.
        # Rows 0..238 must have all-zero 4h block (last 18 columns).
        bars = _make_bars(300)
        feat = MultiTimeframeFeatureEngineer().transform(bars)
        htf_4h = feat[:239, -18:]
        assert (htf_4h == 0).all(), "4h features should be zeros before first complete 4h bar"


class TestICTFeatureEngineer:

    def test_output_shape(self):
        bars = _make_bars(150)
        eng  = ICTFeatureEngineer()
        feat = eng.transform(bars)
        assert feat.shape == (150, N_FEATURES)

    def test_no_nans_after_transform(self):
        bars = _make_bars(150)
        feat = ICTFeatureEngineer().transform(bars)
        assert not np.isnan(feat).any(), "Feature matrix contains NaN"

    def test_feature_count_matches_names(self):
        assert N_FEATURES == len(FEATURE_NAMES)

    def test_premium_discount_range(self):
        bars = _make_bars(100)
        feat = ICTFeatureEngineer().transform(bars)
        pd_col = FEATURE_NAMES.index("premium_discount")
        vals = feat[:, pd_col]
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0

    def test_kill_zone_flags_binary(self):
        bars = _make_bars(100)
        feat = ICTFeatureEngineer().transform(bars)
        for col_name in ("kill_zone_london", "kill_zone_ny"):
            col = FEATURE_NAMES.index(col_name)
            unique = set(feat[:, col].tolist())
            assert unique.issubset({0.0, 1.0}), f"{col_name} has non-binary values"

    def test_market_structure_values(self):
        bars = _make_bars(150)
        feat = ICTFeatureEngineer().transform(bars)
        ms_col = FEATURE_NAMES.index("market_structure")
        unique = set(feat[:, ms_col].tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_float32_dtype(self):
        feat = ICTFeatureEngineer().transform(_make_bars(60))
        assert feat.dtype == np.float32

    def test_single_bar_does_not_crash(self):
        feat = ICTFeatureEngineer().transform(_make_bars(1))
        assert feat.shape == (1, N_FEATURES)


# ---------------------------------------------------------------------------
# Label tests
# ---------------------------------------------------------------------------

class TestMakeLabels:

    def _make_arrays(self, n, start=100, end=110):
        closes = np.linspace(start, end, n)
        highs  = closes + 0.5
        lows   = closes - 0.5
        return closes, highs, lows

    def test_output_shape(self):
        closes, highs, lows = self._make_arrays(200)
        atr    = np.full(200, 1.0)
        labels, sl_tp = make_labels(closes, highs, lows, atr, horizon=10)
        assert labels.shape == (200,)
        assert sl_tp.shape  == (200, 2)

    def test_labels_are_0_1_2(self):
        closes, highs, lows = self._make_arrays(200)
        atr    = np.full(200, 0.5)
        labels, _ = make_labels(closes, highs, lows, atr, horizon=5)
        assert set(labels.tolist()).issubset({0, 1, 2})

    def test_last_horizon_bars_are_flat(self):
        closes, highs, lows = self._make_arrays(100, 100, 200)
        atr    = np.full(100, 1.0)
        labels, _ = make_labels(closes, highs, lows, atr, horizon=10)
        assert (labels[-10:] == 1).all()

    def test_trending_up_produces_buy_labels(self):
        # Strong uptrend → most labels should be Buy
        closes = np.linspace(100, 300, 300)   # large, clean trend
        highs  = closes + 0.5
        lows   = closes - 0.5
        atr    = np.full(300, 0.1)            # tiny ATR → z large
        labels, _ = make_labels(closes, highs, lows, atr, horizon=10, threshold=0.5)
        buy_pct = (labels == 2).mean()
        assert buy_pct > 0.5, f"Expected mostly Buy labels in uptrend, got {buy_pct:.2f}"

    def test_sl_tp_targets_positive(self):
        closes, highs, lows = self._make_arrays(200, 100, 300)
        atr    = np.full(200, 1.0)
        _, sl_tp = make_labels(closes, highs, lows, atr, horizon=10)
        assert (sl_tp > 0).all(), "All SL/TP targets must be positive"


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestSequenceDataset:

    def _make_sl_tp(self, n):
        return np.full((n, 2), [1.5, 2.5], dtype=np.float32)

    def test_length(self):
        features = np.random.rand(200, N_MTF_FEATURES).astype(np.float32)
        labels   = np.ones(200, dtype=np.int8)
        sl_tp    = self._make_sl_tp(200)
        ds       = SequenceDataset(features, labels, sl_tp, seq_len=20)
        # valid positions: seq_len-1 .. 199  → 200 - 20 + 1 = 181
        assert len(ds) == 181

    def test_item_shapes(self):
        features = np.random.rand(100, N_MTF_FEATURES).astype(np.float32)
        labels   = np.zeros(100, dtype=np.int8)
        sl_tp    = self._make_sl_tp(100)
        ds       = SequenceDataset(features, labels, sl_tp, seq_len=15)
        x, y, y_sltp = ds[0]
        assert x.shape      == (15, N_MTF_FEATURES)
        assert y.shape      == ()
        assert y_sltp.shape == (2,)

    def test_custom_indices(self):
        features = np.random.rand(200, N_MTF_FEATURES).astype(np.float32)
        labels   = np.ones(200, dtype=np.int8)
        sl_tp    = self._make_sl_tp(200)
        indices  = list(range(100, 150))
        ds       = SequenceDataset(features, labels, sl_tp, seq_len=10, indices=indices)
        assert len(ds) == len([i for i in indices if i >= 9])


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

class TestWalkForwardSplits:

    def test_no_overlap(self):
        for train_idx, val_idx in walk_forward_splits(1000, 600, 100, 100):
            assert max(train_idx) < min(val_idx)

    def test_expanding_train(self):
        sizes = [len(t) for t, _ in walk_forward_splits(1000, 600, 100, 100)]
        assert sizes == sorted(sizes), "Train sizes should be non-decreasing"

    def test_val_size(self):
        for _, val_idx in walk_forward_splits(1000, 600, 100, 100):
            assert len(val_idx) == 100

    def test_no_folds_if_insufficient(self):
        folds = list(walk_forward_splits(500, 600, 100, 100))
        assert folds == []

    def test_yields_at_least_one_fold(self):
        folds = list(walk_forward_splits(1000, 600, 100, 100))
        assert len(folds) >= 1


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestLSTMSignalModel:

    def test_forward_shape(self):
        model = LSTMSignalModel()   # defaults to N_MTF_FEATURES=90
        x = torch.randn(4, 30, N_MTF_FEATURES)
        logits, sl_tp = model(x)
        assert logits.shape == (4, 3)
        assert sl_tp.shape  == (4, 2)
        assert (sl_tp > 0).all()

    def test_predict_with_sl_tp(self):
        model = LSTMSignalModel()
        x = torch.randn(1, 30, N_MTF_FEATURES)
        cls, conf, sl, tp = model.predict_with_sl_tp(x)
        assert cls in (0, 1, 2)
        assert 0.0 <= conf <= 1.0
        assert sl > 0
        assert tp > 0

    def test_predict_proba_sums_to_one(self):
        model = LSTMSignalModel()
        x = torch.randn(2, 30, N_MTF_FEATURES)
        probs = model.predict_proba(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_returns_valid_class(self):
        model = LSTMSignalModel()
        x = torch.randn(1, 30, N_MTF_FEATURES)
        pred = model.predict(x)
        assert pred in (0, 1, 2)

    def test_parameter_count_is_reasonable(self):
        model = LSTMSignalModel()
        # 90-feature × 128-hidden LSTM — larger than single-TF but still bounded
        assert model.n_parameters < 1_000_000

    def test_save_and_load_via_from_checkpoint(self):
        model = LSTMSignalModel()
        x = torch.randn(1, 30, N_MTF_FEATURES)
        original_pred = model.predict(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            # from_checkpoint auto-detects n_features
            loaded = LSTMSignalModel.from_checkpoint(path, device="cpu")
            loaded_pred = loaded.predict(x)

        assert original_pred == loaded_pred


# ---------------------------------------------------------------------------
# Trainer (quick smoke test — 2 epochs only)
# ---------------------------------------------------------------------------

class TestTrainer:

    def _make_data(self, n=300):
        bars     = _make_bars(n)
        eng      = MultiTimeframeFeatureEngineer()
        features = eng.transform(bars)
        closes   = np.array([b.close for b in bars])
        highs    = np.array([b.high  for b in bars])
        lows     = np.array([b.low   for b in bars])
        atr      = ICTFeatureEngineer._atr(highs, lows, closes, 14)
        labels, sl_tp_targets = make_labels(closes, highs, lows, atr,
                                            horizon=5, threshold=0.5)
        return features, labels, sl_tp_targets

    def test_fit_simple_runs(self):
        features, labels, sl_tp = self._make_data(300)
        model   = LSTMSignalModel(n_features=features.shape[1])
        trainer = Trainer(model, seq_len=20, batch_size=32, epochs=2, patience=2,
                          device="cpu")
        result  = trainer.fit(features, labels, sl_tp, val_split=0.2)
        assert "val_loss" in result
        assert "val_acc"  in result
        assert 0.0 <= result["val_acc"] <= 1.0

    def test_walk_forward_runs(self):
        features, labels, sl_tp = self._make_data(500)
        model   = LSTMSignalModel(n_features=features.shape[1])
        trainer = Trainer(model, seq_len=20, batch_size=32, epochs=2, patience=2,
                          device="cpu")
        metrics = trainer.fit_walk_forward(
            features, labels, sl_tp,
            train_bars=200, val_bars=100, step_bars=100,
        )
        assert len(metrics) >= 1
        for m in metrics:
            assert 0.0 <= m["val_acc"] <= 1.0

    def test_trainer_save_load(self):
        features, labels, sl_tp = self._make_data(300)
        model   = LSTMSignalModel(n_features=features.shape[1])
        trainer = Trainer(model, seq_len=20, batch_size=32, epochs=2, patience=2,
                          device="cpu")
        trainer.fit(features, labels, sl_tp)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            trainer.save(path)
            assert path.exists()
            loaded = LSTMSignalModel.from_checkpoint(path, device="cpu")
            assert loaded is not None


# ---------------------------------------------------------------------------
# Strategy integration test
# ---------------------------------------------------------------------------

class TestNNICTStrategy:

    def test_strategy_runs_on_feed(self):
        from backtesting.engine import BacktestEngine
        from backtesting.portfolio import Portfolio, MarginSpec
        from strategies.nn_ict_strategy import NNICTStrategy

        bars     = _make_bars(300)
        feed     = DataFeed(bars, warmup_bars=50)
        strategy = NNICTStrategy(model_path=None, seq_len=20, contracts=1)
        portfolio = Portfolio(
            initial_cash=500_000,
            margin_specs={"NQ=F": MarginSpec("NQ=F", 21_000, 19_000, 20.0)},
            commission_per_contract=2.0,
            slippage_ticks=1,
            tick_size=0.25,
        )
        result = BacktestEngine(feed, portfolio, [strategy], verbose=False).run()
        # Just verify it completed without crashing and analytics exist
        assert hasattr(result, "analytics")
        assert result.analytics.total_trades >= 0
