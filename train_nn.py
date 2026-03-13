"""
Neural Network ICT Strategy -- Training Script
===============================================

What this script does
---------------------
1. Loads bar data (cache -> Yahoo Finance -> synthetic fallback)
2. Engineers ICT features (FVG, order blocks, swing levels, kill zones ...)
3. Generates ATR-normalised labels (Buy / Flat / Sell)
4. Trains the LSTM via walk-forward cross-validation
5. Saves the best model weights to  models/nn_ict_<interval>.pt
6. Saves full training metrics to   models/nn_ict_<interval>_metrics.json
7. Runs a hold-out backtest and prints results

Walk-forward CV explained
--------------------------
Instead of a random train/test split (which leaks future prices into training),
the data is split chronologically:

    |--- train fold 0 ---|--- val fold 0 ---|
    |------ train fold 1 ------|--- val fold 1 ---|
    ...

Each validation fold is strictly in the future relative to its training fold.
If val accuracy is consistently close to train accuracy, the model generalises.
If train_acc >> val_acc, the model is overfitting -- increase dropout / reduce
hidden_size / reduce seq_len.

Saved files
-----------
  models/nn_ict_5m.pt              -- PyTorch model weights (reloadable)
  models/nn_ict_5m_metrics.json    -- per-fold metrics + config + backtest result

Run:
    python train_nn.py                        # 5m bars, walk-forward CV
    python train_nn.py --interval 1h          # hourly bars
    python train_nn.py --no-wf                # simple 80/20 split (faster)
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from backtesting.data_feed import Bar, DataFeed
from backtesting.engine import BacktestEngine
from backtesting.loaders.cache import save_feed, load_feed_from_cache
from backtesting.ml.dataset import make_labels
from backtesting.ml.trainer import select_device
from backtesting.ml.features import ICTFeatureEngineer, MultiTimeframeFeatureEngineer
from backtesting.ml.model import LSTMSignalModel
from backtesting.ml.trainer import Trainer
from backtesting.portfolio import Portfolio, MarginSpec
from strategies.nn_ict_strategy import NNICTStrategy
from strategies.prop_firm_risk import PropFirmRisk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL     = "NQ=F"
MULTIPLIER = 20.0

# Prop firm account settings
CASH         = 50_000.0    # starting capital
MAX_CONTRACTS = 4           # hard position cap (prop firm rule)
MAX_DAILY_LOSS    = 1_500.0 # stop trading today if daily loss reaches this
MAX_DRAWDOWN_USD  = 2_500.0 # halt account if trailing drawdown reaches this

# Reduced margin to allow up to MAX_CONTRACTS on a $50k account
# ($50k / 4 = $12,500 per contract available for initial margin)
INIT_MARGIN  = 12_000.0
MAINT_MARGIN = 10_000.0
TICK_SIZE    = 0.25

CACHE_DIR  = Path("data")
MODEL_DIR  = Path("models")

# Training hyper-parameters
# Base timeframe is 1m, so all bar counts are in 1-minute units:
#   SEQ_LEN=60  → 60 minutes of context per inference
#   HORIZON=30  → label looks 30 minutes ahead
SEQ_LEN      = 60      # bars per LSTM input window  (60 min)
HORIZON      = 30      # bars forward for label       (30 min)
THRESHOLD    = 0.5     # ATR-normalised return needed to label Buy/Sell
HIDDEN_SIZE  = 128     # 5-TF × 18 = 90 input features
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 0.05
DROPOUT_LSTM = 0.3
DROPOUT_FC   = 0.4
BATCH_SIZE   = 64
PATIENCE     = 10      # early stopping patience

# Walk-forward params  (1m bars — ~1 trading week = ~2,400 bars)
# Sized to allow ~5+ folds on a typical 5–10k bar download.
# Increase WF_TRAIN_BARS if you have 20k+ bars.
WF_TRAIN_BARS = 2500   # initial training window (~1 week of 1m bars)
WF_VAL_BARS   = 500    # validation window per fold
WF_STEP_BARS  = 500    # advance per fold


# ---------------------------------------------------------------------------
# Synthetic data (used if yfinance download fails)
# ---------------------------------------------------------------------------

def make_synthetic_bars(n: int, interval: str) -> list[Bar]:
    random.seed(42)
    np.random.seed(42)
    base  = datetime(2023, 1, 2, 9, 30)
    vol   = {"1m": 8, "5m": 18, "15m": 30, "1h": 60}.get(interval, 40)
    mins  = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(interval, 5)
    bars, price = [], 17_500.0
    for i in range(n):
        chg   = random.gauss(0, vol)
        open_ = price
        close = open_ + chg
        high  = max(open_, close) + abs(random.gauss(0, vol * 0.3))
        low   = min(open_, close) - abs(random.gauss(0, vol * 0.3))
        bars.append(Bar(
            timestamp=base + timedelta(minutes=i * mins),
            symbol=SYMBOL,
            open=round(open_, 2), high=round(high, 2),
            low=round(low, 2),   close=round(close, 2),
            volume=round(random.uniform(500, 5_000)),
            contract_multiplier=MULTIPLIER,
        ))
        price = close
    return bars


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(interval: str) -> list[Bar]:
    cache = CACHE_DIR / f"{SYMBOL.replace('=', '')}_{interval}.parquet"
    try:
        feed = load_feed_from_cache(cache, symbol=SYMBOL,
                                    contract_multiplier=MULTIPLIER)
        print(f"  Loaded {feed._total} bars from cache: {cache}")
        return feed._bars
    except FileNotFoundError:
        pass

    try:
        print(f"  Downloading {interval} bars from Yahoo Finance ...")
        feed = DataFeed.from_yfinance(SYMBOL, interval=interval,
                                      contract_multiplier=MULTIPLIER)
        save_feed(feed, cache)
        return feed._bars
    except Exception as e:
        n = {"1m": 15000, "5m": 5000, "15m": 3000, "1h": 1500}.get(interval, 5000)
        print(f"  Download failed ({e}). Using {n} synthetic bars.")
        return make_synthetic_bars(n, interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1m",
                        choices=["1m", "5m", "15m", "1h"])
    parser.add_argument("--no-wf", action="store_true",
                        help="Use simple 80/20 split instead of walk-forward CV")
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda', 'mps', 'cpu', or omit to auto-detect."
    )
    args = parser.parse_args()

    interval = args.interval
    MODEL_DIR.mkdir(exist_ok=True)
    model_path   = MODEL_DIR / f"nn_ict_{interval}.pt"
    metrics_path = MODEL_DIR / f"nn_ict_{interval}_metrics.json"

    print(f"\n{'='*60}")
    print(f"  NNICTStrategy training  ({interval})")
    print(f"{'='*60}")

    # 1. Load data
    bars = load_bars(interval)
    n = len(bars)
    print(f"  Bars available: {n}")

    if n < WF_TRAIN_BARS + WF_VAL_BARS:
        print(f"  WARNING: only {n} bars -- walk-forward needs "
              f"{WF_TRAIN_BARS + WF_VAL_BARS}. Switching to simple split.")
        args.no_wf = True

    # 2. Feature engineering — 1m + 5m + 15m + 1h + 4h stacked
    print("  Computing multi-timeframe ICT features (1m / 5m / 15m / 1h / 4h) ...")
    engineer = MultiTimeframeFeatureEngineer()
    features = engineer.transform(bars)
    print(f"  Feature matrix: {features.shape}  ({features.shape[1]} features)")

    # 3. Labels + SL/TP regression targets
    closes = np.array([b.close for b in bars])
    highs  = np.array([b.high  for b in bars])
    lows   = np.array([b.low   for b in bars])
    atr    = ICTFeatureEngineer._atr(highs, lows, closes, 14)
    labels, sl_tp_targets = make_labels(
        closes, highs, lows, atr, horizon=HORIZON, threshold=THRESHOLD
    )

    buy_pct  = float((labels == 2).mean() * 100)
    sell_pct = float((labels == 0).mean() * 100)
    flat_pct = float((labels == 1).mean() * 100)
    print(f"  Label distribution -- Buy:{buy_pct:.1f}%  Sell:{sell_pct:.1f}%  Flat:{flat_pct:.1f}%")

    # 4. Model
    torch.manual_seed(42)
    model = LSTMSignalModel(
        n_features=engineer.n_features,
        hidden_size=HIDDEN_SIZE,
        lstm_dropout=DROPOUT_LSTM,
        fc_dropout=DROPOUT_FC,
    )
    print(f"  Model parameters: {model.n_parameters:,}  (n_features={engineer.n_features})")

    device = args.device  # None → auto-detect inside Trainer
    trainer = Trainer(
        model=model,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        device=device,
    )

    # Build the run log that will be saved to JSON
    run_log: dict = {
        "interval":   interval,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": str(model_path),
        "n_bars":     n,
        "n_features":    int(features.shape[1]),
        "timeframes":    ["1m", "5m", "15m", "1h", "4h"],
        "label_pct":  {"buy": round(buy_pct, 2),
                       "sell": round(sell_pct, 2),
                       "flat": round(flat_pct, 2)},
        "hyperparams": {
            "seq_len":      SEQ_LEN,
            "horizon":      HORIZON,
            "threshold":    THRESHOLD,
            "hidden_size":  HIDDEN_SIZE,
            "epochs":       EPOCHS,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size":   BATCH_SIZE,
            "patience":     PATIENCE,
        },
    }

    # 5. Train
    print()
    if args.no_wf:
        print("  Training (simple 80/20 split) ...")
        metrics = trainer.fit(features, labels, sl_tp_targets, val_split=0.2)
        print(f"  val_loss={metrics['val_loss']:.4f}  val_acc={metrics['val_acc']:.3f}")
        run_log["mode"]     = "simple_split"
        run_log["val_loss"] = metrics["val_loss"]
        run_log["val_acc"]  = metrics["val_acc"]
    else:
        print("  Training (walk-forward CV) ...")
        fold_metrics = trainer.fit_walk_forward(
            features, labels, sl_tp_targets,
            train_bars=WF_TRAIN_BARS,
            val_bars=WF_VAL_BARS,
            step_bars=WF_STEP_BARS,
        )
        if fold_metrics:
            avg_acc  = float(np.mean([f["val_acc"]  for f in fold_metrics]))
            avg_loss = float(np.mean([f["val_loss"] for f in fold_metrics]))
            print(f"\n  Avg val_acc={avg_acc:.3f}  avg_val_loss={avg_loss:.4f}")
            print(
                "\n  NOTE: val_acc ~33% = random (3 classes). "
                "Target >38-42% for edge.\n"
                "  Large gap (train_acc >> val_acc) = overfitting.\n"
                "  Consistent val_acc across folds = good generalisation."
            )
            run_log["mode"]         = "walk_forward"
            run_log["avg_val_acc"]  = round(avg_acc,  4)
            run_log["avg_val_loss"] = round(avg_loss, 4)
            run_log["folds"]        = fold_metrics

    # Always save the model regardless of training mode
    trainer.save(model_path)

    # 6. Hold-out backtest
    print(f"\n  Running backtest with trained model ({interval}) ...")
    prop_rules = PropFirmRisk(
        max_contracts=MAX_CONTRACTS,
        max_daily_loss_usd=MAX_DAILY_LOSS,
        max_drawdown_usd=MAX_DRAWDOWN_USD,
        initial_equity=CASH,
    )
    strategy = NNICTStrategy(
        model_path=model_path,
        seq_len=SEQ_LEN,
        min_confidence=0.40,
        contracts=MAX_CONTRACTS,
        prop_rules=prop_rules,
    )
    split_idx = int(n * 0.8)
    # Pre-seed the strategy's internal bar buffer: include the last
    # _MIN_BUFFER bars from training so the warmup check clears immediately.
    ctx_start = max(0, split_idx - NNICTStrategy._MIN_BUFFER)
    test_bars  = bars[ctx_start:]
    n_warmup   = (split_idx - ctx_start) + SEQ_LEN   # these bars don't count for PnL
    if len(test_bars) - n_warmup < 50:
        print("  Insufficient hold-out bars for backtest.")
    else:
        feed = DataFeed(test_bars, warmup_bars=n_warmup)
        portfolio = Portfolio(
            initial_cash=CASH,
            margin_specs={SYMBOL: MarginSpec(SYMBOL, INIT_MARGIN, MAINT_MARGIN, MULTIPLIER)},
            commission_per_contract=2.0,
            slippage_ticks=1,
            tick_size=TICK_SIZE,
        )
        result = BacktestEngine(feed, portfolio, [strategy], verbose=False).run()
        a = result.analytics

        print(f"\n  {'─'*50}")
        print(f"  Hold-out backtest results (last 20% of bars)")
        print(f"  {'─'*50}")
        print(f"  Total return   : {a.total_return_pct:>+8.2f}%")
        print(f"  Sharpe ratio   : {a.sharpe_ratio:>8.3f}")
        print(f"  Max drawdown   : {a.max_drawdown_pct:>8.2f}%")
        print(f"  Total trades   : {a.total_trades:>8}")
        print(f"  Win rate       : {a.win_rate:>8.1f}%")
        print(f"  {'─'*50}\n")
        print(
            "  Realistic expectations:\n"
            "  - Sharpe > 0.5 on unseen data = viable edge\n"
            "  - Sharpe > 1.0 = strong (rare for pure ML strategies)\n"
            "  - Sharpe < 0.3 = no edge, needs more data / better features\n"
            "  - Consistent across multiple hold-out periods = not overfit\n"
        )

        # Add backtest results to the run log
        run_log["backtest"] = {
            "hold_out_pct":     20,
            "n_test_bars":      n - split_idx,
            "total_return_pct": round(a.total_return_pct, 4),
            "sharpe_ratio":     round(a.sharpe_ratio,     4),
            "max_drawdown_pct": round(a.max_drawdown_pct, 4),
            "total_trades":     a.total_trades,
            "win_rate":         round(a.win_rate,         4),
        }

    # 7. Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"  Metrics saved  -> {metrics_path}")
    print(f"  Model saved    -> {model_path}\n")


if __name__ == "__main__":
    main()
