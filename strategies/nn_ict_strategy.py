"""
NNICTStrategy — Neural Network ICT Trading Strategy
-----------------------------------------------------
Uses the trained LSTMSignalModel to generate Buy/Sell/Flat signals and to
set per-trade stop loss and take profit levels.

Multi-timeframe input
~~~~~~~~~~~~~~~~~~~~~
  The model sees 72 features per bar — ICT features computed at four
  timeframes stacked together:
    5m  (18 feats) : base execution timeframe
    15m (18 feats) : last complete 15m candle (3 × 5m)
    1h  (18 feats) : last complete 1h candle  (12 × 5m)
    4h  (18 feats) : last complete 4h candle  (48 × 5m)

Prop firm risk rules (when prop_rules is provided)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  • Max 4 contracts per position
  • Max daily loss $1,500 — no new entries today once hit
  • Max trailing drawdown $2,500 — halt trading permanently once hit
  • RTH only — force-close all positions by 3:55 PM ET; no entries
    outside 9:30 AM – 3:55 PM ET (no overnight holding)

SL/TP — set by the neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The regression head predicts [sl_atr_mult, tp_atr_mult] at trade entry.
  For LONG:  stop = close − sl_mult × ATR,  tp = close + tp_mult × ATR
  For SHORT: stop = close + sl_mult × ATR,  tp = close − tp_mult × ATR

Confidence filter
~~~~~~~~~~~~~~~~~
  Only act if softmax confidence ≥ min_confidence (default 0.40).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from backtesting.data_feed import Bar
from backtesting.strategy import Strategy
from backtesting.ml.features import MultiTimeframeFeatureEngineer
from backtesting.ml.model import LSTMSignalModel
from backtesting.ml.trainer import select_device
from strategies.prop_firm_risk import PropFirmRisk


class NNICTStrategy(Strategy):
    """
    Parameters
    ----------
    model_path : str | Path | None
        Path to a saved model state dict (.pt file).  If None a randomly
        initialised model is used (useful for pipeline testing).
    seq_len : int
        Number of bars to feed into the LSTM per inference.
    min_confidence : float
        Minimum softmax probability required to act on a signal (0–1).
    contracts : int
        Base number of contracts per trade (capped by prop_rules.max_contracts).
    device : str | None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to auto-detect.
    prop_rules : PropFirmRisk | None
        If provided, enforces prop-firm session hours, daily loss limit,
        max drawdown, and contract cap on every bar.  Pass ``None`` to
        disable all rule enforcement (for research/debugging only).
    """

    name = "NN-ICT Strategy"

    # 4h context needs ~10 complete 4h bars = 2,400 1m bars + seq headroom
    _MIN_BUFFER = 2500

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        seq_len: int = 30,
        min_confidence: float = 0.40,
        contracts: int = 1,
        device: str | None = None,
        prop_rules: Optional[PropFirmRisk] = None,
    ) -> None:
        self.seq_len        = seq_len
        self.min_confidence = min_confidence
        self.contracts      = contracts
        self.device         = device if device is not None else select_device()
        self.prop_rules     = prop_rules

        # Load model — auto-detects n_features from checkpoint weights
        if model_path and Path(model_path).exists():
            try:
                self._model = LSTMSignalModel.from_checkpoint(model_path, self.device)
            except Exception as e:
                print(
                    f"  [NNICTStrategy] Warning: could not load {model_path} ({e}). "
                    f"Using random init. Re-run train_nn.py to rebuild checkpoint."
                )
                self._model = LSTMSignalModel()
                self._model.to(self.device)
        else:
            self._model = LSTMSignalModel()
            self._model.to(self.device)

        self._engineer = MultiTimeframeFeatureEngineer()

        # Buffer must hold enough history for 4h aggregation + seq window
        self._bar_buffer: deque[Bar] = deque(maxlen=max(self._MIN_BUFFER, seq_len + self._MIN_BUFFER))

        # Active stop / TP levels (set by the NN at trade entry)
        self._stop_price: Optional[float] = None
        self._tp_price:   Optional[float] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        self._bar_buffer.append(bar)
        bars: List[Bar] = list(self._bar_buffer)
        sym = bar.symbol

        # ── Prop firm: update risk state ────────────────────────────────
        if self.prop_rules is not None:
            self.prop_rules.update(bar.timestamp, self.equity())

        # ── Force-close: outside session or prop firm halted ────────────
        pos = self.position(sym)
        if pos != 0:
            if self._must_close(bar):
                self.close_position(sym)
                self._reset_levels()
                return
            # Manage running SL/TP
            self._check_exit_levels(bar, sym, pos)
            if self.position(sym) == 0:
                return   # just hit SL or TP — wait for next bar

        # ── Warmup: need enough history for 4h aggregation ──────────────
        if len(bars) < self._MIN_BUFFER:
            return

        # ── Gate: prop firm rules block entry ───────────────────────────
        if not self._can_enter(bar):
            return

        # ── Run inference ────────────────────────────────────────────────
        signal, confidence, sl_mult, tp_mult = self._infer(bars)

        # ── Act on signal ────────────────────────────────────────────────
        n_contracts = self._contracts_to_trade()

        if signal == 2 and confidence >= self.min_confidence:   # Buy
            if pos <= 0:
                if pos < 0:
                    self.close_position(sym)
                self._enter_long(bar, bars, sl_mult, tp_mult, n_contracts)

        elif signal == 0 and confidence >= self.min_confidence:  # Sell
            if pos >= 0:
                if pos > 0:
                    self.close_position(sym)
                self._enter_short(bar, bars, sl_mult, tp_mult, n_contracts)

        # signal == 1 (Flat): do nothing

    # ------------------------------------------------------------------
    # Prop firm helpers
    # ------------------------------------------------------------------

    def _must_close(self, bar: Bar) -> bool:
        """Should we force-close because of session end or halt?"""
        if self.prop_rules is None:
            return False
        if self.prop_rules.halted:
            return True
        return self.prop_rules.should_close(bar.timestamp)

    def _can_enter(self, bar: Bar) -> bool:
        """All prop firm checks must pass before a new entry is allowed."""
        if self.prop_rules is None:
            return True
        # No new entries if halted
        if self.prop_rules.halted:
            return False
        # Must be inside RTH session AND before entry cutoff
        if not self.prop_rules.is_in_session(bar.timestamp):
            return False
        if self.prop_rules.should_close(bar.timestamp):
            return False
        # Loss limits
        return self.prop_rules.can_enter(self.equity())

    def _contracts_to_trade(self) -> int:
        """Contracts for next trade, capped by prop firm limit."""
        n = self.contracts
        if self.prop_rules is not None:
            n = min(n, self.prop_rules.max_contracts)
        return max(1, n)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, bars: List[Bar]) -> tuple[int, float, float, float]:
        """Returns (predicted_class, confidence, sl_atr_mult, tp_atr_mult)."""
        feat_matrix = self._engineer.transform(bars)

        # Append drawdown_frac column — must match the feature added during training.
        # Value = fraction of max_drawdown consumed (0=none, 1=limit hit).
        if self.prop_rules is not None:
            eq = self.equity()
            dd_frac = float(np.clip(
                (self.prop_rules._peak_equity - eq) / self.prop_rules.max_drawdown,
                0.0, 2.0,
            ))
        else:
            dd_frac = 0.0
        dd_col = np.full((len(feat_matrix), 1), dd_frac, dtype=np.float32)
        feat_matrix = np.column_stack([feat_matrix, dd_col])

        seq = feat_matrix[-self.seq_len:]
        if len(seq) < self.seq_len:
            return 1, 0.0, 1.5, 2.5   # Flat / insufficient data

        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        return self._model.predict_with_sl_tp(x)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _atr_now(self, bars: List[Bar]) -> float:
        """Quick ATR estimate from last 14 bars."""
        n = min(14, len(bars))
        recent = bars[-n:]
        trs = []
        for i in range(1, len(recent)):
            trs.append(max(
                recent[i].high - recent[i].low,
                abs(recent[i].high - recent[i - 1].close),
                abs(recent[i].low  - recent[i - 1].close),
            ))
        return float(np.mean(trs)) if trs else bars[-1].close * 0.001

    def _enter_long(
        self, bar: Bar, bars: List[Bar],
        sl_mult: float, tp_mult: float, n_contracts: int,
    ) -> None:
        atr = self._atr_now(bars)
        self._stop_price = bar.close - sl_mult * atr
        self._tp_price   = bar.close + tp_mult * atr
        self.buy(bar.symbol, n_contracts)

    def _enter_short(
        self, bar: Bar, bars: List[Bar],
        sl_mult: float, tp_mult: float, n_contracts: int,
    ) -> None:
        atr = self._atr_now(bars)
        self._stop_price = bar.close + sl_mult * atr
        self._tp_price   = bar.close - tp_mult * atr
        self.sell(bar.symbol, n_contracts)

    def _check_exit_levels(self, bar: Bar, sym: str, pos: int) -> None:
        if self._stop_price is None or self._tp_price is None:
            return
        if pos > 0:
            if bar.low <= self._stop_price or bar.high >= self._tp_price:
                self.close_position(sym)
                self._reset_levels()
        elif pos < 0:
            if bar.high >= self._stop_price or bar.low <= self._tp_price:
                self.close_position(sym)
                self._reset_levels()

    def _reset_levels(self) -> None:
        self._stop_price = None
        self._tp_price   = None
