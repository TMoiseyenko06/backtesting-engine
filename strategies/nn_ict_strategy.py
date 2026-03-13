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

  This gives the model context from micro structure (5m) all the way up to
  the macro session trend (4h) without any lookahead.

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


class NNICTStrategy(Strategy):
    """
    Parameters
    ----------
    model_path : str | Path | None
        Path to a saved model state dict (.pt file).  If None, a randomly
        initialised model is used (useful for testing the pipeline).
    seq_len : int
        Number of bars to feed into the LSTM per inference.
    min_confidence : float
        Minimum softmax probability required to act on a signal (0–1).
    contracts : int
        Number of contracts per trade.
    device : str | None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to auto-detect.
    """

    name = "NN-ICT Strategy"

    # 4h context needs ~10 complete 4h bars = 480 base bars + seq_len headroom
    _MIN_BUFFER = 600

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        seq_len: int = 30,
        min_confidence: float = 0.40,
        contracts: int = 1,
        device: str | None = None,
    ) -> None:
        self.seq_len        = seq_len
        self.min_confidence = min_confidence
        self.contracts      = contracts
        self.device         = device if device is not None else select_device()

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
        buf_size = max(self._MIN_BUFFER, seq_len + self._MIN_BUFFER)
        self._bar_buffer: deque[Bar] = deque(maxlen=buf_size)

        # Active stop / TP levels (set by the NN at trade entry)
        self._stop_price: Optional[float] = None
        self._tp_price:   Optional[float] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        self._bar_buffer.append(bar)
        bars: List[Bar] = list(self._bar_buffer)

        # Need enough history for the 4h aggregation (48 bars per candle,
        # ~10 complete candles for ICT lookbacks) plus the LSTM sequence window
        if len(bars) < self._MIN_BUFFER:
            return

        sym = bar.symbol

        # --- Manage open stops / take profits ---
        pos = self.position(sym)
        if pos != 0:
            self._check_exit_levels(bar, sym, pos)
            if self.position(sym) == 0:
                return   # just exited, wait for next bar

        # --- Run inference (direction + SL/TP multiples) ---
        signal, confidence, sl_mult, tp_mult = self._infer(bars)

        # --- Act on signal ---
        if signal == 2 and confidence >= self.min_confidence:   # Buy
            if pos <= 0:
                if pos < 0:
                    self.close_position(sym)
                self._enter_long(bar, bars, sl_mult, tp_mult)

        elif signal == 0 and confidence >= self.min_confidence:  # Sell
            if pos >= 0:
                if pos > 0:
                    self.close_position(sym)
                self._enter_short(bar, bars, sl_mult, tp_mult)

        # signal == 1 (Flat): do nothing

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, bars: List[Bar]) -> tuple[int, float, float, float]:
        """
        Returns (predicted_class, confidence, sl_atr_mult, tp_atr_mult).
        Flat signal returns default multiples that are never used.
        """
        feat_matrix = self._engineer.transform(bars)       # (n_bars, n_feat)
        seq = feat_matrix[-self.seq_len:]                   # last seq_len rows
        if len(seq) < self.seq_len:
            return 1, 0.0, 1.5, 2.5   # Flat / not enough data

        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, seq, feat)
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
        return np.mean(trs) if trs else bars[-1].close * 0.001

    def _enter_long(
        self, bar: Bar, bars: List[Bar], sl_mult: float, tp_mult: float
    ) -> None:
        atr = self._atr_now(bars)
        self._stop_price = bar.close - sl_mult * atr
        self._tp_price   = bar.close + tp_mult * atr
        self.buy(bar.symbol, self.contracts)

    def _enter_short(
        self, bar: Bar, bars: List[Bar], sl_mult: float, tp_mult: float
    ) -> None:
        atr = self._atr_now(bars)
        self._stop_price = bar.close + sl_mult * atr
        self._tp_price   = bar.close - tp_mult * atr
        self.sell(bar.symbol, self.contracts)

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
