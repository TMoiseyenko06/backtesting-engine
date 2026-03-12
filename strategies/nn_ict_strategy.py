"""
NNICTStrategy — Neural Network ICT Trading Strategy
-----------------------------------------------------
Uses the trained LSTMSignalModel to generate Buy/Sell/Flat signals.

Signal mapping
~~~~~~~~~~~~~~
  Model output 2 (Buy)  → enter long   (or exit short)
  Model output 0 (Sell) → enter short  (or exit long)
  Model output 1 (Flat) → do nothing   (or optionally exit)

Position management
~~~~~~~~~~~~~~~~~~~
  - Max 1 contract long or short at a time
  - ATR-based stop loss (1.5× ATR from entry)
  - ATR-based take profit (2.5× ATR from entry)  — 1:1.67 R:R
  - Exit on opposing signal

Confidence filter
~~~~~~~~~~~~~~~~~
  The model outputs 3 softmax probabilities.  A trade is only entered if the
  winning class probability exceeds `min_confidence` (default 0.50).
  Raising this threshold reduces trade frequency but improves signal quality.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from backtesting.data_feed import Bar
from backtesting.strategy import Strategy
from backtesting.ml.features import ICTFeatureEngineer, N_FEATURES
from backtesting.ml.model import LSTMSignalModel
from backtesting.ml.trainer import Trainer


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
    atr_stop_mult : float
        Stop loss = entry_price ± atr_stop_mult × ATR.
    atr_tp_mult : float
        Take profit = entry_price ± atr_tp_mult × ATR.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    name = "NN-ICT Strategy"

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        seq_len: int = 30,
        min_confidence: float = 0.50,
        contracts: int = 1,
        atr_stop_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
        device: str = "cpu",
    ) -> None:
        self.seq_len        = seq_len
        self.min_confidence = min_confidence
        self.contracts      = contracts
        self.atr_stop_mult  = atr_stop_mult
        self.atr_tp_mult    = atr_tp_mult
        self.device         = device

        self._model = LSTMSignalModel()
        if model_path and Path(model_path).exists():
            Trainer.load_model(model_path, self._model, device)

        self._engineer = ICTFeatureEngineer()

        # Rolling buffer of raw Bar objects for feature computation
        # We keep seq_len + headroom bars to compute ATR etc.
        self._bar_buffer: deque[Bar] = deque(maxlen=seq_len + 100)

        # Active stop / TP levels
        self._stop_price: Optional[float] = None
        self._tp_price:   Optional[float] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        self._bar_buffer.append(bar)
        bars: List[Bar] = list(self._bar_buffer)

        # Need enough history for features + one full sequence
        if len(bars) < self.seq_len + 20:
            return

        sym = bar.symbol

        # --- Manage open stops / take profits ---
        pos = self.position(sym)
        if pos != 0:
            self._check_exit_levels(bar, sym, pos)
            if self.position(sym) == 0:
                return   # just exited, wait for next bar

        # --- Run inference ---
        signal, confidence = self._infer(bars)

        # --- Act on signal ---
        if signal == 2 and confidence >= self.min_confidence:   # Buy
            if pos <= 0:
                if pos < 0:
                    self.close_position(sym)
                self._enter_long(bar)

        elif signal == 0 and confidence >= self.min_confidence:  # Sell
            if pos >= 0:
                if pos > 0:
                    self.close_position(sym)
                self._enter_short(bar)

        # signal == 1 (Flat): do nothing

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, bars: List[Bar]):
        """
        Returns (predicted_class, confidence) where class ∈ {0, 1, 2}.
        """
        feat_matrix = self._engineer.transform(bars)       # (n_bars, n_feat)
        seq = feat_matrix[-self.seq_len:]                   # last seq_len rows
        if len(seq) < self.seq_len:
            return 1, 0.0   # Flat / not enough data

        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, seq, feat)
        probs = self._model.predict_proba(x)[0]                 # (3,)
        predicted = int(probs.argmax().item())
        confidence = float(probs[predicted].item())
        return predicted, confidence

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

    def _enter_long(self, bar: Bar) -> None:
        atr = self._atr_now(list(self._bar_buffer))
        self._stop_price = bar.close - self.atr_stop_mult * atr
        self._tp_price   = bar.close + self.atr_tp_mult   * atr
        self.buy(bar.symbol, self.contracts)

    def _enter_short(self, bar: Bar) -> None:
        atr = self._atr_now(list(self._bar_buffer))
        self._stop_price = bar.close + self.atr_stop_mult * atr
        self._tp_price   = bar.close - self.atr_tp_mult   * atr
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
