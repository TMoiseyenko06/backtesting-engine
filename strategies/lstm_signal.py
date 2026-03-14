"""
LSTM Signal Strategy
---------------------
Uses a pre-trained LSTMModel to generate Buy / Flat / Sell signals on every bar.

Signal mapping (matches dataset.py):
    2 → BUY  — go long  (or flip from short)
    1 → FLAT — close any open position
    0 → SELL — go short (or flip from long)

Position logic:
    - High-confidence BUY  → buy if flat or short  (close short first)
    - High-confidence SELL → sell if flat or long   (close long first)
    - FLAT signal          → close any open position
    - Below confidence_threshold → do nothing (stay in current position)
"""

from __future__ import annotations

from backtesting.data_feed import Bar
from backtesting.strategy import Strategy

# Class-index constants (must match backtesting/ml/dataset.py)
_BUY  = 2
_SELL = 0
_FLAT = 1

# Minimum history bars needed so that SMA-50 (the longest indicator in
# make_features) is fully warmed up before the LSTM window begins.
_INDICATOR_WARMUP = 60


class LSTMSignalStrategy(Strategy):
    """
    Parameters
    ----------
    model : LSTMModel
        Trained model instance (already on the correct device).
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    symbol : str
        Futures symbol to trade, e.g. ``"NQ"``.
    seq_len : int
        Sequence length expected by the model (must match training config).
    confidence_threshold : float
        Minimum softmax probability to act on a signal.  Bars where the
        model is uncertain stay in their current position.
    contracts : float
        Number of contracts per trade.
    """

    def __init__(
        self,
        model,
        device: str,
        symbol: str,
        seq_len: int = 30,
        confidence_threshold: float = 0.5,
        contracts: float = 1.0,
    ) -> None:
        super().__init__()
        self._model  = model
        self._device = device
        self._symbol = symbol
        self._seq_len = seq_len
        self._conf_thresh = confidence_threshold
        self._contracts = contracts
        # Need seq_len + indicator_warmup closed bars before first inference
        self._min_bars = seq_len + _INDICATOR_WARMUP

    def on_start(self) -> None:
        self.name = f"LSTM-Signal(seq={self._seq_len},conf≥{self._conf_thresh:.0%})"

    def on_bar(self, bar: Bar) -> None:
        import torch
        from backtesting.ml.features import make_features

        if self.bars_available < self._min_bars:
            return

        # Build feature matrix from recent closed bars only — zero lookahead
        hist = self.history(self._min_bars)
        features = make_features(hist)          # (min_bars, N_FEATURES)
        x = features[-self._seq_len:]           # (seq_len, N_FEATURES)

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
        signal, confidence = self._model.predict(x_t)

        if confidence < self._conf_thresh:
            return  # not confident enough — hold current position

        pos = self.position(self._symbol)       # signed float: >0 long, <0 short, 0 flat

        if signal == _BUY and pos <= 0:
            if pos < 0:
                self.close_position(self._symbol, tag="flip_long")
            self.buy(self._symbol, self._contracts, tag="lstm_long")

        elif signal == _SELL and pos >= 0:
            if pos > 0:
                self.close_position(self._symbol, tag="flip_short")
            self.sell(self._symbol, self._contracts, tag="lstm_short")

        elif signal == _FLAT and pos != 0:
            self.close_position(self._symbol, tag="lstm_flat")
