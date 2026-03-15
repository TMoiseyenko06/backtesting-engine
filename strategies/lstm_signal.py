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

Risk controls:
    - Intraday only: all positions closed at/after ``eod_hour_utc`` (default 20,
      i.e. 20:00 UTC ≈ 4 PM EDT).  No new entries after ``no_entry_hour_utc``
      (default 19).
    - Per-trade stop-loss: if unrealised P&L on the current position falls below
      ``-max_loss_per_trade`` (default $2 500), the position is closed immediately.
      Dollar P&L is calculated using the bar's ``contract_multiplier``.
    - Minimum hold time: signal-driven exits/flips are blocked for the first
      ``min_hold_bars`` bars after entry (default 2).  Stop-loss and EOD close
      always override this.  With 1-minute bars this enforces a ≥2-minute hold,
      ensuring no sub-minute scalping and guaranteeing >50 % of trades exceed
      10 seconds.
"""

from __future__ import annotations

from backtesting.data_feed import Bar
from backtesting.order import Order
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
    max_loss_per_trade : float
        Maximum dollar loss allowed before the position is stopped out.
        Default $2 500.
    eod_hour_utc : int
        UTC hour at (and after) which all open positions are closed and no
        new entries are taken.  Default 20 (20:00 UTC ≈ 4 PM EDT / 3 PM CDT).
    no_entry_hour_utc : int
        UTC hour at (and after) which no new entries are submitted.  Must be
        <= ``eod_hour_utc``.  Default 19.
    min_hold_bars : int
        Minimum number of bars a position must be held before a signal-driven
        exit or flip is allowed.  Stop-loss and EOD close always override this.
        Default 2 (≥ 2 minutes on 1-min bars — no scalping).
    """

    def __init__(
        self,
        model,
        device: str,
        symbol: str,
        seq_len: int = 30,
        confidence_threshold: float = 0.5,
        contracts: float = 1.0,
        max_loss_per_trade: float = 2_500.0,
        eod_hour_utc: int = 20,
        no_entry_hour_utc: int = 19,
        min_hold_bars: int = 2,
    ) -> None:
        super().__init__()
        self._model  = model
        self._device = device
        self._symbol = symbol
        self._seq_len = seq_len
        self._conf_thresh = confidence_threshold
        self._contracts = contracts
        self._max_loss = max_loss_per_trade
        self._eod_hour = eod_hour_utc
        self._no_entry_hour = no_entry_hour_utc
        self._min_hold_bars = min_hold_bars
        # Need seq_len + indicator_warmup closed bars before first inference
        self._min_bars = seq_len + _INDICATOR_WARMUP

        # Tracking state for open position
        self._entry_price: float = 0.0      # actual fill price (set in on_fill)
        self._entry_bar: int = 0            # bars_available at entry

    def on_start(self) -> None:
        self.name = (
            f"LSTM-Signal(seq={self._seq_len},conf≥{self._conf_thresh:.0%},"
            f"stop=${self._max_loss:,.0f},hold≥{self._min_hold_bars}bar,intraday)"
        )

    # ------------------------------------------------------------------
    # Fill callback — capture actual entry price
    # ------------------------------------------------------------------

    def on_fill(self, order: Order) -> None:
        """Record the fill price whenever we open or flip into a new position."""
        pos_after = self.position(self._symbol)
        if abs(pos_after) > 1e-9 and order.fill_price is not None:
            self._entry_price = order.fill_price
            self._entry_bar   = self.bars_available

    # ------------------------------------------------------------------
    # Main bar callback
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        import torch
        from backtesting.ml.features import make_features

        if self.bars_available < self._min_bars:
            return

        pos = self.position(self._symbol)
        bar_hour = bar.timestamp.hour   # Databento timestamps are UTC

        # ── 1. End-of-day flat rule ────────────────────────────────────
        # At/after eod_hour_utc: close any open position and stop trading.
        if bar_hour >= self._eod_hour:
            if pos != 0:
                self.close_position(self._symbol, tag="eod_close")
            return

        # ── 2. Per-trade stop-loss (always active, overrides hold timer) ──
        if pos != 0 and self._entry_price > 0:
            direction = 1.0 if pos > 0 else -1.0
            unrealised = (
                direction
                * (bar.close - self._entry_price)
                * abs(pos)
                * bar.contract_multiplier
            )
            if unrealised < -self._max_loss:
                self.close_position(self._symbol, tag="stop_loss")
                return

        # ── 3. No new entries in the final run-up to EOD ──────────────
        if bar_hour >= self._no_entry_hour:
            return  # let existing position ride until EOD close above

        # ── 4. Minimum hold time — block signal exits/flips ───────────
        bars_held = self.bars_available - self._entry_bar
        signal_exits_blocked = (pos != 0) and (bars_held < self._min_hold_bars)

        # ── 5. Model inference ────────────────────────────────────────
        hist = self.history(self._min_bars)
        features = make_features(hist)          # (min_bars, N_FEATURES)
        x = features[-self._seq_len:]           # (seq_len, N_FEATURES)

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
        signal, confidence = self._model.predict(x_t)

        if confidence < self._conf_thresh:
            return  # not confident enough — hold current position

        # ── 6. Execute signal ─────────────────────────────────────────
        if signal == _BUY and pos <= 0:
            if signal_exits_blocked:
                return   # too soon to flip/exit short
            if pos < 0:
                self.close_position(self._symbol, tag="flip_long")
            self.buy(self._symbol, self._contracts, tag="lstm_long")

        elif signal == _SELL and pos >= 0:
            if signal_exits_blocked:
                return   # too soon to flip/exit long
            if pos > 0:
                self.close_position(self._symbol, tag="flip_short")
            self.sell(self._symbol, self._contracts, tag="lstm_short")

        elif signal == _FLAT and pos != 0:
            if signal_exits_blocked:
                return   # hold a bit longer before flattening
            self.close_position(self._symbol, tag="lstm_flat")
