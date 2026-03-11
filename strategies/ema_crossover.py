"""
EMA Crossover Strategy
-----------------------
Classic dual-EMA trend-following system for futures.

Rules
-----
Entry long  : fast EMA crosses above slow EMA
Entry short : fast EMA crosses below slow EMA
Exit        : opposite crossover OR stop-loss hit

Zero lookahead: all EMA values are computed on closed bars only.
The signal that triggers on bar N causes an order filled on bar N+1's open.
"""

from __future__ import annotations

from typing import Optional

from backtesting.data_feed import Bar
from backtesting.order import Order, OrderType
from backtesting.strategy import Strategy


class EMACrossoverStrategy(Strategy):
    """
    Parameters
    ----------
    fast_period : int
        Period of the fast EMA.
    slow_period : int
        Period of the slow EMA.
    contracts : float
        Number of contracts per trade.
    stop_atr_mult : float
        ATR multiplier for stop-loss distance (0 = no stop).
    atr_period : int
        Period for ATR calculation used in stop-loss sizing.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        contracts: float = 1.0,
        stop_atr_mult: float = 2.0,
        atr_period: int = 14,
    ) -> None:
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.contracts = contracts
        self.stop_atr_mult = stop_atr_mult
        self.atr_period = atr_period

        self._warmup = slow_period + atr_period + 5
        self._prev_fast: Optional[float] = None
        self._prev_slow: Optional[float] = None
        self._stop_order_id: Optional[str] = None

    def on_start(self) -> None:
        self.name = (
            f"EMACross({self.fast_period}/{self.slow_period})"
        )

    def on_bar(self, bar: Bar) -> None:
        if self.bars_available < self._warmup:
            return

        # Fetch enough history for indicators — all closed bars, no lookahead
        n = max(self.slow_period * 3, self.atr_period + 10)
        hist = self.history(n)
        closes = [b.close for b in hist]

        fast_ema = self.ema(closes, self.fast_period)
        slow_ema = self.ema(closes, self.slow_period)
        atr_val = self.atr(hist, self.atr_period)

        symbol = bar.symbol
        pos = self.position(symbol)

        # --- Entry signals ---
        bullish_cross = (
            self._prev_fast is not None
            and self._prev_slow is not None
            and self._prev_fast <= self._prev_slow
            and fast_ema > slow_ema
        )
        bearish_cross = (
            self._prev_fast is not None
            and self._prev_slow is not None
            and self._prev_fast >= self._prev_slow
            and fast_ema < slow_ema
        )

        if bullish_cross and pos <= 0:
            # Close any short first
            if pos < 0:
                self.close_position(symbol, tag="flip_long")
                self._stop_order_id = None
            # Go long — market order fills at next bar open
            self.buy(symbol, self.contracts, tag="ema_long")
            # Place stop below entry (approximate — uses current close)
            if self.stop_atr_mult > 0 and atr_val > 0:
                stop_px = bar.close - self.stop_atr_mult * atr_val
                o = self.sell(
                    symbol, self.contracts,
                    order_type=OrderType.STOP,
                    stop_price=round(stop_px, 2),
                    reduce_only=True,
                    tag="sl_long",
                )
                self._stop_order_id = o.order_id

        elif bearish_cross and pos >= 0:
            if pos > 0:
                self.close_position(symbol, tag="flip_short")
                self._stop_order_id = None
            self.sell(symbol, self.contracts, tag="ema_short")
            if self.stop_atr_mult > 0 and atr_val > 0:
                stop_px = bar.close + self.stop_atr_mult * atr_val
                o = self.buy(
                    symbol, self.contracts,
                    order_type=OrderType.STOP,
                    stop_price=round(stop_px, 2),
                    reduce_only=True,
                    tag="sl_short",
                )
                self._stop_order_id = o.order_id

        self._prev_fast = fast_ema
        self._prev_slow = slow_ema

    def on_fill(self, order: Order) -> None:
        pass  # Could log fills here
