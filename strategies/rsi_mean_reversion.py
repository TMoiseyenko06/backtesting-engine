"""
RSI Mean-Reversion Strategy
-----------------------------
Buys oversold dips and sells overbought rallies in futures.

Rules
-----
Entry long  : RSI < oversold_level  AND  price > long-term SMA (trend filter)
Entry short : RSI > overbought_level AND  price < long-term SMA
Exit long   : RSI > 50 OR stop hit
Exit short  : RSI < 50 OR stop hit

Zero lookahead: RSI is computed only on closed bars.
"""

from __future__ import annotations

from typing import Optional

from backtesting.data_feed import Bar
from backtesting.order import Order, OrderType
from backtesting.strategy import Strategy


class RSIMeanReversionStrategy(Strategy):
    """
    Parameters
    ----------
    rsi_period : int
    oversold : float  (default 30)
    overbought : float  (default 70)
    trend_sma_period : int
        Long-term SMA for trend filter (0 = no filter).
    contracts : float
    stop_atr_mult : float
    atr_period : int
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        trend_sma_period: int = 50,
        contracts: float = 1.0,
        stop_atr_mult: float = 1.5,
        atr_period: int = 14,
    ) -> None:
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.trend_sma_period = trend_sma_period
        self.contracts = contracts
        self.stop_atr_mult = stop_atr_mult
        self.atr_period = atr_period
        self._warmup = max(rsi_period, trend_sma_period, atr_period) + 10
        self._stop_order_id: Optional[str] = None

    def on_start(self) -> None:
        self.name = f"RSI_MeanRev(rsi={self.rsi_period},{self.oversold}/{self.overbought})"

    def on_bar(self, bar: Bar) -> None:
        if self.bars_available < self._warmup:
            return

        n = max(self.rsi_period + 5, self.trend_sma_period + 5, self.atr_period + 5)
        hist = self.history(n)
        closes = [b.close for b in hist]

        rsi_val = self.rsi(closes, self.rsi_period)
        atr_val = self.atr(hist, self.atr_period)

        trend_filter_long = True
        trend_filter_short = True
        if self.trend_sma_period > 0 and len(closes) >= self.trend_sma_period:
            sma_val = self.sma(closes, self.trend_sma_period)
            trend_filter_long = bar.close > sma_val
            trend_filter_short = bar.close < sma_val

        symbol = bar.symbol
        pos = self.position(symbol)

        # Exit logic
        if pos > 0 and rsi_val > 50:
            self.close_position(symbol, tag="rsi_exit_long")
            if self._stop_order_id:
                self.cancel_order(self._stop_order_id)
                self._stop_order_id = None
            return

        if pos < 0 and rsi_val < 50:
            self.close_position(symbol, tag="rsi_exit_short")
            if self._stop_order_id:
                self.cancel_order(self._stop_order_id)
                self._stop_order_id = None
            return

        # Entry logic
        if pos == 0:
            if rsi_val < self.oversold and trend_filter_long:
                self.buy(symbol, self.contracts, tag="rsi_long")
                if self.stop_atr_mult > 0 and atr_val > 0:
                    stop_px = bar.close - self.stop_atr_mult * atr_val
                    o = self.sell(
                        symbol, self.contracts,
                        order_type=OrderType.STOP,
                        stop_price=round(stop_px, 2),
                        reduce_only=True,
                        tag="sl",
                    )
                    self._stop_order_id = o.order_id

            elif rsi_val > self.overbought and trend_filter_short:
                self.sell(symbol, self.contracts, tag="rsi_short")
                if self.stop_atr_mult > 0 and atr_val > 0:
                    stop_px = bar.close + self.stop_atr_mult * atr_val
                    o = self.buy(
                        symbol, self.contracts,
                        order_type=OrderType.STOP,
                        stop_price=round(stop_px, 2),
                        reduce_only=True,
                        tag="sl",
                    )
                    self._stop_order_id = o.order_id
