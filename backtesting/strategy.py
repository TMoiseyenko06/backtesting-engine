"""
Strategy base class.

Users subclass `Strategy` and implement `on_bar()`.  They interact with
the engine exclusively through the methods defined here — all of which
only expose already-closed bar data (zero lookahead).

Usage example
-------------
    class MyStrategy(Strategy):
        def on_start(self):
            self.ema_period = 20

        def on_bar(self, bar: Bar) -> None:
            if self.bars_available < self.ema_period:
                return
            closes = [b.close for b in self.history(self.ema_period)]
            ema = self._ema(closes, self.ema_period)
            if bar.close > ema and self.position(bar.symbol) == 0:
                self.buy(bar.symbol, quantity=1)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from .data_feed import Bar
from .order import Order, OrderSide, OrderType
from .position import Position

if TYPE_CHECKING:
    from .engine import BacktestEngine


class Strategy(ABC):
    """
    Base class for all futures strategies.

    The engine injects references at startup; strategies must not
    access engine internals directly.
    """

    def __init__(self) -> None:
        self._engine: Optional["BacktestEngine"] = None
        self.name: str = self.__class__.__name__

    # ------------------------------------------------------------------
    # Lifecycle hooks — override as needed
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the first bar.  Use for initialisation."""

    def on_bar(self, bar: Bar) -> None:
        """
        Called once per closed bar (after warm-up).

        `bar` is the bar that just closed — the same object returned by
        `self.current_bar()`.  You may safely read all its fields.
        Future bars are not accessible.
        """

    def on_fill(self, order: Order) -> None:
        """Called when one of this strategy's orders is filled."""

    def on_stop(self) -> None:
        """Called once after the last bar."""

    # ------------------------------------------------------------------
    # Data access (zero lookahead)
    # ------------------------------------------------------------------

    @property
    def bars_available(self) -> int:
        """Number of closed bars the strategy can read."""
        self._require_engine()
        return self._engine.feed.bars_available()

    def current_bar(self) -> Optional[Bar]:
        """The most recently closed bar."""
        self._require_engine()
        return self._engine.feed.current_bar()

    def history(self, n: int) -> List[Bar]:
        """
        Last `n` closed bars, oldest first.
        Same as calling feed.history(n).
        """
        self._require_engine()
        return self._engine.feed.history(n)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def buy(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        tag: str = "",
    ) -> Order:
        """Submit a buy order.  Fills on next bar at the earliest."""
        return self._submit(
            symbol, OrderSide.BUY, quantity, order_type,
            limit_price, stop_price, reduce_only, tag,
        )

    def sell(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        tag: str = "",
    ) -> Order:
        """Submit a sell order.  Fills on next bar at the earliest."""
        return self._submit(
            symbol, OrderSide.SELL, quantity, order_type,
            limit_price, stop_price, reduce_only, tag,
        )

    def close_position(self, symbol: str, tag: str = "close") -> Optional[Order]:
        """Flatten an open position with a market order."""
        pos = self.position_obj(symbol)
        if pos.is_flat:
            return None
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        qty = abs(pos.quantity)
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            reduce_only=True,
            tag=tag,
        )
        return self._engine.order_manager.submit(order, self._engine.current_timestamp)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order by ID."""
        self._require_engine()
        return self._engine.order_manager.cancel(order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders (optionally filtered by symbol)."""
        self._require_engine()
        return self._engine.order_manager.cancel_all(symbol)

    # ------------------------------------------------------------------
    # Portfolio / position queries
    # ------------------------------------------------------------------

    def position(self, symbol: str) -> float:
        """Signed position size (positive=long, negative=short, 0=flat)."""
        self._require_engine()
        return self._engine.portfolio.get_position(symbol).quantity

    def position_obj(self, symbol: str) -> Position:
        """Full Position object."""
        self._require_engine()
        return self._engine.portfolio.get_position(symbol)

    def cash(self) -> float:
        self._require_engine()
        return self._engine.portfolio.cash

    def equity(self) -> float:
        self._require_engine()
        bar = self.current_bar()
        if bar is None:
            return self._engine.portfolio.cash
        prices = {bar.symbol: bar.close}
        return self._engine.portfolio.equity(prices)

    @property
    def open_orders(self) -> List[Order]:
        self._require_engine()
        return self._engine.order_manager.open_orders

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float],
        stop_price: Optional[float],
        reduce_only: bool,
        tag: str,
    ) -> Order:
        self._require_engine()
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            tag=tag,
        )
        return self._engine.order_manager.submit(order, self._engine.current_timestamp)

    def _require_engine(self) -> None:
        if self._engine is None:
            raise RuntimeError(
                "Strategy has not been attached to an engine. "
                "Pass it to BacktestEngine.run()."
            )

    # ------------------------------------------------------------------
    # Utility indicators (all use only already-provided history — no lookahead)
    # ------------------------------------------------------------------

    @staticmethod
    def sma(values: List[float], period: int) -> float:
        """Simple moving average of the last `period` values."""
        if len(values) < period:
            raise ValueError(f"Need {period} values, got {len(values)}")
        return sum(values[-period:]) / period

    @staticmethod
    def ema(values: List[float], period: int) -> float:
        """
        Exponential moving average.  Computed iteratively over all values;
        returns the final EMA value.
        """
        if len(values) < 1:
            raise ValueError("values must be non-empty")
        k = 2.0 / (period + 1)
        result = values[0]
        for v in values[1:]:
            result = v * k + result * (1 - k)
        return result

    @staticmethod
    def atr(bars: List[Bar], period: int) -> float:
        """
        Average True Range over `period` bars.
        Requires at least period+1 bars (for the prior close).
        """
        if len(bars) < 2:
            return bars[-1].high - bars[-1].low if bars else 0.0
        trs = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - prev_close),
                abs(bars[i].low - prev_close),
            )
            trs.append(tr)
        return sum(trs[-period:]) / min(period, len(trs))

    @staticmethod
    def rsi(values: List[float], period: int) -> float:
        """
        Relative Strength Index.  Returns value in [0, 100].
        Requires at least period+1 values.
        """
        if len(values) < period + 1:
            raise ValueError(f"RSI needs at least {period + 1} values")
        gains, losses = [], []
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)
