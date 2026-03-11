"""
BacktestEngine — the central event loop.

Lookahead prevention checklist
--------------------------------
[x] Strategy.on_bar() is called AFTER _advance() closes the bar.
[x] Orders submitted during on_bar() are filled on the NEXT bar's open.
[x] Mark-to-market uses the just-closed bar's close price.
[x] Margin calls are checked after MTM, before the next on_bar().
[x] DataFeed._advance() is called by the engine only; strategies cannot
    move the cursor.
[x] DataFeed validates strictly ascending timestamps on construction.
[x] No strategy method exposes the current live bar or future bars.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from .data_feed import DataFeed, Bar
from .order import Order
from .order_manager import OrderManager
from .portfolio import Portfolio
from .strategy import Strategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Drives a backtest for a single data feed and one or more strategies.

    Parameters
    ----------
    feed : DataFeed
        The bar data source.
    portfolio : Portfolio
        Cash and position tracker.
    strategies : list[Strategy]
        One or more strategies to run simultaneously.
    verbose : bool
        Print progress every N bars.
    progress_interval : int
        Print a progress line every N bars when verbose=True.
    """

    def __init__(
        self,
        feed: DataFeed,
        portfolio: Portfolio,
        strategies: List[Strategy],
        verbose: bool = True,
        progress_interval: int = 500,
    ) -> None:
        self.feed = feed
        self.portfolio = portfolio
        self.strategies = strategies
        self.order_manager = OrderManager(portfolio)
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.current_timestamp: datetime = datetime.min
        self._bar_count = 0

        # Attach engine reference to all strategies
        for strat in self.strategies:
            strat._engine = self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "BacktestResult":
        """Execute the full backtest.  Returns a BacktestResult."""
        self._on_start()
        self._event_loop()
        self._on_stop()
        return self._build_result()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        for strat in self.strategies:
            try:
                strat.on_start()
            except Exception as exc:
                logger.error("on_start error in %s: %s", strat.name, exc)
                raise

    def _on_stop(self) -> None:
        for strat in self.strategies:
            try:
                strat.on_stop()
            except Exception as exc:
                logger.error("on_stop error in %s: %s", strat.name, exc)

    def _event_loop(self) -> None:
        while True:
            bar = self.feed._advance()
            if bar is None:
                break

            self.current_timestamp = bar.timestamp
            self._bar_count += 1

            # 1. Process pending/open orders against this bar's prices.
            #    (Orders from on_bar() on the previous bar fill here.)
            filled_orders = self.order_manager.process_bar(bar)
            if filled_orders:
                self._notify_fills(filled_orders)

            # 2. Mark-to-market and check margin calls
            prices = {bar.symbol: bar.close}
            equity = self.portfolio.mark_to_market(prices, bar.timestamp)
            margin_calls = self.portfolio.check_margin_calls(prices, bar.timestamp)
            if margin_calls:
                self._handle_margin_calls(margin_calls, bar)

            # 3. Skip strategy callbacks during warm-up
            if self.feed.is_warming_up:
                continue

            # 4. Call strategy on_bar() — strategies see only closed bars
            for strat in self.strategies:
                try:
                    strat.on_bar(bar)
                except Exception as exc:
                    logger.error(
                        "on_bar error in %s at %s: %s",
                        strat.name, bar.timestamp, exc,
                    )
                    raise

            if self.verbose and self._bar_count % self.progress_interval == 0:
                logger.info(
                    "Bar %d | %s | equity=%.2f",
                    self._bar_count, bar.timestamp, equity,
                )

    def _notify_fills(self, orders: List[Order]) -> None:
        for order in orders:
            for strat in self.strategies:
                try:
                    strat.on_fill(order)
                except Exception as exc:
                    logger.error("on_fill error in %s: %s", strat.name, exc)

    def _handle_margin_calls(self, calls, bar: Bar) -> None:
        for symbol, pos in calls:
            logger.warning(
                "MARGIN CALL: %s %s at %s — force-closing position.",
                symbol, pos, bar.timestamp,
            )
            self.order_manager.cancel_all(symbol)
            from .order import Order, OrderSide, OrderType
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            force_order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(pos.quantity),
                tag="margin_call",
            )
            self.portfolio.execute_fill(force_order, bar.close, bar.timestamp)

    def _build_result(self) -> "BacktestResult":
        from .analytics import Analytics
        return BacktestResult(
            portfolio=self.portfolio,
            analytics=Analytics(self.portfolio),
            bar_count=self._bar_count,
        )


class BacktestResult:
    """Container for backtest output."""

    def __init__(self, portfolio: Portfolio, analytics: "Analytics", bar_count: int) -> None:
        self.portfolio = portfolio
        self.analytics = analytics
        self.bar_count = bar_count

    def summary(self) -> str:
        return self.analytics.summary()

    def print_summary(self) -> None:
        print(self.summary())
