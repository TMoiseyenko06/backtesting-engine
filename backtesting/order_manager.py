"""
OrderManager — simulates a brokerage order book with no lookahead.

Lookahead-prevention rules
---------------------------
1. MARKET orders submitted on bar N are filled at bar N+1's **open**.
2. LIMIT orders submitted on bar N are checked for fills starting on
   bar N+1.  They fill at the limit price if the bar's range includes it.
3. STOP orders trigger at the stop price once the bar's range crosses it;
   they then fill at that price (or the open if gap-through).
4. A fill on bar N can only use price data from bar N — never future bars.
5. `reduce_only` orders are cancelled if the position is already flat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .data_feed import Bar
from .order import Order, OrderSide, OrderStatus, OrderType
from .portfolio import Portfolio


class OrderManager:
    def __init__(self, portfolio: Portfolio) -> None:
        self._portfolio = portfolio
        self._pending: List[Order] = []    # submitted this bar, fill next bar
        self._open: List[Order] = []       # resting limit/stop orders

    # ------------------------------------------------------------------
    # Strategy-facing API
    # ------------------------------------------------------------------

    def submit(self, order: Order, timestamp: datetime) -> Order:
        """
        Accept an order from the strategy.  Market orders move to _pending;
        limit/stop orders move to _open.
        """
        order.submitted_at = timestamp
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.PENDING
            self._pending.append(order)
        else:
            order.status = OrderStatus.OPEN
            self._open.append(order)
        return order

    def cancel(self, order_id: str) -> bool:
        for lst in (self._pending, self._open):
            for order in lst:
                if order.order_id == order_id and order.is_active:
                    order.status = OrderStatus.CANCELLED
                    lst.remove(order)
                    return True
        return False

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        cancelled = 0
        for lst in (self._pending, self._open):
            to_remove = []
            for order in lst:
                if order.is_active and (symbol is None or order.symbol == symbol):
                    order.status = OrderStatus.CANCELLED
                    to_remove.append(order)
                    cancelled += 1
            for order in to_remove:
                lst.remove(order)
        return cancelled

    @property
    def open_orders(self) -> List[Order]:
        return list(self._open)

    @property
    def pending_orders(self) -> List[Order]:
        return list(self._pending)

    # ------------------------------------------------------------------
    # Engine-facing API — called AFTER each bar closes
    # ------------------------------------------------------------------

    def process_bar(self, bar: Bar) -> List[Order]:
        """
        Process all pending and open orders against the newly closed bar.
        Returns list of filled orders.

        Called by the engine with the bar that just closed.  Because the
        strategy only sees `current_bar()` (the *previous* bar), no future
        price information is used here.
        """
        filled: List[Order] = []

        # 1. Market orders from last bar → fill at this bar's open
        pending_snapshot = list(self._pending)
        self._pending.clear()
        for order in pending_snapshot:
            if not order.is_active:
                continue
            if self._reduce_only_check(order):
                order.status = OrderStatus.CANCELLED
                continue
            success = self._portfolio.execute_fill(order, bar.open, bar.timestamp)
            if success:
                filled.append(order)

        # 2. Resting limit / stop orders → check against this bar's range
        still_open: List[Order] = []
        for order in self._open:
            if not order.is_active:
                continue

            if self._reduce_only_check(order):
                order.status = OrderStatus.CANCELLED
                continue

            fill_price = self._check_limit_stop(order, bar)
            if fill_price is not None:
                success = self._portfolio.execute_fill(order, fill_price, bar.timestamp)
                if success:
                    filled.append(order)
            else:
                # DAY orders expire at end of bar's session
                if order.time_in_force == "DAY":
                    order.status = OrderStatus.EXPIRED
                else:
                    still_open.append(order)

        self._open = still_open
        return filled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reduce_only_check(self, order: Order) -> bool:
        """Return True if the order should be cancelled due to reduce_only."""
        if not order.reduce_only:
            return False
        pos = self._portfolio.get_position(order.symbol)
        if pos.is_flat:
            return True
        # Cancel if order would increase (not reduce) the position
        pos_sign = 1 if pos.quantity > 0 else -1
        order_sign = 1 if order.side == OrderSide.BUY else -1
        return pos_sign == order_sign  # same direction → would increase

    def _check_limit_stop(self, order: Order, bar: Bar) -> Optional[float]:
        """
        Determine the fill price for a limit/stop order on `bar`.
        Returns None if the order should not fill on this bar.

        Gap-through logic:
          - If the bar opens through a limit/stop price, fill at the bar open
            (simulates realistic gap fills — conservative for limits, realistic
            for stops).
        """
        lo, hi, op = bar.low, bar.high, bar.open

        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # Buy limit: fill if bar trades at or below limit price
                if lo <= order.limit_price:
                    # Gap-down open through limit → fill at open (better for buyer)
                    return min(op, order.limit_price)
            else:  # SELL
                if hi >= order.limit_price:
                    return max(op, order.limit_price)

        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                # Buy stop: triggers when price rises to stop
                if hi >= order.stop_price:
                    # Gap-up open through stop → fill at open (worse for buyer — realistic)
                    return max(op, order.stop_price)
            else:
                if lo <= order.stop_price:
                    return min(op, order.stop_price)

        elif order.order_type == OrderType.STOP_LIMIT:
            triggered = False
            if order.side == OrderSide.BUY and hi >= order.stop_price:
                triggered = True
            elif order.side == OrderSide.SELL and lo <= order.stop_price:
                triggered = True

            if triggered:
                # Now check limit condition
                if order.side == OrderSide.BUY and lo <= order.limit_price:
                    return min(op, order.limit_price)
                elif order.side == OrderSide.SELL and hi >= order.limit_price:
                    return max(op, order.limit_price)

        return None
