"""
Portfolio — manages cash, margin, positions, and equity curve.

Futures margin model
---------------------
Initial margin  = required per contract to open a position.
Maintenance margin = minimum equity to hold the position.
If equity < maintenance margin → margin call → position force-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .order import Order, OrderSide, OrderStatus
from .position import Position, Trade


@dataclass
class MarginSpec:
    """Per-symbol margin requirements."""
    symbol: str
    initial_margin_per_contract: float       # e.g. $12,000 for ES
    maintenance_margin_per_contract: float   # e.g. $10,900 for ES
    contract_multiplier: float = 1.0


@dataclass
class EquitySnapshot:
    timestamp: datetime
    cash: float
    unrealised_pnl: float
    margin_used: float

    @property
    def equity(self) -> float:
        return self.cash + self.unrealised_pnl

    @property
    def free_margin(self) -> float:
        return self.equity - self.margin_used


class Portfolio:
    """
    Tracks cash, open positions, and equity over time.

    Parameters
    ----------
    initial_cash : float
    margin_specs : dict[symbol, MarginSpec]
        If a symbol has no spec, orders for it will be rejected.
    commission_per_contract : float
        Flat commission charged per contract per side.
    slippage_ticks : float
        Number of ticks of slippage added to fills (0 for frictionless).
    tick_size : float
        Minimum price increment (used with slippage_ticks).
    """

    def __init__(
        self,
        initial_cash: float,
        margin_specs: Optional[Dict[str, MarginSpec]] = None,
        commission_per_contract: float = 2.0,
        slippage_ticks: float = 0.0,
        tick_size: float = 0.25,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.margin_specs: Dict[str, MarginSpec] = margin_specs or {}
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size

        self.positions: Dict[str, Position] = {}
        self.all_trades: List[Trade] = []
        self.equity_curve: List[EquitySnapshot] = []
        self._order_log: List[Order] = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            multiplier = self._multiplier(symbol)
            self.positions[symbol] = Position(symbol=symbol, contract_multiplier=multiplier)
        return self.positions[symbol]

    def _multiplier(self, symbol: str) -> float:
        spec = self.margin_specs.get(symbol)
        return spec.contract_multiplier if spec else 1.0

    def mark_to_market(self, prices: Dict[str, float], timestamp: datetime) -> float:
        """
        Update unrealised P&L and record an equity snapshot.
        Returns current equity.
        """
        unrealised = sum(
            pos.unrealised_pnl(prices[sym])
            for sym, pos in self.positions.items()
            if not pos.is_flat and sym in prices
        )
        margin_used = sum(
            self._margin_used(sym, pos)
            for sym, pos in self.positions.items()
            if not pos.is_flat
        )
        snap = EquitySnapshot(
            timestamp=timestamp,
            cash=self.cash,
            unrealised_pnl=unrealised,
            margin_used=margin_used,
        )
        self.equity_curve.append(snap)
        return snap.equity

    def equity(self, prices: Dict[str, float]) -> float:
        unrealised = sum(
            pos.unrealised_pnl(prices[sym])
            for sym, pos in self.positions.items()
            if not pos.is_flat and sym in prices
        )
        return self.cash + unrealised

    # ------------------------------------------------------------------
    # Margin helpers
    # ------------------------------------------------------------------

    def _margin_used(self, symbol: str, pos: Position) -> float:
        spec = self.margin_specs.get(symbol)
        if spec is None:
            return 0.0
        return abs(pos.quantity) * spec.maintenance_margin_per_contract

    def _initial_margin_required(self, symbol: str, quantity: float) -> float:
        spec = self.margin_specs.get(symbol)
        if spec is None:
            return 0.0
        return abs(quantity) * spec.initial_margin_per_contract

    def check_margin_calls(
        self, prices: Dict[str, float], timestamp: datetime
    ) -> List[Tuple[str, Position]]:
        """
        Return list of (symbol, position) pairs where equity backing the
        position has fallen below maintenance margin.
        """
        calls = []
        for sym, pos in self.positions.items():
            if pos.is_flat or sym not in prices:
                continue
            maint = self._margin_used(sym, pos)
            unrealised = pos.unrealised_pnl(prices[sym])
            if self.cash + unrealised < maint:
                calls.append((sym, pos))
        return calls

    # ------------------------------------------------------------------
    # Fill execution
    # ------------------------------------------------------------------

    def execute_fill(
        self,
        order: Order,
        raw_fill_price: float,
        fill_time: datetime,
    ) -> bool:
        """
        Execute an order fill.  Returns True on success, False if rejected
        (insufficient margin).

        Slippage is applied here (after the bar has closed) — not in the
        strategy, so it cannot be gamed.
        """
        # Apply slippage in the direction of the trade
        slip = self.slippage_ticks * self.tick_size
        if order.side == OrderSide.BUY:
            fill_price = raw_fill_price + slip
        else:
            fill_price = raw_fill_price - slip

        commission = self.commission_per_contract * order.quantity

        # Margin check for new / increasing positions
        pos = self.get_position(order.symbol)
        existing_side = pos.side
        fill_sign = 1.0 if order.side == OrderSide.BUY else -1.0
        existing_sign = 0.0
        if existing_side == "LONG":
            existing_sign = 1.0
        elif existing_side == "SHORT":
            existing_sign = -1.0

        is_increasing = (existing_sign == 0) or (existing_sign == fill_sign)
        if is_increasing:
            margin_req = self._initial_margin_required(order.symbol, order.quantity)
            if self.cash < margin_req + commission:
                order.status = OrderStatus.REJECTED
                order.reject_reason = (
                    f"Insufficient margin: need {margin_req + commission:.2f}, "
                    f"cash={self.cash:.2f}"
                )
                self._order_log.append(order)
                return False

        # Apply fill to position
        fill_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        closed_trade = pos.apply_fill(fill_qty, fill_price, commission, fill_time)

        # Update cash
        # For futures: no premium paid — margin is locked, realised P&L flows to cash
        if closed_trade is not None:
            self.cash += closed_trade.net_pnl
            self.all_trades.append(closed_trade)
        else:
            # Opening/adding: deduct commission only
            self.cash -= commission

        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = fill_time
        order.commission = commission
        self._order_log.append(order)
        return True
