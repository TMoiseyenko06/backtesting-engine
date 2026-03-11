"""
Position tracking for futures contracts.

A futures position is simply a signed quantity plus average entry price.
Unrealised P&L uses mark-to-market (last close price × multiplier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Trade:
    """Record of a completed round-trip (or partial) trade."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str                   # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    exit_price: float
    contract_multiplier: float
    commission: float = 0.0

    @property
    def gross_pnl(self) -> float:
        direction = 1.0 if self.side == "LONG" else -1.0
        return (
            direction
            * (self.exit_price - self.entry_price)
            * self.quantity
            * self.contract_multiplier
        )

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission

    @property
    def return_pct(self) -> float:
        """Return as a fraction of initial margin (notional * multiplier)."""
        notional = self.entry_price * self.quantity * self.contract_multiplier
        return self.net_pnl / notional if notional else 0.0


@dataclass
class Position:
    """
    Live futures position for a single symbol.

    Attributes
    ----------
    symbol : str
    quantity : float
        Positive = long, negative = short, zero = flat.
    avg_entry_price : float
    contract_multiplier : float
    """

    symbol: str
    contract_multiplier: float
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realised_pnl: float = 0.0
    total_commission: float = 0.0
    entry_time: Optional[datetime] = None
    trades: List[Trade] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-9

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "LONG"
        if self.quantity < 0:
            return "SHORT"
        return "FLAT"

    def unrealised_pnl(self, mark_price: float) -> float:
        """Mark-to-market unrealised P&L."""
        return (mark_price - self.avg_entry_price) * self.quantity * self.contract_multiplier

    def notional_value(self, mark_price: float) -> float:
        return abs(self.quantity) * mark_price * self.contract_multiplier

    # ------------------------------------------------------------------
    # Mutation — called by the engine only
    # ------------------------------------------------------------------

    def apply_fill(
        self,
        fill_qty: float,       # positive = buy, negative = sell
        fill_price: float,
        commission: float,
        fill_time: datetime,
    ) -> Optional[Trade]:
        """
        Update position with a fill.  Returns a closed Trade if this fill
        fully or partially closes an existing position.
        """
        self.total_commission += commission
        closed_trade: Optional[Trade] = None

        if self.is_flat:
            # Opening a new position
            self.quantity = fill_qty
            self.avg_entry_price = fill_price
            self.entry_time = fill_time
            return None

        existing_sign = 1.0 if self.quantity > 0 else -1.0
        fill_sign = 1.0 if fill_qty > 0 else -1.0

        if existing_sign == fill_sign:
            # Adding to existing position — update average entry
            total_qty = self.quantity + fill_qty
            self.avg_entry_price = (
                self.avg_entry_price * self.quantity + fill_price * fill_qty
            ) / total_qty
            self.quantity = total_qty
        else:
            # Reducing or flipping position
            close_qty = min(abs(fill_qty), abs(self.quantity))
            close_side = "LONG" if self.quantity > 0 else "SHORT"
            direction = 1.0 if close_side == "LONG" else -1.0
            gross = direction * (fill_price - self.avg_entry_price) * close_qty * self.contract_multiplier
            self.realised_pnl += gross - commission
            closed_trade = Trade(
                symbol=self.symbol,
                entry_time=self.entry_time or fill_time,
                exit_time=fill_time,
                side=close_side,
                quantity=close_qty,
                entry_price=self.avg_entry_price,
                exit_price=fill_price,
                contract_multiplier=self.contract_multiplier,
                commission=commission,
            )
            self.trades.append(closed_trade)

            remaining_fill = abs(fill_qty) - close_qty
            self.quantity += fill_qty  # move toward zero (and possibly flip)

            if abs(self.quantity) < 1e-9:
                # Position closed exactly
                self.quantity = 0.0
                self.avg_entry_price = 0.0
                self.entry_time = None
            elif abs(self.quantity) > 1e-9 and (
                (fill_sign > 0 and self.quantity > 0)
                or (fill_sign < 0 and self.quantity < 0)
            ):
                # Position flipped — new entry at fill_price
                self.avg_entry_price = fill_price
                self.entry_time = fill_time

        return closed_trade

    def __repr__(self) -> str:
        return (
            f"Position({self.symbol} qty={self.quantity:.4f} "
            f"@ {self.avg_entry_price:.4f} [{self.side}])"
        )
