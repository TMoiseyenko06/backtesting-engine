"""
Order definitions for the futures backtesting engine.

All fills happen at the *next bar's open* after an order is submitted
(or at the limit/stop price if triggered later).  This models realistic
execution delay and eliminates any bar-of-signal fill bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional
import uuid


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"           # Stop-market (enter or exit at stop price)
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"         # Submitted, not yet processed
    OPEN = "OPEN"               # Accepted, waiting for fill
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """
    A single futures order.

    Attributes
    ----------
    symbol : str
    side : OrderSide
    order_type : OrderType
    quantity : float
        Number of contracts (positive).
    limit_price : float | None
        Required for LIMIT and STOP_LIMIT orders.
    stop_price : float | None
        Required for STOP and STOP_LIMIT orders.
    time_in_force : str
        'GTC' (good-till-cancelled) or 'DAY'.
    reduce_only : bool
        If True, the order can only reduce an existing position.
    tag : str
        Optional label (e.g. 'entry', 'tp', 'sl') for strategy bookkeeping.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    tag: str = ""

    # Set by the engine
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    reject_reason: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None:
                raise ValueError(f"{self.order_type} order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if self.stop_price is None:
                raise ValueError(f"{self.order_type} order requires stop_price")

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_done(self) -> bool:
        return not self.is_active

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, {self.side.value} {self.quantity} "
            f"{self.symbol} @ {self.order_type.value}, "
            f"status={self.status.value})"
        )
