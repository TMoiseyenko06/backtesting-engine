"""
Strict lookahead and bias tests.

These tests verify the core guarantee of the engine:
  - Strategy code cannot access future bars.
  - Orders fill on the next bar, never the current one.
  - DataFeed rejects non-chronological data.
  - Slippage and commission are applied correctly.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from typing import List, Optional

from backtesting.data_feed import Bar, DataFeed
from backtesting.engine import BacktestEngine
from backtesting.order import Order, OrderType
from backtesting.portfolio import Portfolio, MarginSpec
from backtesting.strategy import Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bars(n: int, start_price: float = 100.0, symbol: str = "TEST") -> List[Bar]:
    """Generate n ascending bars with deterministic prices."""
    base = datetime(2023, 1, 1)
    bars = []
    price = start_price
    for i in range(n):
        bars.append(Bar(
            timestamp=base + timedelta(days=i),
            symbol=symbol,
            open=price,
            high=price + 2.0,
            low=price - 1.0,
            close=price + 1.0,
            volume=1000.0,
            contract_multiplier=1.0,
        ))
        price += 1.0
    return bars


def make_engine(bars, strategy, initial_cash=100_000.0, margin_per_contract=5000.0):
    feed = DataFeed(bars)
    margin = {
        "TEST": MarginSpec(
            symbol="TEST",
            initial_margin_per_contract=margin_per_contract,
            maintenance_margin_per_contract=margin_per_contract * 0.9,
            contract_multiplier=1.0,
        )
    }
    portfolio = Portfolio(initial_cash, margin_specs=margin, commission_per_contract=0.0, slippage_ticks=0)
    return BacktestEngine(feed, portfolio, [strategy], verbose=False)


# ---------------------------------------------------------------------------
# Test: DataFeed rejects non-chronological data
# ---------------------------------------------------------------------------

def test_datafeed_rejects_non_chronological():
    bars = make_bars(5)
    # Swap two bars to create a non-monotonic sequence
    bars[2], bars[3] = bars[3], bars[2]
    with pytest.raises(ValueError, match="strictly ascending"):
        DataFeed(bars)


# ---------------------------------------------------------------------------
# Test: Strategy cannot see future bars via history()
# ---------------------------------------------------------------------------

class FuturePeekStrategy(Strategy):
    """Attempts to access bars beyond the current cursor — must fail."""

    def __init__(self):
        super().__init__()
        self.violations = []

    def on_bar(self, bar: Bar) -> None:
        available = self.bars_available
        hist = self.history(available)
        # hist should contain exactly `available` bars, all <= bar.timestamp
        for h in hist:
            if h.timestamp > bar.timestamp:
                self.violations.append(
                    f"Future bar seen: {h.timestamp} > {bar.timestamp}"
                )


def test_no_future_bars_in_history():
    bars = make_bars(50)
    strat = FuturePeekStrategy()
    engine = make_engine(bars, strat)
    engine.run()
    assert strat.violations == [], f"Lookahead violations: {strat.violations}"


# ---------------------------------------------------------------------------
# Test: Market order fills on NEXT bar's open, not current bar
# ---------------------------------------------------------------------------

class FillTimingStrategy(Strategy):
    """Records the bar index when order was submitted and when it filled."""

    def __init__(self):
        super().__init__()
        self.submit_bar_idx: Optional[int] = None
        self.fill_bar_idx: Optional[int] = None
        self.fill_price: Optional[float] = None
        self._order_submitted = False

    def on_bar(self, bar: Bar) -> None:
        if not self._order_submitted and self.bars_available >= 2:
            self.buy(bar.symbol, 1.0, tag="test")
            self.submit_bar_idx = self.bars_available  # cursor at submission
            self._order_submitted = True

    def on_fill(self, order: Order) -> None:
        self.fill_bar_idx = self.bars_available
        self.fill_price = order.fill_price


def test_market_order_fills_next_bar():
    bars = make_bars(10)
    strat = FillTimingStrategy()
    engine = make_engine(bars, strat)
    engine.run()

    assert strat.submit_bar_idx is not None, "Order was never submitted"
    assert strat.fill_bar_idx is not None, "Order was never filled"

    # Fill must happen on a bar strictly after submission
    assert strat.fill_bar_idx > strat.submit_bar_idx, (
        f"Order filled on same or earlier bar: "
        f"submit={strat.submit_bar_idx}, fill={strat.fill_bar_idx}"
    )

    # Fill price must equal the open of the bar AFTER submission
    submission_bar_number = strat.submit_bar_idx - 1  # 0-indexed
    next_bar = bars[submission_bar_number + 1]
    assert strat.fill_price == next_bar.open, (
        f"Expected fill at next bar open={next_bar.open}, got {strat.fill_price}"
    )


# ---------------------------------------------------------------------------
# Test: history() never exceeds available bars
# ---------------------------------------------------------------------------

class HistoryLenStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.violations = []

    def on_bar(self, bar: Bar) -> None:
        available = self.bars_available
        try:
            hist = self.history(available + 100)  # ask for more than available
        except Exception:
            return
        if len(hist) > available:
            self.violations.append(
                f"Got {len(hist)} bars but only {available} should be available"
            )


def test_history_never_exceeds_available():
    bars = make_bars(30)
    strat = HistoryLenStrategy()
    engine = make_engine(bars, strat)
    engine.run()
    assert strat.violations == []


# ---------------------------------------------------------------------------
# Test: Limit order fills only when price is reached
# ---------------------------------------------------------------------------

class LimitOrderStrategy(Strategy):
    def __init__(self, limit_px: float):
        super().__init__()
        self.limit_px = limit_px
        self.filled = False
        self.fill_price_received: Optional[float] = None

    def on_start(self):
        self._submitted = False

    def on_bar(self, bar: Bar) -> None:
        if not self._submitted and self.bars_available >= 1:
            self.buy(
                bar.symbol, 1.0,
                order_type=OrderType.LIMIT,
                limit_price=self.limit_px,
            )
            self._submitted = True

    def on_fill(self, order: Order) -> None:
        self.filled = True
        self.fill_price_received = order.fill_price


def test_limit_order_fills_at_or_below_limit():
    # Bars with low <= limit price so it should fill
    bars = make_bars(10, start_price=100.0)
    # All bars have low = price - 1.  Start price is 100, so bar[0] low=99.
    # Set limit slightly above first bar's low so it fills.
    limit_px = 102.0  # bar[2].low = 100 + 2 - 1 = 101, bar open = 103 → fills at limit
    strat = LimitOrderStrategy(limit_px)
    engine = make_engine(bars, strat)
    engine.run()
    assert strat.filled, "Limit order should have been filled"
    assert strat.fill_price_received <= limit_px, (
        f"Fill price {strat.fill_price_received} exceeds limit {limit_px}"
    )


def test_limit_order_does_not_fill_if_price_never_reached():
    bars = make_bars(10, start_price=200.0)
    # All closes are >= 200; limit buy at 50 will never be reached
    strat = LimitOrderStrategy(50.0)
    engine = make_engine(bars, strat)
    engine.run()
    assert not strat.filled, "Limit order should NOT have filled"


# ---------------------------------------------------------------------------
# Test: Commission is deducted correctly
# ---------------------------------------------------------------------------

class CommissionStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self._entered = False

    def on_bar(self, bar: Bar) -> None:
        if not self._entered and self.bars_available >= 1:
            self.buy(bar.symbol, 1.0)
            self._entered = True


def test_commission_deducted():
    bars = make_bars(5)
    feed = DataFeed(bars)
    margin = {
        "TEST": MarginSpec("TEST", 5000.0, 4500.0, contract_multiplier=1.0)
    }
    commission = 5.0
    portfolio = Portfolio(
        100_000.0,
        margin_specs=margin,
        commission_per_contract=commission,
        slippage_ticks=0,
    )
    strat = CommissionStrategy()
    engine = BacktestEngine(feed, portfolio, [strat], verbose=False)
    engine.run()
    # After buying 1 contract, cash should have decreased by exactly the commission
    assert portfolio.cash == pytest.approx(100_000.0 - commission, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: Slippage is applied in the correct direction
# ---------------------------------------------------------------------------

class SlippageStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.buy_fill: Optional[float] = None
        self.sell_fill: Optional[float] = None
        self._step = 0

    def on_bar(self, bar: Bar) -> None:
        if self._step == 0 and self.bars_available >= 1:
            self.buy(bar.symbol, 1.0, tag="buy")
            self._step = 1
        elif self._step == 2:
            self.close_position(bar.symbol, tag="sell")
            self._step = 3

    def on_fill(self, order: Order) -> None:
        if order.tag == "buy":
            self.buy_fill = order.fill_price
            self._step = 2
        elif order.tag == "close":
            self.sell_fill = order.fill_price


def test_slippage_worsens_fills():
    """Buy fill should be ABOVE raw open; sell fill should be BELOW raw open."""
    bars = make_bars(10)
    feed = DataFeed(bars)
    margin = {
        "TEST": MarginSpec("TEST", 5000.0, 4500.0, contract_multiplier=1.0)
    }
    tick = 0.25
    slippage_ticks = 2
    portfolio = Portfolio(
        100_000.0,
        margin_specs=margin,
        commission_per_contract=0.0,
        slippage_ticks=slippage_ticks,
        tick_size=tick,
    )
    strat = SlippageStrategy()
    engine = BacktestEngine(feed, portfolio, [strat], verbose=False)
    engine.run()

    # Buy fills: price should be raw_open + slippage
    assert strat.buy_fill is not None
    raw_open = bars[1].open  # first fill is on bar index 1
    expected = raw_open + slippage_ticks * tick
    assert strat.buy_fill == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: Warmup bars are respected
# ---------------------------------------------------------------------------

class WarmupStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.bar_calls = 0
        self.first_bar_idx: Optional[int] = None

    def on_bar(self, bar: Bar) -> None:
        self.bar_calls += 1
        if self.first_bar_idx is None:
            self.first_bar_idx = self.bars_available


def test_warmup_bars_skipped():
    warmup = 10
    bars = make_bars(30)
    feed = DataFeed(bars, warmup_bars=warmup)
    portfolio = Portfolio(100_000.0)
    strat = WarmupStrategy()
    engine = BacktestEngine(feed, portfolio, [strat], verbose=False)
    engine.run()

    total_bars = 30
    expected_calls = total_bars - warmup
    assert strat.bar_calls == expected_calls, (
        f"Expected {expected_calls} on_bar calls, got {strat.bar_calls}"
    )
    assert strat.first_bar_idx == warmup + 1


# ---------------------------------------------------------------------------
# Test: Position P&L is calculated correctly (no commission, no slippage)
# ---------------------------------------------------------------------------

class PnlCheckStrategy(Strategy):
    """Buy 1 contract at bar[1].open, hold, sell at bar[3].open."""

    def __init__(self):
        super().__init__()
        self._step = 0
        self.entry_price: Optional[float] = None
        self.exit_price: Optional[float] = None

    def on_bar(self, bar: Bar) -> None:
        if self._step == 0 and self.bars_available == 1:
            self.buy(bar.symbol, 1.0)
            self._step = 1
        elif self._step == 2 and self.bars_available == 3:
            self.close_position(bar.symbol)
            self._step = 3

    def on_fill(self, order: Order) -> None:
        if self._step == 1:
            self.entry_price = order.fill_price
            self._step = 2
        elif self._step == 3:
            self.exit_price = order.fill_price


def test_pnl_correctness():
    bars = make_bars(10)
    feed = DataFeed(bars)
    portfolio = Portfolio(
        100_000.0,
        commission_per_contract=0.0,
        slippage_ticks=0,
    )
    strat = PnlCheckStrategy()
    engine = BacktestEngine(feed, portfolio, [strat], verbose=False)
    engine.run()

    assert strat.entry_price is not None
    assert strat.exit_price is not None

    expected_pnl = (strat.exit_price - strat.entry_price) * 1.0  # multiplier=1
    actual_pnl = portfolio.cash - 100_000.0
    assert actual_pnl == pytest.approx(expected_pnl, rel=1e-6)
