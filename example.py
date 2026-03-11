"""
Quick-start example — runs all four built-in strategies on synthetic ES data.

Run with:
    python example.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List

from backtesting.data_feed import Bar, DataFeed
from backtesting.engine import BacktestEngine
from backtesting.portfolio import Portfolio, MarginSpec
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy


# ---------------------------------------------------------------------------
# Generate realistic-ish synthetic ES futures data
# ---------------------------------------------------------------------------

def make_synthetic_bars(n: int = 500, symbol: str = "ES") -> List[Bar]:
    random.seed(1234)
    base = datetime(2021, 1, 4)
    bars = []
    price = 3700.0
    for i in range(n):
        # Random-walk with slight upward drift
        change = random.gauss(0.5, 15.0)
        open_ = price
        close = open_ + change
        high = max(open_, close) + abs(random.gauss(0, 5))
        low = min(open_, close) - abs(random.gauss(0, 5))
        bars.append(
            Bar(
                timestamp=base + timedelta(days=i),
                symbol=symbol,
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=round(random.uniform(50_000, 200_000)),
                open_interest=round(random.uniform(2_000_000, 3_000_000)),
                contract_multiplier=50.0,
            )
        )
        price = close
    return bars


# ---------------------------------------------------------------------------
# Helper to run a single strategy
# ---------------------------------------------------------------------------

def run_strategy(strategy, bars: List[Bar], initial_cash: float = 500_000.0):
    symbol = bars[0].symbol
    feed = DataFeed(bars)
    margin_specs = {
        symbol: MarginSpec(
            symbol=symbol,
            initial_margin_per_contract=12_000.0,
            maintenance_margin_per_contract=10_900.0,
            contract_multiplier=50.0,
        )
    }
    portfolio = Portfolio(
        initial_cash=initial_cash,
        margin_specs=margin_specs,
        commission_per_contract=2.0,   # $2 per contract per side
        slippage_ticks=1,              # 1 tick slippage
        tick_size=0.25,
    )
    engine = BacktestEngine(feed, portfolio, [strategy], verbose=False)
    result = engine.run()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bars = make_synthetic_bars(500)

    strategies = [
        EMACrossoverStrategy(fast_period=9, slow_period=21, contracts=1),
        RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70, contracts=1),
        BreakoutStrategy(entry_period=20, exit_period=10, contracts=1),
        TrendFollowingStrategy(ema_period=30, atr_period=14, risk_pct=0.01),
    ]

    for strat in strategies:
        result = run_strategy(strat, bars)
        print(f"\nStrategy: {strat.name}")
        result.print_summary()

    # --- Example: load from CSV ---
    # from backtesting.data_feed import DataFeed
    # feed = DataFeed.from_csv(
    #     "data/ES_daily.csv",
    #     symbol="ES",
    #     contract_multiplier=50.0,
    #     warmup_bars=50,
    # )
    # ... then build Portfolio and BacktestEngine as above
