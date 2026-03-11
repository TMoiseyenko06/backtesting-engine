"""
Quick-start example — runs all four built-in strategies on NQ futures data.

Uses DataFeed.from_yfinance() to pull real NQ=F data, with a synthetic
fallback if the download fails (e.g. no internet connection).

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

# NQ (Nasdaq-100 E-mini) contract spec
NQ_SYMBOL          = "NQ=F"
NQ_MULTIPLIER      = 20.0    # $20 per point
NQ_INITIAL_MARGIN  = 21_000.0
NQ_MAINT_MARGIN    = 19_000.0
NQ_TICK_SIZE       = 0.25
INITIAL_CASH       = 500_000.0


# ---------------------------------------------------------------------------
# Fallback: synthetic NQ-like bars
# ---------------------------------------------------------------------------

def make_synthetic_nq_bars(n: int = 500) -> List[Bar]:
    random.seed(1234)
    base = datetime(2021, 1, 4)
    bars = []
    price = 13_000.0
    for i in range(n):
        change = random.gauss(2.0, 80.0)   # NQ moves more than ES
        open_ = price
        close = open_ + change
        high = max(open_, close) + abs(random.gauss(0, 20))
        low  = min(open_, close) - abs(random.gauss(0, 20))
        bars.append(Bar(
            timestamp=base + timedelta(days=i),
            symbol=NQ_SYMBOL,
            open=round(open_, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close, 2),
            volume=round(random.uniform(20_000, 80_000)),
            open_interest=round(random.uniform(200_000, 400_000)),
            contract_multiplier=NQ_MULTIPLIER,
        ))
        price = close
    return bars


# ---------------------------------------------------------------------------
# Helper to run a single strategy
# ---------------------------------------------------------------------------

def run_strategy(strategy, feed: DataFeed) -> object:
    margin_specs = {
        NQ_SYMBOL: MarginSpec(
            symbol=NQ_SYMBOL,
            initial_margin_per_contract=NQ_INITIAL_MARGIN,
            maintenance_margin_per_contract=NQ_MAINT_MARGIN,
            contract_multiplier=NQ_MULTIPLIER,
        )
    }
    portfolio = Portfolio(
        initial_cash=INITIAL_CASH,
        margin_specs=margin_specs,
        commission_per_contract=2.0,
        slippage_ticks=1,
        tick_size=NQ_TICK_SIZE,
    )
    engine = BacktestEngine(feed, portfolio, [strategy], verbose=False)
    return engine.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Try to load real NQ data from Yahoo Finance
    feed = None
    try:
        print("Downloading NQ=F data from Yahoo Finance...")
        feed = DataFeed.from_yfinance(
            NQ_SYMBOL,
            contract_multiplier=NQ_MULTIPLIER,
            warmup_bars=50,
            period="3y",
            interval="1d",
        )
        print(f"Loaded {feed._total} bars of real NQ data.\n")
    except Exception as e:
        print(f"Download failed ({e}), using synthetic data.\n")

    strategies = [
        EMACrossoverStrategy(fast_period=9, slow_period=21, contracts=1),
        RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70, contracts=1),
        BreakoutStrategy(entry_period=20, exit_period=10, contracts=1),
        TrendFollowingStrategy(ema_period=30, atr_period=14, risk_pct=0.01),
    ]

    for strat in strategies:
        # Each strategy gets a fresh feed from the same data
        if feed is not None:
            strategy_feed = DataFeed.from_yfinance(
                NQ_SYMBOL,
                contract_multiplier=NQ_MULTIPLIER,
                warmup_bars=50,
                period="3y",
                interval="1d",
            )
        else:
            strategy_feed = DataFeed(make_synthetic_nq_bars(500))

        result = run_strategy(strat, strategy_feed)
        print(f"Strategy: {strat.name}")
        result.print_summary()
        print()
