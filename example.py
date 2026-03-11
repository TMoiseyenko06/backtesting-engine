"""
Futures Backtesting Example — NQ across multiple timeframes.

Runs all four built-in strategies on each of:
    1m  · 5m  · 15m  · 1h  · 4h

Real data is pulled from Yahoo Finance (NQ=F).
A synthetic fallback is used if the download fails.

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
# NQ (Nasdaq-100 E-mini) contract spec
# ---------------------------------------------------------------------------

SYMBOL       = "NQ=F"
MULTIPLIER   = 20.0       # $20 per point
INIT_MARGIN  = 21_000.0
MAINT_MARGIN = 19_000.0
TICK_SIZE    = 0.25
CASH         = 500_000.0

# Timeframes to test.
# period=None → DataFeed.from_yfinance auto-selects the maximum allowed period.
# (interval, period, warmup_bars, label)
TIMEFRAMES = [
    ("1m",  None, 20, "1-Minute"),
    ("5m",  None, 20, "5-Minute"),
    ("15m", None, 20, "15-Minute"),
    ("1h",  None, 20, "1-Hour"),
    ("4h",  None, 10, "4-Hour"),
]


# ---------------------------------------------------------------------------
# Synthetic fallback — NQ-like prices at the right volatility scale
# ---------------------------------------------------------------------------

def make_synthetic_bars(n: int, interval: str) -> List[Bar]:
    random.seed(42)
    base = datetime(2024, 1, 2, 9, 30)
    vol = {"1m": 8, "5m": 18, "15m": 30, "1h": 60, "4h": 100}.get(interval, 40)
    bars, price = [], 17_500.0
    for i in range(n):
        chg   = random.gauss(0, vol)
        open_ = price
        close = open_ + chg
        high  = max(open_, close) + abs(random.gauss(0, vol * 0.3))
        low   = min(open_, close) - abs(random.gauss(0, vol * 0.3))
        bars.append(Bar(
            timestamp=base + timedelta(minutes=i),
            symbol=SYMBOL,
            open=round(open_, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close, 2),
            volume=round(random.uniform(500, 5_000)),
            contract_multiplier=MULTIPLIER,
        ))
        price = close
    return bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_feed(interval: str, period, warmup: int) -> DataFeed:
    kwargs = {"interval": interval, "warmup_bars": warmup}
    if period:
        kwargs["period"] = period
    return DataFeed.from_yfinance(SYMBOL, contract_multiplier=MULTIPLIER, **kwargs)


def make_strategies() -> list:
    return [
        EMACrossoverStrategy(fast_period=9, slow_period=21, contracts=1),
        RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70, contracts=1),
        BreakoutStrategy(entry_period=20, exit_period=10, contracts=1),
        TrendFollowingStrategy(ema_period=30, atr_period=14, risk_pct=0.01),
    ]


def run_strategy(strategy, feed: DataFeed) -> object:
    portfolio = Portfolio(
        initial_cash=CASH,
        margin_specs={SYMBOL: MarginSpec(SYMBOL, INIT_MARGIN, MAINT_MARGIN, MULTIPLIER)},
        commission_per_contract=2.0,
        slippage_ticks=1,
        tick_size=TICK_SIZE,
    )
    return BacktestEngine(feed, portfolio, [strategy], verbose=False).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for interval, period, warmup, label in TIMEFRAMES:
        print(f"\n{'='*60}")
        print(f"  TIMEFRAME: {label}  ({interval})")
        print(f"{'='*60}")

        # Download once; reuse bar list across all strategies
        try:
            print(f"  Downloading NQ=F {interval} data from Yahoo Finance...")
            master_feed = load_feed(interval, period, warmup)
            bars = master_feed._bars
            print(f"  Loaded {len(bars)} bars.\n")
        except Exception as e:
            n = {"1m": 500, "5m": 400, "15m": 300, "1h": 200, "4h": 100}[interval]
            print(f"  Download failed ({e}), using {n} synthetic bars.\n")
            bars = make_synthetic_bars(n, interval)

        for strat in make_strategies():
            # Each strategy gets a fresh feed cursor from the same bar list
            feed = DataFeed(bars, warmup_bars=warmup)
            result = run_strategy(strat, feed)
            a = result.analytics
            print(
                f"  {strat.name:<38} "
                f"Return: {a.total_return_pct:>+7.2f}%  "
                f"Trades: {a.total_trades:>3}  "
                f"WinRate: {a.win_rate:>5.1f}%  "
                f"Sharpe: {a.sharpe_ratio:>6.3f}  "
                f"MaxDD: {a.max_drawdown_pct:>5.2f}%"
            )

    print()
