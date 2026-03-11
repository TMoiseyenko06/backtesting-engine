"""
Integration tests: run example strategies on synthetic data and verify
they produce sane (not necessarily profitable) results without errors.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List

import pytest

from backtesting.data_feed import Bar, DataFeed
from backtesting.engine import BacktestEngine
from backtesting.portfolio import Portfolio, MarginSpec
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy


def make_trending_bars(n: int = 200, symbol: str = "ES") -> List[Bar]:
    """Upward-trending bars with some noise."""
    import random
    random.seed(42)
    base = datetime(2022, 1, 1)
    bars = []
    price = 4000.0
    for i in range(n):
        noise = random.gauss(0, 5)
        trend = 0.5
        open_ = price
        close = price + trend + noise
        high = max(open_, close) + random.uniform(0, 3)
        low = min(open_, close) - random.uniform(0, 3)
        bars.append(Bar(
            timestamp=base + timedelta(days=i),
            symbol=symbol,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=10000.0,
            contract_multiplier=50.0,
        ))
        price = close
    return bars


def run_strategy(strategy, bars, initial_cash=500_000.0):
    symbol = bars[0].symbol
    feed = DataFeed(bars)
    margin = {
        symbol: MarginSpec(
            symbol=symbol,
            initial_margin_per_contract=12_000.0,
            maintenance_margin_per_contract=10_900.0,
            contract_multiplier=50.0,
        )
    }
    portfolio = Portfolio(
        initial_cash,
        margin_specs=margin,
        commission_per_contract=2.0,
        slippage_ticks=1,
        tick_size=0.25,
    )
    engine = BacktestEngine(feed, portfolio, [strategy], verbose=False)
    result = engine.run()
    return result


def test_ema_crossover_runs():
    bars = make_trending_bars(300)
    strat = EMACrossoverStrategy(fast_period=9, slow_period=21, contracts=1)
    result = run_strategy(strat, bars)
    assert result.analytics.total_trades >= 0
    assert not math.isnan(result.analytics.final_equity)
    assert result.analytics.final_equity > 0


def test_rsi_mean_reversion_runs():
    bars = make_trending_bars(300)
    strat = RSIMeanReversionStrategy(rsi_period=14, contracts=1)
    result = run_strategy(strat, bars)
    assert result.analytics.total_trades >= 0
    assert not math.isnan(result.analytics.final_equity)


def test_breakout_runs():
    bars = make_trending_bars(300)
    strat = BreakoutStrategy(entry_period=20, exit_period=10, contracts=1)
    result = run_strategy(strat, bars)
    assert result.analytics.total_trades >= 0
    assert not math.isnan(result.analytics.final_equity)


def test_trend_following_runs():
    bars = make_trending_bars(300)
    strat = TrendFollowingStrategy(ema_period=20, atr_period=14, risk_pct=0.01)
    result = run_strategy(strat, bars)
    assert result.analytics.total_trades >= 0
    assert not math.isnan(result.analytics.final_equity)


def test_analytics_metrics_sane():
    bars = make_trending_bars(300)
    strat = EMACrossoverStrategy(fast_period=9, slow_period=21, contracts=1)
    result = run_strategy(strat, bars)
    a = result.analytics

    assert 0 <= a.win_rate <= 100
    assert a.max_drawdown >= 0
    assert a.max_drawdown_pct >= 0
    assert a.profit_factor >= 0
    # Win + lose counts should equal total trades
    assert a.winning_trades + a.losing_trades == a.total_trades


def test_result_summary_prints():
    bars = make_trending_bars(100)
    strat = EMACrossoverStrategy()
    result = run_strategy(strat, bars)
    summary = result.summary()
    assert "BACKTEST RESULTS" in summary
    assert "Sharpe" in summary
