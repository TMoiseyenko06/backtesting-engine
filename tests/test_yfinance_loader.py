"""
Tests for DataFeed.from_yfinance() and the improved from_dataframe().

We mock yfinance.download() so tests run offline and deterministically.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from backtesting.data_feed import DataFeed


# ---------------------------------------------------------------------------
# Helpers — build DataFrames that mimic what yfinance actually returns
# ---------------------------------------------------------------------------

def _make_yf_multiindex_df(n: int = 50, ticker: str = "ES=F") -> pd.DataFrame:
    """
    Mimic yfinance MultiIndex output (yfinance >= 0.2.x, single ticker).
    Columns: MultiIndex [(Price, Ticker), ...]
    Index: DatetimeIndex named 'Date'
    """
    dates = pd.date_range("2023-01-02", periods=n, freq="B", tz="America/New_York")
    price = 4000.0
    rows = []
    for _ in range(n):
        rows.append({
            "open": price,
            "high": price + 5,
            "low": price - 3,
            "close": price + 2,
            "volume": 100_000.0,
        })
        price += 1.0

    df = pd.DataFrame(rows, index=dates)
    df.index.name = "Date"
    # Build MultiIndex columns as yfinance does
    df.columns = pd.MultiIndex.from_tuples(
        [(c.capitalize(), ticker) for c in df.columns],
        names=["Price", "Ticker"],
    )
    return df


def _make_yf_flat_df(n: int = 30) -> pd.DataFrame:
    """Flat-column DataFrame (older yfinance style or single-ticker downloads)."""
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    price = 100.0
    rows = []
    for _ in range(n):
        rows.append({
            "Open": price,
            "High": price + 2,
            "Low": price - 1,
            "Close": price + 1,
            "Volume": 50_000.0,
        })
        price += 0.5
    df = pd.DataFrame(rows, index=dates)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Tests: from_yfinance()
# ---------------------------------------------------------------------------

def test_from_yfinance_multiindex():
    """from_yfinance() correctly parses MultiIndex yfinance output."""
    mock_df = _make_yf_multiindex_df(50, "ES=F")

    with patch("yfinance.download", return_value=mock_df):
        feed = DataFeed.from_yfinance("ES=F", contract_multiplier=50.0, period="3mo")

    assert feed.bars_available() == 0  # no bars consumed yet
    bar = feed._advance()
    assert bar is not None
    assert bar.symbol == "ES=F"
    assert bar.contract_multiplier == 50.0
    assert bar.open > 0
    assert bar.high >= bar.low
    # Timezone stripped — naive datetime
    assert bar.timestamp.tzinfo is None


def test_from_yfinance_symbol_override():
    """symbol parameter overrides the ticker name in bars."""
    mock_df = _make_yf_multiindex_df(10, "ES=F")

    with patch("yfinance.download", return_value=mock_df):
        feed = DataFeed.from_yfinance("ES=F", symbol="ES", contract_multiplier=50.0)

    bar = feed._advance()
    assert bar.symbol == "ES"


def test_from_yfinance_warmup_respected():
    """warmup_bars parameter is passed through correctly."""
    mock_df = _make_yf_multiindex_df(40, "NQ=F")
    warmup = 10

    with patch("yfinance.download", return_value=mock_df):
        feed = DataFeed.from_yfinance("NQ=F", warmup_bars=warmup)

    # Advance past warmup
    for _ in range(warmup):
        feed._advance()
    assert feed.is_warming_up

    feed._advance()
    assert not feed.is_warming_up


def test_from_yfinance_empty_raises():
    """Empty download raises a clear ValueError."""
    empty_df = pd.DataFrame()

    with patch("yfinance.download", return_value=empty_df):
        with pytest.raises(ValueError, match="no data"):
            DataFeed.from_yfinance("FAKE=F")


def test_from_yfinance_missing_package():
    """ImportError raised with helpful message when yfinance not installed."""
    import sys
    original = sys.modules.get("yfinance")
    sys.modules["yfinance"] = None  # simulate missing module

    try:
        with pytest.raises((ImportError, TypeError)):
            DataFeed.from_yfinance("ES=F")
    finally:
        if original is not None:
            sys.modules["yfinance"] = original
        else:
            del sys.modules["yfinance"]


# ---------------------------------------------------------------------------
# Tests: from_dataframe() with MultiIndex (yfinance-style)
# ---------------------------------------------------------------------------

def test_from_dataframe_multiindex():
    """from_dataframe() handles MultiIndex columns transparently."""
    df = _make_yf_multiindex_df(20, "CL=F")
    df = df.reset_index()  # move Date into a column

    feed = DataFeed.from_dataframe(df, symbol="CL", contract_multiplier=1000.0)
    bar = feed._advance()
    assert bar is not None
    assert bar.symbol == "CL"
    assert bar.contract_multiplier == 1000.0


def test_from_dataframe_date_in_index():
    """from_dataframe() handles DataFrames where the date is the index."""
    df = _make_yf_flat_df(15)
    # Leave date in index (common from yfinance)
    feed = DataFeed.from_dataframe(df, symbol="TEST")
    bar = feed._advance()
    assert bar is not None
    assert bar.open > 0


def test_from_dataframe_timezone_stripped():
    """Timezone-aware timestamps are stripped to naive UTC datetimes."""
    df = _make_yf_multiindex_df(10)
    df = df.reset_index()
    feed = DataFeed.from_dataframe(df, symbol="TZ_TEST")
    bar = feed._advance()
    assert bar.timestamp.tzinfo is None


def test_from_dataframe_chronological_order_enforced():
    """Bars returned by from_dataframe() are strictly ascending."""
    df = _make_yf_flat_df(20)
    # Shuffle deliberately — from_dataframe sorts, DataFeed validates
    df_shuffled = df.sample(frac=1, random_state=7).reset_index()
    feed = DataFeed.from_dataframe(df_shuffled, symbol="SHUF")
    prev_ts = None
    while True:
        bar = feed._advance()
        if bar is None:
            break
        if prev_ts is not None:
            assert bar.timestamp > prev_ts
        prev_ts = bar.timestamp


# ---------------------------------------------------------------------------
# Integration: full engine run with mocked yfinance data
# ---------------------------------------------------------------------------

def test_full_backtest_with_yfinance_data():
    """Run a complete backtest using DataFeed.from_yfinance() mock data."""
    from backtesting.engine import BacktestEngine
    from backtesting.portfolio import Portfolio, MarginSpec
    from strategies.ema_crossover import EMACrossoverStrategy

    mock_df = _make_yf_multiindex_df(200, "ES=F")

    with patch("yfinance.download", return_value=mock_df):
        feed = DataFeed.from_yfinance("ES=F", contract_multiplier=50.0, period="1y")

    margin = {
        "ES=F": MarginSpec("ES=F", 12_000, 10_900, contract_multiplier=50.0)
    }
    portfolio = Portfolio(500_000, margin_specs=margin, commission_per_contract=2.0)
    strategy = EMACrossoverStrategy(fast_period=9, slow_period=21, contracts=1)
    engine = BacktestEngine(feed, portfolio, [strategy], verbose=False)
    result = engine.run()

    assert result.bar_count == 200
    assert result.analytics.final_equity > 0
