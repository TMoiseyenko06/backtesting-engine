"""
Tests for Polygon, Alpha Vantage, and cache loaders.
All external HTTP calls are mocked — no internet required.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtesting.data_feed import DataFeed
from backtesting.loaders.cache import save_feed, load_feed_from_cache
from backtesting.loaders.polygon import from_polygon
from backtesting.loaders.alpha_vantage import from_alpha_vantage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feed(n=30, symbol="QQQ") -> DataFeed:
    from datetime import timedelta
    from backtesting.data_feed import Bar
    base = datetime(2023, 1, 2)
    bars = [
        Bar(
            timestamp=base + timedelta(minutes=i * 5),
            symbol=symbol,
            open=400.0 + i,
            high=401.0 + i,
            low=399.0 + i,
            close=400.5 + i,
            volume=10_000.0,
            contract_multiplier=20.0,
        )
        for i in range(n)
    ]
    return DataFeed(bars)


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

def test_save_and_load_parquet():
    feed = _make_feed(50, "NQ")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.parquet"
        save_feed(feed, path)
        loaded = load_feed_from_cache(path)

    assert loaded._total == 50
    orig_bar = feed._bars[10]
    load_bar = loaded._bars[10]
    assert load_bar.open == pytest.approx(orig_bar.open)
    assert load_bar.close == pytest.approx(orig_bar.close)
    assert load_bar.timestamp == orig_bar.timestamp


def test_save_and_load_csv():
    feed = _make_feed(20, "QQQ")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.csv"
        save_feed(feed, path)
        loaded = load_feed_from_cache(path)

    assert loaded._total == 20


def test_load_cache_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Cache file not found"):
        load_feed_from_cache("/nonexistent/path/data.parquet")


def test_cache_symbol_override():
    feed = _make_feed(10, "QQQ")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.parquet"
        save_feed(feed, path)
        loaded = load_feed_from_cache(path, symbol="NQ", contract_multiplier=20.0)

    bar = loaded._bars[0]
    assert bar.symbol == "NQ"
    assert bar.contract_multiplier == 20.0


def test_cache_roundtrip_preserves_all_fields():
    feed = _make_feed(5, "ES")
    orig = feed._bars[2]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.parquet"
        save_feed(feed, path)
        loaded = load_feed_from_cache(path)

    bar = loaded._bars[2]
    assert bar.high == pytest.approx(orig.high)
    assert bar.low  == pytest.approx(orig.low)
    assert bar.volume == pytest.approx(orig.volume)
    assert bar.contract_multiplier == pytest.approx(orig.contract_multiplier)


# ---------------------------------------------------------------------------
# Polygon tests
# ---------------------------------------------------------------------------

def _polygon_response(n=100, ticker="QQQ", offset_bars=0):
    """Build a fake Polygon aggregates API response."""
    base_ms = int(datetime(2023, 1, 2, 9, 30, tzinfo=timezone.utc).timestamp() * 1000)
    results = [
        {
            "t": base_ms + (offset_bars + i) * 5 * 60 * 1000,  # 5-min bars in ms
            "o": 300.0 + i,
            "h": 301.0 + i,
            "l": 299.0 + i,
            "c": 300.5 + i,
            "v": 50_000.0,
            "vw": 300.2 + i,
            "n": 500,
        }
        for i in range(n)
    ]
    return {"status": "OK", "resultsCount": n, "results": results}


def test_from_polygon_basic():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _polygon_response(50, "QQQ")
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        feed = from_polygon(
            api_key="test_key",
            ticker="QQQ",
            multiplier=5,
            timespan="minute",
            start="2023-01-02",
            end="2023-01-31",
            contract_multiplier=20.0,
            symbol="NQ",
        )

    assert feed._total == 50
    bar = feed._bars[0]
    assert bar.symbol == "NQ"
    assert bar.contract_multiplier == 20.0
    assert bar.open == pytest.approx(300.0)


def test_from_polygon_paginates():
    """Simulates two pages of results (next_url present on first response)."""
    page1 = _polygon_response(50, offset_bars=0)
    page1["next_url"] = "https://api.polygon.io/v2/aggs/ticker/QQQ/range/5/minute/2023-01-02/2023-01-31?cursor=abc"
    page2 = _polygon_response(30, offset_bars=50)  # distinct timestamps

    responses = [MagicMock(), MagicMock()]
    responses[0].json.return_value = page1
    responses[0].raise_for_status = MagicMock()
    responses[1].json.return_value = page2
    responses[1].raise_for_status = MagicMock()

    call_count = 0
    def fake_get(url, **kwargs):
        nonlocal call_count
        resp = responses[min(call_count, 1)]
        call_count += 1
        return resp

    with patch("requests.get", side_effect=fake_get):
        with patch("time.sleep"):  # skip rate limit delay in tests
            feed = from_polygon(
                api_key="test_key",
                ticker="QQQ",
                multiplier=5,
                timespan="minute",
                start="2023-01-02",
                end="2023-01-31",
            )

    assert feed._total == 80  # 50 + 30


def test_from_polygon_empty_raises():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "OK", "resultsCount": 0, "results": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="no data"):
            from_polygon(
                api_key="test_key",
                ticker="FAKE",
                multiplier=5,
                timespan="minute",
                start="2023-01-01",
                end="2023-01-31",
            )


def test_from_polygon_api_error_raises():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "ERROR", "error": "Unauthorized"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="Polygon API error"):
            from_polygon(
                api_key="bad_key",
                ticker="QQQ",
                multiplier=5,
                timespan="minute",
                start="2023-01-01",
                end="2023-01-31",
            )


# ---------------------------------------------------------------------------
# Alpha Vantage tests
# ---------------------------------------------------------------------------

def _av_response(interval="5min", n=100):
    """Build a fake Alpha Vantage TIME_SERIES_INTRADAY response."""
    base = datetime(2023, 6, 1, 9, 30)
    from datetime import timedelta
    ts = {}
    for i in range(n):
        dt = base + timedelta(minutes=i * 5)
        ts[dt.strftime("%Y-%m-%d %H:%M:%S")] = {
            "1. open":   f"{350.0 + i:.4f}",
            "2. high":   f"{351.0 + i:.4f}",
            "3. low":    f"{349.0 + i:.4f}",
            "4. close":  f"{350.5 + i:.4f}",
            "5. volume": "75000",
        }
    return {
        "Meta Data": {"2. Symbol": "QQQ"},
        f"Time Series ({interval})": ts,
    }


def test_from_alpha_vantage_single_month():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _av_response("5min", 100)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with patch("time.sleep"):
            feed = from_alpha_vantage(
                api_key="test_key",
                ticker="QQQ",
                interval="5min",
                start="2023-06-01",
                end="2023-06-30",
                contract_multiplier=20.0,
                symbol="NQ",
            )

    assert feed._total == 100
    bar = feed._bars[0]
    assert bar.symbol == "NQ"
    assert bar.open == pytest.approx(350.0)


def test_from_alpha_vantage_multi_month():
    """Fetches 3 months, each with 50 bars in distinct months → 150 total."""
    months = ["2023-01", "2023-02", "2023-03"]
    responses = []
    for month in months:
        year, mo = int(month[:4]), int(month[5:])
        base = datetime(year, mo, 2, 9, 30)
        from datetime import timedelta
        ts = {}
        for i in range(50):
            dt = base + timedelta(minutes=i * 5)
            ts[dt.strftime("%Y-%m-%d %H:%M:%S")] = {
                "1. open": "350.0", "2. high": "351.0",
                "3. low": "349.0", "4. close": "350.5", "5. volume": "75000",
            }
        resp = MagicMock()
        resp.json.return_value = {"Meta Data": {}, "Time Series (5min)": ts}
        resp.raise_for_status = MagicMock()
        responses.append(resp)

    with patch("requests.get", side_effect=responses):
        with patch("time.sleep"):
            feed = from_alpha_vantage(
                api_key="test_key",
                ticker="QQQ",
                interval="5min",
                start="2023-01-01",
                end="2023-03-31",
            )

    assert feed._total == 150


def test_from_alpha_vantage_empty_raises():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"Meta Data": {}}  # no time series key
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with patch("time.sleep"):
            with pytest.raises(ValueError, match="no data"):
                from_alpha_vantage(
                    api_key="test_key",
                    ticker="QQQ",
                    interval="5min",
                    start="2023-01-01",
                    end="2023-01-31",
                )
