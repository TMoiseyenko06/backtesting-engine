"""
Alpha Vantage data loader
--------------------------
Free tier : 25 requests / day.
            Each request with month= gives one full month of intraday bars.
            → 12 requests = 1 year of data (fits easily in a free daily quota).

Sign up   : https://www.alphavantage.co/support/#api-key
API docs  : https://www.alphavantage.co/documentation/#intraday

Supported intervals : 1min, 5min, 15min, 30min, 60min
Supported tickers   : US equities and ETFs (QQQ, SPY, …).
                      No direct futures support — use QQQ as NQ proxy.

Usage
-----
    from backtesting.loaders.alpha_vantage import from_alpha_vantage

    feed = from_alpha_vantage(
        api_key="YOUR_KEY",
        ticker="QQQ",
        interval="5min",
        start="2023-01-01",
        end="2024-01-01",
        contract_multiplier=20.0,
        symbol="NQ",
        warmup_bars=20,
    )
"""

from __future__ import annotations

import time
from datetime import datetime, date
from typing import Optional

import pandas as pd

from backtesting.data_feed import DataFeed


# Free tier: 25 req/day → space requests 15s apart to be safe during a run
_REQUEST_DELAY = 15.0


def from_alpha_vantage(
    api_key: str,
    ticker: str,
    interval: str,
    start: str,
    end: str,
    contract_multiplier: float = 1.0,
    warmup_bars: int = 0,
    symbol: Optional[str] = None,
    request_delay: float = _REQUEST_DELAY,
) -> DataFeed:
    """
    Fetch intraday bars from Alpha Vantage (TIME_SERIES_INTRADAY) and
    return a DataFeed.

    Downloads month-by-month (one API call per month) then stitches the
    results together, so a full year costs only 12 API calls.

    Parameters
    ----------
    api_key : str
        Your Alpha Vantage API key.
    ticker : str
        Stock / ETF ticker, e.g. ``"QQQ"``, ``"SPY"``.
    interval : str
        ``"1min"``, ``"5min"``, ``"15min"``, ``"30min"``, or ``"60min"``.
    start : str
        Start date ``"YYYY-MM-DD"``.
    end : str
        End date ``"YYYY-MM-DD"`` (inclusive).
    contract_multiplier : float
        Futures contract size used for P&L ($ per point).
    warmup_bars : int
        Silent warm-up bars.
    symbol : str | None
        Override the symbol label in Bar objects (defaults to ``ticker``).
    request_delay : float
        Seconds to wait between requests (free tier: 15s recommended).

    Returns
    -------
    DataFeed
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is not installed. Run: pip install requests")

    bar_symbol = symbol or ticker
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

    # Build list of YYYY-MM month strings to fetch
    months = _month_range(start_dt, end_dt)
    all_frames = []

    for i, month in enumerate(months):
        if i > 0:
            time.sleep(request_delay)

        url = (
            "https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_INTRADAY"
            f"&symbol={ticker}"
            f"&interval={interval}"
            f"&month={month}"
            f"&outputsize=full"
            f"&datatype=json"
            f"&apikey={api_key}"
        )

        try:
            import requests as req
            resp = req.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  Warning: failed to fetch {month}: {exc}")
            continue

        if "Note" in data:
            print(f"  Alpha Vantage rate limit hit on {month}. Waiting 60s...")
            time.sleep(60)
            resp = req.get(url, timeout=30)
            data = resp.json()

        if "Error Message" in data:
            raise ValueError(
                f"Alpha Vantage error: {data['Error Message']}. "
                "Check your ticker and API key."
            )

        key = f"Time Series ({interval})"
        ts = data.get(key, {})
        if not ts:
            print(f"  Warning: no data returned for {month}")
            continue

        rows = []
        for ts_str, ohlcv in ts.items():
            rows.append({
                "timestamp": pd.to_datetime(ts_str),
                "open":   float(ohlcv["1. open"]),
                "high":   float(ohlcv["2. high"]),
                "low":    float(ohlcv["3. low"]),
                "close":  float(ohlcv["4. close"]),
                "volume": float(ohlcv["5. volume"]),
            })
        all_frames.append(pd.DataFrame(rows))
        print(f"  Fetched {len(rows):>6} bars for {month}")

    if not all_frames:
        raise ValueError(
            f"Alpha Vantage returned no data for {ticker} {interval} "
            f"{start}→{end}. Check your API key and ticker."
        )

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Trim to exact requested date range
    df = df[
        (df["timestamp"] >= pd.Timestamp(start))
        & (df["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(days=1))
    ]

    return DataFeed.from_dataframe(
        df,
        symbol=bar_symbol,
        contract_multiplier=contract_multiplier,
        warmup_bars=warmup_bars,
    )


def _month_range(start: date, end: date):
    """Yield 'YYYY-MM' strings for every month between start and end."""
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current.strftime("%Y-%m")
        # Advance one month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
