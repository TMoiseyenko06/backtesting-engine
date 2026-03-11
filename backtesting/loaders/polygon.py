"""
Polygon.io data loader
-----------------------
Free tier  : 2 years of 1m/5m/15m/1h aggregates for US stocks & ETFs.
             Rate limit: 5 API calls / minute.
Paid plans : Futures (NQ, ES, CL …) require Starter ($29/mo) or higher.

Free-tier NQ proxy: use "QQQ" (Nasdaq-100 ETF) — tracks NQ-100 almost 1:1.

Sign up   : https://polygon.io  (no credit card for free tier)
API docs  : https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier__timespan__from__to

Usage
-----
    from backtesting.loaders.polygon import from_polygon

    # Free tier — QQQ as NQ proxy (2 years of 1m data)
    feed = from_polygon(
        api_key="YOUR_KEY",
        ticker="QQQ",
        symbol="NQ",               # label bars as NQ
        multiplier=5,
        timespan="minute",         # minute | hour | day
        start="2023-01-01",
        end="2024-01-01",
        contract_multiplier=20.0,
        warmup_bars=20,
    )

    # Paid tier — actual NQ futures
    feed = from_polygon(
        api_key="YOUR_KEY",
        ticker="NQ",               # futures ticker on Polygon
        multiplier=5,
        timespan="minute",
        start="2023-01-01",
        end="2024-01-01",
        contract_multiplier=20.0,
    )
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from backtesting.data_feed import DataFeed


# Polygon free tier: 5 requests/minute
_RATE_LIMIT_DELAY = 12.5  # seconds between requests to stay under limit


def from_polygon(
    api_key: str,
    ticker: str,
    multiplier: int,
    timespan: str,
    start: str,
    end: str,
    contract_multiplier: float = 1.0,
    warmup_bars: int = 0,
    symbol: Optional[str] = None,
    adjusted: bool = True,
    limit: int = 50_000,
    rate_limit_delay: float = _RATE_LIMIT_DELAY,
) -> DataFeed:
    """
    Fetch OHLCV aggregates from Polygon.io and return a DataFeed.

    Parameters
    ----------
    api_key : str
        Your Polygon.io API key.
    ticker : str
        Polygon ticker, e.g. ``"QQQ"``, ``"SPY"``, or futures ``"NQ"``
        (futures require a paid plan).
    multiplier : int
        Bar size multiplier, e.g. ``5`` for 5-minute bars.
    timespan : str
        ``"minute"``, ``"hour"``, or ``"day"``.
    start : str
        Start date ``"YYYY-MM-DD"``.
    end : str
        End date ``"YYYY-MM-DD"`` (inclusive).
    contract_multiplier : float
        Futures contract size ($ per point).
    warmup_bars : int
        Silent warm-up bars before strategies receive on_bar() calls.
    symbol : str | None
        Override the symbol name in Bar objects (defaults to ``ticker``).
    adjusted : bool
        Whether to return adjusted prices (default True).
    limit : int
        Max results per API request (Polygon max is 50,000).
    rate_limit_delay : float
        Seconds to wait between paginated requests (free tier = 12.5s).

    Returns
    -------
    DataFeed
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is not installed. Run: pip install requests")

    bar_symbol = symbol or ticker
    base_url = "https://api.polygon.io/v2/aggs/ticker"
    all_results = []

    # Polygon returns at most `limit` bars per request; paginate with cursor
    url = (
        f"{base_url}/{ticker}/range/{multiplier}/{timespan}"
        f"/{start}/{end}"
        f"?adjusted={'true' if adjusted else 'false'}"
        f"&sort=asc&limit={limit}&apiKey={api_key}"
    )

    page = 0
    while url:
        page += 1
        if page > 1:
            time.sleep(rate_limit_delay)

        resp = requests.get(url, timeout=30)
        if resp.status_code == 429:
            print("Polygon rate limit hit — waiting 60s...")
            time.sleep(60)
            resp = requests.get(url, timeout=30)

        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        if status == "ERROR":
            raise ValueError(
                f"Polygon API error: {data.get('error', data)}"
            )

        results = data.get("results", [])
        all_results.extend(results)

        # Follow pagination cursor if present
        next_url = data.get("next_url")
        url = f"{next_url}&apiKey={api_key}" if next_url else None

    if not all_results:
        raise ValueError(
            f"Polygon returned no data for {ticker} "
            f"{multiplier}/{timespan} {start}→{end}. "
            "Check your API key, ticker, and subscription tier."
        )

    df = pd.DataFrame(all_results)
    # Polygon fields: t=timestamp(ms), o=open, h=high, l=low, c=close, v=volume, vw=vwap, n=trades
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")

    return DataFeed.from_dataframe(
        df,
        symbol=bar_symbol,
        contract_multiplier=contract_multiplier,
        warmup_bars=warmup_bars,
    )
