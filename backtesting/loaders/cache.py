"""
Local file cache for DataFeed bar data.

Download once, store to disk, reload instantly on every subsequent run.
Supports both CSV and Parquet (Parquet is ~10× smaller and faster).

Usage
-----
    from backtesting.loaders.cache import save_feed, load_feed_from_cache
    from backtesting.loaders.polygon import from_polygon

    CACHE = "data/NQ_5m_2023.parquet"

    try:
        feed = load_feed_from_cache(CACHE, symbol="NQ", contract_multiplier=20)
        print("Loaded from cache.")
    except FileNotFoundError:
        feed = from_polygon(api_key="...", ticker="QQQ", ...)
        save_feed(feed, CACHE)
        print("Downloaded and cached.")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from backtesting.data_feed import DataFeed, Bar


def save_feed(feed: DataFeed, path: str | Path) -> None:
    """
    Persist all bars in a DataFeed to a CSV or Parquet file.

    The file format is determined by the extension:
        .parquet  →  Parquet (recommended: faster, smaller)
        .csv      →  CSV (human-readable)

    The saved file can be reloaded with :func:`load_feed_from_cache`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "timestamp":           b.timestamp.isoformat(),
            "symbol":              b.symbol,
            "open":                b.open,
            "high":                b.high,
            "low":                 b.low,
            "close":               b.close,
            "volume":              b.volume,
            "open_interest":       b.open_interest,
            "contract_multiplier": b.contract_multiplier,
        }
        for b in feed._bars
    ]
    df = pd.DataFrame(rows)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"Saved {len(rows)} bars to {path}")


def load_feed_from_cache(
    path: str | Path,
    symbol: Optional[str] = None,
    contract_multiplier: Optional[float] = None,
    warmup_bars: int = 0,
) -> DataFeed:
    """
    Load a DataFeed from a previously cached CSV or Parquet file.

    Parameters
    ----------
    path : str | Path
        File saved by :func:`save_feed`.
    symbol : str | None
        Override the symbol stored in the file (optional).
    contract_multiplier : float | None
        Override the contract multiplier stored in the file (optional).
    warmup_bars : int

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist — caller should then download
        and call :func:`save_feed`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {path}\n"
            "Download data first and call save_feed() to create the cache."
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    bars = [
        Bar(
            timestamp=row["timestamp"].to_pydatetime(),
            symbol=symbol or str(row["symbol"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            open_interest=float(row.get("open_interest", 0.0)),
            contract_multiplier=(
                contract_multiplier
                if contract_multiplier is not None
                else float(row.get("contract_multiplier", 1.0))
            ),
        )
        for _, row in df.iterrows()
    ]

    print(f"Loaded {len(bars)} bars from cache: {path}")
    return DataFeed(bars, warmup_bars=warmup_bars)
