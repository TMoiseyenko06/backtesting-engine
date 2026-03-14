"""
Databento data loader — ohlcv-1m schema
-----------------------------------------
Provides actual CME futures data (NQ, ES, CL, GC, …) with full history.
Pay-per-query pricing — a full year of NQ 1m bars costs roughly $1–3.

Install  : pip install databento
Sign up  : https://databento.com  (free account, pay only for what you download)
API docs : https://docs.databento.com/api-reference-historical/timeseries/timeseries-get-range

Dataset  : GLBX.MDP3  (CME Globex — covers NQ, ES, CL, GC, ZN, …)
Schema   : ohlcv-1m

Continuous-front-month symbols
-------------------------------
  NQ.c.0   Nasdaq-100 futures  (multiplier = 20)
  ES.c.0   S&P 500 futures     (multiplier = 50)
  CL.c.0   Crude Oil futures   (multiplier = 1000)
  GC.c.0   Gold futures        (multiplier = 100)
  ZN.c.0   10-Year Note        (multiplier = 1000)

Usage
-----
    from backtesting.loaders.databento import from_databento

    feed = from_databento(
        api_key="db-YOUR_KEY",
        symbol="NQ.c.0",
        start="2023-01-01",
        end="2024-01-01",
        contract_multiplier=20.0,
        warmup_bars=20,
    )
"""

from __future__ import annotations

from typing import Optional

from backtesting.data_feed import DataFeed


# CME Globex dataset — covers all CME/CBOT/NYMEX/COMEX futures
_DATASET = "GLBX.MDP3"
_SCHEMA = "ohlcv-1m"


def from_databento(
    api_key: str,
    symbol: str,
    start: str,
    end: str,
    contract_multiplier: float = 1.0,
    warmup_bars: int = 0,
    dataset: str = _DATASET,
    bar_symbol: Optional[str] = None,
) -> DataFeed:
    """
    Fetch ohlcv-1m bars from Databento and return a DataFeed.

    Parameters
    ----------
    api_key : str
        Databento API key (starts with ``"db-"``).
    symbol : str
        Continuous-front-month symbol, e.g. ``"NQ.c.0"``, ``"ES.c.0"``.
    start : str
        Start date ``"YYYY-MM-DD"`` or ISO-8601 datetime.
    end : str
        End date ``"YYYY-MM-DD"`` or ISO-8601 datetime (exclusive).
    contract_multiplier : float
        Futures contract size ($ per point), e.g. ``20.0`` for NQ.
    warmup_bars : int
        Silent warm-up bars before strategies receive on_bar() calls.
    dataset : str
        Databento dataset ID. Defaults to ``"GLBX.MDP3"`` (CME Globex).
    bar_symbol : str | None
        Override the symbol name stored in each Bar. Defaults to ``symbol``
        with the ``.c.0`` suffix stripped, e.g. ``"NQ.c.0"`` → ``"NQ"``.

    Returns
    -------
    DataFeed

    Examples
    --------
    >>> feed = from_databento(
    ...     api_key="db-YOUR_KEY",
    ...     symbol="NQ.c.0",
    ...     start="2023-01-01",
    ...     end="2024-01-01",
    ...     contract_multiplier=20.0,
    ... )
    """
    try:
        import databento as db
    except ImportError:
        raise ImportError(
            "databento is not installed. Run: pip install databento"
        )

    import pandas as pd

    client = db.Historical(api_key)

    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=_SCHEMA,
        start=start,
        end=end,
    )

    df = data.to_df()

    if df.empty:
        raise ValueError(
            f"Databento returned no data for {symbol} "
            f"({dataset} / {_SCHEMA}) {start} → {end}. "
            "Check your symbol, date range, and API key."
        )

    # ts_event is a nanosecond timestamp index; move to column
    df = df.reset_index()
    ts_col = next(
        (c for c in df.columns if "ts_event" in c or "timestamp" in c.lower()),
        None,
    )
    if ts_col is None:
        raise ValueError(
            f"Could not find a timestamp column in Databento DataFrame. "
            f"Columns: {list(df.columns)}"
        )
    df = df.rename(columns={ts_col: "timestamp"})

    # Ensure expected OHLCV columns exist
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Databento DataFrame is missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Default bar symbol: strip ".c.0" suffix from continuous symbol
    if bar_symbol is None:
        bar_symbol = symbol.split(".")[0]

    return DataFeed.from_dataframe(
        df[["timestamp", "open", "high", "low", "close", "volume"]],
        symbol=bar_symbol,
        contract_multiplier=contract_multiplier,
        warmup_bars=warmup_bars,
    )
