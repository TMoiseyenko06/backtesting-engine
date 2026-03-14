"""
Databento data loader — ohlcv-1m schema
-----------------------------------------
Supports two workflows:

1. **Batch file** (recommended) — download once from the Databento web UI,
   load the local .dbn / .dbn.zstd file.  No API key needed at runtime.

2. **Live API** — stream directly from the Databento Historical API.
   Pay-per-query; charged on download.

Install  : pip install databento
Sign up  : https://databento.com
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

Usage — batch file
------------------
    from backtesting.loaders.databento import from_databento_file

    feed = from_databento_file(
        path="glbx-mdp3-20210313-20260313.ohlcv-1m.dbn.zstd",
        symbol="NQ",              # label for Bar objects
        contract_multiplier=20.0,
        warmup_bars=20,
    )

Usage — live API
----------------
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

from pathlib import Path
from typing import Optional, Union

from backtesting.data_feed import DataFeed


# CME Globex dataset — covers all CME/CBOT/NYMEX/COMEX futures
_DATASET = "GLBX.MDP3"
_SCHEMA = "ohlcv-1m"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _dbn_store_to_feed(
    store,
    symbol: str,
    contract_multiplier: float,
    warmup_bars: int,
) -> DataFeed:
    """Convert a databento.DBNStore to a DataFeed."""
    import pandas as pd

    df = store.to_df()

    if df.empty:
        raise ValueError("Databento returned an empty DataFrame.")

    # ts_event is the nanosecond-resolution bar-close timestamp and is the
    # index after to_df(); reset so we can rename it uniformly.
    df = df.reset_index()

    ts_col = next(
        (c for c in df.columns if c == "ts_event" or "timestamp" in c.lower()),
        None,
    )
    if ts_col is None:
        raise ValueError(
            f"Could not find a timestamp column. Columns: {list(df.columns)}"
        )
    df = df.rename(columns={ts_col: "timestamp"})

    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing expected columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    return DataFeed.from_dataframe(
        df[["timestamp", "open", "high", "low", "close", "volume"]],
        symbol=symbol,
        contract_multiplier=contract_multiplier,
        warmup_bars=warmup_bars,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def from_databento_file(
    path: Union[str, Path],
    symbol: str,
    contract_multiplier: float = 1.0,
    warmup_bars: int = 0,
) -> DataFeed:
    """
    Load ohlcv-1m bars from a local Databento batch-download file and return
    a DataFeed.

    Supports ``.dbn``, ``.dbn.zstd``, and ``.dbn.zst`` files produced by the
    Databento web UI or ``databento-cli``.

    Parameters
    ----------
    path : str | Path
        Path to the ``.dbn`` or ``.dbn.zstd`` file.
    symbol : str
        Symbol name to store in each Bar, e.g. ``"NQ"``.
    contract_multiplier : float
        Futures contract size ($ per point), e.g. ``20.0`` for NQ.
    warmup_bars : int
        Silent warm-up bars before strategies receive on_bar() calls.

    Returns
    -------
    DataFeed

    Examples
    --------
    >>> feed = from_databento_file(
    ...     path="glbx-mdp3-20210313-20260313.ohlcv-1m.dbn.zstd",
    ...     symbol="NQ",
    ...     contract_multiplier=20.0,
    ...     warmup_bars=20,
    ... )
    """
    try:
        import databento as db
    except ImportError:
        raise ImportError("databento is not installed. Run: pip install databento")

    store = db.DBNStore.from_file(str(path))
    return _dbn_store_to_feed(store, symbol, contract_multiplier, warmup_bars)


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
    Fetch ohlcv-1m bars from the Databento Historical API and return a
    DataFeed.  Charges apply on each download.

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
        Override the symbol name stored in each Bar. Defaults to the root
        of ``symbol`` — e.g. ``"NQ.c.0"`` → ``"NQ"``.

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
        raise ImportError("databento is not installed. Run: pip install databento")

    client = db.Historical(api_key)
    store = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=_SCHEMA,
        start=start,
        end=end,
    )

    resolved_symbol = bar_symbol or symbol.split(".")[0]
    return _dbn_store_to_feed(store, resolved_symbol, contract_multiplier, warmup_bars)
