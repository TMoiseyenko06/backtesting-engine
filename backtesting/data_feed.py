"""
DataFeed — strict lookahead-free OHLCV bar iterator.

Rules enforced here:
  - Bars are yielded one at a time in chronological order.
  - Strategy code can only call `current_bar()` or `history(n)`, which
    both return data for bars that have already *closed*.
  - The current (live) bar is never exposed until it closes.
  - `history(n)` returns at most the last `n` *closed* bars, never the
    current or future bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence
import pandas as pd


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar for one futures contract."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float = 0.0
    # Multiplier (contract size) — e.g. 50 for ES, 20 for NQ
    contract_multiplier: float = 1.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def vwap_proxy(self) -> float:
        """Intra-bar VWAP approximation (typical * volume). NOT lookahead."""
        return self.typical_price * self.volume


class DataFeed:
    """
    Iterates over sorted OHLCV bars without exposing future data.

    Parameters
    ----------
    bars : sequence of Bar
        Pre-sorted (ascending timestamp) bar data.
    warmup_bars : int
        Number of bars to consume silently before the engine starts trading.
        Signals/indicators that need N bars of history should set this to N.
    """

    def __init__(self, bars: Sequence[Bar], warmup_bars: int = 0) -> None:
        if not bars:
            raise ValueError("bars must not be empty")
        # Validate strict chronological order — catches accidental future leaks
        for i in range(1, len(bars)):
            if bars[i].timestamp <= bars[i - 1].timestamp:
                raise ValueError(
                    f"Bars must be strictly ascending. "
                    f"Bar[{i}].timestamp={bars[i].timestamp} <= "
                    f"Bar[{i-1}].timestamp={bars[i-1].timestamp}"
                )
        self._bars: List[Bar] = list(bars)
        self._warmup_bars = warmup_bars
        self._cursor: int = -1          # index of the most recently closed bar
        self._total = len(self._bars)
        self._in_warmup: bool = True

    # ------------------------------------------------------------------
    # Internal iteration (used by the engine only)
    # ------------------------------------------------------------------

    def _advance(self) -> Optional[Bar]:
        """
        Move to the next bar.  Returns the newly closed bar, or None if
        the feed is exhausted.  Called exclusively by BacktestEngine.
        """
        next_idx = self._cursor + 1
        if next_idx >= self._total:
            return None
        self._cursor = next_idx
        if self._cursor >= self._warmup_bars:
            self._in_warmup = False
        return self._bars[self._cursor]

    @property
    def is_warming_up(self) -> bool:
        return self._in_warmup

    @property
    def current_index(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # Public API — safe for strategy code
    # ------------------------------------------------------------------

    def current_bar(self) -> Optional[Bar]:
        """
        Return the most recently *closed* bar.
        Returns None during warm-up or before the first bar.
        """
        if self._cursor < 0:
            return None
        return self._bars[self._cursor]

    def history(self, n: int) -> List[Bar]:
        """
        Return the last `n` closed bars (oldest first).
        Never includes the current live bar.
        Raises ValueError if n < 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if self._cursor < 0:
            return []
        start = max(0, self._cursor - n + 1)
        return self._bars[start : self._cursor + 1]

    def bars_available(self) -> int:
        """Number of closed bars available to the strategy."""
        return max(0, self._cursor + 1)

    def remaining_bars(self) -> int:
        """Number of bars not yet processed."""
        return self._total - self._cursor - 1

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    # Maximum lookback period yfinance supports per intraday interval.
    # Exceeding these silently returns truncated or empty data.
    _YF_MAX_PERIOD: dict = {
        "1m":  "30d",
        "2m":  "60d",
        "5m":  "60d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d",
        "1h":  "730d",
        "4h":  "60d",   # undocumented but works in practice
        "90m": "60d",
    }

    @classmethod
    def from_yfinance(
        cls,
        ticker: str,
        contract_multiplier: float = 1.0,
        warmup_bars: int = 0,
        symbol: Optional[str] = None,
        **download_kwargs,
    ) -> "DataFeed":
        """
        Download data directly from Yahoo Finance and return a DataFeed.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol, e.g. ``"NQ=F"``, ``"ES=F"``, ``"CL=F"``.
        contract_multiplier : float
            Contract size multiplier for P&L calculation.
        warmup_bars : int
            Number of bars to consume silently before the engine starts trading.
        symbol : str | None
            Override the symbol name stored in each Bar (defaults to ``ticker``).
        **download_kwargs
            Passed directly to ``yfinance.download()``.  Common options:

            interval : str
                ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"`` …
                If omitted, defaults to ``"1d"``.
            period : str
                ``"7d"``, ``"60d"``, ``"730d"``, ``"3y"``, ``"max"`` …
                If omitted, the maximum allowed period for the interval is used
                automatically so you never silently get truncated data.
            start / end : str
                Alternative to ``period``, e.g. ``start="2023-01-01"``.
            auto_adjust : bool
                Default ``True``.

        yfinance intraday limits
        ------------------------
        =======  ==========
        1m       last 30 days
        5m       last 60 days
        15m      last 60 days
        1h       last 730 days
        4h       last 60 days
        =======  ==========

        Examples
        --------
        >>> feed = DataFeed.from_yfinance("NQ=F", contract_multiplier=20,
        ...                               interval="5m")          # 60-day default

        >>> feed = DataFeed.from_yfinance("NQ=F", contract_multiplier=20,
        ...                               interval="1h", period="730d")

        >>> feed = DataFeed.from_yfinance("NQ=F", contract_multiplier=20,
        ...                               interval="1d",
        ...                               start="2020-01-01", end="2024-01-01")
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is not installed. Run: pip install yfinance"
            )

        kwargs = {"progress": False, "auto_adjust": True}
        kwargs.update(download_kwargs)

        # Default interval to daily
        interval = kwargs.setdefault("interval", "1d")

        # Auto-set max period for intraday intervals when the caller hasn't
        # specified either period or start/end — prevents silent truncation.
        if "period" not in kwargs and "start" not in kwargs and "end" not in kwargs:
            max_period = cls._YF_MAX_PERIOD.get(interval)
            if max_period:
                kwargs["period"] = max_period

        df = yf.download(ticker, **kwargs)

        if df.empty:
            raise ValueError(
                f"yfinance returned no data for ticker '{ticker}'. "
                "Check the symbol and date range."
            )

        # yfinance returns a MultiIndex when downloading a single ticker with
        # recent versions: columns are (Price, Ticker) tuples.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [price_level.lower() for price_level, _ in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # The date is in the index (named 'Date' or 'Datetime')
        df = df.reset_index()
        idx_col = df.columns[0]  # 'Date' or 'Datetime'
        df = df.rename(columns={idx_col: "timestamp"})

        # Drop rows with NaN OHLCV (can appear at end of intraday data)
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Strip timezone info — engine uses naive datetimes internally
        if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df.sort_values("timestamp").reset_index(drop=True)

        bar_symbol = symbol or ticker
        oi_col = "open_interest" if "open_interest" in df.columns else None

        bars = [
            Bar(
                timestamp=row["timestamp"].to_pydatetime(),
                symbol=bar_symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                open_interest=float(row[oi_col]) if oi_col else 0.0,
                contract_multiplier=contract_multiplier,
            )
            for _, row in df.iterrows()
        ]
        return cls(bars, warmup_bars=warmup_bars)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        contract_multiplier: float = 1.0,
        warmup_bars: int = 0,
    ) -> "DataFeed":
        """
        Build a DataFeed from a pandas DataFrame.

        Handles both flat and MultiIndex column DataFrames (e.g. from yfinance).
        Expected columns (case-insensitive):
            timestamp | datetime | date  →  parsed as datetime
            open, high, low, close, volume
        Optional: open_interest
        """
        df = df.copy()

        # Flatten MultiIndex columns (yfinance returns these for single tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [price_level.lower() for price_level, _ in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # If date is in the index rather than a column, move it to a column
        if df.index.name and df.index.name.lower() in ("date", "datetime", "timestamp", "time"):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamp"})

        # Normalise timestamp column
        ts_col = next(
            (c for c in ("timestamp", "datetime", "date", "time") if c in df.columns),
            None,
        )
        if ts_col is None:
            raise ValueError("DataFrame must have a timestamp/datetime/date column")
        df["timestamp"] = pd.to_datetime(df[ts_col])

        # Strip timezone info
        if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df.sort_values("timestamp").reset_index(drop=True)

        oi_col = "open_interest" if "open_interest" in df.columns else None

        bars = [
            Bar(
                timestamp=row["timestamp"].to_pydatetime(),
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                open_interest=float(row[oi_col]) if oi_col else 0.0,
                contract_multiplier=contract_multiplier,
            )
            for _, row in df.iterrows()
        ]
        return cls(bars, warmup_bars=warmup_bars)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        symbol: str,
        contract_multiplier: float = 1.0,
        warmup_bars: int = 0,
        **read_csv_kwargs,
    ) -> "DataFeed":
        """Load from a CSV file via pandas."""
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_dataframe(
            df, symbol, contract_multiplier=contract_multiplier, warmup_bars=warmup_bars
        )
