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

import csv
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

        Expected columns (case-insensitive):
            timestamp | datetime | date  →  parsed as datetime
            open, high, low, close, volume
        Optional: open_interest
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Normalise timestamp column
        ts_col = next(
            (c for c in ("timestamp", "datetime", "date", "time") if c in df.columns),
            None,
        )
        if ts_col is None:
            raise ValueError("DataFrame must have a timestamp/datetime/date column")
        df["timestamp"] = pd.to_datetime(df[ts_col])
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
