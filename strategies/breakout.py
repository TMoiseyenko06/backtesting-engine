"""
Donchian Channel Breakout Strategy
------------------------------------
A classic channel-breakout system for futures.

Rules
-----
Entry long  : Close > highest high of the last `entry_period` bars
Entry short : Close < lowest  low  of the last `entry_period` bars
Exit long   : Close < lowest  low  of the last `exit_period` bars
Exit short  : Close > highest high of the last `exit_period` bars

Zero lookahead: channel levels computed on the last N *closed* bars,
excluding the current bar's close (which would introduce lookahead if used
to compute the channel AND trigger the entry on the same bar).
The entry signal on bar N fills at bar N+1's open.
"""

from __future__ import annotations

from backtesting.data_feed import Bar
from backtesting.strategy import Strategy


class BreakoutStrategy(Strategy):
    """
    Parameters
    ----------
    entry_period : int
        Channel lookback for entries (default 20).
    exit_period : int
        Channel lookback for exits (default 10).
    contracts : float
    use_atr_filter : bool
        Only trade when ATR > atr_threshold (volatility filter).
    atr_period : int
    atr_threshold : float
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        contracts: float = 1.0,
        use_atr_filter: bool = False,
        atr_period: int = 14,
        atr_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.contracts = contracts
        self.use_atr_filter = use_atr_filter
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self._warmup = entry_period + atr_period + 2

    def on_start(self) -> None:
        self.name = f"Breakout({self.entry_period}/{self.exit_period})"

    def on_bar(self, bar: Bar) -> None:
        if self.bars_available < self._warmup:
            return

        # Use history EXCLUDING the current bar for channel calculation
        # history(n) already returns only closed bars — the current bar IS
        # the most recently closed bar, so we use history(entry_period + 1)
        # and exclude the last element to avoid using today's close in the
        # channel that determines today's signal.
        n = self.entry_period + 1
        hist = self.history(n)

        # Exclude current bar from channel to prevent in-bar lookahead
        channel_bars = hist[:-1]  # bars before the current one
        if len(channel_bars) < self.entry_period:
            return

        entry_high = max(b.high for b in channel_bars[-self.entry_period:])
        entry_low  = min(b.low  for b in channel_bars[-self.entry_period:])

        exit_hist = channel_bars[-self.exit_period:] if len(channel_bars) >= self.exit_period else channel_bars
        exit_high = max(b.high for b in exit_hist)
        exit_low  = min(b.low  for b in exit_hist)

        symbol = bar.symbol
        pos = self.position(symbol)

        # ATR filter
        if self.use_atr_filter:
            full_hist = self.history(self.atr_period + 5)
            atr_val = self.atr(full_hist, self.atr_period)
            if atr_val < self.atr_threshold:
                return

        # Exit logic
        if pos > 0 and bar.close < exit_low:
            self.close_position(symbol, tag="exit_long")
            return
        if pos < 0 and bar.close > exit_high:
            self.close_position(symbol, tag="exit_short")
            return

        # Entry logic — breakout of prior bars' channel
        if pos == 0:
            if bar.close > entry_high:
                self.buy(symbol, self.contracts, tag="breakout_long")
            elif bar.close < entry_low:
                self.sell(symbol, self.contracts, tag="breakout_short")
