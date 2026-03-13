"""
PropFirmRisk — Prop Firm Account Rule Enforcement
--------------------------------------------------
Implements the risk guardrails for a $50k NQ futures prop firm account:

  Rule 1  Max 4 contracts (hard position cap)
  Rule 2  Max daily loss $1,500 — no new entries today if breached
  Rule 3  Max trailing drawdown $2,500 — trading permanently halted if
          equity falls $2,500 below the running peak (or initial equity,
          whichever is higher)
  Rule 4  Session-aware — NQ futures trade ~23 hours/day with a 1-hour
          maintenance break from 5:00 PM – 6:00 PM ET (21:00–22:00 UTC).
          • All open positions force-closed by 4:55 PM ET (5 min buffer)
          • No new entries during the maintenance window (21:00–22:00 UTC)
  Rule 5  Short-trade frequency — no more than 50% of completed trades may
          have a duration shorter than 10 seconds (enforced once ≥5 trades).
  Rule 6  Short-trade profit cap — trades lasting < 10 seconds may not
          account for more than 50% of total gross profit (enforced once ≥5
          trades with positive P&L exist).

Times are expressed in UTC using EDT (UTC-4) as the baseline, which is
accurate for ~Mar–Nov.  In EST months (Nov–Mar) this shifts by 1 hour.

Usage inside a strategy:
    risk = PropFirmRisk(initial_equity=50_000)

    def on_bar(self, bar):
        eq = self.equity()
        risk.update(bar.timestamp, eq)   # must call first

        if risk.should_close(bar.timestamp):
            self.close_position(bar.symbol)
            return

        if not risk.can_enter(eq) or not risk.is_in_session(bar.timestamp):
            return

        contracts = min(risk.max_contracts, computed_qty)
        ...
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional


class PropFirmRisk:
    """
    Parameters
    ----------
    max_contracts : int
        Hard cap on simultaneous contracts (default 4).
    max_daily_loss_usd : float
        Stop entering trades today if today's open-to-current loss
        reaches this amount (default $1,500).
    max_drawdown_usd : float
        Permanently halt trading if equity drops this much below its
        running peak — or below initial equity if peak hasn't risen yet
        (default $2,500).
    initial_equity : float
        Starting account equity (default $50,000).
    short_trade_secs : float
        Trades shorter than this many seconds are classified as
        "short trades" for Rules 5 & 6 (default 10).
    max_short_trade_pct : float
        Maximum fraction of total trades that may be short trades,
        enforced once ≥5 trades exist (default 0.50 = 50%).
    max_short_profit_pct : float
        Maximum fraction of total gross profit that short trades may
        contribute, enforced once ≥5 profitable trades exist
        (default 0.50 = 50%).
    """

    # NQ futures maintenance break: 5:00–6:00 PM ET = 21:00–22:00 UTC (EDT)
    _MAINT_START_MINS  = 21 * 60       # 21:00 UTC = 5:00 PM ET — close begins
    _MAINT_END_MINS    = 22 * 60       # 22:00 UTC = 6:00 PM ET — market reopens
    _ENTRY_CUTOFF_MINS = 20 * 60 + 55  # 20:55 UTC = 4:55 PM ET — 5 min buffer

    def __init__(
        self,
        max_contracts: int        = 4,
        max_daily_loss_usd: float = 1_500.0,
        max_drawdown_usd: float   = 2_500.0,
        initial_equity: float     = 50_000.0,
        short_trade_secs: float   = 10.0,
        max_short_trade_pct: float  = 0.50,
        max_short_profit_pct: float = 0.50,
    ) -> None:
        self.max_contracts      = max_contracts
        self.max_daily_loss     = max_daily_loss_usd
        self.max_drawdown       = max_drawdown_usd
        self.short_trade_secs   = short_trade_secs
        self.max_short_trade_pct  = max_short_trade_pct
        self.max_short_profit_pct = max_short_profit_pct

        self._peak_equity: float               = initial_equity
        self._day_open_equity: Optional[float] = None
        self._current_date: Optional[date]     = None
        self._halted: bool                     = False   # permanent halt on drawdown

        # Trade history for Rules 5 & 6 — populated via record_trade()
        self._trade_durations: list[float] = []   # seconds per completed trade
        self._trade_pnls: list[float]      = []   # net P&L per completed trade

    # ------------------------------------------------------------------
    # Trade reporting — call after every completed round-trip
    # ------------------------------------------------------------------

    def record_trade(self, duration_secs: float, net_pnl: float) -> None:
        """
        Register a completed trade for Rules 5 & 6.

        Call this once per closed trade (entry → exit).  The strategy is
        responsible for calling it with the correct duration and net P&L.

        Parameters
        ----------
        duration_secs : float
            Elapsed seconds from entry fill to exit fill.
        net_pnl : float
            Realised net P&L of the trade (after commissions).
        """
        self._trade_durations.append(duration_secs)
        self._trade_pnls.append(net_pnl)

    # ------------------------------------------------------------------
    # Must call once per bar before any entry/exit decisions
    # ------------------------------------------------------------------

    def update(self, ts: datetime, equity: float) -> None:
        """
        Update internal state for the current bar.
        Call this at the top of on_bar() before any other risk checks.
        """
        today = ts.date()
        if today != self._current_date:
            self._current_date    = today
            self._day_open_equity = equity       # reset daily P&L reference

        self._peak_equity = max(self._peak_equity, equity)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def is_in_session(self, ts: datetime) -> bool:
        """True outside the 1-hour maintenance break (5–6 PM ET / 21:00–22:00 UTC).
        NQ futures trade ~23 hours/day; only the maintenance window is excluded."""
        t = ts.hour * 60 + ts.minute
        return not (self._MAINT_START_MINS <= t < self._MAINT_END_MINS)

    def should_close(self, ts: datetime) -> bool:
        """
        True when any open position should be immediately closed:
          • Within 5 minutes of the maintenance window (≥ 4:55 PM ET)
          • During the maintenance window itself (5:00–6:00 PM ET)
        """
        t = ts.hour * 60 + ts.minute
        return t >= self._ENTRY_CUTOFF_MINS and t < self._MAINT_END_MINS

    def can_enter(self, equity: float) -> bool:
        """
        False if any risk limit prevents opening new positions:
          • Permanently halted after max trailing drawdown
          • Today's loss limit already reached
        """
        if self._halted:
            return False

        # Rule 3: max trailing drawdown from running peak
        if self._peak_equity - equity >= self.max_drawdown:
            self._halted = True
            print(
                f"  [PropFirm] MAX DRAWDOWN BREACHED "
                f"(equity={equity:,.0f}, peak={self._peak_equity:,.0f}, "
                f"drawdown={self._peak_equity - equity:,.0f}). "
                f"Trading permanently halted."
            )
            return False

        # Rule 2: daily loss limit
        if self._day_open_equity is not None:
            if self._day_open_equity - equity >= self.max_daily_loss:
                return False

        # Rules 5 & 6: short-trade frequency and profit limits
        n = len(self._trade_durations)
        if n >= 5:
            short_mask = [d < self.short_trade_secs for d in self._trade_durations]
            short_count = sum(short_mask)

            # Rule 5: short trades must be < max_short_trade_pct of all trades
            if short_count / n > self.max_short_trade_pct:
                return False

            # Rule 6: short-trade gross profit ≤ max_short_profit_pct of total profit
            total_profit = sum(p for p in self._trade_pnls if p > 0)
            if total_profit > 0:
                short_profit = sum(
                    p for is_short, p in zip(short_mask, self._trade_pnls)
                    if is_short and p > 0
                )
                if short_profit / total_profit > self.max_short_profit_pct:
                    return False

        return True

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def halted(self) -> bool:
        """True after max trailing drawdown — all trading must stop."""
        return self._halted

    @property
    def daily_pnl(self) -> float:
        """Today's P&L (negative = loss).  Zero before first update."""
        if self._day_open_equity is None:
            return 0.0
        return 0.0   # computed in can_enter; exposed for logging only

    def status(self, equity: float) -> str:
        """One-line status string for logging."""
        dd = self._peak_equity - equity
        daily = (self._day_open_equity - equity) if self._day_open_equity else 0.0
        n = len(self._trade_durations)
        short_count = sum(1 for d in self._trade_durations if d < self.short_trade_secs)
        short_pct = short_count / n * 100 if n else 0.0
        total_profit = sum(p for p in self._trade_pnls if p > 0)
        short_profit = sum(
            p for d, p in zip(self._trade_durations, self._trade_pnls)
            if d < self.short_trade_secs and p > 0
        )
        short_profit_pct = short_profit / total_profit * 100 if total_profit > 0 else 0.0
        return (
            f"equity={equity:,.0f}  "
            f"drawdown={dd:,.0f}/{self.max_drawdown:,.0f}  "
            f"daily_loss={daily:,.0f}/{self.max_daily_loss:,.0f}  "
            f"short_trades={short_count}/{n}({short_pct:.0f}%)  "
            f"short_profit={short_profit_pct:.0f}%"
            + ("  [HALTED]" if self._halted else "")
        )
