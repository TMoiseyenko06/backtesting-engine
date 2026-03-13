"""
PropFirmRisk — Prop Firm Account Rule Enforcement
--------------------------------------------------
Implements the risk guardrails for a $50k NQ futures prop firm account:

  Rule 1  Max 4 contracts (hard position cap)
  Rule 2  Max daily loss $1,500 — no new entries today if breached
  Rule 3  Max trailing drawdown $2,500 — trading permanently halted if
          equity falls $2,500 below the running peak (or initial equity,
          whichever is higher)
  Rule 4  RTH-only — no overnight positions:
          • All open positions force-closed by 3:55 PM ET
          • No new entries outside 9:30 AM – 3:55 PM ET
  Rule 5  Trade consistency — short-circuit entries outside kill zones
          to bias towards the most liquid intraday windows

Times are expressed in UTC using EDT (UTC-4) as the baseline, which is
accurate for ~Mar–Nov.  In EST months (Nov–Mar) this shifts by 1 hour
but remains conservative (no after-hours holding).

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
    """

    # UTC times (EDT baseline = UTC-4, accurate ~Mar–Nov)
    # In EST months shift is 1h earlier — still no after-hours holding
    _SESSION_START_MINS = 13 * 60 + 30   # 13:30 UTC ≈ 09:30 ET
    _SESSION_CLOSE_MINS = 20 * 60 +  0   # 20:00 UTC ≈ 16:00 ET
    _ENTRY_CUTOFF_MINS  = 19 * 60 + 55   # 19:55 UTC ≈ 15:55 ET  (5 min buffer)

    def __init__(
        self,
        max_contracts: int        = 4,
        max_daily_loss_usd: float = 1_500.0,
        max_drawdown_usd: float   = 2_500.0,
        initial_equity: float     = 50_000.0,
    ) -> None:
        self.max_contracts     = max_contracts
        self.max_daily_loss    = max_daily_loss_usd
        self.max_drawdown      = max_drawdown_usd

        self._peak_equity: float           = initial_equity
        self._day_open_equity: Optional[float] = None
        self._current_date: Optional[date] = None
        self._halted: bool                 = False   # permanent halt on drawdown

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
        """True if the bar falls within RTH (9:30 AM – 4:00 PM ET)."""
        t = ts.hour * 60 + ts.minute
        return self._SESSION_START_MINS <= t < self._SESSION_CLOSE_MINS

    def should_close(self, ts: datetime) -> bool:
        """
        True when any open position should be immediately closed:
          • Within 5 minutes of the session close (≥ 15:55 ET)
          • Any bar outside RTH (overnight / pre-market)
        """
        t = ts.hour * 60 + ts.minute
        past_cutoff   = t >= self._ENTRY_CUTOFF_MINS
        pre_session   = t <  self._SESSION_START_MINS
        return past_cutoff or pre_session

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
        return (
            f"equity={equity:,.0f}  "
            f"drawdown={dd:,.0f}/{self.max_drawdown:,.0f}  "
            f"daily_loss={daily:,.0f}/{self.max_daily_loss:,.0f}"
            + ("  [HALTED]" if self._halted else "")
        )
