"""
Analytics — performance metrics computed from equity curve and trade log.

All metrics use only realised data from the backtest.  No forward-fill or
lookahead is possible here because the equity curve is built bar-by-bar
during the simulation.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import List, Optional, Tuple

from .portfolio import Portfolio, EquitySnapshot
from .position import Trade


class Analytics:
    """Compute and format performance metrics."""

    def __init__(self, portfolio: Portfolio) -> None:
        self._portfolio = portfolio
        self._curve: List[EquitySnapshot] = portfolio.equity_curve
        self._trades: List[Trade] = portfolio.all_trades

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    @property
    def initial_equity(self) -> float:
        return self._portfolio.initial_cash

    @property
    def final_equity(self) -> float:
        return self._curve[-1].equity if self._curve else self._portfolio.cash

    @property
    def total_return(self) -> float:
        """Absolute dollar return."""
        return self.final_equity - self.initial_equity

    @property
    def total_return_pct(self) -> float:
        """Total return as a percentage of initial equity."""
        if self.initial_equity == 0:
            return 0.0
        return self.total_return / self.initial_equity * 100.0

    @property
    def equity_curve(self) -> List[Tuple]:
        """List of (timestamp, equity) tuples."""
        return [(s.timestamp, s.equity) for s in self._curve]

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown in dollars."""
        peak = float("-inf")
        max_dd = 0.0
        for snap in self._curve:
            eq = snap.equity
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as a percentage of the peak equity."""
        peak = float("-inf")
        max_dd_pct = 0.0
        for snap in self._curve:
            eq = snap.equity
            if eq > peak:
                peak = eq
            if peak > 0:
                dd_pct = (peak - eq) / peak * 100.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
        return max_dd_pct

    @property
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualised Sharpe ratio (assumes daily bars; adjust if needed)."""
        returns = self._daily_returns()
        if len(returns) < 2:
            return 0.0
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        excess = mean - risk_free_rate / 252.0
        return (excess / std) * math.sqrt(252)

    @property
    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        returns = self._daily_returns()
        if len(returns) < 2:
            return 0.0
        n = len(returns)
        mean = sum(returns) / n
        downside = [min(r, 0) for r in returns]
        downside_var = sum(d ** 2 for d in downside) / n
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
        if downside_std == 0:
            return 0.0
        excess = mean - risk_free_rate / 252.0
        return (excess / downside_std) * math.sqrt(252)

    @property
    def calmar_ratio(self) -> float:
        """Annualised return / max drawdown (absolute $)."""
        ann_return = self._annualised_return()
        if self.max_drawdown == 0:
            return float("inf")
        return ann_return / self.max_drawdown

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self._trades if t.net_pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self._trades if t.net_pnl <= 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100.0

    @property
    def avg_win(self) -> float:
        wins = [t.net_pnl for t in self._trades if t.net_pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_pnl for t in self._trades if t.net_pnl <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.net_pnl for t in self._trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self._trades if t.net_pnl < 0))
        if gross_loss == 0:
            return float("inf")
        return gross_win / gross_loss

    @property
    def expectancy(self) -> float:
        """Expected dollar value per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_return / self.total_trades

    @property
    def largest_win(self) -> float:
        wins = [t.net_pnl for t in self._trades if t.net_pnl > 0]
        return max(wins) if wins else 0.0

    @property
    def largest_loss(self) -> float:
        losses = [t.net_pnl for t in self._trades if t.net_pnl < 0]
        return min(losses) if losses else 0.0

    @property
    def total_commission(self) -> float:
        return self._portfolio.positions and sum(
            p.total_commission for p in self._portfolio.positions.values()
        ) or 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"  BACKTEST RESULTS — {self._portfolio.initial_cash:,.0f} initial capital",
            "=" * 55,
            f"  Total Return        : ${self.total_return:>12,.2f}  ({self.total_return_pct:+.2f}%)",
            f"  Final Equity        : ${self.final_equity:>12,.2f}",
            f"  Max Drawdown        : ${self.max_drawdown:>12,.2f}  ({self.max_drawdown_pct:.2f}%)",
            f"  Sharpe Ratio        : {self.sharpe_ratio:>12.4f}",
            f"  Sortino Ratio       : {self.sortino_ratio:>12.4f}",
            f"  Calmar Ratio        : {self.calmar_ratio:>12.4f}",
            "-" * 55,
            f"  Total Trades        : {self.total_trades:>12}",
            f"  Win Rate            : {self.win_rate:>11.1f}%",
            f"  Avg Win             : ${self.avg_win:>12,.2f}",
            f"  Avg Loss            : ${self.avg_loss:>12,.2f}",
            f"  Profit Factor       : {self.profit_factor:>12.4f}",
            f"  Expectancy / Trade  : ${self.expectancy:>12,.2f}",
            f"  Largest Win         : ${self.largest_win:>12,.2f}",
            f"  Largest Loss        : ${self.largest_loss:>12,.2f}",
            f"  Total Commission    : ${self.total_commission:>12,.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "total_commission": self.total_commission,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _daily_returns(self) -> List[float]:
        if len(self._curve) < 2:
            return []
        returns = []
        for i in range(1, len(self._curve)):
            prev_eq = self._curve[i - 1].equity
            curr_eq = self._curve[i].equity
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)
        return returns

    def _annualised_return(self) -> float:
        if len(self._curve) < 2:
            return 0.0
        start = self._curve[0].timestamp
        end = self._curve[-1].timestamp
        days = (end - start).total_seconds() / 86400.0
        if days <= 0:
            return 0.0
        years = days / 365.0
        if self.initial_equity <= 0:
            return 0.0
        total_ratio = self.final_equity / self.initial_equity
        if total_ratio <= 0:
            return 0.0
        return (total_ratio ** (1.0 / years) - 1.0) * self.initial_equity
