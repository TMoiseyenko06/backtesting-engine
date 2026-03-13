"""Tests for PropFirmRisk rule enforcement."""

from __future__ import annotations

from datetime import datetime

import pytest

from strategies.prop_firm_risk import PropFirmRisk


def _ts(hour: int, minute: int = 0) -> datetime:
    """UTC timestamp on an arbitrary trading day."""
    return datetime(2024, 6, 15, hour, minute)   # Saturday in tests — doesn't matter


class TestSessionHours:

    def test_inside_rth(self):
        r = PropFirmRisk()
        # 10:00 AM ET = 14:00 UTC (EDT)
        assert r.is_in_session(_ts(14, 0)) is True
        assert r.is_in_session(_ts(13, 30)) is True   # exact open
        assert r.is_in_session(_ts(19, 59)) is True   # one minute before close

    def test_outside_rth_premarket(self):
        r = PropFirmRisk()
        assert r.is_in_session(_ts(12, 0)) is False   # 8:00 AM ET
        assert r.is_in_session(_ts(0, 0))  is False   # midnight

    def test_outside_rth_afterhours(self):
        r = PropFirmRisk()
        assert r.is_in_session(_ts(20, 0)) is False   # 4:00 PM ET exactly
        assert r.is_in_session(_ts(21, 0)) is False   # 5:00 PM ET

    def test_should_close_at_entry_cutoff(self):
        r = PropFirmRisk()
        # Entry cutoff is 19:55 UTC (3:55 PM ET)
        assert r.should_close(_ts(19, 55)) is True
        assert r.should_close(_ts(19, 54)) is False   # one minute before

    def test_should_close_overnight(self):
        r = PropFirmRisk()
        assert r.should_close(_ts(22, 0)) is True    # 6 PM ET — after hours
        assert r.should_close(_ts(8,  0)) is True    # pre-market


class TestDrawdownRules:

    def test_max_drawdown_halts_trading(self):
        r = PropFirmRisk(initial_equity=50_000, max_drawdown_usd=2_500)
        r.update(_ts(14), 50_000)

        # Loss of exactly $2,500 → halted
        assert r.can_enter(47_500) is False
        assert r.halted is True

    def test_drawdown_below_threshold_ok(self):
        # Set daily loss limit high so it doesn't interfere with drawdown test
        r = PropFirmRisk(initial_equity=50_000, max_drawdown_usd=2_500,
                         max_daily_loss_usd=10_000)
        r.update(_ts(14), 50_000)
        assert r.can_enter(48_000) is True    # only $2,000 down — under $2,500 limit

    def test_peak_equity_rises(self):
        r = PropFirmRisk(initial_equity=50_000, max_drawdown_usd=2_500,
                         max_daily_loss_usd=10_000)
        r.update(_ts(14), 53_000)   # equity rose — peak now $53k
        # Must fall $2,500 from $53k = below $50,500 to halt
        assert r.can_enter(51_000) is True    # $2k below new peak — fine
        assert r.can_enter(50_400) is False   # $2,600 below new peak — halted

    def test_halted_blocks_permanently(self):
        r = PropFirmRisk(initial_equity=50_000, max_drawdown_usd=2_500)
        r.update(_ts(14), 50_000)
        r.can_enter(47_000)   # trigger halt
        # Even if equity somehow recovers, still halted
        assert r.can_enter(52_000) is False


class TestDailyLossLimit:

    def test_daily_loss_blocks_entry(self):
        r = PropFirmRisk(initial_equity=50_000, max_daily_loss_usd=1_500)
        r.update(_ts(14), 50_000)    # day opens at $50k
        # Drop $1,500 exactly → blocked
        assert r.can_enter(48_500) is False

    def test_daily_loss_resets_next_day(self):
        r = PropFirmRisk(initial_equity=50_000, max_daily_loss_usd=1_500)
        # Day 1: lose $1,500
        r.update(datetime(2024, 6, 15, 14, 0), 50_000)
        assert r.can_enter(48_500) is False   # blocked today

        # Day 2: new session open
        r.update(datetime(2024, 6, 16, 14, 0), 48_500)
        assert r.can_enter(48_000) is True    # only $500 down today — fine

    def test_daily_loss_under_limit_ok(self):
        r = PropFirmRisk(initial_equity=50_000, max_daily_loss_usd=1_500)
        r.update(_ts(14), 50_000)
        assert r.can_enter(49_000) is True    # $1,000 down — under limit


class TestContractCap:

    def test_max_contracts_default(self):
        r = PropFirmRisk(max_contracts=4)
        assert r.max_contracts == 4

    def test_custom_max_contracts(self):
        r = PropFirmRisk(max_contracts=2)
        assert r.max_contracts == 2
