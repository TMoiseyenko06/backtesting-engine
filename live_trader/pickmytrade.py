"""
PickMyTrade webhook client.

PickMyTrade receives a JSON payload and forwards it as a live broker order
(Tradovate, Rithmic, IB, TradeStation, etc.).

Webhook docs: https://docs.pickmytrade.trade/docs/tradingview-json-alert-configuration/
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Actions the PickMyTrade webhook accepts
ACTION_BUY   = "buy"
ACTION_SELL  = "sell"
ACTION_CLOSE = "close"


class PickMyTradeClient:
    """
    Sends trade signals to PickMyTrade via HTTP webhook.

    Parameters
    ----------
    token        : str   Your PickMyTrade API token (from the dashboard).
    account_id   : str   Your broker account ID (from the dashboard).
    url          : str   Webhook endpoint (Tradovate vs. other brokers differ).
    symbol       : str   Instrument, e.g. "NQ" (without expiry suffix).
    contracts    : int   Number of contracts per trade.
    multiplier   : float Contract point value (NQ=$20, ES=$50).
    reverse_order_close : bool
        Tell PickMyTrade to close any existing position before entering a new
        one.  Strongly recommended to avoid accidental double-entries when the
        live trader restarts mid-trade.
    timeout      : int   HTTP timeout in seconds.
    dry_run      : bool  Log payloads but don't actually send HTTP requests.
    """

    def __init__(
        self,
        token: str,
        account_id: str,
        url: str,
        symbol: str,
        contracts: int = 1,
        multiplier: float = 20.0,
        reverse_order_close: bool = True,
        timeout: int = 10,
        dry_run: bool = False,
    ) -> None:
        self._token    = token
        self._account  = account_id
        self._url      = url
        self._symbol   = symbol.upper().split(".")[0]  # "NQ.c.0" → "NQ"
        self._qty      = contracts
        self._mult     = multiplier
        self._reverse  = reverse_order_close
        self._timeout  = timeout
        self._dry_run  = dry_run
        self._session  = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def buy(self, sl_pts: float, tp_pts: float) -> bool:
        """Place a long market order with bracket SL/TP."""
        return self._send(ACTION_BUY, sl_pts=sl_pts, tp_pts=tp_pts)

    def sell(self, sl_pts: float, tp_pts: float) -> bool:
        """Place a short market order with bracket SL/TP."""
        return self._send(ACTION_SELL, sl_pts=sl_pts, tp_pts=tp_pts)

    def close(self) -> bool:
        """Close any open position (EOD flat or bankruptcy stop)."""
        return self._send(ACTION_CLOSE)

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    def _send(
        self,
        action: str,
        sl_pts: float = 0.0,
        tp_pts: float = 0.0,
    ) -> bool:
        """
        Build and POST the PickMyTrade JSON payload.

        SL/TP are sent as dollar amounts (dollar_sl / dollar_tp) so
        PickMyTrade doesn't need to know the exact fill price upfront.
        For NQ with 1 contract: dollar_sl = sl_pts * 20.

        Returns True on success, False on any error.
        """
        dollar_sl = round(sl_pts * self._mult * self._qty, 2) if sl_pts > 0 else 0
        dollar_tp = round(tp_pts * self._mult * self._qty, 2) if tp_pts > 0 else 0

        payload = {
            "symbol":               self._symbol,
            "date":                 datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data":                 action,
            "quantity":             self._qty,
            "risk_percentage":      0,
            "price":                0,            # market order — no limit price
            "tp":                   0,
            "percentage_tp":        0,
            "dollar_tp":            dollar_tp,
            "sl":                   0,
            "dollar_sl":            dollar_sl,
            "percentage_sl":        0,
            "trail":                0,
            "trail_stop":           0,
            "trail_trigger":        0,
            "trail_freq":           0,
            "update_tp":            False,
            "update_sl":            False,
            "breakeven":            0,
            "token":                self._token,
            "pyramid":              False,
            "reverse_order_close":  self._reverse if action != ACTION_CLOSE else False,
            "order_type":           "MKT",
            "account_id":           self._account,
            "multiple_accounts":    [],
        }

        log.info(
            "PMT webhook | action=%-5s  sl=$%-7.0f  tp=$%-7.0f  symbol=%s",
            action, dollar_sl, dollar_tp, self._symbol,
        )

        if self._dry_run:
            log.info("DRY RUN — payload not sent: %s", payload)
            return True

        try:
            resp = self._session.post(self._url, json=payload, timeout=self._timeout)
            if resp.status_code == 200:
                log.info("PMT response 200: %s", resp.text[:200])
                return True
            else:
                log.error(
                    "PMT webhook failed: HTTP %d — %s",
                    resp.status_code, resp.text[:500],
                )
                return False
        except requests.RequestException as exc:
            log.error("PMT webhook error: %s", exc)
            return False
