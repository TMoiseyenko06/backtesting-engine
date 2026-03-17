"""
LiveTrader — real-time inference engine.

Flow (per 1-minute bar, triggered when Databento delivers a closed bar):
  1. Append bar to rolling buffer.
  2. If not enough bars for warmup, skip.
  3. Check for EOD: if bar_hour >= eod_hour_utc, send close if in trade.
  4. Check for SL/TP hit (local simulation — mirrors the bracket we sent).
  5. If not in trade and bar_hour < no_entry_hour_utc:
       a. Build 22-feature matrix + position/upnl context.
       b. Run model.predict_rl() → action, sl_pts, tp_pts.
       c. If action != FLAT, send PickMyTrade buy/sell webhook.
  6. Log everything.

Position state is tracked locally using the SL/TP prices we originally sent.
This is sufficient because:
  - PickMyTrade manages the actual bracket on the broker side.
  - We only need to know "are we in a trade?" to gate new entry signals.
  - reverse_order_close=True in the payload protects against edge cases.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from backtesting.data_feed import Bar
from backtesting.ml.features import make_features
from live_trader.config import LiveConfig
from live_trader.pickmytrade import PickMyTradeClient

log = logging.getLogger(__name__)

# Minimum bars buffer for feature computation (SMA-50 warmup + seq window)
_WARMUP = 60   # bars needed for SMA-50 to be meaningful
_PRICE_SCALE = 1e9


@dataclass
class _TradeState:
    """Tracks our locally-known position."""
    direction: int   = 0     # +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    sl_price: float    = 0.0
    tp_price: float    = 0.0
    entry_time: Optional[datetime] = None

    @property
    def is_flat(self) -> bool:
        return self.direction == 0

    def reset(self) -> None:
        self.direction   = 0
        self.entry_price = 0.0
        self.sl_price    = 0.0
        self.tp_price    = 0.0
        self.entry_time  = None


class LiveTrader:
    """
    Parameters
    ----------
    config  : LiveConfig
    dry_run : bool   Log signals without sending actual webhooks.
    """

    def __init__(self, config: LiveConfig, dry_run: bool = False) -> None:
        self._cfg     = config
        self._dry_run = dry_run

        # ── Load model ───────────────────────────────────────────────
        self._model, self._device = _load_model(config)
        log.info("Model loaded from %s (device=%s)", config.model_path, self._device)

        # ── PickMyTrade client ───────────────────────────────────────
        symbol_short = config.symbol.split(".")[0]   # "NQ.c.0" → "NQ"
        self._pmt = PickMyTradeClient(
            token               = config.pickmytrade_token,
            account_id          = config.pickmytrade_account_id,
            url                 = config.pickmytrade_url,
            symbol              = symbol_short,
            contracts           = config.contracts,
            multiplier          = config.multiplier,
            reverse_order_close = config.reverse_order_close,
            dry_run             = dry_run,
        )

        # ── Rolling bar buffer ───────────────────────────────────────
        min_bars = _WARMUP + config.seq_len
        self._bar_buffer: deque[Bar] = deque(maxlen=min_bars + 1)
        self._min_bars = min_bars

        # ── Position / context state ─────────────────────────────────
        self._trade = _TradeState()
        self._pos_hist:  deque[float] = deque(maxlen=config.seq_len)
        self._upnl_hist: deque[float] = deque(maxlen=config.seq_len)

        # ── Stats ────────────────────────────────────────────────────
        self._bars_seen    = 0
        self._signals_sent = 0

    # ──────────────────────────────────────────────────────────────────
    # Main entry point — called for every closed 1m bar
    # ──────────────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar) -> None:
        self._bar_buffer.append(bar)
        self._bars_seen += 1
        bar_hour = bar.timestamp.hour   # Databento timestamps are UTC

        log.debug(
            "BAR %s  O=%.2f H=%.2f L=%.2f C=%.2f  bars_seen=%d",
            bar.timestamp, bar.open, bar.high, bar.low, bar.close,
            self._bars_seen,
        )

        # ── Update rolling position/upnl history (mirrors training) ──
        self._update_context(bar)

        # ── 1. Warmup guard ───────────────────────────────────────────
        if len(self._bar_buffer) < self._min_bars:
            log.debug("Warming up: %d / %d bars", len(self._bar_buffer), self._min_bars)
            return

        # ── 2. EOD forced flat ────────────────────────────────────────
        if bar_hour >= self._cfg.eod_hour_utc:
            if not self._trade.is_flat:
                log.info("EOD flat triggered at %s", bar.timestamp)
                self._pmt.close()
                self._trade.reset()
            return

        # ── 3. Check bracket exit (local simulation) ──────────────────
        if not self._trade.is_flat:
            self._check_bracket_exit(bar)
            if not self._trade.is_flat:
                # Still in trade — don't send new signals
                return

        # ── 4. No new entries near EOD ────────────────────────────────
        if bar_hour >= self._cfg.no_entry_hour_utc:
            return

        # ── 5. Model inference ────────────────────────────────────────
        action, sl_pts, tp_pts = self._infer(bar)

        if action == 0:
            log.debug("Model: FLAT")
            return

        # Apply overrides
        if self._cfg.sl_pts_override > 0:
            sl_pts = self._cfg.sl_pts_override
        if self._cfg.tp_pts_override > 0:
            tp_pts = self._cfg.tp_pts_override

        # Hard-cap SL to max_loss
        max_sl = self._cfg.max_loss_usd / (self._cfg.contracts * self._cfg.multiplier)
        sl_pts = min(sl_pts, max_sl)

        direction_str = "LONG" if action == 1 else "SHORT"
        log.info(
            "SIGNAL %s  SL=%.1fpts ($%.0f)  TP=%.1fpts ($%.0f)  bar=%s",
            direction_str, sl_pts, sl_pts * self._cfg.multiplier,
            tp_pts, tp_pts * self._cfg.multiplier, bar.timestamp,
        )

        # ── 6. Send webhook + update local state ─────────────────────
        direction = 1 if action == 1 else -1
        ok = (self._pmt.buy if action == 1 else self._pmt.sell)(
            sl_pts=sl_pts, tp_pts=tp_pts
        )

        if ok:
            self._signals_sent += 1
            # Record expected bracket prices for local simulation
            self._trade.direction   = direction
            self._trade.entry_price = bar.close   # approximate fill = last close
            self._trade.sl_price    = bar.close - direction * sl_pts
            self._trade.tp_price    = bar.close + direction * tp_pts
            self._trade.entry_time  = bar.timestamp
        else:
            log.error("Webhook failed — not tracking position")

    # ──────────────────────────────────────────────────────────────────
    # Start / stop
    # ──────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect to Databento live feed and run until interrupted."""
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "databento is required for live trading.\n"
                "Install it: pip install databento"
            )

        log.info(
            "Connecting to Databento live feed | dataset=%s  symbol=%s",
            self._cfg.dataset, self._cfg.symbol,
        )

        client = db.Live(key=self._cfg.databento_key)
        client.subscribe(
            dataset  = self._cfg.dataset,
            schema   = "ohlcv-1m",
            stype_in = "continuous",
            symbols  = [self._cfg.symbol],
        )

        log.info("Live feed connected. Waiting for bars...")
        try:
            for record in client:
                if not hasattr(record, "open"):
                    # Skip non-OHLCV messages (heartbeats, system messages, etc.)
                    continue
                bar = _databento_to_bar(record, self._cfg)
                self.on_bar(bar)
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        if not self._trade.is_flat:
            log.info("Shutdown: closing open position via PickMyTrade.")
            self._pmt.close()
            self._trade.reset()
        log.info(
            "LiveTrader stopped. bars_seen=%d  signals_sent=%d",
            self._bars_seen, self._signals_sent,
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _check_bracket_exit(self, bar: Bar) -> None:
        """
        Simulate bracket order fills using bar HIGH/LOW.
        Mirrors the same logic used in backtesting/order_manager.py.
        """
        d = self._trade.direction
        hit_sl = (d == 1  and bar.low  <= self._trade.sl_price) or \
                 (d == -1 and bar.high >= self._trade.sl_price)
        hit_tp = (d == 1  and bar.high >= self._trade.tp_price) or \
                 (d == -1 and bar.low  <= self._trade.tp_price)

        if hit_sl or hit_tp:
            tag   = "SL" if hit_sl else "TP"
            price = self._trade.sl_price if hit_sl else self._trade.tp_price
            pnl   = (price - self._trade.entry_price) * d * self._cfg.multiplier
            log.info(
                "BRACKET EXIT (%s)  exit≈%.2f  pnl≈$%.0f  duration=%s",
                tag, price, pnl,
                bar.timestamp - self._trade.entry_time
                if self._trade.entry_time else "?",
            )
            self._trade.reset()

    def _update_context(self, bar: Bar) -> None:
        """Maintain rolling pos/upnl history for model input (matches training)."""
        d = self._trade.direction
        if d != 0 and self._trade.entry_price > 0:
            upnl_raw = ((bar.close - self._trade.entry_price)
                        * d * self._cfg.multiplier / self._cfg.reward_scale)
            upnl = float(np.clip(upnl_raw, -10.0, 10.0))
        else:
            upnl = 0.0
        self._pos_hist.append(float(d))
        self._upnl_hist.append(upnl)

    def _infer(self, bar: Bar):
        """
        Run model inference on the current bar.

        Returns
        -------
        action : int     0=flat, 1=long, 2=short
        sl_pts : float   stop-loss distance in instrument points
        tp_pts : float   take-profit distance in instrument points
        """
        import torch

        bars      = list(self._bar_buffer)
        features  = make_features(bars)[-self._cfg.seq_len:]   # (seq_len, 22)

        seq = self._cfg.seq_len
        pad = seq - len(self._pos_hist)
        pos_enc  = np.array(
            [0.0] * pad + list(self._pos_hist),  dtype=np.float32
        ).reshape(-1, 1)
        upnl_arr = np.array(
            [0.0] * pad + list(self._upnl_hist), dtype=np.float32
        ).reshape(-1, 1)

        x_aug = np.concatenate([features, pos_enc, upnl_arr], axis=-1)  # (seq, 24)
        x_t   = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0).to(self._device)

        desired_pos, _confidence, sl_pts, tp_pts = self._model.predict_rl(x_t)

        # desired_pos: -1 short, 0 flat, +1 long → remap to 0/1/2
        if desired_pos == 0:
            return 0, sl_pts, tp_pts
        action = 1 if desired_pos == 1 else 2
        return action, sl_pts, tp_pts


# ──────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────

def _load_model(cfg: LiveConfig):
    """Load a trained ActorCriticLSTM from disk."""
    import torch
    from backtesting.ml.actor_critic import ActorCriticLSTM

    device = torch.device(cfg.device)
    model  = ActorCriticLSTM(hidden_size=cfg.hidden_size)
    state  = torch.load(cfg.model_path, map_location=device, weights_only=True)
    # Handle both raw state_dict saves and wrapped checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def _databento_to_bar(record, cfg: LiveConfig) -> Bar:
    """
    Convert a Databento OHLCVMsg to a backtesting Bar.

    Databento stores prices as int64 fixed-point (divide by 1e9).
    Timestamps are nanoseconds since Unix epoch (UTC).
    """
    symbol_short = cfg.symbol.split(".")[0]
    ts = datetime.fromtimestamp(record.ts_event / 1e9, tz=timezone.utc).replace(tzinfo=None)

    return Bar(
        timestamp          = ts,
        symbol             = symbol_short,
        open               = record.open  / cfg.price_scale,
        high               = record.high  / cfg.price_scale,
        low                = record.low   / cfg.price_scale,
        close              = record.close / cfg.price_scale,
        volume             = float(record.volume),
        contract_multiplier= cfg.multiplier,
    )
