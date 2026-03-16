"""
RL Trading Strategy — Bracket Orders
--------------------------------------
Wraps a trained ActorCriticLSTM for backtesting.

When the model detects a trade opportunity, it simultaneously places:
  - A market entry order
  - A STOP order at the predicted SL price (reduce_only)
  - A LIMIT order at the predicted TP price (reduce_only)

The SL and TP distances (in NQ points) are predicted by the network itself.
Once a bracket is open the model is completely ignored — the trade either
hits SL or TP.  Only one cancel-and-resubmit event per bar is possible
(the OCO cancel in on_fill).

The ONLY hard rules enforced here are non-negotiable structural controls:
  - Intraday only : forced flat at/after ``eod_hour_utc`` (default 20:00 UTC).
                    No new entries after ``no_entry_hour_utc`` (default 19:00 UTC).
  - No flip       : since a position is always managed by a live bracket, the
                    model cannot reverse while in a trade.  It must wait for the
                    bracket to resolve before entering the opposite direction.
"""

from __future__ import annotations

import numpy as np

from backtesting.data_feed import Bar
from backtesting.order import Order, OrderType
from backtesting.strategy import Strategy

_INDICATOR_WARMUP = 60   # bars needed for SMA-50 to warm up


class RLTradingStrategy(Strategy):
    """
    Parameters
    ----------
    model             : ActorCriticLSTM  Trained model (on the correct device).
    device            : str
    symbol            : str              Futures symbol, e.g. ``"NQ"``.
    seq_len           : int
    contracts         : float            Position size.
    eod_hour_utc      : int              UTC hour to flatten everything.
    no_entry_hour_utc : int              UTC hour to block new entries.
    reward_scale      : float            Must match PPOTrainer.reward_scale.
    """

    def __init__(
        self,
        model,
        device: str,
        symbol: str,
        seq_len: int           = 30,
        contracts: float       = 1.0,
        eod_hour_utc: int      = 20,
        no_entry_hour_utc: int = 19,
        reward_scale: float    = 100.0,
    ) -> None:
        super().__init__()
        self._model         = model
        self._device        = device
        self._symbol        = symbol
        self._seq_len       = seq_len
        self._contracts     = contracts
        self._eod_hour      = eod_hour_utc
        self._no_entry_hour = no_entry_hour_utc
        self._reward_scale  = reward_scale
        self._min_bars      = seq_len + _INDICATOR_WARMUP

        # Bracket order state
        self._in_bracket:     bool        = False
        self._sl_order_id:    str | None  = None
        self._tp_order_id:    str | None  = None
        self._pending_sl_pts: float       = 0.0
        self._pending_tp_pts: float       = 0.0

    def on_start(self) -> None:
        self.name = f"RL-PPO-Bracket(seq={self._seq_len},intraday)"

    # ------------------------------------------------------------------
    # Fill callback — OCO bracket management
    # ------------------------------------------------------------------

    def on_fill(self, order: Order) -> None:
        pos = self.position(self._symbol)

        # ── Entry fill: position just opened → place SL + TP bracket ──
        if abs(pos) > 1e-9 and not self._in_bracket and order.fill_price is not None:
            entry     = order.fill_price
            direction = 1.0 if pos > 0 else -1.0
            sl_price  = entry - direction * self._pending_sl_pts
            tp_price  = entry + direction * self._pending_tp_pts

            if pos > 0:
                sl_order = self.sell(
                    self._symbol, self._contracts,
                    order_type=OrderType.STOP, stop_price=sl_price,
                    reduce_only=True, tag="bracket_sl",
                )
                tp_order = self.sell(
                    self._symbol, self._contracts,
                    order_type=OrderType.LIMIT, limit_price=tp_price,
                    reduce_only=True, tag="bracket_tp",
                )
            else:
                sl_order = self.buy(
                    self._symbol, self._contracts,
                    order_type=OrderType.STOP, stop_price=sl_price,
                    reduce_only=True, tag="bracket_sl",
                )
                tp_order = self.buy(
                    self._symbol, self._contracts,
                    order_type=OrderType.LIMIT, limit_price=tp_price,
                    reduce_only=True, tag="bracket_tp",
                )

            self._sl_order_id = sl_order.order_id
            self._tp_order_id = tp_order.order_id
            self._in_bracket  = True

        # ── Bracket exit: SL or TP fired → cancel the surviving leg ───
        elif self._in_bracket and abs(pos) < 1e-9:
            if self._sl_order_id:
                self.cancel_order(self._sl_order_id)
            if self._tp_order_id:
                self.cancel_order(self._tp_order_id)
            self._sl_order_id = None
            self._tp_order_id = None
            self._in_bracket  = False

    # ------------------------------------------------------------------
    # Main bar callback
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        import torch
        from backtesting.ml.features import make_features

        if self.bars_available < self._min_bars:
            return

        pos      = self.position(self._symbol)
        bar_hour = bar.timestamp.hour   # Databento timestamps are UTC

        # ── 1. EOD forced flat ────────────────────────────────────────
        if bar_hour >= self._eod_hour:
            if self._in_bracket or pos != 0:
                self.cancel_all(self._symbol)
                if pos != 0:
                    self.close_position(self._symbol, tag="eod_close")
                self._in_bracket  = False
                self._sl_order_id = None
                self._tp_order_id = None
            return

        # ── 2. In bracket — model is ignored, orders manage the trade ─
        if self._in_bracket:
            return

        # ── 3. No new entries near EOD ────────────────────────────────
        if bar_hour >= self._no_entry_hour:
            return

        # ── 4. Build augmented state (matches TradingEnv training) ────
        # pos should be 0 here (not in bracket, guaranteed by step 2)
        hist     = self.history(self._min_bars)
        features = make_features(hist)[-self._seq_len:]      # (seq_len, 22)

        pos_enc  = np.zeros((self._seq_len, 1), dtype=np.float32)   # always flat here
        upnl_arr = np.zeros((self._seq_len, 1), dtype=np.float32)

        x_aug = np.concatenate([features, pos_enc, upnl_arr], axis=-1)  # (seq_len, 24)
        x_t   = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0).to(self._device)

        # ── 5. Policy inference — network decides direction + bracket ─
        desired_pos, _confidence, sl_pts, tp_pts = self._model.predict_rl(x_t)

        if desired_pos == 0:
            return  # flat signal, stay flat

        # ── 6. Submit market entry (bracket placed in on_fill) ────────
        self._pending_sl_pts = sl_pts
        self._pending_tp_pts = tp_pts

        if desired_pos == 1:
            self.buy(self._symbol, self._contracts, tag="rl_long")
        else:
            self.sell(self._symbol, self._contracts, tag="rl_short")
