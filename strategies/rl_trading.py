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
  - Bankruptcy    : trading halts if equity falls below ``bankruptcy_floor``
                    (default 10% of initial cash) to prevent unbounded losses.
"""

from __future__ import annotations

from collections import deque

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
    bankruptcy_floor  : float            Stop trading if equity drops below this
                                         fraction of initial cash (default 0.10).
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
        max_loss: float        = 2_500.0,
        multiplier: float      = 20.0,
        bankruptcy_floor: float = 0.10,
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
        self._max_loss      = max_loss
        self._multiplier    = multiplier
        self._bankruptcy_floor = bankruptcy_floor
        self._min_bars      = seq_len + _INDICATOR_WARMUP

        # Bracket order state
        self._in_bracket:     bool        = False
        self._sl_order_id:    str | None  = None
        self._tp_order_id:    str | None  = None
        self._pending_sl_pts: float       = 0.0
        self._pending_tp_pts: float       = 0.0

        # Rolling history for train-consistent pos/upnl state encoding
        self._pos_hist:  deque[float] = deque(maxlen=seq_len)
        self._upnl_hist: deque[float] = deque(maxlen=seq_len)

        # Action distribution counters (for on_stop summary)
        self._act_flat:  int = 0
        self._act_long:  int = 0
        self._act_short: int = 0

        # Per-trade log: (entry_time, entry_price, sl_price, tp_price, direction)
        self._pending_trade_info: tuple | None = None

        # Bankruptcy guard: set True once equity falls below floor
        self._bankrupt: bool = False
        self._initial_cash: float | None = None

    def on_start(self) -> None:
        self.name = f"RL-PPO-Bracket(seq={self._seq_len},intraday)"
        self._initial_cash = self.cash()

    # ------------------------------------------------------------------
    # Fill callback — OCO bracket management
    # ------------------------------------------------------------------

    def on_fill(self, order: Order) -> None:
        pos = self.position(self._symbol)

        # ── Entry fill: position just opened → place SL + TP bracket ──
        if abs(pos) > 1e-9 and not self._in_bracket and order.fill_price is not None:
            entry     = order.fill_price
            direction = 1.0 if pos > 0 else -1.0

            # Hard-cap SL distance so max loss per trade ≤ max_loss dollars
            max_sl_pts = self._max_loss / (self._contracts * self._multiplier)
            sl_pts     = min(self._pending_sl_pts, max_sl_pts)

            sl_price  = entry - direction * sl_pts
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

            dir_str = "LONG" if pos > 0 else "SHORT"
            print(
                f"  [TRADE OPEN ] {order.filled_at}  {dir_str}  "
                f"entry={entry:.2f}  SL={sl_price:.2f}({sl_pts:.1f}pts)  "
                f"TP={tp_price:.2f}({self._pending_tp_pts:.1f}pts)"
            )

        # ── Bracket exit: SL or TP fired → cancel the surviving leg ───
        elif self._in_bracket and abs(pos) < 1e-9:
            if self._sl_order_id:
                self.cancel_order(self._sl_order_id)
            if self._tp_order_id:
                self.cancel_order(self._tp_order_id)
            self._sl_order_id = None
            self._tp_order_id = None
            self._in_bracket  = False

            tag = order.tag if order.tag else "?"
            print(
                f"  [TRADE CLOSE] {order.filled_at}  exit={order.fill_price:.2f}  "
                f"tag={tag}  equity={self.equity():.0f}"
            )

    # ------------------------------------------------------------------
    # Main bar callback
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        import torch
        from backtesting.ml.features import make_features

        if self.bars_available < self._min_bars:
            return

        pos_obj  = self.position_obj(self._symbol)
        pos      = pos_obj.quantity
        bar_hour = bar.timestamp.hour   # Databento timestamps are UTC

        # ── Bankruptcy guard ───────────────────────────────────────────
        if self._initial_cash is not None and not self._bankrupt:
            current_equity = self.equity()
            floor = self._initial_cash * self._bankruptcy_floor
            if current_equity < floor:
                self._bankrupt = True
                if self._in_bracket or pos != 0:
                    self.cancel_all(self._symbol)
                    if pos != 0:
                        self.close_position(self._symbol, tag="bankruptcy_stop")
                    self._in_bracket  = False
                    self._sl_order_id = None
                    self._tp_order_id = None
                print(
                    f"  [BANKRUPTCY ] {bar.timestamp}  equity={current_equity:.0f} "
                    f"< floor={floor:.0f} — trading halted"
                )
                return

        if self._bankrupt:
            return

        # ── Update rolling pos/upnl history on every bar (mirrors training) ──
        direction = 1.0 if pos > 0 else (-1.0 if pos < 0 else 0.0)
        if abs(pos) > 1e-9 and pos_obj.avg_entry_price > 0.0:
            upnl_raw = ((bar.close - pos_obj.avg_entry_price)
                        * pos * self._multiplier / self._reward_scale)
            upnl_scaled = float(np.clip(upnl_raw, -10.0, 10.0))
        else:
            upnl_scaled = 0.0
        self._pos_hist.append(direction)
        self._upnl_hist.append(upnl_scaled)

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
        #    Guard: a margin call can force-close the position without
        #    triggering on_fill, leaving _in_bracket stale.  Detect this
        #    and reset so the strategy can enter new trades.
        if self._in_bracket and abs(pos) < 1e-9:
            self.cancel_all(self._symbol)
            self._in_bracket  = False
            self._sl_order_id = None
            self._tp_order_id = None

        #    Per-bar hard stop: if bracket SL somehow fails to execute,
        #    this guarantees we never lose more than max_loss dollars.
        if self._in_bracket:
            if abs(pos) > 1e-9 and pos_obj.avg_entry_price > 0.0:
                upnl = (bar.close - pos_obj.avg_entry_price) * pos * self._multiplier
                if upnl < -self._max_loss:
                    self.cancel_all(self._symbol)
                    self.close_position(self._symbol, tag="hard_stop")
                    self._in_bracket  = False
                    self._sl_order_id = None
                    self._tp_order_id = None
                    print(
                        f"  [HARD STOP  ] {bar.timestamp}  upnl={upnl:.0f}  "
                        f"entry={pos_obj.avg_entry_price:.2f}  close={bar.close:.2f}"
                    )
            return

        # ── 3. No new entries near EOD ────────────────────────────────
        if bar_hour >= self._no_entry_hour:
            return

        # ── 4. Build augmented state (matches TradingEnv training) ────
        hist     = self.history(self._min_bars)
        features = make_features(hist)[-self._seq_len:]      # (seq_len, 22)

        # Use rolling history so the model sees the same pos/upnl distribution
        # it trained on (non-zero during bracket holds, 0 when flat).
        pad = self._seq_len - len(self._pos_hist)
        pos_enc  = np.array(
            [0.0] * pad + list(self._pos_hist), dtype=np.float32
        ).reshape(-1, 1)
        upnl_arr = np.array(
            [0.0] * pad + list(self._upnl_hist), dtype=np.float32
        ).reshape(-1, 1)

        x_aug = np.concatenate([features, pos_enc, upnl_arr], axis=-1)  # (seq_len, 24)
        x_t   = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0).to(self._device)

        # ── 5. Policy inference — network decides direction + bracket ─
        desired_pos, _confidence, sl_pts, tp_pts = self._model.predict_rl(x_t)

        if desired_pos == 0:
            self._act_flat += 1
            return  # flat signal, stay flat

        if desired_pos == 1:
            self._act_long += 1
        else:
            self._act_short += 1

        # ── 6. Submit market entry (bracket placed in on_fill) ────────
        _max_sl = self._max_loss / (self._contracts * self._multiplier)
        self._pending_sl_pts = min(sl_pts, _max_sl)
        self._pending_tp_pts = tp_pts

        if desired_pos == 1:
            self.buy(self._symbol, self._contracts, tag="rl_long")
        else:
            self.sell(self._symbol, self._contracts, tag="rl_short")

    # ------------------------------------------------------------------
    # End-of-backtest summary
    # ------------------------------------------------------------------

    def on_stop(self) -> None:
        total = self._act_flat + self._act_long + self._act_short
        if total == 0:
            print("  [RL Strategy] No bars processed after warm-up.")
            return
        print(
            f"\n  ── RL Strategy Action Distribution ──────────────────────\n"
            f"  FLAT : {self._act_flat:>7,}  ({100*self._act_flat/total:.1f}%)\n"
            f"  LONG : {self._act_long:>7,}  ({100*self._act_long/total:.1f}%)\n"
            f"  SHORT: {self._act_short:>7,}  ({100*self._act_short/total:.1f}%)\n"
            f"  Total decision bars: {total:,}\n"
            f"  ─────────────────────────────────────────────────────────"
        )
        if self._act_long + self._act_short == 0:
            print(
                "  WARNING: model predicted FLAT on every bar — policy may have collapsed.\n"
                "  Try: --entropy-coef 0.1 --flat-penalty 2.0"
            )
