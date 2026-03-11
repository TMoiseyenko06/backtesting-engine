"""
Trend-Following Strategy with ATR-based position sizing
---------------------------------------------------------
Combines EMA trend direction with ATR for dynamic stop placement
and position sizing (risk a fixed % of equity per trade).

Rules
-----
Entry  : price crosses EMA from below (long) or above (short)
Stop   : 2× ATR from entry price
Target : optional take-profit at risk_reward_ratio × stop distance
Size   : risk_pct of equity / (ATR * multiplier) → contracts to trade
"""

from __future__ import annotations

from typing import Optional

from backtesting.data_feed import Bar
from backtesting.order import Order, OrderType
from backtesting.strategy import Strategy


class TrendFollowingStrategy(Strategy):
    """
    Parameters
    ----------
    ema_period : int
    atr_period : int
    atr_stop_mult : float
        ATR multiplier for initial stop loss.
    risk_pct : float
        Fraction of equity to risk per trade (e.g. 0.01 = 1%).
    max_contracts : float
        Hard cap on position size.
    risk_reward_ratio : float
        Take-profit as a multiple of the stop distance (0 = no TP).
    """

    def __init__(
        self,
        ema_period: int = 50,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        risk_pct: float = 0.01,
        max_contracts: float = 10.0,
        risk_reward_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.risk_pct = risk_pct
        self.max_contracts = max_contracts
        self.risk_reward_ratio = risk_reward_ratio
        self._warmup = ema_period + atr_period + 5
        self._prev_above_ema: Optional[bool] = None
        self._stop_id: Optional[str] = None
        self._tp_id: Optional[str] = None

    def on_start(self) -> None:
        self.name = f"TrendFollow(ema={self.ema_period},atr={self.atr_period})"

    def on_bar(self, bar: Bar) -> None:
        if self.bars_available < self._warmup:
            return

        n = max(self.ema_period * 3, self.atr_period + 10)
        hist = self.history(n)
        closes = [b.close for b in hist]

        ema_val = self.ema(closes, self.ema_period)
        atr_val = self.atr(hist, self.atr_period)

        symbol = bar.symbol
        pos = self.position(symbol)
        above_ema = bar.close > ema_val

        # --- Exit logic ---
        # (stops and TPs are handled as resting orders; here we add EMA exit)
        if pos > 0 and not above_ema:
            self._cancel_attached_orders()
            self.close_position(symbol, tag="ema_exit_long")
            self._prev_above_ema = above_ema
            return

        if pos < 0 and above_ema:
            self._cancel_attached_orders()
            self.close_position(symbol, tag="ema_exit_short")
            self._prev_above_ema = above_ema
            return

        # --- Entry logic ---
        if pos == 0 and self._prev_above_ema is not None:
            equity = self.equity()
            risk_dollars = equity * self.risk_pct
            multiplier = self.position_obj(symbol).contract_multiplier or 1.0

            stop_dist = self.atr_stop_mult * atr_val  # price distance for stop
            if stop_dist <= 0:
                self._prev_above_ema = above_ema
                return

            # Size = risk_dollars / (stop_distance_per_contract)
            size = risk_dollars / (stop_dist * multiplier)
            size = max(1.0, min(self.max_contracts, round(size)))

            # Long entry: price just crossed above EMA
            if not self._prev_above_ema and above_ema:
                self.buy(symbol, size, tag="tf_long")
                stop_px = bar.close - stop_dist
                o_stop = self.sell(
                    symbol, size,
                    order_type=OrderType.STOP,
                    stop_price=round(stop_px, 2),
                    reduce_only=True,
                    tag="sl_long",
                )
                self._stop_id = o_stop.order_id

                if self.risk_reward_ratio > 0:
                    tp_px = bar.close + stop_dist * self.risk_reward_ratio
                    o_tp = self.sell(
                        symbol, size,
                        order_type=OrderType.LIMIT,
                        limit_price=round(tp_px, 2),
                        reduce_only=True,
                        tag="tp_long",
                    )
                    self._tp_id = o_tp.order_id

            # Short entry: price just crossed below EMA
            elif self._prev_above_ema and not above_ema:
                self.sell(symbol, size, tag="tf_short")
                stop_px = bar.close + stop_dist
                o_stop = self.buy(
                    symbol, size,
                    order_type=OrderType.STOP,
                    stop_price=round(stop_px, 2),
                    reduce_only=True,
                    tag="sl_short",
                )
                self._stop_id = o_stop.order_id

                if self.risk_reward_ratio > 0:
                    tp_px = bar.close - stop_dist * self.risk_reward_ratio
                    o_tp = self.buy(
                        symbol, size,
                        order_type=OrderType.LIMIT,
                        limit_price=round(tp_px, 2),
                        reduce_only=True,
                        tag="tp_short",
                    )
                    self._tp_id = o_tp.order_id

        self._prev_above_ema = above_ema

    def on_fill(self, order: Order) -> None:
        # When stop fires, cancel the TP and vice versa
        if order.tag in ("sl_long", "sl_short") and self._tp_id:
            self.cancel_order(self._tp_id)
            self._tp_id = None
        elif order.tag in ("tp_long", "tp_short") and self._stop_id:
            self.cancel_order(self._stop_id)
            self._stop_id = None

    def _cancel_attached_orders(self) -> None:
        if self._stop_id:
            self.cancel_order(self._stop_id)
            self._stop_id = None
        if self._tp_id:
            self.cancel_order(self._tp_id)
            self._tp_id = None
