"""
RL Trading Strategy
--------------------
Wraps a trained ActorCriticLSTM for backtesting.

The strategy feeds the standard OHLCV feature window augmented with two
position-context channels (matching TradingEnv's training state), then calls
model.predict_rl() to get the desired position and trades accordingly.

The network decides EVERYTHING about entry, exit, and hold duration by itself.
No minimum hold time, no magnitude filter, no confidence threshold — the agent
learned all of that during PPO training.  Excessive commission costs in the
environment already discouraged churning; the auxiliary prediction loss taught
the LSTM to identify multi-bar moves.

The ONLY hard rules enforced here are non-negotiable risk controls:
  - Intraday only : forced flat at/after ``eod_hour_utc`` (default 20:00 UTC ≈ 4 PM EDT).
                    No new entries after ``no_entry_hour_utc`` (default 19:00 UTC).
  - Stop-loss     : if unrealised P&L < -``max_loss_per_trade`` → force flat immediately.
"""

from __future__ import annotations

import numpy as np

from backtesting.data_feed import Bar
from backtesting.order import Order
from backtesting.strategy import Strategy

_INDICATOR_WARMUP = 60   # bars needed for SMA-50 to warm up


class RLTradingStrategy(Strategy):
    """
    Parameters
    ----------
    model              : ActorCriticLSTM  Trained model (on the correct device).
    device             : str
    symbol             : str              Futures symbol, e.g. ``"NQ"``.
    seq_len            : int
    contracts          : float            Position size (max 1 per user requirement).
    max_loss_per_trade : float            Dollar stop-loss.
    eod_hour_utc       : int              UTC hour to flatten everything.
    no_entry_hour_utc  : int              UTC hour to block new entries.
    reward_scale       : float            Must match PPOTrainer.reward_scale.
    """

    def __init__(
        self,
        model,
        device: str,
        symbol: str,
        seq_len: int              = 30,
        contracts: float          = 1.0,
        max_loss_per_trade: float = 2_500.0,
        eod_hour_utc: int         = 20,
        no_entry_hour_utc: int    = 19,
        reward_scale: float       = 100.0,
    ) -> None:
        super().__init__()
        self._model         = model
        self._device        = device
        self._symbol        = symbol
        self._seq_len       = seq_len
        self._contracts     = contracts
        self._max_loss      = max_loss_per_trade
        self._eod_hour      = eod_hour_utc
        self._no_entry_hour = no_entry_hour_utc
        self._reward_scale  = reward_scale
        self._min_bars      = seq_len + _INDICATOR_WARMUP

        # Track entry price for stop-loss calculation
        self._entry_price: float = 0.0

    def on_start(self) -> None:
        self.name = (
            f"RL-PPO(seq={self._seq_len},"
            f"stop=${self._max_loss:,.0f},intraday)"
        )

    # ------------------------------------------------------------------
    # Fill callback — track actual entry price for stop-loss
    # ------------------------------------------------------------------

    def on_fill(self, order: Order) -> None:
        pos_after = self.position(self._symbol)
        if abs(pos_after) > 1e-9 and order.fill_price is not None:
            self._entry_price = order.fill_price

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

        # ── 1. EOD forced flat (hard rule) ───────────────────────────
        if bar_hour >= self._eod_hour:
            if pos != 0:
                self.close_position(self._symbol, tag="eod_close")
            return

        # ── 2. Per-trade stop-loss (hard rule) ───────────────────────
        if pos != 0 and self._entry_price > 0:
            direction  = 1.0 if pos > 0 else -1.0
            unrealised = (
                direction
                * (bar.close - self._entry_price)
                * abs(pos)
                * bar.contract_multiplier
            )
            if unrealised < -self._max_loss:
                self.close_position(self._symbol, tag="stop_loss")
                return

        # ── 3. No new entries near EOD ───────────────────────────────
        if bar_hour >= self._no_entry_hour:
            return  # existing position rides until EOD close above

        # ── 4. Build augmented state (matches TradingEnv training) ───
        hist     = self.history(self._min_bars)
        features = make_features(hist)[-self._seq_len:]      # (seq_len, 22)

        pos_enc_val = 0.0 if pos == 0 else (1.0 if pos > 0 else -1.0)
        pos_enc     = np.full((self._seq_len, 1), pos_enc_val, dtype=np.float32)

        if pos != 0 and self._entry_price > 0:
            upnl_val = float(np.clip(
                (bar.close - self._entry_price)
                * pos_enc_val * self._contracts * bar.contract_multiplier
                / self._reward_scale,
                -10.0, 10.0,
            ))
        else:
            upnl_val = 0.0
        upnl_arr = np.full((self._seq_len, 1), upnl_val, dtype=np.float32)

        x_aug = np.concatenate([features, pos_enc, upnl_arr], axis=-1)  # (seq_len, 24)
        x_t   = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0).to(self._device)

        # ── 5. Policy inference (network decides everything) ─────────
        # desired_pos : -1 (short), 0 (flat), +1 (long)
        # pred_return : model's H-bar ahead price prediction (informational)
        desired_pos, _confidence, _pred_return = self._model.predict_rl(x_t)

        # ── 6. Execute desired position ───────────────────────────────
        current_sign = 0 if pos == 0 else (1 if pos > 0 else -1)
        if desired_pos == current_sign:
            return  # already in the right state

        if desired_pos == 0:
            self.close_position(self._symbol, tag="rl_flat")

        elif desired_pos == 1:
            if pos < 0:
                self.close_position(self._symbol, tag="flip_long")
            self.buy(self._symbol, self._contracts, tag="rl_long")

        elif desired_pos == -1:
            if pos > 0:
                self.close_position(self._symbol, tag="flip_short")
            self.sell(self._symbol, self._contracts, tag="rl_short")
