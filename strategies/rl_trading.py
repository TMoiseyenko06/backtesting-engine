"""
RL Trading Strategy
--------------------
Wraps a trained ActorCriticLSTM for backtesting.

The strategy augments the standard OHLCV feature window with two
position-context channels (matching what TradingEnv presents during training),
then calls model.predict_rl() to obtain the desired position and acts on it.

Risk controls (identical to LSTMSignalStrategy)
-----------------------------------------------
  - Intraday only   : forced flat at/after ``eod_hour_utc`` (default 20 ≈ 4 PM EDT).
                      No new entries after ``no_entry_hour_utc`` (default 19).
  - Stop-loss       : if unrealised P&L < -``max_loss_per_trade`` → force flat.
  - Min hold        : signal-driven exits/flips blocked for the first
                      ``min_hold_bars`` bars after entry.  Stop-loss and EOD
                      always override this.
"""

from __future__ import annotations

import numpy as np

from backtesting.data_feed import Bar
from backtesting.order import Order
from backtesting.strategy import Strategy

# Desired-position constants
_LONG  =  1
_FLAT  =  0
_SHORT = -1


class RLTradingStrategy(Strategy):
    """
    Parameters
    ----------
    model              : ActorCriticLSTM  Trained model (on the correct device).
    device             : str
    symbol             : str              Futures symbol, e.g. ``"NQ"``.
    seq_len            : int
    confidence_threshold : float          Min softmax prob to act (else hold).
    contracts          : float
    max_loss_per_trade : float            Dollar stop-loss per trade.
    eod_hour_utc       : int              UTC hour to flatten everything.
    no_entry_hour_utc  : int              UTC hour to block new entries.
    min_hold_bars      : int              Min bars before signal-driven exit/flip.
    reward_scale       : float            Must match PPOTrainer.reward_scale
                                          (used to normalise the upnl context feature).
    """

    def __init__(
        self,
        model,
        device: str,
        symbol: str,
        seq_len: int                = 30,
        confidence_threshold: float = 0.0,   # 0 = always act on policy
        contracts: float            = 1.0,
        max_loss_per_trade: float   = 2_500.0,
        eod_hour_utc: int           = 20,
        no_entry_hour_utc: int      = 19,
        min_hold_bars: int          = 2,
        reward_scale: float         = 100.0,
    ) -> None:
        super().__init__()
        self._model       = model
        self._device      = device
        self._symbol      = symbol
        self._seq_len     = seq_len
        self._conf_thresh = confidence_threshold
        self._contracts   = contracts
        self._max_loss    = max_loss_per_trade
        self._eod_hour    = eod_hour_utc
        self._no_entry_hour = no_entry_hour_utc
        self._min_hold    = min_hold_bars
        self._reward_scale = reward_scale

        # Need indicator warmup before LSTM window (SMA-50 needs 60 bars)
        _INDICATOR_WARMUP = 60
        self._min_bars = seq_len + _INDICATOR_WARMUP

        # Position tracking (updated in on_fill)
        self._entry_price: float = 0.0
        self._entry_bar:   int   = 0

    def on_start(self) -> None:
        self.name = (
            f"RL-PPO(seq={self._seq_len},"
            f"stop=${self._max_loss:,.0f},"
            f"hold≥{self._min_hold}bar,intraday)"
        )

    # ------------------------------------------------------------------
    # Fill callback — track actual entry price
    # ------------------------------------------------------------------

    def on_fill(self, order: Order) -> None:
        pos_after = self.position(self._symbol)
        if abs(pos_after) > 1e-9 and order.fill_price is not None:
            self._entry_price = order.fill_price
            self._entry_bar   = self.bars_available

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
            if pos != 0:
                self.close_position(self._symbol, tag="eod_close")
            return

        # ── 2. Per-trade stop-loss ────────────────────────────────────
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

        # ── 3. No new entries near EOD ────────────────────────────────
        if bar_hour >= self._no_entry_hour:
            return

        # ── 4. Minimum hold guard ─────────────────────────────────────
        bars_held            = self.bars_available - self._entry_bar
        signal_exits_blocked = (pos != 0) and (bars_held < self._min_hold)

        # ── 5. Build augmented state ──────────────────────────────────
        from backtesting.ml.rl_env import TradingEnv
        hist     = self.history(self._min_bars)
        features = make_features(hist)                      # (min_bars, 22)
        x_feat   = features[-self._seq_len:]                # (seq_len, 22)

        # Signed position encoding for this strategy's position
        pos_enc_val = 0.0 if pos == 0 else (1.0 if pos > 0 else -1.0)
        pos_enc  = np.full((self._seq_len, 1), pos_enc_val, dtype=np.float32)

        # Normalised unrealised P&L (same scaling as TradingEnv)
        if pos != 0 and self._entry_price > 0:
            upnl_raw = (
                (bar.close - self._entry_price)
                * pos_enc_val
                * self._contracts
                * bar.contract_multiplier
            ) / self._reward_scale
            upnl_val = float(np.clip(upnl_raw, -10.0, 10.0))
        else:
            upnl_val = 0.0
        upnl_arr = np.full((self._seq_len, 1), upnl_val, dtype=np.float32)

        x_aug = np.concatenate([x_feat, pos_enc, upnl_arr], axis=-1)  # (seq_len, 24)
        x_t   = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0).to(self._device)

        # ── 6. Policy inference ───────────────────────────────────────
        desired_pos, confidence = self._model.predict_rl(x_t)
        # desired_pos : -1 (short), 0 (flat), +1 (long)

        if confidence < self._conf_thresh:
            return

        # ── 7. Execute desired position ───────────────────────────────
        current_sign = 0 if pos == 0 else (1 if pos > 0 else -1)

        if desired_pos == current_sign:
            return  # already in the desired state

        if signal_exits_blocked:
            return  # hold minimum duration before any exit/flip

        if desired_pos == _FLAT:
            self.close_position(self._symbol, tag="rl_flat")

        elif desired_pos == _LONG:
            if pos < 0:
                self.close_position(self._symbol, tag="flip_long")
            self.buy(self._symbol, self._contracts, tag="rl_long")

        elif desired_pos == _SHORT:
            if pos > 0:
                self.close_position(self._symbol, tag="flip_short")
            self.sell(self._symbol, self._contracts, tag="rl_short")
