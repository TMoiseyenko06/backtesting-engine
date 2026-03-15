"""
Intraday futures trading environment for PPO.

One episode = all bars for a single UTC calendar date.  The env steps through
bars one at a time and enforces:
  - Intraday-only (EOD forced-flat at episode end)
  - Per-trade stop-loss (force flat when unrealised P&L < -max_loss)

State
-----
  (seq_len, n_features + 2)  float32
  The last two channels appended to each timestep are context features:
    position_enc  : float  {-1.0 short, 0.0 flat, +1.0 long}
    norm_upnl     : unrealised P&L / reward_scale, clamped to [-10, +10]

Actions
-------
  0  FLAT   — close any open position
  1  LONG   — open / maintain long
  2  SHORT  — open / maintain short

Reward
------
  (mark-to-market P&L change − commission paid this step) / reward_scale

  Commission is charged both sides of every trade (open + close).
  A stop-loss forced close adds one more commission penalty.
  The EOD forced close adds one more commission penalty.
"""

from __future__ import annotations

import numpy as np

FLAT  = 0
LONG  = 1
SHORT = 2

# Maps action int → signed position {-1, 0, +1}
_ACTION_TO_POS = {FLAT: 0, LONG: 1, SHORT: -1}


class TradingEnv:
    """
    Parameters
    ----------
    seq_len       : int    History window fed to the policy network.
    n_features    : int    Raw OHLCV feature count (before augmentation).
    multiplier    : float  Contract point value, e.g. $20 for NQ.
    commission    : float  $ per contract per side.
    contracts     : float  Position size.
    max_loss      : float  Stop-loss threshold in dollars.
    reward_scale  : float  Divide all rewards by this factor (keeps rewards ~O(1)).
    """

    N_EXTRA = 2  # position_enc + norm_upnl

    def __init__(
        self,
        seq_len:      int   = 30,
        n_features:   int   = 22,
        multiplier:   float = 20.0,
        commission:   float = 2.0,
        contracts:    float = 1.0,
        max_loss:     float = 2_500.0,
        reward_scale: float = 100.0,
    ) -> None:
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.state_dim    = n_features + self.N_EXTRA
        self.multiplier   = multiplier
        self.commission   = commission
        self.contracts    = contracts
        self.max_loss     = max_loss
        self.reward_scale = reward_scale

        # Episode state (initialised by reset)
        self._features:    np.ndarray | None = None
        self._prices:      np.ndarray | None = None
        self._n:           int   = 0
        self._cursor:      int   = 0
        self._position:    int   = 0      # −1, 0, +1
        self._entry_price: float = 0.0
        self._prev_close:  float = 0.0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, features: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Begin a new episode.

        features : (n_bars, n_features) — pre-computed for the day
        prices   : (n_bars,)            — bar close prices
        Returns  : initial state (seq_len, state_dim)
        """
        if len(features) != len(prices):
            raise ValueError("features and prices must have the same length")
        if len(features) <= self.seq_len:
            raise ValueError(
                f"episode too short ({len(features)} bars ≤ seq_len={self.seq_len})"
            )
        self._features    = features.astype(np.float32)
        self._prices      = prices.astype(np.float32)
        self._n           = len(features)
        self._cursor      = self.seq_len   # first actionable bar index
        self._position    = 0
        self._entry_price = 0.0
        self._prev_close  = float(prices[self.seq_len - 1])
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute action, advance to the next bar.

        Returns
        -------
        next_state  : (seq_len, state_dim)  — zeros tensor when done=True
        reward      : float
        done        : bool
        """
        if self._features is None:
            raise RuntimeError("call reset() before step()")

        curr_close      = float(self._prices[self._cursor])
        commission_paid = 0.0
        desired         = _ACTION_TO_POS[action]

        # ── Trade execution ───────────────────────────────────────────────
        if desired != self._position:
            if self._position != 0:                      # close existing
                commission_paid += self.commission * self.contracts
            if desired != 0:                             # open new
                commission_paid += self.commission * self.contracts
                self._entry_price = curr_close
            else:
                self._entry_price = 0.0
            self._position = desired

        # ── Mark-to-market reward ─────────────────────────────────────────
        mtm    = (curr_close - self._prev_close) * self._position * self.contracts * self.multiplier
        reward = (mtm - commission_paid) / self.reward_scale

        # ── Per-trade stop-loss ───────────────────────────────────────────
        if self._position != 0 and self._entry_price > 0.0:
            upnl = (
                (curr_close - self._entry_price)
                * self._position * self.contracts * self.multiplier
            )
            if upnl < -self.max_loss:
                reward        -= (self.commission * self.contracts) / self.reward_scale
                self._position = 0
                self._entry_price = 0.0

        self._prev_close = curr_close
        self._cursor    += 1

        # ── End of episode ────────────────────────────────────────────────
        if self._cursor >= self._n:
            if self._position != 0:
                reward        -= (self.commission * self.contracts) / self.reward_scale
                self._position = 0
                self._entry_price = 0.0
            return np.zeros((self.seq_len, self.state_dim), dtype=np.float32), reward, True

        return self._state(), reward, False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state(self) -> np.ndarray:
        """Build (seq_len, state_dim) state from current cursor position."""
        i   = self._cursor
        seq = self._features[i - self.seq_len: i]          # (seq_len, n_features)

        pos_enc = np.full((self.seq_len, 1), float(self._position), dtype=np.float32)

        if self._position != 0 and self._entry_price > 0.0:
            upnl = (
                (float(self._prices[i - 1]) - self._entry_price)
                * self._position * self.contracts * self.multiplier
            ) / self.reward_scale
            upnl = float(np.clip(upnl, -10.0, 10.0))
        else:
            upnl = 0.0
        upnl_arr = np.full((self.seq_len, 1), upnl, dtype=np.float32)

        return np.concatenate([seq, pos_enc, upnl_arr], axis=-1)  # (seq_len, state_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (self.seq_len, self.state_dim)

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def position(self) -> int:
        """Current signed position: -1, 0, or +1."""
        return self._position

    @property
    def unrealised_pnl(self) -> float:
        """Current unrealised P&L in dollars."""
        if self._position == 0 or self._entry_price == 0.0:
            return 0.0
        last_price = float(self._prices[self._cursor - 1]) if self._cursor > 0 else self._entry_price
        return (
            (last_price - self._entry_price)
            * self._position * self.contracts * self.multiplier
        )
