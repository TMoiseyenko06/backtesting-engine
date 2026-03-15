"""
Intraday futures trading environment for PPO.

One episode = all bars for a single UTC calendar date.  The env steps through
bars one at a time and enforces:
  - Intraday-only (EOD forced-flat at episode end)
  - Trailing drawdown stop (force flat when unrealised P&L drops $max_loss from its peak)

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
        self._features:       np.ndarray | None = None
        self._prices:         np.ndarray | None = None
        self._n:              int   = 0
        self._cursor:         int   = 0
        self._position:       int   = 0      # −1, 0, +1
        self._entry_price:    float = 0.0
        self._peak_unrealised: float = 0.0
        self._prev_close:     float = 0.0

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
        self._features        = features.astype(np.float32)
        self._prices          = prices.astype(np.float32)
        self._n               = len(features)
        self._cursor          = self.seq_len   # first actionable bar index
        self._position        = 0
        self._entry_price     = 0.0
        self._peak_unrealised = 0.0
        self._prev_close      = float(prices[self.seq_len - 1])
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
                self._entry_price     = curr_close
                self._peak_unrealised = 0.0
            else:
                self._entry_price     = 0.0
                self._peak_unrealised = 0.0
            self._position = desired

        # ── Mark-to-market reward ─────────────────────────────────────────
        mtm    = (curr_close - self._prev_close) * self._position * self.contracts * self.multiplier
        reward = (mtm - commission_paid) / self.reward_scale

        # ── Trailing drawdown stop ────────────────────────────────────────
        if self._position != 0 and self._entry_price > 0.0:
            upnl = (
                (curr_close - self._entry_price)
                * self._position * self.contracts * self.multiplier
            )
            if upnl > self._peak_unrealised:
                self._peak_unrealised = upnl
            if self._peak_unrealised - upnl >= self.max_loss:
                reward        -= (self.commission * self.contracts) / self.reward_scale
                self._position        = 0
                self._entry_price     = 0.0
                self._peak_unrealised = 0.0

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


class BatchedTradingEnv:
    """
    Vectorised equivalent of running n_envs TradingEnv instances in parallel.

    Instead of calling env.step() in a Python loop (512 calls × 400 bars =
    200K Python dispatches per rollout iteration), all environments are
    stepped simultaneously using numpy array operations.  This removes the
    CPU-side bottleneck that keeps the GPU idle between forward passes.

    API mirrors TradingEnv but operates on all n_envs at once:
        states = env.reset_all(episodes)          # (n, seq_len, state_dim)
        states, rewards, dones = env.step_all(actions)  # all numpy, no loops
    """

    def __init__(
        self,
        n_envs:       int,
        seq_len:      int   = 30,
        n_features:   int   = 22,
        multiplier:   float = 20.0,
        commission:   float = 2.0,
        contracts:    float = 1.0,
        max_loss:     float = 2_500.0,
        reward_scale: float = 100.0,
    ) -> None:
        self.n_envs       = n_envs
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.state_dim    = n_features + TradingEnv.N_EXTRA
        self.multiplier   = multiplier
        self.commission   = commission
        self.contracts    = contracts
        self.max_loss     = max_loss
        self.reward_scale = reward_scale

        # Allocated in reset_all
        self._features        : np.ndarray | None = None  # (n, max_bars, n_features)
        self._prices          : np.ndarray | None = None  # (n, max_bars)
        self._lengths         : np.ndarray | None = None  # (n,) int32
        self._cursors         : np.ndarray | None = None  # (n,) int32
        self._positions       : np.ndarray | None = None  # (n,) int32
        self._entry_prices    : np.ndarray | None = None  # (n,) float32
        self._peak_unrealised : np.ndarray | None = None  # (n,) float32
        self._prev_closes     : np.ndarray | None = None  # (n,) float32
        self._ei              : np.ndarray | None = None  # np.arange(n), cached

    # ------------------------------------------------------------------

    def reset_all(
        self,
        episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        episodes : list of (features, prices, fwd_returns) of length n_envs
        Returns  : initial states (n_envs, seq_len, state_dim)
        """
        n        = self.n_envs
        max_bars = max(len(f) for f, _, _ in episodes)

        self._features = np.zeros((n, max_bars, self.n_features), dtype=np.float32)
        self._prices   = np.zeros((n, max_bars),                   dtype=np.float32)
        self._lengths  = np.empty(n, dtype=np.int32)

        for i, (feat, prices, _) in enumerate(episodes):
            L = len(feat)
            self._features[i, :L] = feat.astype(np.float32)
            self._prices  [i, :L] = prices.astype(np.float32)
            self._lengths [i]     = L

        self._ei              = np.arange(n)
        self._cursors         = np.full(n, self.seq_len, dtype=np.int32)
        self._positions       = np.zeros(n, dtype=np.int32)
        self._entry_prices    = np.zeros(n, dtype=np.float32)
        self._peak_unrealised = np.zeros(n, dtype=np.float32)
        self._prev_closes     = self._prices[self._ei, self.seq_len - 1]
        return self._build_states()

    # ------------------------------------------------------------------

    def step_all(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        actions : (n_envs,) int array — {0 FLAT, 1 LONG, 2 SHORT}
        Returns : states (n, seq_len, state_dim), rewards (n,), dones (n,)
        """
        ei          = self._ei
        curr_closes = self._prices[ei, self._cursors]                     # (n,)

        # Map action integers to signed positions vectorially
        desired = np.where(actions == 1, np.int32(1),
                  np.where(actions == 2, np.int32(-1), np.int32(0)))

        # ── Trade execution (mirrors TradingEnv.step order exactly) ───
        trade      = desired != self._positions
        close_cost = (self._positions != 0) & trade
        open_cost  = (desired != 0)         & trade
        commission = (
            (close_cost.astype(np.float32) + open_cost.astype(np.float32))
            * self.commission * self.contracts
        )

        # Positions and entry prices are updated BEFORE MTM — TradingEnv
        # updates self._position before computing mtm, so a newly opened
        # position receives credit for the current bar's price move.
        new_pos   = np.where(trade, desired, self._positions)
        new_entry = np.where(
            trade & (desired != 0), curr_closes,
            np.where(trade & (desired == 0), np.float32(0.0), self._entry_prices),
        )
        # Reset peak when a trade opens or closes
        new_peak = np.where(trade, np.float32(0.0), self._peak_unrealised)

        # MTM uses the NEW (post-trade) position
        mtm     = ((curr_closes - self._prev_closes)
                   * new_pos * self.contracts * self.multiplier)
        rewards = (mtm - commission) / self.reward_scale

        # Trailing drawdown stop — evaluated on NEW position / entry
        has_pos  = (new_pos != 0) & (new_entry > 0.0)
        upnl     = (curr_closes - new_entry) * new_pos * self.contracts * self.multiplier
        new_peak = np.where(has_pos & (upnl > new_peak), upnl, new_peak)
        stopped  = has_pos & ((new_peak - upnl) >= self.max_loss)
        rewards  -= stopped.astype(np.float32) * self.commission * self.contracts / self.reward_scale
        new_pos   = np.where(stopped, np.int32(0),     new_pos)
        new_entry = np.where(stopped, np.float32(0.0), new_entry)
        new_peak  = np.where(stopped, np.float32(0.0), new_peak)

        self._positions       = new_pos
        self._entry_prices    = new_entry
        self._peak_unrealised = new_peak
        self._prev_closes  = curr_closes
        self._cursors     += 1

        # End-of-episode forced flat
        dones   = self._cursors >= self._lengths
        eod_pos = dones & (self._positions != 0)
        rewards -= eod_pos.astype(np.float32) * self.commission * self.contracts / self.reward_scale
        self._positions       = np.where(dones, np.int32(0),     self._positions)
        self._entry_prices    = np.where(dones, np.float32(0.0), self._entry_prices)
        self._peak_unrealised = np.where(dones, np.float32(0.0), self._peak_unrealised)

        return self._build_states(), rewards.astype(np.float32), dones

    # ------------------------------------------------------------------

    def _build_states(self) -> np.ndarray:
        """
        Build (n_envs, seq_len, state_dim) for all envs with one numpy call.
        Uses advanced integer indexing — zero Python loops.
        """
        ei  = self._ei                              # (n,)
        c   = self._cursors                         # (n,)
        j   = np.arange(self.seq_len)               # (seq_len,)

        # Time indices into the padded features array
        t_idx = (c[:, None] - self.seq_len) + j     # (n, seq_len)
        # Clamp: envs that have run past their episode end reuse last valid bar
        t_idx = np.clip(t_idx, 0, self._features.shape[1] - 1)

        seq = self._features[ei[:, None], t_idx, :] # (n, seq_len, n_features)

        # Position encoding: broadcast scalar per env across the seq window
        pos_enc = (self._positions.astype(np.float32)[:, None, None]
                   * np.ones((1, self.seq_len, 1), dtype=np.float32))

        # Unrealised P&L encoding
        prev_p   = self._prices[ei, np.maximum(c - 1, 0)]
        upnl_raw = np.where(
            (self._positions != 0) & (self._entry_prices > 0.0),
            ((prev_p - self._entry_prices)
             * self._positions * self.contracts * self.multiplier
             / self.reward_scale),
            np.float32(0.0),
        ).astype(np.float32)
        upnl_raw = np.clip(upnl_raw, -10.0, 10.0)
        upnl_arr = (upnl_raw[:, None, None]
                    * np.ones((1, self.seq_len, 1), dtype=np.float32))

        return np.concatenate([seq, pos_enc, upnl_arr], axis=-1)  # (n, seq_len, state_dim)
