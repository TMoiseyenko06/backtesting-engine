"""
Intraday futures trading environment for PPO with bracket orders.

One episode = all bars for a single UTC calendar date.  The env steps through
bars one at a time and enforces:
  - Intraday-only (EOD forced-flat at episode end)
  - Bracket orders: when the model enters, a STOP (SL) and LIMIT (TP) are set
    simultaneously.  The trade closes automatically when either price is hit
    intrabar (checked via bar HIGH/LOW).  The model is ignored while in a bracket.

State
-----
  (seq_len, n_features + 2)  float32
  The last two channels appended to each timestep are context features:
    position_enc  : float  {-1.0 short, 0.0 flat, +1.0 long}
    norm_upnl     : unrealised P&L / reward_scale, clamped to [-10, +10]

Actions / directions
--------------------
  BatchedTradingEnv.step_all() accepts (directions, sl_pts, tp_pts):
    directions : int array (-1, 0, +1)
    sl_pts     : float array — stop-loss distance in NQ points
    tp_pts     : float array — take-profit distance in NQ points

  When an environment is currently in a bracket, its direction is ignored.
  Only flat environments act on the direction signal.

Reward
------
  Standard mode (binary_reward=False):
    Entry bar    : -entry_commission / reward_scale
    In-bracket   : MTM change (close - prev_close) * pos * multiplier / reward_scale
    Exit bar(SL) : (sl_price - prev_close) * pos * multiplier / reward_scale
                   - exit_commission / reward_scale
    Exit bar(TP) : (tp_price - prev_close) * pos * multiplier / reward_scale
                   - exit_commission / reward_scale
    EOD exit     : (close - prev_close) * pos * multiplier / reward_scale
                   - exit_commission / reward_scale

    Total bracket reward = (exit_price - entry_price) * direction
                           * multiplier * contracts / reward_scale
                           - 2 * commission / reward_scale

  Binary mode (binary_reward=True):
    With fixed SL/TP, P&L is a pure linear function of win rate, so maximising
    P&L IS maximising win rate.  The bar-by-bar MTM signal is pure noise —
    intermediate wiggles don't affect the final TP/SL outcome.  Binary mode
    removes that noise and reduces the task to a classification problem:
    "will price hit TP before SL?"

    In-bracket      : 0
    Exit bar(TP)    : +(win_bonus or reward_scale) / reward_scale  ≈ +1.0
    Exit bar(SL)    : -(win_bonus or reward_scale) / reward_scale  ≈ -1.0
    Exit bar(timeout): -(win_bonus or reward_scale) / reward_scale ≈ -1.0
    Flat bar        : -flat_penalty / reward_scale  (same as standard mode)
    Commissions     : omitted (constant cost; adds noise without information)

Time-limit timeout (time_limit_bars > 0):
    If a bracket trade has not been resolved by SL or TP within time_limit_bars
    bars of entry, it is force-closed at the current bar's close and penalised
    as a loss.  In binary mode this gives -1.0; in standard mode the MTM at
    close is used (same as an EOD exit).  This trains the model to only enter
    when it expects a fast, decisive move — avoiding "stuck" trades that tie
    up capital without resolving.
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
    Single-episode environment (kept for API compatibility).
    The BatchedTradingEnv is used for training; this class is for reference.

    Parameters
    ----------
    seq_len       : int    History window fed to the policy network.
    n_features    : int    Raw OHLCV feature count (before augmentation).
    multiplier    : float  Contract point value, e.g. $20 for NQ.
    commission    : float  $ per contract per side.
    contracts     : float  Position size.
    max_loss      : float  Hard cap on SL distance: sl_pts is clamped so that
                           max dollar loss per trade ≤ max_loss.
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
        flat_penalty:   float = 0.0,
        win_bonus:      float = 0.0,
        binary_reward:  bool  = False,
    ) -> None:
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.state_dim    = n_features + self.N_EXTRA
        self.multiplier   = multiplier
        self.commission   = commission
        self.contracts    = contracts
        self.max_loss     = max_loss   # retained for API compat, not used
        self.reward_scale = reward_scale

        # Episode state (initialised by reset)
        self._features    : np.ndarray | None = None
        self._prices      : np.ndarray | None = None
        self._n           : int   = 0
        self._cursor      : int   = 0
        self._position    : int   = 0      # −1, 0, +1
        self._entry_price : float = 0.0
        self._prev_close  : float = 0.0

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
        return self._position

    @property
    def unrealised_pnl(self) -> float:
        if self._position == 0 or self._entry_price == 0.0:
            return 0.0
        last_price = float(self._prices[self._cursor - 1]) if self._cursor > 0 else self._entry_price
        return (
            (last_price - self._entry_price)
            * self._position * self.contracts * self.multiplier
        )


class BatchedTradingEnv:
    """
    Vectorised bracket-order trading environment.

    Runs n_envs episodes simultaneously using numpy array operations.
    Each episode uses HIGH/LOW data to check intrabar SL/TP fills, matching
    the live backtesting engine's order-fill logic.

    API
    ---
        states = env.reset_all(episodes)
        states, rewards, dones, entered = env.step_all(directions, sl_pts, tp_pts)

    Episodes must be 5-tuples: (features, closes, highs, lows, fwd_returns).
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
        flat_penalty:         float = 0.0,
        win_bonus:            float = 0.0,
        binary_reward:        bool  = False,
        time_limit_bars:      int   = 0,   # 0 = disabled; >0 = force exit + penalty after N bars
        max_trades_per_episode: int = 0,   # 0 = unlimited; >0 = cap entries per day
    ) -> None:
        self.n_envs                = n_envs
        self.seq_len               = seq_len
        self.n_features            = n_features
        self.state_dim             = n_features + TradingEnv.N_EXTRA
        self.multiplier            = multiplier
        self.commission      = commission
        self.contracts       = contracts
        self.max_loss        = max_loss
        self.reward_scale    = reward_scale
        self.flat_penalty    = flat_penalty
        self.win_bonus       = win_bonus
        self.binary_reward          = binary_reward
        self.time_limit_bars        = time_limit_bars
        self.max_trades_per_episode = max_trades_per_episode
        self.tp_hit_total      = 0
        self.sl_hit_total      = 0
        self.timeout_hit_total = 0

        # Allocated in reset_all — episode data (padded 2D arrays)
        self._features  : np.ndarray | None = None  # (n, max_bars, n_features)
        self._prices    : np.ndarray | None = None  # (n, max_bars) — closes
        self._highs     : np.ndarray | None = None  # (n, max_bars)
        self._lows      : np.ndarray | None = None  # (n, max_bars)
        self._lengths   : np.ndarray | None = None  # (n,) int32

        # Per-environment state
        self._cursors      : np.ndarray | None = None  # (n,) int32
        self._positions    : np.ndarray | None = None  # (n,) int32  {-1,0,+1}
        self._entry_prices : np.ndarray | None = None  # (n,) float32
        self._sl_prices    : np.ndarray | None = None  # (n,) float32 absolute SL
        self._tp_prices    : np.ndarray | None = None  # (n,) float32 absolute TP
        self._in_bracket   : np.ndarray | None = None  # (n,) bool
        self._prev_closes  : np.ndarray | None = None  # (n,) float32 for MTM
        self._bars_in_trade      : np.ndarray | None = None  # (n,) int32 — bars elapsed since entry
        self._trades_this_episode: np.ndarray | None = None  # (n,) int32 — entries taken today
        self._ei                 : np.ndarray | None = None  # np.arange(n), cached

    # ------------------------------------------------------------------

    def reset_all(
        self,
        episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        episodes : list of (features, closes, highs, lows, fwd_returns), length n_envs
        Returns  : initial states (n_envs, seq_len, state_dim)
        """
        n        = self.n_envs
        max_bars = max(len(f) for f, *_ in episodes)

        self._features = np.zeros((n, max_bars, self.n_features), dtype=np.float32)
        self._prices   = np.zeros((n, max_bars),                   dtype=np.float32)
        self._highs    = np.zeros((n, max_bars),                   dtype=np.float32)
        self._lows     = np.zeros((n, max_bars),                   dtype=np.float32)
        self._lengths  = np.empty(n, dtype=np.int32)

        for i, (feat, prices, highs, lows, _) in enumerate(episodes):
            L = len(feat)
            self._features[i, :L] = feat.astype(np.float32)
            self._prices  [i, :L] = prices.astype(np.float32)
            self._highs   [i, :L] = highs.astype(np.float32)
            self._lows    [i, :L] = lows.astype(np.float32)
            self._lengths [i]     = L

        self._ei             = np.arange(n)
        self._cursors        = np.full(n, self.seq_len, dtype=np.int32)
        self._positions      = np.zeros(n, dtype=np.int32)
        self._entry_prices   = np.zeros(n, dtype=np.float32)
        self._sl_prices      = np.zeros(n, dtype=np.float32)
        self._tp_prices      = np.zeros(n, dtype=np.float32)
        self._in_bracket          = np.zeros(n, dtype=bool)
        self._bars_in_trade       = np.zeros(n, dtype=np.int32)
        self._trades_this_episode = np.zeros(n, dtype=np.int32)
        self._prev_closes    = self._prices[self._ei, self.seq_len - 1]
        self.tp_hit_total      = 0
        self.sl_hit_total      = 0
        self.timeout_hit_total = 0
        return self._build_states()

    # ------------------------------------------------------------------

    def step_all(
        self,
        directions: np.ndarray,
        sl_pts:     np.ndarray,
        tp_pts:     np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments one bar.

        Parameters
        ----------
        directions : (n_envs,) int — desired direction {-1, 0, +1}.
                     Ignored for environments currently in a bracket.
        sl_pts     : (n_envs,) float — stop-loss distance in NQ points.
        tp_pts     : (n_envs,) float — take-profit distance in NQ points.

        Returns
        -------
        states          : (n, seq_len, state_dim)
        rewards         : (n,) float32
        dones           : (n,) bool
        entered_bracket : (n,) bool — True for envs that opened a new bracket
                          this step (used by PPO to attribute sl/tp log_probs).
        """
        ei          = self._ei
        curr_closes = self._prices[ei, self._cursors]   # (n,)
        bar_highs   = self._highs [ei, self._cursors]   # (n,)
        bar_lows    = self._lows  [ei, self._cursors]   # (n,)

        in_b      = self._in_bracket
        long_pos  = self._positions == 1
        short_pos = self._positions == -1

        # ── 1. Check SL/TP intrabar hits ───────────────────────────────
        # SL: bar_low touched SL price (long) or bar_high touched SL price (short)
        sl_hit = (
            (in_b & long_pos  & (bar_lows  <= self._sl_prices)) |
            (in_b & short_pos & (bar_highs >= self._sl_prices))
        )
        # TP: bar_high touched TP price (long) or bar_low touched TP price (short)
        tp_hit = (
            (in_b & long_pos  & (bar_highs >= self._tp_prices)) |
            (in_b & short_pos & (bar_lows  <= self._tp_prices))
        )
        # If both hit on same bar, SL takes precedence (conservative)
        tp_hit = tp_hit & ~sl_hit

        # ── 1b. Time-limit timeout ─────────────────────────────────────
        # If the trade hasn't resolved within time_limit_bars bars, force
        # exit at the current close and penalise like an SL hit.  Teaches
        # the model to only enter when it expects a fast, decisive move.
        if self.time_limit_bars > 0:
            timeout_hit = in_b & (self._bars_in_trade >= self.time_limit_bars) & ~sl_hit & ~tp_hit
        else:
            timeout_hit = np.zeros(self.n_envs, dtype=bool)

        bracket_exit = sl_hit | tp_hit | timeout_hit
        self.tp_hit_total      += int(tp_hit.sum())
        self.sl_hit_total      += int(sl_hit.sum())
        self.timeout_hit_total += int(timeout_hit.sum())

        # ── 2. MTM reward using actual exit price for bracket exits ────
        # For bars where SL fires, price moved to sl_price; for TP, to tp_price.
        # For all other bars (including in-bracket hold), use current close.
        effective_price = np.where(
            sl_hit, self._sl_prices,
            np.where(tp_hit, self._tp_prices, curr_closes)
        )
        exit_commission = bracket_exit.astype(np.float32) * self.commission * self.contracts

        mtm     = ((effective_price - self._prev_closes)
                   * self._positions.astype(np.float32)
                   * self.contracts * self.multiplier)
        # ── 3. Update bracket state after exits ────────────────────────
        post_exit_pos   = np.where(bracket_exit, np.int32(0),     self._positions)
        post_exit_entry = np.where(bracket_exit, np.float32(0.0), self._entry_prices)
        post_exit_sl    = np.where(bracket_exit, np.float32(0.0), self._sl_prices)
        post_exit_tp    = np.where(bracket_exit, np.float32(0.0), self._tp_prices)
        post_exit_in_b  = self._in_bracket & ~bracket_exit

        # ── 4. New entries: only for flat envs not in bracket ──────────
        # Also enforce max_trades_per_episode when set (0 = unlimited).
        trade_quota_ok = (
            (self.max_trades_per_episode <= 0) |
            (self._trades_this_episode < self.max_trades_per_episode)
        )
        can_enter   = ~post_exit_in_b & (post_exit_pos == 0) & (directions != 0) & trade_quota_ok
        entry_price = curr_closes   # enter at current bar's close

        dir_f       = directions.astype(np.float32)
        # Hard-cap sl_pts: max loss per trade must not exceed max_loss dollars
        _max_sl_pts = self.max_loss / (self.contracts * self.multiplier)
        sl_pts      = np.minimum(sl_pts, _max_sl_pts)
        sl_price_new = entry_price - dir_f * sl_pts
        tp_price_new = entry_price + dir_f * tp_pts

        new_pos    = np.where(can_enter, directions,    post_exit_pos)
        new_entry  = np.where(can_enter, entry_price,   post_exit_entry)
        new_sl     = np.where(can_enter, sl_price_new,  post_exit_sl)
        new_tp     = np.where(can_enter, tp_price_new,  post_exit_tp)
        new_in_b   = post_exit_in_b | can_enter

        # Flat penalty: only charge when the model chose to stay flat this bar
        # (not on entry bars — those already pay commission).
        # Re-compute using can_enter so entry bars are excluded.
        flat_cost = ((post_exit_pos == 0) & ~can_enter).astype(np.float32) * self.flat_penalty

        if self.binary_reward:
            # Binary mode: with fixed SL/TP, P&L is a linear function of win rate.
            # Bar-by-bar MTM is pure noise — only the bracket outcome matters.
            # TP → +1,  SL → -1,  timeout → -1 (failed to pick a decisive move).
            binary_scale = self.win_bonus if self.win_bonus > 0.0 else self.reward_scale
            outcome = (tp_hit.astype(np.float32)
                       - sl_hit.astype(np.float32)
                       - timeout_hit.astype(np.float32))
            rewards = (outcome * binary_scale - flat_cost) / self.reward_scale
        else:
            # Standard mode: full MTM + commission signal.
            # timeout exits at current close (effective_price already = curr_closes for non-SL/TP).
            win_loss_bonus = (tp_hit.astype(np.float32) - sl_hit.astype(np.float32)) * self.win_bonus
            rewards = (mtm - exit_commission - flat_cost + win_loss_bonus) / self.reward_scale
            # Entry commission
            rewards -= (can_enter.astype(np.float32)
                        * self.commission * self.contracts / self.reward_scale)

        # ── 5. Commit state ────────────────────────────────────────────
        self._positions    = new_pos.astype(np.int32)
        self._entry_prices = new_entry
        self._sl_prices    = new_sl
        self._tp_prices    = new_tp
        self._in_bracket   = new_in_b
        self._prev_closes  = curr_closes
        self._cursors     += 1
        self._trades_this_episode += can_enter.astype(np.int32)

        # Update bars-in-trade counter:
        #   exited (any reason) or EOD → 0
        #   new entry this bar         → 1
        #   still in bracket           → old + 1
        #   flat                       → 0
        self._bars_in_trade = np.where(
            bracket_exit,
            np.int32(0),
            np.where(can_enter,
                     np.int32(1),
                     np.where(new_in_b,
                              self._bars_in_trade + np.int32(1),
                              np.int32(0))),
        )

        # ── 6. EOD forced flat ─────────────────────────────────────────
        dones          = self._cursors >= self._lengths
        eod_in_bracket = dones & self._in_bracket

        # EOD exit: pays exit commission in standard mode only.
        # Binary mode omits commissions — they are constant and add no signal.
        if not self.binary_reward:
            rewards -= (eod_in_bracket.astype(np.float32)
                        * self.commission * self.contracts / self.reward_scale)

        self._positions    = np.where(dones, np.int32(0),     self._positions)
        self._entry_prices = np.where(dones, np.float32(0.0), self._entry_prices)
        self._sl_prices    = np.where(dones, np.float32(0.0), self._sl_prices)
        self._tp_prices    = np.where(dones, np.float32(0.0), self._tp_prices)
        self._in_bracket   = np.where(dones, False,            self._in_bracket)

        return (
            self._build_states(),
            rewards.astype(np.float32),
            dones,
            can_enter,   # entered_bracket mask
        )

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
