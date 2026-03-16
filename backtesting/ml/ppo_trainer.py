"""
PPO (Proximal Policy Optimization) trainer for intraday futures trading
with bracket orders.

Training objective
------------------
  Total loss = PPO clip loss  +  value_loss_coef × value MSE
             - entropy_coef × entropy bonus
             + pred_loss_coef × MSE(predicted_H_bar_return, actual_H_bar_return)

  The PPO clip loss uses a *combined* log probability:
    log_prob = log_prob_direction + entered_bracket × (log_prob_sl + log_prob_tp)

  This means SL and TP are trained only on steps where the model actually
  opened a bracket (entered_bracket=True).  On hold/flat steps, only the
  direction head is trained.

Bracket order model
-------------------
  When the model selects LONG or SHORT, it simultaneously predicts:
    sl_pts : stop-loss distance in NQ points (range [25, 500])
    tp_pts : take-profit distance in NQ points (range [25, 1500])

  The entry + SL + TP are submitted together.  The model is ignored while
  a bracket is open — the trade exits automatically when SL or TP is hit
  intrabar (checked via HIGH/LOW).

Training loop (per iteration)
------------------------------
  1. Sample ``rollout_days`` random trading-day episodes.
  2. Roll out the current policy, collecting per-step:
       obs, action, sl_sample, tp_sample, log_prob (combined),
       entered_bracket, value, reward, done, actual_fwd_return
  3. Compute GAE advantages and discounted returns.
  4. Run ``ppo_epochs`` mini-batch updates using the full loss above.
  5. Print progress every ``print_every`` iterations.

Multi-GPU
---------
DataParallel is used for the PPO update step.  Rollout collection is
vectorised: all rollout_days episodes are stepped in lock-step, batching
their observations into a single GPU forward pass at every bar.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, TensorDataset

from backtesting.data_feed import Bar
from backtesting.ml.actor_critic import ActorCriticLSTM
from backtesting.ml.rl_env import BatchedTradingEnv, TradingEnv
from backtesting.ml.model import _auto_device


class PPOTrainer:
    """
    Parameters
    ----------
    n_features         : int    Raw OHLCV feature count.
    hidden_size        : int    LSTM hidden units.
    num_layers         : int
    dropout            : float
    seq_len            : int    Observation window length.
    lr                 : float  Adam learning rate.
    n_iterations       : int    Total PPO training iterations.
    rollout_days       : int    Episodes to sample per iteration.
    ppo_epochs         : int    Update passes per iteration.
    minibatch_size     : int
    clip_eps           : float  PPO clipping epsilon.
    gamma              : float  Discount factor.
    gae_lambda         : float  GAE λ.
    value_loss_coef    : float
    entropy_coef       : float  Entropy bonus — higher → more exploration, prevents FLAT collapse.
    flat_penalty       : float  Dollar penalty per flat bar in the training env.  Incentivises
                                the model to trade more frequently.  Scale: same as commission
                                ($2 round-trip ≈ 0.5–2.0/bar for ~5 trades/day).
    pred_loss_coef     : float  Weight on auxiliary prediction MSE loss.
    prediction_horizon : int    H — bars ahead to predict (default 30 = 30 minutes).
    max_grad_norm      : float
    print_every        : int
    device             : str    Auto-detected if None.
    multiplier / commission / contracts / max_loss / reward_scale
        Must match the values used in RLTradingStrategy for consistency.
    """

    def __init__(
        self,
        n_features:          int   = 22,
        hidden_size:         int   = 512,
        num_layers:          int   = 2,
        dropout:             float = 0.3,
        seq_len:             int   = 30,
        lr:                  float = 3e-4,
        n_iterations:        int   = 200,
        rollout_days:        int   = 16,
        ppo_epochs:          int   = 4,
        minibatch_size:      int   = 512,
        clip_eps:            float = 0.2,
        gamma:               float = 0.99,
        gae_lambda:          float = 0.95,
        value_loss_coef:     float = 0.25,   # reduced: return normalisation makes value loss O(1)
        entropy_coef:        float = 0.05,   # higher → more exploration, prevents FLAT collapse
        flat_penalty:        float = 1.0,    # dollar cost per flat bar — same order as commission
        pred_loss_coef:      float = 0.01,   # reduced: auxiliary task should not dominate policy
        prediction_horizon:  int   = 30,      # H-bar ahead prediction target
        max_grad_norm:       float = 0.5,
        print_every:         int   = 10,
        device:              Optional[str] = None,
        multiplier:          float = 20.0,
        commission:          float = 2.0,
        contracts:           float = 1.0,
        max_loss:            float = 2_500.0,
        reward_scale:        float = 100.0,
        weight_decay:        float = 1e-4,    # L2 regularisation — penalises large weights
        val_frac:            float = 0.15,    # fraction of training days held out for validation
        patience:            int   = 80,      # early-stop after this many iters with no val improvement
        pretrained_path:     Optional[str] = None,  # .pt checkpoint to resume from
    ) -> None:
        self.seq_len            = seq_len
        self.n_iterations       = n_iterations
        self.rollout_days       = rollout_days
        self.ppo_epochs         = ppo_epochs
        self.minibatch_size     = minibatch_size
        self.clip_eps           = clip_eps
        self.gamma              = gamma
        self.gae_lambda         = gae_lambda
        self.value_loss_coef    = value_loss_coef
        self.entropy_coef       = entropy_coef
        self.flat_penalty       = flat_penalty
        self.pred_loss_coef     = pred_loss_coef
        self.prediction_horizon = prediction_horizon
        self.max_grad_norm      = max_grad_norm
        self.print_every        = print_every
        self.reward_scale       = reward_scale
        self.device             = device or _auto_device()
        self.val_frac           = val_frac
        self.patience           = patience

        # Store kwargs so _collect_rollout can spin up per-episode envs
        self._env_kwargs = dict(
            seq_len=seq_len,
            n_features=n_features,
            multiplier=multiplier,
            commission=commission,
            contracts=contracts,
            max_loss=max_loss,
            reward_scale=reward_scale,
            flat_penalty=flat_penalty,
        )
        self._env = TradingEnv(**self._env_kwargs)   # kept for external callers

        # ── GPU detection ─────────────────────────────────────────────────
        _gpu_name = (
            torch.cuda.get_device_name(0)
            if self.device == "cuda" and torch.cuda.is_available()
            else ""
        )
        _is_h200 = "H200" in _gpu_name
        _is_a100 = "A100" in _gpu_name

        # ── AMP — LSTM shields itself to float32, linear heads use BF16/FP16
        # Both H200 and A100 support BF16 natively; use FP16 on older GPUs.
        self._use_amp   = (self.device == "cuda")
        self._amp_dtype = torch.bfloat16 if (_is_h200 or _is_a100) else torch.float16
        self._scaler    = torch.amp.GradScaler(
            device="cuda", enabled=self._use_amp
        )

        # ── H200: scale hidden_size BEFORE model construction ────────────
        if _is_h200:
            hidden_size = hidden_size * 4   # 512 → 2048
            print(f"  [H200] hidden_size scaled to {hidden_size}")

        # ── A100: scale hidden_size BEFORE model construction ────────────
        elif _is_a100:
            hidden_size = hidden_size * 2   # 512 → 1024
            print(f"  [A100] hidden_size scaled to {hidden_size}")

        base_model = ActorCriticLSTM(
            n_features=n_features + TradingEnv.N_EXTRA,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        n_gpus = torch.cuda.device_count() if self.device == "cuda" else 1
        if n_gpus > 1:
            self.minibatch_size = minibatch_size * n_gpus
            self.model = nn.DataParallel(base_model)
            print(
                f"  [multi-gpu] DataParallel across {n_gpus} GPUs  "
                f"(minibatch scaled {minibatch_size} → {self.minibatch_size}  "
                f"·  {minibatch_size} per GPU)"
            )
        else:
            self.model = base_model

        if _is_h200:
            self.rollout_days   = rollout_days   * 32   # 16   → 512 episodes
            self.minibatch_size = self.minibatch_size * 8   # → 8192 effective
            self.ppo_epochs     = ppo_epochs * 2            # 4    → 8 epochs
            print(
                f"  [H200] auto-scale: rollout_days={self.rollout_days}  "
                f"minibatch={self.minibatch_size}  ppo_epochs={self.ppo_epochs}"
            )

        elif _is_a100:
            self.rollout_days   = rollout_days   * 8    # 16  → 128 episodes
            self.minibatch_size = self.minibatch_size * 8   # 512 → 4096
            self.ppo_epochs     = max(ppo_epochs, 6)        # 4   → 6 epochs
            print(
                f"  [A100] auto-scale: rollout_days={self.rollout_days}  "
                f"minibatch={self.minibatch_size}  ppo_epochs={self.ppo_epochs}"
            )

        self._policy = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location=device)
            self._policy.load_state_dict(sd, strict=True)
            print(f"  Resumed weights from {pretrained_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, bars: List[Bar], features: np.ndarray) -> dict:
        """
        Train the actor-critic on historical intraday data.

        bars     : training Bar objects (same ordering as features rows)
        features : (n_bars, n_features) pre-computed feature matrix
        """
        episodes = self._make_episodes(bars, features)
        if not episodes:
            raise ValueError("No valid training episodes found (all days too short).")

        # ── Train / validation split (chronological — last val_frac days) ────
        # Chronological split avoids using future data to validate past models.
        n_val          = max(1, int(len(episodes) * self.val_frac))
        train_episodes = episodes[:-n_val]
        val_episodes   = episodes[-n_val:]

        print(
            f"  [PPO] {len(train_episodes)} train days  ·  {len(val_episodes)} val days  ·  "
            f"prediction horizon = {self.prediction_horizon} bars  ·  "
            f"{self.n_iterations} iterations  ·  "
            f"{self.rollout_days} days/rollout  ·  "
            f"{self.ppo_epochs} PPO epochs/iter\n"
        )

        best_val_pnl   = -float("inf")
        best_state     = None
        iters_no_improve = 0

        last_metrics: dict = {}
        for iteration in range(1, self.n_iterations + 1):
            rollout      = self._collect_rollout(train_episodes)
            self._compute_gae(rollout)
            metrics      = self._ppo_update(rollout)
            last_metrics = metrics
            last_metrics["mean_episode_pnl"] = (
                float(np.mean(rollout["episode_returns"])) * self.reward_scale
            )
            last_metrics["trades_per_day"] = float(rollout["trades_per_day"])
            last_metrics["mean_pred_error_pts"] = float(
                np.mean(np.abs(rollout["pred_return_errors"]))
            ) if len(rollout.get("pred_return_errors", [])) > 0 else 0.0

            if iteration % self.print_every == 0 or iteration == 1:
                val_pnl = self._evaluate_val(val_episodes)
                last_metrics["val_pnl"] = val_pnl

                improved = val_pnl > best_val_pnl
                if improved:
                    best_val_pnl = val_pnl
                    best_state   = {k: v.cpu().clone() for k, v in self._policy.state_dict().items()}
                    iters_no_improve = 0
                else:
                    iters_no_improve += self.print_every

                flag = " *" if improved else ""
                print(
                    f"    iter {iteration:>4}  "
                    f"train_pnl=${last_metrics['mean_episode_pnl']:+.0f}  "
                    f"val_pnl=${val_pnl:+.0f}{flag}  "
                    f"trades/day={last_metrics['trades_per_day']:.1f}  "
                    f"policy={metrics['policy_loss']:.4f}  "
                    f"value={metrics['value_loss']:.4f}  "
                    f"entropy={metrics['entropy']:.4f}  "
                    f"clip={metrics['clip_frac']:.3f}"
                )

                if self.patience > 0 and iters_no_improve >= self.patience:
                    print(
                        f"  [early stop] val_pnl has not improved for {self.patience} iters  "
                        f"·  best val_pnl=${best_val_pnl:+.0f}"
                    )
                    break

        if best_state is not None:
            self._policy.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
            print(f"  [PPO] restored best model  (val_pnl=${best_val_pnl:+.0f})")

        return last_metrics

    def save(self, path) -> None:
        self._policy.save(path)

    # ------------------------------------------------------------------
    # Episode construction
    # ------------------------------------------------------------------

    def _make_episodes(
        self, bars: List[Bar], features: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Group bar indices by UTC calendar date.
        Returns list of (features, closes, highs, lows, forward_returns) tuples.

        forward_returns[i] = (closes[i + H] - closes[i]) / closes[i]
          = fraction price change H bars ahead (auxiliary regression target).
        highs and lows are required for intrabar SL/TP simulation in step_all().
        """
        day_map: dict = defaultdict(list)
        for i, bar in enumerate(bars):
            day_map[bar.timestamp.date()].append(i)

        H = self.prediction_horizon
        episodes = []
        for date in sorted(day_map.keys()):
            idx = day_map[date]
            if len(idx) <= self.seq_len:
                continue
            f = features[idx].astype(np.float32)
            p = np.array([bars[i].close for i in idx], dtype=np.float32)
            h = np.array([bars[i].high  for i in idx], dtype=np.float32)
            l = np.array([bars[i].low   for i in idx], dtype=np.float32)

            # H-bar forward returns (regression target)
            n = len(p)
            fwd = np.zeros(n, dtype=np.float32)
            for i in range(n - H):
                if p[i] > 0:
                    fwd[i] = (p[i + H] - p[i]) / p[i]

            episodes.append((f, p, h, l, fwd))
        return episodes

    # ------------------------------------------------------------------
    # Validation evaluation (greedy policy, no gradient)
    # ------------------------------------------------------------------

    def _evaluate_val(self, val_episodes: list) -> float:
        """
        Run the current policy greedily on val_episodes.
        Returns mean daily PnL in dollars.
        """
        self._policy.eval()
        n_ep        = len(val_episodes)
        env         = BatchedTradingEnv(n_envs=n_ep, **self._env_kwargs)
        all_states  = env.reset_all(val_episodes)
        active_mask = np.ones(n_ep, dtype=bool)
        ep_rets     = np.zeros(n_ep, dtype=np.float32)

        with torch.no_grad():
            while active_mask.any():
                active = np.where(active_mask)[0]
                x      = torch.from_numpy(all_states[active]).to(self.device)

                logits, _, _, sl_mean, tp_mean = self._policy(x)
                actions_np = logits.argmax(dim=-1).cpu().numpy()
                sl_np      = sl_mean.cpu().numpy()
                tp_np      = tp_mean.cpu().numpy()

                # Map action ints to directions
                directions_np = np.where(actions_np == 1, np.int32(1),
                                np.where(actions_np == 2, np.int32(-1), np.int32(0)))

                directions_all = np.zeros(n_ep, dtype=np.int32)
                sl_all         = np.zeros(n_ep, dtype=np.float32)
                tp_all         = np.zeros(n_ep, dtype=np.float32)
                directions_all[active] = directions_np
                sl_all        [active] = sl_np
                tp_all        [active] = tp_np

                all_states, rews, dones, _ = env.step_all(directions_all, sl_all, tp_all)
                ep_rets     += rews * active_mask.astype(np.float32)
                active_mask &= ~dones

        return float(ep_rets.mean()) * self.reward_scale

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(
        self,
        episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict:
        """
        Sample ``rollout_days`` episodes and collect trajectories.

        Uses BatchedTradingEnv so all active episodes are stepped with a
        single numpy call.  All per-step bookkeeping uses numpy advanced
        indexing into pre-allocated 2D arrays — zero Python loops in the
        hot path.

        The combined log_prob for PPO is:
          log_prob = log_prob_dir + entered_bracket * (log_prob_sl + log_prob_tp)

        This means sl/tp heads only receive gradients on bracket-entry steps.
        """
        self._policy.eval()

        sampled  = random.choices(episodes, k=self.rollout_days)
        n_ep     = len(sampled)
        max_bars = max(len(f) for f, *_ in sampled)

        # Pre-compute forward-return matrix for vectorised lookup
        fwd_arr = np.zeros((n_ep, max_bars), dtype=np.float32)
        for i, (*_, fwd) in enumerate(sampled):
            fwd_arr[i, :len(fwd)] = fwd

        state_dim = self._env_kwargs["n_features"] + TradingEnv.N_EXTRA

        # Pre-allocated 2D stores (n_ep × max_bars)
        obs_store     = np.zeros((n_ep, max_bars, self.seq_len, state_dim), dtype=np.float32)
        act_store     = np.zeros((n_ep, max_bars),                          dtype=np.int64)
        sl_store      = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        tp_store      = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        rew_store     = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        done_store    = np.zeros((n_ep, max_bars),                          dtype=bool)
        fret_store    = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        entered_store = np.zeros((n_ep, max_bars),                          dtype=bool)
        lp_dir_store  = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        lp_sl_store   = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        lp_tp_store   = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        val_store     = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        pr_store      = np.zeros((n_ep, max_bars),                          dtype=np.float32)

        ep_t    = np.zeros(n_ep, dtype=np.int32)
        ep_rets = np.zeros(n_ep, dtype=np.float32)

        env        = BatchedTradingEnv(n_envs=n_ep, **self._env_kwargs)
        all_states = env.reset_all(sampled)

        active_mask = np.ones(n_ep, dtype=bool)
        cursors     = np.full(n_ep, self.seq_len, dtype=np.int32)

        # GPU tensors accumulated; bulk-transferred after the loop (3 syncs)
        gpu_lp_dir   : list[torch.Tensor] = []
        gpu_lp_sl    : list[torch.Tensor] = []
        gpu_lp_tp    : list[torch.Tensor] = []
        gpu_sl       : list[torch.Tensor] = []
        gpu_tp       : list[torch.Tensor] = []
        gpu_values   : list[torch.Tensor] = []
        gpu_pred_rets: list[torch.Tensor] = []
        active_hist  : list[np.ndarray]   = []
        steps_hist   : list[np.ndarray]   = []

        with torch.no_grad():
            while active_mask.any():
                active   = np.where(active_mask)[0]
                curr     = all_states[active]
                t_active = ep_t[active]

                # ── GPU forward: one batched call for all active episodes ──
                x = torch.from_numpy(curr).to(self.device)
                (actions_t, lp_dir_t, lp_sl_t, lp_tp_t,
                 sl_t, tp_t, values_t, _, pred_rets) = self._policy.act(x)

                gpu_lp_dir   .append(lp_dir_t)
                gpu_lp_sl    .append(lp_sl_t)
                gpu_lp_tp    .append(lp_tp_t)
                gpu_sl       .append(sl_t)
                gpu_tp       .append(tp_t)
                gpu_values   .append(values_t)
                gpu_pred_rets.append(pred_rets)
                active_hist  .append(active)
                steps_hist   .append(t_active.copy())

                # ONE sync per step: actions + sl/tp needed on CPU for env
                actions_np = actions_t.cpu().numpy()
                sl_np      = sl_t.cpu().numpy()
                tp_np      = tp_t.cpu().numpy()

                # Map action ints {0,1,2} → directions {0,+1,-1}
                directions_np = np.where(actions_np == 1, np.int32(1),
                                np.where(actions_np == 2, np.int32(-1), np.int32(0)))

                # ── Vectorised stores: numpy fancy indexing ───────────────
                obs_store [active, t_active] = curr
                act_store [active, t_active] = actions_np
                sl_store  [active, t_active] = sl_np
                tp_store  [active, t_active] = tp_np
                fret_store[active, t_active] = np.where(
                    cursors[active] < fwd_arr.shape[1],
                    fwd_arr[active, cursors[active]],
                    np.float32(0.0),
                )
                ep_t[active]    += 1
                cursors[active] += 1

                # ── Vectorised env step ───────────────────────────────────
                directions_all = np.zeros(n_ep, dtype=np.int32)
                sl_all         = np.zeros(n_ep, dtype=np.float32)
                tp_all         = np.zeros(n_ep, dtype=np.float32)
                directions_all[active] = directions_np
                sl_all        [active] = sl_np
                tp_all        [active] = tp_np

                all_states, rews, dones, entered_b = env.step_all(
                    directions_all, sl_all, tp_all
                )

                rew_store    [active, t_active] = rews    [active]
                done_store   [active, t_active] = dones   [active]
                entered_store[active, t_active] = entered_b[active]
                ep_rets += rews * active_mask.astype(np.float32)

                active_mask &= ~dones

        # ── Bulk GPU→CPU: 3 syncs total for the entire rollout ──────────
        all_lp_dir = torch.cat(gpu_lp_dir   ).cpu().numpy()
        all_lp_sl  = torch.cat(gpu_lp_sl    ).cpu().numpy()
        all_lp_tp  = torch.cat(gpu_lp_tp    ).cpu().numpy()
        all_sl     = torch.cat(gpu_sl        ).cpu().numpy()
        all_tp     = torch.cat(gpu_tp        ).cpu().numpy()
        all_v      = torch.cat(gpu_values    ).cpu().numpy()
        all_pr     = torch.cat(gpu_pred_rets ).cpu().numpy()

        # Scatter into 2D stores (~max_bars iterations, not n_ep×max_bars)
        offset = 0
        for act_step, t_step in zip(active_hist, steps_hist):
            n = len(act_step)
            lp_dir_store[act_step, t_step] = all_lp_dir[offset:offset + n]
            lp_sl_store [act_step, t_step] = all_lp_sl [offset:offset + n]
            lp_tp_store [act_step, t_step] = all_lp_tp [offset:offset + n]
            val_store   [act_step, t_step] = all_v     [offset:offset + n]
            pr_store    [act_step, t_step] = all_pr    [offset:offset + n]
            offset += n

        # ── Flatten in episode order for correct GAE boundaries ─────────
        obs_c = []; act_c = []; sl_c  = []; tp_c  = []
        rew_c = []; don_c = []; frt_c = []; prd_c = []
        ent_c = []; lpd_c = []; lps_c = []; lpt_c = []
        val_c = []

        for i in range(n_ep):
            T = int(ep_t[i])
            if T == 0:
                continue
            obs_c.append(obs_store    [i, :T])
            act_c.append(act_store    [i, :T])
            sl_c .append(sl_store     [i, :T])
            tp_c .append(tp_store     [i, :T])
            rew_c.append(rew_store    [i, :T])
            don_c.append(done_store   [i, :T])
            frt_c.append(fret_store   [i, :T])
            prd_c.append(pr_store     [i, :T])
            ent_c.append(entered_store[i, :T])
            lpd_c.append(lp_dir_store [i, :T])
            lps_c.append(lp_sl_store  [i, :T])
            lpt_c.append(lp_tp_store  [i, :T])
            val_c.append(val_store    [i, :T])

        fwd_np      = np.concatenate(frt_c, axis=0)
        pred_np     = np.concatenate(prd_c, axis=0)
        entered_np  = np.concatenate(ent_c, axis=0)
        lp_dir_np   = np.concatenate(lpd_c, axis=0)
        lp_sl_np    = np.concatenate(lps_c, axis=0)
        lp_tp_np    = np.concatenate(lpt_c, axis=0)

        # Combined log_prob: direction only — SL/TP log_probs (~-10 to -12 magnitude)
        # dwarf the direction term (~-1) and kill the direction gradient once they stabilise.
        combined_lp = lp_dir_np

        total_entries = float(entered_np.sum())
        trades_per_day = total_entries / max(n_ep, 1)

        return {
            "obs":                np.concatenate(obs_c, axis=0).astype(np.float32),
            "actions":            np.concatenate(act_c, axis=0).astype(np.int64),
            "sl_samples":         np.concatenate(sl_c,  axis=0).astype(np.float32),
            "tp_samples":         np.concatenate(tp_c,  axis=0).astype(np.float32),
            "log_probs":          combined_lp.astype(np.float32),
            "values":             np.concatenate(val_c, axis=0).astype(np.float32),
            "rewards":            np.concatenate(rew_c, axis=0).astype(np.float32),
            "dones":              np.concatenate(don_c, axis=0),
            "actual_fwd_rets":    fwd_np,
            "entered_bracket":    entered_np,
            "episode_returns":    ep_rets.tolist(),
            "pred_return_errors": np.abs(pred_np - fwd_np),
            "trades_per_day":     trades_per_day,
        }

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------

    def _compute_gae(self, rollout: dict) -> None:
        rewards = rollout["rewards"]
        values  = rollout["values"]
        dones   = rollout["dones"]
        n       = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            next_val  = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else 0.0)
            delta     = rewards[t] + self.gamma * next_val * (1.0 - float(dones[t])) - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae

        rollout["advantages"] = advantages
        rollout["returns"]    = advantages + values

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, rollout: dict) -> dict:
        """
        Full PPO loss including auxiliary prediction MSE:
          L = -L_clip + c_v * L_value - c_e * L_entropy + c_p * L_pred

        Combined log_prob for the PPO ratio:
          log_prob = log_prob_dir + entered_bracket * (log_prob_sl + log_prob_tp)

        SL and TP log_probs only contribute for bracket-entry steps, so the
        sl/tp heads only receive policy gradients when a trade is actually opened.
        """
        self.model.train()

        # ── Normalise returns BEFORE building tensors ─────────────────────
        rets_np       = rollout["returns"].astype(np.float32)
        ret_mean      = float(rets_np.mean())
        ret_std       = float(rets_np.std()) + 1e-8
        norm_rets_np  = (rets_np - ret_mean) / ret_std

        old_vals_norm_np = (
            rollout["values"].astype(np.float32) - ret_mean
        ) / ret_std

        obs           = torch.tensor(rollout["obs"],             device=self.device)
        actions       = torch.tensor(rollout["actions"],         device=self.device)
        sl_samples    = torch.tensor(rollout["sl_samples"],      device=self.device)
        tp_samples    = torch.tensor(rollout["tp_samples"],      device=self.device)
        entered       = torch.tensor(rollout["entered_bracket"], device=self.device)
        old_lp        = torch.tensor(rollout["log_probs"],       device=self.device)
        norm_returns  = torch.tensor(norm_rets_np,               device=self.device)
        old_vals_norm = torch.tensor(old_vals_norm_np,           device=self.device)
        advantages    = torch.tensor(rollout["advantages"],      device=self.device)
        actual_fwd    = torch.tensor(rollout["actual_fwd_rets"], device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(
            obs, actions, sl_samples, tp_samples, entered,
            old_lp, norm_returns, old_vals_norm, advantages, actual_fwd,
        )
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        amp_ctx = torch.amp.autocast(
            device_type="cuda" if self.device == "cuda" else "cpu",
            enabled=self._use_amp,
            dtype=self._amp_dtype,
        )

        tot_policy = tot_value = tot_entropy = tot_pred = tot_clip = 0.0
        n_batches  = 0

        for (batch_obs, batch_act, batch_sl, batch_tp, batch_entered,
             batch_old_lp, batch_norm_ret, batch_old_val_norm,
             batch_adv, batch_fwd) in loader:

            self.optimiser.zero_grad()

            with amp_ctx:
                logits, values, pred_returns, sl_mean, tp_mean = self.model(batch_obs)

                # Discrete direction distribution
                dir_dist   = Categorical(logits=logits)
                new_lp_dir = dir_dist.log_prob(batch_act)
                entropy    = dir_dist.entropy().mean()

                # Continuous SL/TP distributions (use policy's learnable std)
                sl_std = self._policy.sl_log_std.exp().clamp(1.0, 200.0)
                tp_std = self._policy.tp_log_std.exp().clamp(1.0, 400.0)
                new_lp_sl = Normal(sl_mean, sl_std).log_prob(batch_sl)
                new_lp_tp = Normal(tp_mean, tp_std).log_prob(batch_tp)

                # Direction-only ratio: SL/TP heads learn via shared backbone gradients
                new_lp = new_lp_dir

                ratio     = torch.exp(new_lp - batch_old_lp)
                clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                surr1  = ratio * batch_adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                l_clip = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (PPO standard)
                v_clipped = batch_old_val_norm + torch.clamp(
                    values - batch_old_val_norm, -self.clip_eps, self.clip_eps
                )
                l_val = 0.5 * torch.max(
                    (values    - batch_norm_ret).pow(2),
                    (v_clipped - batch_norm_ret).pow(2),
                ).mean()

                loss = (
                    l_clip
                    + self.value_loss_coef * l_val
                    - self.entropy_coef    * entropy
                )

            # Pred loss computed in float32 outside amp_ctx to prevent bfloat16 overflow
            batch_fwd_safe = batch_fwd.clamp(-0.05, 0.05).float()
            l_pred = torch.nn.functional.mse_loss(pred_returns.float(), batch_fwd_safe)
            loss = loss + self.pred_loss_coef * l_pred

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self._scaler.step(self.optimiser)
            self._scaler.update()

            tot_policy  += l_clip.item()
            tot_value   += l_val.item()
            tot_entropy += entropy.item()
            tot_pred    += l_pred.item()
            tot_clip    += clip_frac
            n_batches   += 1

        d = max(n_batches, 1)
        return {
            "policy_loss": tot_policy  / d,
            "value_loss":  tot_value   / d,
            "entropy":     tot_entropy / d,
            "pred_loss":   tot_pred    / d,
            "clip_frac":   tot_clip    / d,
        }
