"""
PPO (Proximal Policy Optimization) trainer for intraday futures trading.

Training objective
------------------
  Total loss = PPO clip loss  +  value_loss_coef × value MSE
             - entropy_coef × entropy bonus
             + pred_loss_coef × MSE(predicted_H_bar_return, actual_H_bar_return)

The last term (auxiliary prediction loss) forces the shared LSTM to learn
multi-bar price dynamics.  The LSTM hidden state must simultaneously support:
  1. A good trading policy (PPO)
  2. Accurate H-bar forward return prediction (MSE)

This teaches the network to recognise setups where a BIG move is coming and
stay flat when no meaningful move is predicted.  The entropy bonus discourages
the policy from churning on every bar; commission costs in the environment
further penalise excessive trading.

Training loop (per iteration)
------------------------------
  1. Sample ``rollout_days`` random trading-day episodes.
  2. Roll out the current policy, collecting per-step:
       obs, action, log_prob, value, reward, done, actual_fwd_return
  3. Compute GAE advantages and discounted returns.
  4. Run ``ppo_epochs`` mini-batch updates using the full loss above.
  5. Print progress every ``print_every`` iterations.

Prediction horizon
------------------
``prediction_horizon`` (default 30 bars) sets H.  For 1-min bars this means the
model learns to predict the 30-minute ahead price change.  Increasing H makes
the model focus on longer multi-bar swings; decreasing H makes it more reactive.

Multi-GPU
---------
DataParallel is used for the PPO update step.  Rollout collection is
vectorised: all rollout_days episodes are stepped in lock-step, batching
their observations into a single GPU forward pass at every bar, so both GPUs
are fed during inference as well as during the update.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
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
    entropy_coef       : float  Entropy bonus — higher → more exploration / staying flat.
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
        value_loss_coef:     float = 0.5,
        entropy_coef:        float = 0.05,    # higher → more exploration, prevents policy collapse
        pred_loss_coef:      float = 0.3,     # lower → less memorisation of specific returns
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
        )
        self._env = TradingEnv(**self._env_kwargs)   # kept for external callers

        # ── AMP — LSTM shields itself to float32, linear heads use BF16/FP16
        _is_h200 = (
            self.device == "cuda" and torch.cuda.is_available() and
            "H200" in torch.cuda.get_device_name(0)
        )
        self._use_amp   = (self.device == "cuda")
        self._amp_dtype = torch.bfloat16 if _is_h200 else torch.float16
        self._scaler    = torch.amp.GradScaler(
            device="cuda", enabled=self._use_amp
        )

        # ── H200: scale hidden_size BEFORE model construction ────────────
        # LSTM(hidden=512) forward on H200 ≈ 0.2ms < Python overhead ≈ 0.4ms
        # → GPU idles 67% of each rollout step.  4× hidden makes GPU the
        # bottleneck (≈0.8ms compute) so utilisation rises to ~65-75%.
        if _is_h200:
            hidden_size = hidden_size * 4   # 512 → 2048
            print(f"  [H200] hidden_size scaled to {hidden_size}")

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

        # ── H200 auto-scale ──────────────────────────────────────────────────
        # hidden=512 LSTM forward on H200 takes ~0.2ms but Python overhead
        # per rollout step is ~0.4ms, leaving the GPU idle 67% of each cycle.
        # Scaling hidden 4× makes GPU compute the bottleneck again.
        # rollout_days / minibatch / ppo_epochs scale fills the update pipeline.
        if _is_h200:
            self.rollout_days   = rollout_days   * 32   # 16   → 512 episodes
            self.minibatch_size = self.minibatch_size * 8   # → 8192 effective
            self.ppo_epochs     = ppo_epochs * 2            # 4    → 8 epochs
            print(
                f"  [H200] auto-scale: rollout_days={self.rollout_days}  "
                f"minibatch={self.minibatch_size}  ppo_epochs={self.ppo_epochs}"
            )

        self._policy = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

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

        # ── Train / validation split (random, by day) ─────────────────────
        # Hold out val_frac of days to detect memorisation vs generalisation.
        # Episodes are self-contained (one calendar day each) so random
        # splitting does not introduce look-ahead bias.
        rng = random.Random(42)
        eps_shuffled = list(episodes)
        rng.shuffle(eps_shuffled)
        n_val          = max(1, int(len(eps_shuffled) * self.val_frac))
        val_episodes   = eps_shuffled[:n_val]
        train_episodes = eps_shuffled[n_val:]

        print(
            f"  [PPO] {len(train_episodes)} train days  ·  {len(val_episodes)} val days  ·  "
            f"prediction horizon = {self.prediction_horizon} bars  ·  "
            f"{self.n_iterations} iterations  ·  "
            f"{self.rollout_days} days/rollout  ·  "
            f"{self.ppo_epochs} PPO epochs/iter\n"
        )

        best_val_pnl   = -float("inf")
        best_state     = None          # model weights at best val performance
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
                    f"pred_err={last_metrics['mean_pred_error_pts']:.4f}  "
                    f"policy={metrics['policy_loss']:.4f}  "
                    f"value={metrics['value_loss']:.4f}  "
                    f"pred={metrics['pred_loss']:.4f}  "
                    f"clip={metrics['clip_frac']:.3f}"
                )

                if self.patience > 0 and iters_no_improve >= self.patience:
                    print(
                        f"  [early stop] val_pnl has not improved for {self.patience} iters  "
                        f"·  best val_pnl=${best_val_pnl:+.0f}"
                    )
                    break

        # Restore the checkpoint that performed best on the validation set
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
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Group bar indices by UTC calendar date.
        Returns list of (features, prices, forward_returns) tuples.

        forward_returns[i] = (prices[i + H] - prices[i]) / prices[i]
          = the fraction of price change H bars ahead (zero-padded near EOD).
        This is the auxiliary regression target the model is trained to predict.
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

            # H-bar forward returns (regression target)
            n = len(p)
            fwd = np.zeros(n, dtype=np.float32)
            for i in range(n - H):
                if p[i] > 0:
                    fwd[i] = (p[i + H] - p[i]) / p[i]

            episodes.append((f, p, fwd))
        return episodes

    # ------------------------------------------------------------------
    # Validation evaluation (greedy policy, no gradient)
    # ------------------------------------------------------------------

    def _evaluate_val(self, val_episodes: list) -> float:
        """
        Run the current policy greedily on val_episodes.
        Returns mean daily PnL in dollars (same units as mean_episode_pnl).
        """
        self._policy.eval()
        n_ep       = len(val_episodes)
        env        = BatchedTradingEnv(n_envs=n_ep, **self._env_kwargs)
        all_states = env.reset_all(val_episodes)
        active_mask = np.ones(n_ep, dtype=bool)
        ep_rets     = np.zeros(n_ep, dtype=np.float32)

        with torch.no_grad():
            while active_mask.any():
                active  = np.where(active_mask)[0]
                x       = torch.from_numpy(all_states[active]).to(self.device)
                logits, _, _ = self._policy(x)
                actions_np   = logits.argmax(dim=-1).cpu().numpy()
                actions_all  = np.zeros(n_ep, dtype=np.int64)
                actions_all[active] = actions_np
                all_states, rews, dones = env.step_all(actions_all)
                ep_rets     += rews * active_mask.astype(np.float32)
                active_mask &= ~dones

        return float(ep_rets.mean()) * self.reward_scale

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(
        self, episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> dict:
        """
        Sample ``rollout_days`` episodes and collect trajectories.

        Uses BatchedTradingEnv so all active episodes are stepped with a
        single numpy call.  All per-step bookkeeping uses numpy advanced
        indexing into pre-allocated 2D arrays — zero Python loops in the
        hot path.  The remaining Python loops run ~max_bars (≈400) and
        ~n_ep (≈512) iterations each, vs the previous ~n_ep×max_bars
        (≈200K) iterations per Python dict-append loop.
        """
        self._policy.eval()

        sampled  = random.choices(episodes, k=self.rollout_days)
        n_ep     = len(sampled)
        max_bars = max(len(f) for f, _, _ in sampled)

        # Pre-compute forward-return matrix for vectorised lookup
        fwd_arr = np.zeros((n_ep, max_bars), dtype=np.float32)
        for i, (_, _, fwd) in enumerate(sampled):
            fwd_arr[i, :len(fwd)] = fwd

        state_dim = self._env_kwargs["n_features"] + TradingEnv.N_EXTRA

        # Pre-allocated 2D stores (n_ep × max_bars) — written with numpy
        # fancy indexing; no Python loops in the hot path.
        obs_store  = np.zeros((n_ep, max_bars, self.seq_len, state_dim), dtype=np.float32)
        act_store  = np.zeros((n_ep, max_bars),                          dtype=np.int64)
        rew_store  = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        done_store = np.zeros((n_ep, max_bars),                          dtype=bool)
        fret_store = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        lp_store   = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        val_store  = np.zeros((n_ep, max_bars),                          dtype=np.float32)
        pr_store   = np.zeros((n_ep, max_bars),                          dtype=np.float32)

        ep_t    = np.zeros(n_ep, dtype=np.int32)    # steps taken per episode
        ep_rets = np.zeros(n_ep, dtype=np.float32)

        env        = BatchedTradingEnv(n_envs=n_ep, **self._env_kwargs)
        all_states = env.reset_all(sampled)          # (n_ep, seq_len, state_dim)

        active_mask = np.ones(n_ep, dtype=bool)
        cursors     = np.full(n_ep, self.seq_len, dtype=np.int32)

        # GPU tensors accumulated; bulk-transferred after the loop (3 syncs total)
        gpu_log_probs : list[torch.Tensor] = []
        gpu_values    : list[torch.Tensor] = []
        gpu_pred_rets : list[torch.Tensor] = []
        active_hist   : list[np.ndarray]   = []   # active episode indices per step
        steps_hist    : list[np.ndarray]   = []   # ep_t snapshot per step

        with torch.no_grad():
            while active_mask.any():
                active   = np.where(active_mask)[0]   # (n_active,)
                curr     = all_states[active]          # (n_active, seq_len, state_dim)
                t_active = ep_t[active]                # (n_active,) — step index per ep

                # ── GPU forward: one batched call for all active episodes ──
                x = torch.from_numpy(curr).to(self.device)
                logits, values_t, pred_rets = self._policy(x)
                dist      = torch.distributions.Categorical(logits=logits)
                actions_t = dist.sample()
                log_prob  = dist.log_prob(actions_t)

                gpu_log_probs.append(log_prob)
                gpu_values.append(values_t)
                gpu_pred_rets.append(pred_rets)
                active_hist.append(active)
                steps_hist.append(t_active.copy())

                # ONE sync per step — only actions need to reach CPU now
                actions_np = actions_t.cpu().numpy()   # (n_active,)

                # ── Vectorised stores: numpy fancy indexing, zero loops ───
                obs_store [active, t_active] = curr
                act_store [active, t_active] = actions_np
                fret_store[active, t_active] = np.where(
                    cursors[active] < fwd_arr.shape[1],
                    fwd_arr[active, cursors[active]],
                    np.float32(0.0),
                )
                ep_t[active]    += 1
                cursors[active] += 1

                # ── Vectorised env step ──────────────────────────────────
                actions_all         = np.zeros(n_ep, dtype=np.int64)
                actions_all[active] = actions_np
                all_states, rews, dones = env.step_all(actions_all)

                rew_store [active, t_active] = rews [active]
                done_store[active, t_active] = dones[active]
                ep_rets += rews * active_mask.astype(np.float32)

                active_mask &= ~dones

        # ── Bulk GPU→CPU: 3 syncs total for the entire rollout ──────────
        all_lp = torch.cat(gpu_log_probs).cpu().numpy()
        all_v  = torch.cat(gpu_values).cpu().numpy()
        all_pr = torch.cat(gpu_pred_rets).cpu().numpy()

        # Scatter log_probs / values / pred_rets into 2D stores.
        # ~max_bars loop iters (≈400) — not n_ep×max_bars (≈200K).
        offset = 0
        for act_step, t_step in zip(active_hist, steps_hist):
            n = len(act_step)
            lp_store [act_step, t_step] = all_lp[offset:offset + n]
            val_store[act_step, t_step] = all_v [offset:offset + n]
            pr_store [act_step, t_step] = all_pr[offset:offset + n]
            offset += n

        # ── Flatten in episode order for correct GAE boundaries ─────────
        # ~n_ep loop iters (≈512) — not n_ep×T (≈200K).
        obs_c = []; act_c = []; lp_c  = []; val_c = []
        rew_c = []; don_c = []; frt_c = []; prd_c = []
        for i in range(n_ep):
            T = int(ep_t[i])
            if T == 0:
                continue
            obs_c.append(obs_store [i, :T])
            act_c.append(act_store [i, :T])
            lp_c .append(lp_store  [i, :T])
            val_c.append(val_store [i, :T])
            rew_c.append(rew_store [i, :T])
            don_c.append(done_store[i, :T])
            frt_c.append(fret_store[i, :T])
            prd_c.append(pr_store  [i, :T])

        fwd_np  = np.concatenate(frt_c, axis=0)
        pred_np = np.concatenate(prd_c, axis=0)

        return {
            "obs":                np.concatenate(obs_c, axis=0).astype(np.float32),
            "actions":            np.concatenate(act_c, axis=0).astype(np.int64),
            "log_probs":          np.concatenate(lp_c,  axis=0).astype(np.float32),
            "values":             np.concatenate(val_c, axis=0).astype(np.float32),
            "rewards":            np.concatenate(rew_c, axis=0).astype(np.float32),
            "dones":              np.concatenate(don_c, axis=0),
            "actual_fwd_rets":    fwd_np,
            "episode_returns":    ep_rets.tolist(),
            "pred_return_errors": np.abs(pred_np - fwd_np),
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
        """
        self.model.train()

        obs         = torch.tensor(rollout["obs"],           device=self.device)
        actions     = torch.tensor(rollout["actions"],       device=self.device)
        old_lp      = torch.tensor(rollout["log_probs"],     device=self.device)
        returns     = torch.tensor(rollout["returns"],       device=self.device)
        advantages  = torch.tensor(rollout["advantages"],    device=self.device)
        actual_fwd  = torch.tensor(rollout["actual_fwd_rets"], device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(obs, actions, old_lp, returns, advantages, actual_fwd)
        loader  = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        amp_ctx = torch.amp.autocast(
            device_type="cuda" if self.device == "cuda" else "cpu",
            enabled=self._use_amp,
            dtype=self._amp_dtype,
        )

        tot_policy = tot_value = tot_entropy = tot_pred = tot_clip = 0.0
        n_batches  = 0

        for batch_obs, batch_act, batch_old_lp, batch_ret, batch_adv, batch_fwd in loader:
            self.optimiser.zero_grad()

            with amp_ctx:
                # self.model is either the base model or nn.DataParallel —
                # forward() now returns all three heads so DataParallel can
                # scatter/gather across both GPUs correctly.
                logits, values, pred_returns = self.model(batch_obs)

                dist      = torch.distributions.Categorical(logits=logits)
                new_lp    = dist.log_prob(batch_act)
                entropy   = dist.entropy().mean()

                ratio     = torch.exp(new_lp - batch_old_lp)
                clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                surr1  = ratio * batch_adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                l_clip = -torch.min(surr1, surr2).mean()
                l_val  = 0.5 * (values - batch_ret).pow(2).mean()
                l_pred = torch.nn.functional.mse_loss(pred_returns, batch_fwd)

                loss = (
                    l_clip
                    + self.value_loss_coef * l_val
                    - self.entropy_coef    * entropy
                    + self.pred_loss_coef  * l_pred
                )

            # GradScaler is a no-op when AMP is disabled (enabled=False)
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
