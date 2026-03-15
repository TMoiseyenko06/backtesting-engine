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
from backtesting.ml.rl_env import TradingEnv
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
        entropy_coef:        float = 0.02,    # slightly higher → stay flat more
        pred_loss_coef:      float = 1.0,     # aux prediction loss weight
        prediction_horizon:  int   = 30,      # H-bar ahead prediction target
        max_grad_norm:       float = 0.5,
        print_every:         int   = 10,
        device:              Optional[str] = None,
        multiplier:          float = 20.0,
        commission:          float = 2.0,
        contracts:           float = 1.0,
        max_loss:            float = 2_500.0,
        reward_scale:        float = 100.0,
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

        # ── H200 auto-scale: fill the GPU with larger rollout and minibatches.
        # rollout_days controls the batch size during rollout inference (all
        # active episodes are stacked into one forward pass per bar).  16 episodes
        # → batch-16 calls; 512 episodes → batch-512 calls — 32× more GPU work.
        if _is_h200:
            self.rollout_days   = rollout_days   * 32   # 16  → 512 episodes
            self.minibatch_size = self.minibatch_size * 8   # → 8192 effective
            self.ppo_epochs     = ppo_epochs * 2            # 4   → 8 epochs
            print(
                f"  [H200] auto-scale: rollout_days={self.rollout_days}  "
                f"minibatch={self.minibatch_size}  ppo_epochs={self.ppo_epochs}"
            )

        self._policy = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        print(
            f"  [PPO] {len(episodes)} training days  ·  "
            f"prediction horizon = {self.prediction_horizon} bars  ·  "
            f"{self.n_iterations} iterations  ·  "
            f"{self.rollout_days} days/rollout  ·  "
            f"{self.ppo_epochs} PPO epochs/iter\n"
        )

        last_metrics: dict = {}
        for iteration in range(1, self.n_iterations + 1):
            rollout      = self._collect_rollout(episodes)
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
                print(
                    f"    iter {iteration:>4}  "
                    f"mean_daily_pnl=${last_metrics['mean_episode_pnl']:+.0f}  "
                    f"pred_err={last_metrics['mean_pred_error_pts']:.4f}  "
                    f"policy={metrics['policy_loss']:.4f}  "
                    f"value={metrics['value_loss']:.4f}  "
                    f"pred={metrics['pred_loss']:.4f}  "
                    f"clip={metrics['clip_frac']:.3f}"
                )

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
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(
        self, episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> dict:
        """
        Sample ``rollout_days`` episodes and collect trajectories.

        All active episodes are stepped in lock-step so their observations
        can be batched into a single GPU forward pass at every bar — this
        keeps both H200s busy during inference instead of issuing one
        batch-1 call per bar per episode.
        """
        self._policy.eval()

        sampled  = random.choices(episodes, k=self.rollout_days)
        n_ep     = len(sampled)

        # One independent env per episode
        envs     = [TradingEnv(**self._env_kwargs) for _ in range(n_ep)]
        states   = [env.reset(feat, prices)
                    for env, (feat, prices, _) in zip(envs, sampled)]
        cursors  = [self.seq_len] * n_ep   # index into fwd_returns
        ep_rets  = [0.0] * n_ep
        active   = list(range(n_ep))       # episode indices still running

        # Per-episode experience buffers
        ep_bufs  = [{"obs": [], "actions": [], "log_probs": [],
                     "values": [], "rewards": [], "dones": [],
                     "actual_fwd_rets": [], "pred_fwd_rets": []}
                    for _ in range(n_ep)]

        # GPU tensors accumulated across rollout steps; moved to CPU
        # in a single bulk transfer after the loop finishes.
        # This replaces O(steps × episodes) .item() syncs with O(steps)
        # syncs for actions (needed to call env.step) + 3 syncs total for
        # the scalar outputs at the end.
        gpu_log_probs: list[torch.Tensor] = []
        gpu_values:    list[torch.Tensor] = []
        gpu_pred_rets: list[torch.Tensor] = []
        active_history: list[list[int]]  = []

        with torch.no_grad():
            while active:
                # ── Batch all active states into one GPU call ──────────
                x = torch.tensor(
                    np.stack([states[i] for i in active]),
                    dtype=torch.float32,
                ).to(self.device)           # (n_active, seq_len, n_feat)

                logits, values_t, pred_rets = self._policy(x)
                dist     = torch.distributions.Categorical(logits=logits)
                actions  = dist.sample()
                log_prob = dist.log_prob(actions)

                # Store GPU tensors — no .item() yet
                gpu_log_probs.append(log_prob)
                gpu_values.append(values_t)
                gpu_pred_rets.append(pred_rets)
                active_history.append(list(active))

                # ONE sync per step (not per episode) to feed env.step
                actions_np = actions.cpu().numpy()

                next_active = []
                for j, i in enumerate(active):
                    _, _, fwd_returns = sampled[i]
                    action = int(actions_np[j])

                    next_state, reward, done = envs[i].step(action)
                    ep_rets[i] += reward

                    buf = ep_bufs[i]
                    buf["obs"].append(states[i])
                    buf["actions"].append(action)
                    buf["rewards"].append(reward)
                    buf["dones"].append(done)
                    c = cursors[i]
                    buf["actual_fwd_rets"].append(
                        float(fwd_returns[c]) if c < len(fwd_returns) else 0.0
                    )

                    cursors[i] += 1
                    states[i]   = next_state
                    if not done:
                        next_active.append(i)

                active = next_active

        # ── Bulk GPU→CPU transfer (3 syncs total for entire rollout) ──
        all_lp  = torch.cat(gpu_log_probs).cpu().numpy()
        all_v   = torch.cat(gpu_values).cpu().numpy()
        all_pr  = torch.cat(gpu_pred_rets).cpu().numpy()

        offset = 0
        for active_at_step in active_history:
            n = len(active_at_step)
            for j, i in enumerate(active_at_step):
                ep_bufs[i]["log_probs"].append(float(all_lp[offset + j]))
                ep_bufs[i]["values"].append(float(all_v[offset + j]))
                ep_bufs[i]["pred_fwd_rets"].append(float(all_pr[offset + j]))
            offset += n

        # ── Merge all episode buffers ──────────────────────────────────
        obs_buf            = []
        action_buf         = []
        log_prob_buf       = []
        value_buf          = []
        reward_buf         = []
        done_buf           = []
        actual_fwd_ret_buf = []
        pred_fwd_ret_buf   = []

        for buf in ep_bufs:
            obs_buf.extend(buf["obs"])
            action_buf.extend(buf["actions"])
            log_prob_buf.extend(buf["log_probs"])
            value_buf.extend(buf["values"])
            reward_buf.extend(buf["rewards"])
            done_buf.extend(buf["dones"])
            actual_fwd_ret_buf.extend(buf["actual_fwd_rets"])
            pred_fwd_ret_buf.extend(buf["pred_fwd_rets"])

        pred_errors = np.abs(
            np.array(pred_fwd_ret_buf) - np.array(actual_fwd_ret_buf)
        )

        return {
            "obs":                np.array(obs_buf,            dtype=np.float32),
            "actions":            np.array(action_buf,         dtype=np.int64),
            "log_probs":          np.array(log_prob_buf,       dtype=np.float32),
            "values":             np.array(value_buf,          dtype=np.float32),
            "rewards":            np.array(reward_buf,         dtype=np.float32),
            "dones":              np.array(done_buf,           dtype=bool),
            "actual_fwd_rets":    np.array(actual_fwd_ret_buf, dtype=np.float32),
            "episode_returns":    ep_rets,
            "pred_return_errors": pred_errors,
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
