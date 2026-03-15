"""
PPO (Proximal Policy Optimization) trainer for intraday futures trading.

Training loop
-------------
Each PPO iteration:
  1. Sample ``rollout_days`` random trading-day episodes from the training set.
  2. Roll out the current policy through every selected episode, collecting:
       (obs, action, log_prob, value, reward, done)
  3. Compute GAE advantages and discounted returns.
  4. Run ``ppo_epochs`` mini-batch update passes over the collected data.
  5. Print progress every ``print_every`` iterations.

Episode boundaries
------------------
Bars are grouped by UTC calendar date.  Each date group becomes one episode
so the environment naturally enforces the intraday constraint.

Multi-GPU
---------
The PPO update step uses DataParallel (same as the supervised Trainer) so
both H200s contribute to gradient computation.  Rollout collection runs on
the primary GPU because it is sequential and GPU-bound only for the short
LSTM inference calls.

Reward
------
  (MTM P&L change − commission) / reward_scale
  All dollar metrics printed during training are re-scaled back to dollars.
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
    n_features      : int    Raw OHLCV feature count (augmented to +2 internally).
    hidden_size     : int    LSTM hidden units.
    num_layers      : int
    dropout         : float
    seq_len         : int    Sequence window length.
    lr              : float  Adam learning rate.
    n_iterations    : int    Total PPO training iterations.
    rollout_days    : int    Episodes to sample per iteration.
    ppo_epochs      : int    Update passes over collected rollout per iteration.
    minibatch_size  : int    Samples per mini-batch in the update step.
    clip_eps        : float  PPO clipping epsilon.
    gamma           : float  Discount factor.
    gae_lambda      : float  GAE λ.
    value_loss_coef : float  Weight on value loss.
    entropy_coef    : float  Weight on entropy bonus (encourages exploration).
    max_grad_norm   : float  Gradient clipping norm.
    print_every     : int    Print progress every N iterations.
    device          : str    Auto-detected if None.
    multiplier      : float  Contract point value.
    commission      : float  $ per side per contract.
    contracts       : float  Position size.
    max_loss        : float  Per-trade stop-loss in dollars.
    reward_scale    : float  Reward normalisation divisor.
    """

    def __init__(
        self,
        n_features:      int   = 22,
        hidden_size:     int   = 512,
        num_layers:      int   = 2,
        dropout:         float = 0.3,
        seq_len:         int   = 30,
        lr:              float = 3e-4,
        n_iterations:    int   = 200,
        rollout_days:    int   = 16,
        ppo_epochs:      int   = 4,
        minibatch_size:  int   = 512,
        clip_eps:        float = 0.2,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef:    float = 0.01,
        max_grad_norm:   float = 0.5,
        print_every:     int   = 10,
        device:          Optional[str] = None,
        # TradingEnv params (must match LSTMSignalStrategy / backtest config)
        multiplier:      float = 20.0,
        commission:      float = 2.0,
        contracts:       float = 1.0,
        max_loss:        float = 2_500.0,
        reward_scale:    float = 100.0,
    ) -> None:
        self.seq_len         = seq_len
        self.n_iterations    = n_iterations
        self.rollout_days    = rollout_days
        self.ppo_epochs      = ppo_epochs
        self.minibatch_size  = minibatch_size
        self.clip_eps        = clip_eps
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm
        self.print_every     = print_every
        self.reward_scale    = reward_scale
        self.device          = device or _auto_device()

        # Environment (one instance reused across episodes)
        self._env = TradingEnv(
            seq_len=seq_len,
            n_features=n_features,
            multiplier=multiplier,
            commission=commission,
            contracts=contracts,
            max_loss=max_loss,
            reward_scale=reward_scale,
        )

        # Build model
        base_model = ActorCriticLSTM(
            n_features=n_features + TradingEnv.N_EXTRA,  # 22 + 2 = 24
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Multi-GPU DataParallel
        n_gpus = torch.cuda.device_count() if self.device == "cuda" else 1
        if n_gpus > 1:
            self.model = nn.DataParallel(base_model)
            print(
                f"  [multi-gpu] DataParallel across {n_gpus} GPUs  "
                f"(effective mini-batch = {minibatch_size}  "
                f"·  {minibatch_size // n_gpus} per GPU)"
            )
        else:
            self.model = base_model

        # Rollout inference always uses the bare (un-wrapped) model for
        # step_hidden() — DataParallel doesn't expose custom methods.
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
        Returns  : dict with final training metrics
        """
        episodes = self._make_episodes(bars, features)
        if not episodes:
            raise ValueError("No valid training episodes found (all days too short).")
        print(
            f"  [PPO] {len(episodes)} training days  ·  "
            f"{self.n_iterations} iterations  ·  "
            f"{self.rollout_days} days/rollout  ·  "
            f"{self.ppo_epochs} PPO epochs/iter\n"
        )

        last_metrics: dict = {}
        for iteration in range(1, self.n_iterations + 1):
            rollout       = self._collect_rollout(episodes)
            self._compute_gae(rollout)
            metrics       = self._ppo_update(rollout)
            last_metrics  = metrics
            last_metrics["mean_episode_pnl"] = (
                float(np.mean(rollout["episode_returns"])) * self.reward_scale
            )

            if iteration % self.print_every == 0 or iteration == 1:
                print(
                    f"    iter {iteration:>4}  "
                    f"mean_daily_pnl=${last_metrics['mean_episode_pnl']:+.0f}  "
                    f"policy_loss={metrics['policy_loss']:.4f}  "
                    f"value_loss={metrics['value_loss']:.4f}  "
                    f"entropy={metrics['entropy']:.4f}  "
                    f"clip_frac={metrics['clip_frac']:.3f}"
                )

        return last_metrics

    def save(self, path) -> None:
        """Save the underlying model weights (unwrap DataParallel)."""
        self._policy.save(path)

    # ------------------------------------------------------------------
    # Episode construction
    # ------------------------------------------------------------------

    def _make_episodes(
        self, bars: List[Bar], features: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Group bar indices by UTC calendar date, return (features, prices) pairs.
        Episodes shorter than seq_len + 1 are skipped.
        """
        day_map: dict = defaultdict(list)
        for i, bar in enumerate(bars):
            day_map[bar.timestamp.date()].append(i)

        episodes = []
        for date in sorted(day_map.keys()):
            idx = day_map[date]
            if len(idx) <= self.seq_len:
                continue
            f = features[idx].astype(np.float32)
            p = np.array([bars[i].close for i in idx], dtype=np.float32)
            episodes.append((f, p))
        return episodes

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(
        self, episodes: list[tuple[np.ndarray, np.ndarray]]
    ) -> dict:
        """
        Sample ``rollout_days`` episodes and collect one full trajectory
        through each under the current policy.

        Returns a dict of lists (one entry per environment step):
          obs, actions, log_probs, values, rewards, dones, episode_returns
        """
        self._policy.eval()

        obs_buf        = []
        action_buf     = []
        log_prob_buf   = []
        value_buf      = []
        reward_buf     = []
        done_buf       = []
        episode_returns = []

        sampled = random.choices(episodes, k=self.rollout_days)

        for feat, prices in sampled:
            state = self._env.reset(feat, prices)
            ep_return = 0.0
            done = False

            while not done:
                x = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                action, log_prob, value, _ = self._policy.act(x)

                next_state, reward, done = self._env.step(int(action.item()))
                ep_return += reward

                obs_buf.append(state)
                action_buf.append(int(action.item()))
                log_prob_buf.append(float(log_prob.item()))
                value_buf.append(float(value.item()))
                reward_buf.append(reward)
                done_buf.append(done)

                state = next_state

            episode_returns.append(ep_return)

        return {
            "obs":             np.array(obs_buf,      dtype=np.float32),
            "actions":         np.array(action_buf,   dtype=np.int64),
            "log_probs":       np.array(log_prob_buf, dtype=np.float32),
            "values":          np.array(value_buf,    dtype=np.float32),
            "rewards":         np.array(reward_buf,   dtype=np.float32),
            "dones":           np.array(done_buf,     dtype=bool),
            "episode_returns": episode_returns,
        }

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------

    def _compute_gae(self, rollout: dict) -> None:
        """
        Compute GAE advantages and TD-λ returns in-place, adding keys
        ``advantages`` and ``returns`` to the rollout dict.
        """
        rewards = rollout["rewards"]
        values  = rollout["values"]
        dones   = rollout["dones"]
        n       = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            next_value    = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else 0.0)
            delta         = rewards[t] + self.gamma * next_value * (1.0 - float(dones[t])) - values[t]
            last_gae      = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae

        rollout["advantages"] = advantages
        rollout["returns"]    = advantages + values

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, rollout: dict) -> dict:
        """Run ``ppo_epochs`` passes of PPO mini-batch updates."""
        self.model.train()

        obs         = torch.tensor(rollout["obs"],         device=self.device)
        actions     = torch.tensor(rollout["actions"],     device=self.device)
        old_lp      = torch.tensor(rollout["log_probs"],   device=self.device)
        returns     = torch.tensor(rollout["returns"],     device=self.device)
        advantages  = torch.tensor(rollout["advantages"],  device=self.device)

        # Normalize advantages per mini-batch pass
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(obs, actions, old_lp, returns, advantages)
        loader  = DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
        )

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_clip_frac   = 0.0
        n_batches         = 0

        for _ in range(self.ppo_epochs):
            for batch_obs, batch_act, batch_old_lp, batch_ret, batch_adv in loader:
                logits, values = self.model(batch_obs)
                dist           = torch.distributions.Categorical(logits=logits)
                new_lp         = dist.log_prob(batch_act)
                entropy        = dist.entropy().mean()

                ratio     = torch.exp(new_lp - batch_old_lp)
                clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (values - batch_ret).pow(2).mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef    * entropy
                )

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimiser.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                total_clip_frac   += clip_frac
                n_batches         += 1

        denom = max(n_batches, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss":  total_value_loss  / denom,
            "entropy":     total_entropy     / denom,
            "clip_frac":   total_clip_frac   / denom,
        }
