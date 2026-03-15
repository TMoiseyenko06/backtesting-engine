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
DataParallel is used for the PPO update step.  Rollout collection runs on the
primary GPU (sequential, policy inference per bar).
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

        self._env = TradingEnv(
            seq_len=seq_len,
            n_features=n_features,
            multiplier=multiplier,
            commission=commission,
            contracts=contracts,
            max_loss=max_loss,
            reward_scale=reward_scale,
        )

        base_model = ActorCriticLSTM(
            n_features=n_features + TradingEnv.N_EXTRA,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

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
        Sample ``rollout_days`` episodes and collect one full trajectory each.
        Records actual forward returns alongside experience for the aux loss.
        """
        self._policy.eval()

        obs_buf             = []
        action_buf          = []
        log_prob_buf        = []
        value_buf           = []
        reward_buf          = []
        done_buf            = []
        actual_fwd_ret_buf  = []   # H-bar ahead ground truth
        pred_fwd_ret_buf    = []   # model's predicted forward return (for error tracking)
        episode_returns     = []

        sampled = random.choices(episodes, k=self.rollout_days)

        for feat, prices, fwd_returns in sampled:
            state   = self._env.reset(feat, prices)
            ep_ret  = 0.0
            done    = False
            cursor  = self._env.seq_len   # first actionable bar index

            while not done:
                x = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                action, log_prob, value, _, pred_ret = self._policy.act(x)

                next_state, reward, done = self._env.step(int(action.item()))
                ep_ret += reward

                obs_buf.append(state)
                action_buf.append(int(action.item()))
                log_prob_buf.append(float(log_prob.item()))
                value_buf.append(float(value.item()))
                reward_buf.append(reward)
                done_buf.append(done)
                # Ground-truth target for prediction head
                actual_fwd_ret_buf.append(float(fwd_returns[cursor]) if cursor < len(fwd_returns) else 0.0)
                pred_fwd_ret_buf.append(float(pred_ret.item()))

                cursor += 1
                state   = next_state

            episode_returns.append(ep_ret)

        pred_errors = np.abs(
            np.array(pred_fwd_ret_buf) - np.array(actual_fwd_ret_buf)
        )

        return {
            "obs":               np.array(obs_buf,            dtype=np.float32),
            "actions":           np.array(action_buf,         dtype=np.int64),
            "log_probs":         np.array(log_prob_buf,       dtype=np.float32),
            "values":            np.array(value_buf,          dtype=np.float32),
            "rewards":           np.array(reward_buf,         dtype=np.float32),
            "dones":             np.array(done_buf,           dtype=bool),
            "actual_fwd_rets":   np.array(actual_fwd_ret_buf, dtype=np.float32),
            "episode_returns":   episode_returns,
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

        tot_policy = tot_value = tot_entropy = tot_pred = tot_clip = 0.0
        n_batches  = 0

        for batch_obs, batch_act, batch_old_lp, batch_ret, batch_adv, batch_fwd in loader:
            logits, values, pred_returns = self.model.forward_full(batch_obs) \
                if not isinstance(self.model, nn.DataParallel) \
                else self.model.module.forward_full(batch_obs)

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

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimiser.step()

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
