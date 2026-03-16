"""
Actor-Critic LSTM for PPO-based futures trading.

Architecture
------------
  Input  : (batch, seq_len, n_features + 2)
             n_features = OHLCV features (from features.py)
             +1          = position encoding  {-1 short, 0 flat, +1 long}
             +1          = normalised unrealised P&L

  Shared LSTM → LayerNorm on last hidden state

  Five output heads:
    policy_head     : Dropout → Linear(hidden, 3)  — logits over {FLAT, LONG, SHORT}
    value_head      : Dropout → Linear(hidden, 1)  — scalar V(s) for PPO critic
    prediction_head : Dropout → Linear(hidden, 1)  — predicted H-bar forward return
    sl_head         : Dropout → Linear(hidden, 1)  — stop-loss distance in NQ points
    tp_head         : Dropout → Linear(hidden, 1)  — take-profit distance in NQ points

  sl_head and tp_head are sigmoid-bounded to their valid ranges:
    SL: [SL_MIN_PTS, SL_MAX_PTS] = [25, 500] pts
    TP: [TP_MIN_PTS, TP_MAX_PTS] = [25, 1500] pts

  Each has a learnable log-std (sl_log_std, tp_log_std) for stochastic sampling
  during training.  During inference (predict_rl) the mean is used directly.

Action convention (matches TradingEnv)
--------------------------------------
  0  FLAT   — stay flat, do not enter
  1  LONG   — go long with predicted SL/TP bracket
  2  SHORT  — go short with predicted SL/TP bracket

predict_rl() returns (desired_position, confidence, sl_pts, tp_pts):
  desired_position : -1 (short), 0 (flat), +1 (long)
  confidence       : max softmax probability
  sl_pts           : predicted stop-loss distance in NQ points
  tp_pts           : predicted take-profit distance in NQ points
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from backtesting.ml.features import N_FEATURES
from backtesting.ml.rl_env import TradingEnv

# Total input features (raw OHLCV features + position context)
N_INPUT = N_FEATURES + TradingEnv.N_EXTRA   # 22 + 2 = 24

# Bracket order bounds (NQ points)
SL_MIN_PTS =   25.0
SL_MAX_PTS =  500.0
TP_MIN_PTS =   25.0
TP_MAX_PTS = 1500.0


class ActorCriticLSTM(nn.Module):
    """
    Parameters
    ----------
    n_features  : int   Features per timestep (default N_INPUT = 24).
    hidden_size : int   LSTM hidden units per layer.
    num_layers  : int   Stacked LSTM layers.
    dropout     : float Applied between LSTM layers and before heads.
    n_actions   : int   Discrete action count (3).
    """

    def __init__(
        self,
        n_features:  int   = N_INPUT,
        hidden_size: int   = 512,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        n_actions:   int   = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.n_actions   = n_actions

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.policy_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        # Predicts the H-bar forward return at each step (auxiliary task).
        # Trained with MSE loss so the LSTM learns multi-bar price dynamics.
        self.prediction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        # SL and TP distance heads — sigmoid-scaled to valid NQ point ranges
        self.sl_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.tp_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        # Learnable log-std for stochastic bracket sampling during training
        self.sl_log_std = nn.Parameter(torch.zeros(1))
        self.tp_log_std = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Hidden state extraction (shared, called by all heads)
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run LSTM + LayerNorm, return last hidden state (batch, hidden)."""
        # Disable autocast for the LSTM: even with float32 input the LSTM
        # weight matmuls are downcast to BF16 by the outer autocast context,
        # causing NaN in hidden states.  autocast(enabled=False) forces all
        # internal matmuls to stay in float32.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out, _ = self.lstm(x.float())
        return self.norm(out[:, -1, :])

    # ------------------------------------------------------------------
    # Batch forward — all five heads
    # Called by DataParallel during the PPO update step, so it must
    # return all outputs needed by the loss (logits, values, pred_returns,
    # sl_mean, tp_mean).
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (batch, seq_len, n_features)
        Returns:
          logits        (batch, n_actions)
          values        (batch,)
          pred_returns  (batch,)   — predicted H-bar forward return (tanh-bounded)
          sl_mean       (batch,)   — predicted SL distance in NQ pts [SL_MIN, SL_MAX]
          tp_mean       (batch,)   — predicted TP distance in NQ pts [TP_MIN, TP_MAX]
        """
        h            = self._encode(x)
        logits       = self.policy_head(h)
        values       = self.value_head(h).squeeze(-1)
        pred_returns = self.prediction_head(h).squeeze(-1)
        sl_mean = SL_MIN_PTS + (SL_MAX_PTS - SL_MIN_PTS) * torch.sigmoid(
            self.sl_head(h).squeeze(-1)
        )
        tp_mean = TP_MIN_PTS + (TP_MAX_PTS - TP_MIN_PTS) * torch.sigmoid(
            self.tp_head(h).squeeze(-1)
        )
        return logits, values, pred_returns, sl_mean, tp_mean

    # Alias kept for external callers (RLTradingStrategy, act, predict_rl).
    forward_full = forward

    # ------------------------------------------------------------------
    # Stochastic action sampling — rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Sample action stochastically (exploration during training).

        x : (batch, seq_len, n_features)
        Returns:
          action       — sampled discrete action {0,1,2}
          log_prob_dir — log probability of sampled direction
          log_prob_sl  — log probability of sampled sl_pts
          log_prob_tp  — log probability of sampled tp_pts
          sl_sample    — sampled SL distance in NQ pts (clamped to valid range)
          tp_sample    — sampled TP distance in NQ pts (clamped to valid range)
          values       — critic value estimate
          entropy      — policy entropy
          pred_returns — predicted H-bar forward return
        """
        logits, values, pred_returns, sl_mean, tp_mean = self.forward_full(x)

        dist         = Categorical(logits=logits)
        action       = dist.sample()
        log_prob_dir = dist.log_prob(action)
        entropy      = dist.entropy()

        sl_std = self.sl_log_std.exp().clamp(1.0, 200.0)
        tp_std = self.tp_log_std.exp().clamp(1.0, 400.0)

        sl_dist   = Normal(sl_mean, sl_std)
        tp_dist   = Normal(tp_mean, tp_std)
        sl_sample = sl_dist.sample().clamp(SL_MIN_PTS, SL_MAX_PTS)
        tp_sample = tp_dist.sample().clamp(TP_MIN_PTS, TP_MAX_PTS)

        log_prob_sl = sl_dist.log_prob(sl_sample)
        log_prob_tp = tp_dist.log_prob(tp_sample)

        return (
            action, log_prob_dir, log_prob_sl, log_prob_tp,
            sl_sample, tp_sample, values, entropy, pred_returns,
        )

    # ------------------------------------------------------------------
    # Greedy inference — backtesting via RLTradingStrategy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_rl(
        self, x: torch.Tensor
    ) -> tuple[int, float, float, float]:
        """
        Greedy (argmax) action for deployment/backtesting.

        x : (1, seq_len, n_features+2) — state-augmented input
        Returns:
          desired_position : -1 (short), 0 (flat), +1 (long)
          confidence       : max softmax probability
          sl_pts           : predicted stop-loss distance in NQ points
          tp_pts           : predicted take-profit distance in NQ points
        """
        logits, _, _, sl_mean, tp_mean = self.forward_full(x)
        probs     = torch.softmax(logits, dim=-1)
        ac_action = int(probs.argmax(dim=-1).item())
        conf      = float(probs[0, ac_action].item())
        sl_pts    = float(sl_mean[0].item())
        tp_pts    = float(tp_mean[0].item())

        # TradingEnv: FLAT=0, LONG=1, SHORT=2  →  position {0, +1, -1}
        _AC_TO_POS = {0: 0, 1: 1, 2: -1}
        return _AC_TO_POS[ac_action], conf, sl_pts, tp_pts

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "ActorCriticLSTM":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(path, map_location=device, weights_only=True)
        ih          = state["lstm.weight_ih_l0"]
        n_features  = ih.shape[1]
        hidden_size = ih.shape[0] // 4
        num_layers  = sum(1 for k in state if k.startswith("lstm.weight_ih_l"))
        n_actions   = state["policy_head.1.weight"].shape[0]
        model = cls(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_actions=n_actions,
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
