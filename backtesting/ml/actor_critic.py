"""
Actor-Critic LSTM for PPO-based futures trading.

Architecture
------------
  Input  : (batch, seq_len, n_features + 2)
             n_features = OHLCV features (from features.py)
             +1          = position encoding  {-1 short, 0 flat, +1 long}
             +1          = normalised unrealised P&L

  Shared LSTM → LayerNorm on last hidden state

  Three output heads:
    policy_head     : Dropout → Linear(hidden, 3)  — logits over {FLAT, LONG, SHORT}
    value_head      : Dropout → Linear(hidden, 1)  — scalar V(s) for PPO critic
    prediction_head : Dropout → Linear(hidden, 1)  — predicted H-bar forward return

  The prediction head is trained with an MSE auxiliary loss (actual H-bar forward
  return as target).  This forces the shared LSTM to encode multi-bar momentum /
  trend information, so the policy learns to distinguish large moves from noise.

Action convention (matches TradingEnv)
--------------------------------------
  0  FLAT   — close any open position
  1  LONG   — go / stay long
  2  SHORT  — go / stay short

predict_rl() returns (desired_position, confidence, predicted_return_pct):
  desired_position : -1 (short), 0 (flat), +1 (long)
  confidence       : max softmax probability
  predicted_return_pct : predicted H-bar price change / current price
                         (positive = up, negative = down)
  The strategy uses predicted_return_pct × close × multiplier to estimate the
  expected dollar move, then only enters trades above a minimum threshold.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from backtesting.ml.features import N_FEATURES
from backtesting.ml.rl_env import TradingEnv

# Total input features (raw OHLCV features + position context)
N_INPUT = N_FEATURES + TradingEnv.N_EXTRA   # 22 + 2 = 24


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
        # Tanh bounds output to [-1, 1]; without it the unbounded linear head
        # produces O(1) predictions against O(0.001) fractional-return targets,
        # giving MSE in the tens of thousands that drowns the policy gradient.
        self.prediction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

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
    # Batch forward — all three heads
    # Called by DataParallel during the PPO update step, so it must
    # return all outputs needed by the loss (logits, values, pred_returns).
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (batch, seq_len, n_features)
        Returns:
          logits        (batch, n_actions)
          values        (batch,)
          pred_returns  (batch,)   — predicted H-bar forward return
        """
        h            = self._encode(x)
        logits       = self.policy_head(h)
        values       = self.value_head(h).squeeze(-1)
        pred_returns = self.prediction_head(h).squeeze(-1)
        return logits, values, pred_returns

    # Alias kept for external callers (RLTradingStrategy, act, predict_rl).
    forward_full = forward

    # ------------------------------------------------------------------
    # Stochastic action sampling — rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action stochastically (exploration during training).

        x : (1, seq_len, n_features)
        Returns: (action, log_prob, value, entropy, pred_return)  — scalar tensors
        """
        logits, values, pred_returns = self.forward_full(x)
        dist    = Categorical(logits=logits)
        action  = dist.sample()
        return action, dist.log_prob(action), values, dist.entropy(), pred_returns

    # ------------------------------------------------------------------
    # Greedy inference — backtesting via RLTradingStrategy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_rl(
        self, x: torch.Tensor
    ) -> tuple[int, float, float]:
        """
        Greedy (argmax) action for deployment/backtesting.

        x : (1, seq_len, n_features+2) — state-augmented input
        Returns:
          desired_position  : -1 (short), 0 (flat), +1 (long)
          confidence        : max softmax probability
          predicted_return  : predicted H-bar forward return as a fraction of price
                              (e.g. 0.005 = model expects +0.5% price move ahead)
        """
        logits, _, pred_returns = self.forward_full(x)
        probs      = torch.softmax(logits, dim=-1)
        ac_action  = int(probs.argmax(dim=-1).item())
        conf       = float(probs[0, ac_action].item())
        pred_ret   = float(pred_returns[0].item())

        # TradingEnv: FLAT=0, LONG=1, SHORT=2  →  position {0, +1, -1}
        _AC_TO_POS = {0: 0, 1: 1, 2: -1}
        return _AC_TO_POS[ac_action], conf, pred_ret

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
