"""
Actor-Critic LSTM for PPO-based futures trading.

Architecture
------------
  Input  : (batch, seq_len, n_features + 2)
             n_features = OHLCV features (from features.py)
             +1          = position encoding  {-1 short, 0 flat, +1 long}
             +1          = normalised unrealised P&L

  Shared LSTM → LayerNorm on last hidden state
  Policy head : Dropout → Linear(hidden, 3)   logits over {FLAT, LONG, SHORT}
  Value  head : Dropout → Linear(hidden, 1)   scalar V(s)

Action convention (matches TradingEnv)
--------------------------------------
  0  FLAT   — close any open position
  1  LONG   — go / stay long
  2  SHORT  — go / stay short

Serialisation
-------------
  save(path) / load(path, device)  — saved as plain state_dict; architecture
  dims are inferred from weight shapes on load, same as LSTMModel.
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

    # ------------------------------------------------------------------
    # Batch forward — used during PPO update
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, seq_len, n_features)
        Returns
        -------
        logits : (batch, n_actions)
        values : (batch,)
        """
        out, _ = self.lstm(x)
        h      = self.norm(out[:, -1, :])
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values

    # ------------------------------------------------------------------
    # Single-step forward for efficient rollout collection
    # ------------------------------------------------------------------

    def step_hidden(
        self,
        x:      torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Advance the LSTM one timestep, carrying hidden state.

        x      : (1, 1, n_features)   single bar
        hidden : (h, c) or None
        Returns: (logits (1, n_actions), values (1,), new_hidden)
        """
        out, new_hidden = self.lstm(x, hidden)
        h      = self.norm(out[:, -1, :])
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values, new_hidden

    # ------------------------------------------------------------------
    # Stochastic action sampling — used during rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy (training / exploration).

        x : (1, seq_len, n_features)  — on the correct device
        Returns: (action, log_prob, value, entropy)  — all scalar tensors
        """
        logits, values = self.forward(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), values, dist.entropy()

    # ------------------------------------------------------------------
    # Greedy inference — for backtesting via RLTradingStrategy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_rl(
        self, x: torch.Tensor
    ) -> tuple[int, float]:
        """
        Greedy (argmax) action for deployment/backtesting.

        x : (1, seq_len, n_features+2)  — state-augmented input
        Returns: (desired_position, confidence)
          desired_position : -1 (short), 0 (flat), +1 (long)
          confidence       : max softmax probability
        """
        logits, _ = self.forward(x)
        probs     = torch.softmax(logits, dim=-1)
        ac_action = int(probs.argmax(dim=-1).item())
        conf      = float(probs[0, ac_action].item())

        # TradingEnv convention: FLAT=0, LONG=1, SHORT=2  →  position {0, +1, -1}
        _AC_TO_POS = {0: 0, 1: 1, 2: -1}
        return _AC_TO_POS[ac_action], conf

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
        # Infer architecture from saved weights
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
