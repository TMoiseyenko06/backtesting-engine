"""
LSTM signal model — pure price/volume features, 3-class output.

Architecture
------------
  Input  : (batch, seq_len, n_features)
  LSTM   : 2 layers, hidden_size=64, dropout between layers
  Norm   : LayerNorm on final hidden state
  Head   : Dropout → Linear(hidden, 3)  — Buy / Flat / Sell
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backtesting.ml.features import N_FEATURES


class LSTMModel(nn.Module):
    """
    Parameters
    ----------
    n_features  : int   input size (default = N_FEATURES from features.py)
    hidden_size : int   LSTM hidden units per layer
    num_layers  : int   stacked LSTM layers
    dropout     : float dropout applied between LSTM layers and before FC head
    n_classes   : int   output classes (3: Buy=2, Flat=1, Sell=0)
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        n_classes:   int   = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns logits : (batch, n_classes)
        """
        # Disable autocast for the LSTM: even with float32 input the LSTM
        # weight matmuls are downcast to BF16 by the outer autocast context,
        # causing NaN in hidden states.  autocast(enabled=False) forces all
        # internal matmuls to stay in float32.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out, _ = self.lstm(x.float())   # (batch, seq_len, hidden)
        h = out[:, -1, :]            # last timestep
        h = self.norm(h)
        return self.head(h)          # (batch, n_classes)

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        """
        Single-sample inference.

        x : (1, seq_len, n_features)  — already on the correct device
        returns (predicted_class, confidence)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
            cls    = int(probs.argmax(dim=-1).item())
            conf   = float(probs[0, cls].item())
        return cls, conf

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "LSTMModel":
        """
        Load a saved model, inferring architecture from checkpoint weights.
        """
        if device is None:
            device = _auto_device()
        state = torch.load(path, map_location=device, weights_only=True)
        # Infer dims from saved weights
        ih = state["lstm.weight_ih_l0"]
        n_features  = ih.shape[1]
        hidden_size = ih.shape[0] // 4
        num_layers  = sum(1 for k in state if k.startswith("lstm.weight_ih_l"))
        model = cls(n_features=n_features, hidden_size=hidden_size,
                    num_layers=num_layers)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model


def _auto_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  [device] CUDA GPU: {name}")
        return "cuda"
    if torch.backends.mps.is_available():
        print("  [device] Apple MPS")
        return "mps"
    print("  [device] CPU")
    return "cpu"
