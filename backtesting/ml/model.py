"""
LSTM Signal Model
-----------------
Deliberately modest in size to avoid overfitting.

Architecture
~~~~~~~~~~~~
  Input: (batch, seq_len, n_features)
    └─ LSTM(hidden=64, layers=2, dropout=0.3)
         └─ take last hidden state  → (batch, 64)
              └─ Dropout(0.4)
                   └─ Linear(64, 32)  → ReLU
                        └─ Dropout(0.3)
                             └─ Linear(32, 3)   → logits (Buy/Flat/Sell)

Design decisions to reduce overfitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Only 2 LSTM layers (not 3-4)
- Hidden size 64 (not 256+)
- Dropout both inside LSTM and in the FC head
- L2 weight decay applied by the optimiser (set in Trainer)
- Class weights passed to CrossEntropyLoss so the flat-heavy label
  distribution doesn't cause the model to just always predict Flat
"""

from __future__ import annotations

import torch
import torch.nn as nn
from backtesting.ml.features import N_FEATURES


class LSTMSignalModel(nn.Module):
    """
    Parameters
    ----------
    n_features : int
        Number of input features per bar (default N_FEATURES from features.py).
    hidden_size : int
        LSTM hidden units (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    lstm_dropout : float
        Dropout between LSTM layers (default 0.3).
    fc_dropout : float
        Dropout before the FC head (default 0.4).
    n_classes : int
        Output classes: 3 (Buy=2, Flat=1, Sell=0).
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.4,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(fc_dropout * 0.75),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        Tensor of shape (batch, n_classes)  — raw logits
        """
        # out: (batch, seq_len, hidden)
        # h_n: (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # top layer's final hidden state
        return self.head(last_hidden)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (no grad)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> int:
        """
        Return predicted class (0=Sell, 1=Flat, 2=Buy) for a single
        sequence tensor of shape (1, seq_len, n_features).
        """
        probs = self.predict_proba(x)
        return int(probs.argmax(dim=-1).item())

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
