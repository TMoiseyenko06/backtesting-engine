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
                             ├─ Linear(32, 3)      → direction logits (Buy/Flat/Sell)
                             └─ Linear(32, 2) + Softplus → sl_atr_mult, tp_atr_mult

Design decisions to reduce overfitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Only 2 LSTM layers (not 3-4)
- Hidden size 64 (not 256+)
- Dropout both inside LSTM and in the FC head
- L2 weight decay applied by the optimiser (set in Trainer)
- Class weights passed to CrossEntropyLoss so the flat-heavy label
  distribution doesn't cause the model to just always predict Flat

Multi-task outputs
~~~~~~~~~~~~~~~~~~
- direction_head : 3-class logits (Sell=0, Flat=1, Buy=2)
- sl_tp_head     : 2 positive scalars [sl_atr_mult, tp_atr_mult]
                   representing how many ATRs away to place SL and TP.
                   Trained via MSE on max-adverse / max-favorable excursion
                   over the label horizon (only on Buy/Sell bars).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from backtesting.ml.features import N_FEATURES, N_MTF_FEATURES

# Safety bounds for predicted SL/TP ATR multiples
SL_TP_MIN = 0.3
SL_TP_MAX = 6.0


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

    @classmethod
    def from_checkpoint(
        cls,
        path,
        device: str | None = None,
    ) -> "LSTMSignalModel":
        """
        Load a saved model, auto-detecting ``n_features`` from the checkpoint
        so you don't need to know the feature count at call time.
        """
        from backtesting.ml.trainer import select_device  # avoid circular import
        if device is None:
            device = select_device()
        import torch
        state = torch.load(path, map_location=device, weights_only=True)
        # lstm.weight_ih_l0 shape: (4*hidden, n_features)
        n_features = state["lstm.weight_ih_l0"].shape[1]
        model = cls(n_features=n_features)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def __init__(
        self,
        n_features: int = N_MTF_FEATURES,
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

        # Shared trunk — both heads branch off here
        self.shared = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(fc_dropout * 0.75),
        )

        # Head 1: direction classification (Buy / Flat / Sell)
        self.direction_head = nn.Linear(32, n_classes)

        # Head 2: SL/TP regression — outputs sl_atr_mult and tp_atr_mult.
        # Softplus ensures positive outputs; we add SL_TP_MIN so the floor
        # is never zero.
        self.sl_tp_head = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        logits  : Tensor (batch, n_classes)  — raw direction logits
        sl_tp   : Tensor (batch, 2)          — [sl_atr_mult, tp_atr_mult], always > 0
        """
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # top layer's final hidden state
        shared = self.shared(last_hidden)
        logits = self.direction_head(shared)
        sl_tp  = self.sl_tp_head(shared) + SL_TP_MIN
        return logits, sl_tp

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax direction probabilities (no grad)."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> int:
        """
        Return predicted class (0=Sell, 1=Flat, 2=Buy) for a single
        sequence tensor of shape (1, seq_len, n_features).
        """
        probs = self.predict_proba(x)
        return int(probs.argmax(dim=-1).item())

    def predict_with_sl_tp(
        self, x: torch.Tensor
    ) -> tuple[int, float, float, float]:
        """
        Run a single forward pass and return direction + SL/TP multipliers.

        Returns
        -------
        (predicted_class, confidence, sl_atr_mult, tp_atr_mult)
        """
        self.eval()
        with torch.no_grad():
            logits, sl_tp = self.forward(x)
        probs       = torch.softmax(logits, dim=-1)[0]
        predicted   = int(probs.argmax().item())
        confidence  = float(probs[predicted].item())
        sl_mult     = float(sl_tp[0, 0].clamp(SL_TP_MIN, SL_TP_MAX).item())
        tp_mult     = float(sl_tp[0, 1].clamp(SL_TP_MIN, SL_TP_MAX).item())
        return predicted, confidence, sl_mult, tp_mult

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
