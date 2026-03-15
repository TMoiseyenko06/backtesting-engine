"""
Training loop for LSTMModel.

Features
--------
  - Weighted cross-entropy to handle class imbalance (Flat usually dominates)
  - Early stopping on validation loss
  - LR scheduling (ReduceLROnPlateau)
  - Per-fold metrics printed to stdout
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backtesting.ml.model import LSTMModel, _auto_device
from backtesting.ml.dataset import SequenceDataset


class Trainer:
    """
    Parameters
    ----------
    n_features  : int
    hidden_size : int
    num_layers  : int
    dropout     : float
    seq_len     : int
    lr          : float    initial learning rate
    batch_size  : int
    max_epochs  : int
    patience    : int      early-stopping patience (epochs without val improvement)
    device      : str | None  auto-detected if None
    weight_decay: float    L2 regularisation
    """

    def __init__(
        self,
        n_features:      int   = 22,
        hidden_size:     int   = 64,
        num_layers:      int   = 2,
        dropout:         float = 0.3,
        seq_len:         int   = 30,
        lr:              float = 1e-3,
        batch_size:      int   = 256,
        max_epochs:      int   = 50,
        patience:        int   = 7,
        device:          Optional[str] = None,
        weight_decay:    float = 1e-4,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.device      = device or _auto_device()

        self.model = LSTMModel(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Warm-start from checkpoint if available
        if checkpoint_path:
            from pathlib import Path
            if Path(checkpoint_path).exists():
                state = torch.load(checkpoint_path, map_location=self.device,
                                   weights_only=True)
                try:
                    self.model.load_state_dict(state)
                    print(f"  [checkpoint] Warm-started from {checkpoint_path}")
                except RuntimeError:
                    print(f"  [checkpoint] Architecture mismatch — starting fresh")

        self.optimiser = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", factor=0.5, patience=3
        )

    def fit(
        self,
        features:     np.ndarray,
        labels:       np.ndarray,
        train_idx:    np.ndarray,
        val_idx:      np.ndarray,
        fold:         int = 0,
    ) -> dict:
        """
        Train for one walk-forward fold.

        Returns
        -------
        dict with keys: val_acc, val_loss, train_acc, best_epoch
        """
        train_ds = SequenceDataset(features, labels, self.seq_len, train_idx)
        val_ds   = SequenceDataset(features, labels, self.seq_len, val_idx)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=(self.device == "cuda"),
        )

        # Class weights from training labels only
        train_labels = labels[train_idx[train_idx >= self.seq_len]]
        weights = _class_weights(train_labels, n_classes=3)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self.device)
        )

        best_val_loss = float("inf")
        best_val_acc  = 0.0
        best_epoch    = 0
        no_improve    = 0
        best_state    = None

        for epoch in range(1, self.max_epochs + 1):
            train_acc, train_loss = self._run_epoch(
                train_loader, criterion, training=True
            )
            val_acc, val_loss = self._run_epoch(
                val_loader, criterion, training=False
            )

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc  = val_acc
                best_epoch    = epoch
                no_improve    = 0
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"    fold {fold+1}  epoch {epoch:>3}  "
                    f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                    f"val_loss={val_loss:.4f}"
                )

            if no_improve >= self.patience:
                print(f"    fold {fold+1}  early stop at epoch {epoch}")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        # Final train acc from best weights
        train_acc_final, _ = self._run_epoch(train_loader, criterion, training=False)

        return {
            "val_acc":    best_val_acc,
            "val_loss":   best_val_loss,
            "train_acc":  train_acc_final,
            "best_epoch": best_epoch,
        }

    def save(self, path) -> None:
        self.model.save(path)

    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        training: bool,
    ) -> tuple[float, float]:
        self.model.train(training)
        total_loss  = 0.0
        total_correct = 0
        total_samples = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if training:
                    self.optimiser.zero_grad()

                logits = self.model(x)
                loss   = criterion(logits, y)

                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimiser.step()

                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                total_loss    += loss.item() * y.size(0)

        acc  = total_correct / total_samples if total_samples > 0 else 0.0
        loss = total_loss    / total_samples if total_samples > 0 else 0.0
        return acc, loss


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _class_weights(labels: np.ndarray, n_classes: int = 3) -> list:
    """Inverse-frequency weights so rare classes aren't ignored."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    return weights.tolist()
