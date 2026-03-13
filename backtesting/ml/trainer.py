"""
Trainer
-------
Handles:
  * Class-weighted CrossEntropyLoss  (combats the flat-heavy label imbalance)
  * MSE regression loss on SL/TP targets (only on Buy/Sell bars)
  * Multi-task total loss = CE + sl_tp_weight * MSE
  * AdamW optimiser with L2 weight decay  (another overfitting guard)
  * Early stopping on validation loss
  * Walk-forward cross-validation via walk_forward_splits()
  * Saving / loading best model weights
  * Automatic device selection (CUDA → MPS → CPU)

Walk-forward CV is the MOST important protection against overfitting for
financial time series.  The model trains on older data and validates on the
immediately following out-of-sample window — exactly how you would deploy it.
If val performance degrades across folds, the model is over-fitted and should
be made simpler (reduce hidden_size / seq_len / add dropout).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backtesting.ml.dataset import SequenceDataset, walk_forward_splits
from backtesting.ml.model import LSTMSignalModel


def select_device() -> str:
    """
    Auto-select the best available compute device:
      CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  [device] CUDA GPU detected: {name}")
        return "cuda"
    if torch.backends.mps.is_available():
        print("  [device] Apple MPS detected.")
        return "mps"
    # Distinguish between CPU-only PyTorch build and a missing/blocked driver
    cuda_build = torch.__version__
    if "+cu" in cuda_build or "+cuda" in cuda_build:
        print(
            f"  [device] GPU not accessible (PyTorch={cuda_build}). "
            "Check that NVIDIA drivers are installed and the GPU is visible "
            "(run: nvidia-smi). Falling back to CPU."
        )
    else:
        print(
            f"  [device] CPU-only PyTorch build detected ({cuda_build}). "
            "For GPU support run: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    return "cpu"


class EarlyStopping:
    """Stop training when val loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.best_weights: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss    = val_loss
            self.counter      = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class Trainer:
    """
    Parameters
    ----------
    model : LSTMSignalModel
    seq_len : int
        LSTM look-back window in bars.
    batch_size : int
    epochs : int
        Max epochs per fold.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        L2 penalty (AdamW).  0.01–0.1 is typical.
    patience : int
        Early stopping patience (epochs).
    device : str | None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to auto-detect.
    sl_tp_weight : float
        Weight of the SL/TP regression loss relative to direction CE loss.
        Loss = CE + sl_tp_weight * MSE(sl_tp on trading bars only).
    """

    def __init__(
        self,
        model: LSTMSignalModel,
        seq_len: int = 30,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        patience: int = 10,
        device: str | None = None,
        sl_tp_weight: float = 0.5,
    ) -> None:
        self.device      = device if device is not None else select_device()
        self.model       = model.to(self.device)
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.lr          = lr
        self.weight_decay = weight_decay
        self.patience    = patience
        self.sl_tp_weight = sl_tp_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_walk_forward(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sl_tp_targets: np.ndarray,
        train_bars: int,
        val_bars: int,
        step_bars: int,
        save_path: Optional[str | Path] = None,
    ) -> List[dict]:
        """
        Run walk-forward cross-validation.

        Returns
        -------
        list of per-fold metrics dicts:
          {fold, train_size, val_size, val_loss, val_acc, elapsed_s}
        """
        n = len(features)
        folds = list(walk_forward_splits(n, train_bars, val_bars, step_bars))
        if not folds:
            raise ValueError(
                f"No folds generated.  n_bars={n}, train_bars={train_bars}, "
                f"val_bars={val_bars}.  Need at least train_bars + val_bars bars."
            )

        fold_metrics = []
        best_val_acc  = -1.0
        best_state    = None

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            t0 = time.time()
            train_ds = SequenceDataset(features, labels, sl_tp_targets, self.seq_len, train_idx)
            val_ds   = SequenceDataset(features, labels, sl_tp_targets, self.seq_len, val_idx)

            if len(train_ds) == 0 or len(val_ds) == 0:
                print(f"  Fold {fold_idx}: insufficient data, skipping.")
                continue

            class_weights = self._class_weights(labels[train_idx])
            val_loss, val_acc = self._train_fold(train_ds, val_ds, class_weights)

            elapsed = time.time() - t0
            metrics = {
                "fold":       fold_idx,
                "train_size": len(train_ds),
                "val_size":   len(val_ds),
                "val_loss":   round(val_loss, 4),
                "val_acc":    round(val_acc, 4),
                "elapsed_s":  round(elapsed, 1),
            }
            fold_metrics.append(metrics)
            print(
                f"  Fold {fold_idx:>2}  train={len(train_ds):>5}  val={len(val_ds):>5}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                f"  ({elapsed:.1f}s)"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\n  Best fold val_acc = {best_val_acc:.3f}")

        if save_path:
            self.save(save_path)

        return fold_metrics

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sl_tp_targets: np.ndarray,
        val_split: float = 0.2,
    ) -> dict:
        """
        Simple single-pass train/val split (chronological).
        Use fit_walk_forward for more rigorous evaluation.
        """
        n = len(features)
        split = int(n * (1 - val_split))
        train_idx = list(range(split))
        val_idx   = list(range(split, n))

        train_ds = SequenceDataset(features, labels, sl_tp_targets, self.seq_len, train_idx)
        val_ds   = SequenceDataset(features, labels, sl_tp_targets, self.seq_len, val_idx)

        class_weights = self._class_weights(labels[train_idx])
        val_loss, val_acc = self._train_fold(train_ds, val_ds, class_weights)
        return {"val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Model saved → {path}")

    @classmethod
    def load_model(
        cls,
        path: str | Path,
        model: LSTMSignalModel,
        device: str | None = None,
    ) -> LSTMSignalModel:
        if device is None:
            device = select_device()
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        train_ds: SequenceDataset,
        val_ds: SequenceDataset,
        class_weights: torch.Tensor,
    ) -> tuple[float, float]:
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        ce_criterion   = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        mse_criterion  = nn.MSELoss(reduction="none")
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs, eta_min=self.lr * 0.1
        )
        stopper = EarlyStopping(patience=self.patience)

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb, sltp_b in train_loader:
                xb     = xb.to(self.device)
                yb     = yb.to(self.device)
                sltp_b = sltp_b.to(self.device)

                optimiser.zero_grad()
                logits, sl_tp_pred = self.model(xb)

                # Direction classification loss (all bars)
                ce_loss = ce_criterion(logits, yb)

                # SL/TP regression loss — only on Buy (2) and Sell (0) bars
                trading_mask = (yb != 1)  # True for Buy and Sell bars
                if trading_mask.any():
                    mse_raw  = mse_criterion(sl_tp_pred[trading_mask],
                                             sltp_b[trading_mask])
                    mse_loss = mse_raw.mean()
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)

                loss = ce_loss + self.sl_tp_weight * mse_loss
                loss.backward()
                # Gradient clipping prevents exploding gradients in LSTM
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
            scheduler.step()

            val_loss, val_acc = self._evaluate(val_loader, ce_criterion)
            if stopper.step(val_loss, self.model):
                break

        stopper.restore_best(self.model)
        val_loss, val_acc = self._evaluate(val_loader, ce_criterion)
        return val_loss, val_acc

    def _evaluate(
        self,
        loader: DataLoader,
        ce_criterion: nn.Module,
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb, _sltp in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, _ = self.model(xb)
                total_loss += ce_criterion(logits, yb).item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)
        return (total_loss / total), (correct / total)

    @staticmethod
    def _class_weights(labels: np.ndarray) -> torch.Tensor:
        """Inverse-frequency weights so the model doesn't just predict Flat."""
        counts = np.bincount(labels.astype(int), minlength=3).astype(float)
        counts = np.where(counts == 0, 1.0, counts)   # avoid div-by-zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * 3          # normalise to sum=3
        return torch.tensor(weights, dtype=torch.float32)
