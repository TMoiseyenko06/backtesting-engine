"""
Dataset utilities
-----------------
SequenceDataset  : PyTorch Dataset that serves sliding windows of features
                   and a 3-class label (Buy=2, Flat=1, Sell=0).

walk_forward_splits : generator that yields (train_idx, val_idx) index ranges
                      for proper time-series cross-validation.  This is the
                      primary defence against overfitting — the model NEVER
                      sees future data during training.

Label engineering
~~~~~~~~~~~~~~~~~
The label for bar i is based on the *future* return over `horizon` bars,
normalised by ATR to be volatility-adjusted:

    z_ret = (close[i+horizon] - close[i]) / (ATR[i] * sqrt(horizon))

    Buy  (2) : z_ret >  threshold
    Sell (0) : z_ret < -threshold
    Flat (1) : otherwise

Using an ATR-normalised threshold means the model only acts on moves that
are large relative to current noise — realistic rather than chasing every
tick.  The default threshold of 0.5 means roughly half an ATR per root-bar.
"""

from __future__ import annotations

from typing import Generator, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Sliding-window sequence dataset.

    Parameters
    ----------
    features : np.ndarray, shape (n_bars, n_features)
        Pre-computed feature matrix from ICTFeatureEngineer.
    labels : np.ndarray, shape (n_bars,)
        Integer class labels {0, 1, 2}.
    seq_len : int
        Number of bars per sequence fed to the LSTM (look-back window).
    indices : list[int] | None
        If provided, only these bar indices (the *last* bar of each window)
        are included.  Used by the walk-forward splitter.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
        indices: List[int] | None = None,
    ) -> None:
        self.features = features.astype(np.float32)
        self.labels   = labels.astype(np.int64)
        self.seq_len  = seq_len

        if indices is None:
            # All valid end-of-window positions
            self._indices = list(range(seq_len - 1, len(features)))
        else:
            # Only positions that have enough history
            self._indices = [i for i in indices if i >= seq_len - 1]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end   = self._indices[idx]
        start = end - self.seq_len + 1
        x = torch.from_numpy(self.features[start: end + 1])   # (seq_len, n_feat)
        y = torch.tensor(self.labels[end], dtype=torch.long)
        return x, y


def make_labels(
    closes: np.ndarray,
    atr: np.ndarray,
    horizon: int = 12,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Build 3-class labels without lookahead leakage in the *model* — labels
    are only used during *training* (not at inference time).

    Parameters
    ----------
    closes : np.ndarray, shape (n,)
    atr : np.ndarray, shape (n,)
        ATR values (NaN for early bars).
    horizon : int
        How many bars forward to measure the return.
    threshold : float
        Minimum ATR-normalised return to trigger Buy/Sell.

    Returns
    -------
    np.ndarray of int8, shape (n,)
        0=Sell, 1=Flat, 2=Buy.  Last `horizon` bars are labelled Flat
        since no future return is available.
    """
    n = len(closes)
    labels = np.ones(n, dtype=np.int8)   # default Flat

    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a == 0:
            continue
        future_ret = (closes[i + horizon] - closes[i]) / closes[i]
        z = future_ret / (a / closes[i] * np.sqrt(horizon))
        if z > threshold:
            labels[i] = 2   # Buy
        elif z < -threshold:
            labels[i] = 0   # Sell
        # else stays 1 (Flat)

    return labels


def walk_forward_splits(
    n_bars: int,
    train_bars: int,
    val_bars: int,
    step_bars: int,
    min_train_bars: int | None = None,
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Yield (train_indices, val_indices) for expanding-window walk-forward CV.

    Each fold:
      - train set grows by `step_bars` (expanding window)
      - val set is the immediately following `val_bars` bars
      - the sets never overlap

    Example with n=1000, train=600, val=100, step=100:
      fold 0 : train [0..599],   val [600..699]
      fold 1 : train [0..699],   val [700..799]
      fold 2 : train [0..799],   val [800..899]

    Parameters
    ----------
    n_bars : int
    train_bars : int
        Initial training window size.
    val_bars : int
        Validation window size per fold.
    step_bars : int
        How many bars to advance the window each fold.
    min_train_bars : int | None
        Minimum training bars required (defaults to train_bars).
    """
    min_train = min_train_bars or train_bars
    train_end = train_bars

    while train_end + val_bars <= n_bars:
        val_end = train_end + val_bars
        if train_end >= min_train:
            yield list(range(train_end)), list(range(train_end, val_end))
        train_end += step_bars
