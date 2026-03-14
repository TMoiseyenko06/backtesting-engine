"""
Dataset utilities for the ML training pipeline.

Responsibilities:
  - make_labels()          : forward-return labelling (Buy / Flat / Sell)
  - walk_forward_splits()  : time-series cross-validation indices
  - SequenceDataset        : PyTorch Dataset wrapping feature sequences
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from backtesting.data_feed import Bar
from backtesting.ml.features import make_features, _atr


# --------------------------------------------------------------------------- #
# Label generation                                                             #
# --------------------------------------------------------------------------- #

# Class index mapping
BUY  = 2
FLAT = 1
SELL = 0


def make_labels(
    bars: List[Bar],
    horizon: int = 10,
    atr_mult: float = 0.5,
    atr_period: int = 14,
) -> np.ndarray:
    """
    Generate 3-class labels based on forward return vs ATR threshold.

    For each bar i, look `horizon` bars ahead:
        forward_return = (close[i+horizon] - close[i]) / close[i]
        threshold      = atr_mult * atr[i] / close[i]

        label = BUY  (2) if forward_return >  +threshold
        label = SELL (0) if forward_return <  -threshold
        label = FLAT (1) otherwise

    The ATR-based threshold adapts to current volatility, so the label
    distribution stays balanced even as market conditions change.

    Parameters
    ----------
    bars : list of Bar
    horizon : int
        Number of bars to look forward for the return.
    atr_mult : float
        Multiplier on ATR to set the buy/sell threshold.
    atr_period : int
        ATR lookback period.

    Returns
    -------
    np.ndarray, shape (n,), dtype int64
        Labels 0/1/2 for Sell/Flat/Buy.  Last `horizon` bars are FLAT
        (no future data available).
    """
    n = len(bars)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    highs  = np.array([b.high  for b in bars], dtype=np.float64)
    lows   = np.array([b.low   for b in bars], dtype=np.float64)

    atr = _atr(highs, lows, closes, atr_period)

    labels = np.full(n, FLAT, dtype=np.int64)

    for i in range(n - horizon):
        c_now   = closes[i]
        c_ahead = closes[i + horizon]
        if c_now <= 0:
            continue
        fwd_ret   = (c_ahead - c_now) / c_now
        threshold = atr_mult * atr[i] / c_now if c_now > 0 else 0.0
        if fwd_ret > threshold:
            labels[i] = BUY
        elif fwd_ret < -threshold:
            labels[i] = SELL

    return labels


# --------------------------------------------------------------------------- #
# Walk-forward splits                                                          #
# --------------------------------------------------------------------------- #

def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    test_frac: float = 0.15,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce time-series (non-shuffled) train/validation index pairs.

    Each split trains on an expanding window ending before the val period.
    The optional `gap` drops bars between train end and val start to prevent
    label leakage when horizon > 1.

    Parameters
    ----------
    n : int
        Total number of samples.
    n_splits : int
        Number of folds.
    test_frac : float
        Fraction of total bars used as validation in each fold.
    gap : int
        Bars to skip between train end and val start.

    Returns
    -------
    list of (train_indices, val_indices)
    """
    test_size  = max(1, int(n * test_frac))
    splits = []
    for k in range(n_splits):
        # val window slides from the end backwards
        val_end   = n - k * (test_size // n_splits)
        val_start = val_end - test_size
        if val_start <= 0:
            break
        train_end = val_start - gap
        if train_end <= 10:
            break
        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return list(reversed(splits))  # chronological order


# --------------------------------------------------------------------------- #
# PyTorch Dataset                                                              #
# --------------------------------------------------------------------------- #

class SequenceDataset:
    """
    Wraps pre-computed feature matrix + labels into fixed-length sequences
    for LSTM training.

    Parameters
    ----------
    features : np.ndarray, shape (n, n_features)
    labels   : np.ndarray, shape (n,)
    seq_len  : int
        Length of each input sequence.
    indices  : np.ndarray | None
        If provided, only use these row indices (used to enforce the
        train/val split without copying data).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
        indices: np.ndarray | None = None,
    ) -> None:
        import torch
        self._seq_len = seq_len

        if indices is not None:
            # Only keep rows where a full seq_len lookback is available
            valid = indices[indices >= seq_len]
        else:
            valid = np.arange(seq_len, len(features))

        self._x = torch.from_numpy(features).float()
        self._y = torch.from_numpy(labels).long()
        self._valid = valid

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int):
        i = self._valid[idx]
        x = self._x[i - self._seq_len: i]   # (seq_len, n_features)
        y = self._y[i]
        return x, y
