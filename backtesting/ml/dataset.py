"""
Dataset utilities
-----------------
SequenceDataset  : PyTorch Dataset that serves sliding windows of features,
                   a 3-class direction label (Buy=2, Flat=1, Sell=0), and
                   a 2-element SL/TP regression target [sl_atr_mult, tp_atr_mult].

walk_forward_splits : generator that yields (train_idx, val_idx) index ranges
                      for proper time-series cross-validation.  This is the
                      primary defence against overfitting — the model NEVER
                      sees future data during training.

Label engineering
~~~~~~~~~~~~~~~~~
The direction label for bar i is based on the *future* return over `horizon`
bars, normalised by ATR to be volatility-adjusted:

    z_ret = (close[i+horizon] - close[i]) / (ATR[i] * sqrt(horizon))

    Buy  (2) : z_ret >  threshold
    Sell (0) : z_ret < -threshold
    Flat (1) : otherwise

SL/TP regression targets
~~~~~~~~~~~~~~~~~~~~~~~~
For each bar labeled Buy or Sell, the regression targets are the
Max Adverse Excursion (MAE) and Max Favorable Excursion (MFE) over
the next `horizon` bars, each normalised by ATR:

    Buy  label at bar i:
        sl_atr = (close[i] - min(lows[i+1 .. i+horizon])) / ATR[i]   ← downside MAE
        tp_atr = (max(highs[i+1 .. i+horizon]) - close[i]) / ATR[i]  ← upside  MFE

    Sell label at bar i:
        sl_atr = (max(highs[i+1 .. i+horizon]) - close[i]) / ATR[i]  ← upside  MAE
        tp_atr = (close[i] - min(lows[i+1 .. i+horizon])) / ATR[i]   ← downside MFE

Both targets are clipped to [0.3, 6.0].  For Flat bars the values default to
[1.5, 2.5] but are masked out of the regression loss during training.
"""

from __future__ import annotations

from typing import Generator, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

# Fallback SL/TP for flat bars (never used in loss, but stored for completeness)
_FLAT_SL = 1.5
_FLAT_TP = 2.5
_SL_TP_CLIP = (0.3, 6.0)


class SequenceDataset(Dataset):
    """
    Sliding-window sequence dataset.

    Parameters
    ----------
    features : np.ndarray, shape (n_bars, n_features)
        Pre-computed feature matrix from ICTFeatureEngineer.
    labels : np.ndarray, shape (n_bars,)
        Integer class labels {0, 1, 2}.
    sl_tp_targets : np.ndarray, shape (n_bars, 2)
        Regression targets [sl_atr_mult, tp_atr_mult] per bar.
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
        sl_tp_targets: np.ndarray,
        seq_len: int,
        indices: List[int] | None = None,
    ) -> None:
        self.features      = features.astype(np.float32)
        self.labels        = labels.astype(np.int64)
        self.sl_tp_targets = sl_tp_targets.astype(np.float32)
        self.seq_len       = seq_len

        if indices is None:
            self._indices = list(range(seq_len - 1, len(features)))
        else:
            self._indices = [i for i in indices if i >= seq_len - 1]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end   = self._indices[idx]
        start = end - self.seq_len + 1
        x      = torch.from_numpy(self.features[start: end + 1])        # (seq_len, n_feat)
        y_cls  = torch.tensor(self.labels[end], dtype=torch.long)        # scalar
        y_sltp = torch.from_numpy(self.sl_tp_targets[end])               # (2,)
        return x, y_cls, y_sltp


def make_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    horizon: int = 12,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 3-class direction labels and SL/TP regression targets without
    lookahead leakage in the *model* — labels are only used during *training*.

    Parameters
    ----------
    closes, highs, lows : np.ndarray, shape (n,)
    atr : np.ndarray, shape (n,)
        ATR values (NaN for early bars).
    horizon : int
        How many bars forward to measure the return / excursion.
    threshold : float
        Minimum ATR-normalised return to trigger Buy/Sell.

    Returns
    -------
    labels : np.ndarray of int8, shape (n,)
        0=Sell, 1=Flat, 2=Buy.  Last `horizon` bars are labelled Flat.
    sl_tp_targets : np.ndarray of float32, shape (n, 2)
        [sl_atr_mult, tp_atr_mult] per bar (MAE/MFE normalised by ATR).
    """
    n = len(closes)
    labels        = np.ones(n, dtype=np.int8)
    sl_tp_targets = np.full((n, 2), [_FLAT_SL, _FLAT_TP], dtype=np.float32)

    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a == 0:
            continue

        future_ret = (closes[i + horizon] - closes[i]) / closes[i]
        z = future_ret / (a / closes[i] * np.sqrt(horizon))

        future_highs = highs[i + 1: i + horizon + 1]
        future_lows  = lows[i + 1:  i + horizon + 1]

        if z > threshold:
            labels[i] = 2  # Buy
            # MAE = how far down it went (adverse for long)
            # MFE = how far up it went (favorable for long)
            sl_atr = (closes[i] - float(future_lows.min())) / a
            tp_atr = (float(future_highs.max()) - closes[i]) / a
        elif z < -threshold:
            labels[i] = 0  # Sell
            # MAE = how far up it went (adverse for short)
            # MFE = how far down it went (favorable for short)
            sl_atr = (float(future_highs.max()) - closes[i]) / a
            tp_atr = (closes[i] - float(future_lows.min())) / a
        else:
            continue  # Flat — keep defaults

        sl_tp_targets[i, 0] = np.clip(sl_atr, *_SL_TP_CLIP)
        sl_tp_targets[i, 1] = np.clip(tp_atr, *_SL_TP_CLIP)

    return labels, sl_tp_targets


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
