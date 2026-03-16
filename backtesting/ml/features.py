"""
Pure OHLCV feature engineering — no external data, no ICT concepts.

Every feature is scale-invariant (returns, ratios, normalised values) so
the model can generalise across different price levels and time periods.

Features per bar (N_FEATURES = 22):
------------------------------------
  Price structure  (7)
    log_return          log(close / prev_close)
    open_return         (open  - prev_close) / prev_close
    high_return         (high  - prev_close) / prev_close
    low_return          (low   - prev_close) / prev_close
    candle_body         (close - open) / open
    upper_wick          (high  - max(open, close)) / open
    lower_wick          (min(open, close) - low)   / open

  Trend / MA  (6)
    sma5_dist           close / sma(5)  - 1
    sma10_dist          close / sma(10) - 1
    sma20_dist          close / sma(20) - 1
    sma50_dist          close / sma(50) - 1
    ema9_dist           close / ema(9)  - 1
    ema21_dist          close / ema(21) - 1

  Momentum  (3)
    roc5                close / close[t-5]  - 1
    roc10               close / close[t-10] - 1
    roc20               close / close[t-20] - 1

  Volatility  (2)
    atr14_norm          atr(14) / close
    bb_position         (close - bb_mid) / bb_width  (BB period=20, 2σ)

  Range position  (1)
    price_in_range      (close - low) / (high - low)   within-bar position

  Volume  (2)
    log_volume          log(1 + volume)
    vol_ratio           volume / rolling_mean_volume(20)

  RSI  (1)
    rsi14               RSI(14) / 100   (scaled to 0-1)
"""

from __future__ import annotations

from typing import List

import numpy as np

from backtesting.data_feed import Bar

N_FEATURES = 22


def make_features(bars: List[Bar]) -> np.ndarray:
    """
    Compute OHLCV-only features for every bar.

    Parameters
    ----------
    bars : list of Bar
        Chronological 1m bars (minimum ~60 bars recommended for valid MAs).

    Returns
    -------
    np.ndarray, shape (n_bars, N_FEATURES), dtype float32
        Rows with insufficient history are filled with 0.
    """
    n = len(bars)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)

    closes  = np.array([b.close  for b in bars], dtype=np.float64)
    opens   = np.array([b.open   for b in bars], dtype=np.float64)
    highs   = np.array([b.high   for b in bars], dtype=np.float64)
    lows    = np.array([b.low    for b in bars], dtype=np.float64)
    volumes = np.array([b.volume for b in bars], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Pre-compute rolling windows                                          #
    # ------------------------------------------------------------------ #
    sma5   = _rolling_mean(closes, 5)
    sma10  = _rolling_mean(closes, 10)
    sma20  = _rolling_mean(closes, 20)
    sma50  = _rolling_mean(closes, 50)
    ema9   = _ema(closes, 9)
    ema21  = _ema(closes, 21)
    atr14  = _atr(highs, lows, closes, 14)
    vol20  = _rolling_mean(volumes, 20)
    rsi14  = _rsi(closes, 14)
    bb_mid, bb_std = _rolling_mean(closes, 20), _rolling_std(closes, 20)

    for i in range(1, n):
        c   = closes[i]
        o   = opens[i]
        h   = highs[i]
        l   = lows[i]
        pc  = closes[i - 1]  # previous close

        # ── Price structure ──────────────────────────────────────────── #
        log_ret    = np.log(c / pc) if pc > 0 else 0.0
        open_ret   = (o - pc) / pc if pc > 0 else 0.0
        high_ret   = (h - pc) / pc if pc > 0 else 0.0
        low_ret    = (l - pc) / pc if pc > 0 else 0.0
        body       = (c - o) / o   if o  > 0 else 0.0
        up_wick    = (h - max(o, c)) / o if o > 0 else 0.0
        lo_wick    = (min(o, c) - l) / o if o > 0 else 0.0

        # ── Trend / MA ───────────────────────────────────────────────── #
        s5  = (c / sma5[i]  - 1) if sma5[i]  > 0 else 0.0
        s10 = (c / sma10[i] - 1) if sma10[i] > 0 else 0.0
        s20 = (c / sma20[i] - 1) if sma20[i] > 0 else 0.0
        s50 = (c / sma50[i] - 1) if sma50[i] > 0 else 0.0
        e9  = (c / ema9[i]  - 1) if ema9[i]  > 0 else 0.0
        e21 = (c / ema21[i] - 1) if ema21[i] > 0 else 0.0

        # ── Momentum ─────────────────────────────────────────────────── #
        roc5  = (c / closes[i - 5]  - 1) if i >= 5  and closes[i - 5]  > 0 else 0.0
        roc10 = (c / closes[i - 10] - 1) if i >= 10 and closes[i - 10] > 0 else 0.0
        roc20 = (c / closes[i - 20] - 1) if i >= 20 and closes[i - 20] > 0 else 0.0

        # ── Volatility ───────────────────────────────────────────────── #
        atr_n   = atr14[i] / c if c > 0 else 0.0
        bb_w    = 2 * bb_std[i]
        bb_pos  = ((c - bb_mid[i]) / bb_w) if bb_w > 0 else 0.0

        # ── Range position ───────────────────────────────────────────── #
        rng = h - l
        pir = (c - l) / rng if rng > 0 else 0.5

        # ── Volume ───────────────────────────────────────────────────── #
        # Normalise relative to rolling mean so scale matches other features.
        # Raw log1p(volume) ≈ 7–12 for futures, 100× larger than price/return
        # features, which saturates LSTM cell states and kills convergence.
        log_vol  = (np.log1p(volumes[i]) / np.log1p(vol20[i]) - 1.0) if vol20[i] > 0 else 0.0
        vol_rat  = (volumes[i] / vol20[i]) if vol20[i] > 0 else 1.0

        # ── RSI ──────────────────────────────────────────────────────── #
        rsi = rsi14[i] / 100.0

        out[i] = [
            log_ret, open_ret, high_ret, low_ret, body, up_wick, lo_wick,  # 7
            s5, s10, s20, s50, e9, e21,                                      # 6
            roc5, roc10, roc20,                                               # 3
            atr_n, bb_pos,                                                    # 2
            pir,                                                               # 1
            log_vol, vol_rat,                                                  # 2
            rsi,                                                               # 1
        ]

    # Safety net: any NaN / ±Inf that slipped through (e.g. a zero-price bar,
    # floating-point near-zero denominator) would silently corrupt every
    # downstream gradient.  Replace with 0 so the bar is effectively "unseen".
    np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return out


# --------------------------------------------------------------------------- #
# Rolling helpers                                                              #
# --------------------------------------------------------------------------- #

def _rolling_mean(x: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros_like(x)
    cumsum = np.cumsum(x)
    for i in range(period - 1, len(x)):
        out[i] = (cumsum[i] - (cumsum[i - period] if i >= period else 0)) / period
    return out


def _rolling_std(x: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(period - 1, len(x)):
        out[i] = x[i - period + 1: i + 1].std()
    return out


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros_like(x)
    k = 2.0 / (period + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int) -> np.ndarray:
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
    # Wilder smoothing
    atr = np.zeros(n)
    if n > period:
        atr[period] = tr[1:period + 1].mean()
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    rsi = np.full(n, 50.0)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains[i]  = max(delta, 0)
        losses[i] = max(-delta, 0)
    if n <= period:
        return rsi
    avg_gain = gains[1:period + 1].mean()
    avg_loss = losses[1:period + 1].mean()
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i])  / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi
