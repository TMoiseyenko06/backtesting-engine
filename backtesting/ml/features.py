"""
ICT Feature Engineering
-----------------------
All features are computed using only past/current bar data — zero lookahead.

ICT concepts implemented
~~~~~~~~~~~~~~~~~~~~~~~~
fair_value_gap_bull  : magnitude of nearest bullish FVG above current price
fair_value_gap_bear  : magnitude of nearest bearish FVG below current price
order_block_bull_dist: distance to nearest bullish order block (OB), /ATR
order_block_bear_dist: distance to nearest bearish order block (OB), /ATR
swing_high_dist      : distance from close to recent swing high, /ATR
swing_low_dist       : distance from close to recent swing low, /ATR
premium_discount     : position in range (0=discount, 1=premium, 0.5=eq)
market_structure     : +1 bullish (HH+HL), -1 bearish (LH+LL), 0 neutral
kill_zone_london     : 1 if inside London open kill zone (07:00-10:00 UTC)
kill_zone_ny         : 1 if inside NY open kill zone (12:00-15:00 UTC)

Standard technical features (normalized, lookahead-free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
atr_norm             : ATR / close  (volatility regime)
rsi_14               : RSI(14) scaled to [-1, 1]
ret_1..ret_20        : log returns over 1, 5, 10, 20 bars
vol_ratio            : 5-bar ATR / 20-bar ATR (local vs macro vol)
body_ratio           : |close-open| / (high-low+ε)  (candle body strength)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from backtesting.data_feed import Bar


FEATURE_NAMES: List[str] = [
    "fair_value_gap_bull",
    "fair_value_gap_bear",
    "order_block_bull_dist",
    "order_block_bear_dist",
    "swing_high_dist",
    "swing_low_dist",
    "premium_discount",
    "market_structure",
    "kill_zone_london",
    "kill_zone_ny",
    "atr_norm",
    "rsi_14",
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_ratio",
    "body_ratio",
]

N_FEATURES = len(FEATURE_NAMES)


class ICTFeatureEngineer:
    """
    Converts a list of Bar objects into a 2-D numpy array of shape
    (n_bars, N_FEATURES), safe to use as LSTM input.

    All features are normalized so they're on roughly the same scale.
    ATR-relative distances keep values bounded even during trend moves.

    Parameters
    ----------
    swing_lookback : int
        Number of bars to look back when identifying swing highs/lows (default 20).
    ob_lookback : int
        Number of bars to look back when searching for order blocks (default 50).
    fvg_lookback : int
        Number of recent bars to scan for open fair value gaps (default 30).
    range_lookback : int
        Number of bars used to define the premium/discount range (default 50).
    atr_period : int
        Period for ATR calculation (default 14).
    rsi_period : int
        Period for RSI calculation (default 14).
    """

    def __init__(
        self,
        swing_lookback: int = 20,
        ob_lookback: int = 50,
        fvg_lookback: int = 30,
        range_lookback: int = 50,
        atr_period: int = 14,
        rsi_period: int = 14,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.ob_lookback    = ob_lookback
        self.fvg_lookback   = fvg_lookback
        self.range_lookback = range_lookback
        self.atr_period     = atr_period
        self.rsi_period     = rsi_period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, bars: List[Bar]) -> np.ndarray:
        """
        Parameters
        ----------
        bars : list of Bar
            Full bar history (chronological order).

        Returns
        -------
        np.ndarray of shape (len(bars), N_FEATURES)
            NaN rows at the start (before enough history) are forward-filled
            then zero-filled so the array is always fully numeric.
        """
        n = len(bars)
        opens  = np.array([b.open  for b in bars], dtype=np.float64)
        highs  = np.array([b.high  for b in bars], dtype=np.float64)
        lows   = np.array([b.low   for b in bars], dtype=np.float64)
        closes = np.array([b.close for b in bars], dtype=np.float64)
        times  = [b.timestamp for b in bars]

        atr = self._atr(highs, lows, closes, self.atr_period)

        features = np.full((n, N_FEATURES), np.nan)

        for i in range(n):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue

            a = atr[i]
            c = closes[i]

            features[i, 0]  = self._fvg_bull(highs, lows, closes, i, a)
            features[i, 1]  = self._fvg_bear(highs, lows, closes, i, a)
            features[i, 2]  = self._ob_bull_dist(opens, highs, lows, closes, i, a)
            features[i, 3]  = self._ob_bear_dist(opens, highs, lows, closes, i, a)
            features[i, 4]  = self._swing_high_dist(highs, i, c, a)
            features[i, 5]  = self._swing_low_dist(lows, i, c, a)
            features[i, 6]  = self._premium_discount(highs, lows, i, c)
            features[i, 7]  = self._market_structure(highs, lows, closes, i)
            kl, kn          = self._kill_zones(times[i])
            features[i, 8]  = kl
            features[i, 9]  = kn
            features[i, 10] = a / c
            features[i, 11] = self._rsi(closes, i, self.rsi_period)
            features[i, 12] = self._log_ret(closes, i, 1)
            features[i, 13] = self._log_ret(closes, i, 5)
            features[i, 14] = self._log_ret(closes, i, 10)
            features[i, 15] = self._log_ret(closes, i, 20)
            features[i, 16] = self._vol_ratio(highs, lows, closes, i)
            features[i, 17] = self._body_ratio(opens, highs, lows, closes, i)

        # Forward-fill then zero-fill warmup NaNs
        df = pd.DataFrame(features)
        df = df.ffill().fillna(0.0)
        return df.values.astype(np.float32)

    # ------------------------------------------------------------------
    # ICT features
    # ------------------------------------------------------------------

    def _fvg_bull(self, highs, lows, closes, i, atr):
        """Bullish FVG: gap[i-2].high < gap[i].low. Return gap size / ATR."""
        best = 0.0
        start = max(2, i - self.fvg_lookback)
        for j in range(start, i + 1):
            gap = lows[j] - highs[j - 2]
            if gap > 0 and closes[i] >= highs[j - 2]:
                # price is at or above the bottom of the gap (in the zone)
                best = max(best, gap / atr)
        return min(best, 5.0)   # cap at 5 ATRs

    def _fvg_bear(self, highs, lows, closes, i, atr):
        """Bearish FVG: gap[i-2].low > gap[i].high. Return gap size / ATR."""
        best = 0.0
        start = max(2, i - self.fvg_lookback)
        for j in range(start, i + 1):
            gap = lows[j - 2] - highs[j]
            if gap > 0 and closes[i] <= lows[j - 2]:
                best = max(best, gap / atr)
        return min(best, 5.0)

    def _ob_bull_dist(self, opens, highs, lows, closes, i, atr):
        """
        Bullish OB: last bearish candle before a 3-bar upward impulse.
        Return distance from current close to OB high, / ATR.
        Positive = price above OB (already mitigated), negative = below OB.
        """
        lb = min(i, self.ob_lookback)
        for j in range(i - 2, i - lb, -1):
            if j < 1:
                break
            # bearish candle at j
            if closes[j] < opens[j]:
                # followed by up-move: close[j+1] and close[j+2] both higher
                if j + 2 <= i and closes[j + 1] > closes[j] and closes[j + 2] > closes[j + 1]:
                    ob_top = highs[j]
                    return np.clip((closes[i] - ob_top) / atr, -5.0, 5.0)
        return 0.0

    def _ob_bear_dist(self, opens, highs, lows, closes, i, atr):
        """
        Bearish OB: last bullish candle before a 3-bar downward impulse.
        """
        lb = min(i, self.ob_lookback)
        for j in range(i - 2, i - lb, -1):
            if j < 1:
                break
            if closes[j] > opens[j]:
                if j + 2 <= i and closes[j + 1] < closes[j] and closes[j + 2] < closes[j + 1]:
                    ob_bottom = lows[j]
                    return np.clip((ob_bottom - closes[i]) / atr, -5.0, 5.0)
        return 0.0

    def _swing_high_dist(self, highs, i, close, atr):
        """Distance from close to rolling swing high (lookback window), /ATR."""
        lb = max(0, i - self.swing_lookback)
        swing_h = highs[lb: i + 1].max()
        return np.clip((swing_h - close) / atr, 0.0, 10.0)

    def _swing_low_dist(self, lows, i, close, atr):
        """Distance from rolling swing low to close, /ATR."""
        lb = max(0, i - self.swing_lookback)
        swing_l = lows[lb: i + 1].min()
        return np.clip((close - swing_l) / atr, 0.0, 10.0)

    def _premium_discount(self, highs, lows, i, close):
        """
        Where is price in the recent range?
        0 = at the low (discount), 1 = at the high (premium), 0.5 = midpoint.
        """
        lb = max(0, i - self.range_lookback)
        range_high = highs[lb: i + 1].max()
        range_low  = lows[lb:  i + 1].min()
        span = range_high - range_low
        if span == 0:
            return 0.5
        return np.clip((close - range_low) / span, 0.0, 1.0)

    def _market_structure(self, highs, lows, closes, i):
        """
        Simplified market structure: compare last 3 rolling swing points.
        +1 = bullish (higher highs + higher lows)
        -1 = bearish (lower highs + lower lows)
         0 = neutral / consolidation
        """
        if i < self.swing_lookback * 2:
            return 0.0
        w = self.swing_lookback
        h1 = highs[max(0, i - 2*w): i - w].max()
        h2 = highs[max(0, i - w):   i    ].max()
        l1 = lows[max(0, i - 2*w):  i - w].min()
        l2 = lows[max(0, i - w):    i    ].min()

        bull = (h2 > h1) and (l2 > l1)
        bear = (h2 < h1) and (l2 < l1)
        if bull:
            return 1.0
        if bear:
            return -1.0
        return 0.0

    @staticmethod
    def _kill_zones(ts):
        """
        Returns (london_kz, ny_kz) binary flags.
        Timestamps assumed timezone-naive UTC or local — hour checked directly.
        London open KZ: 07:00-10:00 UTC
        NY open KZ    : 12:00-15:00 UTC
        """
        h = ts.hour
        london = 1.0 if 7 <= h < 10 else 0.0
        ny     = 1.0 if 12 <= h < 15 else 0.0
        return london, ny

    # ------------------------------------------------------------------
    # Technical features
    # ------------------------------------------------------------------

    @staticmethod
    def _atr(highs, lows, closes, period):
        n = len(closes)
        atr = np.full(n, np.nan)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
        # Wilder smoothing
        if n >= period:
            atr[period - 1] = tr[:period].mean()
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _rsi(closes, i, period):
        """RSI scaled to [-1, 1]. Returns 0 if insufficient history."""
        if i < period:
            return 0.0
        delta = np.diff(closes[max(0, i - period - 1): i + 1])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 1.0
        rs = avg_gain / avg_loss
        rsi_01 = rs / (1 + rs)       # [0,1]
        return rsi_01 * 2 - 1        # [-1,1]

    @staticmethod
    def _log_ret(closes, i, period):
        if i < period:
            return 0.0
        r = np.log(closes[i] / closes[i - period])
        return np.clip(r, -0.1, 0.1)   # clip extreme moves

    @staticmethod
    def _vol_ratio(highs, lows, closes, i):
        """Ratio of 5-bar ATR to 20-bar ATR. > 1 = vol expansion."""
        if i < 20:
            return 1.0
        def simple_atr(window_h, window_l, window_c):
            trs = [window_h[k] - window_l[k] for k in range(len(window_h))]
            return np.mean(trs) if trs else 1.0
        atr5  = simple_atr(highs[i-4:i+1],  lows[i-4:i+1],  closes[i-4:i+1])
        atr20 = simple_atr(highs[i-19:i+1], lows[i-19:i+1], closes[i-19:i+1])
        return np.clip(atr5 / (atr20 + 1e-9), 0.2, 5.0)

    @staticmethod
    def _body_ratio(opens, highs, lows, closes, i):
        """Candle body / full range. 1 = pure trend candle, 0 = pure doji."""
        body  = abs(closes[i] - opens[i])
        range_ = highs[i] - lows[i]
        return body / (range_ + 1e-9)


# ---------------------------------------------------------------------------
# Multi-timeframe utilities
# ---------------------------------------------------------------------------

# (mult, human-readable name, ICTFeatureEngineer kwargs)
# Base timeframe is 1m.  Multipliers: 1=1m, 5=5m, 15=15m, 60=1h, 240=4h.
# Lookbacks are in bars of the given TF — smaller for higher TFs so the
# bar buffer stays manageable at inference time.
_TF_SPECS = [
    (1,   "1m",  dict(swing_lookback=20, ob_lookback=50,  fvg_lookback=30, range_lookback=50)),
    (5,   "5m",  dict(swing_lookback=20, ob_lookback=50,  fvg_lookback=30, range_lookback=50)),
    (15,  "15m", dict(swing_lookback=15, ob_lookback=30,  fvg_lookback=20, range_lookback=30)),
    (60,  "1h",  dict(swing_lookback=10, ob_lookback=20,  fvg_lookback=12, range_lookback=20)),
    (240, "4h",  dict(swing_lookback=6,  ob_lookback=10,  fvg_lookback=6,  range_lookback=10)),
]

MTF_FEATURE_NAMES: List[str] = [
    f"{tf_name}_{feat}"
    for _, tf_name, _ in _TF_SPECS
    for feat in FEATURE_NAMES
]
N_MTF_FEATURES: int = len(MTF_FEATURE_NAMES)   # 5 × 18 = 90


def aggregate_bars(bars: List[Bar], n: int) -> List[Bar]:
    """
    Group every ``n`` consecutive base-timeframe bars into one OHLCV bar.

    The aggregated bar's timestamp is the last bar in the group (i.e., the
    bar that *closes* the higher-timeframe candle).  Only complete groups
    are returned — trailing bars that don't fill a full group are dropped.
    """
    result = []
    for start in range(0, len(bars) - n + 1, n):
        group = bars[start: start + n]
        result.append(Bar(
            timestamp=group[-1].timestamp,
            symbol=group[0].symbol,
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low  for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
            contract_multiplier=group[0].contract_multiplier,
        ))
    return result


class MultiTimeframeFeatureEngineer:
    """
    Stacks ICT features computed at five timeframes into one wide vector.

    Base timeframe is 1-minute bars:

        1m  (mult=1)   : ICT features on raw 1m bars            →  18 features
        5m  (mult=5)   : ICT features on 5-bar aggregates        →  18 features
        15m (mult=15)  : ICT features on 15-bar aggregates       →  18 features
        1h  (mult=60)  : ICT features on 60-bar aggregates       →  18 features
        4h  (mult=240) : ICT features on 240-bar aggregates      →  18 features

    Total: 90 features per base bar.

    Higher-TF features are aligned with ZERO lookahead: for base bar ``i``
    the higher-TF feature used is from the *last complete* HTF candle, i.e.
    HTF candle index ``(i + 1) // mult - 1``.  Early bars that precede the
    first complete HTF candle receive all-zeros for that TF block.

    Minimum bar buffer needed (conservative):
        4h lookback (10 agg bars) × 240 = 2,400 base bars
    Use ``maxlen ≈ 2500`` in the inference deque to be safe.
    """

    def __init__(self) -> None:
        self._specs = _TF_SPECS
        self._engineers = {
            mult: ICTFeatureEngineer(**kwargs)
            for mult, _, kwargs in _TF_SPECS
        }

    @property
    def n_features(self) -> int:
        return N_MTF_FEATURES

    def transform(self, bars: List[Bar]) -> np.ndarray:
        """
        Parameters
        ----------
        bars : list of Bar  (base timeframe: 1m)

        Returns
        -------
        np.ndarray  shape (n_bars, N_MTF_FEATURES=90), dtype float32
        """
        n = len(bars)
        parts: List[np.ndarray] = []

        for mult, _, _ in self._specs:
            if mult == 1:
                parts.append(self._engineers[1].transform(bars))
            else:
                agg = aggregate_bars(bars, mult)
                if len(agg) == 0:
                    parts.append(np.zeros((n, N_FEATURES), dtype=np.float32))
                    continue

                feat_htf = self._engineers[mult].transform(agg)   # (n_agg, 18)

                # Vectorised alignment — no lookahead:
                # last complete HTF bar for base bar i  =  (i+1)//mult - 1
                idx = np.arange(n)
                htf_idx = (idx + 1) // mult - 1
                valid   = htf_idx >= 0
                clamped = np.clip(htf_idx, 0, len(feat_htf) - 1)
                aligned = feat_htf[clamped].copy()
                aligned[~valid] = 0.0
                parts.append(aligned)

        return np.concatenate(parts, axis=1)
