"""Built-in example strategies."""
from .ema_crossover import EMACrossoverStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .breakout import BreakoutStrategy
from .trend_following import TrendFollowingStrategy

__all__ = [
    "EMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "BreakoutStrategy",
    "TrendFollowingStrategy",
]
