"""
LiveConfig — all settings for the live trading module.

Load from YAML:
    cfg = LiveConfig.from_yaml("live_config.yaml")

Override via env vars (prefixed LIVE_):
    LIVE_DATABENTO_KEY=abc123
    LIVE_PMT_TOKEN=xyz
    LIVE_PMT_ACCOUNT_ID=12345
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class LiveConfig:
    # ── Model ────────────────────────────────────────────────────────
    model_path: str = "models/nq_lstm_ohlcv1m.pt"
    seq_len: int = 30
    hidden_size: int = 512
    device: str = "cpu"           # "cpu" or "cuda"

    # ── Databento ────────────────────────────────────────────────────
    # Sign up at https://databento.com — pay-as-you-go, ~$0.27/GB.
    # Live streaming for NQ 1m bars costs cents per day.
    databento_key: str = ""
    symbol: str = "NQ.c.0"       # continuous front-month NQ
    dataset: str = "GLBX.MDP3"   # CME Globex futures
    # multiplier used to decode Databento fixed-point prices (always 1e9)
    price_scale: float = 1e9

    # ── PickMyTrade ──────────────────────────────────────────────────
    # Get your token + account_id from https://pickmytrade.trade dashboard.
    # Use the Tradovate endpoint if your broker is Tradovate, otherwise .io
    pickmytrade_token: str = ""
    pickmytrade_account_id: str = ""
    pickmytrade_url: str = "https://api.pickmytrade.trade/v2/add-trade-data"
    # Set to True to have PickMyTrade close any open position before entering
    # a new one (recommended — avoids accidental double-entries).
    reverse_order_close: bool = True

    # ── Trading rules ────────────────────────────────────────────────
    contracts: int = 1
    multiplier: float = 20.0      # NQ = $20/point; ES = $50/point
    eod_hour_utc: int = 20        # 4 PM EDT — force flat
    no_entry_hour_utc: int = 19   # 3 PM EDT — stop new entries
    reward_scale: float = 100.0   # must match training reward_scale
    max_loss_usd: float = 2500.0  # hard stop per trade (dollars)

    # ── SL/TP overrides ──────────────────────────────────────────────
    # Set > 0 to override the model's dynamic SL/TP predictions.
    # 0 means "use whatever the model predicts".
    sl_pts_override: float = 0.0
    tp_pts_override: float = 0.0

    # ── Misc ─────────────────────────────────────────────────────────
    # Bars needed before model can run: seq_len + 60 (SMA-50 warmup)
    warmup_bars: int = 90
    log_level: str = "INFO"

    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LiveConfig":
        if not _HAS_YAML:
            raise ImportError("PyYAML is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        cfg._apply_env_overrides()
        return cfg

    @classmethod
    def from_env(cls) -> "LiveConfig":
        """Construct from environment variables only (no YAML needed)."""
        cfg = cls()
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        """Env vars override YAML / defaults.  Prefix: LIVE_"""
        mapping = {
            "LIVE_DATABENTO_KEY":    "databento_key",
            "LIVE_PMT_TOKEN":        "pickmytrade_token",
            "LIVE_PMT_ACCOUNT_ID":   "pickmytrade_account_id",
            "LIVE_PMT_URL":          "pickmytrade_url",
            "LIVE_MODEL_PATH":       "model_path",
            "LIVE_DEVICE":           "device",
            "LIVE_SYMBOL":           "symbol",
            "LIVE_CONTRACTS":        "contracts",
            "LIVE_EOD_HOUR":         "eod_hour_utc",
        }
        for env_key, attr in mapping.items():
            val = os.environ.get(env_key)
            if val is not None:
                # Cast to the existing type
                existing = getattr(self, attr)
                setattr(self, attr, type(existing)(val))

    def validate(self) -> None:
        """Raise ValueError if required fields are missing."""
        if not self.databento_key:
            raise ValueError(
                "databento_key is required.\n"
                "Set it in your YAML config or via LIVE_DATABENTO_KEY env var.\n"
                "Sign up at https://databento.com (pay-as-you-go, ~$0.27/GB)."
            )
        if not self.pickmytrade_token:
            raise ValueError(
                "pickmytrade_token is required.\n"
                "Set it in your YAML config or via LIVE_PMT_TOKEN env var.\n"
                "Get it from your PickMyTrade dashboard."
            )
        if not self.pickmytrade_account_id:
            raise ValueError(
                "pickmytrade_account_id is required.\n"
                "Set it via LIVE_PMT_ACCOUNT_ID or in your YAML config."
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Train a model first with train_and_backtest.py."
            )
