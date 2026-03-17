#!/usr/bin/env python3
"""
live_trade.py — Launch the live trading module.

Usage
-----
  # Using a YAML config file:
  python live_trade.py --config live_config.yaml

  # Using env vars only (no YAML):
  LIVE_DATABENTO_KEY=abc LIVE_PMT_TOKEN=xyz LIVE_PMT_ACCOUNT_ID=123 \\
      python live_trade.py

  # Dry run (no webhooks sent — safe for testing):
  python live_trade.py --config live_config.yaml --dry-run

  # Override symbol and contracts on the command line:
  python live_trade.py --config live_config.yaml --symbol ES.c.0 --contracts 2

  # Use a specific trained model:
  python live_trade.py --config live_config.yaml --model models/nq_rl_best.pt

Requirements
------------
  pip install databento pyyaml requests torch

Data cost (Databento)
---------------------
  Databento charges ~$0.27/GB pay-as-you-go for live streaming.
  NQ 1-minute OHLCV bars use < 1 MB/day — roughly $0.01/day.
  Sign up at https://databento.com (free $125 credit on signup).

PickMyTrade setup
-----------------
  1. Create an account at https://pickmytrade.trade
  2. Connect your broker (Tradovate, Rithmic, IB, etc.)
  3. Copy your API token and account ID from the dashboard
  4. Paste them into live_config.yaml (or set env vars)
  5. Use the Tradovate webhook URL for Tradovate, .io URL for others:
       Tradovate : https://api.pickmytrade.trade/v2/add-trade-data
       Others    : https://api.pickmytrade.io/v2/add-trade-data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Live trading: Databento → ActorCriticLSTM → PickMyTrade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: none — use env vars).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model path (default: from config).",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Override Databento symbol, e.g. 'ES.c.0' (default: from config).",
    )
    p.add_argument(
        "--contracts",
        type=int,
        default=None,
        help="Override number of contracts (default: from config).",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Override compute device (default: from config).",
    )
    p.add_argument(
        "--sl-pts",
        type=float,
        default=None,
        dest="sl_pts_override",
        help="Fix stop-loss distance in points (overrides model prediction).",
    )
    p.add_argument(
        "--tp-pts",
        type=float,
        default=None,
        dest="tp_pts_override",
        help="Fix take-profit distance in points (overrides model prediction).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals without sending actual PickMyTrade webhooks.",
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Load config ───────────────────────────────────────────────────
    from live_trader.config import LiveConfig

    if args.config:
        cfg = LiveConfig.from_yaml(args.config)
    else:
        cfg = LiveConfig.from_env()

    # Apply CLI overrides
    if args.model       is not None: cfg.model_path        = args.model
    if args.symbol      is not None: cfg.symbol            = args.symbol
    if args.contracts   is not None: cfg.contracts         = args.contracts
    if args.device      is not None: cfg.device            = args.device
    if args.sl_pts_override is not None: cfg.sl_pts_override = args.sl_pts_override
    if args.tp_pts_override is not None: cfg.tp_pts_override = args.tp_pts_override
    if args.log_level   is not None: cfg.log_level         = args.log_level

    # ── Logging ───────────────────────────────────────────────────────
    logging.basicConfig(
        level   = getattr(logging, cfg.log_level, logging.INFO),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        stream  = sys.stdout,
    )
    log = logging.getLogger(__name__)

    if args.dry_run:
        log.warning("DRY RUN MODE — no webhooks will be sent")

    # ── Validate config ───────────────────────────────────────────────
    try:
        cfg.validate()
    except (ValueError, FileNotFoundError) as exc:
        log.error("Config error: %s", exc)
        sys.exit(1)

    # ── Launch ────────────────────────────────────────────────────────
    from live_trader.trader import LiveTrader

    log.info("=" * 60)
    log.info("Live Trader starting")
    log.info("  model      : %s", cfg.model_path)
    log.info("  symbol     : %s", cfg.symbol)
    log.info("  contracts  : %d", cfg.contracts)
    log.info("  device     : %s", cfg.device)
    log.info("  eod_utc    : %d:00", cfg.eod_hour_utc)
    log.info("  sl_override: %s pts", cfg.sl_pts_override or "model")
    log.info("  tp_override: %s pts", cfg.tp_pts_override or "model")
    log.info("  dry_run    : %s", args.dry_run)
    log.info("=" * 60)

    trader = LiveTrader(config=cfg, dry_run=args.dry_run)
    trader.start()


if __name__ == "__main__":
    main()
