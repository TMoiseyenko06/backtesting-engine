"""
NQ LSTM / PPO-RL — Train on all data except last year, Backtest on last 1 year
===============================================================================

Usage
-----
    python train_and_backtest.py <path-to-dbn-file> [options]

    python train_and_backtest.py data/nq.dbn                          # RL (default)
    python train_and_backtest.py data/nq.dbn --no-rl                  # supervised
    python train_and_backtest.py data/nq.dbn --contracts 2 --cash 1000000

Training modes
--------------
  RL (default, --rl):
    Proximal Policy Optimization — the network directly learns to trade for P&L.
    The actor-critic LSTM outputs a desired position (long/flat/short) each bar
    and is rewarded by the mark-to-market P&L change minus transaction costs.

  Supervised (--no-rl):
    3-class cross-entropy on forward-return labels.  The model predicts direction;
    a rule-based layer translates predictions into trades.

Arguments
---------
    path            Path to the Databento .dbn or .dbn.zstd batch file.

Options (shared)
----------------
    --symbol        Bar symbol label (default: NQ)
    --multiplier    Contract multiplier, $ per point (default: 20.0)
    --cash          Starting capital (default: 500000)
    --init-margin   Initial margin per contract (default: 21000)
    --maint-margin  Maintenance margin per contract (default: 19000)
    --contracts     Contracts per trade (default: 1)
    --seq-len       LSTM sequence length, bars (default: 30)
    --hidden        LSTM hidden size (default: 512)
    --save-model    Path to save model weights (default: models/nq_lstm_ohlcv1m.pt)
    --no-save       Skip saving the model to disk
    --max-loss      Max dollar loss per trade / stop-loss (default: 2500)
    --eod-hour      UTC hour to flatten all positions (default: 20 ≈ 4 PM EDT)
    --no-entry-hour UTC hour after which no new entries (default: 19)
    --min-hold-bars Min bars to hold before signal-driven exit/flip (default: 2)

Options (RL-specific)
---------------------
    --rl-iters      PPO training iterations (default: 200)
    --rl-days       Episodes (trading days) per PPO rollout (default: 16)
    --rl-ppo-epochs PPO update epochs per iteration (default: 4)
    --rl-lr         PPO Adam learning rate (default: 3e-4)

Options (supervised-specific)
------------------------------
    --conf          Minimum signal confidence 0-1 (default: 0.50)
    --folds         Walk-forward training folds (default: 3)
    --epochs        Max training epochs per fold (default: 50)
    --batch-size    Training batch size (default: 4096)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_device() -> dict:
    """
    Probe for CUDA → MPS → CPU.
    Returns a dict with keys: device, name, vram_gb, cuda_version, torch_version.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Run: pip install torch")

    info = {"torch_version": torch.__version__, "cuda_version": "n/a",
            "device": "cpu", "name": "CPU", "vram_gb": None}

    if torch.cuda.is_available():
        info["device"]  = "cuda"
        info["n_gpus"]  = torch.cuda.device_count()
        info["name"]    = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"]      = round(props.total_memory / 1024 ** 3, 1)
        info["cuda_version"] = torch.version.cuda or "n/a"
        info["compute_cap"]  = f"{props.major}.{props.minor}"
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["name"]   = "Apple MPS"
    return info


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

W = 70   # dashboard width

def _banner(title: str) -> str:
    pad = W - 2
    return f"{'=' * W}\n  {title}\n{'=' * W}"

def _section(label: str) -> str:
    return f"  [{label}]"

def _rule() -> str:
    return "  " + "-" * (W - 4)

def _kv(key: str, value: str, indent: int = 12) -> str:
    return f"  {key:<{indent}} {value}"

def _print_device_section(info: dict) -> None:
    if info["device"] == "cuda":
        n    = info.get("n_gpus", 1)
        vram = info['vram_gb']
        cap  = info.get("compute_cap", "")
        print(_section("DEVICE"))
        gpu_label = f"{n}x " if n > 1 else ""
        print(_kv("GPU :", f"{gpu_label}NVIDIA {info['name']}  "
                           f"({vram} GB VRAM each  ·  {vram * n:.0f} GB total)"))
        print(_kv("",      f"CUDA {info['cuda_version']}  ·  Compute {cap}  "
                           f"·  PyTorch {info['torch_version']}  ·  TF32"))
    elif info["device"] == "mps":
        print(_section("DEVICE"))
        print(_kv("GPU :", f"Apple MPS  ·  PyTorch {info['torch_version']}"))
    else:
        print(_section("DEVICE"))
        print(_kv("CPU :", f"No GPU detected  ·  PyTorch {info['torch_version']}"))
    print()


def _print_data_section(path: str, bars: list, split: int) -> None:
    total  = len(bars)
    t_bars = split
    b_bars = total - split
    t_start = bars[0].timestamp.strftime("%Y-%m-%d")
    t_end   = bars[split - 1].timestamp.strftime("%Y-%m-%d")
    b_start = bars[split].timestamp.strftime("%Y-%m-%d")
    b_end   = bars[-1].timestamp.strftime("%Y-%m-%d")

    print(_section("DATA"))
    print(_kv("File :",   Path(path).name))
    print(_kv("Bars :",   f"{total:,}  ({bars[0].timestamp.strftime('%Y-%m-%d')} "
                           f"→ {bars[-1].timestamp.strftime('%Y-%m-%d')})"))
    pct_train = 100.0 * t_bars / (t_bars + b_bars)
    pct_test  = 100.0 * b_bars / (t_bars + b_bars)
    print(_kv("Train :",  f"{t_bars:,} bars  ({pct_train:.1f}%)  {t_start} → {t_end}"))
    print(_kv("Test  :",  f"{b_bars:,} bars  ({pct_test:.1f}%)  {b_start} → {b_end}  [last 1 year]"))
    print()


def _print_train_section(fold_results: list, model_path: str | None, elapsed: float) -> None:
    print(_section("TRAIN"))
    for i, r in enumerate(fold_results):
        print(
            f"  fold {i+1}  "
            f"val_acc={r['val_acc']:.4f}  "
            f"train_acc={r['train_acc']:.4f}  "
            f"val_loss={r['val_loss']:.4f}  "
            f"best_epoch={r['best_epoch']}"
        )
    print(_rule())
    mean_val = np.mean([r["val_acc"] for r in fold_results])
    best_val = max(r["val_acc"] for r in fold_results)
    print(_kv("Mean val_acc  :", f"{mean_val:.4f}"))
    print(_kv("Best val_acc  :", f"{best_val:.4f}"))
    print(_kv("Train time    :", f"{elapsed:.1f}s"))
    if model_path:
        print(_kv("Model saved   :", model_path))
    print()


def _print_rl_train_section(metrics: dict, model_path: str | None, elapsed: float) -> None:
    print(_section("TRAIN  (PPO-RL)"))
    print(_kv("Mean daily P&L :", f"${metrics.get('mean_episode_pnl', 0):+,.0f}"))
    print(_kv("Policy loss    :", f"{metrics.get('policy_loss', 0):.4f}"))
    print(_kv("Value  loss    :", f"{metrics.get('value_loss',  0):.4f}"))
    print(_kv("Entropy        :", f"{metrics.get('entropy',     0):.4f}"))
    print(_kv("Clip fraction  :", f"{metrics.get('clip_frac',   0):.3f}"))
    print(_kv("Train time     :", f"{elapsed:.1f}s"))
    if model_path:
        print(_kv("Model saved    :", model_path))
    print()


def _print_backtest_section(analytics, symbol: str, test_bars: int,
                             b_start: str, b_end: str) -> None:
    a = analytics
    print(_banner(
        f"BACKTEST RESULTS — {symbol} LSTM Signal  ·  "
        f"last 1 year  ({b_start} → {b_end})"
    ))
    print(f"  {'Bars tested':<24}: {test_bars:>10,}")
    print(_rule())
    sign = "+" if a.total_return >= 0 else ""
    print(f"  {'Total Return':<24}: ${a.total_return:>12,.2f}  ({sign}{a.total_return_pct:.2f}%)")
    print(f"  {'Final Equity':<24}: ${a.final_equity:>12,.2f}")
    print(f"  {'Max Drawdown':<24}: ${a.max_drawdown:>12,.2f}  ({a.max_drawdown_pct:.2f}%)")
    print(f"  {'Sharpe Ratio':<24}: {a.sharpe_ratio:>12.4f}")
    print(f"  {'Sortino Ratio':<24}: {a.sortino_ratio:>12.4f}")
    print(f"  {'Calmar Ratio':<24}: {a.calmar_ratio:>12.4f}")
    print(_rule())
    print(f"  {'Total Trades':<24}: {a.total_trades:>12,}")
    print(f"  {'Win Rate':<24}: {a.win_rate:>11.1f}%")
    print(f"  {'Avg Win':<24}: ${a.avg_win:>12,.2f}")
    print(f"  {'Avg Loss':<24}: ${a.avg_loss:>12,.2f}")
    print(f"  {'Profit Factor':<24}: {a.profit_factor:>12.4f}")
    print(f"  {'Expectancy / Trade':<24}: ${a.expectancy:>12,.2f}")
    print(f"  {'Largest Win':<24}: ${a.largest_win:>12,.2f}")
    print(f"  {'Largest Loss':<24}: ${a.largest_loss:>12,.2f}")
    print(f"  {'Total Commission':<24}: ${a.total_commission:>12,.2f}")
    print("=" * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LSTM on 80% of Databento OHLCV-1m data, "
                    "then backtest on the remaining 20%."
    )
    p.add_argument("path", help="Path to .dbn or .dbn.zstd batch file")
    p.add_argument("--symbol",       default="NQ",        help="Bar symbol label")
    p.add_argument("--multiplier",   type=float, default=20.0)
    p.add_argument("--cash",         type=float, default=500_000.0)
    p.add_argument("--init-margin",  type=float, default=21_000.0, dest="init_margin")
    p.add_argument("--maint-margin", type=float, default=19_000.0, dest="maint_margin")
    p.add_argument("--contracts",    type=float, default=1.0)
    p.add_argument("--seq-len",      type=int,   default=30,  dest="seq_len")
    p.add_argument("--conf",         type=float, default=0.50)
    p.add_argument("--folds",        type=int,   default=3)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--hidden",       type=int,   default=512)
    p.add_argument("--batch-size",   type=int,   default=4096, dest="batch_size")
    p.add_argument("--save-model",   default="models/nq_lstm_ohlcv1m.pt", dest="save_model")
    p.add_argument("--no-save",      action="store_true", dest="no_save")
    p.add_argument("--max-loss",     type=float, default=2_500.0, dest="max_loss",
                   help="Max dollar loss per trade before stop-out (default $2500)")
    p.add_argument("--eod-hour",     type=int,   default=20, dest="eod_hour",
                   help="UTC hour to flatten all positions / stop trading (default 20 ≈ 4 PM EDT)")
    p.add_argument("--no-entry-hour", type=int,  default=19, dest="no_entry_hour",
                   help="UTC hour after which no new entries are taken (default 19)")
    p.add_argument("--min-hold-bars", type=int,  default=2, dest="min_hold_bars",
                   help="Min bars to hold before a signal-driven exit/flip (default 2)")
    # Training mode
    p.add_argument("--rl",    dest="rl", action="store_true",  default=True,
                   help="Use PPO reinforcement learning (default)")
    p.add_argument("--no-rl", dest="rl", action="store_false",
                   help="Use supervised cross-entropy training instead of RL")
    # RL-specific
    p.add_argument("--rl-iters",           type=int,   default=200,  dest="rl_iters",
                   help="PPO training iterations (default 200)")
    p.add_argument("--rl-days",            type=int,   default=16,   dest="rl_days",
                   help="Episodes per PPO rollout (default 16)")
    p.add_argument("--rl-ppo-epochs",      type=int,   default=4,    dest="rl_ppo_epochs",
                   help="PPO update epochs per iteration (default 4)")
    p.add_argument("--rl-lr",              type=float, default=3e-4, dest="rl_lr",
                   help="PPO Adam learning rate (default 3e-4)")
    p.add_argument("--prediction-horizon", type=int,   default=30,   dest="prediction_horizon",
                   help="Bars ahead for the model to predict (default 30 = 30 min)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── 0. imports (deferred so --help is fast) ───────────────────────────
    from backtesting.data_feed import DataFeed
    from backtesting.engine import BacktestEngine
    from backtesting.loaders.databento import from_databento_file
    from backtesting.ml.features import make_features
    from backtesting.portfolio import Portfolio, MarginSpec

    # ── 1. GPU detection ──────────────────────────────────────────────────
    device_info = _detect_device()
    device = device_info["device"]

    mode_label = "PPO-RL" if args.rl else "SUPERVISED"
    print()
    print(_banner(f"NQ LSTM  ·  DATABENTO OHLCV-1m  ·  {mode_label}  ·  TRAIN ALL  /  BACKTEST LAST 1 YEAR"))
    print()
    _print_device_section(device_info)

    # ── 2. Load data ───────────────────────────────────────────────────────
    print(f"  Loading {args.path} …", flush=True)
    feed_all = from_databento_file(
        path=args.path,
        symbol=args.symbol,
        contract_multiplier=args.multiplier,
    )
    bars: List = feed_all._bars
    print(f"  Loaded {len(bars):,} bars.\n")

    # ── 3. Split: train = everything before last 1 year ───────────────────
    from datetime import timedelta
    cutoff = bars[-1].timestamp - timedelta(days=365)
    split = next(i for i, b in enumerate(bars) if b.timestamp >= cutoff)
    train_bars = bars[:split]
    test_bars  = bars[split:]

    _print_data_section(args.path, bars, split)

    # ── 4. Feature engineering on training set ────────────────────────────
    print("  Building features …", flush=True)
    features = make_features(train_bars)
    print(f"  Features shape : {features.shape}  (train set)")

    if not args.rl:
        from backtesting.ml.dataset import make_labels
        labels = make_labels(train_bars)
        label_counts = np.bincount(labels, minlength=3)
        print(
            f"  Label balance  : "
            f"Sell={label_counts[0]:,}  Flat={label_counts[1]:,}  Buy={label_counts[2]:,}"
        )
    print()

    # ── 5. Training ───────────────────────────────────────────────────────
    model_path: str | None = None
    if not args.no_save:
        model_path = args.save_model
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.rl:
        # ── PPO reinforcement learning ────────────────────────────────
        from backtesting.ml.ppo_trainer import PPOTrainer
        from strategies.rl_trading import RLTradingStrategy

        print(f"  Training with PPO (RL) — {args.rl_iters} iterations …\n", flush=True)
        ppo_trainer = PPOTrainer(
            hidden_size=args.hidden,
            seq_len=args.seq_len,
            lr=args.rl_lr,
            n_iterations=args.rl_iters,
            rollout_days=args.rl_days,
            ppo_epochs=args.rl_ppo_epochs,
            prediction_horizon=args.prediction_horizon,
            device=device,
            multiplier=args.multiplier,
            commission=2.0,
            contracts=args.contracts,
            max_loss=args.max_loss,
        )
        rl_metrics = ppo_trainer.fit(train_bars, features)
        elapsed = time.time() - t0

        if model_path:
            ppo_trainer.save(model_path)

        print()
        _print_rl_train_section(rl_metrics, model_path, elapsed)

        trained_model = ppo_trainer._policy   # bare ActorCriticLSTM

        # ── 6a. Backtest (RL) ─────────────────────────────────────────
        warmup   = args.seq_len + 60
        feed_test = DataFeed(test_bars, warmup_bars=warmup)
        strategy = RLTradingStrategy(
            model=trained_model,
            device=device,
            symbol=args.symbol,
            seq_len=args.seq_len,
            contracts=args.contracts,
            max_loss_per_trade=args.max_loss,
            eod_hour_utc=args.eod_hour,
            no_entry_hour_utc=args.no_entry_hour,
        )

    else:
        # ── Supervised walk-forward training ─────────────────────────
        from backtesting.ml.dataset import walk_forward_splits
        from backtesting.ml.trainer import Trainer
        from strategies.lstm_signal import LSTMSignalStrategy

        print(f"  Training LSTM — {args.folds} walk-forward folds …\n", flush=True)
        trainer = Trainer(
            hidden_size=args.hidden,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            device=device,
        )
        splits = walk_forward_splits(len(train_bars), n_splits=args.folds)
        if not splits:
            print("  ERROR: not enough training data for walk-forward splits.", file=sys.stderr)
            sys.exit(1)

        fold_results = []
        for i, (train_idx, val_idx) in enumerate(splits):
            fold_results.append(trainer.fit(features, labels, train_idx, val_idx, fold=i))
        elapsed = time.time() - t0

        if model_path:
            trainer.save(model_path)

        print()
        _print_train_section(fold_results, model_path, elapsed)

        trained_model = (
            trainer.model.module
            if isinstance(trainer.model, torch.nn.DataParallel)
            else trainer.model
        )

        # ── 6b. Backtest (supervised) ─────────────────────────────────
        warmup   = args.seq_len + 60
        feed_test = DataFeed(test_bars, warmup_bars=warmup)
        strategy = LSTMSignalStrategy(
            model=trained_model,
            device=device,
            symbol=args.symbol,
            seq_len=args.seq_len,
            confidence_threshold=args.conf,
            contracts=args.contracts,
            max_loss_per_trade=args.max_loss,
            eod_hour_utc=args.eod_hour,
            no_entry_hour_utc=args.no_entry_hour,
            min_hold_bars=args.min_hold_bars,
        )

    portfolio = Portfolio(
        initial_cash=args.cash,
        margin_specs={
            args.symbol: MarginSpec(
                args.symbol,
                args.init_margin,
                args.maint_margin,
                args.multiplier,
            )
        },
        commission_per_contract=2.0,
        slippage_ticks=1,
        tick_size=0.25,
    )

    print("  Running backtest on test set …", flush=True)
    result = BacktestEngine(feed_test, portfolio, [strategy], verbose=False).run()

    # ── 7. Dashboard ──────────────────────────────────────────────────────
    print()
    b_start = test_bars[0].timestamp.strftime("%Y-%m-%d")
    b_end   = test_bars[-1].timestamp.strftime("%Y-%m-%d")
    _print_backtest_section(result.analytics, args.symbol, len(test_bars), b_start, b_end)
    print()


if __name__ == "__main__":
    main()
