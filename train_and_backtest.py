"""
NQ LSTM — Train on all data except last year, Backtest on last 1 year
======================================================================

Usage
-----
    python train_and_backtest.py <path-to-dbn-file> [options]

    python train_and_backtest.py glbx-mdp3-20210313-20260313.ohlcv-1m.dbn.zstd
    python train_and_backtest.py data/nq.dbn.zstd --contracts 2 --cash 1000000

Arguments
---------
    path            Path to the Databento .dbn or .dbn.zstd batch file.

Options
-------
    --symbol        Bar symbol label (default: NQ)
    --multiplier    Contract multiplier, $ per point (default: 20.0)
    --cash          Starting capital (default: 500000)
    --init-margin   Initial margin per contract (default: 21000)
    --maint-margin  Maintenance margin per contract (default: 19000)
    --contracts     Contracts per trade (default: 1)
    --seq-len       LSTM sequence length, bars (default: 30)
    --conf          Minimum signal confidence 0-1 (default: 0.50)
    --folds         Walk-forward training folds (default: 3)
    --epochs        Max training epochs per fold (default: 50)
    --hidden        LSTM hidden size (default: 512)
    --batch-size    Training batch size (default: 4096)
    --save-model    Path to save the trained model .pt file
                    (default: models/nq_lstm_ohlcv1m.pt)
    --no-save       Skip saving the model to disk
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


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
                           f"·  PyTorch {info['torch_version']}  ·  BF16 AMP"))
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
    from backtesting.ml.dataset import make_labels, walk_forward_splits
    from backtesting.ml.features import make_features
    from backtesting.ml.trainer import Trainer
    from backtesting.portfolio import Portfolio, MarginSpec
    from strategies.lstm_signal import LSTMSignalStrategy

    # ── 1. GPU detection ──────────────────────────────────────────────────
    device_info = _detect_device()
    device = device_info["device"]

    print()
    print(_banner(f"NQ LSTM  ·  DATABENTO OHLCV-1m  ·  TRAIN ALL  /  BACKTEST LAST 1 YEAR"))
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

    # ── 4. Feature engineering & labels on training set ───────────────────
    print("  Building features & labels …", flush=True)
    features = make_features(train_bars)
    labels   = make_labels(train_bars)
    print(f"  Features shape : {features.shape}  (train set)")

    label_counts = np.bincount(labels, minlength=3)
    print(
        f"  Label balance  : "
        f"Sell={label_counts[0]:,}  Flat={label_counts[1]:,}  Buy={label_counts[2]:,}\n"
    )

    # ── 5. Walk-forward training ──────────────────────────────────────────
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
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(splits):
        result = trainer.fit(features, labels, train_idx, val_idx, fold=i)
        fold_results.append(result)
    elapsed = time.time() - t0

    # Save model
    model_path: str | None = None
    if not args.no_save:
        model_path = args.save_model
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(model_path)

    print()
    _print_train_section(fold_results, model_path, elapsed)

    # ── 6. Backtest on held-out 20% ───────────────────────────────────────
    warmup = args.seq_len + 60   # seq_len + SMA-50 warmup
    feed_test = DataFeed(test_bars, warmup_bars=warmup)

    strategy = LSTMSignalStrategy(
        model=trainer.model,
        device=device,
        symbol=args.symbol,
        seq_len=args.seq_len,
        confidence_threshold=args.conf,
        contracts=args.contracts,
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
