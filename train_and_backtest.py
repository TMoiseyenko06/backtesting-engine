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
    --cash          Starting capital (default: 100000)
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
    --entropy-coef  Entropy bonus (default: 0.05) — prevents FLAT collapse
    --flat-penalty  Dollar cost per flat bar (default: 1.0) — incentivises trading frequency
                    Start with 0.5–1.0 for ~5 trades/day; same scale as commission ($2/side)

Options (supervised-specific)
------------------------------
    --conf          Minimum signal confidence 0-1 (default: 0.50)
    --folds         Walk-forward training folds (default: 3)
    --epochs        Max training epochs per fold (default: 50)
    --batch-size    Training batch size (default: 4096)
"""

from __future__ import annotations

import argparse
import os
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
# Telegram notification
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """Parse a .env file and populate os.environ (existing vars take priority)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _notify_telegram(bot_token: str, chat_id: str, analytics, elapsed: float, mode: str) -> None:
    """
    Send a Telegram message via the Bot API when training + backtest finish.
    Silently does nothing if the request fails — never crashes the run.
    """
    import urllib.request, urllib.parse
    a = analytics
    msg = (
        f"*Backtest done — {mode}*\n"
        f"Finished in {elapsed:.0f}s\n\n"
        f"P&L: `${a.total_return:+,.0f}` ({a.total_return_pct:+.1f}%)\n"
        f"Trades: {a.total_trades}  |  Win rate: {a.win_rate:.0f}%\n"
        f"Sharpe: {a.sharpe_ratio:.2f}  |  Max DD: `${a.max_drawdown:,.0f}`"
    )
    params = urllib.parse.urlencode({
        "chat_id":    chat_id,
        "text":       msg,
        "parse_mode": "Markdown",
    }).encode()
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        urllib.request.urlopen(urllib.request.Request(url, data=params), timeout=10)
    except Exception:
        pass   # notification is best-effort; never crash the run


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
    print(_kv("Trades / day   :", f"{metrics.get('trades_per_day', 0):.1f}"))
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
# NN-based backtest helpers (replaces BacktestEngine for RL mode)
# ---------------------------------------------------------------------------

def _build_test_episodes(
    bars, features: np.ndarray, seq_len: int, prediction_horizon: int
) -> list:
    """
    Group test bars by UTC calendar date into episode tuples.
    Mirrors PPOTrainer._make_episodes() so the test env is identical to training.

    Returns list of (features, closes, highs, lows, fwd_returns) tuples.
    """
    from collections import defaultdict
    day_map: dict = defaultdict(list)
    for i, bar in enumerate(bars):
        day_map[bar.timestamp.date()].append(i)

    H = prediction_horizon
    episodes = []
    for date in sorted(day_map.keys()):
        idx = day_map[date]
        if len(idx) <= seq_len:
            continue
        f = features[idx].astype(np.float32)
        p = np.array([bars[i].close     for i in idx], dtype=np.float32)
        h = np.array([bars[i].high      for i in idx], dtype=np.float32)
        l = np.array([bars[i].low       for i in idx], dtype=np.float32)
        t = [bars[i].timestamp          for i in idx]
        n = len(p)
        fwd = np.zeros(n, dtype=np.float32)
        for i in range(n - H):
            if p[i] > 0:
                fwd[i] = (p[i + H] - p[i]) / p[i]
        episodes.append((f, p, h, l, fwd, t))
    return episodes


def _nn_backtest(model, test_episodes: list, env_kwargs: dict,
                 device: str, initial_cash: float,
                 trade_log_path: str | None = None,
                 fixed_sl_pts: float | None = None,
                 fixed_tp_pts: float | None = None):
    """
    Run the trained model greedily on every test episode using BatchedTradingEnv.
    Identical evaluation logic to PPOTrainer._evaluate_val(), run day-by-day so
    we can track individual trade P&L for analytics.

    Returns an analytics object with the same attributes used by
    _print_backtest_section() and _notify_telegram().
    """
    import csv
    import torch
    from backtesting.ml.rl_env import BatchedTradingEnv

    model.eval()
    reward_scale = env_kwargs["reward_scale"]

    # During the test we measure real P&L — strip training artifacts:
    # flat_penalty, win_bonus, and binary_reward all distort the rew_dollars
    # accumulator used for day_pnl/trade_pnl tracking below.
    bt_kwargs = {**env_kwargs, "flat_penalty": 0.0, "win_bonus": 0.0, "binary_reward": False}

    daily_pnls: list[float] = []
    trade_pnls: list[float] = []
    trade_log:  list[dict]  = []
    n_trades = 0

    for episode in test_episodes:
        # Episodes are 6-tuples: (features, closes, highs, lows, fwd_returns, timestamps)
        ep_data       = episode[:5]
        ep_timestamps = episode[5] if len(episode) == 6 else None

        env    = BatchedTradingEnv(n_envs=1, **bt_kwargs)
        states = env.reset_all([ep_data])

        day_pnl       = 0.0
        in_trade      = False
        trade_pnl_acc = 0.0
        entry_ts      = None
        entry_price   = 0.0
        entry_dir     = 0
        entry_sl      = 0.0
        entry_tp      = 0.0

        with torch.no_grad():
            while True:
                x       = torch.from_numpy(states).to(device)
                logits, _, _, sl_mean, tp_mean = model(x)

                actions = logits.argmax(dim=-1).cpu().numpy()   # (1,)
                sl_arr  = (np.array([fixed_sl_pts], dtype=np.float32)
                           if fixed_sl_pts is not None
                           else sl_mean.cpu().numpy().flatten())
                tp_arr  = (np.array([fixed_tp_pts], dtype=np.float32)
                           if fixed_tp_pts is not None
                           else tp_mean.cpu().numpy().flatten())

                directions = np.where(
                    actions == 1, np.int32(1),
                    np.where(actions == 2, np.int32(-1), np.int32(0))
                )

                states, rews, dones, entered = env.step_all(
                    directions.astype(np.int32), sl_arr, tp_arr
                )

                # cursor was incremented inside step_all; bar just processed is cursor-1
                bar_idx = int(env._cursors[0]) - 1
                bar_ts  = ep_timestamps[bar_idx] if ep_timestamps is not None else None

                rew_dollars = float(rews[0]) * reward_scale
                day_pnl    += rew_dollars

                # ── Track individual trade P&L ──────────────────────────
                if entered[0]:
                    n_trades += 1
                    entry_comm_dollars = env.commission * env.contracts
                    if in_trade:
                        # Previous bracket closed AND new entry in same bar.
                        # rew_dollars = old_exit_mtm - exit_comm - entry_comm
                        # Old trade's correct final-bar P&L = rew_dollars + entry_comm
                        trade_pnl_acc += rew_dollars + entry_comm_dollars
                        exit_type = "TP" if trade_pnl_acc > 0 else "SL"
                        trade_pnls.append(trade_pnl_acc)
                        trade_log.append({
                            "entry_time":  entry_ts,
                            "exit_time":   bar_ts,
                            "direction":   "LONG" if entry_dir == 1 else "SHORT",
                            "entry_price": entry_price,
                            "sl_price":    entry_sl,
                            "tp_price":    entry_tp,
                            "exit_type":   exit_type,
                            "pnl":         round(trade_pnl_acc, 2),
                        })
                    in_trade      = True
                    trade_pnl_acc = -entry_comm_dollars  # new trade: only entry commission
                    entry_ts      = bar_ts
                    entry_price   = float(env._entry_prices[0])
                    entry_dir     = int(directions[0])
                    entry_sl      = float(env._sl_prices[0])
                    entry_tp      = float(env._tp_prices[0])
                elif in_trade:
                    trade_pnl_acc += rew_dollars
                    # Check EOD first: env zeros _positions on dones, so we must
                    # not mistake an EOD forced-flat for an intraday SL/TP hit.
                    if dones[0]:
                        trade_pnls.append(trade_pnl_acc)
                        trade_log.append({
                            "entry_time":  entry_ts,
                            "exit_time":   bar_ts,
                            "direction":   "LONG" if entry_dir == 1 else "SHORT",
                            "entry_price": entry_price,
                            "sl_price":    entry_sl,
                            "tp_price":    entry_tp,
                            "exit_type":   "EOD",
                            "pnl":         round(trade_pnl_acc, 2),
                        })
                        in_trade = False
                    elif env._positions[0] == 0:         # intraday SL/TP hit
                        exit_type = "TP" if trade_pnl_acc > 0 else "SL"
                        trade_pnls.append(trade_pnl_acc)
                        trade_log.append({
                            "entry_time":  entry_ts,
                            "exit_time":   bar_ts,
                            "direction":   "LONG" if entry_dir == 1 else "SHORT",
                            "entry_price": entry_price,
                            "sl_price":    entry_sl,
                            "tp_price":    entry_tp,
                            "exit_type":   exit_type,
                            "pnl":         round(trade_pnl_acc, 2),
                        })
                        in_trade      = False
                        trade_pnl_acc = 0.0

                if dones[0]:
                    break

        daily_pnls.append(day_pnl)

    # ── Write trade log to CSV ─────────────────────────────────────────────
    if trade_log_path and trade_log:
        Path(trade_log_path).parent.mkdir(parents=True, exist_ok=True)
        fields = ["entry_time", "exit_time", "direction",
                  "entry_price", "sl_price", "tp_price", "exit_type", "pnl"]
        with open(trade_log_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trade_log)
        print(f"  Trade log saved → {trade_log_path}  ({len(trade_log)} trades)")

    # ── Analytics ─────────────────────────────────────────────────────────
    daily_arr  = np.array(daily_pnls, dtype=np.float64)
    cum_equity = initial_cash + np.cumsum(daily_arr)

    total_return = float(daily_arr.sum())

    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = float(daily_arr.mean() / daily_arr.std() * np.sqrt(252))

    sortino  = 0.0
    neg_days = daily_arr[daily_arr < 0]
    if len(neg_days) > 1 and neg_days.std() > 0:
        sortino = float(daily_arr.mean() / neg_days.std() * np.sqrt(252))

    peak   = np.maximum.accumulate(cum_equity)
    max_dd = float((peak - cum_equity).max()) if len(cum_equity) > 0 else 0.0
    max_dd_pct = max_dd / initial_cash * 100 if initial_cash > 0 else 0.0
    calmar     = total_return / max_dd if max_dd > 0 else 0.0

    win_rate = avg_win = avg_loss = 0.0
    largest_win = largest_loss = profit_factor = expectancy = 0.0
    if trade_pnls:
        tp_arr_np    = np.array(trade_pnls, dtype=np.float64)
        wins         = tp_arr_np[tp_arr_np > 0]
        losses       = tp_arr_np[tp_arr_np <= 0]
        win_rate     = 100.0 * len(wins) / len(tp_arr_np)
        avg_win      = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_loss     = float(losses.mean()) if len(losses) > 0 else 0.0
        largest_win  = float(wins.max())    if len(wins)   > 0 else 0.0
        largest_loss = float(losses.min())  if len(losses) > 0 else 0.0
        gross_profit = float(wins.sum())             if len(wins)   > 0 else 0.0
        gross_loss   = abs(float(losses.sum()))      if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss    if gross_loss  > 0 else float("inf")
        expectancy   = float(tp_arr_np.mean())

    total_commission = n_trades * env_kwargs["commission"] * env_kwargs["contracts"] * 2

    class _A:
        pass
    a = _A()
    a.total_return     = total_return
    a.total_return_pct = total_return / initial_cash * 100
    a.final_equity     = initial_cash + total_return
    a.max_drawdown     = max_dd
    a.max_drawdown_pct = max_dd_pct
    a.sharpe_ratio     = sharpe
    a.sortino_ratio    = sortino
    a.calmar_ratio     = calmar
    a.total_trades     = n_trades
    a.win_rate         = win_rate
    a.avg_win          = avg_win
    a.avg_loss         = avg_loss
    a.profit_factor    = profit_factor
    a.expectancy       = expectancy
    a.largest_win      = largest_win
    a.largest_loss     = largest_loss
    a.total_commission = total_commission
    return a


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    _load_dotenv()
    p = argparse.ArgumentParser(
        description="Train LSTM on 80% of Databento OHLCV-1m data, "
                    "then backtest on the remaining 20%."
    )
    p.add_argument("path", help="Path to .dbn or .dbn.zstd batch file")
    p.add_argument("--symbol",       default="NQ",        help="Bar symbol label")
    p.add_argument("--multiplier",   type=float, default=20.0)
    p.add_argument("--cash",         type=float, default=100_000.0)
    p.add_argument("--init-margin",  type=float, default=21_000.0, dest="init_margin")
    p.add_argument("--maint-margin", type=float, default=19_000.0, dest="maint_margin")
    p.add_argument("--contracts",    type=float, default=1.0)
    p.add_argument("--seq-len",      type=int,   default=60,  dest="seq_len",
                   help="LSTM lookback window in bars (default: 60 = 1 hour on 1-min data)."
                        " More bars = richer context but slower training."
                        " Good range: 60 (1h) to 120 (2h).")
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
    p.add_argument("--entropy-coef",       type=float, default=0.05, dest="entropy_coef",
                   help="PPO entropy bonus coefficient — higher prevents FLAT collapse (default 0.05)")
    p.add_argument("--flat-penalty",       type=float, default=1.0,  dest="flat_penalty",
                   help="Dollar penalty per flat bar during training — incentivises more trading "
                        "(default 1.0; same scale as commission)")
    p.add_argument("--prediction-horizon", type=int,   default=30,   dest="prediction_horizon",
                   help="Bars ahead for the model to predict (default 30 = 30 min)")
    p.add_argument("--weight-decay",       type=float, default=1e-4, dest="weight_decay",
                   help="Adam L2 weight decay / regularisation (default 1e-4)")
    p.add_argument("--val-frac",           type=float, default=0.15, dest="val_frac",
                   help="Fraction of training days held out for validation (default 0.15)")
    p.add_argument("--patience",           type=int,   default=80,   dest="patience",
                   help="Early-stop after this many iters with no val improvement (default 80, 0=off)")
    p.add_argument("--resume",             type=str,   default=None, dest="resume", metavar="PATH",
                   help="Path to .pt checkpoint to resume PPO training from (e.g. models/nq_lstm_ohlcv1m.pt)")
    p.add_argument("--backtest-only",      action="store_true",      dest="backtest_only",
                   help="Skip training; load model from --resume (or --save-model path) and run backtest only")
    p.add_argument("--tg-token",   default=os.environ.get("TELEGRAM_BOT_TOKEN", ""), dest="tg_token",
                   help="Telegram bot token — overrides TELEGRAM_BOT_TOKEN in .env")
    p.add_argument("--tg-chat",    default=os.environ.get("TELEGRAM_CHAT_ID", ""),   dest="tg_chat",
                   help="Telegram chat ID — overrides TELEGRAM_CHAT_ID in .env")
    p.add_argument("--trade-log",          type=str,   default="logs/trade_log.csv", dest="trade_log",
                   metavar="PATH",
                   help="CSV file to write per-trade log after backtesting (default: logs/trade_log.csv)")
    p.add_argument("--compile",            action="store_true",      dest="compile_model",
                   help="Enable torch.compile for fused CUDA kernels (PyTorch >= 2.0, ~10-30%% speedup)")
    p.add_argument("--win-bonus", type=float, default=0.0, dest="win_bonus",
                   metavar="DOLLARS",
                   help="Dollar bonus added to reward on TP hit, subtracted on SL hit."
                        " Trains the model to care about winrate independently of P&L magnitude."
                        " Good starting point: 100-500. Default: 0 (off).")
    p.add_argument("--val-winrate-coef", type=float, default=0.0, dest="val_winrate_coef",
                   metavar="DOLLARS",
                   help="Winrate weight in model selection score: score = val_pnl + coef * winrate."
                        " The best checkpoint saved is now the one with the highest combined score."
                        " E.g. 5000 means a 10%% winrate improvement is worth $500/day in selection."
                        " Default: 0 (select purely on P&L).")
    p.add_argument("--binary-reward",  action="store_true", default=False, dest="binary_reward",
                   help="Replace bar-by-bar MTM reward with a pure ±1 win/loss signal."
                        " With fixed --sl-pts/--tp-pts, P&L is a linear function of win rate,"
                        " so maximising P&L IS maximising win rate.  Binary mode removes"
                        " intermediate price noise and turns training into a classification"
                        " problem: 'will price hit TP before SL?'  Recommended with"
                        " --sl-pts and --tp-pts.  Default: off.")
    p.add_argument("--time-limit-bars", type=int, default=0, dest="time_limit_bars",
                   metavar="BARS",
                   help="Force-exit and penalise any trade that has not hit SL or TP within"
                        " this many bars of entry.  0 = disabled (default).  Good value: 30"
                        " (= 30 min on 1-min data).  Works with both standard and binary reward"
                        " modes; in binary mode the penalty equals -1.0 (same as an SL hit)."
                        " Trains the model to only enter when a fast, decisive move is expected.")
    p.add_argument("--max-trades", type=int, default=0, dest="max_trades_per_episode",
                   metavar="N",
                   help="Maximum trades per episode (day).  0 = unlimited (default)."
                        " Once N trades have been taken the model is locked out for the"
                        " rest of that day, forcing selectivity.  E.g. --max-trades 3.")
    p.add_argument("--sl-pts",  type=float, default=None, dest="sl_pts",
                   metavar="PTS",
                   help="Fixed SL distance in points (e.g. 100). Overrides model SL head."
                        " When set, every trade uses exactly this stop-loss distance.")
    p.add_argument("--tp-pts",  type=float, default=None, dest="tp_pts",
                   metavar="PTS",
                   help="Fixed TP distance in points (e.g. 200). Overrides model TP head."
                        " When set, every trade uses exactly this take-profit distance.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── 0. imports (deferred so --help is fast) ───────────────────────────
    from backtesting.loaders.databento import from_databento_file
    from backtesting.ml.features import make_features

    # ── 1. GPU detection + VRAM reset ─────────────────────────────────────
    device_info = _detect_device()
    device = device_info["device"]
    if device == "cuda":
        import torch as _torch
        _torch.cuda.empty_cache()
        _torch.cuda.reset_peak_memory_stats()

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
        from backtesting.ml.actor_critic import ActorCriticLSTM
        from backtesting.ml.ppo_trainer import PPOTrainer

        if args.backtest_only:
            # Skip training — load weights from --resume or default save path
            load_path = args.resume or args.save_model
            if not Path(load_path).exists():
                print(f"ERROR: --backtest-only requires a saved model at '{load_path}'.\n"
                      f"       Train first, or pass --resume <path>.", file=sys.stderr)
                sys.exit(1)
            print(f"  Skipping training — loading model from {load_path}\n", flush=True)
            trained_model = ActorCriticLSTM.load(load_path, device=device)
            elapsed = 0.0
        else:
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
                entropy_coef=args.entropy_coef,
                flat_penalty=args.flat_penalty,
                weight_decay=args.weight_decay,
                val_frac=args.val_frac,
                patience=args.patience,
                pretrained_path=args.resume,
                fixed_sl_pts=args.sl_pts,
                fixed_tp_pts=args.tp_pts,
                win_bonus=args.win_bonus,
                val_winrate_coef=args.val_winrate_coef,
                binary_reward=args.binary_reward,
                time_limit_bars=args.time_limit_bars,
                max_trades_per_episode=args.max_trades_per_episode,
            )
            rl_metrics = ppo_trainer.fit(train_bars, features)
            elapsed = time.time() - t0

            if model_path:
                ppo_trainer.save(model_path)

            print()
            _print_rl_train_section(rl_metrics, model_path, elapsed)

            trained_model = ppo_trainer._policy   # bare ActorCriticLSTM
            trained_model.eval()                  # disable dropout for deterministic inference

        # ── Model fingerprint — confirms weights changed between runs ──
        import torch as _torch
        _wsum = sum(float(p.detach().cpu().sum()) for p in trained_model.parameters())
        print(f"  Model fingerprint (weight sum): {_wsum:.6f}")
        print(f"  (This should differ across training runs — if identical, weights are not updating)\n")

        # ── 6a. NN backtest (direct evaluation — same logic as val) ───
        print("  Building test-set features …", flush=True)
        test_features = make_features(test_bars)

        env_kwargs = dict(
            seq_len=args.seq_len,
            n_features=22,
            multiplier=args.multiplier,
            commission=2.0,
            contracts=args.contracts,
            max_loss=args.max_loss,
            reward_scale=100.0,
            flat_penalty=args.flat_penalty,
            win_bonus=args.win_bonus,
            binary_reward=args.binary_reward,
            time_limit_bars=args.time_limit_bars,
            max_trades_per_episode=args.max_trades_per_episode,
        )
        test_episodes = _build_test_episodes(
            test_bars, test_features, args.seq_len, args.prediction_horizon
        )
        print(f"  {len(test_episodes)} test-set trading days", flush=True)
        print("  Running NN backtest on test set …", flush=True)
        analytics = _nn_backtest(trained_model, test_episodes, env_kwargs, device, args.cash,
                                  trade_log_path=args.trade_log,
                                  fixed_sl_pts=args.sl_pts,
                                  fixed_tp_pts=args.tp_pts)

    else:
        # ── Supervised walk-forward training ─────────────────────────
        from backtesting.data_feed import DataFeed
        from backtesting.engine import BacktestEngine
        from backtesting.ml.dataset import walk_forward_splits
        from backtesting.ml.trainer import Trainer
        from backtesting.portfolio import Portfolio, MarginSpec
        from strategies.lstm_signal import LSTMSignalStrategy

        print(f"  Training LSTM — {args.folds} walk-forward folds …\n", flush=True)
        trainer = Trainer(
            hidden_size=args.hidden,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            device=device,
            compile_model=args.compile_model,
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
        warmup    = args.seq_len + 60
        feed_test = DataFeed(test_bars, warmup_bars=warmup)
        strategy  = LSTMSignalStrategy(
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
        result   = BacktestEngine(feed_test, portfolio, [strategy], verbose=False).run()
        analytics = result.analytics

    # ── 7. Dashboard ──────────────────────────────────────────────────────
    print()
    b_start = test_bars[0].timestamp.strftime("%Y-%m-%d")
    b_end   = test_bars[-1].timestamp.strftime("%Y-%m-%d")
    _print_backtest_section(analytics, args.symbol, len(test_bars), b_start, b_end)
    print()

    # ── 8. Telegram notification ───────────────────────────────────────────
    if args.tg_token and args.tg_chat:
        total_elapsed = time.time() - t0
        mode_label = "PPO-RL" if args.rl else "Supervised"
        _notify_telegram(args.tg_token, args.tg_chat, analytics, total_elapsed, mode_label)
        print(f"  Notification sent to Telegram chat {args.tg_chat}")


if __name__ == "__main__":
    main()
