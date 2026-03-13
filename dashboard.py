"""
Dashboard — Interactive Training & Backtesting UI
===================================================
Run:
    python dashboard.py

Then open http://localhost:8000 in your browser.

Features
--------
* Start / stop model training with configurable epochs and interval
* Live training log streamed via WebSocket
* Run a hold-out backtest and view results
* Interactive candlestick chart with trade entry/exit markers
* Equity curve chart
* Detailed trade table
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="NQ Futures ML Dashboard")


# ---------------------------------------------------------------------------
# Shared training state
# ---------------------------------------------------------------------------

class _TrainState:
    def __init__(self) -> None:
        self.running: bool = False
        self.proc: Optional[object] = None          # subprocess.Popen
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def reset_and_start(self) -> None:
        with self._lock:
            self.lines = []
            self.running = True

    def add(self, text: str) -> None:
        with self._lock:
            self.lines.append(text)

    def snapshot(self, from_idx: int) -> tuple[list[str], int]:
        with self._lock:
            chunk = self.lines[from_idx:]
            return chunk, len(self.lines)

    def stop(self) -> None:
        import subprocess
        if self.proc and isinstance(self.proc, subprocess.Popen):
            if self.proc.poll() is None:
                self.proc.terminate()
        self.running = False


_train = _TrainState()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class TrainParams(BaseModel):
    interval: str = "1m"
    epochs: int = 50
    walk_forward: bool = True
    refresh: bool = False


class BacktestParams(BaseModel):
    interval: str = "1m"


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = ROOT / "dashboard" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status() -> dict:
    intervals = ["1m", "5m", "15m", "1h"]
    return {
        "training_running": _train.running,
        "models": {
            iv: (ROOT / "models" / f"nn_ict_{iv}.pt").exists()
            for iv in intervals
        },
        "data": {
            iv: (ROOT / "data" / f"NQF_{iv}.parquet").exists()
            for iv in intervals
        },
    }


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

@app.post("/api/train/start")
async def api_train_start(params: TrainParams) -> dict:
    if _train.running:
        return {"error": "Training already running"}

    _train.reset_and_start()

    cmd = [
        sys.executable,
        str(ROOT / "train_nn.py"),
        "--interval", params.interval,
        "--epochs",   str(params.epochs),
    ]
    if not params.walk_forward:
        cmd.append("--no-wf")
    if params.refresh:
        cmd.append("--refresh")

    def _run() -> None:
        import subprocess
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
                bufsize=1,
            )
            _train.proc = proc
            for raw in proc.stdout:
                _train.add(raw.rstrip("\n"))
            proc.wait()
            code = proc.returncode
            _train.add(
                f"\n{'✔ Training completed' if code == 0 else f'✖ Process exited (code {code})'}"
            )
        except Exception as exc:
            _train.add(f"✖ Error: {exc}")
        finally:
            _train.running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.post("/api/train/stop")
async def api_train_stop() -> dict:
    _train.stop()
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# WebSocket — live training log
# ---------------------------------------------------------------------------

@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket) -> None:
    await ws.accept()
    cursor = 0
    try:
        while True:
            chunk, cursor = _train.snapshot(cursor)
            for line in chunk:
                await ws.send_text(json.dumps({"type": "line", "text": line}))
            # Signal idle when training finished
            if not _train.running and chunk == []:
                await ws.send_text(json.dumps({"type": "idle"}))
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Backtest endpoint
# ---------------------------------------------------------------------------

@app.post("/api/backtest/run")
async def api_backtest_run(params: BacktestParams) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_backtest, params.interval)


def _run_backtest(interval: str) -> dict:
    """Run hold-out backtest synchronously; returns OHLCV + trades as dicts."""
    import numpy as np

    from backtesting.data_feed import DataFeed
    from backtesting.engine import BacktestEngine
    from backtesting.loaders.cache import load_feed_from_cache
    from backtesting.portfolio import Portfolio, MarginSpec
    from strategies.nn_ict_strategy import NNICTStrategy
    from strategies.prop_firm_risk import PropFirmRisk

    SYMBOL        = "NQ=F"
    MULTIPLIER    = 20.0
    CASH          = 50_000.0
    MAX_CONTRACTS = 1
    MAX_DAILY_LOSS    = 1_500.0
    MAX_DRAWDOWN_USD  = 2_500.0
    INIT_MARGIN   = 12_000.0
    MAINT_MARGIN  = 10_000.0
    TICK_SIZE     = 0.25
    SEQ_LEN       = 60

    cache_path = ROOT / "data" / f"{SYMBOL.replace('=', '')}_{interval}.parquet"
    model_path = ROOT / "models" / f"nn_ict_{interval}.pt"

    if not cache_path.exists():
        return {"error": f"No cached data for {interval}. Train first (it downloads the data)."}
    if not model_path.exists():
        return {"error": f"No trained model for {interval}. Run training first."}

    feed_obj = load_feed_from_cache(cache_path, symbol=SYMBOL,
                                    contract_multiplier=MULTIPLIER)
    bars = feed_obj._bars
    n    = len(bars)

    prop_rules = PropFirmRisk(
        max_contracts=MAX_CONTRACTS,
        max_daily_loss_usd=MAX_DAILY_LOSS,
        max_drawdown_usd=MAX_DRAWDOWN_USD,
        initial_equity=CASH,
    )
    strategy = NNICTStrategy(
        model_path=model_path,
        seq_len=SEQ_LEN,
        min_confidence=0.40,
        contracts=MAX_CONTRACTS,
        prop_rules=prop_rules,
    )

    split_idx = int(n * 0.8)
    ctx_start = max(0, split_idx - NNICTStrategy._MIN_BUFFER)
    test_bars = bars[ctx_start:]
    n_warmup  = (split_idx - ctx_start) + SEQ_LEN

    if len(test_bars) - n_warmup < 50:
        return {"error": "Not enough hold-out bars for a meaningful backtest."}

    data_feed = DataFeed(test_bars, warmup_bars=n_warmup)
    portfolio = Portfolio(
        initial_cash=CASH,
        margin_specs={SYMBOL: MarginSpec(SYMBOL, INIT_MARGIN, MAINT_MARGIN, MULTIPLIER)},
        commission_per_contract=2.0,
        slippage_ticks=1,
        tick_size=TICK_SIZE,
    )
    result   = BacktestEngine(data_feed, portfolio, [strategy], verbose=False).run()
    a        = result.analytics

    # OHLCV bars for the chart (hold-out only, after warmup)
    chart_bars = test_bars[n_warmup:]
    ohlcv = [
        {
            "time":   int(b.timestamp.timestamp()),
            "open":   round(b.open,  2),
            "high":   round(b.high,  2),
            "low":    round(b.low,   2),
            "close":  round(b.close, 2),
            "volume": int(b.volume),
        }
        for b in chart_bars
    ]

    # Completed trades
    trades = [
        {
            "entry_time":  int(t.entry_time.timestamp()),
            "exit_time":   int(t.exit_time.timestamp()),
            "entry_label": t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "exit_label":  t.exit_time.strftime("%Y-%m-%d %H:%M"),
            "side":        t.side,
            "qty":         int(t.quantity),
            "entry_price": round(t.entry_price, 2),
            "exit_price":  round(t.exit_price,  2),
            "gross_pnl":   round(t.gross_pnl,   2),
            "net_pnl":     round(t.net_pnl,     2),
            "duration_s":  round((t.exit_time - t.entry_time).total_seconds(), 1),
        }
        for t in portfolio.all_trades
    ]

    # Equity curve (sampled to ≤2 000 points for browser performance)
    curve = portfolio.equity_curve
    step  = max(1, len(curve) // 2000)
    equity_curve = [
        {"time": int(s.timestamp.timestamp()), "value": round(s.equity, 2)}
        for s in curve[::step]
    ]

    pf = a.profit_factor
    return {
        "analytics": {
            "total_return_pct": round(a.total_return_pct, 2),
            "sharpe_ratio":     round(a.sharpe_ratio,     3),
            "max_drawdown_pct": round(a.max_drawdown_pct, 2),
            "total_trades":     a.total_trades,
            "win_rate":         round(a.win_rate,         1),
            "profit_factor":    round(pf, 3) if pf != float("inf") else 999.0,
            "avg_win":          round(a.avg_win,  2),
            "avg_loss":         round(a.avg_loss, 2),
            "expectancy":       round(a.expectancy, 2),
        },
        "ohlcv":        ohlcv,
        "trades":       trades,
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Dashboard -> http://localhost:8000")
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8000, reload=False)
