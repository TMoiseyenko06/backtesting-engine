"""
NQ LSTM Backtesting Dashboard
==============================
Interactive Streamlit dashboard for training the LSTM model and reviewing
backtest results — including a candlestick chart with ghost candles that show
what the algorithm predicted the market would do at each bar.

Run
---
    streamlit run dashboard.py

Optional CLI argument (pass after the Streamlit separator):
    streamlit run dashboard.py -- --data path/to/file.dbn.zstd
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NQ LSTM Backtesting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional default data path from CLI   streamlit run dashboard.py -- --data x
# ──────────────────────────────────────────────────────────────────────────────
def _cli_default_data() -> str:
    args = sys.argv[1:]
    if "--data" in args:
        idx = args.index("--data")
        if idx + 1 < len(args):
            return args[idx + 1]
    return "data/nq.dbn.zstd"


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 NQ LSTM Dashboard")
    st.markdown("---")

    data_path = st.text_input("Data file", _cli_default_data())
    cycles    = st.number_input("Training cycles", min_value=1, max_value=20, value=1, step=1,
                                help="Each cycle = full train + backtest on the same data")

    st.markdown("**Model hyperparameters**")
    c1, c2 = st.columns(2)
    hidden     = c1.number_input("Hidden", 32, 512, 64,  step=32)
    seq_len    = c1.number_input("Seq len", 10, 120, 30,  step=5)
    folds      = c1.number_input("Folds",   1,  10,   3)
    epochs     = c2.number_input("Epochs",  5,  200, 50,  step=5)
    batch_size = c2.number_input("Batch",   64, 1024, 256, step=64)
    conf       = c2.number_input("Min conf", 0.10, 0.99, 0.50, step=0.05)

    st.markdown("**Portfolio**")
    cash         = st.number_input("Capital ($)",    100_000, 10_000_000, 500_000, step=50_000)
    contracts    = st.number_input("Contracts/trade", 1, 20, 1)
    multiplier   = st.number_input("Multiplier ($/pt)", 1.0, 100.0, 20.0)
    init_margin  = st.number_input("Init margin ($)",  1_000, 100_000, 21_000, step=1_000)
    maint_margin = st.number_input("Maint margin ($)", 1_000, 100_000, 19_000, step=1_000)
    symbol       = st.text_input("Symbol", "NQ")

    st.markdown("**Chart**")
    TF_MAP = {
        "1 min": "1min", "5 min": "5min", "15 min": "15min",
        "1 hour": "1h",  "4 hour": "4h",  "1 day": "1D",
    }
    tf_label  = st.selectbox("Timeframe", list(TF_MAP.keys()), index=3)
    timeframe = TF_MAP[tf_label]

    st.markdown("---")
    start_btn = st.button("▶  Start Training", type="primary", width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# State initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "cycle_results" not in st.session_state:
    st.session_state.cycle_results: list = []
if "bars"        not in st.session_state:
    st.session_state.bars        = None
if "split"       not in st.session_state:
    st.session_state.split       = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ──────────────────────────────────────────────────────────────────────────────
# Recording strategy — extends LSTMSignalStrategy to log every prediction
# ──────────────────────────────────────────────────────────────────────────────
def _make_recording_strategy(model, device, sym, seq, conf_thr, ctrs):
    """Return a RecordingLSTMStrategy (defined here to avoid a new module file)."""
    from strategies.lstm_signal import LSTMSignalStrategy

    class RecordingLSTMStrategy(LSTMSignalStrategy):
        """Identical to LSTMSignalStrategy but appends every inference to
        self.predictions as (timestamp, signal, confidence, bar_close)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.predictions: List[Tuple] = []

        def on_bar(self, bar):
            import torch
            from backtesting.ml.features import make_features

            if self.bars_available < self._min_bars:
                return

            hist     = self.history(self._min_bars)
            features = make_features(hist)
            x        = features[-self._seq_len:]
            x_t      = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
            signal, confidence = self._model.predict(x_t)

            # Record regardless of confidence so ghost candles show all guesses
            self.predictions.append((bar.timestamp, int(signal), float(confidence), float(bar.close)))

            if confidence < self._conf_thresh:
                return

            pos = self.position(self._symbol)
            if signal == 2 and pos <= 0:        # BUY
                if pos < 0:
                    self.close_position(self._symbol, tag="flip_long")
                self.buy(self._symbol, self._contracts, tag="lstm_long")
            elif signal == 0 and pos >= 0:      # SELL
                if pos > 0:
                    self.close_position(self._symbol, tag="flip_short")
                self.sell(self._symbol, self._contracts, tag="lstm_short")
            elif signal == 1 and pos != 0:      # FLAT
                self.close_position(self._symbol, tag="lstm_flat")

    return RecordingLSTMStrategy(
        model=model,
        device=device,
        symbol=sym,
        seq_len=seq,
        confidence_threshold=conf_thr,
        contracts=ctrs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (cached to avoid re-reading the .dbn on every Streamlit rerun)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading bar data …")
def load_bars(path: str, sym: str, mult: float):
    from backtesting.loaders.databento import from_databento_file
    feed = from_databento_file(path=path, symbol=sym, contract_multiplier=mult)
    return feed._bars


def compute_split(bars, days: int = 365):
    cutoff = bars[-1].timestamp - timedelta(days=days)
    return next(i for i, b in enumerate(bars) if b.timestamp >= cutoff)


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────
def _bars_to_df(bars) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [b.timestamp for b in bars],
            "open":      [b.open      for b in bars],
            "high":      [b.high      for b in bars],
            "low":       [b.low       for b in bars],
            "close":     [b.close     for b in bars],
            "volume":    [b.volume    for b in bars],
        }
    ).set_index("timestamp")


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = tf
    agg = df.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return agg


def _compute_atr(agg: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple ATR on aggregated OHLCV."""
    high, low, prev_close = agg["high"], agg["low"], agg["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _build_ghost_df(
    predictions: list,
    agg: pd.DataFrame,
    tf: str,
    conf_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw 1-min predictions to ghost candle OHLC DataFrames,
    one for BUY signals and one for SELL signals.

    Each ghost candle is placed at the period AFTER the prediction bar so it
    represents "what the model thinks the next bar will look like."
    """
    if not predictions or agg.empty:
        empty = pd.DataFrame(columns=["open", "high", "low", "close"])
        return empty, empty

    atr = _compute_atr(agg)
    freq = pd.tseries.frequencies.to_offset(tf)

    # Map each prediction to its aggregated period index
    pred_df = pd.DataFrame(predictions, columns=["ts", "signal", "conf", "close"])
    pred_df = pred_df[pred_df["conf"] >= conf_threshold]
    pred_df = pred_df[pred_df["signal"] != 1]  # drop FLAT

    if pred_df.empty:
        empty = pd.DataFrame(columns=["open", "high", "low", "close"])
        return empty, empty

    pred_df["ts"] = pd.to_datetime(pred_df["ts"])
    pred_df["period"] = pred_df["ts"].dt.floor(tf)

    # For each period, aggregate: weighted-average direction and last close
    grouped = (
        pred_df.groupby("period")
        .agg(
            net_signal=("signal", lambda s: (s == 2).sum() - (s == 0).sum()),
            avg_conf=("conf", "mean"),
            last_close=("close", "last"),
        )
        .reset_index()
    )

    ghost_rows_buy  = []
    ghost_rows_sell = []

    for _, row in grouped.iterrows():
        period   = row["period"]
        net_sig  = row["net_signal"]
        avg_conf = row["avg_conf"]
        ref_close = row["last_close"]

        next_period = period + freq
        if next_period not in agg.index:
            continue

        atr_val = float(atr.get(next_period, atr.iloc[-1]))
        scale   = avg_conf * atr_val          # ghost candle body height

        if net_sig > 0:                        # bullish ghost
            g_open  = ref_close
            g_close = ref_close + scale
            g_high  = g_close  + 0.2 * atr_val
            g_low   = g_open   - 0.1 * atr_val
            ghost_rows_buy.append(
                {"timestamp": next_period, "open": g_open, "high": g_high, "low": g_low, "close": g_close}
            )
        elif net_sig < 0:                      # bearish ghost
            g_open  = ref_close
            g_close = ref_close - scale
            g_high  = g_open   + 0.1 * atr_val
            g_low   = g_close  - 0.2 * atr_val
            ghost_rows_sell.append(
                {"timestamp": next_period, "open": g_open, "high": g_high, "low": g_low, "close": g_close}
            )

    def _to_df(rows):
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        return pd.DataFrame(rows).set_index("timestamp")

    return _to_df(ghost_rows_buy), _to_df(ghost_rows_sell)


def _build_chart(
    test_bars,
    predictions: list,
    equity_curve: list,
    trades: list,
    timeframe: str,
    conf_threshold: float,
    cycle_num: int,
) -> go.Figure:
    """Build the full 2-row Plotly figure: candlestick + ghost candles / equity curve."""

    raw_df  = _bars_to_df(test_bars)
    agg     = _resample(raw_df, timeframe)
    buy_gh, sell_gh = _build_ghost_df(predictions, agg, timeframe, conf_threshold)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.03,
        subplot_titles=(
            f"Cycle {cycle_num} — Actual price + ghost candles ({tf_label})",
            "Equity curve",
        ),
    )

    # ── Actual candles ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=agg.index,
            open=agg["open"],  high=agg["high"],
            low=agg["low"],    close=agg["close"],
            name="Actual",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            line_width=1,
        ),
        row=1, col=1,
    )

    # ── BUY ghost candles (bullish — green, transparent) ────────────────────
    if not buy_gh.empty:
        fig.add_trace(
            go.Candlestick(
                x=buy_gh.index,
                open=buy_gh["open"],   high=buy_gh["high"],
                low=buy_gh["low"],     close=buy_gh["close"],
                name="Predicted ↑",
                increasing_line_color="rgba(0,220,130,0.55)",
                decreasing_line_color="rgba(0,220,130,0.55)",
                increasing_fillcolor="rgba(0,220,130,0.30)",
                decreasing_fillcolor="rgba(0,220,130,0.30)",
                line_width=1,
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ── SELL ghost candles (bearish — red, transparent) ─────────────────────
    if not sell_gh.empty:
        fig.add_trace(
            go.Candlestick(
                x=sell_gh.index,
                open=sell_gh["open"],   high=sell_gh["high"],
                low=sell_gh["low"],     close=sell_gh["close"],
                name="Predicted ↓",
                increasing_line_color="rgba(255,80,80,0.55)",
                decreasing_line_color="rgba(255,80,80,0.55)",
                increasing_fillcolor="rgba(255,80,80,0.30)",
                decreasing_fillcolor="rgba(255,80,80,0.30)",
                line_width=1,
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ── Trade entry / exit markers ───────────────────────────────────────────
    if trades:
        long_entries  = [(t.open_time,  t.open_price)  for t in trades if t.quantity > 0]
        short_entries = [(t.open_time,  t.open_price)  for t in trades if t.quantity < 0]
        exits         = [(t.close_time, t.close_price) for t in trades]

        for times, prices, sym_shape, color, lbl in [
            (long_entries,  [p for _, p in long_entries],  "triangle-up",   "#00e676", "Buy entry"),
            (short_entries, [p for _, p in short_entries], "triangle-down", "#ff1744", "Sell entry"),
            (exits,         [p for _, p in exits],         "x",             "#ffea00", "Exit"),
        ]:
            if times:
                fig.add_trace(
                    go.Scatter(
                        x=[t for t, _ in times], y=prices,
                        mode="markers",
                        marker=dict(symbol=sym_shape, size=8, color=color,
                                    line=dict(width=1, color="black")),
                        name=lbl,
                    ),
                    row=1, col=1,
                )

    # ── Equity curve ────────────────────────────────────────────────────────
    if equity_curve:
        eq_times  = [t for t, _ in equity_curve]
        eq_values = [v for _, v in equity_curve]
        fig.add_trace(
            go.Scatter(
                x=eq_times, y=eq_values,
                mode="lines",
                name="Equity",
                line=dict(color="#42a5f5", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(66,165,245,0.10)",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        height=780,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=11),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#1f2630", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1f2630", row=2, col=1)
    fig.update_xaxes(showgrid=False)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Metric display helpers
# ──────────────────────────────────────────────────────────────────────────────
def _metric_colour(val: float) -> str:
    return "green" if val >= 0 else "red"


def _show_metrics(analytics, test_bars) -> None:
    a = analytics
    b_start = test_bars[0].timestamp.strftime("%Y-%m-%d")
    b_end   = test_bars[-1].timestamp.strftime("%Y-%m-%d")

    st.caption(f"Backtest period: **{b_start}** → **{b_end}**  ({len(test_bars):,} bars)")
    st.markdown("---")

    cols = st.columns(4)
    cols[0].metric("Total Return",
                   f"${a.total_return:,.0f}",
                   f"{a.total_return_pct:+.2f}%")
    cols[1].metric("Final Equity",  f"${a.final_equity:,.0f}")
    cols[2].metric("Max Drawdown",  f"${a.max_drawdown:,.0f}",
                   f"{a.max_drawdown_pct:.2f}%", delta_color="inverse")
    cols[3].metric("Sharpe Ratio",  f"{a.sharpe_ratio:.3f}")

    cols2 = st.columns(4)
    cols2[0].metric("Total Trades", f"{a.total_trades:,}")
    cols2[1].metric("Win Rate",     f"{a.win_rate:.1f}%")
    cols2[2].metric("Profit Factor",f"{a.profit_factor:.3f}")
    cols2[3].metric("Sortino",      f"{a.sortino_ratio:.3f}")

    with st.expander("Full metrics"):
        st.json(a.to_dict())


# ──────────────────────────────────────────────────────────────────────────────
# Core cycle runner
# ──────────────────────────────────────────────────────────────────────────────
def run_cycle(
    cycle_num: int,
    train_bars,
    test_bars,
    device_info: dict,
    args: dict,
    log_placeholder,
    progress_placeholder,
) -> dict:
    """Train one cycle, backtest, return results dict."""
    import torch
    from backtesting.data_feed import DataFeed
    from backtesting.engine import BacktestEngine
    from backtesting.ml.dataset import make_labels, walk_forward_splits
    from backtesting.ml.features import make_features
    from backtesting.ml.trainer import Trainer
    from backtesting.portfolio import Portfolio, MarginSpec

    device = device_info["device"]
    log    = []

    def _log(msg: str):
        log.append(msg)
        log_placeholder.code("\n".join(log[-60:]))   # keep last 60 lines

    _log(f"── Cycle {cycle_num}  ·  training on {len(train_bars):,} bars ──")

    # Feature engineering
    _log("Building features & labels …")
    features = make_features(train_bars)
    labels   = make_labels(train_bars)
    counts   = np.bincount(labels, minlength=3)
    _log(f"Features shape : {features.shape}")
    _log(f"Label balance  : Sell={counts[0]:,}  Flat={counts[1]:,}  Buy={counts[2]:,}")

    # Walk-forward training with stdout captured → log
    trainer = Trainer(
        hidden_size=args["hidden"],
        seq_len=args["seq_len"],
        batch_size=args["batch_size"],
        max_epochs=args["epochs"],
        device=device,
    )
    splits = walk_forward_splits(len(train_bars), n_splits=args["folds"])
    if not splits:
        st.error("Not enough training data for walk-forward splits.")
        return {}

    fold_results = []
    t0 = time.time()
    for i, (tr_idx, val_idx) in enumerate(splits):
        _log(f"\n  Fold {i+1}/{args['folds']} …")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = trainer.fit(features, labels, tr_idx, val_idx, fold=i)
        for line in buf.getvalue().splitlines():
            _log(line)
        fold_results.append(result)
        progress_placeholder.progress(
            (cycle_num - 1) / args["cycles"] + (i + 1) / args["folds"] / args["cycles"],
            text=f"Cycle {cycle_num}/{args['cycles']}  ·  fold {i+1}/{args['folds']}",
        )

    elapsed = time.time() - t0
    mean_val = float(np.mean([r["val_acc"] for r in fold_results]))
    best_val = float(max(r["val_acc"] for r in fold_results))
    _log(f"\nTraining done in {elapsed:.1f}s — mean val_acc={mean_val:.4f}  best={best_val:.4f}")

    # Backtest
    _log("Running backtest …")
    warmup   = args["seq_len"] + 60
    feed     = DataFeed(test_bars, warmup_bars=warmup)
    strategy = _make_recording_strategy(
        model=trainer.model,
        device=device,
        sym=args["symbol"],
        seq=args["seq_len"],
        conf_thr=args["conf"],
        ctrs=args["contracts"],
    )
    portfolio = Portfolio(
        initial_cash=args["cash"],
        margin_specs={
            args["symbol"]: MarginSpec(
                args["symbol"],
                args["init_margin"],
                args["maint_margin"],
                args["multiplier"],
            )
        },
        commission_per_contract=2.0,
        slippage_ticks=1,
        tick_size=0.25,
    )
    bt_result = BacktestEngine(feed, portfolio, [strategy], verbose=False).run()
    _log(f"Backtest complete — {bt_result.bar_count:,} bars processed.")

    return {
        "cycle":        cycle_num,
        "fold_results": fold_results,
        "elapsed":      elapsed,
        "mean_val_acc": mean_val,
        "best_val_acc": best_val,
        "analytics":    bt_result.analytics,
        "predictions":  strategy.predictions,
        "trades":       bt_result.portfolio.all_trades,
        "equity_curve": bt_result.analytics.equity_curve,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main rendering
# ──────────────────────────────────────────────────────────────────────────────
st.title("NQ LSTM — Backtesting Dashboard")

if start_btn:
    st.session_state.cycle_results = []

    # Detect device
    try:
        import torch
        if torch.cuda.is_available():
            device_info = {"device": "cuda", "name": torch.cuda.get_device_name(0)}
        elif torch.backends.mps.is_available():
            device_info = {"device": "mps",  "name": "Apple MPS"}
        else:
            device_info = {"device": "cpu",  "name": "CPU"}
    except ImportError:
        st.error("PyTorch is not installed.")
        st.stop()

    st.info(f"Device: **{device_info['name']}**", icon="🖥️")

    # Load data (cached)
    try:
        all_bars = load_bars(data_path, symbol, multiplier)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    split      = compute_split(all_bars)
    train_bars = all_bars[:split]
    test_bars  = all_bars[split:]

    st.success(
        f"Loaded **{len(all_bars):,}** bars.  "
        f"Train: **{len(train_bars):,}**  ·  Test (last 1 yr): **{len(test_bars):,}**"
    )

    args = dict(
        hidden=int(hidden), seq_len=int(seq_len), folds=int(folds),
        epochs=int(epochs), batch_size=int(batch_size), conf=float(conf),
        cash=float(cash), contracts=float(contracts), multiplier=float(multiplier),
        init_margin=float(init_margin), maint_margin=float(maint_margin),
        symbol=symbol, cycles=int(cycles),
    )

    overall_progress = st.progress(0.0, text="Starting …")

    for cycle_num in range(1, int(cycles) + 1):
        st.markdown(f"## Cycle {cycle_num} / {int(cycles)}")
        log_area  = st.empty()
        result    = run_cycle(
            cycle_num=cycle_num,
            train_bars=train_bars,
            test_bars=test_bars,
            device_info=device_info,
            args=args,
            log_placeholder=log_area,
            progress_placeholder=overall_progress,
        )
        if not result:
            continue
        st.session_state.cycle_results.append(result)
        overall_progress.progress(cycle_num / int(cycles), text=f"Cycle {cycle_num} complete")

    overall_progress.empty()


# ──────────────────────────────────────────────────────────────────────────────
# Display stored results (persists across Streamlit reruns)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.cycle_results:
    results = st.session_state.cycle_results

    # All-cycle equity comparison
    if len(results) > 1:
        st.markdown("## All Cycles — Equity Comparison")
        fig_all = go.Figure()
        for r in results:
            if r.get("equity_curve"):
                ts, eq = zip(*r["equity_curve"])
                fig_all.add_trace(go.Scatter(
                    x=list(ts), y=list(eq),
                    mode="lines", name=f"Cycle {r['cycle']}",
                ))
        fig_all.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_all, width="stretch")

    # Per-cycle tabs
    tab_labels = [f"Cycle {r['cycle']}" for r in results]
    tabs       = st.tabs(tab_labels)

    for tab, result in zip(tabs, results):
        with tab:
            # Training fold summary
            with st.expander("Training fold details", expanded=False):
                fold_df = pd.DataFrame(result["fold_results"])
                fold_df.index = fold_df.index + 1
                fold_df.index.name = "Fold"
                st.dataframe(fold_df.style.format("{:.4f}"), width="stretch")
            st.caption(
                f"Mean val_acc: **{result['mean_val_acc']:.4f}**  "
                f"Best val_acc: **{result['best_val_acc']:.4f}**  "
                f"Train time: {result['elapsed']:.1f}s"
            )

            # Need the original test bars — re-derive them from session cache
            try:
                all_bars_cached = load_bars(data_path, symbol, multiplier)
                split_cached    = compute_split(all_bars_cached)
                test_bars_cached = all_bars_cached[split_cached:]
            except Exception:
                test_bars_cached = []

            # Chart
            if test_bars_cached:
                fig = _build_chart(
                    test_bars=test_bars_cached,
                    predictions=result["predictions"],
                    equity_curve=result["equity_curve"],
                    trades=result["trades"],
                    timeframe=timeframe,
                    conf_threshold=float(conf),
                    cycle_num=result["cycle"],
                )
                st.plotly_chart(fig, width="stretch")

                ghost_buy  = sum(1 for _, s, c, _ in result["predictions"]
                                 if s == 2 and c >= float(conf))
                ghost_sell = sum(1 for _, s, c, _ in result["predictions"]
                                 if s == 0 and c >= float(conf))
                st.caption(
                    f"Ghost candles: **{ghost_buy}** bullish  ·  **{ghost_sell}** bearish  "
                    f"(conf ≥ {conf:.0%})"
                )

            # Performance metrics
            st.markdown("### Performance Metrics")
            _show_metrics(result["analytics"], test_bars_cached or [])
