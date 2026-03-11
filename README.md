# Futures Backtesting Engine

A production-quality Python backtesting engine for futures trading with **zero lookahead bias** and **zero survivorship bias**.

## Architecture

```
backtesting/
  data_feed.py       — Bar type + strict chronological iterator
  order.py           — Order dataclass (MARKET / LIMIT / STOP / STOP_LIMIT)
  order_manager.py   — Order book with next-bar fill logic
  position.py        — Position + Trade records
  portfolio.py       — Cash, margin, equity curve
  engine.py          — Central event loop
  analytics.py       — Sharpe, drawdown, win-rate, etc.
  strategy.py        — Strategy base class

strategies/
  ema_crossover.py        — Dual-EMA crossover
  rsi_mean_reversion.py   — RSI overbought/oversold
  breakout.py             — Donchian channel breakout
  trend_following.py      — EMA trend + ATR sizing
```

## Zero Lookahead Guarantees

| Rule | Enforcement |
|------|-------------|
| Market orders fill at **next bar's open** | `OrderManager.process_bar()` fills pending orders first, before `on_bar()` |
| Limit/stop orders use the **bar's own range** | `_check_limit_stop()` uses `bar.low/high` of the bar being processed |
| `history(n)` only returns **closed bars** | `DataFeed._cursor` is advanced by the engine; strategies read `_cursor` via public API only |
| DataFeed rejects **non-chronological** data | Constructor validates strict ascending timestamps |
| Strategies **cannot advance** the feed | `DataFeed._advance()` is package-private; strategies call `current_bar()` / `history()` only |
| Slippage applied **after** bar closes | Portfolio applies slippage inside `execute_fill()`, not visible to strategy |

## Quick Start

```python
from backtesting.data_feed import DataFeed
from backtesting.engine import BacktestEngine
from backtesting.portfolio import Portfolio, MarginSpec
from strategies.ema_crossover import EMACrossoverStrategy

# Load data
feed = DataFeed.from_csv("data/ES_daily.csv", symbol="ES", contract_multiplier=50.0)

# Configure portfolio
portfolio = Portfolio(
    initial_cash=500_000,
    margin_specs={
        "ES": MarginSpec("ES", initial_margin_per_contract=12_000,
                         maintenance_margin_per_contract=10_900,
                         contract_multiplier=50.0)
    },
    commission_per_contract=2.0,
    slippage_ticks=1,
    tick_size=0.25,
)

# Run
strategy = EMACrossoverStrategy(fast_period=9, slow_period=21)
engine = BacktestEngine(feed, portfolio, [strategy])
result = engine.run()
result.print_summary()
```

## Writing Your Own Strategy

```python
from backtesting.strategy import Strategy
from backtesting.data_feed import Bar
from backtesting.order import OrderType

class MyStrategy(Strategy):
    def on_start(self):
        self.period = 20      # set parameters here

    def on_bar(self, bar: Bar) -> None:
        if self.bars_available < self.period:
            return            # not enough history yet

        # Safe — all closed bars, zero lookahead
        hist = self.history(self.period)
        closes = [b.close for b in hist]
        sma = self.sma(closes, self.period)

        pos = self.position(bar.symbol)

        if bar.close > sma and pos == 0:
            self.buy(bar.symbol, quantity=1)         # fills next bar open

        elif bar.close < sma and pos > 0:
            self.close_position(bar.symbol)          # fills next bar open

    def on_fill(self, order):
        print(f"Filled: {order}")
```

### Available Strategy Methods

| Method | Description |
|--------|-------------|
| `self.history(n)` | Last n closed bars (oldest first) |
| `self.current_bar()` | Most recent closed bar |
| `self.bars_available` | Count of closed bars |
| `self.buy(symbol, qty, ...)` | Submit buy order |
| `self.sell(symbol, qty, ...)` | Submit sell order |
| `self.close_position(symbol)` | Flatten position |
| `self.cancel_order(id)` | Cancel specific order |
| `self.cancel_all(symbol)` | Cancel all orders |
| `self.position(symbol)` | Signed position size |
| `self.cash()` | Available cash |
| `self.equity()` | Mark-to-market equity |
| `self.sma(values, period)` | Simple moving average |
| `self.ema(values, period)` | Exponential moving average |
| `self.atr(bars, period)` | Average True Range |
| `self.rsi(values, period)` | RSI (0–100) |

### Order Types

```python
from backtesting.order import OrderType

self.buy(symbol, 1, order_type=OrderType.MARKET)          # default
self.buy(symbol, 1, order_type=OrderType.LIMIT, limit_price=4500.0)
self.sell(symbol, 1, order_type=OrderType.STOP, stop_price=4400.0)
self.sell(symbol, 1, order_type=OrderType.STOP_LIMIT,
          stop_price=4400.0, limit_price=4395.0)
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Running the Example

```bash
python example.py
```
