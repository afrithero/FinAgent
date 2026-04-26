import backtrader as bt
import pandas as pd
from typing import Any


def _log_order(strategy: bt.Strategy, order: bt.Order) -> None:
    """Append a completed order to strategy.trade_log."""
    if order.status == order.Completed:
        action = "BUY" if order.isbuy() else "SELL"
        strategy.trade_log.append({
            "date": strategy.datas[0].datetime.date(0).isoformat(),
            "action": action,
            "price": order.executed.price,
            "size": order.executed.size,
            "value": order.executed.value,
            "commission": order.executed.comm,
        })


class SmaCross(bt.Strategy):
    """SMA crossover strategy: buy on golden cross, sell on death cross."""

    params = (("fast", 5), ("slow", 20),)

    def __init__(self) -> None:
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        self.trade_log: list[dict[str, Any]] = []

    def next(self) -> None:
        if self.position.size == 0 and self.crossover > 0:
            self.buy()
        elif self.position.size > 0 and self.crossover < 0:
            self.close()

    def notify_order(self, order: bt.Order) -> None:
        _log_order(self, order)


class RSIStrategy(bt.Strategy):
    """RSI mean-reversion strategy: buy when oversold, sell when overbought."""

    params = (
        ("rsi_period", 14),
        ("rsi_upper", 70),
        ("rsi_lower", 30),
    )

    def __init__(self) -> None:
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.trade_log: list[dict[str, Any]] = []

    def next(self) -> None:
        if self.position.size == 0:
            if self.rsi < self.params.rsi_lower:
                self.buy()
        elif self.rsi > self.params.rsi_upper:
            self.close()

    def notify_order(self, order: bt.Order) -> None:
        _log_order(self, order)


class MomentumStrategy(bt.Strategy):
    """Momentum strategy: buy when recent return exceeds threshold, sell when it drops below."""

    params = (
        ("lookback_period", 20),
        ("threshold", 0.05),
    )

    def __init__(self) -> None:
        self.trade_log: list[dict[str, Any]] = []

    def next(self) -> None:
        if len(self.data) <= self.params.lookback_period:
            return
        momentum = (self.data.close[0] / self.data.close[-self.params.lookback_period]) - 1
        if self.position.size == 0:
            if momentum > self.params.threshold:
                self.buy()
        elif momentum < -self.params.threshold:
            self.close()

    def notify_order(self, order: bt.Order) -> None:
        _log_order(self, order)


STRATEGY_REGISTRY: dict[str, type[bt.Strategy]] = {
    "SmaCross": SmaCross,
    "RSIStrategy": RSIStrategy,
    "MomentumStrategy": MomentumStrategy,
}

DEFAULT_STRATEGY_PARAMS: dict[str, dict] = {
    "SmaCross": {"fast": 3, "slow": 37},
    "RSIStrategy": {"rsi_period": 14, "rsi_upper": 70, "rsi_lower": 30},
    "MomentumStrategy": {"lookback_period": 20, "threshold": 0.05},
}

# Market-specific broker defaults.
# Taiwan (tw): minimum lot = 1000 shares; commission ~0.003 (blended, accounts for
#              0.1425% brokerage + ~0.3% securities transaction tax on sell side).
# US (us):     stake = 10 shares; commission ~0.001.
MARKET_DEFAULTS: dict[str, dict] = {
    "tw": {"stake": 1000, "commission": 0.003,  "default_cash": 1_000_000},
    "us": {"stake":   10, "commission": 0.001,  "default_cash":    50_000},
}


class Backtester:
    def __init__(
        self,
        csv_path: str | None,
        strategy,
        cash: float = 10000,
        market: str = "us",
        data_df: pd.DataFrame | None = None,
        **kwargs,
    ):
        self.csv_path = csv_path
        self.strategy = strategy
        self.cash = cash
        self.market = market
        self.data_df = data_df
        self.kwargs = kwargs
        self.results = None
        self.strat = None
        self.cerebro = None

    def run(self):
        if self.data_df is not None:
            df = self.data_df.copy()
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            else:
                df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
        else:
            if not self.csv_path:
                raise ValueError("Either csv_path or data_df must be provided.")
            df = pd.read_csv(self.csv_path, parse_dates=True, index_col="Date")
        data = bt.feeds.PandasData(dataname=df,
                                   open='Open',
                                   high='High',
                                   low='Low',
                                   close='Close',
                                   volume='Volume',)
        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        cerebro.addstrategy(self.strategy, **self.kwargs)

        market_cfg = MARKET_DEFAULTS.get(self.market, MARKET_DEFAULTS["us"])
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=market_cfg["commission"])
        cerebro.addsizer(bt.sizers.FixedSize, stake=market_cfg["stake"])

        # analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        self.results = cerebro.run()
        self.strat = self.results[0]
        self.cerebro = cerebro

    def get_performance(self):
        if self.strat is None:
            raise RuntimeError("Must run() before getting performance.")

        return {
            "initial_cash": self.cash,
            "final_cash": self.cerebro.broker.getvalue(),
            "return_pct": self.strat.analyzers.returns.get_analysis().get("rtot", None),
            "max_drawdown_pct": self.strat.analyzers.drawdown.get_analysis()["max"]["drawdown"],
            "sharpe_ratio": self.strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        }

    def get_trades(self):
        if self.strat is None:
            raise RuntimeError("Must run() before getting trades.")
        return getattr(self.strat, "trade_log", [])

    def to_tool_result(self) -> dict[str, Any]:
        """Return a ToolResult-conformant dict for use by backtest_tool."""
        perf = self.get_performance()
        trades = self.get_trades()
        final_cash = perf["final_cash"]
        initial_cash = perf["initial_cash"]
        ret_pct = perf["return_pct"]
        sharpe = perf["sharpe_ratio"]
        ret_text = f"{ret_pct:.2%}" if isinstance(ret_pct, (int, float)) else "N/A"
        summary = (
            f"Backtest complete. "
            f"Initial={initial_cash:.2f}  Final={final_cash:.2f}  "
            f"Return={ret_text}  Sharpe={sharpe}  Trades={len(trades)}"
        )
        return {
            "status": "ok",
            "summary": summary,
            "data": {"performance": perf, "trades": trades},
            "debug_hint": None,
        }
