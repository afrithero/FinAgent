import backtrader as bt
import pandas as pd

class SmaCross(bt.Strategy):
    params = (("fast",5), ("slow", 20),)
    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        self.trade_log = []

    def next(self):
        if self.position.size == 0:
            if self.crossover > 0:
                self.buy()

        if self.position.size > 0:
            if self.crossover < 0:
                self.close()
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self.trade_log.append({
                "date": self.datas[0].datetime.date(0).isoformat(),
                "action": action,
                "price": order.executed.price,
                "size": order.executed.size,
                "value": order.executed.value,
                "commission": order.executed.comm
            })

class Backtester:
    def __init__(self, csv_path: str, strategy, cash: float = 10000, **kwargs):
        self.csv_path = csv_path
        self.strategy = strategy
        self.cash = cash
        self.kwargs = kwargs
        self.results = None
        self.strat = None
        self.cerebro = None

    def run(self):
        df = pd.read_csv(self.csv_path, parse_dates=True, index_col="Date")
        data = bt.feeds.PandasData(dataname=df,
                                   open='Open',
                                   high='High',
                                   low='Low',
                                   close='Close',
                                   volume='Volume',)
        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # 用傳進來的 strategy
        cerebro.addstrategy(self.strategy, **self.kwargs)

        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)

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