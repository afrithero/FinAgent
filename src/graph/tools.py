from langchain_core.tools import tool
from stock.trader import Backtester, SmaCross

@tool(
    "backtest",
    description=(
        "Run SMA cross backtest on stock data. "
        "If the user asks about AAPL, use csv_path='../data/us_stock/AAPL.csv' by default."
    ),
)
def backtest_tool(
    csv_path: str = "../data/us_stock/AAPL.csv", 
    cash: float = 50000,
    fast: int = 3,
    slow: int = 37,
):
    bt_runner = Backtester(csv_path, strategy=SmaCross, cash=cash, fast=fast, slow=slow)
    bt_runner.run()
    return {"performance": bt_runner.get_performance(), "trades": bt_runner.get_trades()}

def create_retriever_tool(retriever):
    @tool(
        "retrieve_financial_docs",
        description=(
            "Retrieve 1-3 relevant financial context snippets for the given query. "
            "Use this FIRST when the user asks for stock/company outlook or market potential."
        ),
    )
    def retriever_tool(query: str):
        return retriever.retrieve(query)[:3] 
    return retriever_tool

