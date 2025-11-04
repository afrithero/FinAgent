from langchain_core.tools import tool
from stock.trader import Backtester, SmaCross
import httpx

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

@tool(
    "search_stock_info",
    description=(
        "Search for stock outlook, predictions, or market potential using the SerpAPI FastAPI endpoint."
        "Returns summarized search results from Google."
    ),
)
def search_stock_info(query: str, num_results: int = 3, hl: str = "en", gl: str = "us"):
    BASE_URL = "http://localhost:8000" 
    with httpx.Client(timeout=20.0) as client:
        params = {
            "query": query,
            "num_results": num_results,
            "hl": hl,
            "gl": gl
        }
        resp = client.get(f"{BASE_URL}/search", params=params)
        if resp.status_code != 200:
            return f"Search failed: {resp.text}"
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No search results found."
        summary = "\n".join(
            [f"- {r['title']}\n  {r['link']}" for r in results]
        )
        return f"Search results for '{query}':\n{summary}"

