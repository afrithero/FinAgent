import re
from typing import Optional, Dict, Any


def should_run_backtest(query: str) -> Optional[str]:
    # Heuristic: backtest-related keywords + ticker in query.
    backtest_keywords = ["backtest", "back-testing", "回測", "量化", "SMA"]
    if not any(k.lower() in query.lower() for k in backtest_keywords):
        return None
    # Find ticker like AAPL, TSLA, 2330 (numeric) - prefer uppercase alpha tickers.
    m = re.search(r"\b([A-Z]{2,5})\b", query)
    if m:
        return m.group(1)
    # Fallback: look for common ticker mention inside parentheses e.g. (AAPL).
    m2 = re.search(r"\(([A-Za-z0-9\-]{1,6})\)", query)
    if m2:
        return m2.group(1).upper()
    return None


def should_search_stock_info(query: str) -> bool:
    # Heuristic: outlook/prediction/market potential questions benefit from web search.
    keywords = [
        "outlook", "prediction", "forecast", "growth potential", "market potential",
        "future", "price target", "analyst", "rating", "guidance", "news",
        "前景", "預測", "成長", "市場", "目標價", "展望",
    ]
    q = query.lower()
    return any(k in q for k in keywords)


def route_state(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    ticker = should_run_backtest(query)
    run_search = should_search_stock_info(query)
    base = {"run_search": run_search}
    if not ticker:
        return {**base, "run_backtest": False}
    csv_map = {"AAPL": "../data/us_stock/AAPL.csv", "2330": "../data/tw_stock/2330.csv"}
    csv_path = csv_map.get(ticker, f"../data/us_stock/{ticker}.csv")
    return {**base, "run_backtest": True, "ticker": ticker, "csv_path": csv_path}
