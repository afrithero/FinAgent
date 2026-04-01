import re
from typing import Optional, Dict, Any

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def extract_start_date(query: str) -> Optional[str]:
    """Extract start date from natural language query. Returns YYYY-MM-DD or None."""
    q = query.lower()

    # "since/from January 2023" or "since/from Jan 2023"
    m = re.search(r"(?:since|from)\s+([a-z]+)\s+(20\d{2})", q)
    if m:
        month_str, year = m.group(1), m.group(2)
        month = _MONTH_MAP.get(month_str)
        if month:
            return f"{year}-{month}-01"

    # "since/from 2023"
    m = re.search(r"(?:since|from)\s+(20\d{2})\b", q)
    if m:
        return f"{m.group(1)}-01-01"

    # Chinese: "2023 年 1 月" / "2023年1月"
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", query)
    if m:
        year, month = m.group(1), m.group(2).zfill(2)
        return f"{year}-{month}-01"

    # Chinese: "2023 年" (year only)
    m = re.search(r"(20\d{2})\s*年", query)
    if m:
        return f"{m.group(1)}-01-01"

    # Explicit ISO date "2023-01-01"
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    if m:
        return m.group(1)

    return None


def should_run_backtest(query: str) -> Optional[str]:
    # Heuristic: backtest-related keywords + ticker in query.
    backtest_keywords = ["backtest", "back-testing", "回測", "量化", "SMA"]
    if not any(k.lower() in query.lower() for k in backtest_keywords):
        return None
    # Find uppercase alpha ticker e.g. AAPL, TSLA (2-5 chars).
    m = re.search(r"\b([A-Z]{2,5})\b", query)
    if m:
        return m.group(1)
    # Taiwan numeric ticker: 4-5 digit code, inside ASCII or full-width parentheses.
    m2 = re.search(r"[(\uff08](\d{4,5})[)\uff09]", query)
    if m2:
        return m2.group(1)
    # Fallback: bare numeric ticker e.g. "2454 股票".
    m3 = re.search(r"\b(\d{4,5})\b", query)
    if m3:
        return m3.group(1)
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
    market = "tw" if ticker.isdigit() else "us"
    csv_map = {"AAPL": "../data/us_stock/AAPL.csv", "2330": "../data/tw_stock/2330.csv"}
    default_dir = "tw_stock" if market == "tw" else "us_stock"
    csv_path = csv_map.get(ticker, f"../data/{default_dir}/{ticker}.csv")
    result = {
        **base,
        "run_backtest": True,
        "ticker": ticker,
        "market": market,
        "csv_path": csv_path,
    }
    start_date = extract_start_date(query)
    if start_date:
        result["start_date"] = start_date
    return result
