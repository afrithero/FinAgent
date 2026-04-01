from langchain_core.tools import tool
from stock.trader import Backtester, SmaCross
from stock.stock_loader import resolve_stock_data
import httpx
import pandas as pd

# Re-export ToolResult so tools.py callers can import it from here.
from graph.state import ToolResult


@tool(
    "backtest",
    description=(
        "Run SMA cross backtest on stock data with deterministic data resolution "
        "(cache -> csv -> live fetch for US). "
        "Supports forcing a fresh download via download_stock_data=true."
    ),
)
def backtest_tool(
    ticker: str = "AAPL",
    market: str = "us",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    csv_path: str = "../data/us_stock/AAPL.csv",
    cash: float = 50000,
    fast: int = 3,
    slow: int = 37,
    download_stock_data: bool = False,
):
    try:
        if not end_date:
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        resolved = resolve_stock_data(
            ticker=ticker,
            market=market,
            start_date=start_date,
            end_date=end_date,
            csv_path=csv_path,
            download_stock_data=download_stock_data,
        )
        bt_runner = Backtester(
            csv_path=resolved.get("csv_path") or csv_path,
            data_df=resolved["df"],
            strategy=SmaCross,
            cash=cash,
            fast=fast,
            slow=slow,
        )
        bt_runner.run()
        result = bt_runner.to_tool_result()
        result["summary"] = (
            f"{result['summary']} | Data source: {resolved['source']} "
            f"({ticker.upper()} {start_date} to {end_date})"
        )
        return result
    except NotImplementedError as exc:
        return ToolResult(
            status="error",
            summary=f"Backtest failed: {exc}",
            data=None,
            debug_hint=str(exc),
        ).model_dump()
    except FileNotFoundError as exc:
        return ToolResult(
            status="error",
            summary=f"Backtest failed: CSV not found at {csv_path}",
            data=None,
            debug_hint=str(exc),
        ).model_dump()
    except Exception as exc:
        return ToolResult(
            status="error",
            summary=f"Backtest error: {exc}",
            data=None,
            debug_hint=str(exc),
        ).model_dump()

def create_retriever_tool(retriever):
    @tool(
        "retrieve_financial_docs",
        description=(
            "Retrieve 1-3 relevant financial context snippets for the given query. "
            "Use this FIRST when the user asks for stock/company outlook or market potential."
        ),
    )
    def retriever_tool(query: str):
        docs = retriever.retrieve(query)[:3]
        if not docs:
            return ToolResult(
                status="empty",
                summary="No financial documents retrieved for this query.",
                data=None,
                debug_hint=None,
            ).model_dump()
        excerpt = ";  ".join(docs)
        return ToolResult(
            status="ok",
            summary=excerpt,
            data=docs,
            debug_hint=None,
        ).model_dump()
    return retriever_tool

@tool(
    "search_stock_info",
    description=(
        "Search for stock outlook, predictions, or market potential using the SerpAPI FastAPI endpoint."
        "Returns summarized search results from Google."
    ),
)
def search_stock_info(query: str, num_results: int = 3, hl: str = "en", gl: str = "us"):
    BASE_URL = "http://mcp_server:8000"
    try:
        with httpx.Client(timeout=20.0) as client:
            params = {
                "query": query,
                "num_results": num_results,
                "hl": hl,
                "gl": gl,
            }
            resp = client.get(f"{BASE_URL}/search", params=params)
            if resp.status_code != 200:
                return ToolResult(
                    status="error",
                    summary=f"Search failed: HTTP {resp.status_code}",
                    data=None,
                    debug_hint=resp.text[:200],
                ).model_dump()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return ToolResult(
                    status="empty",
                    summary="No search results found for this query.",
                    data=None,
                    debug_hint=None,
                ).model_dump()
            summary_text = "\n".join(
                [f"- {r['title']}  ({r['link']})" for r in results]
            )
            return ToolResult(
                status="ok",
                summary=summary_text,
                data=results,
                debug_hint=None,
            ).model_dump()
    except httpx.TimeoutException as exc:
        return ToolResult(
            status="error",
            summary="Search timed out after 20 seconds.",
            data=None,
            debug_hint=str(exc),
        ).model_dump()
    except Exception as exc:
        return ToolResult(
            status="error",
            summary=f"Search error: {exc}",
            data=None,
            debug_hint=str(exc),
        ).model_dump()
