from typing import TypedDict, List, Dict, Any, Literal, Optional
from pydantic import BaseModel, ConfigDict


class ToolResult(BaseModel):
    """Standardised output envelope for every tool result.

    All four keys are always present.  The LLMNode reads only `summary`
    when assembling the answer prompt; `data` carries the structured
    payload and `debug_hint` is an optional trace string.
    """
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error", "empty"]
    summary: str
    data: Any | None
    debug_hint: str | None


class State(TypedDict):
    query: str
    docs: List[str]
    backtest: Optional[Dict[str, Any]]
    answer: Optional[Dict[str, Any]]
    run_backtest: Optional[bool]
    run_search: Optional[bool]
    ticker: Optional[str]
    market: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    download_stock_data: Optional[bool]
    csv_path: Optional[str]
    search_results: Optional[Dict[str, Any]]
    debug: Dict[str, Any]
