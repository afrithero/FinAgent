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
    backtest: ToolResult
    answer: str
    run_backtest: Optional[bool]
    run_search: Optional[bool]
    ticker: Optional[str]
    csv_path: Optional[str]
    search_results: Optional[ToolResult]
    debug: Dict[str, str] 
