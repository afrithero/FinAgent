from typing import TypedDict, List, Dict, Any, Optional

class State(TypedDict):
    query: str
    docs: List[str]
    backtest: Dict[str, Any]
    answer: str
    run_backtest: Optional[bool]
    run_search: Optional[bool]
    ticker: Optional[str]
    csv_path: Optional[str]
    search_results: Optional[str]
    debug: Dict[str, str] 
