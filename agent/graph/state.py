from typing import TypedDict, List, Dict, Any

class State(TypedDict):
    query: str
    docs: List[str]
    backtest: Dict[str, Any]
    answer: str
    debug: Dict[str, str] 