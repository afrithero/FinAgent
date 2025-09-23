from typing import TypedDict, List, Dict

class State(TypedDict):
    query: str
    docs: List[str]
    answer: str
    debug: Dict[str, str] 