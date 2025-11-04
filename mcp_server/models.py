from typing import List, Dict, Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: Optional[str]
    link: Optional[str]
    snippet: Optional[str]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]