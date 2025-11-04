from fastapi import FastAPI, Query, HTTPException
from fastapi_mcp import FastApiMCP
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from models import SearchResponse
from fastapi import Request
import httpx
import os
import uvicorn
import anyio


load_dotenv()
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

app = FastAPI(title="SerpAPI Server")
client = httpx.AsyncClient(timeout=30.0)

@app.get("/hello", operation_id="hello", tags=["tools"])
async def hello():
    return {"message": "Hello from FastAPI!"}

@app.get("/search", response_model=SearchResponse, operation_id="search_tool", tags=["tools"])
async def serp_search(
    query: str = Query(..., description="Search query"),
    num_results: int = Query(5, description="Number of results"),
    hl: str = Query("en", description="Language code"),
    gl: str = Query("us", description="Country code")
):
    
    if not SERP_API_KEY:
        return {"error": "Missing SERPAPI_API_KEY. Please set it in .env"}

    params = {
        "api_key": SERP_API_KEY,
        "q": query,
        "hl": hl,
        "gl": gl,
        "num": num_results
    }

    try:
        def run_search() -> Dict[str, Any]:
            search = GoogleSearch(params)
            return search.get_dict()

        data = await anyio.to_thread.run_sync(run_search)
        organic = data.get("organic_results", []) or []

        results: List[Dict[str, Optional[str]]] = [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in organic[:num_results]
        ]

        return SearchResponse(query=query, results=results)

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to query SerpAPI: {str(e)}")

mcp = FastApiMCP(
    app,
    name="SerpAPI MCP",
    include_tags=["tools"] 
)

mcp.mount_http()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP client on shutdown"""
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)