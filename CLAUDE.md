# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r agent/requirements.txt
pip install -r mcp_server/requirements.txt
```

### Docker (primary development environment)
```bash
docker compose up --build          # start both services
docker exec langgraph_agent bash   # shell into agent container
```

### Tests
```bash
# All tests (from repo root)
pytest -q

# Inside container
docker exec langgraph_agent pytest -q -o "pythonpath=."

# Single test
docker exec langgraph_agent pytest -q -o "pythonpath=." tests/test_dataset.py::TestFinderDataset::test_schema_columns
```

`pytest.ini` sets `pythonpath = agent`, so all imports in `agent/` are importable without the `agent.` prefix in tests.

### Build vector index
```bash
cd ./agent && python build_index.py    # builds Faiss index, persists to ../data/FinDER/
```

### Run agents
```bash
cd ./agent && python run_react_agent.py       # LangChain ReAct agent
cd ./agent && python run_stategraph_agent.py  # LangGraph StateGraph agent
```

## Architecture

### System overview

FinAgent is a financial AI agent combining RAG (LlamaIndex + Faiss), live web search (SerpAPI via FastAPI MCP server), and Backtrader backtesting to answer investment queries.

Two Docker services:
- **`mcp_server`** (port 8000): FastAPI service exposing `/search` (SerpAPI wrapper)
- **`agent`** (`langgraph_agent` container): LangGraph agent with RAG, backtest, and LLM components

### LangGraph StateGraph flow

```
retrieve → route → (search?) → (backtest?) → llm → END
```

- **RetrieverNode** (`graph/node.py`): Queries Faiss vector DB for financial docs
- **route** (`graph/utils.py`): Keyword heuristic determines `run_backtest` / `run_search` flags and extracts ticker
- **SearchNode** (`graph/node.py`): Calls `http://mcp_server:8000/search` — only reachable inside Docker network
- **BacktestNode** (`graph/node.py`): Delegates to `backtest_tool` which calls `resolve_stock_data` then `Backtester`
- **LLMNode** (`graph/node.py`): Reads **only** `ToolResult.summary` fields (never raw data) to build the LLM prompt; outputs `LLMOutputSchema` (verdict, recommendation, backtest_summary)

### ToolResult contract

Every tool function must return a dict matching `ToolResult` (defined in `agent/graph/state.py`):
```python
ToolResult(status="ok"|"error"|"empty", summary=str, data=Any|None, debug_hint=str|None)
```
`LLMNode` reads `.summary` only. The `data` field carries the structured payload for downstream use.

### Stock data resolution (`agent/stock/stock_loader.py`)

`resolve_stock_data(ticker, market, start_date, end_date, ...)` resolves data in priority order:
1. **In-memory cache** (`STOCK_DATA_CACHE`, module-level `StockDataCache` singleton)
2. **CSV file** if it covers the requested date range
3. **yfinance live fetch** (US tickers only; Taiwan raises `NotImplementedError`)

`download_stock_data=True` bypasses cache/CSV and forces a live fetch + CSV write.

### LLM / Embedder factories

`LLMFactory` and `EmbedderFactory` (`agent/llm/`, `agent/embedder/`) use a provider string (`"huggingface"`, `"gemini"`, `"openai"`) to instantiate the right backend. Add new providers by extending the factory.

### State fields (`agent/graph/state.py`)

Key `State` TypedDict fields: `query`, `docs`, `backtest` (ToolResult), `answer`, `run_backtest`, `run_search`, `ticker`, `market`, `start_date`, `end_date`, `download_stock_data`, `csv_path`, `search_results` (ToolResult), `debug`.

`debug` is an additive dict — always copy before writing: `debug = state.get("debug", {}).copy()`.

## Coding conventions

- 4-space indent, `snake_case` functions/variables, `CamelCase` classes
- Partial type hints — annotate function signatures but not every local variable
- Pydantic `BaseModel` with `ConfigDict(extra="forbid")` for structured outputs
- Fail fast with descriptive exceptions (no silent swallowing)
- No lint/format tools committed — follow surrounding style
