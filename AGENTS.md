# FinAgent Developer Guide

> **Agent Instructions**: This file provides guidance for AI agents working in this repository.
> `.cursorrules`, `.cursor/rules/`, and `.github/copilot-instructions.md` were **not found** in this repository.

---

## Repository Overview

**FinAgent** is a Financial AI Agent combining RAG retrieval, web search, and Backtrader-based backtesting for investment analysis.

### Directory Structure

```
llm_agent/
├── agent/                    # Main agent package
│   ├── dataset/              # FinDER dataset loader
│   ├── embedder/              # HuggingFace embedder factory
│   ├── graph/                 # LangGraph StateGraph agent
│   │   ├── node.py           # RetrieverNode, LLMNode, BacktestNode, SearchNode
│   │   ├── state.py          # TypedDict State
│   │   ├── tools.py          # @tool-decorated functions
│   │   └── utils.py          # Route helpers
│   ├── llm/                   # LLM factory + HuggingFace backend
│   ├── retriever/             # Faiss-backed retrieval
│   ├── stock/                 # Backtrader SmaCross + StockLoader
│   ├── vectordb/              # FaissVectorDB
│   ├── build_index.py         # Build FinDER index
│   ├── download_stock_data.py  # Manual stock data download
│   ├── query_index.py         # Query the index
│   ├── run_react_agent.py      # ReAct agent entrypoint
│   └── run_stategraph_agent.py # StateGraph agent entrypoint
├── mcp_server/               # FastAPI + MCP search server
│   ├── server.py             # FastAPI app with /search endpoint
│   ├── models.py             # Pydantic models
│   └── requirements.txt
├── tests/                     # Pytest tests (run inside langgraph_agent container)
├── data/                      # Persisted data (indices, CSVs)
├── .github/workflows/ci.yml   # GitHub Actions CI
├── pytest.ini                 # Sets pythonpath = agent
└── docker-compose.yml
```

---

## Build, Run, and Test Commands

```bash
# Setup (no root requirements.txt)
pip install -r agent/requirements.txt
pip install -r mcp_server/requirements.txt

# Index
cd agent && python build_index.py      # → ../data/FinDER/
cd agent && python query_index.py

# Stock data
cd agent && python download_stock_data.py  # yfinance (US) / twstock (TW)

# Run agents
cd agent && python run_react_agent.py
cd agent && python run_stategraph_agent.py

# Tests (run inside langgraph_agent container)
docker exec langgraph_agent pytest -q -o "pythonpath=."                    # all tests
docker exec langgraph_agent pytest -q -o "pythonpath=." tests/test_dataset.py::TestFinderDataset::test_schema_columns
```

**CI**: GitHub Actions runs on Python 3.10.12 with `pytest -q` (see `.github/workflows/ci.yml`)

---

## Architecture Summary

### ReAct Agent (`run_react_agent.py`)
- LangGraph `create_react_agent` with hardcoded sample query, no multi-turn memory
- Tools: `retrieve_financial_docs`, `backtest`, `search_stock_info`
- Calls `http://mcp_server:8000/search`

### StateGraph Agent (`run_stategraph_agent.py`)
- Custom `State` TypedDict: `query`, `docs`, `backtest`, `answer`, `run_backtest`, `run_search`, `ticker`, `csv_path`, `search_results`, `debug`
- Flow: `retrieve` → `route` → (`search` | `backtest`) → `llm` → END
- `route_state` heuristic routes based on keywords; structured JSON via Pydantic `LLMOutputSchema`

### MCP Server (`mcp_server/server.py`)
- FastAPI on port 8000, `/search` calls SerpAPI, returns `SearchResponse`

### Stock Data (`agent/stock/`)
- `SmaCross`: Backtrader SMA crossover strategy
- `StockLoader`: Downloads US (yfinance) or TW (twstock) to CSV

---

## Code Style Guidance

**No committed lint/format config was found.** Follow observed conventions:

| Convention | Rule |
|------------|------|
| Indentation | 4-space |
| Imports | Grouped: stdlib → third-party → local (blank line between) |
| Variables/functions | `snake_case` |
| Classes/Pydantic models | `CamelCase` |
| Constants | `UPPER_SNAKE_CASE` |
| Type hints | Partial (not exhaustive) |

### Patterns Observed
- **Pydantic for structured output**: `LLMOutputSchema(BaseModel)` with `ConfigDict(extra="forbid")`
- **Explicit exceptions**: `raise ValueError(...)`, `raise RuntimeError(...)`
- **Debug state**: Additive dict `debug = state.get("debug", {}).copy(); debug["key"] = value`
- **Structured LLM output**: LLMNode retries JSON parsing with fallback to raw string

---

## Known Limitations / Future Work

> **Handoff**: Active implementation work is tracked in `IMPLEMENTATION_BRIEF.md`. That document is the source of truth for workstream scoping and delivery order.

1. **No multi-turn memory**: Single `invoke()` per run, no session history
2. **Hardcoded queries**: No CLI or user input in entrypoints
3. **Limited backtesting**: Only `SmaCross` strategy in `agent/stock/trader.py`
4. **Hardcoded CSV paths**: `route_state` only maps AAPL/2330; others default to US path
5. **Manual stock downloads**: `download_stock_data.py` requires manual execution
6. **Hardcoded search URL**: `http://mcp_server:8000` in `graph/node.py` and `graph/tools.py` — secondary technical debt; tracked in `IMPLEMENTATION_BRIEF.md`
7. **No fallback search**: SerpAPI required; no graceful degradation

---

## Environment Variables

Explicitly read in code:
```bash
SERPAPI_API_KEY=...   # Required for /search in mcp_server/server.py
GOOGLE_API_KEY=...   # Used when selecting the Google LLM provider in agent/llm/llm_factory.py
```

Provider-dependent / implicit:
```bash
# OPENAI_API_KEY may be needed when using OpenAI models
# HuggingFace credentials may be needed depending on HuggingFace endpoint configuration
```

---

## Docker

```bash
docker compose up --build
```

- `mcp_server` container on port 8000
- `agent` container runs `sleep infinity` (interactive debugging)
- Both share `finagent_net` network
