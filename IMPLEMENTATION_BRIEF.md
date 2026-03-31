# Implementation Brief

> **Purpose**: Handoff document for the implementation-focused session.
> This session handles concept convergence and code review; the implementation session executes workstreams and writes code/tests.
> This document summarizes the five known unfinished items as concrete, scoped workstreams.

---

## Collaboration Contract

### Roles

| Session | Responsibility |
|---------|----------------|
| **Implementation Session** | Writes code and tests; executes workstreams; proposes file changes |
| **Review/Concept Session** | Converges architecture; reviews code; controls scope changes |

### Division of Labor

- **Implementation session** does not decide architectural direction. If a workstream surfaces an architectural question (e.g., "is this node placement correct?"), implementation pauses and the review session is consulted before proceeding.
- **Review session** does not write code or tests directly. It provides guidance and the implementation session implements it.
- **Scope changes** require review session sign-off. Implementation session may flag scope creep but does not unilaterally expand it.

### Information Flow

1. Implementation session reads `AGENTS.md` and `IMPLEMENTATION_BRIEF.md` at the start of each session
2. Implementation session reports progress and any blocking questions before requesting review
3. Review session provides architectural feedback and merge guidance only
4. Implementation session owns all code changes

### Constraints

- **Do not assume tools/config do not exist** — verify before claiming a capability
- **Keep implementations minimal and focused** — one workstream at a time
- **Write accurate, not aspirational, code** — no TODO-only stubs
- **Prefer additive changes** — avoid large refactors without review session sign-off
- **No lint/coverage/infra tooling** — none is committed; follow surrounding code style

---

## Recommended Delivery Order

| # | Workstream | Rationale |
|---|-----------|-----------|
| 1 | Output Template Standardization | Unblocks memory clarity for all downstream reasoning work |
| 2 | Stock Data Decision Logic | All backtesting depends on having the right data path resolved |
| 3 | Backtrader Strategies & Parameters | Builds on data layer; isolated strategy additions |
| 4 | Multi-Round Reasoning Verification | Can be verified without changing architecture; confirms WS1 effect |
| 5 | LangGraph Architecture Audit | Requires full context from WS1–WS4; highest uncertainty |

---

## Workstreams

---

### Workstream 1: Output Template Standardization

**Goal**: Define a stable, machine-readable output shape for every tool result so multi-round reasoning stays coherent regardless of what the LLM generates.

**Current State**:
- Tool results returned to the LLM have inconsistent shapes (`backtest` returns verbose text, `search_stock_info` returns unstructured strings, `retrieve_financial_docs` returns raw doc list)
- The `answer` field in `State` is free-form; no template constrains it
- No standardized error payload shape

**Target State**:
- Each tool function returns a `ToolResult` Pydantic model (or equivalent TypedDict) with: `status` (`"ok"` | `"error"` | `"empty"`), `summary` (short human-readable string), `data` (tool-specific payload or null), and `debug_hint` (optional trace string)
- `LLMNode` assembles `answer` from the union of tool `summary` fields, not raw tool output
- The `answer` field never mixes structured and unstructured content in the same round

**Implementation Scope**:
1. Define `ToolResult` schema (or TypedDict equivalent) in `agent/graph/state.py`
2. Refactor all tool functions in `agent/graph/tools.py` to return `ToolResult`-conforming dicts
3. Update `LLMNode` in `agent/graph/node.py` to read `tool_result.summary` for answer assembly
4. Update `agent/stock/trader.py` backtest output to emit `ToolResult`-conformant dicts
5. Add `pytest` tests for `ToolResult` shape consistency across tools (basic round-trip checks)

**Out of Scope**:
- Changing the LLM prompt template itself
- Historical answer history reformatting
- Tool result persistence (beyond the current State lifetime)

**Acceptance Criteria**:
- Every call to `backtest_tool`, `retrieve_financial_docs`, and `search_stock_info` returns a dict with `status`, `summary`, `data`, and `debug_hint`
- `agent/graph/node.py` LLMNode reads `summary` field only for answer assembly
- At least one test verifies the `ToolResult` shape for each tool function
- Multi-round conversation: tool results from round N do not include raw full-text from round N−1

**Likely Touched Files**:
- `agent/graph/state.py`
- `agent/graph/tools.py`
- `agent/graph/node.py`
- `agent/stock/trader.py`
- `tests/` (new or existing)

**Review Questions**:
1. Should `ToolResult` be a Pydantic `BaseModel` with `ConfigDict(extra="forbid")` or a TypedDict? (Pydantic aligns with existing `LLMOutputSchema` pattern but adds a runtime dependency per call.)
2. Is `summary` the right field name, or should it be `readable_payload` to avoid confusion with financial summaries?
3. Should the review session pre-approve the exact `ToolResult` schema before implementation begins?

#### Implementation Update (2026-03-30)

**Status**: Completed

**Decisions Made**:
- Chose Pydantic `BaseModel` with `ConfigDict(extra="forbid")` for `ToolResult`
- Confirmed field names: `status` (Literal["ok", "error", "empty"]), `summary` (str), `data` (Any), `debug_hint` (Optional[str])
- Retained `summary` as field name — no rename to `readable_payload`
- `LLMNode` assembles prompt from `summary` field only, not raw tool output
- Scope kept minimal: only WS1 files touched; WS2–WS5 not affected

**Files Changed**:
- `agent/graph/state.py`
- `agent/graph/tools.py`
- `agent/graph/node.py`
- `agent/stock/trader.py`
- `tests/test_tool_result.py`

**Validation**:
```
python3 -m pytest tests/test_tool_result.py -v
```
Result: **14 passed**

**Reviewer Checklist**:
- [ ] `ToolResult` schema matches acceptance criteria in spec
- [ ] All three tools (`backtest_tool`, `retrieve_financial_docs`, `search_stock_info`) return `ToolResult`-conformant dicts
- [ ] `LLMNode` reads `summary` field only for answer assembly
- [ ] Tests cover shape consistency for each tool
- [ ] No `extra` fields permitted on `ToolResult` instances

---

### Workstream 2: Stock Data Decision Logic

**Goal**: Give the agent a deterministic way to resolve "download, fetch live, or reuse cached data" so it no longer requires manual pre-work.

**Current State**:
- `StockLoader` in `agent/stock/stock_loader.py` has `save_one_stock_to_csv()` and `run()` methods that write CSVs to disk
- No in-memory cache
- No live-fetch path (yfinance supports it but the loader doesn't expose it)
- The agent cannot decide between these paths; it has no data-path tool

**Target State**:
- A `resolve_stock_data(ticker, market, start_date, end_date)` function that:
  - Returns in-memory cached data if all parameters are within the cache window
  - Returns the existing CSV file if it covers the full requested range
  - Falls back to `yfinance` live fetch for US tickers (no new CSV written unless explicitly requested)
  - Raises a clear error for Taiwan tickers where live fetch is not implemented
- The `backtest_tool` uses `resolve_stock_data` internally instead of calling `StockLoader` methods directly
- Agent can explain its data choice in the `ToolResult.summary`

**Implementation Scope**:
1. Add a `StockDataCache` class (in-memory dict keyed by `(ticker, market)`) in `agent/stock/stock_loader.py`
2. Add `resolve_stock_data()` function that checks cache → CSV file → live fetch in order
3. Add a `download_stock_data` flag to `backtest_tool` in `agent/graph/tools.py` so users can force a fresh CSV download
4. Update `agent/stock/trader.py` to accept a pre-loaded DataFrame instead of always reading from disk
5. Add `pytest` tests for `resolve_stock_data` covering: cache hit, CSV hit, live fetch, and error path for unimplemented Taiwan live fetch

**Out of Scope**:
- Persistent disk cache (e.g., SQLite or file-based LRU cache)
- Implementing live fetch for Taiwan tickers (twstock has rate-limiting concerns; leave as error with clear message)
- Changing the `download_stock_data.py` CLI

**Acceptance Criteria**:
- Calling `resolve_stock_data("AAPL", "us", "2024-01-01", "2024-06-01")` twice in the same session returns the cached DataFrame (no second HTTP request)
- `backtest_tool` works for US tickers without any prior `download_stock_data.py` execution
- Taiwan tickers raise `NotImplementedError` with a message mentioning manual download
- Cache does not persist across separate agent invocations

**Likely Touched Files**:
- `agent/stock/stock_loader.py`
- `agent/stock/trader.py`
- `agent/graph/tools.py`
- `tests/` (new or existing)

**Review Questions**:
1. Should the cache be a module-level global dict or a class with an explicit `.clear()` method? (Global is simpler but harder to test; class is more explicit.)
2. Should `resolve_stock_data` live in `stock_loader.py` or be a tool function in `tools.py` so the agent can call it explicitly?
3. Is a 24-hour cache TTL needed, or is "same session" the correct cache scope?

---

### Workstream 3: Backtrader Strategies & Parameters

**Goal**: Expand the strategy library and expose parameters through the agent tool interface so users can specify what they want to test.

**Current State**:
- `SmaCross` in `agent/stock/trader.py` is the only strategy
- Its parameters (`fast`, `slow`) are hardcoded in the `params` tuple
- `backtest_tool` in `agent/graph/tools.py` has no `strategy` or `params` argument

**Target State**:
- At minimum three strategies: `SmaCross` (existing), `RSIStrategy`, `MomentumStrategy`
- Each strategy exposes its controllable parameters as constructor kwargs
- `backtest_tool` accepts optional `strategy` (str) and `params` (dict) arguments
- Strategy selection is validated: unknown strategy name raises a clear `ValueError`

**Implementation Scope**:
1. Implement `RSIStrategy` in `agent/stock/trader.py` with controllable `rsi_period` and `rsi_upper`/`rsi_lower` thresholds
2. Implement `MomentumStrategy` in `agent/stock/trader.py` with controllable `lookback_period` and `threshold`
3. Create a `StrategyRegistry` dict mapping strategy name strings to strategy classes
4. Update `backtest_tool` in `agent/graph/tools.py` to accept `strategy` (str, default `"SmaCross"`) and `params` (dict, default `{}`)
5. Add `pytest` tests: strategy instantiation with params, unknown strategy raises, and `StrategyRegistry` lookup

**Out of Scope**:
- Plotting or visualization of backtest results
- Portfolio-level multi-strategy backtesting
- Performance metrics beyond what backtrader already provides
- Strategy optimization or parameter sweep

**Acceptance Criteria**:
- `backtest_tool` can be called with `strategy="RSIStrategy"` and `params={"rsi_period": 14, "rsi_lower": 30}` without error
- `backtest_tool(strategy="UnknownStrategy")` raises `ValueError` with a message listing available strategies
- All three strategies run a backtest on AAPL and produce a `ToolResult` with a `summary` field
- At least one test covers each strategy's instantiation with default params

**Likely Touched Files**:
- `agent/stock/trader.py`
- `agent/graph/tools.py`
- `tests/` (new or existing)

**Review Questions**:
1. Should strategy params be validated (e.g., `rsi_period > 0`) inside the strategy `__init__` or in `backtest_tool` before passing them?
2. Should the `summary` field include the strategy name and key params used? (Helps traceability in multi-round reasoning.)
3. Is the `StrategyRegistry` pattern acceptable, or does the review session prefer a different dispatch mechanism?

---

### Workstream 4: Multi-Round Reasoning Verification

**Goal**: Confirm whether the current agent implementation actually triggers multi-round reasoning from a single user prompt, and expose any gaps.

**Current State**:
- `run_react_agent.py` uses `create_react_agent` with no explicit loop control visible
- `run_stategraph_agent.py` runs a fixed graph: `retrieve` → `route` → (`search` | `backtest`) → `llm` → END — one pass only
- It is unclear whether the LLM ever re-calls a tool in the same `invoke()` call or whether the graph enforces a hard stop after one tool call

**Target State**:
- Verified behavior documented in code comments:
  - Does the ReAct agent re-call tools in a loop? (Should — that's the ReAct pattern.)
  - Does the StateGraph agent stop after one tool call or can it loop?
- If multi-round is broken, the specific breakage point is identified and a fix is scoped
- Agent entrypoints (`run_react_agent.py`, `run_stategraph_agent.py`) include a comment block summarizing their loop behavior

**Implementation Scope**:
1. Trace the execution path in `run_react_agent.py` with a small set of prompts (e.g., one that triggers search, one that triggers backtest) and count how many LLM calls occur
2. Trace `run_stategraph_agent.py` similarly — determine if the graph allows revisiting `route` after a tool completes
3. Add a `max_llm_calls` parameter (default 5) to both entrypoints for safety
4. Document findings in comments at the top of each entrypoint
5. If the graph is confirmed single-pass: add a conditional edge from tool completion back to `route` so it can re-route (minimal change)
6. Add a `pytest` test that runs a simple two-tool prompt through the StateGraph and verifies two tools were called

**Out of Scope**:
- Full conversation memory (session IDs, history threading) — this is a separate workstream
- Changing the ReAct agent's internal `create_react_agent` behavior
- Rewriting the graph topology beyond adding one conditional edge

**Acceptance Criteria**:
- A prompt that requires two distinct tool calls (e.g., "backtest AAPL with RSI and also search for news") results in two tool calls in the same session — not one
- Both entrypoints have a comment block at the top describing their loop behavior
- A safety `max_llm_calls` cap exists to prevent runaway loops
- At least one test exercises a two-tool prompt and asserts the correct number of tool invocations

**Likely Touched Files**:
- `agent/run_react_agent.py`
- `agent/run_stategraph_agent.py`
- `agent/graph/node.py` (potential edge addition)
- `tests/` (new or existing)

**Review Questions**:
1. Is the ReAct agent's multi-round behavior trustworthy as-is, or should it be explicitly tested with a mock tool that returns a retry signal?
2. If the StateGraph needs a loop edge back to `route`, should this be gated by an `allow_loop` flag in the State, or always-on?
3. Should the review session pre-approve the `max_llm_calls` default value (5) before implementation?

---

### Workstream 5: LangGraph Architecture Audit

**Goal**: Evaluate the current StateGraph topology for correctness, identify architectural gaps, and produce a short written assessment.

**Current State**:
- StateGraph flow: `retrieve` → `route` → (`search` | `backtest`) → `llm` → END
- Nodes: `RetrieverNode`, `LLMNode`, `BacktestNode`, `SearchNode` (and `RouteNode` as a conditional)
- `state.py` defines `State` TypedDict with fields: `query`, `docs`, `backtest`, `answer`, `run_backtest`, `run_search`, `ticker`, `csv_path`, `search_results`, `debug`
- `route_state` in `utils.py` performs keyword-based routing with no fallback for mixed or unrouteable queries

**Audit Scope**:
1. **State completeness**: Are all fields used by at least one node? Are any fields written but never read?
2. **Node isolation**: Does each node have a single responsibility? (e.g., `LLMNode` should not also fetch data.)
3. **Edge coverage**: Are all terminal states handled? Are there dead-end nodes?
4. **Error propagation**: If `BacktestNode` or `SearchNode` raises, what happens? Does the graph halt or continue?
5. **Routing robustness**: Does `route_state` handle queries that mention both `backtest` and `search`? What does it do with completely unrelated queries?
6. **LLMNode output coupling**: How does `LLMNode` consume tool results — does it read from `State` fields directly or from `ToolResult.summary` (after Workstream 1)?

**Implementation Scope**:
1. Read all node files and trace field reads/writes
2. Produce a brief written audit document (1–2 pages) covering: findings per bullet above, a list of identified issues, and a recommendation for each issue (fix, defer, or accept risk)
3. Implement fixes for issues where the fix is a single-node change and the fix is low-risk (e.g., adding a missing `run_search = False` reset after search completes)
4. Flag high-risk or multi-node changes for review session decision

**Out of Scope**:
- Rewriting any node from scratch
- Adding new node types (e.g., a "ValidationNode" or "MemoryNode")
- Changing the LLM prompt or model selection
- Threading conversation history through the State

**Acceptance Criteria**:
- Written audit document exists and covers all six audit items
- Each identified issue is tagged as: `fix_now`, `defer`, or `accept_risk`
- Fixes for `fix_now` items are implemented and pass existing tests
- `defer` and `accept_risk` items have a one-sentence rationale
- No new `TODO` comments without a corresponding issue tag

**Likely Touched Files**:
- `agent/graph/node.py`
- `agent/graph/state.py`
- `agent/graph/tools.py`
- `agent/graph/utils.py`
- A new `docs/ARCHITECTURE_AUDIT.md` (or section in this file, if preferred)

**Review Questions**:
1. Should the audit be a separate markdown file or a section within this document?
2. Should `defer` items be tracked as in-repo TODO comments or only in the audit document?
3. Is there a preferred format for the issue tags (e.g., `[FIX]`, `[DEFER]`, `[RISK]`)?

---

## Non-Goals for This Phase

The following are intentionally excluded from this implementation session and will be revisited after the five workstreams above are delivered:

- **Persistent conversation history**: Session IDs, history serialization, history threading through `State` — requires multi-round reasoning (WS4) to be verified first
- **Environment-based search URL configuration**: Moving `http://mcp_server:8000` to an env var is a minor config hygiene item; it is secondary technical debt and does not affect correctness
- **Docker/local parity for search**: The MCP server URL difference between local and containerized runs is out of scope until the architecture audit (WS5) confirms the search node placement
- **Strategy optimization or parameter sweep**: Parameter grids and multi-strategy portfolios are future work once the strategy registry (WS3) is in place
- **Vector DB / FinDER index improvements**: Retrieval quality and index freshness are separate concerns not covered by the five user concerns
- **CI/tooling improvements**: No lint tools, coverage tooling, or issue trackers are committed; implementation session does not add them

---

## Appendix: File Map by Workstream

| File | Workstreams |
|------|-------------|
| `agent/graph/state.py` | 1, 5 |
| `agent/graph/tools.py` | 1, 2, 3, 5 |
| `agent/graph/node.py` | 1, 4, 5 |
| `agent/graph/utils.py` | 5 |
| `agent/stock/stock_loader.py` | 2 |
| `agent/stock/trader.py` | 1, 3 |
| `agent/run_react_agent.py` | 4 |
| `agent/run_stategraph_agent.py` | 4 |
| `docs/ARCHITECTURE_AUDIT.md` | 5 (new file) |
| `tests/` | 1, 2, 3, 4 |

---

## Notes for Implementation Session

- `pytest.ini` sets `pythonpath = agent` — tests run from repo root
- CI uses `pytest -q` on Python 3.10.12
- No lint/format tools committed — follow surrounding style (4-space indent, snake_case, CamelCase for classes, partial type hints)
- Pydantic `BaseModel` with `ConfigDict(extra="forbid")` for structured outputs
- Additive debug dict pattern: `debug = state.get("debug", {}).copy(); debug["key"] = value`
- All new functions that can fail should raise descriptive exceptions (fail fast, fail loud)
