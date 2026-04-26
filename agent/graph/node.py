from typing import Any, Dict

from .tools import backtest_tool, search_stock_info
from pydantic import BaseModel, ConfigDict, ValidationError
from stock.trader import DEFAULT_STRATEGY_PARAMS, MARKET_DEFAULTS, STRATEGY_REGISTRY


class LLMOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str
    recommendation: str
    backtest_summary: str | None = None


class RetrieverNode:
    def __init__(self, retriever: Any, top_k: int = 1) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs = self.retriever.retrieve(state["query"])
        selected_docs = docs[:self.top_k]
        debug = state.get("debug", {}).copy()
        debug["retriever_output"] = "\n---\n".join(selected_docs) if selected_docs else "[No docs]"
        return {
            "docs": selected_docs,
            "debug": debug,
        }

class BacktestNode:
    def __init__(
        self,
        cash: float | None = None,
        csv_path: str | None = None,
        start_date: str = "2024-01-01",
        end_date: str | None = None,
        download_stock_data: bool = False,
        strategy_params: dict[str, dict] | None = None,
    ) -> None:
        self.cash = cash
        self.csv_path = csv_path
        self.start_date = start_date
        self.end_date = end_date
        self.download_stock_data = download_stock_data
        # Merge user-supplied overrides on top of defaults per strategy.
        self.strategy_params: dict[str, dict] = {}
        for name in STRATEGY_REGISTRY:
            base = dict(DEFAULT_STRATEGY_PARAMS.get(name, {}))
            overrides = (strategy_params or {}).get(name, {})
            base.update(overrides)
            self.strategy_params[name] = base

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state.get("ticker")
        if not ticker:
            raise ValueError("BacktestNode requires 'ticker' in state; route_state must set it before backtest runs.")
        market = state.get("market", "tw" if str(ticker).isdigit() else "us")
        cash = self.cash if self.cash is not None else MARKET_DEFAULTS.get(market, MARKET_DEFAULTS["us"])["default_cash"]
        csv_path = state.get("csv_path") or self.csv_path
        start_date = state.get("start_date", self.start_date)
        end_date = state.get("end_date", self.end_date)
        download = state.get("download_stock_data", self.download_stock_data)

        per_strategy: dict[str, Any] = {}
        summary_lines: list[str] = []

        for strategy_name in STRATEGY_REGISTRY:
            raw = backtest_tool.invoke({
                "ticker": ticker,
                "market": market,
                "start_date": start_date,
                "end_date": end_date,
                "csv_path": csv_path,
                "cash": cash,
                "strategy": strategy_name,
                "params": self.strategy_params[strategy_name],
                "download_stock_data": download,
            })
            if raw.get("status") == "ok":
                perf = raw.get("data", {}).get("performance", {})
                trades = raw.get("data", {}).get("trades", [])
                ret_pct = perf.get("return_pct")
                ret_text = f"{ret_pct:.2%}" if isinstance(ret_pct, (int, float)) else "N/A"
                summary_lines.append(
                    f"- {strategy_name}: Return={ret_text} | Sharpe={perf.get('sharpe_ratio')} | Trades={len(trades)}"
                )
                per_strategy[strategy_name] = {"performance": perf, "trades": trades, "status": "ok"}
            else:
                error_msg = raw.get("summary", "unknown error")
                summary_lines.append(f"- {strategy_name}: Error — {error_msg}")
                per_strategy[strategy_name] = {
                    "performance": None, "trades": [], "status": "error", "error": error_msg,
                }

        all_failed = all(v["status"] == "error" for v in per_strategy.values())
        end_display = end_date or "today"
        combined_summary = (
            f"Multi-strategy backtest ({ticker.upper()}, {start_date} to {end_display}):\n"
            + "\n".join(summary_lines)
        )

        result = {
            "status": "error" if all_failed else "ok",
            "summary": combined_summary,
            "data": per_strategy,
            "debug_hint": None,
        }
        debug = state.get("debug", {}).copy()
        debug.update({
            "backtest_ticker": ticker,
            "backtest_market": market,
            "backtest_csv_path": csv_path,
            "backtest_output": combined_summary,
        })
        return {"docs": state["docs"], "backtest": result, "debug": debug}

def _derive_verdict(raw_text: str) -> str:
    """Derive verdict deterministically from raw LLM text.
    
    Returns first non-empty line trimmed to reasonable length.
    Falls back to safe default if no valid content found.
    """
    if not raw_text:
        return "No response from LLM"
    
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    
    if not lines:
        return "Empty response from LLM"
    
    first_line = lines[0]
    
    # Return first non-empty line, truncated
    return first_line[:200]


class LLMNode:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        # Default system prompt / response format
        self.system_prompt = (
            "You are a financial assistant. Provide a concise plain-text analysis.\n"
            "Include your verdict (short conclusion about growth potential) and "
            "recommendation (concise next actions or explanation).\n"
            "If backtest results are provided, summarize them briefly.\n"
            "Output plain text, not JSON.\n"
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs = state.get("docs", [])
        doc_context = (
            f"[{len(docs)} document(s) retrieved — see RetrieverNode debug for full text]"
            if docs
            else "[No documents retrieved]"
        )
        # Backtest: read only .summary from ToolResult; stay silent for legacy shapes.
        backtest_summary = ""
        backtest_data = state.get("backtest")
        if backtest_data is not None and isinstance(backtest_data, dict):
            backtest_summary = backtest_data.get("summary", "")
        # Search: same safe pattern.
        search_summary = ""
        search_data = state.get("search_results")
        if search_data is not None and isinstance(search_data, dict):
            search_summary = search_data.get("summary", "")
        # Assemble prompt — LLMNode reads only summary fields; no raw doc or data payloads.
        prompt = (
            self.system_prompt + "\n"
            "Context:\n" + doc_context + "\n\n"
            "Search Result:\n" + search_summary + "\n\n"
            "Backtest Result:\n" + backtest_summary + "\n\n"
            "Question:\n" + state["query"] + "\n\n"
            "Please provide your analysis in plain text."
        )

        # Treat LLM output as raw string - no JSON parsing
        raw_answer = self.llm.generate(prompt)
        raw_text = raw_answer.strip() if raw_answer else ""
        
        debug = state.get("debug", {}).copy()
        debug["llm_input"] = prompt
        debug["llm_output_raw"] = raw_text

        # Build deterministic custom JSON structure from raw text
        # verdict: short deterministic summary derived from raw text
        verdict = _derive_verdict(raw_text)
        
        # recommendation: full raw text (trimmed)
        recommendation = raw_text[:5000] if raw_text else "Empty response from LLM"
        
        # backtest_summary: from ToolResult summary when available
        bt_summary: str | None = backtest_summary if backtest_summary else None

        answer = {
            "verdict": verdict,
            "recommendation": recommendation,
            "backtest_summary": bt_summary,
        }
        
        # Validate the answer dict matches schema
        try:
            validated = LLMOutputSchema.model_validate(answer)
            debug["llm_output_parsed"] = validated.model_dump()
            answer = validated.model_dump()
        except ValidationError as e:
            # This should never happen since we build it deterministically
            debug["llm_output_validation_error"] = str(e)
            # Fallback to ensure schema compliance
            answer = {
                "verdict": "LLM output construction failed",
                "recommendation": recommendation,
                "backtest_summary": bt_summary,
            }

        return {
            "answer": answer,
            "debug": debug
        }


class SearchNode:
    def __init__(self, num_results: int = 3, hl: str = "en", gl: str = "us") -> None:
        self.num_results = num_results
        self.hl = hl
        self.gl = gl

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = search_stock_info.invoke({
            "query": state["query"],
            "num_results": self.num_results,
            "hl": self.hl,
            "gl": self.gl,
        })
        debug = state.get("debug", {}).copy()
        debug["search_output"] = result.get("summary", "")
        return {"search_results": result, "debug": debug}
