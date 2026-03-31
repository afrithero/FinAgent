from .tools import backtest_tool
from .state import ToolResult
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LLMOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str
    recommendation: str
    backtest_summary: str | None = None


class RetrieverNode:
    def __init__(self, retriever, top_k=1):
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, state):
        docs = self.retriever.retrieve(state["query"])
        selected_docs = docs[:self.top_k]
        debug = state.get("debug", {}).copy()
        debug["retriever_output"] = "\n---\n".join(selected_docs) if selected_docs else "[No docs]"
        
        return {
            "docs": selected_docs, 
            "debug": debug
            }

class BacktestNode:
    def __init__(self, csv_path, cash, fast, slow):
        self.csv_path = csv_path
        self.cash = cash
        self.fast = fast
        self.slow = slow
    
    def __call__(self, state):
        csv_path = state.get("csv_path", self.csv_path)
        result = backtest_tool.invoke({
            "csv_path": csv_path,
            "cash": self.cash,
            "fast": self.fast,
            "slow": self.slow
        })

        debug = state.get("debug", {}).copy()
        debug["backtest_csv_path"] = csv_path
        debug["backtest_output"] = str(result)
        
        return {
            "docs": state["docs"],
            "backtest": result,
            "debug": debug
        }

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
    return first_line[:200] if len(first_line) > 200 else first_line


class LLMNode:
    def __init__(self, llm):
        self.llm = llm
        # Default system prompt / response format
        self.system_prompt = (
            "You are a financial assistant. Provide a concise plain-text analysis.\n"
            "Include your verdict (short conclusion about growth potential) and "
            "recommendation (concise next actions or explanation).\n"
            "If backtest results are provided, summarize them briefly.\n"
            "Output plain text, not JSON.\n"
        )

    def __call__(self, state):
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
    def __init__(self, num_results=3, hl="en", gl="us"):
        self.num_results = num_results
        self.hl = hl
        self.gl = gl
        self.base_url = "http://mcp_server:8000"

    def __call__(self, state):
        query = state["query"]
        with httpx.Client(timeout=20.0) as client:
            params = {
                "query": query,
                "num_results": self.num_results,
                "hl": self.hl,
                "gl": self.gl,
            }
            resp = client.get(f"{self.base_url}/search", params=params)
            if resp.status_code != 200:
                result = ToolResult(
                    status="error",
                    summary=f"Search failed: HTTP {resp.status_code}",
                    data=None,
                    debug_hint=resp.text[:200],
                ).model_dump()
                debug = state.get("debug", {}).copy()
                debug["search_output"] = result["summary"]
                return {"search_results": result, "debug": debug}
            data = resp.json()
            items = data.get("results", [])
            if not items:
                result = ToolResult(
                    status="empty",
                    summary="No search results found for this query.",
                    data=None,
                    debug_hint=None,
                ).model_dump()
                debug = state.get("debug", {}).copy()
                debug["search_output"] = result["summary"]
                return {"search_results": result, "debug": debug}
            summary_text = "\n".join(
                [f"- {r['title']}  ({r['link']})" for r in items]
            )
            result = ToolResult(
                status="ok",
                summary=summary_text,
                data=items,
                debug_hint=None,
            ).model_dump()
        debug = state.get("debug", {}).copy()
        debug["search_output"] = summary_text
        return {
            "search_results": result,
            "debug": debug
        }
