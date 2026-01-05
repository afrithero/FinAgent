from .tools import backtest_tool
import httpx
import json
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

class LLMNode:
    def __init__(self, llm):
        self.llm = llm
        # Default system prompt / response format
        self.system_prompt = (
            "You are a financial assistant. Respond in JSON with the following keys:\n"
            "- verdict: string, short conclusion about growth potential\n"
            "- recommendation: string, concise next actions or explanation\n"
            "- backtest_summary: string or null, optional summary of backtest results\n"
            "Return ONLY valid JSON. Do not include extra keys.\n"
            "Example:\n"
            "{\n"
            "  \"verdict\": \"Moderate growth potential.\",\n"
            "  \"recommendation\": \"Consider small allocation and monitor earnings.\",\n"
            "  \"backtest_summary\": null\n"
            "}\n"
        )

    def __call__(self, state):
        context = "\n".join(state["docs"])
        backtest_result = ""
        if "backtest" in state and state["backtest"]:
            backtest_result = json.dumps(state["backtest"], indent=2)
        search_result = ""
        if state.get("search_results"):
            search_result = str(state["search_results"])
        # assemble prompt with system guidance asking for JSON output
        prompt = (
            self.system_prompt + "\n"
            "Context:\n" + context + "\n\n"
            "Search Result:\n" + search_result + "\n\n"
            "Backtest Result:\n" + backtest_result + "\n\n"
            "Question:\n" + state["query"] + "\n\n"
            "Please produce only valid JSON matching the schema above."
        )

        raw_answer = self.llm.generate(prompt)
        answer = raw_answer.strip()
        debug = state.get("debug", {}).copy()
        debug["llm_input"] = prompt
        debug["llm_output_raw"] = answer

        # try to parse JSON, then validate schema; if it fails, retry once with correction
        parsed = None
        try:
            parsed = json.loads(answer)
            validated = LLMOutputSchema.model_validate(parsed)
            debug["llm_output_parsed"] = validated.model_dump()
            parsed = validated.model_dump()
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            debug["llm_output_error"] = str(e)
            repair_prompt = (
                "Your previous response did not match the required JSON schema.\n"
                f"Error: {str(e)}\n"
                "Return ONLY valid JSON that matches the schema in the system prompt."
            )
            repair_answer = self.llm.generate(repair_prompt).strip()
            debug["llm_output_raw_repair"] = repair_answer
            try:
                parsed = json.loads(repair_answer)
                validated = LLMOutputSchema.model_validate(parsed)
                debug["llm_output_parsed"] = validated.model_dump()
                parsed = validated.model_dump()
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e2:
                debug["llm_output_error_repair"] = str(e2)

        return {
            "answer": parsed if parsed is not None else answer,
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
                results = f"Search failed: {resp.text}"
            else:
                data = resp.json()
                items = data.get("results", [])
                if not items:
                    results = "No search results found."
                else:
                    summary = "\n".join(
                        [f"- {r['title']}\n  {r['link']}" for r in items]
                    )
                    results = f"Search results for '{query}':\n{summary}"
        debug = state.get("debug", {}).copy()
        debug["search_output"] = str(results)
        return {
            "search_results": results,
            "debug": debug
        }
