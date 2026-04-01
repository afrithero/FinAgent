"""Tests for Workstream 1: Output Template Standardization.

Each tool function must return a dict with exactly the four ToolResult keys:
  status, summary, data, debug_hint
and LLMNode must assemble its prompt using only the summary field.
"""
import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from graph.state import ToolResult
from graph.tools import backtest_tool, create_retriever_tool, search_stock_info
from stock.trader import Backtester, SmaCross


# -------------------------------------------------------------------------- #
# ToolResult schema
# -------------------------------------------------------------------------- #
class TestToolResultSchema:
    def test_tool_result_ok_minimal(self):
        r = ToolResult(status="ok", summary="test", data=None, debug_hint=None)
        assert r.status == "ok"
        assert r.summary == "test"
        assert r.data is None
        assert r.debug_hint is None

    def test_tool_result_error_with_data(self):
        r = ToolResult(
            status="error",
            summary="failed",
            data={"msg": "boom"},
            debug_hint="trace123",
        )
        assert r.status == "error"
        assert r.data["msg"] == "boom"

    def test_extra_keys_forbidden(self):
        with pytest.raises(ValidationError):
            ToolResult(
                status="ok",
                summary="test",
                data=None,
                debug_hint=None,
                extra_field="not allowed",
            )

    def test_valid_status_literals(self):
        for s in ("ok", "error", "empty"):
            r = ToolResult(status=s, summary="x", data=None, debug_hint=None)
            assert r.status == s


# -------------------------------------------------------------------------- #
# backtest_tool
# -------------------------------------------------------------------------- #
class TestBacktestTool:
    def _stub_backtester(self):
        """Return a Backtester that reads a non-existent path and will
        trigger FileNotFoundError, exercising the error path."""

        class StubBacktester(Backtester):
            def run(self):
                raise FileNotFoundError("stub CSV not found")

        return StubBacktester(
            csv_path="DOES_NOT_EXIST.csv",
            strategy=SmaCross,
            cash=10000,
            fast=5,
            slow=20,
        )

    def test_backtest_tool_error_on_missing_file(self):
        # Patch Backtester so it raises FileNotFoundError immediately
        with patch("graph.tools.Backtester") as MockBT:
            MockBT.side_effect = FileNotFoundError("CSV not found")
            result = backtest_tool.invoke({"csv_path": "bad.csv"})

        assert isinstance(result, dict)
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["status"] == "error"
        assert isinstance(result["summary"], str)
        assert result["data"] is None

    def test_backtest_tool_ok_on_valid_backtester(self):
        # Build a real Backtester backed by a tiny in-memory CSV
        csv_content = (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-01,100,105,99,101,1000\n"
            "2024-01-02,101,106,100,102,1000\n"
            "2024-01-03,102,107,101,103,1000\n"
            "2024-01-04,103,108,102,104,1000\n"
            "2024-01-05,104,109,103,105,1000\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            bt_runner = Backtester(
                csv_path, strategy=SmaCross, cash=10000, fast=2, slow=3
            )
            bt_runner.run()
            result = bt_runner.to_tool_result()
        finally:
            os.unlink(csv_path)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["status"] == "ok"
        assert isinstance(result["summary"], str)
        assert "performance" in result["data"]
        assert "trades" in result["data"]

    def test_backtest_tool_generic_exception(self):
        with patch("graph.tools.Backtester") as MockBT:
            MockBT.side_effect = RuntimeError("backtest exploded")
            result = backtest_tool.invoke({"csv_path": "any.csv"})

        assert result["status"] == "error"
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}


# -------------------------------------------------------------------------- #
# retrieve_financial_docs
# -------------------------------------------------------------------------- #
class TestRetrieverTool:
    def test_retriever_tool_returns_toolresult_ok(self):
        stub_docs = [
            "Revenue grew 20% YoY in Q3 2024.",
            "Gross margin expanded 150 bps to 68%.",
            "Analyst target price raised to $180.",
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = stub_docs

        tool_fn = create_retriever_tool(mock_retriever)
        result = tool_fn.invoke({"query": "AAPL outlook"})

        assert isinstance(result, dict)
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["status"] == "ok"
        assert result["data"] == stub_docs
        assert len(result["summary"]) > 0

    def test_retriever_tool_empty_result(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        tool_fn = create_retriever_tool(mock_retriever)
        result = tool_fn.invoke({"query": "nonexistent query"})

        assert isinstance(result, dict)
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["status"] == "empty"
        assert result["data"] is None


# -------------------------------------------------------------------------- #
# search_stock_info
# -------------------------------------------------------------------------- #
class TestSearchStockInfo:
    def test_search_tool_returns_toolresult_ok(self):
        fake_results = [
            {"title": "Apple Q4 Earnings Beat", "link": "https://example.com/1"},
            {"title": "AAPL Target Price Upgrade", "link": "https://example.com/2"},
        ]
        with patch("graph.tools.httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.json.return_value = {
                "results": fake_results
            }
            MockClient.return_value = mock_instance

            result = search_stock_info.invoke({"query": "AAPL earnings"})

        assert isinstance(result, dict)
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["status"] == "ok"
        assert result["data"] == fake_results
        assert len(result["summary"]) > 0

    def test_search_tool_http_error(self):
        with patch("graph.tools.httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value.status_code = 500
            mock_instance.get.return_value.text = "Internal Server Error"
            MockClient.return_value = mock_instance

            result = search_stock_info.invoke({"query": "AAPL"})

        assert result["status"] == "error"
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["data"] is None

    def test_search_tool_empty_results(self):
        with patch("graph.tools.httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.json.return_value = {"results": []}
            MockClient.return_value = mock_instance

            result = search_stock_info.invoke({"query": "xyz"})

        assert result["status"] == "empty"
        assert set(result.keys()) == {"status", "summary", "data", "debug_hint"}
        assert result["data"] is None


# -------------------------------------------------------------------------- #
# SearchNode (node.py)
# -------------------------------------------------------------------------- #
class TestSearchNode:
    def test_searchnode_success(self):
        from graph.node import SearchNode

        fake_items = [
            {"title": "Apple Q4 Earnings Beat", "link": "https://example.com/1"},
            {"title": "AAPL Target Price Upgrade", "link": "https://example.com/2"},
        ]
        with patch("graph.node.httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.json.return_value = {
                "results": fake_items
            }
            MockClient.return_value = mock_instance

            node = SearchNode(num_results=3, hl="en", gl="us")
            result = node({"query": "AAPL earnings", "debug": {}})

        assert isinstance(result, dict)
        assert "search_results" in result
        sr = result["search_results"]
        assert set(sr.keys()) == {"status", "summary", "data", "debug_hint"}
        assert sr["status"] == "ok"
        assert sr["data"] == fake_items
        assert "Apple Q4 Earnings Beat" in sr["summary"]
        assert "https://example.com/1" in sr["summary"]
        assert isinstance(sr["debug_hint"], type(None))

    def test_searchnode_empty(self):
        from graph.node import SearchNode

        with patch("graph.node.httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.json.return_value = {"results": []}
            MockClient.return_value = mock_instance

            node = SearchNode()
            result = node({"query": "nonexistent query xyz", "debug": {}})

        assert isinstance(result, dict)
        sr = result["search_results"]
        assert set(sr.keys()) == {"status", "summary", "data", "debug_hint"}
        assert sr["status"] == "empty"
        assert sr["data"] is None
        assert len(sr["summary"]) > 0

    def test_searchnode_http_error(self):
        from graph.node import SearchNode

        with patch("graph.node.httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.get.return_value.status_code = 500
            mock_instance.get.return_value.text = "Internal Server Error"
            MockClient.return_value = mock_instance

            node = SearchNode()
            result = node({"query": "AAPL", "debug": {}})

        assert isinstance(result, dict)
        sr = result["search_results"]
        assert set(sr.keys()) == {"status", "summary", "data", "debug_hint"}
        assert sr["status"] == "error"
        assert sr["data"] is None
        assert "500" in sr["summary"]


# -------------------------------------------------------------------------- #
# LLMNode: summary-only prompt assembly
# -------------------------------------------------------------------------- #
class TestLLMNodeSummaryOnly:
    def test_llmnode_uses_summary_only_in_prompt(self):
        from graph.node import LLMNode

        # Capture whatever the mock LLM stores when generate() is called
        captured_prompts = []

        class CapturingLLM:
            def generate(self, prompt):
                captured_prompts.append(prompt)
                return "Growth potential is moderate. Consider allocation."

        llm_node = LLMNode(CapturingLLM())

        # State contains full ToolResult shapes (including raw data)
        state = {
            "docs": [
                "Revenue grew 20% YoY in Q3 2024.",
                "Gross margin expanded 150 bps to 68%.",
                "Analyst target price raised to $180.",
            ],
            "backtest": {
                "status": "ok",
                "summary": "Backtest: Return 12.5%  Sharpe 1.2",
                "data": {
                    "performance": {
                        "initial_cash": 50000.0,
                        "final_cash": 56250.0,
                        "return_pct": 0.125,
                        "sharpe_ratio": 1.2,
                    },
                    "trades": [
                        {
                            "date": "2024-01-03",
                            "action": "BUY",
                            "price": 101.5,
                            "size": 10,
                            "value": 1015.0,
                            "commission": 0.001,
                        }
                    ],
                },
                "debug_hint": None,
            },
            "search_results": {
                "status": "ok",
                "summary": "Search: Apple Q4 earnings beat expectations.",
                "data": [
                    {
                        "title": "Apple Q4 Earnings Beat",
                        "link": "https://example.com/apple-q4",
                    }
                ],
                "debug_hint": None,
            },
            "query": "Should I buy AAPL?",
            "debug": {},
        }

        llm_node(state)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # The summary fields must appear in the prompt
        assert "Backtest: Return 12.5%  Sharpe 1.2" in prompt
        assert "Search: Apple Q4 earnings beat expectations." in prompt

        # The safe docs placeholder must appear (count-based, not raw text)
        assert "3 document(s) retrieved" in prompt

        # Raw doc text must NOT appear in prompt
        assert "Revenue grew 20%" not in prompt
        assert "Gross margin expanded" not in prompt
        assert "Analyst target price" not in prompt

        # The raw structured data must NOT appear
        assert "50000.0" not in prompt
        assert "56250.0" not in prompt
        assert "initial_cash" not in prompt
        assert "final_cash" not in prompt
        assert "Apple Q4 Earnings Beat" not in prompt  # title is in data, not summary
        assert "https://example.com" not in prompt
        assert "2024-01-03" not in prompt
        assert "1015.0" not in prompt

    def test_llmnode_handles_legacy_string_shapes_silently(self):
        """Non-dict backtest/search values (legacy strings) must not appear in prompt."""
        from graph.node import LLMNode

        captured_prompts = []

        class CapturingLLM:
            def generate(self, prompt):
                captured_prompts.append(prompt)
                return "Growth potential is moderate."

        llm_node = LLMNode(CapturingLLM())

        state = {
            "docs": ["Raw doc text that must not appear in prompt."],
            # Legacy: backtest is a plain string, not a ToolResult dict
            "backtest": "old backtest result string",
            # Legacy: search_results is None
            "search_results": None,
            "query": "test query",
            "debug": {},
        }

        llm_node(state)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Safe doc placeholder appears
        assert "1 document(s) retrieved" in prompt
        # Legacy raw strings must NOT appear
        assert "old backtest result string" not in prompt
        assert "Raw doc text" not in prompt


# -------------------------------------------------------------------------- #
# LLMNode: deterministic wrapping of raw string output
# -------------------------------------------------------------------------- #
class TestLLMNodeDeterministicWrapping:
    def test_llmnode_wraps_raw_string_into_dict(self):
        """Raw LLM string should be wrapped into dict with required keys."""
        from graph.node import LLMNode, LLMOutputSchema

        class RawTextLLM:
            def generate(self, prompt):
                return "Growth potential is moderate. Consider small allocation."

        llm_node = LLMNode(RawTextLLM())

        state = {
            "docs": [],
            "query": "test query",
            "debug": {},
        }

        result = llm_node(state)

        # answer must always be dict matching LLMOutputSchema
        assert isinstance(result["answer"], dict), "answer must be dict-shaped"
        
        # Validate against schema - should not raise
        validated = LLMOutputSchema.model_validate(result["answer"])
        assert validated.verdict is not None
        assert validated.recommendation is not None

    def test_llmnode_recommendation_contains_raw_text(self):
        """recommendation field should contain the raw LLM text."""
        from graph.node import LLMNode

        raw_text = "This is the full LLM response with detailed reasoning."

        class RawTextLLM:
            def generate(self, prompt):
                return raw_text

        llm_node = LLMNode(RawTextLLM())

        state = {
            "docs": [],
            "query": "test query",
            "debug": {},
        }

        result = llm_node(state)

        assert raw_text in result["answer"]["recommendation"]

    def test_llmnode_verdict_deterministic_derivation(self):
        """verdict should be deterministically derived from raw text."""
        from graph.node import LLMNode

        class RawTextLLM:
            def generate(self, prompt):
                return "First line of response.\nSecond line."
        class RawTextLLM2:
            def generate(self, prompt):
                return "First line of response.\nSecond line."

        llm_node1 = LLMNode(RawTextLLM())
        llm_node2 = LLMNode(RawTextLLM2())

        state = {"docs": [], "query": "test", "debug": {}}

        result1 = llm_node1(state)
        result2 = llm_node2(state)

        # Verdict should be derived deterministically from same input
        assert result1["answer"]["verdict"] == result2["answer"]["verdict"]
        # Should be first line
        assert result1["answer"]["verdict"] == "First line of response."

    def test_llmnode_backtest_summary_from_toolresult(self):
        """backtest_summary should be taken from ToolResult summary when available."""
        from graph.node import LLMNode

        class RawTextLLM:
            def generate(self, prompt):
                return "Analysis complete."

        llm_node = LLMNode(RawTextLLM())

        bt_summary = "Backtest: Return 12.5% Sharpe 1.2"
        state = {
            "docs": [],
            "backtest": {
                "status": "ok",
                "summary": bt_summary,
                "data": {"performance": {}, "trades": []},
                "debug_hint": None,
            },
            "search_results": None,
            "query": "test query",
            "debug": {},
        }

        result = llm_node(state)

        assert result["answer"]["backtest_summary"] == bt_summary

    def test_llmnode_backtest_summary_none_when_missing(self):
        """backtest_summary should be None when no backtest in state."""
        from graph.node import LLMNode

        class RawTextLLM:
            def generate(self, prompt):
                return "Analysis complete."

        llm_node = LLMNode(RawTextLLM())

        state = {
            "docs": [],
            "backtest": None,
            "search_results": None,
            "query": "test query",
            "debug": {},
        }

        result = llm_node(state)

        assert result["answer"]["backtest_summary"] is None

    def test_llmnode_empty_response_handling(self):
        """Empty LLM response should be handled gracefully."""
        from graph.node import LLMNode, LLMOutputSchema

        class EmptyLLM:
            def generate(self, prompt):
                return ""

        llm_node = LLMNode(EmptyLLM())

        state = {
            "docs": [],
            "query": "test query",
            "debug": {},
        }

        result = llm_node(state)

        # Still returns valid dict
        assert isinstance(result["answer"], dict)
        validated = LLMOutputSchema.model_validate(result["answer"])
        # Should have fallback verdict
        assert validated.verdict is not None

    def test_llmnode_raw_output_in_debug(self):
        """Raw LLM output should be stored in debug."""
        from graph.node import LLMNode

        raw_text = "Full raw LLM response."

        class RawTextLLM:
            def generate(self, prompt):
                return raw_text

        llm_node = LLMNode(RawTextLLM())

        state = {
            "docs": [],
            "query": "test query",
            "debug": {},
        }

        result = llm_node(state)

        assert "llm_output_raw" in result["debug"]
        assert result["debug"]["llm_output_raw"] == raw_text

    def test_llmnode_prompt_safety_still_holds(self):
        """Prompt must still only include summary fields, not raw docs/data."""
        from graph.node import LLMNode

        captured_prompts = []

        class CapturingLLM:
            def generate(self, prompt):
                captured_prompts.append(prompt)
                return "Analysis complete."

        llm_node = LLMNode(CapturingLLM())

        state = {
            "docs": ["Revenue $10B", "Profit $1B"],
            "backtest": {
                "status": "ok",
                "summary": "Return 15%",
                "data": {"trades": [{"date": "2024-01-01"}]},
                "debug_hint": None,
            },
            "search_results": {
                "status": "ok",
                "summary": "Found 5 results",
                "data": [{"title": "Secret Data"}],
                "debug_hint": None,
            },
            "query": "test",
            "debug": {},
        }

        llm_node(state)

        prompt = captured_prompts[0]
        # Should have summary-only
        assert "Return 15%" in prompt
        assert "Found 5 results" in prompt
        # Should NOT have raw data
        assert "Revenue $10B" not in prompt
        assert "Profit $1B" not in prompt
        assert "Secret Data" not in prompt
        assert "2024-01-01" not in prompt
