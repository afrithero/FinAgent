"""Tests for trace_utils: Stage execution visibility and contract validation.

These tests use synthetic stream chunks/payloads to verify:
- ToolResult contract validation
- LLMOutputSchema validation
- Event formatting helpers
"""
import pytest
from agent.utils.trace_utils import (
    validate_tool_result,
    validate_llm_output,
    format_tool_result_summary,
    print_stage,
    print_contract_check,
    is_final_answer_message,
    get_planning_tool_names,
)


# -------------------------------------------------------------------------- #
# ToolResult validation
# -------------------------------------------------------------------------- #
class TestValidateToolResult:
    """Tests for validate_tool_result function."""

    def test_valid_tool_result_ok(self):
        data = {
            "status": "ok",
            "summary": "Test summary",
            "data": {"key": "value"},
            "debug_hint": "trace info",
        }
        result = validate_tool_result(data)
        assert result["valid"] is True
        assert "valid" in result["message"].lower()

    def test_valid_tool_result_error(self):
        data = {
            "status": "error",
            "summary": "Error occurred",
            "data": None,
            "debug_hint": "stack trace",
        }
        result = validate_tool_result(data)
        assert result["valid"] is True

    def test_valid_tool_result_empty(self):
        data = {
            "status": "empty",
            "summary": "No results",
            "data": None,
            "debug_hint": None,
        }
        result = validate_tool_result(data)
        assert result["valid"] is True

    def test_none_input(self):
        result = validate_tool_result(None)
        assert result["valid"] is False
        assert "None" in result["message"]

    def test_non_dict_input(self):
        result = validate_tool_result("not a dict")
        assert result["valid"] is False
        assert "not a dict" in result["message"]

    def test_missing_keys(self):
        data = {"status": "ok", "summary": "test"}
        result = validate_tool_result(data)
        assert result["valid"] is False
        assert "missing keys" in result["message"]

    def test_extra_keys(self):
        data = {
            "status": "ok",
            "summary": "test",
            "data": None,
            "debug_hint": None,
            "extra_field": "not allowed",
        }
        result = validate_tool_result(data)
        assert result["valid"] is False
        assert "extra keys" in result["message"]

    def test_invalid_status(self):
        data = {
            "status": "invalid_status",
            "summary": "test",
            "data": None,
            "debug_hint": None,
        }
        result = validate_tool_result(data)
        assert result["valid"] is False
        assert "Invalid status" in result["message"]

    def test_missing_status(self):
        data = {
            "summary": "test",
            "data": None,
            "debug_hint": None,
        }
        result = validate_tool_result(data)
        assert result["valid"] is False


# -------------------------------------------------------------------------- #
# LLMOutputSchema validation
# -------------------------------------------------------------------------- #
class TestValidateLLMOutput:
    """Tests for validate_llm_output function."""

    def test_valid_llm_output_full(self):
        data = {
            "verdict": "Moderate growth potential",
            "recommendation": "Consider small allocation",
            "backtest_summary": "Return 12.5%, Sharpe 1.2",
        }
        result = validate_llm_output(data)
        assert result["valid"] is True
        assert "valid" in result["message"].lower()

    def test_valid_llm_output_null_backtest(self):
        data = {
            "verdict": "High growth potential",
            "recommendation": "Buy more",
            "backtest_summary": None,
        }
        result = validate_llm_output(data)
        assert result["valid"] is True

    def test_none_input(self):
        result = validate_llm_output(None)
        assert result["valid"] is False
        assert "None" in result["message"]

    def test_non_dict_input(self):
        result = validate_llm_output(123)
        assert result["valid"] is False
        assert "not a dict" in result["message"]

    def test_missing_verdict(self):
        data = {
            "recommendation": "Buy",
            "backtest_summary": None,
        }
        result = validate_llm_output(data)
        assert result["valid"] is False
        assert "missing keys" in result["message"]

    def test_missing_recommendation(self):
        data = {
            "verdict": "Growth",
            "backtest_summary": None,
        }
        result = validate_llm_output(data)
        assert result["valid"] is False
        assert "missing keys" in result["message"]

    def test_invalid_verdict_type(self):
        data = {
            "verdict": 123,  # should be str
            "recommendation": "Buy",
            "backtest_summary": None,
        }
        result = validate_llm_output(data)
        assert result["valid"] is False
        assert "verdict must be str" in result["message"]

    def test_invalid_recommendation_type(self):
        data = {
            "verdict": "Growth",
            "recommendation": ["a", "b"],  # should be str
            "backtest_summary": None,
        }
        result = validate_llm_output(data)
        assert result["valid"] is False
        assert "recommendation must be str" in result["message"]

    def test_invalid_backtest_summary_type(self):
        data = {
            "verdict": "Growth",
            "recommendation": "Buy",
            "backtest_summary": {"nested": "object"},  # should be str or null
        }
        result = validate_llm_output(data)
        assert result["valid"] is False
        assert "backtest_summary must be str or null" in result["message"]

    def test_extra_keys_allowed_with_debug(self):
        """Extra debug/trace keys should be allowed."""
        data = {
            "verdict": "Growth",
            "recommendation": "Buy",
            "backtest_summary": None,
            "debug": {"extra": "info"},
            "llm_input": "prompt",
            "llm_output_raw": "response",
        }
        result = validate_llm_output(data)
        assert result["valid"] is True


# -------------------------------------------------------------------------- #
# Formatting helpers
# -------------------------------------------------------------------------- #
class TestFormatToolResultSummary:
    """Tests for format_tool_result_summary function."""

    def test_format_valid_tool_result(self):
        data = {
            "status": "ok",
            "summary": "This is a test summary",
            "data": {"key": "value"},
            "debug_hint": None,
        }
        result = format_tool_result_summary(data)
        assert "[ok]" in result
        assert "This is a test summary" in result

    def test_format_long_summary_truncated(self):
        data = {
            "status": "ok",
            "summary": "A" * 200,
            "data": None,
            "debug_hint": None,
        }
        result = format_tool_result_summary(data)
        assert len(result) < 120  # Should be truncated
        assert "..." in result

    def test_format_none(self):
        result = format_tool_result_summary(None)
        assert result == "<none>"

    def test_format_non_dict(self):
        result = format_tool_result_summary("just a string")
        assert result == "<str>"


# -------------------------------------------------------------------------- #
# Print helpers (smoke tests - just ensure they don't raise)
# -------------------------------------------------------------------------- #
class TestPrintHelpers:
    """Smoke tests for print helper functions."""

    def test_print_stage_start(self, capsys):
        print_stage("test", "start")
        captured = capsys.readouterr()
        assert ">>>" in captured.out
        assert "TEST" in captured.out

    def test_print_stage_complete(self, capsys):
        print_stage("test", "complete", result="ok")
        captured = capsys.readouterr()
        assert "<<<" in captured.out
        assert "TEST" in captured.out
        assert "result=ok" in captured.out

    def test_print_stage_skip(self, capsys):
        print_stage("test", "skip")
        captured = capsys.readouterr()
        assert "---" in captured.out

    def test_print_stage_error(self, capsys):
        print_stage("test", "error", reason="failed")
        captured = capsys.readouterr()
        assert "!!!" in captured.out
        assert "reason=failed" in captured.out

    def test_print_contract_check_valid(self, capsys):
        print_contract_check("test", {"valid": True, "message": "Valid"})
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "Valid" in captured.out

    def test_print_contract_check_invalid(self, capsys):
        print_contract_check("test", {"valid": False, "message": "Invalid"})
        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "Invalid" in captured.out


# -------------------------------------------------------------------------- #
# Synthetic stream event tests
# -------------------------------------------------------------------------- #
class TestSyntheticStreamEvents:
    """Test validation with synthetic stream chunks mimicking LangGraph events."""

    def test_tool_result_from_backtest_event(self):
        """Simulate a backtest tool result from stream event."""
        tool_result = {
            "status": "ok",
            "summary": "Backtest: Return 12.5%, Sharpe 1.2",
            "data": {
                "performance": {
                    "initial_cash": 50000,
                    "final_cash": 56250,
                    "return_pct": 0.125,
                },
                "trades": [],
            },
            "debug_hint": None,
        }
        validation = validate_tool_result(tool_result)
        assert validation["valid"] is True

    def test_tool_result_from_search_event(self):
        """Simulate a search tool result from stream event."""
        tool_result = {
            "status": "ok",
            "summary": "- Apple Q4 Earnings (link)",
            "data": [{"title": "Apple Q4 Earnings", "link": "https://example.com"}],
            "debug_hint": None,
        }
        validation = validate_tool_result(tool_result)
        assert validation["valid"] is True

    def test_llm_output_from_final_node(self):
        """Simulate final LLM output from StateGraph."""
        llm_output = {
            "verdict": "Moderate growth potential",
            "recommendation": "Consider small allocation and monitor earnings",
            "backtest_summary": "Return 12.5%, Sharpe 1.2",
        }
        validation = validate_llm_output(llm_output)
        assert validation["valid"] is True

    def test_react_agent_thinking_event(self):
        """Simulate ReAct agent thinking/planning event (should not be ToolResult)."""
        # This is internal reasoning - we don't validate it as ToolResult
        thinking_content = "Let me search for AAPL stock information and run a backtest."
        # Just verify it's not validated as tool result
        result = validate_tool_result(thinking_content)
        assert result["valid"] is False  # It's a string, not a dict

    def test_react_tool_call_event(self):
        """Simulate tool call event from ReAct stream."""
        # Tool calls are dicts with 'name' and 'args', not ToolResults
        tool_call = {
            "name": "backtest",
            "args": {"csv_path": "../data/us_stock/AAPL.csv", "cash": 50000},
        }
        result = validate_tool_result(tool_call)
        assert result["valid"] is False  # Missing required ToolResult keys


# -------------------------------------------------------------------------- #
# Final answer detection and safe formatting helpers
# -------------------------------------------------------------------------- #
class TestFinalAnswerDetection:
    """Tests for is_final_answer_message and get_planning_tool_names."""

    def test_ai_message_without_tool_calls_is_final(self):
        """AIMessage without tool_calls should be detected as final answer."""
        from langchain_core.messages import AIMessage
        
        msg = AIMessage(content="Here is the final answer.")
        assert is_final_answer_message(msg) is True

    def test_ai_message_with_tool_calls_is_not_final(self):
        """AIMessage with tool_calls should NOT be detected as final."""
        from langchain_core.messages import AIMessage
        
        msg = AIMessage(
            content="Let me search for that.",
            tool_calls=[{"id": "call_1", "name": "search", "args": {"query": "AAPL"}}]
        )
        assert is_final_answer_message(msg) is False

    def test_ai_message_with_tool_calls_in_additional_kwargs(self):
        """AIMessage with tool_calls in additional_kwargs should not be final."""
        from langchain_core.messages import AIMessage
        
        msg = AIMessage(
            content="Running backtest now.",
            additional_kwargs={"tool_calls": [{"name": "backtest", "args": {}}]}
        )
        assert is_final_answer_message(msg) is False

    def test_human_message_is_not_final(self):
        """HumanMessage should never be considered final."""
        from langchain_core.messages import HumanMessage
        
        msg = HumanMessage(content="Should I buy AAPL?")
        assert is_final_answer_message(msg) is False

    def test_tool_message_is_not_final(self):
        """ToolMessage should never be considered final."""
        from langchain_core.messages import ToolMessage
        
        msg = ToolMessage(content="Backtest complete", tool_call_id="abc123")
        assert is_final_answer_message(msg) is False

    def test_none_message_is_not_final(self):
        """None should return False."""
        assert is_final_answer_message(None) is False


class TestGetPlanningToolNames:
    """Tests for get_planning_tool_names helper."""

    def test_extracts_tool_names_from_tool_calls(self):
        """Should extract tool names from tool_calls attribute."""
        from langchain_core.messages import AIMessage
        
        msg = AIMessage(
            content="Planning to run backtest and search.",
            tool_calls=[
                {"id": "call_1", "name": "backtest", "args": {"csv_path": "test.csv"}},
                {"id": "call_2", "name": "search_stock_info", "args": {"query": "AAPL"}},
            ]
        )
        names = get_planning_tool_names(msg)
        assert "backtest" in names
        assert "search_stock_info" in names
        assert len(names) == 2

    def test_extracts_tool_names_from_additional_kwargs(self):
        """Should extract tool names from additional_kwargs.tool_calls."""
        from langchain_core.messages import AIMessage
        
        msg = AIMessage(
            content="Planning.",
            additional_kwargs={"tool_calls": [{"name": "retriever", "args": {}}]}
        )
        names = get_planning_tool_names(msg)
        assert "retriever" in names

    def test_returns_empty_list_when_no_tool_calls(self):
        """Should return empty list when no tool calls present."""
        from langchain_core.messages import AIMessage
        
        msg = AIMessage(content="Final answer.")
        names = get_planning_tool_names(msg)
        assert names == []

    def test_returns_empty_list_for_non_ai_message(self):
        """Should return empty list for non-AIMessage."""
        from langchain_core.messages import HumanMessage
        
        msg = HumanMessage(content="Test query")
        names = get_planning_tool_names(msg)
        assert names == []
