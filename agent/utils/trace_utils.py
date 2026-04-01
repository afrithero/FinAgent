"""Shared tracing utilities for stage execution visibility.

This module provides minimal, pure helper functions to:
- Validate ToolResult contracts for tool outputs
- Validate LLMOutputSchema for final answers
- Format and print stage execution events
"""

from typing import Any, TypedDict


# Extra keys that LLMNode may add to the answer dict for internal debugging.
# validate_llm_output permits these in addition to the required schema keys.
_ALLOWED_LLM_DEBUG_KEYS: frozenset[str] = frozenset({
    "debug", "llm_input", "llm_output_raw", "llm_output_error",
    "llm_output_parsed", "llm_output_raw_repair", "llm_output_error_repair",
})


class ValidationResult(TypedDict):
    """Result of a contract validation check."""
    valid: bool
    message: str


def validate_tool_result(data: Any | None) -> ValidationResult:
    """Validate whether a dict conforms to the ToolResult contract.
    
    The ToolResult contract requires exactly these keys:
    - status: Literal["ok", "error", "empty"]
    - summary: str
    - data: Any | None
    - debug_hint: str | None
    
    Returns a validation result with valid flag and descriptive message.
    """
    if data is None:
        return {"valid": False, "message": "ToolResult is None"}
    
    if not isinstance(data, dict):
        return {"valid": False, "message": f"ToolResult is not a dict, got {type(data).__name__}"}
    
    required_keys = {"status", "summary", "data", "debug_hint"}
    actual_keys = set(data.keys())
    
    if actual_keys != required_keys:
        missing = required_keys - actual_keys
        extra = actual_keys - required_keys
        msg_parts = []
        if missing:
            msg_parts.append(f"missing keys: {missing}")
        if extra:
            msg_parts.append(f"extra keys: {extra}")
        return {"valid": False, "message": f"ToolResult contract violated ({', '.join(msg_parts)})"}
    
    valid_statuses = {"ok", "error", "empty"}
    status = data.get("status")
    if status not in valid_statuses:
        return {"valid": False, "message": f"Invalid status: {status!r} (must be one of {valid_statuses})"}
    
    return {"valid": True, "message": "ToolResult valid"}


def validate_llm_output(data: Any | None) -> ValidationResult:
    """Validate whether data conforms to the LLMOutputSchema.
    
    The LLMOutputSchema requires:
    - verdict: str
    - recommendation: str
    - backtest_summary: str | None
    
    Returns a validation result with valid flag and descriptive message.
    """
    if data is None:
        return {"valid": False, "message": "LLMOutput is None"}
    
    if not isinstance(data, dict):
        return {"valid": False, "message": f"LLMOutput is not a dict, got {type(data).__name__}"}
    
    required_keys = {"verdict", "recommendation", "backtest_summary"}
    actual_keys = set(data.keys())
    
    missing = required_keys - actual_keys
    if missing:
        return {"valid": False, "message": f"LLMOutputSchema missing keys: {missing}"}
    
    extra = actual_keys - required_keys - _ALLOWED_LLM_DEBUG_KEYS
    if extra:
        return {"valid": False, "message": f"LLMOutputSchema has extra keys: {extra}"}
    
    # Validate types
    verdict = data.get("verdict")
    recommendation = data.get("recommendation")
    backtest_summary = data.get("backtest_summary")
    
    if not isinstance(verdict, str):
        return {"valid": False, "message": f"verdict must be str, got {type(verdict).__name__}"}
    if not isinstance(recommendation, str):
        return {"valid": False, "message": f"recommendation must be str, got {type(recommendation).__name__}"}
    if backtest_summary is not None and not isinstance(backtest_summary, str):
        return {"valid": False, "message": f"backtest_summary must be str or null, got {type(backtest_summary).__name__}"}
    
    return {"valid": True, "message": "LLMOutputSchema valid"}


def print_stage(stage: str, status: str = "start", **details) -> None:
    """Print a formatted stage execution line.
    
    Args:
        stage: Name of the stage (e.g., "retrieve", "backtest", "llm")
        status: One of "start", "complete", "skip", "error"
        **details: Additional key-value pairs to display
    """
    status_symbols = {
        "start": ">>>",
        "complete": "<<<",
        "skip": "---",
        "error": "!!!",
    }
    symbol = status_symbols.get(status, "?  ")
    
    detail_parts = [f"{k}={v}" for k, v in details.items()]
    detail_str = f" | {', '.join(detail_parts)}" if detail_parts else ""
    
    print(f"{symbol} {stage.upper()}{detail_str}")


def format_tool_result_summary(data: Any | None) -> str:
    """Extract and format a summary from ToolResult data for display."""
    if data is None:
        return "<none>"
    
    if not isinstance(data, dict):
        return f"<{type(data).__name__}>"
    
    status = data.get("status", "?")
    summary = data.get("summary", "")
    
    # Truncate long summaries
    if len(summary) > 100:
        summary = summary[:97] + "..."
    
    return f"[{status}] {summary}"


def print_contract_check(stage: str, validation: ValidationResult) -> None:
    """Print a contract validation check result."""
    symbol = "✓" if validation["valid"] else "✗"
    print(f"   {symbol} CONTRACT: {validation['message']}")


def is_final_answer_message(msg: Any) -> bool:
    """Check if a message is a final answer (AIMessage without tool calls).
    
    Returns True only if the message is an AI message type and does NOT contain
    tool calls - i.e., it's an actual final response rather than reasoning.
    """
    if msg is None:
        return False
    
    msg_type = type(msg).__name__
    is_ai_message = msg_type == "AIMessage" or (hasattr(msg, "type") and msg.type == "ai")
    
    if not is_ai_message:
        return False
    
    # Check for tool calls in two places: direct attribute and additional_kwargs
    has_tool_calls = getattr(msg, "tool_calls", None) or (
        hasattr(msg, "additional_kwargs") and 
        msg.additional_kwargs.get("tool_calls")
    )
    
    return not has_tool_calls


def get_planning_tool_names(msg: Any) -> list[str]:
    """Extract tool call names from a message that has tool calls (planning phase).
    
    Returns list of tool names being planned, or empty list if no tool calls.
    """
    if msg is None:
        return []
    
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        # Check additional_kwargs for tool_calls
        if hasattr(msg, "additional_kwargs"):
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
    
    if not tool_calls:
        return []
    
    return [tc.get("name", "unknown") for tc in tool_calls]


def print_react_event(event: dict) -> None:
    """Print a formatted ReAct agent stream event.
    
    Handles LangGraph streaming events defensively with early-exit guards.
    Avoids exposing chain-of-thought-like text - shows safe stage summaries only.
    """
    # Handle agent planning/decision events
    if "agent" in event:
        agent_data = event.get("agent", {})
        if "messages" in agent_data:
            msgs = agent_data["messages"]
            if msgs:
                last_msg = msgs[-1]
                # Check for tool calls to determine if this is planning vs final answer
                has_tool_calls = getattr(last_msg, "tool_calls", None) or (
                    hasattr(last_msg, "additional_kwargs") and 
                    last_msg.additional_kwargs.get("tool_calls")
                )
                if has_tool_calls:
                    # Safe: show generic planning marker instead of reasoning content
                    tool_call_names = [
                        tc.get("name", "unknown") for tc in (
                            getattr(last_msg, "tool_calls", []) or 
                            last_msg.additional_kwargs.get("tool_calls", [])
                        )
                    ]
                    print(f"   > Agent planning: {', '.join(tool_call_names)}")
                else:
                    # Safe: show final answer preview (no tool calls = actual response)
                    content = getattr(last_msg, "content", "")
                    if content:
                        # Truncate for preview, but only show final answers
                        if len(content) > 150:
                            content = content[:147] + "..."
                        print(f"   > Agent: {content}")
    
    # Handle tool call events
    if "tools" in event:
        tools_data = event.get("tools", {})
        if isinstance(tools_data, list):
            for tool_call in tools_data:
                tool_name = tool_call.get("name", "unknown")
                args = tool_call.get("args", {})
                args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
                print(f"   > TOOL CALL: {tool_name}({args_str})")
        elif isinstance(tools_data, dict):
            # Sometimes tools data comes as dict with tool_calls
            if "tool_calls" in tools_data:
                for tc in tools_data["tool_calls"]:
                    tool_name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
                    print(f"   > TOOL CALL: {tool_name}({args_str})")

    # Handle tool result events (these come as messages with tool role)
    if "messages" in event:
        msgs = event.get("messages", [])
        for msg in msgs:
            msg_type = type(msg).__name__
            if msg_type == "ToolMessage":
                tool_name = getattr(msg, "name", "unknown")
                content = getattr(msg, "content", "")
                # Validate tool result if it's a ToolResult-shaped dict
                validation = validate_tool_result(content)
                summary = format_tool_result_summary(content) if isinstance(content, dict) else str(content)[:80]
                print(f"   < TOOL RESULT ({tool_name}): {summary}")
                print_contract_check("tool", validation)
            elif hasattr(msg, "type") and msg.type == "tool":
                # Handle raw tool messages
                content = getattr(msg, "content", "")
                validation = validate_tool_result(content)
                summary = format_tool_result_summary(content) if isinstance(content, dict) else str(content)[:80]
                print(f"   < TOOL: {summary}")
                print_contract_check("tool", validation)
