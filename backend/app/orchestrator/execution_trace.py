"""Builds the auditable execution summary the spec requires (selected task, tools/models used,
key parameters, confidence). Built from the orchestrator's own record of what it actually invoked,
never from the LLM's own narration -- so the trace stays trustworthy even if step 6's NL answer
synthesis says something imprecise."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.orchestrator.tool_registry import ToolResult


@dataclass
class ExecutionTrace:
    selected_task: str
    tools_used: list[dict[str, Any]]
    input_summary: str
    confidence: float
    confidence_bucket: str
    warnings: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def build_trace(
    selected_task: str,
    executed: list[tuple[ToolResult, dict[str, Any], str]],
    input_summary: str,
    confidence: float,
    confidence_bucket: str,
    warnings: list[str],
    rounds: list[int] = None,
) -> ExecutionTrace:
    """`executed` is a list of (tool_result, params_passed_to_it, checkpoint_id) tuples, in the
    order they actually ran. `rounds[i]` is which agentic-loop round produced `executed[i]` (see
    controller.handle_query) -- 1 for every entry when the caller doesn't track rounds (e.g. the
    validation-failure path, where `executed` is always empty anyway). Additive: every entry already
    had a stable {name, params, checkpoint_id} shape, this only adds a new key to it."""
    if rounds is None:
        rounds = [1] * len(executed)
    tools_used = [
        {"name": result.tool_name, "params": params, "checkpoint_id": checkpoint_id, "round": round_num}
        for (result, params, checkpoint_id), round_num in zip(executed, rounds)
    ]
    return ExecutionTrace(
        selected_task=selected_task,
        tools_used=tools_used,
        input_summary=input_summary,
        confidence=confidence,
        confidence_bucket=confidence_bucket,
        warnings=warnings,
    )
