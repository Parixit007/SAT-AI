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
) -> ExecutionTrace:
    """`executed` is a list of (tool_result, params_passed_to_it, checkpoint_id) tuples, in the
    order they actually ran."""
    tools_used = [
        {"name": result.tool_name, "params": params, "checkpoint_id": checkpoint_id}
        for result, params, checkpoint_id in executed
    ]
    return ExecutionTrace(
        selected_task=selected_task,
        tools_used=tools_used,
        input_summary=input_summary,
        confidence=confidence,
        confidence_bucket=confidence_bucket,
        warnings=warnings,
    )
