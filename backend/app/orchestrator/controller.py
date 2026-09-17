"""The agentic loop. Deterministic Python owns validation, compatibility checking, execution, and
trace-building; the LLM provider owns exactly one decision (which tool(s) to call). See
CLAUDE.md / the plan for the full per-step rationale."""

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.orchestrator.confidence import combine_confidence
from app.orchestrator.execution_trace import ExecutionTrace, build_trace
from app.orchestrator.input_validation import validate_images
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall
from app.orchestrator.tool_registry import QueryInput, ToolRegistry, ToolResult, ToolSpec

# Caps how many specialists run concurrently for one query (see the parallel-execution block in
# handle_query below) -- not unbounded, since CLAUDE.md documents three resident models as already
# tight on a 16GB machine and concurrent inference adds transient memory pressure on top of that.
MAX_PARALLEL_TOOLS = 3


@dataclass
class QueryResult:
    answer_text: str
    confidence: float
    confidence_bucket: str
    evidence_image_paths: list[Path]
    tool_results: list[ToolResult]
    trace: ExecutionTrace


def _synthesize_answer(executed: list[tuple[ToolResult, dict, Optional[str]]]) -> str:
    """MVP answer synthesis: join each executed tool's own text summary. No second LLM call in
    Phase 0 -- deliberately deterministic so numbers/boxes in the answer always match the trace.
    Swap in an LLM-phrased rewrite here later; keep it constrained to rephrasing, never to
    inventing numbers not present in the tool outputs it's given."""
    if not executed:
        return "No specialist tool produced a result for this query."
    return " ".join(result.text_summary for result, _, _ in executed)


def _validation_failure(reason: str) -> QueryResult:
    trace = build_trace(
        selected_task="input_validation_failed",
        executed=[],
        input_summary=reason,
        confidence=0.0,
        confidence_bucket="Low",
        warnings=[reason],
    )
    return QueryResult(
        answer_text="Could not process the input: " + reason,
        confidence=0.0,
        confidence_bucket="Low",
        evidence_image_paths=[],
        tool_results=[],
        trace=trace,
    )


def _run_tool(item: tuple[ToolCall, ToolSpec], query_input: QueryInput) -> tuple:
    """Runs one already-compatibility-checked tool call. Returns a tagged tuple rather than
    mutating any shared list directly -- this function runs inside a worker thread (see the
    ThreadPoolExecutor in handle_query below), and building the result purely functionally means
    the caller can safely append to `executed`/`skip_warnings` back on the main thread without
    relying on GIL-specific reasoning about concurrent list mutation."""
    call, spec = item
    try:
        result = spec.handler(query_input, call.arguments)
    except Exception as exc:
        # A specialist can fail for reasons that have nothing to do with the query: an
        # uninstalled dependency, a gated model with no token, an upstream API refusing the
        # call. None of those should take down the whole request with a 500 -- record what
        # went wrong in the trace and carry on, so the user still gets an auditable answer
        # that says which tool failed and why. The message reaches the UI's warnings section.
        return ("error", spec.name, str(exc))
    return ("ok", result, call.arguments, spec.checkpoint_id)


def handle_query(
    query_text: str,
    query_input: QueryInput,
    llm_provider: LLMProvider,
    registry: ToolRegistry,
    forced_tool_calls: Optional[list[ToolCall]] = None,
) -> QueryResult:
    if not query_input.images and query_input.location is None:
        return _validation_failure("No images and no location were provided.")

    validation = validate_images(query_input.images)
    if not validation.ok:
        return _validation_failure("; ".join(validation.errors))

    tool_specs = registry.list_specs()
    input_summary = validation.summary_text()
    if query_input.location:
        input_summary += f"\nQuery location: ({query_input.location.lat:.5f}, {query_input.location.lon:.5f})"
    # A caller can bypass the LLM's own tool choice entirely (e.g. the UI's manual "Advanced"
    # picker) by passing forced_tool_calls -- everything downstream (compatibility checks,
    # execution, trace, tool_results) treats it identically to an LLM-selected list.
    tool_calls = forced_tool_calls if forced_tool_calls is not None else llm_provider.select_tools(
        query_text, tool_specs, input_summary
    )

    executed: list[tuple[ToolResult, dict, Optional[str]]] = []
    skip_warnings: list[str] = []
    retried = False
    to_run: list[tuple[ToolCall, ToolSpec]] = []
    i = 0
    while i < len(tool_calls):
        call = tool_calls[i]
        spec = registry.get(call.tool_name)

        if spec is None:
            skip_warnings.append(f"Requested unknown tool '{call.tool_name}'; skipped.")
            i += 1
            continue

        incompatible_reason = spec.is_compatible(query_input, validation.modalities)
        if incompatible_reason:
            if not retried:
                retried = True
                retry_query = (
                    f"{query_text}\n\nNote: the tool '{call.tool_name}' is not usable here "
                    f"({incompatible_reason}). Pick a different, compatible tool, or none."
                )
                retry_calls = llm_provider.select_tools(retry_query, tool_specs, input_summary)
                # Keep the prefix already processed (successes and skips before index i) exactly as
                # it is; only the still-unprocessed remainder gets replaced. Without this, resetting
                # i to 0 over a brand-new full list let a tool that already ran successfully get
                # re-selected and re-executed, duplicating it in the trace -- also drop anything the
                # retry re-suggests that's already queued in `to_run`, as a second line of defense.
                # (Checked against `to_run`, not `executed`: execution now happens in one batch
                # after resolution finishes, so nothing has "executed" yet at retry time -- what
                # must not be re-suggested is anything already resolved as compatible-and-queued.)
                already_queued = {queued_call.tool_name for queued_call, _ in to_run}
                tool_calls = tool_calls[:i] + [c for c in retry_calls if c.tool_name not in already_queued]
                continue
            skip_warnings.append(incompatible_reason)
            i += 1
            continue

        to_run.append((call, spec))
        i += 1

    # Independent specialists (none reads another's output) run concurrently instead of paying
    # each one's latency serially -- safe because every adapter's lazy model-singleton getter is
    # already thread-safe (app/concurrency.py's serialize_first_call), and that lock only guards
    # getting-or-building the model, not the inference call itself. `pool.map` yields results in
    # input order (not completion order), so `executed`'s ordering -- and everything downstream
    # that depends on it (the trace, tool_results) -- stays deterministic regardless of which
    # tool actually finishes first.
    if to_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(to_run), MAX_PARALLEL_TOOLS)) as pool:
            for outcome in pool.map(lambda item: _run_tool(item, query_input), to_run):
                if outcome[0] == "ok":
                    _, result, arguments, checkpoint_id = outcome
                    executed.append((result, arguments, checkpoint_id))
                else:
                    _, name, message = outcome
                    skip_warnings.append(f"{name} failed: {message}")

    all_warnings = validation.warnings + skip_warnings
    confidences = [result.confidence for result, _, _ in executed]
    confidence, bucket = combine_confidence(confidences, warning_count=len(all_warnings))

    selected_task = ", ".join(result.tool_name for result, _, _ in executed) or "unclassified"
    answer_text = _synthesize_answer(executed)
    if not executed and not all_warnings:
        all_warnings = all_warnings + ["No tool call matched this query."]

    trace = build_trace(
        selected_task=selected_task,
        executed=executed,
        input_summary=input_summary,
        confidence=confidence,
        confidence_bucket=bucket,
        warnings=all_warnings,
    )

    evidence_paths = [result.evidence_image_path for result, _, _ in executed if result.evidence_image_path]

    return QueryResult(
        answer_text=answer_text,
        confidence=confidence,
        confidence_bucket=bucket,
        evidence_image_paths=evidence_paths,
        tool_results=[result for result, _, _ in executed],
        trace=trace,
    )
