"""The agentic loop. Deterministic Python owns validation, compatibility checking, execution, and
trace-building; the LLM provider owns exactly one *kind* of decision -- which tool(s) to call next
-- but is allowed to make it more than once per query. See CLAUDE.md / the plan for the full
per-step rationale.

**Multi-round routing** (added 2026-09-23, replacing a strictly one-shot select-then-execute
design): after a round of tools has actually run, `select_tools()` is called again with `history`
set to what those tools found (the same evidence formatting the final answer is composed from --
see `answer_composer.build_evidence`), so the model can decide whether anything more would help
before the query is considered answered -- the same "look at what I just learned, then decide what's
next" step a genuinely agentic assistant makes after every action, not before it. This is what lets
a query like "how many buildings are there, and does the scene description's answer actually hold
up?" turn into more than one tool call informed by what the first one returned, and what lets a
thin or one-word result get a follow-up from a better-suited tool instead of just being accepted.
It is bounded (MAX_AGENT_ROUNDS) and self-terminating (a round that proposes nothing new -- empty,
or a repeat of an already-executed call -- ends the loop). **This check-in is not free**: even a
query that only ever needed one tool now costs at least two LLM calls instead of one (the pick, then
the "does anything more help?" check that comes back empty) -- accepted deliberately, since routing
calls are fast next to what most specialists themselves cost, and skipping the check-in on some
cheap heuristic guess would be exactly the kind of shortcut that stops the loop from noticing when
more genuinely would help. The manual "Advanced" tool picker (`forced_tool_calls`) is unaffected: it
is an explicit, one-shot user choice and never loops."""

import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings
from app.orchestrator.answer_composer import build_evidence, compose_answer
from app.orchestrator.confidence import combine_confidence
from app.orchestrator.execution_trace import ExecutionTrace, build_trace
from app.orchestrator.input_validation import validate_images
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall
from app.orchestrator.tool_registry import QueryInput, ToolRegistry, ToolResult, ToolSpec

# Caps how many specialists run concurrently *within one round* (see the parallel-execution block in
# handle_query below) -- not unbounded, since CLAUDE.md documents three resident models as already
# tight on a 16GB machine and concurrent inference adds transient memory pressure on top of that.
MAX_PARALLEL_TOOLS = 3

# Caps how many rounds of tool calls one query can go through (see the module docstring). Not a
# tuning knob to reach for lightly: raising it lets a query that genuinely needs more steps go
# further, but also raises the worst-case latency/cost of a query that never naturally terminates.
# 4 rounds already allows up to MAX_PARALLEL_TOOLS x 4 = 12 tool invocations for one query, against
# a registry of 9 tools -- generous for anything this app's tools can actually chain into today.
MAX_AGENT_ROUNDS = 4


@dataclass
class QueryResult:
    answer_text: str
    confidence: float
    confidence_bucket: str
    evidence_image_paths: list[Path]
    tool_results: list[ToolResult]
    trace: ExecutionTrace


def _synthesize_answer(executed: list[tuple[ToolResult, dict, Optional[str]]]) -> str:
    """The deterministic answer: each executed tool's own text summary, joined. It is the fallback
    for the LLM-phrased answer (answer_composer.py) and what that answer's numbers are checked
    against, so it always matches the trace."""
    if not executed:
        return "No specialist tool produced a result for this query."
    return " ".join(result.text_summary for result, _, _ in executed)


def _redirect_calls(calls: list[ToolCall], registry: ToolRegistry) -> tuple[list[ToolCall], list[str]]:
    """Swap calls a tool says it cannot answer (ToolSpec.redirect) for the tool that can -- e.g. "how many
    buildings" picked for the object detector, which has no building class and returns a stray box or two on
    a scene with dozens. Returns the new list and one note per swap for the trace. A swap is dropped if its
    target is not registered, or if the target is already being called; calls that are not redirected are
    never de-duplicated (asking the detector about two different objects is legitimate)."""
    out: list[ToolCall] = []
    notes: list[str] = []
    for call in calls:
        spec = registry.get(call.tool_name)
        target = spec.redirect(call.arguments) if spec is not None and spec.redirect is not None else None
        if target is not None and registry.get(target[0]) is not None:
            name, arguments, reason = target
            notes.append(reason)
            if any(c.tool_name == name for c in calls) or any(c.tool_name == name for c in out):
                continue
            call = ToolCall(tool_name=name, arguments=arguments)
        out.append(call)
    return out, notes


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
    # Every LLM selection goes through the redirect safety net (see _redirect_calls); a forced list is an
    # explicit choice and is never rewritten. The notes are kept out of `warnings` on purpose: a warning
    # lowers the reported confidence, and swapping in the right tool says nothing against the analysis.
    redirect_notes: list[str] = []

    def select(query: str, history: str = "") -> list[ToolCall]:
        calls, notes = _redirect_calls(llm_provider.select_tools(query, tool_specs, input_summary, history), registry)
        redirect_notes.extend(n for n in notes if n not in redirect_notes)
        return calls

    executed: list[tuple[ToolResult, dict, Optional[str]]] = []
    rounds: list[int] = []  # rounds[i] is which round produced executed[i] -- for the trace only
    skip_warnings: list[str] = []
    incompatible_retried = False  # one retry total across the whole query, not one per round

    for round_num in range(1, MAX_AGENT_ROUNDS + 1):
        if forced_tool_calls is not None:
            # The UI's manual "Advanced" picker bypasses select_tools() entirely -- an explicit,
            # one-shot user choice, never subject to the agentic loop below (see the module docstring).
            tool_calls = forced_tool_calls
        elif round_num == 1:
            tool_calls = select(query_text)
            # Tool-calling is probabilistic, not deterministic -- an LLM can decline to call anything
            # on a borderline-phrased query and then call the right tool on an identical retry
            # (confirmed live: "will it burn, past 1000 days" against wildfire_detection, whose own
            # description already covers "fire risk"/"burning"). Round-1-only: from round 2 on, an
            # empty proposal means "I have enough" (see below), not "reconsider."
            if not tool_calls:
                retry_query = (
                    f"{query_text}\n\nNote: no tool seemed to match on the first pass. Reconsider each "
                    f"tool's description once more -- if the query is even loosely related to what a tool "
                    f"does, call it rather than declining."
                )
                tool_calls = select(retry_query)
        else:
            # Give the model what earlier rounds actually found (the same evidence formatting the
            # final answer is composed from) and let it decide whether more calls would still help.
            # Unlike round 1's own select() (whose failure legitimately blocks the whole query --
            # there is nothing to fall back on yet), a round >= 2 check-in is optional refinement on
            # top of an already-successful round 1: if it fails, that says nothing about the
            # reliability of what's already been found, so it stops the loop instead of discarding a
            # good answer. Found live, not hypothesized: Groq's gpt-oss-120b sometimes invents a
            # fake tool named "none" to signal "I'm done" instead of returning zero calls, which the
            # API itself rejects (400, "tool 'none' not in request.tools") -- surfacing that as a 503
            # would have thrown away a perfectly good round-1 answer over an optional follow-up check.
            try:
                history = build_evidence(executed, [])
                tool_calls = select(query_text, history)
            except Exception:
                break

        # Never re-run a call already executed in an earlier round -- both a correctness guard (an
        # LLM or a naive caller re-proposing the same thing shouldn't duplicate work or evidence) and
        # the loop's own termination signal: once every proposed call is a repeat, there is nothing
        # new left to do. Keyed by JSON (not e.g. tuple(sorted(items()))) so an argument value that
        # happens to be a list/dict some future tool takes still hashes fine -- every parameter in
        # every schema today is a plain string/number, but this doesn't assume that stays true.
        def _call_key(name: str, arguments: dict) -> str:
            return name + "\0" + json.dumps(arguments, sort_keys=True, default=str)

        already_ran = {_call_key(r.tool_name, p) for r, p, _ in executed}
        tool_calls = [c for c in tool_calls if _call_key(c.tool_name, c.arguments) not in already_ran]
        if not tool_calls:
            break  # nothing new proposed this round -- the natural way the loop ends

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
                if not incompatible_retried:
                    incompatible_retried = True
                    retry_query = (
                        f"{query_text}\n\nNote: the tool '{call.tool_name}' is not usable here "
                        f"({incompatible_reason}). Pick a different, compatible tool, or none."
                    )
                    retry_calls = select(retry_query)
                    # Keep the prefix already processed (successes and skips before index i) exactly as
                    # it is; only the still-unprocessed remainder gets replaced. Without this, resetting
                    # i to 0 over a brand-new full list let a tool that already ran successfully get
                    # re-selected and re-executed, duplicating it in the trace -- also drop anything the
                    # retry re-suggests that's already queued in `to_run`, as a second line of defense.
                    # (Checked against `to_run`, not `executed`: this round's execution hasn't happened
                    # yet at retry time -- what must not be re-suggested is anything already resolved
                    # as compatible-and-queued this round.)
                    already_queued = {queued_call.tool_name for queued_call, _ in to_run}
                    tool_calls = tool_calls[:i] + [c for c in retry_calls if c.tool_name not in already_queued]
                    continue
                skip_warnings.append(incompatible_reason)
                i += 1
                continue

            to_run.append((call, spec))
            i += 1

        if not to_run:
            break  # everything proposed this round was unknown/incompatible -- nothing left to run

        # Independent specialists (none reads another's output *within a round*) run concurrently
        # instead of paying each one's latency serially -- safe because every adapter's lazy
        # model-singleton getter is already thread-safe (app/concurrency.py's serialize_first_call),
        # and that lock only guards getting-or-building the model, not the inference call itself.
        # `pool.map` yields results in input order (not completion order), so this round's slice of
        # `executed` -- and everything downstream that depends on it (the trace, tool_results) --
        # stays deterministic regardless of which tool actually finishes first.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(to_run), MAX_PARALLEL_TOOLS)) as pool:
            for outcome in pool.map(lambda item: _run_tool(item, query_input), to_run):
                if outcome[0] == "ok":
                    _, result, arguments, checkpoint_id = outcome
                    executed.append((result, arguments, checkpoint_id))
                    rounds.append(round_num)
                else:
                    _, name, message = outcome
                    skip_warnings.append(f"{name} failed: {message}")

        if forced_tool_calls is not None:
            break  # exactly one round for a manual override, regardless of MAX_AGENT_ROUNDS

    all_warnings = validation.warnings + skip_warnings
    confidences = [result.confidence for result, _, _ in executed]
    confidence, bucket = combine_confidence(confidences, warning_count=len(all_warnings))

    selected_task = ", ".join(result.tool_name for result, _, _ in executed) or "unclassified"
    answer_text = _synthesize_answer(executed)
    if not executed and not all_warnings:
        all_warnings = all_warnings + ["No tool call matched this query."]

    # Phrase the answer in plain language from what the tools found. Any failure keeps the
    # deterministic text above; a note about it goes in the trace, but only after the confidence was
    # computed -- a wording problem says nothing about how sure the analysis itself was.
    trace_warnings = all_warnings + redirect_notes
    if executed and settings.compose_answers:
        composed, composer_problem = compose_answer(llm_provider, query_text, executed, all_warnings, input_summary)
        if composed:
            answer_text = composed
        if composer_problem:
            trace_warnings = trace_warnings + [composer_problem]

    trace = build_trace(
        selected_task=selected_task,
        executed=executed,
        input_summary=input_summary,
        confidence=confidence,
        confidence_bucket=bucket,
        warnings=trace_warnings,
        rounds=rounds,
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
