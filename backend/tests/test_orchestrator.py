from app.orchestrator.controller import handle_query
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall
from app.orchestrator.tool_registry import LatLon, QueryInput, ToolRegistry, ToolResult, ToolSpec
from app.specialists import build_default_registry
from tests.conftest import StubProvider


def test_routes_to_correct_tool(sample_image):
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="water_body_segmentation", arguments={})])

    result = handle_query("How much water is in this image?", QueryInput(images=[sample_image]), provider, registry)

    assert result.trace.selected_task == "water_body_segmentation"
    assert len(result.trace.tools_used) == 1
    assert result.trace.tools_used[0]["name"] == "water_body_segmentation"
    assert result.trace.tools_used[0]["checkpoint_id"] == "water_body_unet_final.pt"
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_bucket in {"High", "Medium", "Low"}


def test_routes_bitemporal_query_to_change_detection(change_pair_images):
    before, after, _ = change_pair_images
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="change_detection", arguments={})])

    result = handle_query(
        "What changed between these two dates?", QueryInput(images=[before, after]), provider, registry
    )

    assert result.trace.selected_task == "change_detection"
    assert result.trace.tools_used[0]["name"] == "change_detection"
    # None for Stage 1 (training-free), the checkpoint file for Stage 2 -- which one is active
    # depends on a gitignored file, so compare against what the registry itself reports.
    assert result.trace.tools_used[0]["checkpoint_id"] == registry.get("change_detection").checkpoint_id
    assert 0.0 <= result.confidence <= 1.0


def test_routes_optical_sar_pair_to_fusion(optical_sar_pair):
    optical, sar = optical_sar_pair
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="optical_sar_fusion", arguments={})])

    result = handle_query(
        "Use the optical and SAR images together to find water and built-up regions.",
        QueryInput(images=[optical, sar]),
        provider,
        registry,
    )

    assert result.trace.selected_task == "optical_sar_fusion"
    assert result.trace.tools_used[0]["name"] == "optical_sar_fusion"
    # None for Stage 1 only, the classifier checkpoint once Stage 2 is installed -- same reasoning
    # as change_detection above: compare against what the registry itself reports, not a hardcoded
    # assumption that breaks the moment a developer has the (gitignored) checkpoint on disk.
    assert result.trace.tools_used[0]["checkpoint_id"] == registry.get("optical_sar_fusion").checkpoint_id
    assert 0.0 <= result.confidence <= 1.0


def test_fusion_rejects_two_images_of_the_same_modality(georeferenced_tif, tmp_path):
    """required_modality_pair (tool_registry.py): compatible_modalities=["optical","sar"] alone
    would wrongly accept two optical images (both individually allowed) -- required_modality_pair
    must catch that neither is actually SAR."""
    import shutil

    second_optical = tmp_path / "second_optical.tif"
    shutil.copy(georeferenced_tif, second_optical)

    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="optical_sar_fusion", arguments={})])

    result = handle_query(
        "fuse these", QueryInput(images=[georeferenced_tif, second_optical]), provider, registry
    )

    assert result.trace.tools_used == []
    assert any("needs exactly one optical and one sar" in w for w in result.trace.warnings)


def test_incompatible_input_is_rejected_not_executed(sample_image, tmp_path):
    import shutil

    second_image = tmp_path / "sample2.png"
    shutil.copy(sample_image, second_image)

    registry = build_default_registry()
    # water_body_segmentation only accepts exactly 1 image -- 2 images should never execute it.
    provider = StubProvider([ToolCall(tool_name="water_body_segmentation", arguments={})])

    result = handle_query("water?", QueryInput(images=[sample_image, second_image]), provider, registry)

    assert result.trace.tools_used == []
    assert any("needs between 1 and 1 image" in w for w in result.trace.warnings)
    assert result.confidence == 0.0
    assert result.confidence_bucket == "Low"


def test_unknown_tool_name_is_skipped_gracefully(sample_image):
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="not_a_real_tool", arguments={})])

    result = handle_query("do something", QueryInput(images=[sample_image]), provider, registry)

    assert result.trace.tools_used == []
    assert any("unknown tool" in w for w in result.trace.warnings)


def test_no_tool_selected_is_a_valid_auditable_outcome(sample_image):
    registry = build_default_registry()
    provider = StubProvider([])  # LLM decided nothing fits

    result = handle_query("what's the meaning of life?", QueryInput(images=[sample_image]), provider, registry)

    assert result.trace.tools_used == []
    assert result.trace.selected_task == "unclassified"
    assert "No tool call matched this query." in result.trace.warnings


def test_execution_trace_has_required_fields(sample_image):
    """The spec requires an 'auditable execution summary containing the selected task,
    model/tool names, and key parameters' (problem_statement.txt) -- checks every required field
    actually holds a correct value for this scenario. (Previously this only asserted `hasattr` on
    a dataclass's own declared fields, which is trivially true for any successfully-constructed
    instance regardless of whether the values themselves are right -- e.g. a build that always
    hardcoded confidence=0 would still have passed.)"""
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="water_body_segmentation", arguments={})])

    result = handle_query("water?", QueryInput(images=[sample_image]), provider, registry)
    trace = result.trace

    assert trace.selected_task == "water_body_segmentation"
    assert trace.tools_used == [
        {"name": "water_body_segmentation", "params": {}, "checkpoint_id": "water_body_unet_final.pt"}
    ]
    assert "1 image(s)" in trace.input_summary
    assert 0.0 <= trace.confidence <= 1.0
    assert trace.confidence_bucket in {"High", "Medium", "Low"}
    assert trace.warnings == []
    assert trace.timestamp  # non-empty ISO timestamp


def test_missing_image_is_a_validation_error(tmp_path):
    registry = build_default_registry()
    provider = StubProvider([])
    missing_path = tmp_path / "does_not_exist.png"

    result = handle_query("water?", QueryInput(images=[missing_path]), provider, registry)

    assert result.confidence == 0.0
    assert result.trace.selected_task == "input_validation_failed"


def test_tool_failure_becomes_a_trace_warning_not_a_crash(sample_image):
    """A specialist blowing up (uninstalled dependency, gated model, upstream API refusing) must
    not take the whole query down -- it gets recorded in the trace and the request still returns.
    Found live: an unhandled ModuleNotFoundError from the grounding tool produced a bare 500."""

    def exploding_handler(query_input, arguments):
        raise ModuleNotFoundError("No module named 'groundingdino'")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="exploding_tool",
            description="Always fails.",
            parameters_schema={"type": "object", "properties": {}},
            min_images=1,
            max_images=1,
            compatible_modalities=["optical"],
            handler=exploding_handler,
        )
    )
    provider = StubProvider([ToolCall(tool_name="exploding_tool", arguments={})])

    result = handle_query("find the ships", QueryInput(images=[sample_image]), provider, registry)

    assert result.trace.tools_used == []
    assert any("exploding_tool failed" in w and "groundingdino" in w for w in result.trace.warnings)
    assert result.confidence == 0.0


def test_retry_does_not_reexecute_an_already_succeeded_tool(sample_image):
    """Regression test: the LLM selects [water_body_segmentation, groundwater_potential] with no
    location given, so groundwater_potential is incompatible and triggers the one-retry path. If
    the retry naively re-suggests water_body_segmentation (which already ran and succeeded), it
    must not run a second time -- previously it did, duplicating the trace entry and re-running an
    already-successful (and possibly expensive) specialist."""

    class RetryingProvider(LLMProvider):
        def __init__(self):
            self.call_count = 0

        def select_tools(self, query, tool_specs, input_summary):
            self.call_count += 1
            if self.call_count == 1:
                return [
                    ToolCall(tool_name="water_body_segmentation", arguments={}),
                    ToolCall(tool_name="groundwater_potential", arguments={}),
                ]
            # Retry: naively re-suggests the tool that already ran -- exactly the failure mode found.
            return [ToolCall(tool_name="water_body_segmentation", arguments={})]

    registry = build_default_registry()
    provider = RetryingProvider()

    result = handle_query("water and groundwater?", QueryInput(images=[sample_image]), provider, registry)

    assert provider.call_count == 2  # confirms the retry path was actually exercised
    assert len(result.trace.tools_used) == 1
    assert result.trace.tools_used[0]["name"] == "water_body_segmentation"


def test_no_images_and_no_location_is_a_validation_error():
    registry = build_default_registry()
    provider = StubProvider([])

    result = handle_query("hello?", QueryInput(), provider, registry)

    assert result.confidence == 0.0
    assert result.trace.selected_task == "input_validation_failed"


# --- location-based tool routing (generic mechanism -- see tool_registry.QueryInput/is_compatible;
# the concrete groundwater tool gets its own dedicated tests) -----------------------------------


def _fake_location_tool_registry() -> ToolRegistry:
    def handler(query_input: QueryInput, arguments: dict) -> ToolResult:
        lat, lon = query_input.location.lat, query_input.location.lon
        return ToolResult(
            tool_name="fake_location_tool",
            text_summary=f"Looked up ({lat}, {lon}).",
            structured_data={"lat": lat, "lon": lon},
            evidence_image_path=None,
            confidence=0.9,
        )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="fake_location_tool",
            description="A fake location-only tool for testing.",
            parameters_schema={"type": "object", "properties": {}},
            min_images=0,
            max_images=0,
            compatible_modalities=[],
            handler=handler,
            uses_images=False,
            requires_location=True,
        )
    )
    return registry


def test_location_only_tool_runs_with_no_images():
    registry = _fake_location_tool_registry()
    provider = StubProvider([ToolCall(tool_name="fake_location_tool", arguments={})])

    result = handle_query(
        "should I dig a well here?", QueryInput(location=LatLon(lat=12.9, lon=77.6)), provider, registry
    )

    assert result.trace.selected_task == "fake_location_tool"
    assert result.confidence == 0.9


def test_location_only_tool_rejected_without_location(sample_image):
    registry = _fake_location_tool_registry()
    provider = StubProvider([ToolCall(tool_name="fake_location_tool", arguments={})])

    # Images present but no location -- fake_location_tool ignores images entirely (uses_images=False)
    # and should be rejected purely for missing the location it requires.
    result = handle_query("well?", QueryInput(images=[sample_image]), provider, registry)

    assert result.trace.tools_used == []
    assert any("needs a location" in w for w in result.trace.warnings)


def test_input_summary_includes_location_when_present():
    registry = _fake_location_tool_registry()
    provider = StubProvider([ToolCall(tool_name="fake_location_tool", arguments={})])

    result = handle_query(
        "well?", QueryInput(location=LatLon(lat=12.9, lon=77.6)), provider, registry
    )

    assert "12.9" in result.trace.input_summary and "77.6" in result.trace.input_summary


# --- manual tool override + parallel execution (see the UI-overhaul plan) ----------------------


def test_forced_tool_calls_bypasses_llm_selection(sample_image):
    """QueryRequest.forced_tools (the UI's manual 'Advanced' picker) must run exactly what's
    forced, never consulting the LLM provider at all -- proven with a provider scripted to select
    a *different* tool than what's forced; if the bypass didn't work, that tool would run instead."""
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="text_guided_grounding", arguments={"query": "ship"})])
    forced = [ToolCall(tool_name="water_body_segmentation", arguments={})]

    result = handle_query(
        "irrelevant query text", QueryInput(images=[sample_image]), provider, registry, forced_tool_calls=forced
    )

    assert result.trace.selected_task == "water_body_segmentation"
    assert [t["name"] for t in result.trace.tools_used] == ["water_body_segmentation"]


def test_multiple_independent_tools_all_execute_and_preserve_call_order(sample_image):
    """Two independent tools selected together must both execute (not just the first, the way the
    old sequential loop that this replaces would still have done) and land in tool_results/the
    trace in the *original call order*, regardless of which finishes first -- the tool registered
    to finish slower is listed first in the call list, so an implementation that reordered by
    completion time (rather than ThreadPoolExecutor.map's documented input-order guarantee) would
    fail this."""
    import time

    calls_started = []

    def make_handler(name: str, delay: float):
        def handler(query_input: QueryInput, arguments: dict) -> ToolResult:
            calls_started.append(name)
            time.sleep(delay)
            return ToolResult(
                tool_name=name, text_summary=f"{name} done.", structured_data={},
                evidence_image_path=None, confidence=0.5,
            )
        return handler

    def make_spec(name: str, delay: float) -> ToolSpec:
        return ToolSpec(
            name=name, description="", parameters_schema={"type": "object", "properties": {}},
            min_images=1, max_images=1, compatible_modalities=["optical", "sar", "unknown"],
            handler=make_handler(name, delay),
        )

    registry = ToolRegistry()
    registry.register(make_spec("slow_tool", 0.15))
    registry.register(make_spec("fast_tool", 0.0))
    provider = StubProvider([ToolCall(tool_name="slow_tool", arguments={}), ToolCall(tool_name="fast_tool", arguments={})])

    result = handle_query("run both", QueryInput(images=[sample_image]), provider, registry)

    assert set(calls_started) == {"slow_tool", "fast_tool"}  # both actually ran
    assert [r.tool_name for r in result.tool_results] == ["slow_tool", "fast_tool"]
    assert [t["name"] for t in result.trace.tools_used] == ["slow_tool", "fast_tool"]


# --- retry when the LLM selects zero tools (see the "will it burn" live finding) ----------------


def test_empty_first_selection_gets_one_retry_and_can_recover(sample_image):
    """Tool-calling is probabilistic -- confirmed live, the same query ("will it burn, past 1000
    days") against the real Groq-backed orchestrator declined to call anything once, then routed
    correctly to wildfire_detection on an identical retry. A provider that returns [] on its first
    call and a real tool on its second must actually recover, not just report the empty result."""

    class DeclineThenPickProvider(LLMProvider):
        def __init__(self):
            self.call_count = 0

        def select_tools(self, query, tool_specs, input_summary):
            self.call_count += 1
            if self.call_count == 1:
                return []
            return [ToolCall(tool_name="water_body_segmentation", arguments={})]

    registry = build_default_registry()
    provider = DeclineThenPickProvider()

    result = handle_query("vague query", QueryInput(images=[sample_image]), provider, registry)

    assert provider.call_count == 2  # confirms the retry path was actually exercised
    assert result.trace.selected_task == "water_body_segmentation"
    assert result.trace.warnings == []


def test_still_empty_after_retry_is_a_clean_no_match_not_a_crash(sample_image):
    """The retry is one extra attempt, not a loop -- a provider that declines twice in a row still
    ends in the same honest "no tool matched" outcome, just after two calls instead of one."""

    class AlwaysDeclineProvider(LLMProvider):
        def __init__(self):
            self.call_count = 0

        def select_tools(self, query, tool_specs, input_summary):
            self.call_count += 1
            return []

    registry = build_default_registry()
    provider = AlwaysDeclineProvider()

    result = handle_query("vague query", QueryInput(images=[sample_image]), provider, registry)

    assert provider.call_count == 2
    assert result.trace.selected_task == "unclassified"
    assert "No tool call matched this query." in result.trace.warnings


def test_forced_empty_tool_list_is_not_retried(sample_image):
    """forced_tool_calls=[] (the UI's Advanced picker with no tools checked) is a deliberate 'run
    nothing' request, not an LLM failure -- it must not trigger the empty-selection retry, which
    would incorrectly fall back to asking the LLM to choose after all."""
    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="water_body_segmentation", arguments={})])

    result = handle_query(
        "irrelevant text", QueryInput(images=[sample_image]), provider, registry, forced_tool_calls=[]
    )

    assert result.trace.selected_task == "unclassified"
    assert result.trace.tools_used == []
