"""The controller's redirect safety net (ToolSpec.redirect): a deterministic fix for a routing mistake the LLM
makes -- picking the object detector for "how many buildings", which it cannot answer -- recorded in the trace
without lowering the confidence, never applied to the UI's manual tool picker."""

import pytest

from app.orchestrator.confidence import combine_confidence
from app.orchestrator.controller import handle_query
from app.orchestrator.llm_providers.base import ToolCall
from app.orchestrator.tool_registry import QueryInput, ToolRegistry, ToolResult, ToolSpec
from app.specialists import grounding_adapter
from tests.conftest import StubProvider


def _spec(name, handler, redirect=None):
    return ToolSpec(name=name, description=name, parameters_schema={"type": "object", "properties": {}, "required": []},
                    min_images=1, max_images=1, compatible_modalities=["optical"], handler=handler,
                    checkpoint_id=f"{name}.pt", redirect=redirect)


@pytest.fixture
def world():
    """A detector stand-in that records what it was asked and a land-cover stand-in, wired like the real pair."""
    calls = {"detector": [], "land": [], "water": []}

    def detector(qi, args):
        calls["detector"].append(args)
        return ToolResult("text_guided_grounding", "1 box", {"count": 1}, None, 0.29)

    def land(qi, args):
        calls["land"].append(args)
        return ToolResult("land_cover_analysis", "Land cover: building 25%. About 57 building outlines.", {"building_count": 57}, None, 0.8)

    def water(qi, args):
        calls["water"].append(args)
        return ToolResult("water_body_segmentation", "Water covers 12%.", {"water_fraction": 0.12}, None, 0.7)

    registry = ToolRegistry()
    registry.register(_spec("text_guided_grounding", detector, redirect=grounding_adapter.redirect_unanswerable_query))
    registry.register(_spec("land_cover_analysis", land))
    registry.register(_spec("water_body_segmentation", water))
    return registry, calls


def _run(registry, provider_calls, sample_image, query="how many buildings are there?", forced=None):
    return handle_query(query, QueryInput(images=[sample_image]), StubProvider(provider_calls), registry, forced_tool_calls=forced)


def test_a_buildings_question_picked_for_the_detector_is_answered_by_the_land_cover_model(world, sample_image):
    registry, calls = world
    result = _run(registry, [ToolCall("text_guided_grounding", {"query": "how many buildings are there"})], sample_image)

    assert calls["detector"] == [] and calls["land"] == [{}]                       # the detector never ran
    assert [r.tool_name for r in result.tool_results] == ["land_cover_analysis"]
    assert result.trace.selected_task == "land_cover_analysis"
    assert [t["name"] for t in result.trace.tools_used] == ["land_cover_analysis"]  # the trace names what actually ran
    assert result.trace.tools_used[0]["checkpoint_id"] == "land_cover_analysis.pt"
    assert any("no 'building' class" in w and "land-cover model answered" in w for w in result.trace.warnings)


def test_the_substitution_is_recorded_but_does_not_lower_the_confidence(world, sample_image):
    registry, _ = world
    result = _run(registry, [ToolCall("text_guided_grounding", {"query": "buildings"})], sample_image)
    assert result.confidence == combine_confidence([0.8], warning_count=0)[0]


def test_a_water_question_is_redirected_to_the_water_model(world, sample_image):
    registry, calls = world
    result = _run(registry, [ToolCall("text_guided_grounding", {"query": "lake"})], sample_image)
    assert calls["water"] == [{}] and calls["detector"] == [] and result.trace.selected_task == "water_body_segmentation"


def test_the_manual_tool_picker_is_never_rewritten(world, sample_image):
    registry, calls = world
    result = _run(registry, [], sample_image, forced=[ToolCall("text_guided_grounding", {"query": "building"})])
    assert calls["detector"] == [{"query": "building"}] and calls["land"] == []
    assert not any("answered this instead" in w for w in result.trace.warnings)


def test_ordinary_queries_are_left_alone(world, sample_image):
    registry, calls = world
    _run(registry, [ToolCall("text_guided_grounding", {"query": "airplane"})], sample_image)
    assert calls["detector"] == [{"query": "airplane"}] and calls["land"] == []


def test_if_the_model_already_chose_both_the_target_runs_once(world, sample_image):
    registry, calls = world
    result = _run(registry, [ToolCall("text_guided_grounding", {"query": "building"}), ToolCall("land_cover_analysis", {})], sample_image)
    assert len(calls["land"]) == 1 and calls["detector"] == []
    assert [r.tool_name for r in result.tool_results] == ["land_cover_analysis"]


def test_two_legitimate_calls_to_the_same_tool_are_both_kept(world, sample_image):
    registry, calls = world
    _run(registry, [ToolCall("text_guided_grounding", {"query": "airplane"}), ToolCall("text_guided_grounding", {"query": "ship"})], sample_image)
    assert [c["query"] for c in calls["detector"]] == ["airplane", "ship"]


def test_a_redirect_to_a_tool_that_is_not_registered_leaves_the_call_alone(sample_image):
    registry = ToolRegistry()
    registry.register(_spec("text_guided_grounding", lambda qi, a: ToolResult("text_guided_grounding", "1 box", {}, None, 0.3),
                            redirect=grounding_adapter.redirect_unanswerable_query))
    result = _run(registry, [ToolCall("text_guided_grounding", {"query": "building"})], sample_image)
    assert [r.tool_name for r in result.tool_results] == ["text_guided_grounding"]
