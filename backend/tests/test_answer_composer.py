"""The LLM-phrased final answer (orchestrator/answer_composer.py) and its guards. No real LLM: a
scripted provider returns fixed drafts, so these pin down what happens around the model -- the
number check, the corrective retry, the fallbacks -- not what any particular model writes."""

import numpy as np
import pytest

from app.config import settings
from app.orchestrator import answer_composer as composer
from app.orchestrator.controller import handle_query
from app.orchestrator.llm_providers.base import ToolCall
from app.orchestrator.tool_registry import QueryInput, ToolRegistry, ToolResult, ToolSpec
from tests.conftest import StubProvider


class ScriptedProvider(StubProvider):
    """select_tools returns fixed calls; generate_text pops scripted drafts (an Exception is raised)."""

    def __init__(self, calls=None, drafts=()):
        super().__init__(calls or [])
        self.drafts = list(drafts)
        self.prompts = []

    def generate_text(self, system, user):
        self.prompts.append(user)
        draft = self.drafts.pop(0)
        if isinstance(draft, Exception):
            raise draft
        return draft


def _executed(summary="Found 39 region(s) matching 'airplane' (detection scores 0.31-0.44).", data=None, confidence=0.44):
    data = data if data is not None else {"count": 39, "by_phrase": {"airplane": 39}, "query": "airplane"}
    return [(ToolResult("text_guided_grounding", summary, data, None, confidence), {"query": "airplane"}, "ckpt")]


# ---------------------------------------------------------------- the number check

def test_numbers_quoted_verbatim_rounded_or_as_percent_are_supported():
    evidence = "Water covers approximately 58.3% of the image. Details: {\"water_fraction\": 0.5832} score 0.76"
    assert composer.unsupported_numbers("About 58% is water (0.5832), or 58.3 percent.", evidence) == []


def test_invented_multi_digit_numbers_are_flagged():
    assert composer.unsupported_numbers("I count 40 airplanes.", "Found 39 region(s)") == ["40"]


def test_single_digit_integers_are_not_checked():
    # 'one of the ...', 'two runways' -- ordinary words far more often than invented counts
    assert composer.unsupported_numbers("There are 3 of them, one per gate.", "Found 39 region(s)") == []


def test_numbers_from_the_question_and_input_summary_are_allowed():
    provider = ScriptedProvider(drafts=["The 1024x731 image was searched for the last 1000 days of fires."])
    text, problem = composer.compose_answer(provider, "fires in the past 1000 days?", _executed("x", {}), [], "1 image(s)\n- a.tif: TIFF 1024x731")
    assert problem is None and text.startswith("The 1024x731")


# ---------------------------------------------------------------- the evidence block

def test_evidence_summarises_detection_lists_instead_of_dumping_boxes():
    detections = [{"phrase": "airplane", "bbox_xyxy": [1.5, 2.5, 30.5, 40.5], "score": 0.3 + i / 1000} for i in range(39)]
    block = composer.build_evidence(_executed(data={"detections": detections, "count": 39}), [])
    assert '"count": 39' in block and "score_range" in block
    assert "bbox_xyxy" not in block  # 39 boxes would swamp the prompt for no benefit


def test_evidence_lists_warnings_and_uses_plain_names():
    block = composer.build_evidence(_executed(), ["water_body_segmentation failed: no checkpoint"])
    assert "object detector" in block and "text_guided_grounding" not in block
    assert "Problems reported while running: water_body_segmentation failed: no checkpoint" in block


# ---------------------------------------------------------------- compose_answer

def test_a_clean_draft_is_returned_as_is():
    provider = ScriptedProvider(drafts=["About 39 airplanes are visible."])
    assert composer.compose_answer(provider, "how many planes?", _executed(), [], "1 image(s)") == ("About 39 airplanes are visible.", None)


def test_invented_numbers_trigger_one_corrective_retry():
    provider = ScriptedProvider(drafts=["Roughly 45 airplanes.", "About 39 airplanes."])
    text, problem = composer.compose_answer(provider, "how many planes?", _executed(), [], "1 image(s)")
    assert (text, problem) == ("About 39 airplanes.", None)
    assert "not in the tool results: 45" in provider.prompts[1]


def test_two_bad_drafts_fall_back_with_a_reason():
    provider = ScriptedProvider(drafts=["Roughly 45 airplanes.", "Around 47 airplanes."])
    text, problem = composer.compose_answer(provider, "how many planes?", _executed(), [], "1 image(s)")
    assert text is None and "numbers the tools did not report" in problem


def test_a_provider_without_text_generation_is_silent():
    # StubProvider inherits LLMProvider.generate_text, which raises NotImplementedError
    assert composer.compose_answer(StubProvider([]), "q", _executed(), [], "1 image(s)") == (None, None)


def test_a_provider_error_is_reported_not_raised():
    provider = ScriptedProvider(drafts=[RuntimeError("429 rate limited")])
    text, problem = composer.compose_answer(provider, "q", _executed(), [], "1 image(s)")
    assert text is None and "429 rate limited" in problem


@pytest.mark.parametrize("draft", ["", "x" * 3500])
def test_empty_or_overlong_drafts_are_rejected(draft):
    text, problem = composer.compose_answer(ScriptedProvider(drafts=[draft]), "q", _executed(), [], "1 image(s)")
    assert text is None and "unusable draft" in problem


# ---------------------------------------------------------------- inside the controller

def _fake_registry(confidence=0.44):
    def handler(query_input, arguments):
        return ToolResult("text_guided_grounding", "Found 39 region(s) matching 'airplane'.", {"count": 39}, None, confidence)

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="text_guided_grounding", description="d", parameters_schema={"type": "object", "properties": {}},
        min_images=1, max_images=1, compatible_modalities=["optical"], handler=handler,
    ))
    return registry


def test_the_controller_uses_the_composed_answer(sample_image):
    provider = ScriptedProvider([ToolCall("text_guided_grounding", {"query": "airplane"})], ["There are about 39 airplanes."])
    result = handle_query("how many planes?", QueryInput(images=[sample_image]), provider, _fake_registry())
    assert result.answer_text == "There are about 39 airplanes."
    assert result.tool_results[0].text_summary == "Found 39 region(s) matching 'airplane'."  # the raw summary is still there


def test_a_composer_failure_keeps_the_deterministic_answer_and_notes_it_without_lowering_confidence(sample_image):
    calls = [ToolCall("text_guided_grounding", {"query": "airplane"})]
    baseline = handle_query("q", QueryInput(images=[sample_image]), StubProvider(calls), _fake_registry())
    provider = ScriptedProvider(calls, [RuntimeError("boom")])
    result = handle_query("q", QueryInput(images=[sample_image]), provider, _fake_registry())
    assert result.answer_text == "Found 39 region(s) matching 'airplane'."
    assert any("boom" in w for w in result.trace.warnings)
    assert result.confidence == baseline.confidence  # a wording problem says nothing about the analysis


def test_composing_can_be_switched_off(sample_image, monkeypatch):
    monkeypatch.setattr(settings, "compose_answers", False)
    provider = ScriptedProvider([ToolCall("text_guided_grounding", {})], ["should never be used"])
    result = handle_query("q", QueryInput(images=[sample_image]), provider, _fake_registry())
    assert result.answer_text == "Found 39 region(s) matching 'airplane'."
    assert provider.prompts == []


def test_nothing_is_composed_when_no_tool_ran(sample_image):
    provider = ScriptedProvider([], ["should never be used"])
    result = handle_query("q", QueryInput(images=[sample_image]), provider, _fake_registry())
    assert result.answer_text == "No specialist tool produced a result for this query."
    assert provider.prompts == []
