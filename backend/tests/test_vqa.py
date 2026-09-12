"""Tests for the VQA specialist that don't require the ~11.7GB gated PaliGemma download.

Real answer-quality checks need the model plus an HF token, and are run manually against
RSVQA-LR's own ground-truth answers (see the plan's verification section) rather than here -- a
6GB model load per CI run isn't a reasonable default."""

import pytest

from app.orchestrator.controller import handle_query
from app.orchestrator.llm_providers.base import ToolCall
from app.orchestrator.tool_registry import QueryInput, ToolResult
from app.specialists import build_default_registry
from tests.conftest import StubProvider


def test_vqa_tool_is_registered():
    spec = build_default_registry().get("visual_question_answering")

    assert spec is not None
    assert spec.uses_images is True
    assert spec.min_images == 1 and spec.max_images == 1
    assert spec.requires_location is False
    assert spec.checkpoint_id == "paligemma-3b-ft-rsvqa-lr-224"
    assert "question" in spec.parameters_schema["required"]


def test_vqa_handler_raises_clean_error_without_hf_token(monkeypatch, sample_image):
    # Force the unconfigured state rather than depending on whatever .env happens to hold on this
    # machine -- a real token may well be present in dev.
    import app.specialists.vqa_adapter as adapter

    monkeypatch.setattr(adapter.settings, "hf_token", None)
    monkeypatch.setattr(adapter, "_vqa_tool", None)

    with pytest.raises(RuntimeError, match="VQA model is not configured"):
        adapter._handle(QueryInput(images=[sample_image]), {"question": "Is there a road?"})


def test_vqa_routes_end_to_end_with_model_mocked(monkeypatch, sample_image):
    """The orchestrator path -- routing, execution, trace -- with only the model call faked."""
    import app.specialists.vqa_adapter as adapter

    class FakeTool:
        def answer(self, image_path, question):
            return {"answer": "yes", "confidence": 0.93}

    monkeypatch.setattr(adapter, "_get_tool", lambda: FakeTool())

    registry = build_default_registry()
    provider = StubProvider(
        [ToolCall(tool_name="visual_question_answering", arguments={"question": "Is there a road?"})]
    )

    result = handle_query("Is there a road?", QueryInput(images=[sample_image]), provider, registry)

    assert result.trace.selected_task == "visual_question_answering"
    assert result.answer_text == "yes"
    assert result.confidence == pytest.approx(0.93, abs=0.01)
    assert result.trace.tools_used[0]["checkpoint_id"] == "paligemma-3b-ft-rsvqa-lr-224"


def test_vqa_rejected_when_no_image_supplied(monkeypatch):
    """VQA needs exactly one image -- a location-only query must not reach the handler."""
    import app.specialists.vqa_adapter as adapter

    def _explode():
        raise AssertionError("handler must not be reached without an image")

    monkeypatch.setattr(adapter, "_get_tool", _explode)

    from app.orchestrator.tool_registry import LatLon

    registry = build_default_registry()
    provider = StubProvider(
        [ToolCall(tool_name="visual_question_answering", arguments={"question": "Is there a road?"})]
    )

    result = handle_query(
        "Is there a road?", QueryInput(location=LatLon(lat=12.9, lon=77.6)), provider, registry
    )

    assert result.trace.tools_used == []
    assert any("needs between 1 and 1 image" in w for w in result.trace.warnings)


def test_vqa_result_carries_no_evidence_image(monkeypatch, sample_image):
    """A text answer has nothing to render -- evidence_image_path stays None, and the API layer
    must cope with that (it builds evidence_image_urls by filtering on it)."""
    import app.specialists.vqa_adapter as adapter

    class FakeTool:
        def answer(self, image_path, question):
            return {"answer": "urban", "confidence": 0.8}

    monkeypatch.setattr(adapter, "_get_tool", lambda: FakeTool())

    result: ToolResult = adapter._handle(QueryInput(images=[sample_image]), {"question": "rural or urban?"})

    assert result.evidence_image_path is None
    assert result.structured_data["answer"] == "urban"
