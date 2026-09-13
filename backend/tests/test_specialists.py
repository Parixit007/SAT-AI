"""Smoke tests for the specialist adapters themselves (bypassing the LLM/orchestrator entirely) --
non-empty/well-shaped output, confidence in range, no crashes. These check plumbing, not model
accuracy -- there's no labeled satellite imagery bundled yet (see CLAUDE.md's data-acquisition
follow-up), so a real accuracy check has to wait for that."""

import pytest

from app.orchestrator.tool_registry import QueryInput
from app.specialists.water_segmentation_adapter import _handle as water_handle


def test_water_segmentation_smoke(sample_image):
    result = water_handle(QueryInput(images=[sample_image]), {})

    assert result.tool_name == "water_body_segmentation"
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.structured_data["water_fraction"] <= 1.0
    assert result.evidence_image_path is not None
    assert result.evidence_image_path.exists()


def test_water_segmentation_respects_custom_threshold(sample_image):
    low = water_handle(QueryInput(images=[sample_image]), {"threshold": 0.01})
    high = water_handle(QueryInput(images=[sample_image]), {"threshold": 0.99})

    # A near-0 threshold should never classify less water than a near-1 threshold.
    assert low.structured_data["water_fraction"] >= high.structured_data["water_fraction"]
    # Regression check: confidence is documented as [0.5, 1.0] regardless of threshold. The
    # previous formula (mean(where(mask, probs, 1-probs))) only stayed in that range at the
    # default threshold=0.5 -- it would silently report ~threshold-magnitude numbers at the
    # extremes tested here instead.
    assert 0.5 <= low.confidence <= 1.0
    assert 0.5 <= high.confidence <= 1.0


def test_grounding_smoke(sample_image):
    pytest.importorskip("groundingdino", reason="groundingdino not installed -- see CLAUDE.md Setup")
    from app.specialists.grounding_adapter import _handle as grounding_handle

    result = grounding_handle(QueryInput(images=[sample_image]), {"query": "building"})

    assert result.tool_name == "text_guided_grounding"
    assert 0.0 <= result.confidence <= 1.0
    for det in result.structured_data["detections"]:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        assert 0 <= x1 <= 128 and 0 <= x2 <= 128
        assert 0 <= y1 <= 128 and 0 <= y2 <= 128
