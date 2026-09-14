"""Smoke tests for the specialist adapters themselves (bypassing the LLM/orchestrator entirely) --
non-empty/well-shaped output, confidence in range, no crashes. These check plumbing, not model
accuracy -- there's no labeled satellite imagery bundled yet (see CLAUDE.md's data-acquisition
follow-up), so a real accuracy check has to wait for that."""

import pytest

from app.orchestrator.tool_registry import QueryInput
from app.specialists.change_detection_adapter import TOOL_SPEC as CHANGE_DETECTION_TOOL_SPEC
from app.specialists.change_detection_adapter import _handle as change_handle
from app.specialists.fusion_adapter import TOOL_SPEC as FUSION_TOOL_SPEC
from app.specialists.fusion_adapter import _handle as fusion_handle
from app.specialists.water_segmentation_adapter import _handle as water_handle


def test_change_detection_tool_is_registered():
    spec = CHANGE_DETECTION_TOOL_SPEC

    assert spec.name == "change_detection"
    assert spec.uses_images is True
    assert spec.min_images == 2 and spec.max_images == 2
    assert spec.compatible_modalities == ["optical"]
    assert spec.requires_location is False
    assert spec.checkpoint_id is None  # Stage 1 is training-free


def test_fusion_tool_is_registered():
    spec = FUSION_TOOL_SPEC

    assert spec.name == "optical_sar_fusion"
    assert spec.uses_images is True
    assert spec.min_images == 2 and spec.max_images == 2
    assert spec.compatible_modalities == ["optical", "sar"]
    assert spec.required_modality_pair == ("optical", "sar")
    assert spec.requires_location is False
    assert spec.checkpoint_id is None  # Stage 1 is training-free


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


def test_change_detection_finds_the_known_injected_region(change_pair_images):
    before, after, expected_bbox = change_pair_images

    result = change_handle(QueryInput(images=[before, after]), {})

    assert result.tool_name == "change_detection"
    assert 0.0 <= result.confidence <= 1.0
    assert result.structured_data["largest_region_bbox"] == expected_bbox
    # The injected block is a 40x40 region in a 128x128 image -- ~9.8% of the area.
    assert result.structured_data["change_fraction"] == pytest.approx(1600 / (128 * 128), abs=0.005)
    assert result.evidence_image_path is not None
    assert result.evidence_image_path.exists()


def test_change_detection_reports_no_change_for_identical_images(sample_image):
    result = change_handle(QueryInput(images=[sample_image, sample_image]), {})

    assert result.structured_data["change_fraction"] == 0.0
    assert result.structured_data["largest_region_bbox"] is None


def test_fusion_detects_known_sar_regions_and_agrees_with_optical(optical_sar_pair, monkeypatch):
    """water_segmentation's own model is mocked here (not re-tested -- test_water_segmentation_*
    already covers it) so this isolates fusion_adapter's own SAR-detection + reconciliation logic:
    the injected SAR water block is ~9% of the 100x100 image, so a mocked optical read of 10%
    should land within WATER_AGREEMENT_TOLERANCE and report 'agree'."""
    import app.specialists.fusion_adapter as adapter

    optical, sar = optical_sar_pair

    class FakeWaterTool:
        def segment(self, image_path, threshold=0.5):
            return {"water_fraction": 0.10, "confidence": 0.9}

    monkeypatch.setattr(adapter, "_get_water_tool", lambda: FakeWaterTool())

    result = fusion_handle(QueryInput(images=[optical, sar]), {})

    assert result.tool_name == "optical_sar_fusion"
    assert 0.0 <= result.confidence <= 1.0
    # Injected blocks are each 30x30 in a 100x100 image -- 9% of the area. Recursive-Otsu doesn't
    # cut exactly at the injected block's own value range -- it settles wherever separation is
    # maximized against the random background, typically a little outside the true edge -- so this
    # checks "found roughly the right amount," not pixel-exact area (that's what the bbox-based
    # change_detection test above already verifies more strictly, since that one has a much larger
    # value gap between background and injected region).
    assert result.structured_data["water_fraction_sar"] == pytest.approx(0.09, abs=0.03)
    assert result.structured_data["builtup_fraction_sar"] == pytest.approx(0.09, abs=0.03)
    assert result.structured_data["water_agreement"] == "agree"
    assert result.evidence_image_path is not None
    assert result.evidence_image_path.exists()


def test_fusion_surfaces_disagreement_between_modalities(optical_sar_pair, monkeypatch):
    import app.specialists.fusion_adapter as adapter

    optical, sar = optical_sar_pair

    class FakeWaterTool:
        def segment(self, image_path, threshold=0.5):
            return {"water_fraction": 0.80, "confidence": 0.9}  # wildly different from SAR's ~9%

    monkeypatch.setattr(adapter, "_get_water_tool", lambda: FakeWaterTool())

    result = fusion_handle(QueryInput(images=[optical, sar]), {})

    assert result.structured_data["water_agreement"] == "disagree"
    assert "DISAGREE" in result.text_summary


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
