"""Smoke tests for the specialist adapters themselves (bypassing the LLM/orchestrator entirely) --
non-empty/well-shaped output, confidence in range, no crashes. These check plumbing, not model
accuracy -- there's no labeled satellite imagery bundled yet (see CLAUDE.md's data-acquisition
follow-up), so a real accuracy check has to wait for that."""

import pytest

import app.specialists.change_detection_adapter as change_adapter
import app.specialists.fusion_adapter as fusion_adapter
from app.orchestrator.tool_registry import QueryInput
from app.specialists.change_detection_adapter import TOOL_SPEC as CHANGE_DETECTION_TOOL_SPEC
from app.specialists.change_detection_adapter import _handle as change_handle
from app.specialists.fusion_adapter import TOOL_SPEC as FUSION_TOOL_SPEC
from app.specialists.fusion_adapter import _handle as fusion_handle
from app.specialists.water_segmentation_adapter import _handle as water_handle


@pytest.fixture
def stage1_only(monkeypatch):
    """The Stage 2 checkpoint is gitignored, so whether change_detection runs Stage 1 or Stage 2 is
    decided by a file that exists on a developer's machine but not in CI (change_detection_adapter
    .USE_STAGE2, fixed at import). Tests about Stage 1's own behaviour pin it off rather than
    silently passing in one environment and failing in the other."""
    monkeypatch.setattr(change_adapter, "USE_STAGE2", False)


@pytest.fixture
def fusion_stage1_only(monkeypatch):
    """Same reasoning as stage1_only above, for optical_sar_fusion's own optional Stage 2
    checkpoint (fusion_adapter.USE_CLASSIFIER)."""
    monkeypatch.setattr(fusion_adapter, "USE_CLASSIFIER", False)


def test_change_detection_tool_is_registered():
    spec = CHANGE_DETECTION_TOOL_SPEC

    assert spec.name == "change_detection"
    assert spec.uses_images is True
    assert spec.min_images == 2 and spec.max_images == 2
    assert spec.compatible_modalities == ["optical"]
    assert spec.requires_location is False
    # Stage 1 is training-free (no checkpoint); Stage 2 reports its checkpoint file -- whichever the
    # adapter decided on at import, the trace must agree with it.
    expected = change_adapter.CHANGE_SEG_CHECKPOINT.name if change_adapter.USE_STAGE2 else None
    assert spec.checkpoint_id == expected


def test_fusion_tool_is_registered():
    spec = FUSION_TOOL_SPEC

    assert spec.name == "optical_sar_fusion"
    assert spec.uses_images is True
    assert spec.min_images == 2 and spec.max_images == 2
    assert spec.compatible_modalities == ["optical", "sar"]
    assert spec.required_modality_pair == ("optical", "sar")
    assert spec.requires_location is False
    # Stage 1 is training-free; Stage 2 reports its checkpoint when a developer has it installed --
    # same "the trace must agree with what the adapter actually decided" reasoning as change_detection.
    expected = fusion_adapter.FUSION_CLASSIFIER_CHECKPOINT.name if fusion_adapter.USE_CLASSIFIER else None
    assert spec.checkpoint_id == expected


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


def test_change_detection_finds_the_known_injected_region(change_pair_images, stage1_only):
    before, after, expected_bbox = change_pair_images

    result = change_handle(QueryInput(images=[before, after]), {})

    assert result.tool_name == "change_detection"
    assert 0.0 <= result.confidence <= 1.0
    assert result.structured_data["largest_region_bbox"] == expected_bbox
    # The injected block is a 40x40 region in a 128x128 image -- ~9.8% of the area.
    assert result.structured_data["change_fraction"] == pytest.approx(1600 / (128 * 128), abs=0.005)
    assert result.evidence_image_path is not None
    assert result.evidence_image_path.exists()


def test_change_detection_reports_no_change_for_identical_images(sample_image, stage1_only):
    result = change_handle(QueryInput(images=[sample_image, sample_image]), {})

    assert result.structured_data["change_fraction"] == 0.0
    assert result.structured_data["largest_region_bbox"] is None


def test_fusion_detects_known_sar_regions_and_agrees_with_optical(optical_sar_pair, monkeypatch, fusion_stage1_only):
    """water_segmentation's own model is mocked here (not re-tested -- test_water_segmentation_*
    already covers it) so this isolates fusion_adapter's own SAR-detection + reconciliation logic:
    the injected SAR water block is ~9% of the 100x100 image, so a mocked optical read of 10%
    should land within WATER_AGREEMENT_TOLERANCE and report 'agree'. Pinned to Stage 1 (see
    fusion_stage1_only) since Stage 2's own behaviour is covered separately below -- otherwise this
    would silently also exercise the classifier on a machine that happens to have it installed."""
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
    assert "land_cover_class" not in result.structured_data  # Stage 1 only -- see the fixture note above
    assert "no optical cross-check yet" in result.text_summary


def test_fusion_surfaces_disagreement_between_modalities(optical_sar_pair, monkeypatch, fusion_stage1_only):
    import app.specialists.fusion_adapter as adapter

    optical, sar = optical_sar_pair

    class FakeWaterTool:
        def segment(self, image_path, threshold=0.5):
            return {"water_fraction": 0.80, "confidence": 0.9}  # wildly different from SAR's ~9%

    monkeypatch.setattr(adapter, "_get_water_tool", lambda: FakeWaterTool())

    result = fusion_handle(QueryInput(images=[optical, sar]), {})

    assert result.structured_data["water_agreement"] == "disagree"
    assert "DISAGREE" in result.text_summary


def test_fusion_stage2_adds_learned_land_cover_class(optical_sar_pair, monkeypatch):
    """Stage 2 active: the classifier's read is threaded into structured_data and the text. The
    fixture's injected bright block is ~9-11% of the frame -- below the adapter's 0.15 "SAR sees
    meaningful built-up" threshold (confirmed by the Stage 1 test above, which measures it directly)
    -- so a mocked classification that ALSO doesn't call the scene urban should read as consistent,
    not a disagreement (both readings agree there's no strong built-up signal here)."""
    import app.specialists.fusion_adapter as adapter

    optical, sar = optical_sar_pair

    class FakeWaterTool:
        def segment(self, image_path, threshold=0.5):
            return {"water_fraction": 0.10, "confidence": 0.9}

    class FakeClassifier:
        def classify(self, optical_path, sar_path):
            return {"land_cover_class": "grassland", "probabilities": {"agri": 0.05, "barrenland": 0.05, "grassland": 0.85, "urban": 0.05}, "confidence": 0.85}

    monkeypatch.setattr(adapter, "_get_water_tool", lambda: FakeWaterTool())
    monkeypatch.setattr(adapter, "USE_CLASSIFIER", True)
    monkeypatch.setattr(adapter, "_get_classifier", lambda: FakeClassifier())

    result = fusion_handle(QueryInput(images=[optical, sar]), {})

    assert result.structured_data["land_cover_class"] == "grassland"
    assert result.structured_data["land_cover_probabilities"]["grassland"] == 0.85
    assert result.structured_data["land_cover_confidence"] == 0.85
    assert "grassland" in result.text_summary
    assert "consistent with the SAR-only built-up read" in result.text_summary
    assert "no optical cross-check yet" not in result.text_summary  # Stage 2 replaces that sentence


def test_fusion_stage2_flags_disagreement_with_sar_builtup_read(optical_sar_pair, monkeypatch):
    """The fixture's SAR built-up signal stays below the 0.15 "meaningful" threshold (see the test
    above), but a mocked classifier that confidently calls the scene 'urban' anyway should surface
    that as a disagreement -- e.g. dense low-rise buildings that don't give a strong SAR double-
    bounce return but are unambiguous in the optical image, exactly the case this cross-check
    exists to catch (the reverse of SAR's own documented mountainous-terrain false-positive risk)."""
    import app.specialists.fusion_adapter as adapter

    optical, sar = optical_sar_pair

    class FakeWaterTool:
        def segment(self, image_path, threshold=0.5):
            return {"water_fraction": 0.10, "confidence": 0.9}

    class FakeClassifier:
        def classify(self, optical_path, sar_path):
            return {"land_cover_class": "urban", "probabilities": {"agri": 0.03, "barrenland": 0.02, "grassland": 0.05, "urban": 0.90}, "confidence": 0.90}

    monkeypatch.setattr(adapter, "_get_water_tool", lambda: FakeWaterTool())
    monkeypatch.setattr(adapter, "USE_CLASSIFIER", True)
    monkeypatch.setattr(adapter, "_get_classifier", lambda: FakeClassifier())

    result = fusion_handle(QueryInput(images=[optical, sar]), {})

    assert result.structured_data["land_cover_class"] == "urban"
    assert "DISAGREES with the SAR-only built-up read" in result.text_summary


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
