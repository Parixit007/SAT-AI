"""Registry adapter for the optical-SAR cross-modal fusion specialist (models/fusion/). Two stages:

* **Stage 1** (always on): training-free SAR-backscatter analysis (fusion_tool.py), cross-checked
  against the existing optical-based water_segmentation tool's own read.
* **Stage 2** (active when `config.FUSION_CLASSIFIER_CHECKPOINT` exists, decided once at import
  like every other optional checkpoint in this app -- `USE_CLASSIFIER`): a small early-fusion CNN
  (fusion_classifier.py) that looks at BOTH modalities together and predicts a broad land-cover
  class (agricultural / barren / grassland / urban). Its P(urban) is a genuinely learned,
  cross-modal built-up signal -- the optical cross-check Stage 1's built-up read never had (Stage 1
  is SAR-only there, and SAR brightness alone can false-positive on mountainous terrain).

The actual "fusion" (combining what both modalities say) happens here, not in fusion_tool.py, since
only this adapter layer can import from another specialist's adapter (models/*/ scripts stay
standalone, per CLAUDE.md)."""

import uuid
from typing import Any

from app.concurrency import serialize_first_call
from app.config import EVIDENCE_DIR, FUSION_CLASSIFIER_CHECKPOINT, MODELS_DIR
from app.orchestrator.input_validation import validate_images
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module
from app.specialists.water_segmentation_adapter import _get_tool as _get_water_tool

# Two independent, very different measurement methods (SAR backscatter thresholding vs. an
# optical CNN) agreeing to within this absolute fraction counts as "agree" -- generous on purpose,
# since exact numeric agreement between such different techniques isn't the bar; rough consensus is.
WATER_AGREEMENT_TOLERANCE = 0.15

USE_CLASSIFIER = FUSION_CLASSIFIER_CHECKPOINT.exists()

_fusion_tool = None  # lazy singletons -- see specialists/_loader.py for the shared shape
_fusion_classifier = None


@serialize_first_call
def _get_tool():
    global _fusion_tool
    if _fusion_tool is None:
        module = load_module(MODELS_DIR / "fusion" / "fusion_tool.py", "fusion_tool")
        _fusion_tool = module.FusionTool()
    return _fusion_tool


@serialize_first_call
def _get_classifier():
    global _fusion_classifier
    if _fusion_classifier is None:
        module = load_module(MODELS_DIR / "fusion" / "fusion_classifier.py", "fusion_classifier")
        _fusion_classifier = module.FusionClassifier(str(FUSION_CLASSIFIER_CHECKPOINT))
    return _fusion_classifier


def _split_by_modality(images: list) -> tuple:
    """`is_compatible()` (tool_registry.py) already guaranteed exactly one optical + one SAR image
    reached this handler -- this just figures out which list position is which."""
    validation = validate_images(images)
    optical_path = next(img.path for img in validation.images if img.modality_guess == "optical")
    sar_path = next(img.path for img in validation.images if img.modality_guess == "sar")
    return optical_path, sar_path


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    module = load_module(MODELS_DIR / "fusion" / "fusion_tool.py", "fusion_tool")
    optical_path, sar_path = _split_by_modality(query_input.images)

    sar_result = _get_tool().analyze(str(sar_path))
    optical_result = _get_water_tool().segment(str(optical_path))
    water_frac_sar = sar_result["water_fraction"]
    water_frac_optical = optical_result["water_fraction"]

    agree = abs(water_frac_sar - water_frac_optical) <= WATER_AGREEMENT_TOLERANCE
    # Two independent modalities agreeing is stronger evidence than either alone; disagreeing
    # doesn't mean either is "wrong" -- e.g. cloud-obscured optical vs. a clear SAR read is
    # exactly the scenario SAR exists for -- so it's surfaced, not silently averaged away.
    combined_confidence = (
        min(1.0, (sar_result["confidence"] + optical_result["confidence"]) / 2 + 0.1)
        if agree
        else min(sar_result["confidence"], optical_result["confidence"]) * 0.5
    )

    evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
    module.draw_fusion_overlay(str(sar_path), sar_result, str(evidence_path))

    agreement_text = (
        f"Optical ({water_frac_optical * 100:.1f}%) and SAR ({water_frac_sar * 100:.1f}%) water "
        f"reads agree." if agree else
        f"Optical ({water_frac_optical * 100:.1f}%) and SAR ({water_frac_sar * 100:.1f}%) water "
        "reads DISAGREE -- worth a closer look (e.g. cloud cover in the optical image, or recent "
        "flooding SAR would catch and optical might miss)."
    )

    structured_data = {
        "water_fraction_sar": water_frac_sar,
        "water_fraction_optical": water_frac_optical,
        "water_agreement": "agree" if agree else "disagree",
        "builtup_fraction_sar": sar_result["builtup_fraction"],
    }

    if USE_CLASSIFIER:
        # Stage 2: a learned scene class from BOTH modalities together -- P(urban) is the actual
        # optical cross-check for built-up-ness the sentence below used to say didn't exist.
        cls_result = _get_classifier().classify(str(optical_path), str(sar_path))
        structured_data["land_cover_class"] = cls_result["land_cover_class"]
        structured_data["land_cover_probabilities"] = cls_result["probabilities"]
        structured_data["land_cover_confidence"] = cls_result["confidence"]
        p_urban = cls_result["probabilities"].get("urban", 0.0)
        sar_sees_builtup = sar_result["builtup_fraction"] >= 0.15  # a meaningful presence, not noise
        classifier_sees_urban = cls_result["land_cover_class"] == "urban"
        if sar_sees_builtup == classifier_sees_urban:
            builtup_text = (
                f"A learned model that looks at both images together separately classifies the overall "
                f"scene as '{cls_result['land_cover_class']}' ({cls_result['confidence'] * 100:.0f}% "
                f"confidence, urban likelihood {p_urban * 100:.0f}%) -- consistent with the SAR-only "
                "built-up read above."
            )
        else:
            builtup_text = (
                f"A learned model that looks at both images together classifies the overall scene as "
                f"'{cls_result['land_cover_class']}' ({cls_result['confidence'] * 100:.0f}% confidence, "
                f"urban likelihood {p_urban * 100:.0f}%) -- this DISAGREES with the SAR-only built-up "
                "read above (SAR brightness alone can come from mountainous terrain, not just buildings, "
                "which is exactly the kind of case this cross-check exists to catch)."
            )
    else:
        builtup_text = (
            "Built-up detection is SAR-only in this version -- no optical cross-check yet, so treat it "
            "as the coarser of the two reads (SAR brightness can also come from mountainous terrain, "
            "not just buildings)."
        )

    text_summary = (
        f"SAR analysis: approximately {sar_result['builtup_fraction'] * 100:.1f}% of the scene shows "
        f"built-up-like backscatter, {water_frac_sar * 100:.1f}% shows water-like backscatter. "
        f"{agreement_text} {builtup_text}"
    )

    return ToolResult(
        tool_name="optical_sar_fusion",
        text_summary=text_summary,
        structured_data=structured_data,
        evidence_image_path=evidence_path,
        confidence=round(combined_confidence, 4),
    )


TOOL_SPEC = ToolSpec(
    name="optical_sar_fusion",
    description=(
        "Jointly analyze a co-registered optical image and SAR image of the SAME place at the "
        "SAME time (not two dates -- use change_detection for that) to identify water and "
        "built-up regions, cross-checking what each modality independently shows. Use this when "
        "the query asks to combine/use optical AND SAR together, or when cloud cover might make "
        "an optical-only read unreliable. Requires exactly one optical image and one SAR image "
        "(any order)."
    ),
    parameters_schema={"type": "object", "properties": {}},
    min_images=2,
    max_images=2,
    compatible_modalities=["optical", "sar"],
    required_modality_pair=("optical", "sar"),
    handler=_handle,
    # Stage 1 (SAR-backscatter thresholding) is training-free; Stage 2 (a learned optical+SAR land-
    # cover classifier) reports its checkpoint here when installed -- same decide-once-at-import
    # pattern as change_detection's USE_STAGE2, so the trace never disagrees with what actually ran.
    checkpoint_id=FUSION_CLASSIFIER_CHECKPOINT.name if USE_CLASSIFIER else None,
)
