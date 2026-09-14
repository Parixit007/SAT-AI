"""Registry adapter for the optical-SAR cross-modal fusion specialist (models/fusion/). Stage 1
(see the roadmap plan): training-free SAR-backscatter analysis, cross-checked against the existing
optical-based water_segmentation tool's own read -- the actual "fusion" (combining what both
modalities say) happens here, not in fusion_tool.py, since only this adapter layer can import from
another specialist's adapter (models/*/ scripts stay standalone, per CLAUDE.md)."""

import uuid
from typing import Any

from app.concurrency import serialize_first_call
from app.config import EVIDENCE_DIR, MODELS_DIR
from app.orchestrator.input_validation import validate_images
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module
from app.specialists.water_segmentation_adapter import _get_tool as _get_water_tool

# Two independent, very different measurement methods (SAR backscatter thresholding vs. an
# optical CNN) agreeing to within this absolute fraction counts as "agree" -- generous on purpose,
# since exact numeric agreement between such different techniques isn't the bar; rough consensus is.
WATER_AGREEMENT_TOLERANCE = 0.15

_fusion_tool = None  # lazy singleton -- see specialists/_loader.py for the shared shape


@serialize_first_call
def _get_tool():
    global _fusion_tool
    if _fusion_tool is None:
        module = load_module(MODELS_DIR / "fusion" / "fusion_tool.py", "fusion_tool")
        _fusion_tool = module.FusionTool()
    return _fusion_tool


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
    text_summary = (
        f"SAR analysis: approximately {sar_result['builtup_fraction'] * 100:.1f}% of the scene shows "
        f"built-up-like backscatter, {water_frac_sar * 100:.1f}% shows water-like backscatter. "
        f"{agreement_text} Built-up detection is SAR-only in this version -- no optical cross-check "
        "yet, so treat it as the coarser of the two reads (SAR brightness can also come from "
        "mountainous terrain, not just buildings)."
    )

    return ToolResult(
        tool_name="optical_sar_fusion",
        text_summary=text_summary,
        structured_data={
            "water_fraction_sar": water_frac_sar,
            "water_fraction_optical": water_frac_optical,
            "water_agreement": "agree" if agree else "disagree",
            "builtup_fraction_sar": sar_result["builtup_fraction"],
        },
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
    checkpoint_id=None,  # Stage 1 is training-free
)
