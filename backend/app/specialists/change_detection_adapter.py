"""Registry adapter for the bi-temporal change-detection specialist (models/change_detection/).
Stage 1 (see the roadmap plan): training-free pixel-differencing -- no checkpoint, so unlike the
other image-based adapters this one never fails to construct. Stage 2 (a semantic segmentation
model trained on SECOND-CC) will add class-aware change descriptions later without changing this
adapter's shape -- only ChangeDetectionTool's internals."""

import uuid
from typing import Any

from app.concurrency import serialize_first_call
from app.config import EVIDENCE_DIR, MODELS_DIR
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

_change_tool = None  # lazy singleton -- see module docstring


@serialize_first_call
def _get_tool():
    global _change_tool
    if _change_tool is None:
        module = load_module(MODELS_DIR / "change_detection" / "change_detection_tool.py", "change_detection_tool")
        _change_tool = module.ChangeDetectionTool()
    return _change_tool


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    module = load_module(MODELS_DIR / "change_detection" / "change_detection_tool.py", "change_detection_tool")
    image1_path, image2_path = query_input.images[0], query_input.images[1]

    tool = _get_tool()
    result = tool.detect(str(image1_path), str(image2_path))

    evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
    module.draw_change_overlay(str(image1_path), str(image2_path), result, str(evidence_path))

    pct = result["change_fraction"] * 100
    if result["largest_region_bbox"]:
        x1, y1, x2, y2 = result["largest_region_bbox"]
        location_text = f" The most significant change is concentrated around pixels ({x1},{y1})-({x2},{y2}) of the second image."
    else:
        location_text = ""
    text_summary = (
        f"Approximately {pct:.1f}% of the image area changed between the two dates.{location_text} "
        "This is a pixel-level difference check -- it can say THAT and roughly WHERE something "
        "changed, not WHICH kind of land cover changed (a semantic version is planned)."
    )

    return ToolResult(
        tool_name="change_detection",
        text_summary=text_summary,
        structured_data={
            "change_fraction": result["change_fraction"],
            "largest_region_bbox": result["largest_region_bbox"],
        },
        evidence_image_path=evidence_path,
        confidence=result["confidence"],
    )


TOOL_SPEC = ToolSpec(
    name="change_detection",
    description=(
        "Compare two images of the SAME location taken at DIFFERENT times (a bi-temporal pair) "
        "and report how much of the scene changed and roughly where. Use this for queries about "
        "what changed between two dates, or whether an area changed at all. Currently a "
        "pixel-level difference check: it reports the fraction of the image that changed and the "
        "location of the largest changed region, but not which kind of land cover changed -- for "
        "'has the built-up area increased/decreased' style questions naming a specific class, say "
        "so in the answer that this can't yet isolate that class specifically. Requires exactly "
        "two images, in acquisition order (first image = earlier date, second = later date)."
    ),
    parameters_schema={"type": "object", "properties": {}},
    min_images=2,
    max_images=2,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id=None,  # Stage 1 is training-free -- no checkpoint to report
)
