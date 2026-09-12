"""Registry adapter for the water-body segmentation specialist (models/water_segmentation/)."""

import uuid
from typing import Any

from app.config import EVIDENCE_DIR, MODELS_DIR, WATER_SEG_CHECKPOINT
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

_water_tool = None  # lazy singleton


def _get_tool():
    global _water_tool
    if _water_tool is None:
        module = load_module(
            MODELS_DIR / "water_segmentation" / "water_segmentation_tool.py", "water_segmentation_tool"
        )
        _water_tool = module.WaterSegmentationTool(checkpoint_path=str(WATER_SEG_CHECKPOINT))
    return _water_tool


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    module = load_module(MODELS_DIR / "water_segmentation" / "water_segmentation_tool.py", "water_segmentation_tool")
    threshold = arguments.get("threshold", 0.5)
    image_path = query_input.images[0]

    tool = _get_tool()
    result = tool.segment(str(image_path), threshold=threshold)

    evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
    module.draw_mask(str(image_path), result, str(evidence_path))

    pct = result["water_fraction"] * 100
    text_summary = f"Water covers approximately {pct:.1f}% of the image."

    return ToolResult(
        tool_name="water_body_segmentation",
        text_summary=text_summary,
        structured_data={"water_fraction": result["water_fraction"]},
        evidence_image_path=evidence_path,
        confidence=result["confidence"],
    )


TOOL_SPEC = ToolSpec(
    name="water_body_segmentation",
    description=(
        "Segment water bodies (rivers, lakes, coastline, flooding) in a single optical image and "
        "estimate what fraction of the image is water. Use this for queries about water coverage, "
        "flooding extent, or highlighting/mapping water bodies."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "description": "Probability threshold in [0,1] for classifying a pixel as water (default 0.5).",
            },
        },
        "required": [],
    },
    min_images=1,
    max_images=1,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id="water_body_unet_final.pt",
)
