"""Registry adapter for the bi-temporal change-detection specialist (models/change_detection/).

One tool name, two stages behind it:

* **Stage 2** -- active when `config.CHANGE_SEG_CHECKPOINT` exists: the Siamese semantic-change
  network trained on SECOND-CC (`semantic_change_tool.py`). Reports how much of the scene changed,
  where, WHICH land-cover classes gained or lost area, and the main from->to transitions -- enough
  to answer "has the built-up area increased, decreased, or remained unchanged?".
* **Stage 1** -- the fallback when there's no checkpoint: training-free pixel differencing
  (`change_detection_tool.py`). Says THAT and roughly WHERE something changed, not WHICH class.

Which stage is active is decided once, when this module is imported (`USE_STAGE2`), and the
trace's `checkpoint_id` reports the same decision -- so adding or removing the checkpoint takes a
backend restart to show up, in both the behaviour and the audit trail, rather than the two
disagreeing mid-run. Both stages share the tool name, the `change_fraction` /
`largest_region_bbox` fields, and the evidence-image shape, so nothing downstream (the
orchestrator, the UI's Stage 1 rendering) needs to know which one ran."""

import uuid
from typing import Any

from app.concurrency import serialize_first_call
from app.config import CHANGE_SEG_CHECKPOINT, EVIDENCE_DIR, MODELS_DIR
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

USE_STAGE2 = CHANGE_SEG_CHECKPOINT.exists()

_stage1_tool = None  # lazy singletons -- see module docstring
_stage2_tool = None


def _stage1_module():
    return load_module(MODELS_DIR / "change_detection" / "change_detection_tool.py", "change_detection_tool")


def _stage2_module():
    return load_module(MODELS_DIR / "change_detection" / "semantic_change_tool.py", "semantic_change_tool")


@serialize_first_call
def _get_stage1_tool():
    global _stage1_tool
    if _stage1_tool is None:
        _stage1_tool = _stage1_module().ChangeDetectionTool()
    return _stage1_tool


@serialize_first_call
def _get_stage2_tool():
    global _stage2_tool
    if _stage2_tool is None:
        _stage2_tool = _stage2_module().SemanticChangeTool(checkpoint_path=str(CHANGE_SEG_CHECKPOINT))
    return _stage2_tool


def _bbox_sentence(bbox) -> str:
    if not bbox:
        return ""
    x1, y1, x2, y2 = bbox
    return f" The most significant change is concentrated around pixels ({x1},{y1})-({x2},{y2}) of the second image."


def _handle_stage2(query_input: QueryInput) -> ToolResult:
    module = _stage2_module()
    image1_path, image2_path = str(query_input.images[0]), str(query_input.images[1])

    result = _get_stage2_tool().analyze(image1_path, image2_path)

    evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
    module.draw_semantic_change_overlay(image1_path, image2_path, result, str(evidence_path))

    text_summary = (
        module.describe_changes(result)
        + _bbox_sentence(result["largest_region_bbox"])
        + " Class-level figures are estimates from a model trained on SECOND-CC aerial imagery "
        "(256x256 crops), so treat small percentages as indicative rather than exact."
    )
    return ToolResult(
        tool_name="change_detection",
        text_summary=text_summary,
        structured_data={
            "method": "semantic",
            "change_fraction": result["change_fraction"],
            "largest_region_bbox": result["largest_region_bbox"],
            "class_changes": result["class_changes"],
            "transitions": result["transitions"],
        },
        evidence_image_path=evidence_path,
        confidence=result["confidence"],
    )


def _handle_stage1(query_input: QueryInput) -> ToolResult:
    module = _stage1_module()
    image1_path, image2_path = str(query_input.images[0]), str(query_input.images[1])

    result = _get_stage1_tool().detect(image1_path, image2_path)

    evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
    module.draw_change_overlay(image1_path, image2_path, result, str(evidence_path))

    pct = result["change_fraction"] * 100
    text_summary = (
        f"Approximately {pct:.1f}% of the image area changed between the two dates."
        f"{_bbox_sentence(result['largest_region_bbox'])} "
        "This is a pixel-level difference check -- it can say THAT and roughly WHERE something "
        "changed, not WHICH kind of land cover changed (a semantic version is planned)."
    )
    return ToolResult(
        tool_name="change_detection",
        text_summary=text_summary,
        structured_data={
            "method": "pixel_difference",
            "change_fraction": result["change_fraction"],
            "largest_region_bbox": result["largest_region_bbox"],
        },
        evidence_image_path=evidence_path,
        confidence=result["confidence"],
    )


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    return _handle_stage2(query_input) if USE_STAGE2 else _handle_stage1(query_input)


_STAGE1_DESCRIPTION = (
    "Compare two images of the SAME location taken at DIFFERENT times (a bi-temporal pair) "
    "and report how much of the scene changed and roughly where. Use this for queries about "
    "what changed between two dates, or whether an area changed at all. Currently a "
    "pixel-level difference check: it reports the fraction of the image that changed and the "
    "location of the largest changed region, but not which kind of land cover changed -- for "
    "'has the built-up area increased/decreased' style questions naming a specific class, say "
    "so in the answer that this can't yet isolate that class specifically. Requires exactly "
    "two images, in acquisition order (first image = earlier date, second = later date)."
)

_STAGE2_DESCRIPTION = (
    "Compare two images of the SAME location taken at DIFFERENT times (a bi-temporal pair) and "
    "report how much of the scene changed, where, and WHICH land-cover classes gained or lost "
    "area (buildings, trees, low vegetation, bare ground, water, playground) along with the "
    "main from->to transitions. Use this for queries about what changed between two dates, "
    "whether an area changed at all, or whether a specific class -- built-up area (buildings), "
    "vegetation, water -- increased, decreased, or stayed the same. Requires exactly two "
    "images, in acquisition order (first image = earlier date, second = later date)."
)

TOOL_SPEC = ToolSpec(
    name="change_detection",
    description=_STAGE2_DESCRIPTION if USE_STAGE2 else _STAGE1_DESCRIPTION,
    parameters_schema={"type": "object", "properties": {}},
    min_images=2,
    max_images=2,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id=CHANGE_SEG_CHECKPOINT.name if USE_STAGE2 else None,  # Stage 1 is training-free
)
