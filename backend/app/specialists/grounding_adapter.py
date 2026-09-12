"""Registry adapter for the text-guided region-grounding specialist (models/grounding/). The
underlying `groundingdino` package isn't installed by default (see CLAUDE.md Setup) -- the
GroundingTool singleton is only constructed on the first actual call, so registering this tool
(cheap, metadata-only) never fails even before that setup step is done; only running a grounding
query does."""

import uuid
from typing import Any

from app.config import EVIDENCE_DIR, GROUNDING_CHECKPOINT, GROUNDING_CONFIG, MODELS_DIR
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

_grounding_tool = None  # lazy singleton -- see module docstring


def _get_tool():
    global _grounding_tool
    if _grounding_tool is None:
        module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
        _grounding_tool = module.GroundingTool(
            config_path=str(GROUNDING_CONFIG),
            checkpoint_path=str(GROUNDING_CHECKPOINT),
        )
    return _grounding_tool


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)
    image_path = query_input.images[0]

    tool = _get_tool()
    detections = tool.ground(str(image_path), query, top_k=top_k)

    evidence_path = None
    if detections:
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
        module.draw_boxes(str(image_path), detections, str(evidence_path))

    if detections:
        top = detections[0]
        text_summary = (
            f"Found {len(detections)} region(s) matching '{query}'. "
            f"Best match: '{top['phrase']}' at {top['bbox_xyxy']} (score {top['score']:.2f})."
        )
        confidence = top["score"]
    else:
        text_summary = f"No regions matching '{query}' were found above the detection threshold."
        confidence = 0.0

    return ToolResult(
        tool_name="text_guided_grounding",
        text_summary=text_summary,
        structured_data={"detections": detections},
        evidence_image_path=evidence_path,
        confidence=confidence,
    )


TOOL_SPEC = ToolSpec(
    name="text_guided_grounding",
    description=(
        "Locate the region(s) in a single image that match a text description or referring "
        "expression (e.g. 'the ship near the harbor entrance', or a category like 'building'). "
        "Returns bounding boxes. Use this when the query asks to find, locate, or highlight "
        "something specific in one image."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The object/category or referring expression to locate in the image.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max number of matches to return (default 5).",
            },
        },
        "required": ["query"],
    },
    min_images=1,
    max_images=1,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id="dior_rsvg_finetuned.pth",
)
