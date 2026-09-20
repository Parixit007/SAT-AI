"""Registry adapter for "describe / explain / what's in this image" requests.

Today this is an honest object inventory, not a caption: it counts the objects the grounding
detector has proven reliable on real aerial imagery (airplanes, ships, storage tanks) and says so.
The other 21 categories the checkpoint was trained on were tried on 20 landmark scenes and are not
trustworthy enough to report -- tennis courts came back as swimming pools, a stadium as a basketball
court, bridges/dams/roundabouts/cars not at all -- and the two other sources that looked usable for
a description, the water model (58% "water" on an airport apron) and the VQA checkpoint ("rural" for
Heathrow), fail on exactly the imagery a user is likely to upload. A wrong confident sentence is
worse than a short honest one, so they are not folded in.

Meant to grow: a remote-sensing captioner (VRSBench fine-tune) becomes one more source in this same
tool, feeding a real sentence to the answer composer alongside the counts."""

import uuid
from collections import Counter
from typing import Any

from app.config import EVIDENCE_DIR, MODELS_DIR
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists import grounding_adapter
from app.specialists._loader import load_module

SCAN_CATEGORIES = ["airplane", "ship", "storage tank"]
BOX_THRESHOLD = 0.30  # a little stricter than a single-category count: an inventory shouldn't invent objects


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
    image_path = query_input.images[0]

    detections = grounding_adapter._get_tool().scan(str(image_path), SCAN_CATEGORIES, box_threshold=BOX_THRESHOLD)

    evidence_path = None
    if detections:
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
        module.draw_boxes(str(image_path), detections, str(evidence_path))

    counts = Counter(d["phrase"] for d in detections)
    objects = [
        {"label": label, "count": counts[label], "best_score": max(d["score"] for d in detections if d["phrase"] == label)}
        for label in sorted(counts, key=lambda k: -counts[k])
    ]
    covered = ", ".join(SCAN_CATEGORIES[:-1]) + f" and {SCAN_CATEGORIES[-1]}"
    if objects:
        found = "; ".join(f"{o['count']} {o['label']}(s)" for o in objects)
        text_summary = f"Object scan (checks only {covered}): found {found}."
    else:
        text_summary = f"Object scan (checks only {covered}): none found."

    return ToolResult(
        tool_name="scene_description",
        text_summary=text_summary,
        structured_data={
            "objects": objects,
            "scanned_categories": SCAN_CATEGORIES,
            "detections": detections,
            "total_objects": len(detections),
        },
        evidence_image_path=evidence_path,
        confidence=max((d["score"] for d in detections), default=0.0),
    )


TOOL_SPEC = ToolSpec(
    name="scene_description",
    description=(
        "General, open-ended requests about ONE image: 'describe / explain / summarize this image', "
        "'what is in this image?', 'what do you see?', 'what objects are here?'. It scans for the "
        "objects the detector handles reliably -- airplanes, ships and storage tanks -- and counts "
        "them, returning a marked-up image; it does NOT describe terrain, buildings, roads or land "
        "cover. Use it for general requests only: for a specific question use the specific tool "
        "(counting or locating a named object -> text_guided_grounding; water coverage -> "
        "water_body_segmentation; a yes/no or rural-vs-urban question -> visual_question_answering)."
    ),
    parameters_schema={"type": "object", "properties": {}, "required": []},
    min_images=1,
    max_images=1,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id="dior_rsvg_finetuned.pth",
)
