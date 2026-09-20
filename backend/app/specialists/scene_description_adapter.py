"""Registry adapter for "describe / explain / what's in this image" requests.

Two sources, each optional and independently failing:

  * a written description from the remote-sensing captioner (models/captioning/, SmolVLM-500M
    fine-tuned on VRSBench captions) -- only when its checkpoint is installed, decided once at import
    like change_detection's Stage 2. Fluent but a small model: it can be wrong about details and is
    bad at counting, so it is presented as a general impression, never as the source of numbers.
  * an object inventory from the grounding detector, restricted to the categories it has proven
    reliable on real aerial imagery (airplanes, ships, storage tanks). The other 21 categories the
    checkpoint was trained on were tried on 20 landmark scenes and are not trustworthy enough to
    report -- tennis courts came back as swimming pools, a stadium as a basketball court, bridges,
    dams, roundabouts and cars not at all. Its counts are the numbers to believe.

The other two sources that looked usable for a description were measured wrong on exactly the
imagery a user is likely to upload -- the water model (58% "water" on an airport apron) and the VQA
checkpoint ("rural" for Heathrow) -- so they are not folded in. A wrong confident sentence is worse
than a short honest one."""

import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.concurrency import serialize_first_call
from app.config import CAPTION_CHECKPOINT_DIR, EVIDENCE_DIR, MODELS_DIR
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists import grounding_adapter
from app.specialists._loader import load_module

SCAN_CATEGORIES = ["airplane", "ship", "storage tank"]
BOX_THRESHOLD = 0.30  # a little stricter than a single-category count: an inventory shouldn't invent objects

# Decided once at import (restart the backend after installing the checkpoint), so the description the
# router LLM sees, the trace and the behaviour always agree -- same pattern as change_detection.
USE_CAPTIONS = (CAPTION_CHECKPOINT_DIR / "caption_meta.json").exists()

_captioner = None  # lazy singleton -- the model is ~1GB and only loads on the first description


@serialize_first_call
def _get_captioner():
    global _captioner
    if _captioner is None:
        module = load_module(MODELS_DIR / "captioning" / "caption_tool.py", "caption_tool")
        _captioner = module.CaptionTool(str(CAPTION_CHECKPOINT_DIR))
    return _captioner


def _scan(image_path: str):
    module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
    detections = grounding_adapter._get_tool().scan(image_path, SCAN_CATEGORIES, box_threshold=BOX_THRESHOLD)
    evidence_path = None
    if detections:
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
        module.draw_boxes(image_path, detections, str(evidence_path))
    return detections, evidence_path


def _caption(image_path: str) -> dict[str, Any]:
    return _get_captioner().describe(image_path)


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    image_path = str(query_input.images[0])

    # The detector runs on CPU and the captioner on the GPU (MPS), so they overlap.
    with ThreadPoolExecutor(max_workers=2) as pool:
        scan_future = pool.submit(_scan, image_path)
        caption_future = pool.submit(_caption, image_path) if USE_CAPTIONS else None
        errors, detections, evidence_path, caption = [], [], None, None
        scan_ok = caption_ok = False
        try:
            detections, evidence_path = scan_future.result()
            scan_ok = True
        except Exception as exc:
            errors.append(f"object scan unavailable ({exc})")
        if caption_future is not None:
            try:
                caption = caption_future.result()
                caption_ok = True
            except Exception as exc:
                errors.append(f"description model unavailable ({exc})")

    if not scan_ok and not caption_ok:
        raise RuntimeError("; ".join(errors))

    counts = Counter(d["phrase"] for d in detections)
    objects = [
        {"label": label, "count": counts[label], "best_score": max(d["score"] for d in detections if d["phrase"] == label)}
        for label in sorted(counts, key=lambda k: -counts[k])
    ]
    covered = ", ".join(SCAN_CATEGORIES[:-1]) + f" and {SCAN_CATEGORIES[-1]}"

    parts = []
    if caption_ok:
        parts.append(f"Description (written by a small remote-sensing captioning model; it can be wrong about details and counts): {caption['caption']}")
    if scan_ok:
        if objects:
            found = "; ".join(f"{o['count']} {o['label']}(s)" for o in objects)
            parts.append(f"Object scan (checks only {covered}): found {found}.")
        else:
            parts.append(f"Object scan (checks only {covered}): none found.")
    parts.extend(f"({e}.)" for e in errors)

    source_scores = []
    if caption_ok:
        source_scores.append(caption["confidence"])
    if scan_ok and detections:
        source_scores.append(max(d["score"] for d in detections))
    elif scan_ok and not caption_ok:
        source_scores.append(0.0)

    return ToolResult(
        tool_name="scene_description",
        text_summary=" ".join(parts),
        structured_data={
            "caption": caption["caption"] if caption_ok else None,
            "caption_confidence": caption["confidence"] if caption_ok else None,
            "objects": objects,
            "scanned_categories": SCAN_CATEGORIES,
            "detections": detections,
            "total_objects": len(detections),
            "notes": errors,
        },
        evidence_image_path=evidence_path,
        confidence=sum(source_scores) / len(source_scores) if source_scores else 0.0,
    )


_WHAT_IT_RETURNS = (
    "It returns a written description of the scene from a remote-sensing captioning model, plus counts "
    "of the objects the detector handles reliably -- airplanes, ships and storage tanks -- with a "
    "marked-up image. The description can be wrong about details; the counts are the reliable numbers."
    if USE_CAPTIONS
    else "It scans for the objects the detector handles reliably -- airplanes, ships and storage tanks -- "
    "and counts them, returning a marked-up image; it does NOT describe terrain, buildings, roads or "
    "land cover."
)

TOOL_SPEC = ToolSpec(
    name="scene_description",
    description=(
        "General, open-ended requests about ONE image: 'describe / explain / summarize this image', "
        "'what is in this image?', 'what do you see?', 'what objects are here?'. "
        + _WHAT_IT_RETURNS
        + " Use it for general requests only: for a specific question use the specific tool "
        "(counting or locating a named object -> text_guided_grounding; water coverage -> "
        "water_body_segmentation; a yes/no or rural-vs-urban question -> visual_question_answering)."
    ),
    parameters_schema={"type": "object", "properties": {}, "required": []},
    min_images=1,
    max_images=1,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id="caption_model + dior_rsvg_finetuned.pth" if USE_CAPTIONS else "dior_rsvg_finetuned.pth",
)
