"""Registry adapter for "describe / explain / what's in this image" requests.

Three sources, each optional and independently failing:

  * a written description from the remote-sensing captioner (models/captioning/, SmolVLM-500M
    fine-tuned on VRSBench captions) -- used when ENABLE_CAPTIONER is true (the default) AND its
    checkpoint is installed, decided once at import like change_detection's Stage 2. It is fluent and
    covers what the detector cannot (roads, buildings, vegetation, roof colours) but it is a small model
    trained on object-centric (DOTA/DIOR) captions and is bad at counting -- so it is presented as a
    general impression, never as the source of numbers.
  * land cover from the segmentation model (models/landcover/, land_cover_adapter.py): the share of the image that
    is buildings, roads, trees, water, farmland and so on, an approximate building count and the roof colours --
    the only source that can say anything quantitative about what the detector has no class for. Used when
    ENABLE_LANDCOVER is true (the default) and its checkpoint is installed.
  * an object inventory from the grounding detector, restricted to the categories it has proven
    reliable on real aerial imagery (airplanes, ships, storage tanks). The other 21 categories the
    checkpoint was trained on were tried on 20 landmark scenes and are not trustworthy enough to
    report -- tennis courts came back as swimming pools, a stadium as a basketball court, bridges,
    dams, roundabouts and cars not at all. Its counts are the numbers to believe.

The other two sources that looked usable for a description were measured wrong on exactly the
imagery a user is likely to upload -- the water model (58% "water" on an airport apron) and the VQA
checkpoint ("rural" for Heathrow) -- so they are not folded in. A wrong confident sentence is worse
than a short honest one."""

import logging
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.concurrency import serialize_first_call
from app.config import CAPTION_CHECKPOINT_DIR, EVIDENCE_DIR, MODELS_DIR, settings
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists import grounding_adapter, land_cover_adapter
from app.specialists._loader import load_module

logger = logging.getLogger(__name__)

SCAN_CATEGORIES = ["airplane", "ship", "storage tank"]
BOX_THRESHOLD = 0.30  # a little stricter than a single-category count: an inventory shouldn't invent objects

# Precision guard for the inventory (not for a user's explicit "how many X?", where a small count is a
# legitimate answer). Measured on 21 real scenes at the 0.30 threshold: the true detections were
# clusters (41 airplanes, 69 tanks, 6-11 ships) and a lone or paired weak hit was spurious every time
# -- 2 "storage tanks" on Wembley Stadium (best 0.39), 3 on a golf course (0.32), 2 on Flushing Meadows,
# 1 on a mall car park, an "airplane" in a parking lot. So a category is reported only when it has at
# least MIN_COUNT boxes with a best score of MIN_BEST_SCORE, or a single hit so strong that it stands
# alone. Conservative on purpose: it drops a couple of genuine tiny detections, and a description that
# says "nothing found" is far less harmful than one that invents two storage tanks.
MIN_COUNT = 3
MIN_BEST_SCORE = 0.35
CONFIDENT_SINGLE = 0.55

def captioner_available(enabled: bool, checkpoint_dir: Path) -> bool:
    """The captioner is used only when switched on AND installed -- having the (gitignored) checkpoint on
    disk must not by itself change what users are told about their imagery."""
    return enabled and (checkpoint_dir / "caption_meta.json").exists()


# Decided once at import (restart the backend after changing the setting or installing the checkpoint),
# so the description the router LLM sees, the trace and the behaviour always agree -- same pattern as
# change_detection.
USE_CAPTIONS = captioner_available(settings.enable_captioner, CAPTION_CHECKPOINT_DIR)
USE_LANDCOVER = land_cover_adapter.USE_LANDCOVER
if settings.enable_captioner and not USE_CAPTIONS:
    logger.info("no captioner checkpoint at %s; scene_description will use the object scan only", CAPTION_CHECKPOINT_DIR)

_captioner = None  # lazy singleton -- the model is ~1GB and only loads on the first description


@serialize_first_call
def _get_captioner():
    global _captioner
    if _captioner is None:
        module = load_module(MODELS_DIR / "captioning" / "caption_tool.py", "caption_tool")
        _captioner = module.CaptionTool(str(CAPTION_CHECKPOINT_DIR))
    return _captioner


def _reportable(scores: list[float]) -> bool:
    best = max(scores)
    return (len(scores) >= MIN_COUNT and best >= MIN_BEST_SCORE) or best >= CONFIDENT_SINGLE


def _scan(image_path: str):
    """Returns (reported detections, evidence image path, categories dropped by the precision guard)."""
    module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
    found = grounding_adapter._get_tool().scan(image_path, SCAN_CATEGORIES, box_threshold=BOX_THRESHOLD)

    by_label: dict[str, list[float]] = {}
    for d in found:
        by_label.setdefault(d["phrase"], []).append(d["score"])
    kept = {label for label, scores in by_label.items() if _reportable(scores)}
    unreported = [
        {"label": label, "count": len(scores), "best_score": max(scores)}
        for label, scores in by_label.items()
        if label not in kept
    ]
    detections = [d for d in found if d["phrase"] in kept]

    evidence_path = None
    if detections:
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
        module.draw_boxes(image_path, detections, str(evidence_path))
    return detections, evidence_path, unreported


def _caption(image_path: str) -> dict[str, Any]:
    return _get_captioner().describe(image_path)


def _land_cover(image_path: str) -> dict[str, Any]:
    facts, _class_map = land_cover_adapter.analyze(image_path)  # the overlay image belongs to the land_cover_analysis tool
    return facts


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    image_path = str(query_input.images[0])

    # The detector and the land-cover model run on CPU and the captioner on the GPU (MPS), so they overlap.
    with ThreadPoolExecutor(max_workers=3) as pool:
        scan_future = pool.submit(_scan, image_path)
        caption_future = pool.submit(_caption, image_path) if USE_CAPTIONS else None
        land_future = pool.submit(_land_cover, image_path) if USE_LANDCOVER else None
        errors, detections, evidence_path, caption, unreported, land = [], [], None, None, [], None
        scan_ok = caption_ok = land_ok = False
        try:
            detections, evidence_path, unreported = scan_future.result()
            scan_ok = True
        except Exception as exc:
            errors.append(f"object scan unavailable ({exc})")
        if caption_future is not None:
            try:
                caption = caption_future.result()
                caption_ok = True
            except Exception as exc:
                errors.append(f"description model unavailable ({exc})")
        if land_future is not None:
            try:
                land = land_future.result()
                land_ok = True
            except Exception as exc:
                errors.append(f"land-cover model unavailable ({exc})")

    if not (scan_ok or caption_ok or land_ok):
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
    if land_ok:
        parts.append(land_cover_adapter.summarize(land))
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
    if land_ok:
        source_scores.append(land["confidence"])
    if scan_ok and detections:
        source_scores.append(max(d["score"] for d in detections))
    elif scan_ok and not (caption_ok or land_ok):
        source_scores.append(0.0)

    return ToolResult(
        tool_name="scene_description",
        text_summary=" ".join(parts),
        structured_data={
            "caption": caption["caption"] if caption_ok else None,
            "caption_confidence": caption["confidence"] if caption_ok else None,
            "land_cover": land if land_ok else None,
            "objects": objects,
            "scanned_categories": SCAN_CATEGORIES,
            "detections": detections,
            "total_objects": len(detections),
            "unreported": unreported,  # weak lone/paired detections the precision guard held back (audit trail only)
            "notes": errors,
        },
        evidence_image_path=evidence_path,
        confidence=sum(source_scores) / len(source_scores) if source_scores else 0.0,
    )


def _what_it_returns() -> str:
    if not (USE_CAPTIONS or USE_LANDCOVER):
        return (
            "It scans for the objects the detector handles reliably -- airplanes, ships and storage tanks -- "
            "and counts them, returning a marked-up image; it does NOT describe terrain, buildings, roads or "
            "land cover."
        )
    sources = []
    if USE_CAPTIONS:
        sources.append("a written description of the scene from a remote-sensing captioning model")
    if USE_LANDCOVER:
        sources.append(
            "the share of the image that is buildings, roads, trees, water and farmland, an approximate "
            "building count and the roof colours from a land-cover segmentation model"
        )
    sources.append("counts of the objects the detector handles reliably -- airplanes, ships and storage tanks -- with a marked-up image")
    text = "It returns " + "; ".join(sources[:-1]) + "; plus " + sources[-1] + "."
    if USE_CAPTIONS:
        text += " The description can be wrong about details and counts; the measured shares and the detector's counts are the reliable numbers."
    return text


_WHAT_IT_RETURNS = _what_it_returns()

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
    checkpoint_id=" + ".join(
        (["caption_model"] if USE_CAPTIONS else []) + (["landcover_unet.pt"] if USE_LANDCOVER else []) + ["dior_rsvg_finetuned.pth"]
    ),
)
