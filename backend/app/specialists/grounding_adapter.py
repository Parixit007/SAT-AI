"""Registry adapter for the text-guided region-grounding specialist (models/grounding/). It needs a
git-cloned Open-GroundingDino checkout and a few pip packages (see grounding_tool.py's Setup) --
the GroundingTool singleton is only constructed on the first actual call, so registering this tool
(cheap, metadata-only) never fails even before that setup step is done; only running a grounding
query does.

This is also the tool that answers "how many X?": every detection above the threshold comes back,
so the number of boxes is the count (the VQA checkpoint answers count questions with a single
unreliable word). The detector is sensitive to wording -- a bare category noun works, a whole
question doesn't -- so category-style queries are normalised before they reach it."""

import os
import re
import tempfile
import uuid
from collections import Counter
from typing import Any

from app.concurrency import serialize_first_call
from app.config import EVIDENCE_DIR, GROUNDING_CHECKPOINT, MODELS_DIR
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

_grounding_tool = None  # lazy singleton -- see module docstring

# Measured on a real airport scene (~50 aircraft): "airplane" 39 boxes, "aeroplane" 41, but the
# British plural "aeroplanes" only 19, and the whole question "how many aeroplanes do you see" 3.
_QUESTION_PREFIX = re.compile(
    r"^(?:please\s+)?(?:how\s+many|what(?:'s|\s+is)\s+the\s+(?:number|count)\s+of|(?:the\s+)?number\s+of|count(?:\s+the)?(?:\s+number\s+of)?|"
    r"are\s+there(?:\s+any)?|is\s+there(?:\s+an?)?|do\s+you\s+see(?:\s+any)?|can\s+you\s+(?:see|find|count)(?:\s+any)?|"
    r"find|locate|highlight|show(?:\s+me)?|detect|identify|mark|outline)(?:\s+all)?(?:\s+the)?\s+",
    re.IGNORECASE,
)
_QUESTION_SUFFIX = re.compile(
    r"\s+(?:do\s+you\s+see|can\s+you\s+see|are\s+there|is\s+there|are\s+visible|are\s+present|"
    r"(?:visible\s+|present\s+)?(?:in|on|within)\s+(?:the|this)\s+(?:satellite\s+)?(?:image|picture|photo|scene|area|view|frame)|"
    r"visible|present|is|are|there|here)$",
    re.IGNORECASE,
)
_SYNONYMS = {
    "aeroplane": "airplane", "aircraft": "airplane", "plane": "airplane", "airliner": "airplane",
    "boat": "ship", "vessel": "ship", "harbour": "harbor",
    "car": "vehicle", "truck": "vehicle", "automobile": "vehicle",
}
_MAX_CATEGORY_WORDS = 3


def _singular(word: str) -> str:
    if len(word) <= 3 or word.endswith("ss") or word.endswith("us"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def _normalize_part(part: str) -> str:
    text = re.sub(r"[?!,;:\"']", " ", part.strip().rstrip(".")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    stripped = _QUESTION_PREFIX.sub("", text)
    while True:  # "...storage tanks are there in this image" carries two trailing phrases
        shorter = _QUESTION_SUFFIX.sub("", stripped).strip()
        if shorter == stripped:
            break
        stripped = shorter
    stripped = re.sub(r"^(?:all\s+)?(?:the\s+)?", "", stripped)
    words = stripped.split()
    if not words or len(words) > _MAX_CATEGORY_WORDS:
        return part.strip()  # a real referring expression ("the ship near the harbor entrance") -- leave alone
    words[-1] = _singular(words[-1])
    if len(words) == 1:
        words[0] = _SYNONYMS.get(words[0], words[0])
    return " ".join(words)


def normalize_category_query(query: str) -> str:
    """Best-effort: turn what the router (or a user, via the manual picker) passed into the short
    singular category noun the detector responds to, and leave real referring expressions alone.
    Multi-category queries ("ship . vehicle") are handled per part."""
    parts = [p for p in re.split(r"\s*\.\s+|\s*\.$", query.strip()) if p.strip()]
    if not parts:
        return query.strip()
    return " . ".join(_normalize_part(p) for p in parts)


# Small objects on a big frame: the detector shrinks any input to ~800 px, so a car that is 20 px long in a
# 1024 px street view is a dozen pixels to it. Measured on three real zoomed street scenes (~0.2 m/px), boxes
# on overlapping 512 px tiles at native resolution -- Brooklyn rows / a suburb / a parking lot:
#   "vehicle" 9 / 14 / 9    "car" 87 / 48 / 14    "small vehicle" 91 / 52 / 15    "cars" 140 / 69 / 34
# (a whole 1024 px frame: "car" 19 / 26 / 7 boxes, "vehicle" 1). The plural "cars" is the wording this
# checkpoint responds to best -- looked at, its boxes sit on the curbside cars -- so a vehicle question is asked
# as "cars" (and labelled "vehicle"). It is still a LOWER BOUND: a lot with ~80 tightly packed cars produced
# only a handful of boxes in the lot, so the summary says so. Only these queries pay the roughly nine-fold cost;
# a stadium or a dam would be cut in pieces by tiling.
TILED_QUERIES = {"vehicle"}
DETECTOR_PROMPTS = {"vehicle": "cars"}
MIN_TILED_SIDE = 700  # below this the whole frame is already about one detector input
TILE, TILE_OVERLAP = 512, 160
MERGE_IOU = 0.5


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of `tile`-sized windows covering [0, length) with at least `overlap` pixels shared."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def merge_boxes(detections: list[dict[str, Any]], iou_threshold: float = MERGE_IOU) -> list[dict[str, Any]]:
    """Greedy non-maximum suppression, best score first -- the same object seen in two overlapping tiles is one."""
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda d: -d["score"]):
        ax1, ay1, ax2, ay2 = det["bbox_xyxy"]
        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        duplicate = False
        for other in kept:
            bx1, by1, bx2, by2 = other["bbox_xyxy"]
            inter = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(min(ay2, by2) - max(ay1, by1), 0)
            union = area_a + max(bx2 - bx1, 0) * max(by2 - by1, 0) - inter
            if union > 0 and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _ground_tiled(tool, image_path: str, query: str) -> tuple[list[dict[str, Any]], int]:
    """Run the detector on overlapping native-resolution tiles and merge; returns (detections, tile count)."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    found: list[dict[str, Any]] = []
    tiles = 0
    with tempfile.TemporaryDirectory() as tmp:
        for y0 in tile_starts(height, TILE, TILE_OVERLAP):
            for x0 in tile_starts(width, TILE, TILE_OVERLAP):
                tile_path = os.path.join(tmp, "tile.png")
                image.crop((x0, y0, min(x0 + TILE, width), min(y0 + TILE, height))).save(tile_path)
                tiles += 1
                for det in tool.ground(tile_path, query):
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    found.append({**det, "bbox_xyxy": [round(x1 + x0, 1), round(y1 + y0, 1), round(x2 + x0, 1), round(y2 + y0, 1)]})
    return merge_boxes(found), tiles


@serialize_first_call
def _get_tool():
    global _grounding_tool
    if _grounding_tool is None:
        module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
        _grounding_tool = module.GroundingTool(checkpoint_path=str(GROUNDING_CHECKPOINT))
    return _grounding_tool


def _longest_side(image_path) -> int:
    from PIL import Image

    with Image.open(image_path) as image:
        return max(image.size)


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    module = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool")
    original_query = str(arguments.get("query", ""))
    query = normalize_category_query(original_query)
    top_k = arguments.get("top_k")  # None = every match above the threshold, which is what a count needs
    image_path = query_input.images[0]

    tool = _get_tool()
    tiled_over = 0
    prompt = DETECTOR_PROMPTS.get(query.lower(), query)  # what the detector is actually asked (see DETECTOR_PROMPTS)
    if query.lower() in TILED_QUERIES and _longest_side(image_path) >= MIN_TILED_SIDE:
        detections, tiled_over = _ground_tiled(tool, str(image_path), prompt)
        detections.sort(key=lambda d: d["score"], reverse=True)
        if top_k is not None:
            detections = detections[:top_k]
    else:
        detections = tool.ground(str(image_path), prompt, top_k=top_k)
    if prompt != query:
        detections = [{**d, "phrase": query} for d in detections]  # label what the user asked about, not the prompt wording

    evidence_path = None
    if detections:
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
        module.draw_boxes(str(image_path), detections, str(evidence_path))

    by_phrase = dict(Counter(d["phrase"] for d in detections))
    if detections:
        top = detections[0]
        lo, hi = min(d["score"] for d in detections), max(d["score"] for d in detections)
        text_summary = f"Found {len(detections)} region(s) matching '{query}' (detection scores {lo:.2f}-{hi:.2f})."
        if len(detections) <= 3:
            text_summary += f" Best match: '{top['phrase']}' at {top['bbox_xyxy']} (score {top['score']:.2f})."
        confidence = top["score"]
    else:
        text_summary = f"No regions matching '{query}' were found above the detection threshold."
        confidence = 0.0

    if tiled_over and detections:
        text_summary += (
            f" Counted on {tiled_over} overlapping full-resolution tiles: treat the number as a lower bound, because "
            "tightly packed vehicles (a dense parking lot) are mostly missed."
        )
    structured = {"detections": detections, "count": len(detections), "by_phrase": by_phrase, "query": query}
    if tiled_over:
        structured["tiles"] = tiled_over
    if query != original_query:
        structured["original_query"] = original_query
    return ToolResult(
        tool_name="text_guided_grounding",
        text_summary=text_summary,
        structured_data=structured,
        evidence_image_path=evidence_path,
        confidence=confidence,
    )


TOOL_SPEC = ToolSpec(
    name="text_guided_grounding",
    description=(
        "Locate the region(s) in a single image that match an object category or a text "
        "description (e.g. 'airplane', 'storage tank', or 'the ship near the harbor entrance'), "
        "returning a bounding box for every match. This is also THE tool for counting: use it for "
        "'how many airplanes/ships/vehicles/storage tanks...?' -- the number of boxes returned is "
        "the count, and the marked-up image is the evidence. Prefer it over visual_question_answering "
        "for any counting or 'where is / show me / highlight' request."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Just the object to find: a short, singular category noun in American English "
                    "('airplane', 'ship', 'vehicle', 'storage tank', 'bridge') -- NOT the user's whole "
                    "question -- or, for one specific object, a short referring expression."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Optional cap on the number of matches returned. Omit it to get every match (required for counting).",
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
