"""Registry adapter for the land-cover / buildings specialist (models/landcover/).

A U-Net trained on OpenEarthMap labels every pixel as building, road, tree, water, agriculture land, rangeland
(grass / scrub), developed space (paved lots, plazas, yards) or bareland. From that map this adapter reports what
no other specialist in the app can: the share of the scene in each class, an approximate count of distinct
buildings, and the roof colours read from the pixels inside the building mask. `scene_description` also calls
analyze() for its own summary, so "describe this image" mentions roads, buildings and vegetation.

Two honest limits are stated in the text the user and the answer-phrasing model both see: touching buildings
merge into one region, so dense blocks are under-counted; and roof colours are read from the image, where
shadow can look like a dark roof."""

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.concurrency import serialize_first_call
from app.config import EVIDENCE_DIR, LANDCOVER_CHECKPOINT, MODELS_DIR, settings
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

_MODULE_PATH = MODELS_DIR / "landcover" / "landcover_tool.py"
SHOW_MIN_SHARE = 0.01  # classes under 1% of the image stay in the data but are left out of the sentence
MAX_ROOF_COLOURS = 3


def landcover_available(enabled: bool, checkpoint: Path) -> bool:
    """Used only when switched on AND installed, like the captioner -- decided once at import."""
    return enabled and checkpoint.exists()


USE_LANDCOVER = landcover_available(settings.enable_landcover, LANDCOVER_CHECKPOINT)

_tool = None  # lazy singleton -- the model is ~100MB and loads on first use


@serialize_first_call
def _get_tool():
    global _tool
    if _tool is None:
        module = load_module(_MODULE_PATH, "landcover_tool")
        _tool = module.LandCoverTool(str(LANDCOVER_CHECKPOINT))
    return _tool


def analyze(image_path: str) -> tuple[dict[str, Any], np.ndarray]:
    """(facts, class_map). `facts` is JSON-able: the class shares, the dominant class, the building count, the
    roof colours and the model's mean top-class probability (a model score, not a calibrated probability)."""
    result = _get_tool().analyze(image_path)
    fractions = result["fractions"]
    facts = {
        "fractions": {name: round(share, 4) for name, share in fractions.items()},
        "dominant": max(fractions, key=fractions.get),
        "building_count": int(result["building_count"]),
        "roof_colors": result["roof_colors"],
        "confidence": round(float(result["confidence"]), 4),
        "image_size": result["image_size"],
    }
    return facts, result["class_map"]


def _percent(share: float) -> str:
    return f"{round(share * 100)}%"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def summarize(facts: dict[str, Any]) -> str:
    """The plain-language summary of a land-cover analysis, with its limits stated."""
    fractions = facts["fractions"]
    shown = [(name, share) for name, share in sorted(fractions.items(), key=lambda kv: -kv[1]) if share >= SHOW_MIN_SHARE]
    sentences = ["Land cover (from a segmentation model, so approximate): " + ", ".join(f"{name} {_percent(share)}" for name, share in shown) + "."]

    count, buildings = facts["building_count"], fractions.get("building", 0.0)
    if buildings < 0.005 and count == 0:
        sentences.append("No buildings were found.")
    elif count == 0:
        sentences.append("Buildings cover part of the image but no separate building outlines could be told apart.")
    else:
        sentences.append(
            f"About {count} separate building outlines (buildings that touch merge into one, so dense blocks are under-counted)."
        )
    roofs = [c for c in facts.get("roof_colors", []) if c["share"] >= 0.05][:MAX_ROOF_COLOURS]
    if roofs and buildings >= SHOW_MIN_SHARE:
        lead = "mostly" if roofs[0]["share"] >= 0.5 else "a mix of"  # "mostly" only when one colour family really dominates
        sentences.append(
            f"Roofs are {lead} " + _join([f"{c['name']} ({_percent(c['share'])})" for c in roofs])
            + " (colours read from the image; shadows can look like dark roofs)."
        )
    return " ".join(sentences)


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    image_path = str(query_input.images[0])
    facts, class_map = analyze(image_path)

    module = load_module(_MODULE_PATH, "landcover_tool")
    evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.jpg"
    module.draw_overlay(image_path, class_map, str(evidence_path))

    return ToolResult(
        tool_name="land_cover_analysis",
        text_summary=summarize(facts),
        structured_data=facts,
        evidence_image_path=evidence_path,
        confidence=facts["confidence"],
    )


TOOL_SPEC = ToolSpec(
    name="land_cover_analysis",
    description=(
        "THE tool for buildings, roads and vegetation in ONE optical image -- how many buildings, where they are "
        "('highlight / show the buildings'), what colour the roofs are, how built-up or green the area is. It "
        "measures what covers the ground by labelling every pixel as building, road, tree, "
        "water, agriculture land, rangeland (grass or scrub), developed space (paved lots, plazas, yards) or "
        "bareland. Returns each class's share of the image, an approximate count of distinct buildings (buildings "
        "that touch merge, so dense blocks are under-counted), the roof colours, and a colour-coded map. Use it for "
        "questions about how much of the area is built-up / buildings / roads / trees / vegetation / farmland / "
        "water, how many buildings there are, what colour the roofs are, or how green versus paved an area is. It "
        "does NOT count cars or other small objects (text_guided_grounding does that), and a general 'describe "
        "this image' request should use scene_description."
    ),
    parameters_schema={"type": "object", "properties": {}, "required": []},
    min_images=1,
    max_images=1,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id="landcover_unet.pt",
)
