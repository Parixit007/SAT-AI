"""Registry adapter for the groundwater-potential specialist (backend/app/gee/). Needs a configured
GEE service account (see .env.example) -- registering the tool is always safe (metadata-only);
running it raises a clean RuntimeError (caught as a 503 in routes_query.py, same as the LLM
providers) until GEE is actually set up."""

import logging
import uuid
from typing import Any

from app.config import EVIDENCE_DIR
from app.gee.groundwater import GroundwaterScore, assess_groundwater_potential, render_thumbnail_url
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


def _data_completeness_confidence(score: GroundwaterScore) -> float:
    """Not a statistical confidence -- this is a deterministic GIS score, not a probabilistic
    model prediction. "Confidence" here is data completeness (how many of the layers -- currently
    5 -- had data for this location), so the UI doesn't overstate certainty it doesn't have."""
    total = len(score.layers)  # always the 5 fixed keys compute_groundwater_score() populates
    available = total - len(score.missing_layers)
    return available / total


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    import requests

    location = query_input.location
    lat, lon = location.lat, location.lon
    radius_m = arguments.get("radius_m", 1000.0)

    score = assess_groundwater_potential(lat, lon, radius_m)

    evidence_path = None
    try:
        thumb_url = render_thumbnail_url(lat, lon, radius_m)
        resp = requests.get(thumb_url, timeout=30)
        resp.raise_for_status()
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.png"
        evidence_path.write_bytes(resp.content)
    except Exception:
        # Thumbnail is a bonus -- don't sink the whole result over a rendering failure. Still
        # worth a log line, not total silence: a *systematic* failure here (a broken thumbnail
        # URL, GEE visualization params drifting invalid) would otherwise never surface anywhere.
        logger.warning("Groundwater evidence thumbnail failed for (%s, %s)", lat, lon, exc_info=True)

    layer_bits = [
        f"{name}={ls.normalized:.2f}" if ls.normalized is not None else f"{name}=missing"
        for name, ls in score.layers.items()
    ]
    text_summary = (
        f"Groundwater potential near ({lat:.5f}, {lon:.5f}): {score.category} "
        f"(score {score.overall_score:.2f}). Factors: {', '.join(layer_bits)}. "
        "This is a satellite-data screening estimate, not a substitute for a professional "
        "hydrogeological survey or test drilling."
    )
    if score.missing_layers:
        text_summary += f" Note: {', '.join(score.missing_layers)} data was unavailable for this location."

    return ToolResult(
        tool_name="groundwater_potential",
        text_summary=text_summary,
        structured_data={
            "category": score.category,
            "score": score.overall_score,
            "layers": {name: {"raw": ls.raw_value, "normalized": ls.normalized} for name, ls in score.layers.items()},
        },
        evidence_image_path=evidence_path,
        confidence=_data_completeness_confidence(score),
    )


TOOL_SPEC = ToolSpec(
    name="groundwater_potential",
    description=(
        "Estimate groundwater potential -- how favorable a location is for finding water if "
        "digging a well or tubewell -- from satellite-derived rainfall, topographic wetness, "
        "recent soil moisture, land cover, and proximity to surface water. Returns a Very "
        "Low..Very High rating. This is a "
        "heuristic screening estimate from public satellite data, not a substitute for a "
        "professional hydrogeological survey or test drilling. Use this for queries about "
        "groundwater, well/tubewell siting, or water availability underground at a location."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "radius_m": {
                "type": "number",
                "description": "Radius in meters around the location to average over (default 1000).",
            },
        },
        "required": [],
    },
    min_images=0,
    max_images=0,
    compatible_modalities=[],
    handler=_handle,
    checkpoint_id=None,
    uses_images=False,
    requires_location=True,
)
