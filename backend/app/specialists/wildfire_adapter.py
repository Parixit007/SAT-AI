"""Registry adapter for the wildfire-detection specialist (backend/app/gee/wildfire.py). Needs a
configured GEE service account (see .env.example) -- registering the tool is always safe
(metadata-only); running it raises a clean RuntimeError (caught as a 503 in routes_query.py, same
as the LLM providers and groundwater_potential) until GEE is actually set up."""

import uuid
from typing import Any

from app.config import EVIDENCE_DIR
from app.gee.wildfire import (
    FIRE_LOOKBACK_DAYS,
    DEFAULT_RADIUS_M,
    WildfireAssessment,
    assess_wildfire_risk,
    render_thumbnail_url,
)
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec


def _detection_confidence(assessment: WildfireAssessment) -> float:
    """Not a statistical confidence -- like groundwater_potential, this is a deterministic read of
    satellite detections, not a probabilistic model prediction. 0 when nothing was detected
    (nothing to be confident *about*, not "confidently no fire"); otherwise FIRMS' own 0-100
    detection confidence for the strongest hit, rescaled to 0-1, so the badge reflects how solid
    the underlying read is."""
    if assessment.max_confidence is None:
        return 0.0
    return min(1.0, assessment.max_confidence / 100.0)


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    import requests

    location = query_input.location
    lat, lon = location.lat, location.lon
    radius_m = arguments.get("radius_m", DEFAULT_RADIUS_M)
    lookback_days = arguments.get("lookback_days", FIRE_LOOKBACK_DAYS)

    assessment = assess_wildfire_risk(lat, lon, radius_m, lookback_days)

    evidence_path = None
    try:
        thumb_url = render_thumbnail_url(lat, lon, radius_m, lookback_days)
        resp = requests.get(thumb_url, timeout=30)
        resp.raise_for_status()
        evidence_path = EVIDENCE_DIR / f"{uuid.uuid4().hex}.png"
        evidence_path.write_bytes(resp.content)
    except Exception:
        pass  # thumbnail is a bonus -- don't sink the whole result over a rendering failure

    text_summary = (
        f"Wildfire check near ({lat:.5f}, {lon:.5f}) over the last {lookback_days} days: "
        f"{assessment.status}."
    )
    if assessment.detection_days > 0:
        text_summary += (
            f" Detected on {assessment.detection_days} of the last {lookback_days} days "
            f"(most recently {assessment.most_recent_date}), peak confidence "
            f"{assessment.max_confidence:.0f}/100, peak brightness {assessment.max_brightness_k:.0f}K."
        )
    text_summary += (
        " Based on NASA FIRMS satellite thermal-anomaly detections (~1km resolution, MODIS) -- "
        "a screening signal, not a substitute for official fire-agency alerts or ground reports."
    )

    return ToolResult(
        tool_name="wildfire_detection",
        text_summary=text_summary,
        structured_data={
            "status": assessment.status,
            "detection_days": assessment.detection_days,
            "lookback_days": assessment.lookback_days,
            "max_confidence": assessment.max_confidence,
            "max_brightness_k": assessment.max_brightness_k,
            "most_recent_date": assessment.most_recent_date,
        },
        evidence_image_path=evidence_path,
        confidence=_detection_confidence(assessment),
    )


TOOL_SPEC = ToolSpec(
    name="wildfire_detection",
    description=(
        "Check for active wildfire or fire activity near a location using satellite "
        "thermal-anomaly detections over a recent lookback window (default 10 days). Returns "
        "whether a fire was detected, how many of the recent days showed a detection, peak "
        "detection confidence, and peak fire-pixel brightness temperature. This is a screening "
        "signal from NASA FIRMS satellite data (~1km resolution), not a substitute for official "
        "fire-agency alerts or ground reports. Use this for queries about wildfires, fire risk, "
        "burning, or fire/smoke activity at a location."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "radius_m": {
                "type": "number",
                "description": "Radius in meters around the location to check (default 5000).",
            },
            "lookback_days": {
                "type": "number",
                "description": "How many days back to check for fire detections (default 10).",
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
