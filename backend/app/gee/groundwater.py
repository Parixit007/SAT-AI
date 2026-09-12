"""Groundwater-potential scoring: a deterministic weighted overlay of satellite-derived layers
(rainfall, topographic wetness, land cover, distance to surface water), following the published
"Groundwater Potential Zone" (GPZ) mapping methodology. Deliberately split in two:

  - `fetch_raw_layer_values()` talks to Earth Engine (network I/O, needs real credentials to run
    or meaningfully test -- see the plan's Verification section).
  - `compute_groundwater_score()` is pure Python over already-fetched numbers -- fully unit
    testable without any GEE dependency at all, which is where the actual scoring logic
    (normalization ranges, weights, classification thresholds) lives and can be checked/tuned.

Weights and normalization ranges below are a documented starting point (see the plan), not
rigorously fit -- calibrate against real local well outcomes once available."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

WEIGHTS = {"rainfall": 0.30, "twi": 0.30, "dist_to_water": 0.20, "landcover": 0.20}

RAINFALL_RANGE_MM = (0.0, 2500.0)  # annual mm; higher = better (more recharge)
TWI_RANGE = (2.0, 15.0)  # topographic wetness index; higher = better (water accumulates)
DIST_TO_WATER_RANGE_M = (0.0, 5000.0)  # meters; CLOSER = better (inverted during normalization)

# ESA WorldCover v200 class codes -> recharge suitability, 0 (poor) to 1 (good). Built-up/impervious
# surfaces score lowest; forest/cropland/grassland score highest (decent infiltration); open water
# scores 0 (the question "will digging here find water" doesn't apply to a lake).
LAND_COVER_SUITABILITY = {
    10: 0.7,  # tree cover
    20: 0.6,  # shrubland
    30: 0.6,  # grassland
    40: 0.65,  # cropland
    50: 0.1,  # built-up
    60: 0.3,  # bare / sparse vegetation
    70: 0.2,  # snow / ice
    80: 0.0,  # permanent water body
    90: 0.5,  # herbaceous wetland
    95: 0.5,  # mangrove
    100: 0.3,  # moss / lichen
}

CATEGORY_THRESHOLDS = [
    (0.8, "Very High"),
    (0.6, "High"),
    (0.4, "Moderate"),
    (0.2, "Low"),
]


@dataclass
class LayerScore:
    raw_value: Optional[float]
    normalized: Optional[float]  # 0-1, None if raw_value was missing/unavailable


@dataclass
class GroundwaterScore:
    overall_score: float  # 0-1
    category: str  # "Very Low" .. "Very High"
    layers: dict[str, LayerScore] = field(default_factory=dict)
    missing_layers: list[str] = field(default_factory=list)


def _normalize(value: Optional[float], low: float, high: float, invert: bool = False) -> Optional[float]:
    if value is None:
        return None
    score = (float(value) - low) / (high - low)
    score = max(0.0, min(1.0, score))
    return 1.0 - score if invert else score


def _classify(score: float) -> str:
    for threshold, label in CATEGORY_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Low"


def compute_groundwater_score(raw: dict[str, Optional[float]]) -> GroundwaterScore:
    """`raw` maps layer name -> raw value (or None/missing if Earth Engine had no data there --
    e.g. CHIRPS doesn't cover high latitudes). Missing layers are excluded from the weighted sum
    with the remaining weights renormalized, rather than treated as 0 (which would silently and
    incorrectly penalize a location just because one dataset didn't cover it)."""
    layers: dict[str, LayerScore] = {
        "rainfall": LayerScore(raw.get("rainfall"), _normalize(raw.get("rainfall"), *RAINFALL_RANGE_MM)),
        "twi": LayerScore(raw.get("twi"), _normalize(raw.get("twi"), *TWI_RANGE)),
        "dist_to_water": LayerScore(
            raw.get("dist_to_water"), _normalize(raw.get("dist_to_water"), *DIST_TO_WATER_RANGE_M, invert=True)
        ),
    }
    lc_raw = raw.get("landcover")
    # round(), not int(): a mode-reducer over a categorical band can come back as e.g.
    # 49.999999999999996 due to floating-point noise in Earth Engine's own pipeline -- int()
    # truncates that to 49 (a miss), silently dropping a layer that's actually present.
    lc_normalized = LAND_COVER_SUITABILITY.get(round(lc_raw)) if lc_raw is not None else None
    layers["landcover"] = LayerScore(lc_raw, lc_normalized)

    missing = [name for name, ls in layers.items() if ls.normalized is None]
    available_weight = sum(WEIGHTS[name] for name, ls in layers.items() if ls.normalized is not None)

    if available_weight == 0:
        overall = 0.0
    else:
        overall = (
            sum(WEIGHTS[name] * ls.normalized for name, ls in layers.items() if ls.normalized is not None)
            / available_weight
        )

    return GroundwaterScore(overall_score=overall, category=_classify(overall), layers=layers, missing_layers=missing)


def fetch_raw_layer_values(lat: float, lon: float, radius_m: float = 1000.0) -> dict[str, Optional[float]]:
    """The Earth Engine I/O half -- needs a configured service account (see gee/client.py) to run
    at all, and hasn't been exercised against the live service yet (no credentials in this
    environment). Two separate reduceRegion calls because land cover is categorical (mode) while
    the rest are continuous (mean) -- one reducer can't correctly apply to both band types."""
    import ee

    from app.gee import layers as L
    from app.gee.client import ensure_initialized

    ensure_initialized()

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m)
    scale = 30  # meters -- matches SRTM/WorldCover native resolution

    continuous = (
        L.annual_rainfall_mm(date.today().isoformat())
        .addBands(L.topographic_wetness_index())
        .addBands(L.distance_to_water_m())
    )
    continuous_stats = continuous.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, bestEffort=True
    ).getInfo()
    landcover_stats = L.land_cover().reduceRegion(
        reducer=ee.Reducer.mode(), geometry=region, scale=scale, bestEffort=True
    ).getInfo()

    return {
        "rainfall": continuous_stats.get("rainfall"),
        "twi": continuous_stats.get("twi"),
        "dist_to_water": continuous_stats.get("dist_to_water"),
        "landcover": landcover_stats.get("landcover"),
    }


def render_thumbnail_url(lat: float, lon: float, radius_m: float = 1000.0) -> str:
    """A small TWI (topographic wetness) map around the query point, as visual evidence -- warmer
    colors (the "red" end of the palette) mark higher water-accumulation potential."""
    import ee

    from app.gee import layers as L
    from app.gee.client import ensure_initialized

    ensure_initialized()

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m * 2).bounds()
    vis = L.topographic_wetness_index().clip(region).visualize(min=2, max=15, palette=["blue", "green", "yellow", "red"])
    return vis.getThumbURL({"region": region, "dimensions": 400})


def assess_groundwater_potential(lat: float, lon: float, radius_m: float = 1000.0) -> GroundwaterScore:
    raw = fetch_raw_layer_values(lat, lon, radius_m)
    return compute_groundwater_score(raw)
