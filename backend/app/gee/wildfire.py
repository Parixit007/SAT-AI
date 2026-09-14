"""Active-fire detection: a deterministic read of NASA FIRMS thermal-anomaly detections near a
location, not a trained model -- same pure-Python-classification / impure-GEE-I/O split as
groundwater.py, for the same reason (testable without live GEE access). See
`gee/layers.py`'s `active_fire_collection()` docstring for the dataset/band details.

Live-verified (Sept 2026, western US, real fire-season activity): a 5km-radius / 10-day query
around a real detected fire pixel correctly found exactly the one day with a hit (confidence 79,
T21 325K); the same query at a nearby fire-free point correctly found zero. A deliberately huge
stress-test region (the whole western US, 30 days) hit Earth Engine's "too many concurrent
aggregations" limit -- not a concern at this tool's realistic per-query scale (one point, a
few-km radius), but a real constraint worth knowing if `radius_m`/`lookback_days` were ever opened
up much wider than the defaults below."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

FIRE_LOOKBACK_DAYS = 10  # long enough to catch at least one cloud-free overpass -- fire detection,
# unlike a smooth physical field like soil moisture, needs a genuinely clear-sky pass and can't be
# averaged through cloud cover. Short enough that a positive read still means "recent."

DEFAULT_RADIUS_M = 5000.0  # larger than groundwater's 1000m default: FIRMS pixels are ~1km native
# resolution, so a small radius risks missing a real nearby fire on a grid-alignment technicality;
# "is there a fire near here" is naturally a neighborhood-scale question, not a single-pixel one.

HIGH_CONFIDENCE_THRESHOLD = 80  # NASA FIRMS' own documented nominal-vs-high boundary


@dataclass
class DailyFireHit:
    date: str
    max_confidence: float
    max_brightness_k: float


@dataclass
class WildfireAssessment:
    status: str  # "Active Fire Detected" | "Possible Fire Activity" | "No Fire Detected"
    detection_days: int
    max_confidence: Optional[float]
    max_brightness_k: Optional[float]
    most_recent_date: Optional[str]
    lookback_days: int
    hits: list[DailyFireHit] = field(default_factory=list)


def classify_fire_activity(daily_hits: list[dict], lookback_days: int) -> WildfireAssessment:
    """Pure Python -- `daily_hits` is a list of {"date", "max_confidence", "max_brightness_k"}
    dicts for days that already had at least one fire-confidence>0 pixel in the query region
    (filtered server-side by `fetch_recent_fire_activity()`). No GEE dependency, fully unit
    testable."""
    if not daily_hits:
        return WildfireAssessment(
            status="No Fire Detected",
            detection_days=0,
            max_confidence=None,
            max_brightness_k=None,
            most_recent_date=None,
            lookback_days=lookback_days,
            hits=[],
        )

    hits = sorted((DailyFireHit(**h) for h in daily_hits), key=lambda h: h.date)
    max_confidence = max(h.max_confidence for h in hits)
    max_brightness = max(h.max_brightness_k for h in hits)
    most_recent = hits[-1].date  # ISO YYYY-MM-DD strings sort chronologically as plain text

    status = "Active Fire Detected" if max_confidence >= HIGH_CONFIDENCE_THRESHOLD else "Possible Fire Activity"

    return WildfireAssessment(
        status=status,
        detection_days=len(hits),
        max_confidence=max_confidence,
        max_brightness_k=max_brightness,
        most_recent_date=most_recent,
        lookback_days=lookback_days,
        hits=hits,
    )


def fetch_recent_fire_activity(
    lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M, lookback_days: int = FIRE_LOOKBACK_DAYS
) -> list[dict]:
    """The Earth Engine I/O half -- needs a configured service account to run. Maps one
    `reduceRegion` per image in the window server-side (not a Python loop -- one round trip) rather
    than reducing to a single time-collapsed composite, because `detection_days`/`most_recent_date`
    need day-level granularity that compositing away would lose."""
    import ee

    from app.gee import layers as L
    from app.gee.client import ensure_initialized

    ensure_initialized()

    region = ee.Geometry.Point([lon, lat]).buffer(radius_m)
    scale = 1000  # FIRMS' native resolution

    col = L.active_fire_collection(date.today().isoformat(), lookback_days).filterBounds(region)

    def _per_day(img):
        stats = img.reduceRegion(reducer=ee.Reducer.max(), geometry=region, scale=scale, bestEffort=True)
        return ee.Feature(
            None,
            {
                "date": img.date().format("YYYY-MM-dd"),
                "max_confidence": stats.get("confidence"),
                "max_brightness_k": stats.get("T21"),
            },
        )

    daily = ee.FeatureCollection(col.map(_per_day))
    hits = daily.filter(ee.Filter.gt("max_confidence", 0))
    return [f["properties"] for f in hits.getInfo()["features"]]


def render_thumbnail_url(
    lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M, lookback_days: int = FIRE_LOOKBACK_DAYS
) -> str:
    """A small map of max fire-detection confidence around the query point over the lookback
    window -- warmer colors mark higher-confidence detections, black is no detection."""
    import ee

    from app.gee import layers as L
    from app.gee.client import ensure_initialized

    ensure_initialized()

    region = ee.Geometry.Point([lon, lat]).buffer(radius_m * 2).bounds()
    col = L.active_fire_collection(date.today().isoformat(), lookback_days)
    composite = col.select("confidence").max().clip(region)
    vis = composite.visualize(min=0, max=100, palette=["black", "yellow", "orange", "red"])
    return vis.getThumbURL({"region": region, "dimensions": 400})


def assess_wildfire_risk(
    lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M, lookback_days: int = FIRE_LOOKBACK_DAYS
) -> WildfireAssessment:
    hits = fetch_recent_fire_activity(lat, lon, radius_m, lookback_days)
    return classify_fire_activity(hits, lookback_days)
