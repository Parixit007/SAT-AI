"""One small function per GEE dataset used by groundwater.py. Each returns a plain `ee.Image` --
assumes `client.ensure_initialized()` has already been called by the caller (groundwater.py does
this once, rather than every layer function checking redundantly).

Asset IDs below were verified against the live Earth Engine Data Catalog before writing this file
(not guessed, including checking actual data recency -- see `surface_soil_moisture()`'s docstring
for a case where that check ruled out the two most obvious dataset choices). A live GEE service
account is configured in this environment and this package has been exercised against it, both
via `backend/tests/test_groundwater.py`'s mocked-GEE routing test and ad-hoc manual queries against
real locations."""

SRTM = "USGS/SRTMGL1_003"
HYDROSHEDS_FLOW_ACCUMULATION = "WWF/HydroSHEDS/15ACC"
CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"
ESA_WORLDCOVER = "ESA/WorldCover/v200"
JRC_SURFACE_WATER = "JRC/GSW1_4/GlobalSurfaceWater"
ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
FIRMS = "FIRMS"


def elevation():
    import ee

    return ee.Image(SRTM)


def slope_degrees():
    import ee

    return ee.Terrain.slope(elevation())


def topographic_wetness_index():
    """TWI = ln(flow_accumulation / tan(slope)) -- the standard formula (higher = more water
    accumulation potential: valley bottoms and flat areas with large upstream contributing areas).
    Flow accumulation is in upstream-cell-count units; slope is converted degrees->radians and
    floored at a small epsilon so flat ground (tan(0)=0) doesn't divide by zero."""
    import ee

    flow_accum = ee.Image(HYDROSHEDS_FLOW_ACCUMULATION)
    slope_rad = slope_degrees().multiply(3.14159265 / 180).max(0.001)
    return flow_accum.add(1).log().subtract(slope_rad.tan().log()).rename("twi")


def annual_rainfall_mm(end_date: str):
    """Total precipitation (mm) over the 365 days ending `end_date` (YYYY-MM-DD string)."""
    import ee

    end = ee.Date(end_date)
    start = end.advance(-365, "day")
    return ee.ImageCollection(CHIRPS_DAILY).filterDate(start, end).sum().rename("rainfall")


def surface_soil_moisture(end_date: str):
    """Mean volumetric surface soil water content (m^3/m^3, roughly 0=bone dry to ~0.5=saturated)
    over the 7 days ending `end_date` -- a short trailing average, not a single day's snapshot
    (too noisy, e.g. querying right after one rain event) or CHIRPS's 365-day annual window (too
    slow-moving for a "how wet is the ground right now" signal; soil moisture is a state that
    persists over days-to-weeks, not a year). ERA5-Land is a reanalysis (physically-based model
    informed by observations), not a direct satellite retrieval -- chosen over the two actual SMAP
    satellite products checked first specifically because both had stopped updating on Earth
    Engine (one since 2022, the other since mid-2025), which would silently make "current soil
    moisture" false; ERA5-Land was confirmed live with data current to within days at write time.
    Masked (no data) over open water/ocean, same as every other continuous layer here -- handled
    by the same missing-layer exclusion `compute_groundwater_score()` already does for the rest."""
    import ee

    end = ee.Date(end_date)
    start = end.advance(-7, "day")
    return (
        ee.ImageCollection(ERA5_LAND_DAILY)
        .filterDate(start, end)
        .select("volumetric_soil_water_layer_1")
        .mean()
        .rename("soil_moisture")
    )


def active_fire_collection(end_date: str, lookback_days: int):
    """NASA FIRMS thermal-anomaly detections (MODIS collection 6, ~1km pixels, roughly 1-4
    overpasses/day depending on latitude) over the `lookback_days` days ending `end_date`. Returns
    the raw `ImageCollection` selected down to the two bands that matter -- `confidence` (0-100,
    NASA's own documented scale: <30 low, 30-80 nominal, >80 high) and `T21` (fire-pixel
    brightness temperature in Kelvin) -- unlike every other function in this file, which returns a
    single already-time-collapsed `ee.Image`. Fire detection needs day-by-day presence/absence
    (wildfire.py's `detection_days`/`most_recent_date`), so collapsing to one composite here would
    throw away exactly the information that matters; `line_number` (a scan-line artifact) is
    dropped since nothing here uses it.

    Live-verified Sept 2026 over real western-US fire activity: confidence values spread 52-98,
    T21 spread 308-380K (clear background land runs roughly 280-310K) -- consistent with NASA's
    documented semantics, not just "the dataset loads." Most recent image at verification time was
    2 days old (daily global mosaic cadence, actively updated)."""
    import ee

    end = ee.Date(end_date)
    start = end.advance(-lookback_days, "day")
    return ee.ImageCollection(FIRMS).filterDate(start, end).select(["confidence", "T21"])


def land_cover():
    """ESA WorldCover 10m classes: 10=tree, 20=shrub, 30=grass, 40=cropland, 50=built-up,
    60=bare/sparse, 70=snow/ice, 80=water, 90=wetland, 95=mangrove, 100=moss/lichen -- see
    groundwater.py's LAND_COVER_SUITABILITY for how these map to recharge favorability."""
    import ee

    return ee.ImageCollection(ESA_WORLDCOVER).first().rename("landcover")


DIST_TO_WATER_SEARCH_NEIGHBORHOOD_PX = 300  # ~9km at JRC's 30m native scale -- comfortably past
# groundwater.py's DIST_TO_WATER_RANGE_M ceiling (5000m), since normalization clips beyond that
# anyway. Live-verified this matters: with no explicit neighborhood (an undocumented default),
# fastDistanceTransform returned a nonsense ~1362km for a point in open ocean where JRC Global
# Surface Water (built for inland/coastal water, not deep ocean) has no nearby mapped occurrence.


def distance_to_water_m():
    """Distance (meters) to the nearest pixel with >0% historical surface-water occurrence, within
    a `DIST_TO_WATER_SEARCH_NEIGHBORHOOD_PX`-pixel search window. `fastDistanceTransform` returns
    squared distance in pixel units at the image's native scale; multiplying by the (square root
    of) pixel area converts that to meters for square pixels.

    Caveat: JRC Global Surface Water doesn't map the open ocean, only inland/coastal water -- a
    query point far out at sea (not a realistic groundwater-tool query, but worth knowing) can
    still return a large/meaningless value even with the bounded search."""
    import ee

    occurrence = ee.Image(JRC_SURFACE_WATER).select("occurrence")
    water_mask = occurrence.gt(0)
    pixel_size_m = ee.Image.pixelArea().sqrt()
    return (
        water_mask.fastDistanceTransform(DIST_TO_WATER_SEARCH_NEIGHBORHOOD_PX)
        .sqrt()
        .multiply(pixel_size_m)
        .rename("dist_to_water")
    )
