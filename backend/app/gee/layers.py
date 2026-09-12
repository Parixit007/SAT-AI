"""One small function per GEE dataset used by groundwater.py. Each returns a plain `ee.Image` --
assumes `client.ensure_initialized()` has already been called by the caller (groundwater.py does
this once, rather than every layer function checking redundantly).

Asset IDs below were verified against the live Earth Engine Data Catalog before writing this file
(not guessed) -- see the plan for sources. The TWI formula and the distance-transform-to-meters
conversion are standard textbook approaches, but -- like the rest of this package -- have NOT been
run against live Earth Engine yet (no service account configured in this environment); verify
against a couple of known locations once GEE is set up, per the plan's Verification section."""

SRTM = "USGS/SRTMGL1_003"
HYDROSHEDS_FLOW_ACCUMULATION = "WWF/HydroSHEDS/15ACC"
CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"
ESA_WORLDCOVER = "ESA/WorldCover/v200"
JRC_SURFACE_WATER = "JRC/GSW1_4/GlobalSurfaceWater"


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
