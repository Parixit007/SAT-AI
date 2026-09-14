"""Fetches real satellite imagery for a user-selected map area from Esri's free, keyless World
Imagery export service -- the same basemap already shown on the map (`MapPicker.tsx`), so a
captured image is visually exactly what the user selected. Deliberately exports ONLY the base
imagery layer, not the two reference overlays (roads/boundaries) also stacked on the map: those
are a UI aid for picking a location, and baking road lines/place labels into the pixels would
pollute what the vision tools (grounding, water segmentation, VQA, change detection) actually see.
Live-verified against a real ~2km-wide selection over New Delhi: a genuinely detailed ~900KB PNG
(buildings, roads, individual trees legible), not a downsampled or placeholder tile.

Same pure/impure split as gee/groundwater.py and gee/wildfire.py: `compute_export_size()` is pure
Python (unit-tested without network access), `fetch_satellite_image()` is the actual HTTP call.

Written out as a real GeoTIFF (not a plain PNG) -- found live, not guessed: a plain PNG has no
embedded CRS/band info, so `input_validation.py`'s modality/geo re-derivation (which every query
runs again from scratch against the stored file, not from the upload response) came back
`modality_guess="unknown"` at query time even though the /api/capture response itself said
"optical" -- a real inconsistency that would have silently broken pairing a captured image with a
real SAR upload for the fusion tool's exact-match modality check. A GeoTIFF with a real embedded
CRS/transform and 3 bands makes `extract_geotiff_metadata()` derive "optical" (band_count=3) and
the correct geo bounds organically, the same already-tested path a real georeferenced upload takes
-- consistent at both capture time and query time, no special-casing needed in the route handler."""

import math

ESRI_EXPORT_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

MAX_EXPORT_DIMENSION_PX = 1024  # longer side of the exported image -- live-verified this produces
# a genuinely detailed export for a city-block-to-few-km selection, not a guess.
MIN_EXPORT_DIMENSION_PX = 256  # floor for the shorter side, so an extreme (thin sliver) selection
# never rounds down to something too small to be a useful image.


def compute_export_size(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, max_dimension_px: int = MAX_EXPORT_DIMENSION_PX
) -> tuple[int, int]:
    """Picks a (width, height) in pixels proportional to the selection's real-world aspect ratio,
    capped at `max_dimension_px` on the longer side -- a wide/thin selection shouldn't be squashed
    into a square. Longitude degrees cover less real ground distance than latitude degrees away
    from the equator, so the longitude span is scaled by cos(mean latitude) before comparing the
    two spans -- otherwise a selection at high latitude would compute a visibly wrong aspect ratio."""
    lat_span = max(max_lat - min_lat, 1e-9)
    lon_span = max(max_lon - min_lon, 1e-9)
    mean_lat_rad = math.radians((min_lat + max_lat) / 2)
    real_width_ratio = lon_span * math.cos(mean_lat_rad)
    real_height_ratio = lat_span

    if real_width_ratio >= real_height_ratio:
        width = max_dimension_px
        height = max(MIN_EXPORT_DIMENSION_PX, round(max_dimension_px * real_height_ratio / real_width_ratio))
    else:
        height = max_dimension_px
        width = max(MIN_EXPORT_DIMENSION_PX, round(max_dimension_px * real_width_ratio / real_height_ratio))
    return width, height


def fetch_satellite_image(min_lat: float, min_lon: float, max_lat: float, max_lon: float, timeout: float = 30.0) -> bytes:
    """Raw PNG bytes from Esri -- raises `requests.RequestException` (incl. `HTTPError`) on
    failure, which the caller (the /api/capture route) turns into a clean HTTP error response, the
    same pattern every other external-service call in this codebase follows (GEE, the LLM
    providers). Pair with `png_to_geotiff()` below to get a file `validate_images()` will
    correctly recognize as georeferenced optical imagery."""
    import requests

    width, height = compute_export_size(min_lat, min_lon, max_lat, max_lon)
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{width},{height}",
        "format": "png",
        "transparent": "false",
        "f": "image",
    }
    resp = requests.get(ESRI_EXPORT_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def png_to_geotiff(png_bytes: bytes, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> bytes:
    """Wraps flat PNG pixel data in a real GeoTIFF (EPSG:4326, 3-band RGB) covering exactly the
    requested bounds -- no network access, just reprojection-free georeferencing math, since the
    PNG's own pixel grid already exactly spans this bbox by construction (that's what `bbox`/`size`
    in the Esri export request guaranteed)."""
    import io

    import numpy as np
    from PIL import Image
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    with Image.open(io.BytesIO(png_bytes)) as im:
        arr = np.array(im.convert("RGB"))  # (height, width, 3) -- drop any alpha, keep it plain RGB

    height, width = arr.shape[0], arr.shape[1]
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=3, dtype="uint8", crs="EPSG:4326", transform=transform
        ) as dst:
            for band in range(3):
                dst.write(arr[:, :, band], band + 1)
        return memfile.read()
