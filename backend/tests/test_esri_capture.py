"""Tests for compute_export_size() -- pure Python, no network involved (see esri_capture.py's own
docstring for why this is split from fetch_satellite_image(), the actual HTTP call) -- plus
png_to_geotiff(), which also needs no network (just PNG bytes to wrap in georeferencing)."""

import io

import numpy as np
import pytest
from PIL import Image

from app.gis.esri_capture import MAX_EXPORT_DIMENSION_PX, MIN_EXPORT_DIMENSION_PX, compute_export_size, png_to_geotiff
from app.orchestrator.geo_metadata import extract_geotiff_metadata


def test_square_selection_at_the_equator_is_square():
    # At the equator, cos(0) = 1, so equal lat/lon spans really are square in real-world distance.
    width, height = compute_export_size(0.0, 0.0, 0.1, 0.1)
    assert width == height == MAX_EXPORT_DIMENSION_PX


def test_wide_selection_is_not_squashed_into_a_square():
    # Much wider in longitude than latitude -- the longer side (width) should hit the cap, the
    # shorter side (height) should come out well below it, not also be forced to the cap.
    width, height = compute_export_size(0.0, 0.0, 0.1, 1.0)
    assert width == MAX_EXPORT_DIMENSION_PX
    assert height < width


def test_tall_selection_is_not_squashed_into_a_square():
    width, height = compute_export_size(0.0, 0.0, 1.0, 0.1)
    assert height == MAX_EXPORT_DIMENSION_PX
    assert width < height


def test_high_latitude_selection_accounts_for_longitude_convergence():
    # Same raw lat/lon span (0.2 x 0.2 degrees) but centered at 70N, where a degree of longitude
    # covers much less real ground distance than a degree of latitude -- the real-world shape is
    # wide-and-short (lat span dominates), which should show up as height > width, unlike the
    # equator case above where an equal raw span comes out square.
    width, height = compute_export_size(69.9, 10.0, 70.1, 10.2)
    assert height > width


def test_degenerate_selection_does_not_crash_or_divide_by_zero():
    # Both points effectively identical (e.g. an accidental double-click on the same spot).
    width, height = compute_export_size(12.9, 77.6, 12.9, 77.6)
    assert width > 0 and height > 0


def test_output_dimensions_stay_within_configured_bounds():
    for min_lat, min_lon, max_lat, max_lon in [(0, 0, 0.001, 50), (0, 0, 50, 0.001), (10, 10, 10.5, 10.5)]:
        width, height = compute_export_size(min_lat, min_lon, max_lat, max_lon)
        assert MIN_EXPORT_DIMENSION_PX <= width <= MAX_EXPORT_DIMENSION_PX
        assert MIN_EXPORT_DIMENSION_PX <= height <= MAX_EXPORT_DIMENSION_PX


def test_respects_a_custom_max_dimension():
    width, height = compute_export_size(0.0, 0.0, 0.1, 0.1, max_dimension_px=512)
    assert width == height == 512


# --- png_to_geotiff: wraps flat PNG bytes in real georeferencing, no network involved -----------


def _fake_png_bytes(width: int = 40, height: int = 30) -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_png_to_geotiff_roundtrips_through_the_real_geo_extractor(tmp_path):
    """This is the actual regression this conversion exists to fix: a plain exported PNG has no
    embedded CRS, so extract_geo_metadata() -- the same function every query re-runs against the
    stored file from scratch -- found nothing. A real GeoTIFF should come back with the exact
    requested bounds and band_count=3 (-> "optical"), through the same code path a real
    georeferenced upload takes, not a special case."""
    min_lat, min_lon, max_lat, max_lon = 12.9, 77.5, 13.0, 77.6
    tiff_bytes = png_to_geotiff(_fake_png_bytes(), min_lat, min_lon, max_lat, max_lon)

    path = tmp_path / "captured.tif"
    path.write_bytes(tiff_bytes)

    geo = extract_geotiff_metadata(path)
    assert geo is not None
    assert geo.band_count == 3
    assert geo.center_lat == pytest.approx((min_lat + max_lat) / 2, abs=1e-6)
    assert geo.center_lon == pytest.approx((min_lon + max_lon) / 2, abs=1e-6)
    assert geo.bounds_wgs84 == pytest.approx((min_lon, min_lat, max_lon, max_lat), abs=1e-6)


def test_png_to_geotiff_preserves_pixel_dimensions():
    tiff_bytes = png_to_geotiff(_fake_png_bytes(width=80, height=50), 0.0, 0.0, 0.1, 0.1)

    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(tiff_bytes) as memfile, memfile.open() as src:
        assert src.width == 80
        assert src.height == 50
        assert src.count == 3
        assert src.crs == rasterio.crs.CRS.from_epsg(4326)
