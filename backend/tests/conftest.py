from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.orchestrator.llm_providers.base import LLMProvider, ToolCall


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """A synthetic RGB PNG -- good enough to exercise shapes/plumbing, not model accuracy (there's
    no real satellite imagery bundled yet; see CLAUDE.md's data-acquisition follow-up)."""
    path = tmp_path / "sample.png"
    arr = np.random.default_rng(0).integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def georeferenced_tif(tmp_path: Path) -> Path:
    """A synthetic 3-band GeoTIFF with a real (UTM 43N, ~India) CRS + transform."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "geo.tif"
    transform = from_bounds(500000, 1430000, 510000, 1440000, 50, 50)
    arr = np.random.default_rng(0).integers(0, 255, size=(3, 50, 50), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=50, width=50, count=3,
        dtype="uint8", crs="EPSG:32643", transform=transform,
    ) as dst:
        dst.write(arr)
        dst.update_tags(TIFFTAG_DATETIME="2026:06:15 10:30:00")
    return path


@pytest.fixture
def sar_geotiff(tmp_path: Path) -> Path:
    """A synthetic single-band GeoTIFF -- input_validation.py infers 'sar' from band count <= 2."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "sar.tif"
    transform = from_bounds(500000, 1430000, 510000, 1440000, 50, 50)
    arr = np.random.default_rng(4).integers(90, 140, size=(1, 50, 50), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=50, width=50, count=1,
        dtype="uint8", crs="EPSG:32643", transform=transform,
    ) as dst:
        dst.write(arr)
    return path


@pytest.fixture
def optical_sar_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A co-registered optical (3-band GeoTIFF) + SAR (1-band GeoTIFF) pair, with a known dark
    (water-like) block and a known bright (built-up-like) block injected into the SAR band, so
    fusion tests can assert against known regions rather than just 'something was detected'."""
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(500000, 1430000, 510000, 1440000, 100, 100)

    optical_path = tmp_path / "pair_optical.tif"
    optical_arr = np.random.default_rng(5).integers(0, 255, size=(3, 100, 100), dtype=np.uint8)
    with rasterio.open(
        optical_path, "w", driver="GTiff", height=100, width=100, count=3,
        dtype="uint8", crs="EPSG:32643", transform=transform,
    ) as dst:
        dst.write(optical_arr)

    sar_path = tmp_path / "pair_sar.tif"
    rng = np.random.default_rng(6)
    sar_arr = rng.integers(90, 140, size=(1, 100, 100), dtype=np.uint8)
    sar_arr[0, 10:40, 10:40] = rng.integers(0, 15, size=(30, 30))       # dark block -> water
    sar_arr[0, 60:90, 60:90] = rng.integers(230, 256, size=(30, 30))    # bright block -> built-up
    with rasterio.open(
        sar_path, "w", driver="GTiff", height=100, width=100, count=1,
        dtype="uint8", crs="EPSG:32643", transform=transform,
    ) as dst:
        dst.write(sar_arr)

    return optical_path, sar_path


@pytest.fixture
def plain_tif(tmp_path: Path) -> Path:
    """A TIFF with no georeferencing at all."""
    path = tmp_path / "plain.tif"
    arr = np.random.default_rng(1).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def change_pair_images(tmp_path: Path) -> tuple[Path, Path, list[int]]:
    """A bi-temporal pair with a KNOWN injected difference -- a solid block pasted into one corner
    of an otherwise-identical low-contrast background, so change-detection tests can assert against
    an exact expected bbox rather than just "some change was found somewhere"."""
    rng = np.random.default_rng(3)
    base = rng.integers(80, 120, size=(128, 128, 3), dtype=np.uint8)  # muted, low-contrast background
    before = base.copy()
    after = base.copy()
    after[20:60, 70:110] = [240, 30, 30]  # rows 20-60, cols 70-110 -- a bright, high-contrast block

    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    Image.fromarray(before).save(before_path)
    Image.fromarray(after).save(after_path)
    return before_path, after_path, [70, 20, 110, 60]  # [x1, y1, x2, y2]


@pytest.fixture
def exif_gps_jpeg(tmp_path: Path) -> Path:
    """A JPEG with EXIF GPS tags set to Sydney's approximate coordinates (S/E hemisphere)."""
    from PIL.ExifTags import Base, GPS
    from PIL.TiffImagePlugin import IFDRational

    def rat(n, d=1):
        return IFDRational(n, d)

    path = tmp_path / "geo.jpg"
    arr = np.random.default_rng(2).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[Base.GPSInfo.value] = {
        GPS.GPSLatitudeRef: "S", GPS.GPSLatitude: (rat(33), rat(52), rat(4, 10)),
        GPS.GPSLongitudeRef: "E", GPS.GPSLongitude: (rat(151), rat(12), rat(30, 10)),
    }
    img.save(path, exif=exif)
    return path


@pytest.fixture
def exif_gps_tif(tmp_path: Path) -> Path:
    """A plain TIFF (no embedded CRS/GeoTIFF transform) that still carries EXIF GPS tags -- e.g. a
    drone photo saved as .tiff rather than .jpg. Same coordinates as exif_gps_jpeg, just format
    swapped, to test extract_geo_metadata()'s TIFF-then-EXIF-fallback path specifically.

    Built by round-tripping through a JPEG first: Pillow's TIFF writer errors
    (`AttributeError: 'Exif' object has no attribute 'fp'`) when asked to serialize a freshly
    constructed `Image.Exif()` with a nested GPS IFD directly -- it expects `Exif.fp` to exist,
    which is only true for an Exif object attached to an already-opened file. Saving as JPEG first
    (Pillow's Exif writer handles that case fine, per exif_gps_jpeg above), then reopening and
    passing its already-serialized raw `exif` bytes into the TIFF save, sidesteps that code path
    entirely. Verified by reading it back the same way extract_exif_gps() does
    (`Image.open(...).getexif().get_ifd(0x8825)`) before trusting this as a fixture."""
    from PIL.ExifTags import Base, GPS
    from PIL.TiffImagePlugin import IFDRational

    def rat(n, d=1):
        return IFDRational(n, d)

    jpeg_path = tmp_path / "_geo_no_crs_source.jpg"
    path = tmp_path / "geo_no_crs.tif"
    arr = np.random.default_rng(2).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[Base.GPSInfo.value] = {
        GPS.GPSLatitudeRef: "S", GPS.GPSLatitude: (rat(33), rat(52), rat(4, 10)),
        GPS.GPSLongitudeRef: "E", GPS.GPSLongitude: (rat(151), rat(12), rat(30, 10)),
    }
    img.save(jpeg_path, exif=exif)

    reopened = Image.open(jpeg_path)
    reopened.save(path, format="TIFF", exif=reopened.info.get("exif"))
    return path


class StubProvider(LLMProvider):
    """Returns a fixed, pre-scripted list of tool calls -- lets orchestrator tests assert routing
    behavior without spending real LLM API quota."""

    def __init__(self, calls: list[ToolCall]):
        self._calls = calls

    def select_tools(self, query, tool_specs, input_summary):
        return self._calls
