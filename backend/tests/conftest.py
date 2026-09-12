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
def plain_tif(tmp_path: Path) -> Path:
    """A TIFF with no georeferencing at all."""
    path = tmp_path / "plain.tif"
    arr = np.random.default_rng(1).integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


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


class StubProvider(LLMProvider):
    """Returns a fixed, pre-scripted list of tool calls -- lets orchestrator tests assert routing
    behavior without spending real LLM API quota."""

    def __init__(self, calls: list[ToolCall]):
        self._calls = calls

    def select_tools(self, query, tool_specs, input_summary):
        return self._calls
