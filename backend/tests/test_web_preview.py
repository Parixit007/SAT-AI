import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from app.api.web_preview import web_preview_path


def test_a_browser_safe_image_is_used_as_is(sample_image):
    assert web_preview_path(sample_image) == sample_image


def test_a_geotiff_gets_a_same_size_png_beside_it(georeferenced_tif):
    preview = web_preview_path(georeferenced_tif)
    assert preview.suffix == ".png" and preview.parent == georeferenced_tif.parent
    assert Image.open(preview).size == (50, 50)  # boxes are in the source's pixel space, so the size must not change
    assert web_preview_path(georeferenced_tif) == preview  # reused, not rewritten


def test_a_single_band_float_image_is_stretched_into_a_visible_rgb(tmp_path):
    path = tmp_path / "sar.tif"
    data = np.linspace(-30.0, 0.0, 40 * 40, dtype="float32").reshape(1, 40, 40)
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, 40, 40)) as dst:
        dst.write(data)
    rendered = np.asarray(Image.open(web_preview_path(path)))
    assert rendered.shape == (40, 40, 3)
    assert rendered.min() < 30 and rendered.max() > 225  # percentile-stretched to the full 8-bit range


def test_an_unreadable_file_falls_back_to_the_original_path(tmp_path):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not a tiff")
    assert web_preview_path(bad) == bad
