from app.orchestrator.input_validation import validate_images


def test_paired_images_with_mismatched_dimensions_warn_but_stay_ok(tmp_path):
    import numpy as np
    from PIL import Image

    small = tmp_path / "small.png"
    big = tmp_path / "big.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(small)
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(big)

    result = validate_images([small, big])

    # A mismatch is a warning, not a hard error -- pair-based tools can still attempt the query.
    assert result.ok
    assert any("mismatched dimensions" in w for w in result.warnings)


def test_plain_image_has_unknown_modality_and_no_geo(sample_image):
    result = validate_images([sample_image])

    assert result.ok
    assert result.images[0].modality_guess == "unknown"
    assert result.images[0].geo is None
    assert "location=" not in result.summary_text()


def test_georeferenced_multiband_tif_infers_optical_and_location(georeferenced_tif):
    result = validate_images([georeferenced_tif])

    assert result.ok
    img = result.images[0]
    assert img.modality_guess == "optical"  # 3 bands -> above SAR_BAND_COUNT_MAX
    assert img.geo is not None
    assert "location=" in result.summary_text()


def test_single_band_geotiff_infers_sar(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "sar.tif"
    transform = from_bounds(500000, 1430000, 510000, 1440000, 50, 50)
    arr = np.random.default_rng(0).integers(0, 255, size=(1, 50, 50), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=50, width=50, count=1,
        dtype="uint8", crs="EPSG:32643", transform=transform,
    ) as dst:
        dst.write(arr)

    result = validate_images([path])

    assert result.ok
    assert result.images[0].modality_guess == "sar"
