from app.orchestrator.geo_metadata import extract_exif_gps, extract_geotiff_metadata, extract_geo_metadata


def test_georeferenced_tif_extracts_reprojected_location(georeferenced_tif):
    meta = extract_geotiff_metadata(georeferenced_tif)

    assert meta is not None
    assert meta.source == "geotiff"
    assert meta.band_count == 3
    assert meta.crs == "EPSG:32643"
    # UTM 43N covers ~India -- sanity-check the reprojection landed in a plausible range
    assert 10 < meta.center_lat < 15
    assert 73 < meta.center_lon < 78
    assert meta.acquisition_datetime == "2026:06:15 10:30:00"


def test_non_georeferenced_tif_returns_none(plain_tif):
    assert extract_geotiff_metadata(plain_tif) is None


def test_exif_gps_extraction(exif_gps_jpeg):
    meta = extract_exif_gps(exif_gps_jpeg)

    assert meta is not None
    assert meta.source == "exif"
    # S/E hemisphere -> negative lat, positive lon (this is Sydney's approximate coordinates)
    assert meta.center_lat < 0
    assert meta.center_lon > 0
    assert -34 < meta.center_lat < -33
    assert 151 < meta.center_lon < 152


def test_plain_jpeg_no_exif_returns_none(sample_image):
    assert extract_exif_gps(sample_image) is None


def test_extract_geo_metadata_dispatches_by_format(georeferenced_tif, exif_gps_jpeg, plain_tif):
    assert extract_geo_metadata(georeferenced_tif, "TIFF") is not None
    assert extract_geo_metadata(exif_gps_jpeg, "JPEG") is not None
    assert extract_geo_metadata(plain_tif, "TIFF") is None


def test_tiff_falls_back_to_exif_gps_when_not_georeferenced(exif_gps_tif):
    """Regression test: a TIFF with no embedded CRS/transform (e.g. a drone photo saved as .tiff)
    can still carry EXIF GPS tags the same way a JPEG would -- extract_geo_metadata() used to try
    GeoTIFF extraction only for "TIFF" and never fall back, silently dropping this location."""
    meta = extract_geo_metadata(exif_gps_tif, "TIFF")

    assert meta is not None
    assert meta.source == "exif"
    assert -34 < meta.center_lat < -33
    assert 151 < meta.center_lon < 152


def test_missing_file_does_not_raise(tmp_path):
    missing = tmp_path / "does_not_exist.tif"
    assert extract_geotiff_metadata(missing) is None
    assert extract_exif_gps(missing) is None
