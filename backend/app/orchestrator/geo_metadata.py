"""Extracts real georeferencing/acquisition metadata from uploaded images instead of discarding
it -- GeoTIFF carries embedded CRS + spatial bounds (via `rasterio`), and JPEG/PNG sometimes carry
EXIF GPS tags (drone/aerial exports). Both feed `GeoMetadata.center_lat/lon`, which is exactly the
coordinate the map-based location features (GEE hazard layers) need -- an uploaded image can supply
its own location instead of requiring a separate manual map pick.

Every extractor is defensive: a file with no embedded geo info, a corrupt tag, or a missing/failed
CRS transform returns None rather than raising -- one image's missing/bad metadata must never break
validation for the whole upload."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeoMetadata:
    source: str  # "geotiff" | "exif"
    center_lat: float
    center_lon: float
    bounds_wgs84: Optional[tuple[float, float, float, float]] = None  # (west, south, east, north)
    crs: Optional[str] = None
    band_count: Optional[int] = None
    acquisition_datetime: Optional[str] = None


def extract_geotiff_metadata(path: Path) -> Optional[GeoMetadata]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        return None

    try:
        with rasterio.open(path) as src:
            if src.crs is None or src.transform.is_identity:
                return None  # opened fine, but not actually georeferenced

            bounds = src.bounds
            if src.crs.to_epsg() != 4326:
                west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *bounds)
            else:
                west, south, east, north = bounds

            tags = src.tags()
            acquisition = tags.get("TIFFTAG_DATETIME") or tags.get("ACQUISITIONDATETIME")

            return GeoMetadata(
                source="geotiff",
                center_lat=(south + north) / 2,
                center_lon=(west + east) / 2,
                bounds_wgs84=(west, south, east, north),
                crs=str(src.crs),
                band_count=src.count,
                acquisition_datetime=acquisition,
            )
    except Exception:
        return None  # corrupt/unreadable geo tags -- degrade to "no location", don't fail the upload


def _dms_to_decimal(dms: tuple[float, float, float], ref: str) -> float:
    degrees, minutes, seconds = dms
    decimal = degrees + minutes / 60 + seconds / 3600
    return -decimal if ref in ("S", "W") else decimal


def extract_exif_gps(path: Path) -> Optional[GeoMetadata]:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(0x8825)  # GPSInfo tag
            if not gps_ifd:
                return None

            lat_dms, lat_ref = gps_ifd.get(2), gps_ifd.get(1)
            lon_dms, lon_ref = gps_ifd.get(4), gps_ifd.get(3)
            if not (lat_dms and lat_ref and lon_dms and lon_ref):
                return None

            lat = _dms_to_decimal(tuple(float(v) for v in lat_dms), lat_ref)
            lon = _dms_to_decimal(tuple(float(v) for v in lon_dms), lon_ref)

            return GeoMetadata(source="exif", center_lat=lat, center_lon=lon)
    except Exception:
        return None


def extract_geo_metadata(path: Path, fmt: str) -> Optional[GeoMetadata]:
    """Dispatches to the right extractor for the given (already-detected) image format. A TIFF
    tries GeoTIFF extraction first, then falls back to EXIF GPS -- "TIFF" doesn't imply
    "georeferenced": a plain TIFF with no embedded CRS/transform (e.g. a drone photo saved as
    .tiff rather than .jpg) can still carry EXIF GPS tags the same way a JPEG would, and skipping
    the fallback here used to silently drop that location entirely."""
    if fmt == "TIFF":
        return extract_geotiff_metadata(path) or extract_exif_gps(path)
    return extract_exif_gps(path)
