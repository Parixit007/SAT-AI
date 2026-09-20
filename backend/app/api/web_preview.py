"""Browser-displayable renditions of uploaded images.

Browsers can't render TIFF, and TIFF is exactly what a georeferenced upload -- or a map capture,
which is written as a GeoTIFF on purpose (see gis/esri_capture.py) -- is. The UI overlays detector
boxes on the *source* image (boxes are in that image's own pixel space), so it needs a version of it
an <img> tag can show. The preview keeps the source's exact pixel dimensions so the boxes still line
up; it is written once, next to the upload, and reused."""

import os
from pathlib import Path

BROWSER_SAFE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _display_array(path: Path):
    import numpy as np

    try:
        import rasterio

        with rasterio.open(path) as src:
            bands = src.read(indexes=list(range(1, min(src.count, 3) + 1)))
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))

    if bands.shape[0] == 1:  # e.g. a single-band SAR image
        bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] == 2:
        bands = np.concatenate([bands, bands[:1]], axis=0)
    if bands.dtype != np.uint8:
        stretched = []
        for band in bands.astype("float64"):
            lo, hi = np.nanpercentile(band, [2, 98])
            stretched.append(np.clip((band - lo) / max(hi - lo, 1e-9), 0, 1) * 255)
        bands = np.stack(stretched).astype(np.uint8)
    return np.moveaxis(bands, 0, -1)


def web_preview_path(path: Path) -> Path:
    """`path` itself if a browser can already show it, else a same-size PNG rendition of it. Falls
    back to `path` if the rendition can't be made -- the caller's image then simply fails to load,
    as it would have without this."""
    if path.suffix.lower() in BROWSER_SAFE_SUFFIXES:
        return path
    preview = path.with_name(f"{path.stem}.preview.png")
    try:
        if preview.exists() and preview.stat().st_mtime >= path.stat().st_mtime:
            return preview
        from PIL import Image

        tmp = preview.with_name(f"{preview.name}.{os.getpid()}.tmp")
        Image.fromarray(_display_array(path)).save(tmp, format="PNG")
        os.replace(tmp, preview)
        return preview
    except Exception:
        return path
