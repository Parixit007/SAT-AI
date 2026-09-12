"""Deterministic, no-LLM-call input checks that run before task classification. Cheap format/count
validation, plus real georeferencing/acquisition metadata extraction (see geo_metadata.py) instead
of discarding it -- an uploaded GeoTIFF's embedded CRS+bounds (or a JPEG's EXIF GPS tags) can supply
a query's location directly, and GeoTIFF band count gives a real (if imperfect) modality signal."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.orchestrator.geo_metadata import GeoMetadata, extract_geo_metadata

SUPPORTED_FORMATS = {"PNG", "JPEG", "TIFF"}
DIMENSION_MISMATCH_TOLERANCE = 0.02  # fractional difference allowed between paired images
SAR_BAND_COUNT_MAX = 2  # SAR products are typically single- or dual-polarization; optical/MSI is 3+


@dataclass
class ImageMeta:
    path: Path
    format: str
    width: int
    height: int
    modality_guess: str  # "optical" | "sar" | "unknown"
    geo: Optional[GeoMetadata] = None


@dataclass
class ValidationResult:
    ok: bool
    images: list[ImageMeta]
    warnings: list[str]
    errors: list[str]

    @property
    def modalities(self) -> list[str]:
        return [img.modality_guess for img in self.images]

    def summary_text(self) -> str:
        parts = [f"{len(self.images)} image(s)"]
        for img in self.images:
            line = f"- {img.path.name}: {img.format} {img.width}x{img.height}, modality={img.modality_guess}"
            if img.geo:
                line += f", location=({img.geo.center_lat:.5f}, {img.geo.center_lon:.5f})"
                if img.geo.acquisition_datetime:
                    line += f", acquired={img.geo.acquisition_datetime}"
            parts.append(line)
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(parts)


def validate_images(paths: list[Path]) -> ValidationResult:
    """An empty `paths` list is valid on its own (`ok=True`, zero images) -- location-only queries
    (e.g. the groundwater tool) never touch images at all. Whether "zero images and no location
    either" is actually an error is a decision `controller.handle_query` makes, since only it knows
    about both images and location; this function only ever judges the images it's given."""
    from PIL import Image

    warnings: list[str] = []
    errors: list[str] = []
    metas: list[ImageMeta] = []

    for p in paths:
        if not p.exists():
            errors.append(f"File not found: {p}")
            continue
        try:
            with Image.open(p) as im:
                fmt = (im.format or "").upper()
                if fmt not in SUPPORTED_FORMATS:
                    errors.append(f"{p.name}: unsupported format '{fmt}' (supported: {sorted(SUPPORTED_FORMATS)})")
                    continue
                width, height = im.width, im.height
        except Exception as exc:
            errors.append(f"{p.name}: could not open image ({exc})")
            continue

        geo = extract_geo_metadata(p, fmt)
        modality_guess = "unknown"
        if geo and geo.band_count is not None:
            modality_guess = "sar" if geo.band_count <= SAR_BAND_COUNT_MAX else "optical"

        metas.append(ImageMeta(path=p, format=fmt, width=width, height=height, modality_guess=modality_guess, geo=geo))

    if len(metas) == 2:
        w0, h0 = metas[0].width, metas[0].height
        w1, h1 = metas[1].width, metas[1].height
        dw = abs(w0 - w1) / max(w0, w1)
        dh = abs(h0 - h1) / max(h0, h1)
        if dw > DIMENSION_MISMATCH_TOLERANCE or dh > DIMENSION_MISMATCH_TOLERANCE:
            warnings.append(
                f"Paired images have mismatched dimensions ({w0}x{h0} vs {w1}x{h1}) -- "
                "results for pair-based tasks may be unreliable."
            )

    return ValidationResult(ok=len(errors) == 0, images=metas, warnings=warnings, errors=errors)
