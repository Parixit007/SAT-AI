import shutil
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOADS_DIR
from app.gis.esri_capture import fetch_satellite_image, png_to_geotiff
from app.orchestrator.input_validation import validate_images
from app.schemas.models import CaptureRequest, GeoMetadataOut, UploadedImageInfo, UploadResponse
from app.store import save_input

router = APIRouter()


def _validate_save_and_respond(input_id: str, saved_paths: list[Path]) -> UploadResponse:
    """Shared tail for both /upload and /capture: validate, persist only what's valid, build the
    response. Identical either way once a file exists on disk -- /capture's file just came from
    Esri instead of a form upload."""
    validation = validate_images(saved_paths)
    # Persist only the images that actually validated -- one bad file (e.g. an unsupported format)
    # must not make every future query against this input_id fail input validation, when the other
    # files were fine.
    save_input(input_id, [img.path for img in validation.images])

    images = [
        UploadedImageInfo(
            filename=img.path.name,
            stored_path=str(img.path),
            format=img.format,
            width=img.width,
            height=img.height,
            modality_guess=img.modality_guess,
            geo=GeoMetadataOut(**vars(img.geo)) if img.geo else None,
        )
        for img in validation.images
    ]
    return UploadResponse(input_id=input_id, images=images, warnings=validation.warnings, errors=validation.errors)


@router.post("/upload", response_model=UploadResponse)
async def upload_images(files: list[UploadFile] = File(...)) -> UploadResponse:
    input_id = uuid.uuid4().hex
    dest_dir = UPLOADS_DIR / input_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # `f.filename` is client-supplied and untrusted: `Path(...).name` strips both `../` traversal
    # and an absolute path (which would otherwise make `dest_dir / f.filename` discard `dest_dir`
    # entirely, per pathlib's join semantics). The `{i}_` prefix keeps two same-named files in one
    # batch -- e.g. a co-registered optical+SAR pair both called "export.tif" -- from clobbering
    # each other on disk.
    saved_paths = []
    for i, f in enumerate(files):
        safe_name = Path(f.filename).name if f.filename else ""
        dest = dest_dir / f"{i}_{safe_name or 'upload'}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    return _validate_save_and_respond(input_id, saved_paths)


@router.post("/capture", response_model=UploadResponse)
async def capture_area(req: CaptureRequest) -> UploadResponse:
    """Fetches real satellite imagery for a map-selected rectangle from Esri (see
    gis/esri_capture.py) and feeds it through the same save/validate/store path as a real file
    upload -- the frontend treats the response identically either way, so a captured area is
    immediately usable with every image-based tool. Written out as a real GeoTIFF, not a plain
    PNG: `validate_images()` then derives modality ("optical", from the embedded 3-band count) and
    geo bounds organically from the file, the same already-tested path a real georeferenced
    upload takes -- no special-casing needed here, and (unlike a plain PNG, which was tried first)
    it stays correct on re-validation at query time too, not just in this response."""
    if req.min_lat >= req.max_lat or req.min_lon >= req.max_lon:
        raise HTTPException(status_code=400, detail="Selected area must have positive width and height.")

    try:
        png_bytes = fetch_satellite_image(req.min_lat, req.min_lon, req.max_lat, req.max_lon)
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Could not fetch satellite imagery: {exc}") from exc
    geotiff_bytes = png_to_geotiff(png_bytes, req.min_lat, req.min_lon, req.max_lat, req.max_lon)

    input_id = uuid.uuid4().hex
    dest_dir = UPLOADS_DIR / input_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "0_captured_area.tif"
    dest.write_bytes(geotiff_bytes)

    return _validate_save_and_respond(input_id, [dest])
