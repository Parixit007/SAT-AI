import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.config import UPLOADS_DIR
from app.orchestrator.input_validation import validate_images
from app.schemas.models import GeoMetadataOut, UploadedImageInfo, UploadResponse
from app.store import save_input

router = APIRouter()


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
