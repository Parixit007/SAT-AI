import shutil
import uuid

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

    saved_paths = []
    for f in files:
        dest = dest_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    save_input(input_id, saved_paths)
    validation = validate_images(saved_paths)

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
    return UploadResponse(input_id=input_id, images=images, warnings=validation.warnings + validation.errors)
