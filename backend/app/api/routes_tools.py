from fastapi import APIRouter

from app.schemas.models import ToolSpecOut
from app.specialists import DEFAULT_REGISTRY

router = APIRouter()


@router.get("/tools", response_model=list[ToolSpecOut])
async def list_tools() -> list[ToolSpecOut]:
    """Registry metadata for every registered specialist -- drives the frontend's capabilities
    gallery and its manual tool-override picker from one live source (see CLAUDE.md / the plan),
    so neither can drift from what the backend actually runs."""
    return [
        ToolSpecOut(
            name=spec.name,
            description=spec.description,
            parameters_schema=spec.parameters_schema,
            min_images=spec.min_images,
            max_images=spec.max_images,
            compatible_modalities=spec.compatible_modalities,
            uses_images=spec.uses_images,
            requires_location=spec.requires_location,
            checkpoint_id=spec.checkpoint_id,
        )
        for spec in DEFAULT_REGISTRY.list_specs()
    ]
