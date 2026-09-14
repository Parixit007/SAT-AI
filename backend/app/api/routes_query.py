import uuid

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.orchestrator.controller import handle_query
from app.orchestrator.input_validation import validate_images
from app.orchestrator.llm_providers.base import get_provider
from app.orchestrator.tool_registry import LatLon, QueryInput
from app.schemas.models import ExecutionTraceOut, QueryRequest, QueryResponse, ToolUsage
from app.specialists import build_default_registry
from app.store import get_input, save_query

router = APIRouter()

# Built once per process -- registering a tool is cheap/metadata-only (see specialists/__init__.py);
# the underlying models load lazily on first actual use.
_registry = build_default_registry()


def _resolve_location(payload: QueryRequest, image_paths: list) -> LatLon | None:
    """Explicit request location wins; otherwise fall back to the first uploaded image that has
    its own georeferencing (see orchestrator/geo_metadata.py) -- lets a georeferenced upload supply
    a location-based tool's coordinates with no separate map pick or manual entry needed."""
    if payload.location:
        return LatLon(lat=payload.location.lat, lon=payload.location.lon)
    for img in validate_images(image_paths).images:
        if img.geo:
            return LatLon(lat=img.geo.center_lat, lon=img.geo.center_lon)
    return None


def _run_query_blocking(payload: QueryRequest, image_paths: list, provider, registry):
    """Everything that does blocking I/O for one query: resolving the location (which re-opens
    each image via validate_images for the geo fallback above) and then handle_query itself
    (specialist model inference / GEE network calls). Both used to be split across the async/sync
    boundary -- only handle_query ran via run_in_threadpool, while _resolve_location's own
    PIL/rasterio work ran directly on the event loop despite being the same kind of blocking call
    threadpooling handle_query was meant to keep off it."""
    location = _resolve_location(payload, image_paths)
    query_input = QueryInput(images=image_paths, location=location)
    return handle_query(payload.query_text, query_input, provider, registry)


@router.post("/query", response_model=QueryResponse)
async def run_query(payload: QueryRequest) -> QueryResponse:
    image_paths = []
    if payload.input_id is not None:
        image_paths = get_input(payload.input_id)
        if image_paths is None:
            raise HTTPException(status_code=404, detail=f"Unknown input_id '{payload.input_id}'. Upload images first.")

    try:
        provider = get_provider(settings.llm_provider)
        result = await run_in_threadpool(_run_query_blocking, payload, image_paths, provider, _registry)
    except RuntimeError as exc:
        # Missing API key, or a provider SDK surfacing a config/auth problem -- report it as a
        # clean 503 rather than an opaque 500 (see backend/.env.example for setup).
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    query_id = uuid.uuid4().hex
    save_query(query_id, result)

    evidence_urls = [f"/evidence/{p.name}" for p in result.evidence_image_paths]
    trace = result.trace
    trace_out = ExecutionTraceOut(
        selected_task=trace.selected_task,
        tools_used=[ToolUsage(**t) for t in trace.tools_used],
        input_summary=trace.input_summary,
        confidence=trace.confidence,
        confidence_bucket=trace.confidence_bucket,
        warnings=trace.warnings,
        timestamp=trace.timestamp,
    )

    return QueryResponse(
        query_id=query_id,
        answer_text=result.answer_text,
        confidence=result.confidence,
        confidence_bucket=result.confidence_bucket,
        evidence_image_urls=evidence_urls,
        execution_trace=trace_out,
    )
