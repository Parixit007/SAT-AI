"""Pydantic request/response shapes for the API layer. Internal orchestrator dataclasses
(orchestrator/tool_registry.py, orchestrator/execution_trace.py) are separate and get converted
into these at the API boundary -- keeps the orchestrator importable/testable without FastAPI."""

from typing import Any, Optional

from pydantic import BaseModel


class GeoMetadataOut(BaseModel):
    source: str  # "geotiff" | "exif" | "map_capture"
    center_lat: float
    center_lon: float
    bounds_wgs84: Optional[tuple[float, float, float, float]] = None
    crs: Optional[str] = None
    band_count: Optional[int] = None
    acquisition_datetime: Optional[str] = None


class UploadedImageInfo(BaseModel):
    filename: str
    stored_path: str
    format: str
    width: int
    height: int
    modality_guess: str  # "optical" | "sar" | "unknown"
    geo: Optional[GeoMetadataOut] = None


class UploadResponse(BaseModel):
    input_id: str
    images: list[UploadedImageInfo]
    warnings: list[str] = []
    errors: list[str] = []  # files that failed validation and were NOT saved (e.g. bad format)


class LocationIn(BaseModel):
    lat: float
    lon: float


class CaptureRequest(BaseModel):
    """A rectangular area picked on the map (two opposite corners) -- min/max rather than the raw
    corner points since the frontend doesn't guarantee any particular click order."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class QueryRequest(BaseModel):
    input_id: Optional[str] = None  # omit for a location-only query (no uploaded image needed)
    query_text: str
    location: Optional[LocationIn] = None  # falls back to an uploaded image's own geo metadata if omitted


class ToolUsage(BaseModel):
    name: str
    params: dict[str, Any]
    checkpoint_id: Optional[str] = None


class ExecutionTraceOut(BaseModel):
    selected_task: str
    tools_used: list[ToolUsage]
    input_summary: str
    confidence: float
    confidence_bucket: str
    warnings: list[str]
    timestamp: str


class QueryResponse(BaseModel):
    query_id: str
    answer_text: str
    confidence: float
    confidence_bucket: str
    evidence_image_urls: list[str]
    execution_trace: ExecutionTraceOut
