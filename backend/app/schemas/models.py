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


class ForcedToolCall(BaseModel):
    """One entry in QueryRequest.forced_tools -- lets a caller (the UI's manual 'Advanced' picker)
    pick exactly which tool(s) run, bypassing the LLM's own selection for that query."""

    tool_name: str
    arguments: dict[str, Any] = {}


class QueryRequest(BaseModel):
    input_id: Optional[str] = None  # omit for a location-only query (no uploaded image needed)
    query_text: str
    location: Optional[LocationIn] = None  # falls back to an uploaded image's own geo metadata if omitted
    # None (default) = automatic LLM tool selection, unchanged. A non-None list bypasses
    # select_tools() entirely and runs exactly these tools -- see orchestrator/controller.py.
    forced_tools: Optional[list[ForcedToolCall]] = None


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


class ToolResultOut(BaseModel):
    """One specialist's full output -- the rich, tool-specific structured_data every adapter
    already computes (backend/app/specialists/*_adapter.py) but that used to be discarded before
    reaching the API response. Additive alongside the existing flat answer_text/evidence_image_urls
    fields on QueryResponse, not a replacement for them."""

    tool_name: str
    text_summary: str
    structured_data: dict[str, Any]
    confidence: float
    evidence_image_url: Optional[str] = None
    source_image_url: Optional[str] = None  # the original uploaded image this result is about, if any


class QueryResponse(BaseModel):
    query_id: str
    answer_text: str
    confidence: float
    confidence_bucket: str
    evidence_image_urls: list[str]
    tool_results: list[ToolResultOut] = []
    execution_trace: ExecutionTraceOut


class ToolSpecOut(BaseModel):
    """Registry metadata for one specialist, as served by GET /api/tools -- mirrors ToolSpec
    (orchestrator/tool_registry.py) minus its non-serializable `handler`. Drives both the
    frontend's capabilities gallery and its manual tool-override picker from one live source, so
    neither can drift from what the backend actually runs."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    min_images: int
    max_images: int
    compatible_modalities: list[str]
    uses_images: bool
    requires_location: bool
    checkpoint_id: Optional[str] = None
