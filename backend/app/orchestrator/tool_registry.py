"""The registry of specialist tools the orchestrator can route queries to. Each ToolSpec pairs a
JSON-schema description (handed to the LLM for function-calling) with a plain Python handler
(what actually executes -- the LLM never touches model internals or file paths directly)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class LatLon:
    lat: float
    lon: float


@dataclass
class QueryInput:
    """Everything a tool handler might need about what the user gave it. Images and location are
    independent and both optional at this level -- a given ToolSpec declares which one(s) it
    actually needs via `uses_images`/`requires_location`, checked in `is_compatible()`. `location`
    can come from an explicit request field or be inferred from an uploaded image's own
    georeferencing (see orchestrator/geo_metadata.py) -- callers building this don't need to care
    which source it came from."""

    images: list[Path] = field(default_factory=list)
    location: Optional[LatLon] = None


@dataclass
class ToolResult:
    tool_name: str
    text_summary: str
    structured_data: dict[str, Any]
    evidence_image_path: Optional[Path]
    confidence: float  # in [0, 1]


@dataclass
class ToolSpec:
    name: str
    description: str  # sent to the LLM verbatim -- write it for the model, not for humans
    parameters_schema: dict[str, Any]  # JSON schema for the LLM's function-call arguments
    min_images: int
    max_images: int
    compatible_modalities: list[str]  # subset of {"optical", "sar", "unknown"}; "unknown" always allowed
    handler: Callable[[QueryInput, dict[str, Any]], ToolResult]
    checkpoint_id: Optional[str] = None  # e.g. "dior_rsvg_finetuned.pth" -- surfaced in the execution trace
    uses_images: bool = True  # False for location-only tools (e.g. GEE-backed) -- skips the image-count check entirely
    requires_location: bool = False  # True for tools that need a lat/lon (explicit or from image geo metadata)
    # For a tool that needs one image of EACH of two specific modalities (e.g. optical+SAR fusion)
    # rather than any mix of its compatible_modalities -- e.g. two optical images would otherwise
    # pass compatible_modalities=["optical","sar"] even though the tool needs exactly one of each.
    # None (the default) means no such constraint -- every existing tool leaves this unset.
    required_modality_pair: Optional[tuple[str, str]] = None

    def is_compatible(self, query_input: QueryInput, modalities: list[str]) -> Optional[str]:
        """Returns None if compatible, else a human-readable reason it isn't."""
        if self.uses_images:
            num_images = len(query_input.images)
            if not (self.min_images <= num_images <= self.max_images):
                return (
                    f"{self.name} needs between {self.min_images} and {self.max_images} image(s), "
                    f"got {num_images}."
                )
            for m in modalities:
                if m != "unknown" and m not in self.compatible_modalities:
                    return f"{self.name} does not support modality '{m}' (supports {self.compatible_modalities})."
            if self.required_modality_pair and set(modalities) != set(self.required_modality_pair):
                a, b = self.required_modality_pair
                return f"{self.name} needs exactly one {a} and one {b} image, got modalities {modalities}."
        if self.requires_location and query_input.location is None:
            return f"{self.name} needs a location (pass one explicitly, or upload a georeferenced image)."
        return None


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())
