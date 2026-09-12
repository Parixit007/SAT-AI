"""Builds the default tool registry. Adding a new specialist in a later phase means writing one
adapter module (see grounding_adapter.py / water_segmentation_adapter.py for the pattern) and
registering its TOOL_SPEC here -- no orchestrator changes needed."""

from app.orchestrator.tool_registry import ToolRegistry
from app.specialists.grounding_adapter import TOOL_SPEC as GROUNDING_TOOL_SPEC
from app.specialists.groundwater_adapter import TOOL_SPEC as GROUNDWATER_TOOL_SPEC
from app.specialists.vqa_adapter import TOOL_SPEC as VQA_TOOL_SPEC
from app.specialists.water_segmentation_adapter import TOOL_SPEC as WATER_SEG_TOOL_SPEC


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(VQA_TOOL_SPEC)
    registry.register(GROUNDING_TOOL_SPEC)
    registry.register(WATER_SEG_TOOL_SPEC)
    registry.register(GROUNDWATER_TOOL_SPEC)
    return registry
