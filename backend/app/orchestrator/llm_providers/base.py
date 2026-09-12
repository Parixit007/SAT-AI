"""The whole LLM-provider swap point. One method, one shape -- deliberately not a plugin registry
since this is a 2-provider (Gemini vs Groq) comparison, not a general extensibility surface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.orchestrator.tool_registry import ToolSpec


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]


class LLMProvider(ABC):
    @abstractmethod
    def select_tools(self, query: str, tool_specs: list[ToolSpec], input_summary: str) -> list[ToolCall]:
        """Given the user's query and a text summary of the validated input (image count, format,
        modality guesses -- see input_validation.py), return the ordered list of tool calls to run.
        Both providers receive the same text-only input_summary (not raw image bytes) so the
        Gemini-vs-Groq comparison is measuring tool-calling quality, not vision understanding."""
        raise NotImplementedError


def get_provider(name: str) -> LLMProvider:
    if name == "gemini":
        from app.orchestrator.llm_providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == "groq":
        from app.orchestrator.llm_providers.groq_provider import GroqProvider

        return GroqProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}', expected 'gemini' or 'groq'")
