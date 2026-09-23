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


def format_routing_prompt(query: str, input_summary: str, history: str = "") -> str:
    """The one piece of per-call content shared verbatim by both providers, so a prompt change here
    can't accidentally diverge between them. `history` (see LLMProvider.select_tools) is blank on
    the first round; when present it's shown as what's already been found, so the model can build on
    it instead of re-deciding from scratch."""
    content = f"Query: {query}\n\nInput summary:\n{input_summary}"
    if history:
        content += (
            "\n\nWhat earlier tool calls in this same request already found (you may call more "
            "tools that would still add something, or call none if this is already enough):\n"
            f"{history}"
        )
    return content


class LLMProvider(ABC):
    @abstractmethod
    def select_tools(
        self, query: str, tool_specs: list[ToolSpec], input_summary: str, history: str = ""
    ) -> list[ToolCall]:
        """Given the user's query and a text summary of the validated input (image count, format,
        modality guesses -- see input_validation.py), return the ordered list of tool calls to run
        next. Both providers receive the same text-only input_summary (not raw image bytes) so the
        Gemini-vs-Groq comparison is measuring tool-calling quality, not vision understanding.

        `history` is empty on the first call. controller.py's agentic loop (see `handle_query`) can
        call this again after running a round of tools, with `history` set to those tools' own
        results (pre-formatted by answer_composer.build_evidence, the same formatting the final
        answer is composed from) -- the model can use what it already learned to decide whether more
        calls would help, and with what arguments, or call none to signal it has enough. A provider
        that ignores `history` entirely still behaves exactly as before (single-round routing)."""
        raise NotImplementedError

    def generate_text(self, system: str, user: str) -> str:
        """Plain text completion -- what the answer composer (orchestrator/answer_composer.py) uses
        to phrase the final answer. Deliberately NOT abstract: select_tools is the only thing a
        provider must do, and a provider that can't generate text just leaves the composer switched
        off (it treats NotImplementedError like any other failure and keeps the deterministic
        answer). Implementations normalise SDK errors to RuntimeError, like select_tools does."""
        raise NotImplementedError


def get_provider(name: str) -> LLMProvider:
    if name == "gemini":
        from app.orchestrator.llm_providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == "groq":
        from app.orchestrator.llm_providers.groq_provider import GroqProvider

        return GroqProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}', expected 'gemini' or 'groq'")
