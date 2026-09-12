"""Gemini (google-genai) tool-calling adapter -- the default provider (see base.py / config.py)."""

import json

from app.config import settings
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall
from app.orchestrator.tool_registry import ToolSpec

SYSTEM_PROMPT = (
    "You are the task router for SatQuery AI, a remote-sensing image analysis assistant. "
    "Given a user's natural-language query and a description of the uploaded image(s), decide "
    "which specialist tool(s) to call and with what arguments. Call one or more of the provided "
    "functions; do not answer in plain text. If no tool fits the query, do not call any function."
)


class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see backend/.env.example).")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def select_tools(self, query: str, tool_specs: list[ToolSpec], input_summary: str) -> list[ToolCall]:
        from google.genai import types

        function_declarations = [
            types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters_schema,
            )
            for spec in tool_specs
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=function_declarations)],
        )
        contents = f"Query: {query}\n\nInput summary:\n{input_summary}"

        try:
            response = self._client.models.generate_content(model=self._model, contents=contents, config=config)
        except Exception as exc:
            # Retired model ids, rate limits, auth problems -- all arrive as provider-specific
            # exception types. Normalize to RuntimeError so routes_query turns them into a clean
            # 503 with the provider's own message, instead of a 500 and a stack trace. (A retired
            # "gemini-2.0-flash" default is exactly how this surfaced the first time.)
            raise RuntimeError(f"Gemini request failed ({self._model}): {exc}") from exc

        calls = response.function_calls or []
        return [ToolCall(tool_name=c.name, arguments=dict(c.args or {})) for c in calls]
