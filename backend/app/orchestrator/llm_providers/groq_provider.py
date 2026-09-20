"""Groq (OpenAI-compatible) tool-calling adapter -- the comparison provider (see base.py)."""

import json

from app.config import settings
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall
from app.orchestrator.tool_registry import ToolSpec
from app.orchestrator.llm_providers.gemini_provider import SYSTEM_PROMPT


class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set (see backend/.env.example).")
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model

    def select_tools(self, query: str, tool_specs: list[ToolSpec], input_summary: str) -> list[ToolCall]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters_schema,
                },
            }
            for spec in tool_specs
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nInput summary:\n{input_summary}"},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as exc:
            # Same normalization as gemini_provider: retired model ids, rate limits (Groq's free
            # tier is 30 RPM / 1000 RPD) and auth failures become a clean 503, not a 500.
            raise RuntimeError(f"Groq request failed ({self._model}): {exc}") from exc

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        result = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                # TypeError covers a None/non-string `arguments` -- outside the normal API
                # contract, but not something this code controls either; gemini_provider's
                # equivalent (`dict(c.args or {})`) already handles its own version of this.
                args = {}
            result.append(ToolCall(tool_name=tc.function.name, arguments=args))
        return result

    def generate_text(self, system: str, user: str) -> str:
        try:
            # Phrasing an answer from facts already in the prompt needs no long chain of thought --
            # "low" keeps the reasoning model's extra call to about a second.
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_completion_tokens=1200,
                reasoning_effort="low",
            )
        except Exception as exc:
            raise RuntimeError(f"Groq request failed ({self._model}): {exc}") from exc
        return (response.choices[0].message.content or "").strip()
