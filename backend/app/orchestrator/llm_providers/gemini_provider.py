"""Gemini (google-genai) tool-calling adapter -- the default provider (see base.py / config.py)."""

from app.config import settings
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall, format_routing_prompt
from app.orchestrator.tool_registry import ToolSpec

SYSTEM_PROMPT = (
    "You are the task router for SatQuery AI, a remote-sensing image-analysis assistant. Given a "
    "user's query, a description of the uploaded image(s)/location, and -- from the second time "
    "you're asked about this same query on -- what earlier tool calls already found, decide which "
    "specialist tool(s) to call next and with what arguments. Call functions only; never answer in "
    "plain text.\n\n"
    "How to decide:\n"
    "1. Read every tool's own description before choosing, not just its name -- several explicitly "
    "say when to prefer them over a similar-sounding one (text_guided_grounding is THE tool for "
    "counting or locating a named object; land_cover_analysis is THE tool for buildings, roads and "
    "vegetation; scene_description is only for a general 'describe this image' with no specific "
    "ask; visual_question_answering is for a short presence/comparison/rural-vs-urban question, not "
    "counting or describing). Match the query's actual words to that guidance instead of picking "
    "the first plausible-sounding tool.\n"
    "2. A query with more than one distinct ask (e.g. 'how many buildings are there, and is there "
    "any fire risk nearby?') needs a call for EACH part, not just one -- decompose it and cover "
    "every part before you're done, in one round or several.\n"
    "3. A broad, open-ended query about one image with no specific ask ('describe this', 'what's "
    "here?', 'tell me about this place') means scene_description, not a guess at some narrower "
    "thing the user might specifically want.\n"
    "4. If nothing you're given confidently matches any tool, still call the closest reasonable "
    "match rather than nothing -- a wrong-but-explainable answer beats silence, and the execution "
    "trace will show exactly what ran and why it might not fit.\n"
    "5. Once you have prior results to look at, use them: if an earlier result was thin, low-"
    "confidence, a bare one- or two-word answer, or its own summary said its numbers aren't fully "
    "reliable, and a DIFFERENT available tool would give a firmer answer to the same question, call "
    "it too (e.g. a scene description's building count is a rough guess -- land_cover_analysis's is "
    "the real count; a one-word visual_question_answering answer to a question with more nuance may "
    "need a follow-up from a tool built for that nuance). Never repeat a call you already made with "
    "the same arguments -- if every call you would still make is one already run, or nothing further "
    "would change the answer, call no functions. That is how you signal you are finished."
)


class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see backend/.env.example).")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def select_tools(
        self, query: str, tool_specs: list[ToolSpec], input_summary: str, history: str = ""
    ) -> list[ToolCall]:
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
        contents = format_routing_prompt(query, input_summary, history)

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

    def generate_text(self, system: str, user: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=0.3, max_output_tokens=2048)
        try:
            response = self._client.models.generate_content(model=self._model, contents=user, config=config)
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed ({self._model}): {exc}") from exc
        return (response.text or "").strip()
