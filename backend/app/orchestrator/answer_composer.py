"""Turns the specialists' own outputs into a readable answer.

The specialists are terse by nature -- a VQA checkpoint that answers in one word, a detector that
returns boxes, a GIS score -- and joining their raw text summaries (controller._synthesize_answer)
reads like a log line, not an answer. This asks the LLM provider to phrase what the tools found the
way an analyst would, with two guards that keep the project's original promise that the numbers in
the answer are the numbers in the trace:

  * the prompt confines the model to the tool results, and
  * every multi-digit / decimal number in the draft is checked against the evidence text (with
    rounding and fraction-to-percent allowed); a draft with unsupported numbers gets one corrective
    retry, then is discarded.

Any failure -- provider without text generation, API error, empty or over-long draft, numbers that
won't check out -- returns None and the caller keeps the deterministic summary. The raw per-tool
summaries stay visible in the UI's tool cards either way."""

import json
import re
from typing import Any, Optional

from app.orchestrator.llm_providers.base import LLMProvider
from app.orchestrator.tool_registry import ToolResult

SYSTEM_PROMPT = (
    "You write the final answer for SatQuery AI, a remote-sensing image-analysis assistant. You get the "
    "user's question and the results of the specialist tools that analysed their imagery. Explain the "
    "answer the way a careful analyst would to a non-expert: plain language, complete sentences, "
    "usually 2-6 sentences (a little longer only when there are several distinct findings).\n\n"
    "Rules:\n"
    "1. Use ONLY facts that appear in the tool results. Never add objects, counts, places, dates, causes "
    "or land-cover details that are not there. Copy numbers exactly as given (percentages may be "
    "rounded sensibly); do not round counts up or down.\n"
    "2. Lead with the direct answer to the question, then the supporting numbers, then any caveat that "
    "matters.\n"
    "3. Be honest about limits. If a tool's answer is a single word, low-confidence, or doesn't really "
    "address the question, say what it does and doesn't tell us instead of dressing it up. If a tool "
    "failed, say which analysis was unavailable and what that means for the answer. Objects the "
    "detector did not report are not proof they are absent.\n"
    "4. Detector counts are what it found, not a guarantee: say 'about' or 'at least', and note that "
    "some may be missed. When a generated scene description and the detector disagree about what is "
    "there or how many, trust the detector's counts and present the description as a general "
    "impression rather than fact.\n"
    "5. Confidence values are the tools' own scores, not probabilities of being right. Don't quote them "
    "as numbers; use words (0.75+ fairly confident, 0.5-0.75 moderately, under 0.5 not very confident). "
    "Give fractions as percentages (0.58 -> 58%), not raw decimals.\n"
    "6. Name analyses by what they are ('the object detector', 'the water-mapping model'), not by "
    "internal ids or file names.\n"
    "7. Plain prose. No preamble like 'Based on the results', no headings, and lists only if there are "
    "3+ separate findings. Do not repeat the question."
)

FRIENDLY_NAMES = {
    "visual_question_answering": "visual question-answering model (trained on low-resolution 10 m Sentinel-2 tiles, so it can be unreliable on sharp aerial imagery; answers in one or two words)",
    "text_guided_grounding": "object detector / text-guided grounding model (its scores run low even for correct boxes -- 0.3-0.5 is normal for real objects, so a low-looking score is not a reason for doubt; it can still miss small or partly hidden objects)",
    "water_body_segmentation": "water-mapping model (can over-report water on bright grey surfaces such as concrete or airport aprons)",
    "change_detection": "change-detection model",
    "optical_sar_fusion": "optical + SAR fusion analysis",
    "groundwater_potential": "groundwater-potential estimate (GIS layer overlay, not a trained model)",
    "wildfire_detection": "satellite fire-detection lookup (NASA FIRMS)",
    "scene_description": "scene description (a written description from a small remote-sensing captioning model when one is installed -- fluent, but it can be wrong about details and cannot count reliably -- plus an object scan by the detector, limited to airplanes, ships and storage tanks; the scan's counts are the reliable numbers, and its scores run low even for correct boxes, 0.3-0.5 is normal)",
}

# scene_description with no caption (no captioner installed, or it failed): don't let the label suggest
# a captioning model exists -- the composer would otherwise write "the captioning model did not
# provide a description", which is about a component the user never had.
_SCENE_SCAN_ONLY = (
    "object scan (object detector limited to airplanes, ships and storage tanks -- it cannot describe "
    "terrain, buildings or roads, so say plainly that no fuller description of the scene is available; "
    "its scores run low even for correct boxes, 0.3-0.5 is normal)"
)

_MAX_LIST = 8
_MAX_EVIDENCE_CHARS = 1800
_MAX_ANSWER_CHARS = 3000
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")


def _compact(value: Any, key: str = "") -> Any:
    """JSON-able copy of a tool's structured_data small enough to put in a prompt."""
    if key == "detections" and isinstance(value, list):
        scores = [d.get("score") for d in value if isinstance(d, dict) and d.get("score") is not None]
        summary: dict[str, Any] = {"count": len(value)}
        if scores:
            summary["score_range"] = [round(min(scores), 3), round(max(scores), 3)]
        return summary
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _compact(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_LIST:
            return {"items": len(value), "first": [_compact(v) for v in value[:3]]}
        return [_compact(v) for v in value]
    if isinstance(value, str):
        return value[:300]
    return value


def build_evidence(executed: list[tuple[ToolResult, dict, Optional[str]]], warnings: list[str]) -> str:
    blocks = []
    for i, (result, _arguments, _checkpoint) in enumerate(executed, start=1):
        label = FRIENDLY_NAMES.get(result.tool_name, result.tool_name)
        if result.tool_name == "scene_description" and not result.structured_data.get("caption"):
            label = _SCENE_SCAN_ONLY
        details = json.dumps(_compact(result.structured_data), ensure_ascii=False, default=str)
        if len(details) > _MAX_EVIDENCE_CHARS:
            details = details[:_MAX_EVIDENCE_CHARS] + " ...(truncated)"
        blocks.append(
            f"[{i}] {label} -- the tool's own confidence score: {result.confidence:.2f}\n"
            f"Summary: {result.text_summary}\nDetails: {details}"
        )
    if warnings:
        blocks.append("Problems reported while running: " + "; ".join(warnings))
    return "\n\n".join(blocks)


def unsupported_numbers(answer: str, evidence: str) -> list[str]:
    """Multi-digit or decimal numbers in `answer` that no number in `evidence` accounts for (verbatim,
    rounded, or as a fraction shown as a percent). Single-digit integers are not checked -- they are
    too often ordinary counting words ('one of the ...') to be a useful signal."""
    allowed: set[float] = set()
    for token in _NUMBER.findall(evidence):
        value = float(token.replace(",", ""))
        for decimals in range(5):
            allowed.add(round(value, decimals))
        if 0 < value <= 1:
            for decimals in range(4):
                allowed.add(round(value * 100, decimals))
    bad = []
    for token in _NUMBER.findall(answer):
        digits = token.replace(",", "").replace(".", "")
        if len(digits) < 2 and "." not in token:
            continue
        if round(float(token.replace(",", "")), 4) not in allowed:
            bad.append(token)
    return bad


def compose_answer(
    provider: LLMProvider,
    query: str,
    executed: list[tuple[ToolResult, dict, Optional[str]]],
    warnings: list[str],
    input_summary: str,
) -> tuple[Optional[str], Optional[str]]:
    """Returns (answer, problem). `answer` is None when the caller should keep its deterministic text;
    `problem` is a short human-readable reason worth surfacing in the trace, or None when there is
    nothing to report (e.g. the provider simply doesn't generate text)."""
    evidence = build_evidence(executed, warnings)
    user = f"Question: {query}\n\nInput: {input_summary}\n\nTool results:\n{evidence}"

    def draft(prompt: str) -> str:
        return provider.generate_text(SYSTEM_PROMPT, prompt).strip()

    try:
        text = draft(user)
    except NotImplementedError:
        return None, None
    except Exception as exc:
        return None, f"answer phrasing was unavailable ({exc}); showing the tools' own summaries"

    if not text or len(text) > _MAX_ANSWER_CHARS:
        return None, "answer phrasing returned an unusable draft; showing the tools' own summaries"

    allowed_source = f"{query}\n{input_summary}\n{evidence}"
    bad = unsupported_numbers(text, allowed_source)
    if bad:
        try:
            text = draft(
                f"{user}\n\nYour previous draft used numbers that are not in the tool results: {', '.join(bad)}. "
                "Rewrite the answer using only numbers that appear above."
            )
        except Exception as exc:
            return None, f"answer phrasing failed on retry ({exc}); showing the tools' own summaries"
        if not text or len(text) > _MAX_ANSWER_CHARS or unsupported_numbers(text, allowed_source):
            return None, "answer phrasing kept adding numbers the tools did not report; showing the tools' own summaries"
    return text, None
