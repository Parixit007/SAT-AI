"""Registry adapter for the visual-question-answering specialist (models/vqa/). The model is gated
on the Hugging Face Hub and ~11.7GB to download, so the VQATool singleton is only built on the
first actual call -- registering this tool (cheap, metadata-only) always works, even with no token
configured and nothing downloaded."""

from typing import Any

from app.config import MODELS_DIR, settings
from app.orchestrator.tool_registry import QueryInput, ToolResult, ToolSpec
from app.specialists._loader import load_module

_vqa_tool = None  # lazy singleton -- see module docstring


def _get_tool():
    global _vqa_tool
    if _vqa_tool is None:
        if not settings.hf_token:
            raise RuntimeError(
                "The VQA model is not configured -- set HF_TOKEN in .env with a Hugging Face read "
                "token whose account has accepted the license for "
                f"{settings.vqa_model_id} (see .env.example)."
            )
        module = load_module(MODELS_DIR / "vqa" / "vqa_tool.py", "vqa_tool")
        _vqa_tool = module.VQATool(model_id=settings.vqa_model_id, hf_token=settings.hf_token)
    return _vqa_tool


def _handle(query_input: QueryInput, arguments: dict[str, Any]) -> ToolResult:
    question = arguments.get("question", "")
    image_path = query_input.images[0]

    result = _get_tool().answer(str(image_path), question)

    return ToolResult(
        tool_name="visual_question_answering",
        text_summary=result["answer"],
        structured_data={"question": question, "answer": result["answer"]},
        evidence_image_path=None,  # the answer is the output; there's nothing to render
        confidence=result["confidence"],
    )


TOOL_SPEC = ToolSpec(
    name="visual_question_answering",
    description=(
        "Answer a free-form natural-language question about a single image -- presence ('is there "
        "a road?'), counting ('how many buildings?'), comparison, or scene type (rural vs urban). "
        "Use this for any question about image content that isn't asking to locate/outline a "
        "specific object (use text_guided_grounding for that) or to measure water coverage (use "
        "water_body_segmentation). Returns a short text answer. Its confidence score is the "
        "model's generation likelihood, not a calibrated probability that the answer is correct."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the image, in plain language.",
            },
        },
        "required": ["question"],
    },
    min_images=1,
    max_images=1,
    compatible_modalities=["optical"],
    handler=_handle,
    checkpoint_id="paligemma-3b-ft-rsvqa-lr-224",
)
