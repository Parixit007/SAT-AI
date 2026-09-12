"""
vqa_tool.py

Local (Mac-side) wrapper around PaliGemma fine-tuned on RSVQA-LR -- the mandatory single-image
visual-question-answering specialist. Same shape as ../grounding/grounding_tool.py and
../water_segmentation/water_segmentation_tool.py: one class, lazy imports, one inference method,
and no rendering (VQA answers are text, so there's nothing to draw).

Setup (once):
    pip install transformers accelerate pillow torch

    # The model is GATED on the Hugging Face Hub. Accept Google's Gemma license at
    #     https://huggingface.co/google/paligemma-3b-ft-rsvqa-lr-224
    # with the account whose read token you pass as `hf_token` (the backend reads HF_TOKEN from
    # .env -- see .env.example). Without that, loading fails with a GatedRepoError.
    #
    # ~11.7GB download: the Hub stores these weights in float32. They're cast to bfloat16 on load,
    # so the resident memory cost is roughly half the download.

Usage as a library (what the orchestrator calls):

    from vqa_tool import VQATool

    tool = VQATool(hf_token="hf_...")
    tool.answer("scene.jpg", "Is there a road in this image?")
    # -> {"answer": "yes", "confidence": 0.97}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_MODEL_ID = "google/paligemma-3b-ft-rsvqa-lr-224"
MAX_NEW_TOKENS = 20  # RSVQA answers are short: yes/no, a count, "urban"/"rural", a comparison


class VQATool:
    """Wraps a PaliGemma VQA checkpoint. One instance = one loaded model (~6GB resident in
    bfloat16) -- construct it once in the orchestrator process and reuse it. Note that this sits
    alongside the grounding and segmentation models in the same process; on a 16GB machine that's
    tight, so it loads lazily and only on first use."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Imported lazily so this module can be imported (for its return-type shape, or by the
        # registry at startup) without transformers installed or a model on disk.
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_id = model_id

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            token=hf_token,
            dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()

    def answer(self, image_path: str, question: str) -> Dict[str, Any]:
        """Answer one question about one image.

        Returns {"answer": str, "confidence": float}. `confidence` is the mean per-token
        probability of the generated answer -- a generation-likelihood proxy, NOT a calibrated
        probability that the answer is correct. A fluent wrong answer can score high; treat it as
        "how committed the model was", nothing stronger.
        """
        from PIL import Image

        torch = self._torch
        image = Image.open(image_path).convert("RGB")

        # Task-specific PaliGemma fine-tunes take the bare question as the prompt -- the
        # "answer en"/"caption en" task prefixes belong to the general-purpose `-mix-` checkpoints,
        # not to a checkpoint that was trained on one task.
        inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        new_tokens = generation.sequences[0][input_len:]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        probs = []
        for step_scores, token_id in zip(generation.scores, new_tokens):
            step_probs = torch.softmax(step_scores[0].float(), dim=-1)
            probs.append(step_probs[token_id].item())
        confidence = sum(probs) / len(probs) if probs else 0.0

        return {"answer": answer, "confidence": round(confidence, 4)}


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-token", default=None, help="HF read token (gated model)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")

    tool = VQATool(model_id=args.model_id, hf_token=args.hf_token)
    print(json.dumps(tool.answer(args.image, args.question), indent=2))


if __name__ == "__main__":
    _cli()
