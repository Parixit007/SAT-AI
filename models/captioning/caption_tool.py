"""
caption_tool.py

Local (Mac-side) wrapper around the remote-sensing captioner trained by
notebooks/kaggle_finetune_caption_vrsbench.ipynb: SmolVLM-500M-Instruct with a VRSBench LoRA merged
into the weights (a plain half-precision checkpoint directory, ~1GB -- no PEFT needed here).
Same shape as ../vqa/vqa_tool.py: one class, lazy imports, one inference method, no rendering.

Setup (once): download the `caption_model/` folder from the Kaggle notebook's Output tab into
models/captioning/checkpoints/caption_model/ (gitignored).

Usage as a library (what the orchestrator calls):

    from caption_tool import CaptionTool

    tool = CaptionTool("models/captioning/checkpoints/caption_model")
    tool.describe("scene.jpg")
    # -> {"caption": "The image shows a large airport apron with ...", "confidence": 0.61}

`confidence` is the mean per-token probability of the generated text -- a generation-likelihood
proxy for how committed the model was, NOT the probability that the description is true. A fluent
caption can still be wrong about details (counts especially); the orchestrator pairs it with the
detector's counts for that reason.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "caption_model"
MAX_NEW_TOKENS = 140  # VRSBench captions average 53 words (~75 tokens); this leaves room for the long ones


def trim_to_sentence(text: str) -> str:
    """A caption cut off by the token limit ends mid-sentence; keep only the complete sentences."""
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    return text[: cut + 1] if cut > 0 else text


class CaptionTool:
    """One instance = one loaded model (~1GB in bfloat16) -- construct it once and reuse it."""

    def __init__(self, checkpoint_dir: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        checkpoint = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        meta_path = checkpoint / "caption_meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"No caption model at {checkpoint} (missing caption_meta.json). Download the notebook's "
                "caption_model/ output folder there -- see this file's docstring."
            )
        meta = json.loads(meta_path.read_text())
        self.prompt = meta["prompt"]
        self.metrics = meta.get("fine_tuned_metrics", {})

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._torch = torch

        self.processor = AutoProcessor.from_pretrained(checkpoint)
        # The notebook trained and evaluated with splitting off: one 512px image = 64 tokens, not 1,088.
        self.processor.image_processor.do_image_splitting = bool(meta.get("do_image_splitting", False))
        self.model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device).eval()

    def describe(self, image_path: str, max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
        """Describe one image. Returns {"caption": str, "confidence": float}."""
        from PIL import Image

        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[[image]], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,  # the setting the notebook's evaluation used
                output_scores=True,
                return_dict_in_generate=True,
            )

        new_tokens = generation.sequences[:, inputs["input_ids"].shape[1]:]
        caption = trim_to_sentence(self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0])

        log_probs = self.model.compute_transition_scores(generation.sequences, generation.scores, normalize_logits=True)[0]
        real = new_tokens[0] != self.processor.tokenizer.pad_token_id
        confidence = float(log_probs[real].float().exp().mean()) if bool(real.any()) else 0.0
        return {"caption": re.sub(r"\s+", " ", caption), "confidence": round(confidence, 4)}


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")
    print(json.dumps(CaptionTool(args.checkpoint_dir).describe(args.image), indent=2))


if __name__ == "__main__":
    _cli()
