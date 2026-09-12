"""
grounding_tool.py

Local (Mac-side) wrapper around a Grounding DINO checkpoint fine-tuned with Open-GroundingDino
(see kaggle_finetune_grounding_dino_dior_rsvg.ipynb). Exposes exactly the interface an orchestrator
needs: (image, text query) -> structured box coordinates. No image generation anywhere in this
file -- drawing is a separate, explicit step so the orchestrator can choose whether to render
anything at all.

Setup (once):
    git clone https://github.com/longzw1997/Open-GroundingDino.git
    cd Open-GroundingDino && pip install -r requirements.txt
    pip install pillow

    # No NVIDIA GPU on a Mac, so skip the CUDA op build entirely:
    #     cd models/GroundingDINO/ops && python setup.py build install
    # Grounding DINO's deformable-attention op has a pure-PyTorch fallback that's used
    # automatically when the compiled CUDA extension isn't available, so CPU/MPS inference
    # still works -- just slower than on a CUDA GPU. If `import groundingdino` errors out
    # looking for the compiled op, that fallback path is the thing to check in
    # models/GroundingDINO/ops/modules/ms_deform_attn.py.

    # The config (GroundingDINO_SwinT_OGC.py) lives next to this script; the checkpoint
    # downloaded from the Kaggle notebook's Output tab lives in checkpoints/dior_rsvg_finetuned.pth.

Usage as a library (what your orchestrator calls):

    from grounding_tool import GroundingTool

    tool = GroundingTool(
        config_path="GroundingDINO_SwinT_OGC.py",
        checkpoint_path="checkpoints/dior_rsvg_finetuned.pth",
    )
    detections = tool.ground("scene.jpg", "the ship near the harbor entrance")
    # -> [{"phrase": "the ship near the harbor entrance", "bbox_xyxy": [x1, y1, x2, y2], "score": 0.83}, ...]

Usage from the command line (quick sanity check + draws an annotated copy):

    python grounding_tool.py --image scene.jpg --query "the ship near the harbor entrance" \
        --checkpoint checkpoints/dior_rsvg_finetuned.pth --config GroundingDINO_SwinT_OGC.py --draw out.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch


class GroundingTool:
    """Wraps a (fine-tuned) Grounding DINO checkpoint. One instance = one loaded model in memory --
    construct it once in your orchestrator process and re-use it across calls; re-loading the
    checkpoint per-call is the main thing that would make this slow."""

    def __init__(self, config_path: str, checkpoint_path: str, device: Optional[str] = None):
        # Imported lazily so this module can be imported (e.g. for the dataclass-ish return type)
        # without requiring groundingdino to be installed yet.
        from groundingdino.util.inference import load_model

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        self.model = load_model(config_path, checkpoint_path, device=device)

    def ground(
        self,
        image_path: str,
        query: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run one (image, query) pair through the model.

        `query` can be a single category word ("ship"), several categories separated by " . "
        (Grounding DINO's own convention: "ship . vehicle . airplane ."), or a full referring
        expression ("the ship near the harbor entrance") -- whatever DIOR-RSVG-style phrasing your
        orchestrator sends, since that's what the model was fine-tuned on.

        Returns a list of detections sorted by score (highest first), each:
            {"phrase": str, "bbox_xyxy": [x1, y1, x2, y2], "score": float}
        Coordinates are absolute pixels in the original image. No image is generated or modified --
        drawing is a separate step (see draw_boxes below).
        """
        from groundingdino.util.inference import load_image, predict

        image_source, image = load_image(image_path)
        h, w = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        detections = []
        for box_norm, score, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box_norm.tolist()
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            detections.append({
                "phrase": phrase,
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "score": round(float(score), 4),
            })

        detections.sort(key=lambda d: d["score"], reverse=True)
        if top_k is not None:
            detections = detections[:top_k]
        return detections


def draw_boxes(image_path: str, detections: List[Dict[str, Any]], output_path: str) -> None:
    """Separate, explicit drawing step -- takes the JSON-able output of GroundingTool.ground()
    and writes an annotated copy of the image. Doesn't touch the model at all."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        label = f'{det["phrase"]} ({det["score"]:.2f})'
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill="red")
        draw.text((x1, y1), label, fill="white", font=font)

    image.save(output_path)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--query", required=True, help="Text query, e.g. 'the ship near the harbor entrance'")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned .pth checkpoint")
    parser.add_argument("--config", required=True, help="Path to GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--draw", default=None, help="If set, write an annotated copy of the image here")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")

    tool = GroundingTool(config_path=args.config, checkpoint_path=args.checkpoint)
    detections = tool.ground(
        args.image, args.query,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold, top_k=args.top_k,
    )

    print(json.dumps(detections, indent=2))

    if args.draw:
        draw_boxes(args.image, detections, args.draw)
        print(f"\nAnnotated image written to {args.draw}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
