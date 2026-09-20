"""
grounding_tool.py

Local (Mac-side) wrapper around a Grounding DINO checkpoint fine-tuned with Open-GroundingDino
(see the kaggle_finetune_grounding_dino_* notebooks). Exposes exactly the interface an orchestrator
needs: (image, text query) -> structured box coordinates. No image generation anywhere in the
model path -- drawing is a separate, explicit step so the orchestrator can choose whether to render
anything at all.

Setup (once):
    git clone --depth 1 https://github.com/longzw1997/Open-GroundingDino.git \
        models/grounding/vendor/Open-GroundingDino          # gitignored; or set OPEN_GROUNDINGDINO_DIR
    pip install addict yapf termcolor pycocotools opencv-python-headless

    The checkpoint downloaded from the Kaggle notebook's Output tab lives in
    checkpoints/dior_rsvg_finetuned.pth. There is no separate config file to keep: the model is
    built from the checkout's own config/cfg_odvg.py, the same one training used.

Why this file does more than call the library -- three things in the checkout don't survive a Mac
running current PyTorch/transformers, each found by actually running it, and each handled here
without editing the checkout (so a fresh `git clone` works):
  1. `groundingdino.util.inference` (the "official" inference helpers) still imports module paths
     the repo moved away from, so load/predict are reimplemented against the paths training uses.
  2. models/GroundingDINO/ms_deform_attn.py imports the compiled CUDA op at the top and raises if it
     is missing, even though its forward() already falls back to a pure-PyTorch path off-CUDA. A
     stub module satisfies the import; it is never called unless a CUDA build is present.
  3. transformers>=5 removed BertModel.get_head_mask and changed BertEncoder's signature, which
     breaks the repo's BertModelWarper. It is patched to a minimal equivalent (verified to match the
     stock BERT forward exactly for 2D masks, and to keep GroundingDINO's block-diagonal per-phrase
     3D attention masks isolated). On transformers 4.x the original is used untouched.

Usage as a library (what the orchestrator calls):

    from grounding_tool import GroundingTool

    tool = GroundingTool(checkpoint_path="checkpoints/dior_rsvg_finetuned.pth")
    detections = tool.ground("scene.jpg", "the ship near the harbor entrance")
    # -> [{"phrase": "the ship near the harbor entrance", "bbox_xyxy": [x1, y1, x2, y2], "score": 0.83}, ...]

Usage from the command line (quick sanity check + draws an annotated copy):

    python grounding_tool.py --image scene.jpg --query "airplane" \
        --checkpoint checkpoints/dior_rsvg_finetuned.pth --draw out.jpg
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "Open-GroundingDino"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SHORT_SIDE, MAX_SIDE = 800, 1333  # the resize the model was trained/evaluated with

_open_groundingdino = None  # the imported pieces of the checkout, built once per process


def _patch_bert_warper() -> None:
    """See point 3 in the module docstring."""
    from transformers import BertModel

    if hasattr(BertModel, "get_head_mask"):
        return  # transformers 4.x: the checkout's own BertModelWarper works as written

    import torch.nn as nn
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

    import models.GroundingDINO.bertwarper as bertwarper

    def __init__(self, bert_model):
        nn.Module.__init__(self)
        self.config = bert_model.config
        self.embeddings = bert_model.embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **_unused):
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        # 2D [B, L] padding mask or 3D [B, L, L] self-attention mask (GroundingDINO's per-phrase
        # block-diagonal one) -> additive [B, 1, *, L] mask broadcastable over heads.
        mask = attention_mask[:, None, :, :] if attention_mask.dim() == 3 else attention_mask[:, None, None, :]
        mask = mask.to(dtype=embedding_output.dtype)
        mask = (1.0 - mask) * torch.finfo(mask.dtype).min

        encoded = self.encoder(embedding_output, attention_mask=mask).last_hidden_state
        pooled = self.pooler(encoded) if self.pooler is not None else None
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=encoded, pooler_output=pooled)

    bertwarper.BertModelWarper.__init__ = __init__
    bertwarper.BertModelWarper.forward = forward


def _import_open_groundingdino(repo_dir: Path):
    global _open_groundingdino
    if _open_groundingdino is not None:
        return _open_groundingdino

    if not (repo_dir / "config" / "cfg_odvg.py").exists():
        raise RuntimeError(
            f"Open-GroundingDino checkout not found at {repo_dir}. Run: "
            f"git clone --depth 1 https://github.com/longzw1997/Open-GroundingDino.git {repo_dir} "
            "(or point OPEN_GROUNDINGDINO_DIR at an existing checkout), then "
            "pip install addict yapf termcolor pycocotools opencv-python-headless"
        )
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    has_cuda_op = True
    try:
        import MultiScaleDeformableAttention  # noqa: F401
    except ImportError:
        has_cuda_op = False
        sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")

    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from models.GroundingDINO import build_groundingdino
    from util.slconfig import SLConfig

    _patch_bert_warper()
    _open_groundingdino = types.SimpleNamespace(
        SLConfig=SLConfig,
        build_groundingdino=build_groundingdino,
        clean_state_dict=clean_state_dict,
        get_phrases_from_posmap=get_phrases_from_posmap,
        has_cuda_op=has_cuda_op,
    )
    return _open_groundingdino


def _resize_target(width: int, height: int) -> tuple:
    """DETR-style: shorter side to SHORT_SIDE unless that would push the longer side past MAX_SIDE."""
    size = SHORT_SIDE
    low, high = float(min(width, height)), float(max(width, height))
    if high / low * size > MAX_SIDE:
        size = int(round(MAX_SIDE * low / high))
    if width < height:
        return int(size * height / width), size
    return size, int(size * width / height)


def _preprocess(image) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    out_h, out_w = _resize_target(*image.size)
    resized = image.resize((out_w, out_h), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class GroundingTool:
    """Wraps a (fine-tuned) Grounding DINO checkpoint. One instance = one loaded model in memory --
    construct it once in your orchestrator process and re-use it across calls; re-loading the
    checkpoint per-call is the main thing that would make this slow.

    Runs on CPU by default (~2s per image on an M-series Mac, deterministic); set
    GROUNDING_DEVICE=mps|cuda to override."""

    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, device: Optional[str] = None):
        repo_dir = Path(os.environ.get("OPEN_GROUNDINGDINO_DIR", VENDOR_DIR))
        og = _import_open_groundingdino(repo_dir)

        device = device or os.environ.get("GROUNDING_DEVICE") or "cpu"
        if device.startswith("cuda") and not og.has_cuda_op:
            # Without the compiled op the CUDA branch of the attention forward can't run; the
            # pure-PyTorch path only engages for non-CUDA tensors.
            device = "cpu"
        self.device = device

        args = og.SLConfig.fromfile(str(config_path or repo_dir / "config" / "cfg_odvg.py"))
        args.device = device
        # cfg_odvg.py defaults to use_coco_eval=True, which makes the builder look for a COCO
        # annotation file that only exists at training time. Inference never touches postprocessors.
        args.use_coco_eval = False
        args.label_list = ["object"]
        model, _, _ = og.build_groundingdino(args)

        # weights_only=False: a training checkpoint also carries non-tensor state (epoch, args).
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(og.clean_state_dict(checkpoint["model"]), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} doesn't match the model architecture "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys, e.g. "
                f"{(list(missing) + list(unexpected))[:3]}) -- refusing to run with partly random weights."
            )
        self.model = model.eval().to(device)
        self._get_phrases = og.get_phrases_from_posmap

    def _forward(self, image, caption: str):
        """One model pass. Returns (sigmoid token logits [queries, text_len] on CPU, boxes [queries, 4]
        as normalised cx,cy,w,h on CPU)."""
        with torch.no_grad():
            outputs = self.model(_preprocess(image).to(self.device)[None], captions=[caption])
        return outputs["pred_logits"].sigmoid()[0].cpu(), outputs["pred_boxes"][0].cpu()

    def ground(
        self,
        image_path: str,
        query: str,
        box_threshold: float = 0.25,
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
        Coordinates are absolute pixels in the original image. `top_k=None` returns every detection
        above the threshold -- what a count needs. The 0.25/0.25 defaults are the thresholds the
        fine-tune was evaluated with. Near-duplicate boxes of the same phrase (IoU > 0.6) are
        merged so a count isn't inflated by one object being boxed twice. No image is generated or
        modified -- drawing is a separate step (see draw_boxes below).
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        caption = query.lower().strip()
        if not caption.endswith("."):
            caption += "."

        logits, boxes = self._forward(image, caption)
        keep = logits.max(dim=1)[0] > box_threshold
        logits, boxes = logits[keep], boxes[keep]

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        phrases = [self._get_phrases(row > text_threshold, tokenized, tokenizer) for row in logits]

        scores = logits.max(dim=1)[0]
        # A box below text_threshold on every token has no phrase of its own -- fall back to the query.
        phrases = [phrase or caption.rstrip(".").strip() for phrase in phrases]

        xyxy = torch.stack([
            (boxes[:, 0] - boxes[:, 2] / 2) * width, (boxes[:, 1] - boxes[:, 3] / 2) * height,
            (boxes[:, 0] + boxes[:, 2] / 2) * width, (boxes[:, 1] + boxes[:, 3] / 2) * height,
        ], dim=1) if len(boxes) else torch.zeros((0, 4))
        if len(xyxy):
            from torchvision.ops import batched_nms

            group = {p: i for i, p in enumerate(dict.fromkeys(phrases))}
            keep_idx = batched_nms(xyxy, scores, torch.tensor([group[p] for p in phrases]), iou_threshold=0.6)
        else:
            keep_idx = []

        detections = [
            {
                "phrase": phrases[i],
                "bbox_xyxy": [round(v, 1) for v in xyxy[i].tolist()],
                "score": round(float(scores[i]), 4),
            }
            for i in keep_idx.tolist() if len(xyxy)
        ]

        detections.sort(key=lambda d: d["score"], reverse=True)
        if top_k is not None:
            detections = detections[:top_k]
        return detections


    def scan(
        self,
        image_path: str,
        categories: List[str],
        box_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Find every instance of any of `categories` in one pass -- an object inventory.

        Unlike ground(), which labels a box with whichever caption tokens fire (so a weak box can end
        up called "airport dam expressway toll station ship"), this scores each category as the mean
        of its own tokens' logits, gives each box the single best category, and keeps it if that
        score clears `box_threshold`. Same-category boxes overlapping by IoU > 0.6 are merged, and a
        box claimed by two categories keeps only the stronger (IoU > 0.7). Sorted by score."""
        from PIL import Image
        from torchvision.ops import batched_nms, nms

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        names = [c.lower().strip() for c in categories]
        caption = " . ".join(names) + " ."
        logits, boxes = self._forward(image, caption)

        tokenizer = self.model.tokenizer
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(caption)["input_ids"])
        spans: List[List[int]] = [[] for _ in names]
        category = 0
        for position, token in enumerate(tokens):
            if token in ("[CLS]", "[SEP]"):
                continue
            if token == ".":
                category += 1
            elif category < len(spans):
                spans[category].append(position)

        per_category = torch.stack(
            [logits[:, idx].mean(dim=1) if idx else torch.zeros(len(logits)) for idx in spans], dim=1
        )
        best, label = per_category.max(dim=1)
        keep = best > box_threshold
        if not keep.any():
            return []
        best, label, boxes = best[keep], label[keep], boxes[keep]

        xyxy = torch.stack([
            (boxes[:, 0] - boxes[:, 2] / 2) * width, (boxes[:, 1] - boxes[:, 3] / 2) * height,
            (boxes[:, 0] + boxes[:, 2] / 2) * width, (boxes[:, 1] + boxes[:, 3] / 2) * height,
        ], dim=1)
        kept = batched_nms(xyxy, best, label, iou_threshold=0.6)
        xyxy, best, label = xyxy[kept], best[kept], label[kept]
        kept = nms(xyxy, best, iou_threshold=0.7)

        detections = [
            {"phrase": names[int(label[i])], "bbox_xyxy": [round(v, 1) for v in xyxy[i].tolist()], "score": round(float(best[i]), 4)}
            for i in kept.tolist()
        ]
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

# One colour per phrase so a multi-category scan reads at a glance.
_PALETTE = [(230, 57, 70), (29, 143, 226), (46, 160, 67), (245, 158, 11), (147, 51, 234), (20, 184, 166), (236, 72, 153), (120, 113, 108)]
_LABEL_LIMIT = 12  # above this many boxes, per-box text labels just cover the image


def draw_boxes(image_path: str, detections: List[Dict[str, Any]], output_path: str) -> None:
    """Separate, explicit drawing step -- takes the JSON-able output of GroundingTool.ground()
    and writes an annotated copy of the image. Doesn't touch the model at all. With many boxes
    (a count) it draws outlines only, colour-coded by phrase, plus a per-phrase tally banner."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1.0, max(image.size) / 1000)
    try:
        font = ImageFont.truetype("Arial.ttf", int(16 * scale))
    except OSError:
        font = ImageFont.load_default()

    colors: Dict[str, tuple] = {}
    for det in detections:
        colors.setdefault(det["phrase"], _PALETTE[len(colors) % len(_PALETTE)])
    labelled = len(detections) <= _LABEL_LIMIT
    line_width = max(2, int((3 if labelled else 2) * scale))

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = colors[det["phrase"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if labelled:
            label = f'{det["phrase"]} ({det["score"]:.2f})'
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle([text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
            draw.text((x1, y1), label, fill="white", font=font)

    if not labelled:
        tally: Dict[str, int] = {}
        for det in detections:
            tally[det["phrase"]] = tally.get(det["phrase"], 0) + 1
        y = int(8 * scale)
        for phrase, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            banner = f"{phrase}: {n}"
            text_bbox = draw.textbbox((int(8 * scale), y), banner, font=font)
            draw.rectangle([text_bbox[0] - 4, text_bbox[1] - 3, text_bbox[2] + 4, text_bbox[3] + 3], fill=colors[phrase])
            draw.text((int(8 * scale), y), banner, fill="white", font=font)
            y = text_bbox[3] + int(8 * scale)

    image.save(output_path)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--query", required=True, help="Text query, e.g. 'airplane' or 'the ship near the harbor entrance'")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned .pth checkpoint")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--draw", default=None, help="If set, write an annotated copy of the image here")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")

    tool = GroundingTool(checkpoint_path=args.checkpoint)
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
