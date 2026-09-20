"""
semantic_change_tool.py

Stage 2 of the bi-temporal change specialist (Stage 1 = change_detection_tool.py's training-free
pixel differencing). A Siamese semantic-change network trained on SECOND-CC
(notebooks/kaggle_finetune_change_segmentation_second.ipynb): given a before/after optical pair it
predicts (1) WHERE the scene changed and (2) the land-cover class of every changed pixel at each
date -- so it can say WHICH classes gained or lost area ("buildings +3.2% of the scene, low
vegetation -2.9%") and the dominant from->to transitions, not just that "something changed".

Why a change network rather than a plain land-cover segmenter (found by inspecting the real data,
not assumed): SECOND-CC's semantic maps only carry a class label INSIDE changed regions --
unchanged pixels are one shared "no change" color in both maps -- so per-image land-cover fractions
at each date aren't learnable from these labels. What IS fully determined by them is the net area
change per class: unchanged pixels have the same class at both dates and cancel out of
(area at t2) - (area at t1), leaving only changed pixels. That is exactly what
`compute_class_changes` computes.

Mirrors ../water_segmentation/water_segmentation_tool.py: one *Tool class + one inference method,
a separate top-level draw_*() for visualization. Only segmentation_models_pytorch's public
`get_encoder` is used -- the decoders below are written out here rather than reusing smp's, whose
decoder constructor arguments changed between releases.

Setup (once):
    pip install segmentation-models-pytorch pillow scipy

    # Checkpoint lives at checkpoints/semantic_change_unet.pt next to this script -- a plain
    # torch.save'd dict: {"model_state_dict", "encoder_name", "img_size", "class_names", "metrics"}.

Usage as a library:

    from semantic_change_tool import SemanticChangeTool

    tool = SemanticChangeTool(checkpoint_path="checkpoints/semantic_change_unet.pt")
    result = tool.analyze("before.png", "after.png")
    # -> {"change_mask", "change_fraction", "largest_region_bbox", "class_changes", "transitions",
    #     "sem_before", "sem_after", "confidence"}

Usage from the command line:

    python semantic_change_tool.py --image1 before.png --image2 after.png \
        --checkpoint checkpoints/semantic_change_unet.pt --draw out.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- BEGIN SHARED MODEL BLOCK -----------------------------------------------------------------
# Duplicated verbatim in notebooks/kaggle_finetune_change_segmentation_second.ipynb (each models/*/
# script stays standalone -- see CLAUDE.md -- and a Kaggle notebook can't import from this repo).
# backend/tests/test_semantic_change.py FAILS if the two copies drift, so edit both together.

# Class index -> (name, color in SECOND-CC's RGB-coded semantic maps). White (255,255,255) is NOT a
# class: it marks "no change" and is shared by both dates' maps (0 of 10.7M pixels differed between
# the A and B maps across 163 sampled pairs). The map contains exactly these 7 colors, nothing else.
# The two greens are easy to swap by eye and were, at first: (0,255,0)=trees and (0,128,0)=low
# vegetation is what the captions say -- among pairs containing only the bright green, 89% of
# captions mention "tree(s)" vs 33% for pairs containing only the dark green, whose captions talk
# about sparse vegetation / green fields / farmland instead.
CLASS_NAMES = ["water", "bare ground", "low vegetation", "trees", "buildings", "playground"]
CLASS_COLORS_RGB = [(0, 0, 255), (128, 128, 128), (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0)]
NO_CHANGE_COLOR_RGB = (255, 255, 255)
IGNORE_INDEX = 255  # semantic-loss ignore label for unchanged pixels
NUM_CLASSES = len(CLASS_NAMES)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    """U-Net decoder over a 5-level encoder pyramid: `feats` = [input, f1(H/2), f2(H/4), f3(H/8),
    f4(H/16), f5(H/32)] as returned by smp's get_encoder; the input-resolution entry is unused."""

    def __init__(self, enc_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        c1, c2, c3, c4, c5 = enc_channels[1:]
        d1, d2, d3, d4, d5 = decoder_channels
        self.b1 = DecoderBlock(c5, c4, d1)
        self.b2 = DecoderBlock(d1, c3, d2)
        self.b3 = DecoderBlock(d2, c2, d3)
        self.b4 = DecoderBlock(d3, c1, d4)
        self.b5 = DecoderBlock(d4, 0, d5)
        self.out_channels = d5

    def forward(self, feats):
        f1, f2, f3, f4, f5 = feats[1:]
        x = self.b1(f5, f4)
        x = self.b2(x, f3)
        x = self.b3(x, f2)
        x = self.b4(x, f1)
        return self.b5(x, None)


class SiameseSCDNet(nn.Module):
    """(before, after) -> (change logit [N,1,H,W], semantic logits before [N,K,H,W], semantic logits
    after [N,K,H,W]). One ResNet encoder shared by both dates; ONE semantic decoder shared by both
    dates (the same land-cover appearance model at t1 and t2); a separate change decoder over
    per-level fused features [fA, fB, |fA-fB|]."""

    def __init__(self, encoder_name="resnet34", encoder_weights=None, num_classes=NUM_CLASSES):
        super().__init__()
        from segmentation_models_pytorch.encoders import get_encoder

        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        ch = list(self.encoder.out_channels)
        self.sem_decoder = UNetDecoder(ch)
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
                for c in ch[1:]
            ]
        )
        self.change_decoder = UNetDecoder(ch)
        self.sem_head = nn.Conv2d(self.sem_decoder.out_channels, num_classes, 3, padding=1)
        self.change_head = nn.Conv2d(self.change_decoder.out_channels, 1, 3, padding=1)

    def forward(self, a, b):
        n = a.shape[0]
        feats = self.encoder(torch.cat([a, b], dim=0))  # one pass for both dates
        sem = self.sem_head(self.sem_decoder(feats))  # [2N,K,H,W], shared weights
        fa = [f[:n] for f in feats]
        fb = [f[n:] for f in feats]
        fused = [fa[0]] + [m(torch.cat([x, y, (x - y).abs()], dim=1)) for m, x, y in zip(self.fuse, fa[1:], fb[1:])]
        change = self.change_head(self.change_decoder(fused))
        return change, sem[:n], sem[n:]


# --- END SHARED MODEL BLOCK -------------------------------------------------------------------

# A class's net area change under this share of the whole scene is reported as "unchanged" rather
# than as noise-level increase/decrease. Heuristic, not fit to data.
NET_CHANGE_TOLERANCE = 0.005


def compute_class_changes(
    change_mask: np.ndarray,
    sem_before: np.ndarray,
    sem_after: np.ndarray,
    class_names: List[str],
    tolerance: float = NET_CHANGE_TOLERANCE,
) -> Dict[str, Any]:
    """Pure numpy -- no model. `change_mask` [H,W] bool; `sem_before`/`sem_after` [H,W] integer class
    indices (only read where change_mask is True). All fractions are shares of the WHOLE scene.

    Net change per class is exact from changed pixels alone: unchanged pixels have the same class at
    both dates, so they cancel out of (area at t2) - (area at t1). A consequence worth testing:
    the net changes of all classes sum to zero, since every changed pixel adds one to `after` and
    one to `before`."""
    k = len(class_names)
    total = change_mask.size
    changed = change_mask.astype(bool)
    before = sem_before[changed].astype(np.int64)
    after = sem_after[changed].astype(np.int64)

    before_counts = np.bincount(before, minlength=k)[:k]
    after_counts = np.bincount(after, minlength=k)[:k]
    pair_counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(pair_counts, (before, after), 1)

    def direction(net: float) -> str:
        if net > tolerance:
            return "increased"
        if net < -tolerance:
            return "decreased"
        return "unchanged"

    class_changes = []
    for i, name in enumerate(class_names):
        net = float(after_counts[i] - before_counts[i]) / total
        class_changes.append(
            {
                "class": name,
                "before": round(float(before_counts[i]) / total, 4),
                "after": round(float(after_counts[i]) / total, 4),
                "net": round(net, 4),
                "direction": direction(net),
            }
        )

    transitions = []
    for i in range(k):
        for j in range(k):
            if i != j and pair_counts[i, j] > 0:
                transitions.append({"from": class_names[i], "to": class_names[j], "fraction": round(float(pair_counts[i, j]) / total, 4)})
    transitions.sort(key=lambda t: t["fraction"], reverse=True)

    return {
        "change_fraction": round(float(changed.sum()) / total, 4),
        "class_changes": class_changes,
        "transitions": transitions[:8],
    }


def describe_changes(result: Dict[str, Any]) -> str:
    """Plain-language summary of `analyze()` output (or anything with the same keys). The building
    line is always stated, increase/decrease/unchanged -- 'has the built-up area increased,
    decreased, or remained unchanged?' is one of the spec's own representative queries, and a
    silent omission would read as 'no answer' rather than 'no change'."""
    pct = result["change_fraction"] * 100
    if result["change_fraction"] <= 0:
        return "No land-cover change was detected between the two dates."

    parts = [f"About {pct:.1f}% of the scene changed between the two dates."]
    moved = [c for c in result["class_changes"] if c["direction"] != "unchanged"]
    moved.sort(key=lambda c: abs(c["net"]), reverse=True)
    if moved:
        listed = ", ".join(f"{c['class']} {c['net'] * 100:+.1f}%" for c in moved)
        parts.append(f"Net land-cover change (share of the whole scene): {listed}.")
    else:
        parts.append("No single land-cover class gained or lost a meaningful share of the scene.")

    if result["transitions"]:
        top = "; ".join(f"{t['from']} to {t['to']} ({t['fraction'] * 100:.1f}%)" for t in result["transitions"][:3])
        parts.append(f"Largest transitions: {top}.")

    building = next((c for c in result["class_changes"] if c["class"] == "buildings"), None)
    if building is not None:
        if building["direction"] == "unchanged":
            parts.append("Built-up area (buildings) is essentially unchanged.")
        else:
            parts.append(f"Built-up area (buildings) {building['direction']} by {abs(building['net']) * 100:.1f}% of the scene.")
    return " ".join(parts)


class SemanticChangeTool:
    """Wraps the trained Siamese semantic-change network. One instance = one loaded model in
    memory -- construct once per process and re-use."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        # weights_only=True: the dict is tensors + plain builtins only (see the notebook's export cell).
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.img_size = checkpoint["img_size"]
        self.class_names = list(checkpoint["class_names"])
        self.metrics = checkpoint.get("metrics", {})
        # Picked on the validation split by the training notebook (see its threshold sweep).
        self.change_threshold = float(checkpoint.get("change_threshold", 0.5))

        self.model = SiameseSCDNet(
            encoder_name=checkpoint["encoder_name"], encoder_weights=None, num_classes=len(self.class_names)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _prepare(self, image) -> torch.Tensor:
        from PIL import Image

        resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def analyze(self, image1_path: str, image2_path: str, change_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Compare a before/after pair. Both are resized to the training resolution for the forward
        pass (SECOND-CC crops are 256x256) and the predictions are resampled back to image1's
        original size, so fractions/bbox are in the caller's own pixel space.

        `confidence` is a model score, NOT a calibrated probability: the mean of (a) how far the
        change head's probability sits from the decision boundary, rescaled to [0.5, 1], and -- when
        anything changed -- (b) the mean top-class softmax probability of both semantic heads over
        the changed pixels."""
        from PIL import Image

        threshold = self.change_threshold if change_threshold is None else change_threshold
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
        orig_w, orig_h = img1.size
        if img2.size != img1.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)

        with torch.no_grad():
            change_logit, sem_a, sem_b = self.model(self._prepare(img1), self._prepare(img2))
            size = (orig_h, orig_w)
            change_prob = F.interpolate(torch.sigmoid(change_logit), size=size, mode="bilinear", align_corners=False)[0, 0]
            prob_a = F.interpolate(torch.softmax(sem_a, dim=1), size=size, mode="bilinear", align_corners=False)[0]
            prob_b = F.interpolate(torch.softmax(sem_b, dim=1), size=size, mode="bilinear", align_corners=False)[0]

        change_np = change_prob.cpu().numpy()
        change_mask = change_np >= threshold
        top_a, cls_a = prob_a.max(dim=0)
        top_b, cls_b = prob_b.max(dim=0)
        cls_a = cls_a.cpu().numpy().astype(np.uint8)
        cls_b = cls_b.cpu().numpy().astype(np.uint8)

        summary = compute_class_changes(change_mask, cls_a, cls_b, self.class_names)

        change_conf = float(np.mean(0.5 + 0.5 * np.abs(2 * change_np - 1)))
        if change_mask.any():
            sem_conf = float(0.5 * (top_a.cpu().numpy()[change_mask].mean() + top_b.cpu().numpy()[change_mask].mean()))
            confidence = 0.5 * (change_conf + sem_conf)
        else:
            confidence = change_conf

        sem_before = np.where(change_mask, cls_a, IGNORE_INDEX).astype(np.uint8)
        sem_after = np.where(change_mask, cls_b, IGNORE_INDEX).astype(np.uint8)
        return {
            "change_mask": change_mask,
            "change_fraction": summary["change_fraction"],
            "largest_region_bbox": _largest_region_bbox(change_mask),
            "class_changes": summary["class_changes"],
            "transitions": summary["transitions"],
            "sem_before": sem_before,
            "sem_after": sem_after,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        }


def _largest_region_bbox(mask) -> Optional[list]:
    """Bounding box [x1, y1, x2, y2] (exclusive x2/y2) of the largest 4-connected True region, or
    None if empty. Same helper as change_detection_tool.py (each models/*/ script stays standalone)."""
    from scipy import ndimage

    if not mask.any():
        return None
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return None
    sizes = ndimage.sum(mask, labeled, index=range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == largest_label)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def draw_semantic_change_overlay(image1_path: str, image2_path: str, result: Dict[str, Any], output_path: str) -> None:
    """Separate, explicit drawing step -- takes SemanticChangeTool.analyze()'s output and writes a
    single composite: before | after | after with each changed region tinted by the class it
    BECAME, plus a legend of the classes shown. One composite because ToolResult.evidence_image_path
    is a single path."""
    from PIL import Image, ImageDraw

    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)

    sem_after = result["sem_after"]
    overlay = np.asarray(img2, dtype=np.float32).copy()
    alpha = 0.55
    present = []
    for idx, color in enumerate(CLASS_COLORS_RGB[: len(CLASS_NAMES)]):
        sel = sem_after == idx
        if sel.any():
            present.append(idx)
            overlay[sel] = overlay[sel] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    panel3 = Image.fromarray(overlay.astype(np.uint8))

    if result.get("largest_region_bbox"):
        x1, y1, x2, y2 = result["largest_region_bbox"]
        ImageDraw.Draw(panel3).rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 255, 0), width=max(1, img1.width // 128))

    w, h = img1.size
    legend_h = 24 if present else 0
    composite = Image.new("RGB", (w * 3, h + legend_h), (255, 255, 255))
    composite.paste(img1, (0, 0))
    composite.paste(img2, (w, 0))
    composite.paste(panel3, (2 * w, 0))
    if present:
        draw = ImageDraw.Draw(composite)
        x = 6
        for idx in present:
            draw.rectangle([x, h + 6, x + 12, h + 18], fill=CLASS_COLORS_RGB[idx], outline=(0, 0, 0))
            label = f"-> {CLASS_NAMES[idx]}"
            draw.text((x + 16, h + 6), label, fill=(0, 0, 0))
            x += 16 + 7 * len(label) + 12
    composite.save(output_path)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image1", required=True, help="Path to the 'before' image")
    parser.add_argument("--image2", required=True, help="Path to the 'after' image")
    parser.add_argument("--checkpoint", required=True, help="Path to semantic_change_unet.pt")
    parser.add_argument("--draw", default=None, help="If set, write a before/after/overlay composite here")
    args = parser.parse_args()

    for p in (args.image1, args.image2):
        if not Path(p).exists():
            sys.exit(f"Image not found: {p}")

    tool = SemanticChangeTool(checkpoint_path=args.checkpoint)
    result = tool.analyze(args.image1, args.image2)
    print(describe_changes(result))
    print(json.dumps({k: result[k] for k in ("change_fraction", "largest_region_bbox", "class_changes", "transitions", "confidence")}, indent=2))
    print("metrics at training time:", tool.metrics, file=sys.stderr)

    if args.draw:
        draw_semantic_change_overlay(args.image1, args.image2, result, args.draw)
        print(f"\nComposite image written to {args.draw}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
