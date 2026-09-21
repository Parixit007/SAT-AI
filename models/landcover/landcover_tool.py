"""
landcover_tool.py

Local (Mac-side) wrapper around the land-cover segmenter trained by
notebooks/kaggle_finetune_landcover_openearthmap.ipynb: a U-Net with a ResNet34 encoder that labels every
pixel as one of eight OpenEarthMap classes -- building, road, tree, water, agriculture land, rangeland
(grass / scrub), developed space (paved lots, plazas, yards) or bareland. From that map it derives what the
app's other specialists cannot give: how much of the scene is roofs / roads / vegetation / water, how many
distinct buildings there are, and what colour the roofs are.

Setup (once): download `landcover_unet.pt` from the Kaggle notebook's Output tab (lc_out/) into
models/landcover/checkpoints/ (gitignored). Same shape as ../water_segmentation: one class, lazy imports,
one inference method, drawing kept separate.

Usage as a library (what the orchestrator calls):

    from landcover_tool import LandCoverTool, draw_overlay

    tool = LandCoverTool()                       # or LandCoverTool("path/to/landcover_unet.pt")
    result = tool.analyze("street.jpg")
    result["fractions"]["building"]              # 0.34  (a share of the image's pixels)
    result["building_count"]                     # 62    (distinct connected footprints -- touching buildings merge)
    result["roof_colors"]                        # [{"name": "white", "share": 0.41, "rgb": [231, 228, 222]}, ...]
    draw_overlay("street.jpg", result["class_map"], "overlay.jpg")

The derived numbers are honest about what they are: the building count is the number of connected regions
of the building mask, so terraces and dense blocks are under-counted; roof colours are read from the image's
own pixels (haze, shadow and the sensor shift them); the confidence is the mean top-class probability, a
model score and not a calibrated chance of being right.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CLASSES = ["bareland", "rangeland", "developed space", "road", "tree", "water", "agriculture land", "building"]
BUILDING = CLASSES.index("building")
# RGB used by draw_overlay (and mirrored by the frontend's legend)
PALETTE = {
    "bareland": (140, 110, 90), "rangeland": (170, 230, 110), "developed space": (160, 160, 160), "road": (255, 235, 0),
    "tree": (0, 110, 40), "water": (0, 120, 255), "agriculture land": (255, 150, 0), "building": (230, 0, 0),
}
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "landcover_unet.pt"
# A connected building region smaller than this is noise, not a building -- the same rule the notebook used
# when it measured the count error, so the reported error applies to the number this returns.
MIN_BUILDING_PIXELS = 30
WHOLE_IMAGE_LIMIT = 1536  # up to this many pixels on the longer side the network sees the whole image at once
TILE, OVERLAP = 768, 128  # beyond it: overlapping tiles, class probabilities averaged where they overlap


# --------------------------------------------------------------------------- pure functions

def class_fractions(class_map: np.ndarray) -> Dict[str, float]:
    """Share of the image's pixels in each class (sums to 1)."""
    counts = np.bincount(class_map.ravel(), minlength=len(CLASSES))[: len(CLASSES)]
    total = max(int(counts.sum()), 1)
    return {name: float(counts[i]) / total for i, name in enumerate(CLASSES)}


def count_components(mask: np.ndarray, min_pixels: int = MIN_BUILDING_PIXELS) -> int:
    """Connected regions (8-neighbourhood) of a boolean mask with at least `min_pixels` pixels."""
    from scipy import ndimage

    labelled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    sizes = np.bincount(labelled.ravel())[1:]
    return int((sizes >= min_pixels).sum())


COLOR_NAMES = [
    "white", "light grey", "grey", "dark grey", "very dark",
    "red", "orange", "brown", "tan", "yellow", "green", "blue", "purple",
]


def color_name_indices(rgb: np.ndarray) -> np.ndarray:
    """Plain colour name (index into COLOR_NAMES) for each RGB row (uint8, shape (N, 3)), by simple HSV rules:
    low saturation -> a grey level by brightness; otherwise by hue, with brown/tan for dark or washed-out
    oranges and yellows (most roofs and bare ground). Deterministic and dependency-free on purpose."""
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(1), x.min(1)
    v, delta = mx, mx - mn
    s = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    safe = np.maximum(delta, 1e-6)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    h = np.where(delta < 1e-6, 0.0, h)

    idx = np.full(len(x), COLOR_NAMES.index("grey"))
    ci = COLOR_NAMES.index
    grey = s < 0.16
    idx = np.where(grey & (v >= 0.80), ci("white"), idx)
    idx = np.where(grey & (v >= 0.62) & (v < 0.80), ci("light grey"), idx)
    idx = np.where(grey & (v >= 0.38) & (v < 0.62), ci("grey"), idx)
    idx = np.where(grey & (v >= 0.20) & (v < 0.38), ci("dark grey"), idx)
    idx = np.where(v < 0.20, ci("very dark"), idx)  # includes shadow, which is indistinguishable from a black roof

    chroma = ~grey & (v >= 0.20)
    idx = np.where(chroma & ((h < 14) | (h >= 345)), np.where(v < 0.45, ci("brown"), ci("red")), idx)
    idx = np.where(chroma & (h >= 14) & (h < 45), np.where((v < 0.50) | (s < 0.38), np.where(v < 0.50, ci("brown"), ci("tan")), ci("orange")), idx)
    idx = np.where(chroma & (h >= 45) & (h < 70), np.where(s < 0.40, ci("tan"), ci("yellow")), idx)
    idx = np.where(chroma & (h >= 70) & (h < 170), ci("green"), idx)
    idx = np.where(chroma & (h >= 170) & (h < 260), ci("blue"), idx)
    idx = np.where(chroma & (h >= 260) & (h < 345), ci("purple"), idx)
    return idx


# Roofs are reported by colour FAMILY: the fine names split one visual impression into pieces (a grey-and-white
# district measured 37% grey, 20% light grey, 15% white, 6% dark grey), and the boundaries between them are set by
# lighting, not by the roofs.
COLOR_FAMILY = {
    "white": "white or light grey", "light grey": "white or light grey", "grey": "grey",
    "dark grey": "dark grey or black", "very dark": "dark grey or black",
    "red": "red or orange", "orange": "red or orange", "brown": "brown or tan", "tan": "brown or tan",
    "yellow": "yellow", "green": "green", "blue": "blue", "purple": "purple",
}
FAMILY_NAMES = list(dict.fromkeys(COLOR_FAMILY[n] for n in COLOR_NAMES))


def name_color(rgb) -> str:
    """The plain name of one RGB colour."""
    return COLOR_NAMES[int(color_name_indices(np.array([rgb], dtype=np.uint8))[0])]


def roof_colors(image: np.ndarray, building_mask: np.ndarray, min_share: float = 0.05, max_samples: int = 40000) -> List[Dict[str, Any]]:
    """The roof colours of the building mask: the share of building pixels in each colour FAMILY (largest first,
    only shares of at least `min_share`), with the mean RGB of those pixels for display. The mask is eroded by one
    pixel first so the mixed pixels at roof edges do not vote."""
    from scipy import ndimage

    mask = building_mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    if int(eroded.sum()) >= 50:
        mask = eroded
    pixels = image[mask]
    if len(pixels) < 50:
        return []
    if len(pixels) > max_samples:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), max_samples, replace=False)]
    family_of = np.array([FAMILY_NAMES.index(COLOR_FAMILY[name]) for name in COLOR_NAMES])
    idx = family_of[color_name_indices(pixels)]
    out = []
    for i in np.argsort(-np.bincount(idx, minlength=len(FAMILY_NAMES))):
        share = float((idx == i).sum()) / len(idx)
        if share <= 0 or share < min_share:  # families with no pixels at all must never be listed (their mean is undefined)
            break
        out.append({"name": FAMILY_NAMES[int(i)], "share": round(share, 4), "rgb": [int(v) for v in pixels[idx == i].mean(0).round()]})
    return out


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    """Start offsets of tiles of size `tile` that cover [0, length) with at least `overlap` pixels shared."""
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


# --------------------------------------------------------------------------- the model

class LandCoverTool:
    """One instance = one loaded model (~100MB) -- construct it once and reuse it."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        import segmentation_models_pytorch as smp
        import torch

        path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not path.exists():
            raise RuntimeError(
                f"No land-cover checkpoint at {path}. Download landcover_unet.pt from the Kaggle notebook's "
                "Output tab (lc_out/) -- see this file's docstring."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if list(checkpoint["classes"]) != CLASSES:
            raise RuntimeError(f"The checkpoint's classes {checkpoint['classes']} do not match this tool's {CLASSES}.")
        self.metrics = checkpoint.get("metrics", {})
        self._mean = torch.tensor(checkpoint["mean"], dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(checkpoint["std"], dtype=torch.float32)[:, None, None]

        self.model = smp.Unet(encoder_name=checkpoint["encoder_name"], encoder_weights=None, classes=checkpoint["num_classes"], activation=None)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # CPU by default: the captioner already uses the GPU (MPS) and two models on it at once is not safe;
        # a 1024px image takes a few seconds on the CPU, in parallel with the other specialists.
        self.device = device or "cpu"
        self.model = self.model.to(self.device).eval()
        self._torch = torch

    def _probabilities(self, image: np.ndarray) -> np.ndarray:
        """(classes, H, W) softmax probabilities for a whole RGB image, tiled when it is large."""
        torch, F = self._torch, self._torch.nn.functional
        h, w = image.shape[:2]
        x = (torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 - self._mean) / self._std

        def run(patch):  # (3, ph, pw) -> (classes, ph, pw)
            ph, pw = patch.shape[-2:]
            padded = F.pad(patch[None], (0, (-pw) % 32, 0, (-ph) % 32), mode="reflect")
            with torch.inference_mode():
                logits = self.model(padded.to(self.device))
            return F.softmax(logits.float(), dim=1)[0, :, :ph, :pw].cpu()

        if max(h, w) <= WHOLE_IMAGE_LIMIT:
            return run(x).numpy()
        total = torch.zeros(len(CLASSES), h, w)
        weight = torch.zeros(1, h, w)
        for y0 in tile_starts(h, TILE, OVERLAP):
            for x0 in tile_starts(w, TILE, OVERLAP):
                patch = x[:, y0:y0 + TILE, x0:x0 + TILE]
                total[:, y0:y0 + patch.shape[1], x0:x0 + patch.shape[2]] += run(patch)
                weight[:, y0:y0 + patch.shape[1], x0:x0 + patch.shape[2]] += 1
        return (total / weight).numpy()

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Label every pixel and derive the scene's numbers. `class_map` is (H, W) uint8 indexing CLASSES."""
        from PIL import Image

        image = np.array(Image.open(image_path).convert("RGB"))
        probabilities = self._probabilities(image)
        class_map = probabilities.argmax(0).astype(np.uint8)
        building_mask = class_map == BUILDING
        return {
            "class_map": class_map,
            "fractions": class_fractions(class_map),
            "building_count": count_components(building_mask),
            "roof_colors": roof_colors(image, building_mask),
            "confidence": float(probabilities.max(0).mean()),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
        }


def draw_overlay(image_path: str, class_map: np.ndarray, out_path: str, alpha: float = 0.55) -> None:
    """The class map blended over the image, with a legend strip of the classes actually present."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    if image.size != (class_map.shape[1], class_map.shape[0]):
        image = image.resize((class_map.shape[1], class_map.shape[0]))
    color = np.zeros(class_map.shape + (3,), dtype=np.uint8)
    for i, name in enumerate(CLASSES):
        color[class_map == i] = PALETTE[name]
    blended = Image.blend(image, Image.fromarray(color), alpha)

    present = [(n, PALETTE[n], float((class_map == i).mean())) for i, n in enumerate(CLASSES) if (class_map == i).any()]
    present.sort(key=lambda t: -t[2])
    font = ImageFont.load_default()
    strip_h = 22
    canvas = Image.new("RGB", (blended.width, blended.height + strip_h), (24, 24, 24))
    canvas.paste(blended, (0, 0))
    draw, x = ImageDraw.Draw(canvas), 6
    for name, rgb, share in present:
        label = f"{name} {share:.0%}"
        draw.rectangle([x, blended.height + 5, x + 11, blended.height + 16], fill=rgb)
        draw.text((x + 15, blended.height + 5), label, fill=(235, 235, 235), font=font)
        x += 15 + int(draw.textlength(label, font=font)) + 12
    canvas.save(out_path, quality=90)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--overlay", help="Write the class overlay image to this path")
    args = parser.parse_args()
    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")
    result = LandCoverTool(args.checkpoint).analyze(args.image)
    if args.overlay:
        draw_overlay(args.image, result["class_map"], args.overlay)
    print(json.dumps({k: v for k, v in result.items() if k != "class_map"}, indent=2))


if __name__ == "__main__":
    _cli()
