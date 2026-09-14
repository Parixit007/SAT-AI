"""
change_detection_tool.py

Local (Mac-side) wrapper for training-free bi-temporal change detection between two co-registered
optical images. Mirrors the shape of ../water_segmentation/water_segmentation_tool.py: one class,
one inference method, a separate top-level draw_*() function for visualization.

This is Stage 1 (see the roadmap plan) -- classic pixel-differencing, no model/checkpoint, so it
needs no training and works today. Stage 2
(notebooks/kaggle_finetune_change_segmentation_second.ipynb, trained on SECOND-CC) will add
semantic-class-aware change descriptions ("building area increased") -- this stage can only say
THAT something changed and roughly WHERE, not WHICH land-cover class changed.

Setup (once):
    pip install numpy pillow scipy   # scipy for connected-component labeling only

Usage as a library (what your orchestrator calls):

    from change_detection_tool import ChangeDetectionTool

    tool = ChangeDetectionTool()
    result = tool.detect("before.png", "after.png")
    # -> {"change_mask": np.ndarray[H,W] bool, "change_fraction": 0.12,
    #     "largest_region_bbox": [x1,y1,x2,y2] | None, "confidence": 0.71}

Usage from the command line (quick sanity check + draws an annotated copy):

    python change_detection_tool.py --image1 before.png --image2 after.png --draw out.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class ChangeDetectionTool:
    """Training-free bi-temporal change detector: absolute pixel difference between two
    (assumed co-registered) images, Otsu-thresholded, largest connected changed region reported
    as a bounding box. No model to load -- construction is cheap; kept as a class (rather than a
    bare function) to match every other specialist's shape."""

    def __init__(self):
        pass

    def detect(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """Compare two images acquired at different times. Resizes image2 to image1's size if they
        differ (input_validation.py already warns on a size mismatch beyond its own tolerance --
        this just makes the comparison possible rather than crashing on a shape mismatch).

        Returns {"change_mask", "change_fraction", "largest_region_bbox", "confidence"}.
        `confidence` is Otsu's between-class variance ratio (how cleanly the difference separates
        into "changed"/"unchanged"), a heuristic signal, NOT a calibrated probability -- documented
        honestly, same as the groundwater/VQA tools' own confidence caveats.
        """
        import numpy as np
        from PIL import Image

        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
        if img2.size != img1.size:
            img2 = img2.resize(img1.size, Image.BILINEAR)

        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)

        # Grayscale (per-pixel channel-mean) difference -- robust to per-channel noise without
        # needing separate per-channel thresholds.
        diff = np.abs(arr1 - arr2).mean(axis=2)

        threshold, confidence = _otsu_threshold(diff)
        change_mask = diff >= threshold
        change_fraction = float(change_mask.mean())

        largest_region_bbox = _largest_region_bbox(change_mask)

        return {
            "change_mask": change_mask,
            "change_fraction": round(change_fraction, 4),
            "largest_region_bbox": largest_region_bbox,
            "confidence": round(confidence, 4),
        }


def _otsu_threshold(diff) -> tuple:
    """Otsu's method on the difference image's histogram -- picks the threshold that maximizes
    between-class variance, so no image-specific manual tuning is needed. Returns
    (threshold, normalized_between_class_variance) -- the second value is used as `confidence`:
    a well-separated bimodal histogram (a real, localized change) scores high; a diffuse,
    unimodal histogram (sensor noise / illumination drift, no real change) scores low."""
    import numpy as np

    hist, bin_edges = np.histogram(diff, bins=256, range=(0, max(float(diff.max()), 1e-6)))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.0, 0.0

    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    cum_hist_x = np.cumsum(hist * bin_centers)
    grand_total = cum_hist_x[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = cum_hist_x / weight1
        mean2 = (grand_total - cum_hist_x) / weight2
        between_class_variance = weight1 * weight2 * (mean1 - mean2) ** 2
    between_class_variance = np.nan_to_num(between_class_variance, nan=0.0, posinf=0.0, neginf=0.0)

    best_idx = int(np.argmax(between_class_variance))
    threshold = float(bin_centers[best_idx])

    # Normalize the winning variance by the histogram's overall variance so `confidence` lands
    # roughly in [0, 1] regardless of image contrast/scale.
    overall_mean = float(grand_total / total)
    overall_variance = float(np.sum(hist * (bin_centers - overall_mean) ** 2) / total)
    confidence = (
        0.0 if overall_variance <= 0 else min(1.0, float(between_class_variance[best_idx]) / overall_variance)
    )

    return threshold, confidence


def _largest_region_bbox(mask) -> Optional[list]:
    """Bounding box [x1, y1, x2, y2] (exclusive on x2/y2, like slice indexing) of the largest
    4-connected True region in `mask`, or None if the mask is empty."""
    import numpy as np
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


def draw_change_overlay(image1_path: str, image2_path: str, result: Dict[str, Any], output_path: str) -> None:
    """Separate, explicit drawing step -- takes ChangeDetectionTool.detect()'s output and writes a
    single side-by-side composite: image1 | image2-with-the-change-mask-and-bbox-overlaid. One
    composite image because ToolResult.evidence_image_path (tool_registry.py) is a single path,
    not a list. Doesn't touch the model/algorithm at all."""
    import numpy as np
    from PIL import Image, ImageDraw

    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)

    mask = result["change_mask"]
    overlay = np.asarray(img2, dtype=np.float32).copy()
    color = np.array([255, 60, 60], dtype=np.float32)  # red, matches the project's --alert token intent
    alpha = 0.45
    overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    img2_annotated = Image.fromarray(overlay.astype(np.uint8))

    if result.get("largest_region_bbox"):
        x1, y1, x2, y2 = result["largest_region_bbox"]
        draw = ImageDraw.Draw(img2_annotated)
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 255, 0), width=max(1, img1.width // 128))

    composite = Image.new("RGB", (img1.width + img2_annotated.width, max(img1.height, img2_annotated.height)))
    composite.paste(img1, (0, 0))
    composite.paste(img2_annotated, (img1.width, 0))
    composite.save(output_path)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image1", required=True, help="Path to the 'before' image")
    parser.add_argument("--image2", required=True, help="Path to the 'after' image")
    parser.add_argument("--draw", default=None, help="If set, write a before/after+overlay composite here")
    args = parser.parse_args()

    if not Path(args.image1).exists():
        sys.exit(f"Image not found: {args.image1}")
    if not Path(args.image2).exists():
        sys.exit(f"Image not found: {args.image2}")

    tool = ChangeDetectionTool()
    result = tool.detect(args.image1, args.image2)

    print(json.dumps({
        "change_fraction": result["change_fraction"],
        "largest_region_bbox": result["largest_region_bbox"],
        "confidence": result["confidence"],
    }, indent=2))

    if args.draw:
        draw_change_overlay(args.image1, args.image2, result, args.draw)
        print(f"\nComposite image written to {args.draw}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
