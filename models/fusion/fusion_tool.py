"""
fusion_tool.py

Local (Mac-side) wrapper for optical-SAR cross-modal fusion: given one optical image and one
co-registered SAR image of the SAME place at (roughly) the SAME time, extract information neither
modality gives alone.

Stage 1 (see the roadmap plan) is training-free, built on well-established SAR backscatter
physics: water is smooth and reflects away from the sensor (very low backscatter -- the standard
basis for real operational SAR flood mapping), while buildings produce strong double-bounce/
corner-reflector returns (high backscatter). Both are separable with the same adaptive (Otsu)
thresholding models/change_detection/change_detection_tool.py already uses for a different
purpose -- duplicated here (not imported) since each models/*/ script is standalone (see
CLAUDE.md), applied recursively: one Otsu split finds the broad low/high divide, then a second
Otsu pass within each half finds the water threshold (very-low side) and built-up threshold
(very-high side) -- a standard multilevel/recursive-Otsu technique, not an ad hoc heuristic.

This script only computes the SAR-side read (`analyze()` below) -- it has no way to also produce
an optical water read without duplicating models/water_segmentation/water_segmentation_tool.py
(exactly what the standalone-scripts convention says not to do). The genuinely "fusion" part --
reconciling this SAR read against the existing optical water_segmentation tool's own read -- lives
in backend/app/specialists/fusion_adapter.py, which can import from both.

Setup (once):
    pip install numpy pillow scipy

Usage as a library:
    from fusion_tool import FusionTool
    tool = FusionTool()
    tool.analyze("sar.png")
    # -> {"water_mask": np.ndarray[H,W] bool, "water_fraction": 0.15,
    #     "builtup_mask": np.ndarray[H,W] bool, "builtup_fraction": 0.22,
    #     "confidence": 0.71}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


class FusionTool:
    """SAR-backscatter water/built-up detector. No model to load -- construction is cheap; kept
    as a class (rather than a bare function) to match every other specialist's shape."""

    def __init__(self):
        pass

    def analyze(self, sar_path: str) -> Dict[str, Any]:
        """Analyze one SAR image for water (very low backscatter) and built-up (very high
        backscatter) candidate regions.

        Returns {"water_mask", "water_fraction", "builtup_mask", "builtup_fraction", "confidence"}.
        `confidence` averages the two recursive-Otsu separation strengths (same heuristic meaning
        as change_detection_tool's -- how cleanly each split's histogram separates into two
        classes, NOT a calibrated probability).
        """
        import numpy as np
        from PIL import Image

        img = Image.open(sar_path).convert("L")  # SAR is inherently single-intensity-channel
        arr = np.asarray(img, dtype=np.float32)

        water_mask, builtup_mask, water_conf, builtup_conf = _detect_water_and_builtup(arr)

        return {
            "water_mask": water_mask,
            "water_fraction": round(float(water_mask.mean()), 4),
            "builtup_mask": builtup_mask,
            "builtup_fraction": round(float(builtup_mask.mean()), 4),
            "confidence": round((water_conf + builtup_conf) / 2, 4),
        }


def _detect_water_and_builtup(sar_gray):
    """Recursive (two-level) Otsu thresholding: one pass splits the whole image into a low-
    backscatter half and a high-backscatter half; a second pass *within* the low half finds the
    water threshold, and within the high half finds the built-up threshold. Adaptive per-image,
    no manual/fixed dB cutoffs."""
    import numpy as np

    global_thresh, _ = _otsu_threshold(sar_gray)
    low_side = sar_gray[sar_gray <= global_thresh]
    high_side = sar_gray[sar_gray > global_thresh]

    water_thresh, water_conf = _otsu_threshold(low_side) if low_side.size > 1 else (float(global_thresh), 0.0)
    builtup_thresh, builtup_conf = (
        _otsu_threshold(high_side) if high_side.size > 1 else (float(global_thresh), 0.0)
    )

    water_mask = sar_gray <= water_thresh
    builtup_mask = sar_gray >= builtup_thresh
    return water_mask, builtup_mask, water_conf, builtup_conf


def _otsu_threshold(values) -> tuple:
    """Otsu's method on `values`' histogram -- picks the threshold maximizing between-class
    variance. Returns (threshold, normalized_between_class_variance), the second used as a
    heuristic confidence signal. Identical approach to
    models/change_detection/change_detection_tool.py's own `_otsu_threshold` (duplicated, not
    imported -- see module docstring)."""
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    hist, bin_edges = np.histogram(values, bins=256, range=(0, max(float(values.max()), 1e-6)))
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

    overall_mean = float(grand_total / total)
    overall_variance = float(np.sum(hist * (bin_centers - overall_mean) ** 2) / total)
    confidence = (
        0.0 if overall_variance <= 0 else min(1.0, float(between_class_variance[best_idx]) / overall_variance)
    )

    return threshold, confidence


def draw_fusion_overlay(sar_path: str, result: Dict[str, Any], output_path: str) -> None:
    """Separate, explicit drawing step -- writes a single composite: the SAR image with water
    (blue) and built-up (red) masks overlaid side by side with the original. One image because
    ToolResult.evidence_image_path (tool_registry.py) is a single path, not a list."""
    import numpy as np
    from PIL import Image

    sar_img = Image.open(sar_path).convert("RGB")
    overlay = np.asarray(sar_img, dtype=np.float32).copy()

    alpha = 0.5
    blue = np.array([40, 120, 255], dtype=np.float32)
    red = np.array([255, 60, 60], dtype=np.float32)
    overlay[result["water_mask"]] = overlay[result["water_mask"]] * (1 - alpha) + blue * alpha
    overlay[result["builtup_mask"]] = overlay[result["builtup_mask"]] * (1 - alpha) + red * alpha
    annotated = Image.fromarray(overlay.astype(np.uint8))

    composite = Image.new("RGB", (sar_img.width * 2, sar_img.height))
    composite.paste(sar_img, (0, 0))
    composite.paste(annotated, (sar_img.width, 0))
    composite.save(output_path)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sar", required=True, help="Path to the SAR image")
    parser.add_argument("--draw", default=None, help="If set, write a SAR/overlay composite here")
    args = parser.parse_args()

    if not Path(args.sar).exists():
        sys.exit(f"Image not found: {args.sar}")

    tool = FusionTool()
    result = tool.analyze(args.sar)

    print(json.dumps({
        "water_fraction": result["water_fraction"],
        "builtup_fraction": result["builtup_fraction"],
        "confidence": result["confidence"],
    }, indent=2))

    if args.draw:
        draw_fusion_overlay(args.sar, result, args.draw)
        print(f"\nComposite image written to {args.draw}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
