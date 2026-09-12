"""
water_segmentation_tool.py

Local (Mac-side) wrapper around a segmentation_models_pytorch U-Net fine-tuned for binary
water-body segmentation. Mirrors the shape of ../grounding/grounding_tool.py: exposes exactly the
interface an orchestrator needs -- (image) -> structured mask/water-fraction/confidence. No image
is drawn on here; overlay rendering is a separate, explicit step (see draw_mask below).

Setup (once):
    pip install segmentation-models-pytorch pillow

    # Checkpoint lives at checkpoints/water_body_unet_final.pt next to this script -- a plain
    # torch.save'd dict: {"model_state_dict", "encoder_name", "img_size", "val_iou"}.
    #
    # KNOWN GAP: no training notebook exists for this checkpoint (unlike the grounding tool), so
    # the exact preprocessing used at training time (normalization stats, resize interpolation)
    # is NOT confirmed -- this wrapper uses the standard ImageNet mean/std + bilinear resize to
    # img_size (the convention `segmentation_models_pytorch` + `imagenet`-pretrained encoders
    # expect), which is a reasonable default but unverified against ground truth. Re-validate
    # qualitatively against a real water-containing satellite image once one is available (e.g.
    # after the BigEarthNet/RSVQA data pull), and try alternate normalization if masks look wrong.
    #
    # Synthetic images (flat color blocks, random noise) are NOT a useful stand-in for this check
    # -- already tried; the model's output on them varies with the RNG seed/exact color used and
    # doesn't cleanly separate "land" vs "water" logits, because they're too out-of-distribution
    # for a model trained on real satellite texture. Only real imagery will validate calibration.

Usage as a library (what your orchestrator calls):

    from water_segmentation_tool import WaterSegmentationTool

    tool = WaterSegmentationTool(checkpoint_path="checkpoints/water_body_unet_final.pt")
    result = tool.segment("scene.jpg")
    # -> {"mask": np.ndarray[H,W] bool, "water_fraction": 0.18, "confidence": 0.83}

Usage from the command line (quick sanity check + draws an annotated copy):

    python water_segmentation_tool.py --image scene.jpg \
        --checkpoint checkpoints/water_body_unet_final.pt --draw out.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WaterSegmentationTool:
    """Wraps a (fine-tuned) water-body segmentation U-Net. One instance = one loaded model in
    memory -- construct it once in your orchestrator process and re-use it across calls."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        # Imported lazily so this module can be imported without segmentation_models_pytorch
        # installed yet (e.g. just to get the return-type shape).
        import segmentation_models_pytorch as smp

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.img_size = checkpoint["img_size"]
        self.val_iou = checkpoint.get("val_iou")

        self.model = smp.Unet(
            encoder_name=checkpoint["encoder_name"],
            encoder_weights=None,
            classes=1,
            activation=None,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Run one image through the model.

        Returns:
            {
                "mask": np.ndarray[H, W] bool -- True where water, at the ORIGINAL image resolution,
                "water_fraction": float -- fraction of pixels classified as water,
                "confidence": float -- mean prediction certainty over the image, in [0.5, 1.0],
            }
        No image is generated or modified -- drawing is a separate step (see draw_mask below).
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        probs_full = np.asarray(
            Image.fromarray((probs * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        mask = probs_full >= threshold

        water_fraction = float(mask.mean())
        confidence = float(np.mean(np.where(mask, probs_full, 1.0 - probs_full)))

        return {"mask": mask, "water_fraction": round(water_fraction, 4), "confidence": round(confidence, 4)}


def draw_mask(image_path: str, result: Dict[str, Any], output_path: str, color=(0, 120, 255), alpha: float = 0.45) -> None:
    """Separate, explicit drawing step -- takes the output of WaterSegmentationTool.segment()
    and writes a copy of the image with the water mask overlaid. Doesn't touch the model at all."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    mask = result["mask"]

    overlay = np.asarray(image, dtype=np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1 - alpha) + color_arr * alpha

    Image.fromarray(overlay.astype(np.uint8)).save(output_path)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned .pt checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--draw", default=None, help="If set, write an image with the mask overlaid here")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")

    tool = WaterSegmentationTool(checkpoint_path=args.checkpoint)
    result = tool.segment(args.image, threshold=args.threshold)

    print(json.dumps({
        "water_fraction": result["water_fraction"],
        "confidence": result["confidence"],
        "val_iou_at_training": tool.val_iou,
    }, indent=2))

    if args.draw:
        draw_mask(args.image, result, args.draw)
        print(f"\nOverlay image written to {args.draw}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
