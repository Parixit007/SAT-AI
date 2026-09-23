"""
fusion_classifier.py

Local (Mac-side) wrapper for the optical-SAR fusion Stage 2 model: a small early-fusion CNN that
looks at a co-registered optical + SAR patch *together* and predicts which of four broad land-cover
types it shows (agricultural, barren, grassland, urban) -- see
notebooks/kaggle_finetune_fusion_sen12.ipynb for how it was trained and measured against
single-modality baselines, and CLAUDE.md for the real numbers.

This is a genuinely different thing from fusion_tool.py's Stage 1: Stage 1 is training-free
SAR-backscatter physics (recursive Otsu thresholding, no model at all); this is a trained model that
sees both modalities at once. It gives a second, learned cross-modal signal -- most usefully,
P(urban) as an actual optical+SAR-informed built-up estimate, which is the cross-check
backend/app/specialists/fusion_adapter.py's own docstring has said was missing ("Built-up detection
is SAR-only for now -- no optical cross-check exists yet").

Architecture: ResNet18 (torchvision) whose first conv is expanded from 3 to 4 input channels
(optical RGB + one SAR channel stacked) -- the pretrained RGB weights are kept in channels 0-2, the
SAR channel is initialized as their mean, a standard transfer trick. At inference time no ImageNet
download happens at all -- the checkpoint carries the full trained state dict, architecture is
rebuilt from scratch (random weights) and immediately overwritten by `load_state_dict`.

Setup (once):
    pip install torch torchvision pillow numpy

Usage as a library:
    from fusion_classifier import FusionClassifier
    tool = FusionClassifier()
    tool.classify("optical.png", "sar.png")
    # -> {"land_cover_class": "urban", "probabilities": {"agri": 0.02, "barrenland": 0.01,
    #     "grassland": 0.05, "urban": 0.92}, "confidence": 0.92}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "fusion_classifier.pt"


class FusionClassifier:
    """One instance = one loaded model (~45MB) -- construct it once and reuse it."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        import torch
        import torchvision
        import torch.nn as nn

        path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not path.exists():
            raise RuntimeError(
                f"No fusion-classifier checkpoint at {path}. Download fusion_classifier.pt from the "
                "Kaggle notebook's Output tab (fusion_out/) -- see this file's docstring."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = list(checkpoint["classes"])
        self.crop = int(checkpoint["crop"])
        self.imagenet_mean = torch.tensor(checkpoint["imagenet_mean"])[:, None, None]
        self.imagenet_std = torch.tensor(checkpoint["imagenet_std"])[:, None, None]
        self.sar_mean = torch.tensor([checkpoint["sar_mean"]])[:, None, None]
        self.sar_std = torch.tensor([checkpoint["sar_std"]])[:, None, None]
        self.metrics = checkpoint.get("metrics", {})

        model = torchvision.models.resnet18(weights=None)
        in_channels = int(checkpoint.get("in_channels", 4))
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state_dict"])

        self.device = device or "cpu"
        self.model = model.to(self.device).eval()
        self._torch = torch
        self._F = torch.nn.functional

    def classify(self, optical_path: str, sar_path: str) -> Dict[str, Any]:
        """Classify a co-registered optical + SAR image pair into one of `self.classes`.

        Returns {"land_cover_class", "probabilities": {class: prob}, "confidence"}. `confidence` is
        the top-class softmax probability -- a model score, not a calibrated chance of being right."""
        import numpy as np
        from PIL import Image

        torch, F = self._torch, self._F
        optical = np.array(Image.open(optical_path).convert("RGB"), dtype=np.uint8)
        sar = np.array(Image.open(sar_path).convert("L"), dtype=np.uint8)

        opt = torch.from_numpy(optical).permute(2, 0, 1).float() / 255.0
        sr = torch.from_numpy(sar).float()[None] / 255.0
        opt = F.interpolate(opt[None], size=(self.crop, self.crop), mode="bilinear", align_corners=False)[0]
        sr = F.interpolate(sr[None], size=(self.crop, self.crop), mode="bilinear", align_corners=False)[0]
        opt = (opt - self.imagenet_mean) / self.imagenet_std
        sr = (sr - self.sar_mean) / self.sar_std
        x = torch.cat([opt, sr], dim=0)[None].to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(x).float(), dim=1)[0].cpu().numpy()

        top = int(probs.argmax())
        return {
            "land_cover_class": self.classes[top],
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "confidence": round(float(probs[top]), 4),
        }


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optical", required=True, help="Path to the optical image")
    parser.add_argument("--sar", required=True, help="Path to the SAR image")
    parser.add_argument("--checkpoint", default=None, help="Override the default checkpoint path")
    args = parser.parse_args()

    for label, p in (("optical", args.optical), ("sar", args.sar)):
        if not Path(p).exists():
            sys.exit(f"{label} image not found: {p}")

    tool = FusionClassifier(args.checkpoint)
    print(json.dumps(tool.classify(args.optical, args.sar), indent=2))


if __name__ == "__main__":
    _cli()
