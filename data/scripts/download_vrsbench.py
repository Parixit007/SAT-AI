"""Downloads VRSBench (xiang709/VRSBench on Hugging Face) -- eval dataset for captioning,
grounding, and VQA.

VRSBench does not expose individual images as separate repo files -- they're bundled into one zip
per split. Verified file sizes on the actual HF repo (not the ~11.5GB "whole dataset" figure from
CLAUDE.md's original estimate -- that number was for train+val combined):
    VRSBench_EVAL_Cap.json         4.8MB   (eval captioning annotations)
    VRSBench_EVAL_referring.json  10.3MB   (eval grounding annotations)
    VRSBench_EVAL_vqa.json         9.4MB   (eval VQA annotations)
    Annotations_val.zip           12.8MB   (per-image annotation detail for the val/eval split)
    VRSBench_train.json           65.0MB   (train-split annotations, LLaVA-style conversations --
                                             used by notebooks/kaggle_finetune_grounding_dino_v2_continued.ipynb
                                             to add VRSBench's referring data to grounding training)
    Images_val.zip                 4.0GB   (all eval-split images -- bigger than planned; opt-in)
    Images_train.zip               8.4GB   (train-split images -- pulled inside the Kaggle notebook, not here)

Default run pulls the small annotation JSONs + Annotations_val.zip + VRSBench_train.json (~105MB
total) -- enough to build/inspect eval question sets and to know exactly what training data exists
before running the Kaggle notebook (which fetches Images_train.zip itself). Pass --images to also
pull the 4GB Images_val.zip.

Usage:
    python data/scripts/download_vrsbench.py
    python data/scripts/download_vrsbench.py --images
"""

import argparse
from pathlib import Path

from common import print_summary

REPO_ID = "xiang709/VRSBench"
DEST = Path(__file__).resolve().parents[1] / "raw" / "vrsbench"

ALWAYS = [
    "VRSBench_EVAL_Cap.json",
    "VRSBench_EVAL_referring.json",
    "VRSBench_EVAL_vqa.json",
    "Annotations_val.zip",
    "VRSBench_train.json",
]
IMAGES = ["Images_val.zip"]


def main(include_images: bool) -> None:
    from huggingface_hub import hf_hub_download

    files = ALWAYS + (IMAGES if include_images else [])
    if not include_images:
        print("[skip] Images_val.zip (4.0GB) -- pass --images to include it")

    DEST.mkdir(parents=True, exist_ok=True)
    for filename in files:
        print(f"[hf] {REPO_ID}/{filename} -> {DEST}")
        try:
            hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=str(DEST))
        except Exception as exc:
            print(f"[error] failed to fetch {filename}: {exc}")
            print(f"[fallback] browse manually at https://huggingface.co/datasets/{REPO_ID}")

    print_summary(DEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", action="store_true", help="also download the 4GB Images_val.zip")
    args = parser.parse_args()
    main(include_images=args.images)
