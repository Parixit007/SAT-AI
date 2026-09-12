"""Downloads RSVQA (Lobry et al.) -- eval dataset for single-image VQA.

RSVQA-LR (772 images, ~0.15GB): downloaded in full, per CLAUDE.md's data-acquisition plan.
RSVQA-HR (10,659 images, ~1.07M QA pairs, image archive alone is ~13.5GB): only the *test*-split
JSON annotation files are downloaded by default (~220MB) -- pass --hr-images to also pull the full
Images.tar (13.5GB); you almost certainly don't want that on a laptop.

Sources (verified against the live Zenodo API before writing this script):
    RSVQA-LR: https://zenodo.org/records/6344333 (DOI 10.5281/zenodo.6344333)
    RSVQA-HR: https://zenodo.org/records/6344366 (DOI 10.5281/zenodo.6344366)

Usage:
    python data/scripts/download_rsvqa.py
    python data/scripts/download_rsvqa.py --hr-images   # also pull the 13.5GB HR image archive
"""

import argparse
from pathlib import Path

from common import download_url, print_summary, zenodo_record_files

LR_RECORD = "6344333"
HR_RECORD = "6344366"

DEST = Path(__file__).resolve().parents[1] / "raw" / "rsvqa"


def download_lr() -> None:
    dest = DEST / "LR"
    for f in zenodo_record_files(LR_RECORD):
        download_url(f["links"]["self"], dest / f["key"], expected_min_bytes=max(1, f.get("size", 1024) - 1))
    print_summary(dest)


def download_hr(include_images: bool) -> None:
    dest = DEST / "HR"
    for f in zenodo_record_files(HR_RECORD):
        key = f["key"]
        is_test_split = "test" in key
        is_full_image_archive = key == "Images.tar"
        if is_full_image_archive and not include_images:
            print(f"[skip] {key} (13.5GB full image archive -- pass --hr-images to include it)")
            continue
        if not is_test_split and not is_full_image_archive:
            print(f"[skip] {key} (not part of the test split)")
            continue
        download_url(f["links"]["self"], dest / key, expected_min_bytes=max(1, f.get("size", 1024) - 1))
    print_summary(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hr-images", action="store_true", help="also download the 13.5GB HR Images.tar")
    args = parser.parse_args()

    download_lr()
    download_hr(include_images=args.hr_images)
