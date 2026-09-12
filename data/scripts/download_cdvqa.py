"""Downloads CDVQA (Yuan et al., "Change Detection Meets Visual Question Answering") -- the
mandatory bi-temporal change-VQA eval dataset.

Source: https://github.com/YZHJessica/CDVQA (verified live -- Apache-2.0, all annotation JSONs
hosted directly in the repo, not Google Drive, so this is simpler than CLAUDE.md's original
gdown-based plan).

IMPORTANT GAP: this repo only contains the question/answer JSON annotations (Train/Val/Test/Test2
splits), NOT the actual pre/post image pairs. CDVQA is built on the public subset of the "SECOND"
semantic change detection dataset, which is hosted separately and hasn't been located/verified yet
-- downloading pixels for this dataset is a follow-up, not done by this script. See the printed
warning at the end of this script for what's still missing.

Usage:
    python data/scripts/download_cdvqa.py
"""

from pathlib import Path

from common import download_url, print_summary

RAW_BASE = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
FILES = [
    "Train_questions.json", "Train_answers.json", "Train_images.json",
    "Val_questions.json", "Val_answers.json", "Val_images.json",
    "Test_questions.json", "Test_answers.json", "Test_images.json",
    "Test2_questions.json", "Test2_answers.json", "Test2_images.json",
]

DEST = Path(__file__).resolve().parents[1] / "raw" / "cdvqa"


def main() -> None:
    for filename in FILES:
        download_url(f"{RAW_BASE}/{filename}", DEST / filename, expected_min_bytes=1)
    print_summary(DEST)
    print(
        "\n[NOTE] Only Q/A annotations were downloaded. The actual pre/post satellite image pairs "
        "come from the 'SECOND' change-detection dataset and are NOT hosted in the CDVQA repo -- "
        "locating and downloading those images is still an open follow-up.",
    )


if __name__ == "__main__":
    main()
