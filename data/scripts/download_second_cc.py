"""Downloads SECOND-CC (Zenodo 16937571, CC-BY-4.0) -- the bi-temporal semantic-change dataset the
Stage 2 change-detection model (notebooks/kaggle_finetune_change_segmentation_second.ipynb) trains
on: 6,041 original 256x256 optical pairs (+4,814 offline-augmented copies in the "AUG" variant),
per-timestep semantic maps, and 5 human change captions per pair.

One 2.5GB zip (`SECOND-CC-AUG.zip`) -- verified against the live Zenodo API before writing this
script. Zenodo rate-limits many small HTTP range requests (a first attempt to inspect the zip
remotely with 6 parallel `remotezip` connections got a 429) and then dropped a plain streaming
download after only 7.5MB, so this uses `common.download_url_resumable` (Range-request resume with
backoff, `.part` kept between attempts) and then checks the result against the md5 Zenodo
publishes for it.

Layout inside the zip (verified by listing its central directory):
    SECOND-CC-AUG/SECOND-CC-AUG.json        captions + split + filename for all 10,855 entries
    SECOND-CC-AUG/{train,val,test}/rgb/{A,B}/<name>.png   optical images, 256x256 RGB (A=before, B=after)
    SECOND-CC-AUG/{train,val,test}/sem/{A,B}/<name>.png   semantic maps, RGB color-coded (see notebook)

Usage:
    python data/scripts/download_second_cc.py              # download + md5-verify the zip
    python data/scripts/download_second_cc.py --extract    # also unzip into data/raw/second_cc/
"""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

from common import download_url_resumable, human_size, print_summary, zenodo_record_files

RECORD = "16937571"
DEST = Path(__file__).resolve().parents[1] / "raw" / "second_cc"
EXPECTED_ENTRIES = 43443  # zip entries, from the central directory listing (dirs + files)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify() -> bool:
    ok = True
    for f in zenodo_record_files(RECORD):
        path = DEST / f["key"]
        if not download_url_resumable(f["links"]["self"], path, expected_size=f["size"]):
            ok = False
            continue
        published = (f.get("checksum") or "").removeprefix("md5:")
        if published:
            actual = md5_of(path)
            if actual != published:
                print(f"[error] {path.name}: md5 {actual} != published {published} -- deleting the corrupt file", file=sys.stderr)
                path.unlink()
                ok = False
            else:
                print(f"[ok] {path.name}: md5 matches Zenodo's published checksum ({human_size(path.stat().st_size)})", file=sys.stderr)
    return ok


def extract() -> None:
    zip_path = DEST / "SECOND-CC-AUG.zip"
    out_dir = DEST / "SECOND-CC-AUG"
    if out_dir.exists() and sum(1 for _ in out_dir.rglob("*.png")) >= 43420:
        print(f"[skip] {out_dir} already extracted", file=sys.stderr)
        return
    print(f"[extract] {zip_path} -> {DEST}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as z:
        assert len(z.infolist()) == EXPECTED_ENTRIES, f"unexpected entry count {len(z.infolist())}"
        z.extractall(DEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help="also unzip into data/raw/second_cc/")
    args = parser.parse_args()

    if not download_and_verify():
        sys.exit(1)
    if args.extract:
        extract()
    print_summary(DEST)
