"""Downloads (a shard of) BigEarthNet -- the mandatory remote-sensing adaptation dataset.

REVISED FINDING vs. CLAUDE.md's original plan: there is no small, individually-fetchable
BigEarthNet subset available. Every non-gated Hugging Face mirror checked bundles the data into
multi-GB chunks with no cheap partial-download path:

    torchgeo/bigearthnet   -- V2, split .tar.gz parts (need ALL parts to reassemble one archive)
    lc-col/bigearthnet     -- HDF5 shards, ~4.3GB EACH (one shard = one usable file, smallest option)
    GFM-Bench/BigEarthNet  -- 5GB raw split-archive parts (also need reassembly, custom loader script)

(danielz01/BigEarthNet-S2-v1.0 has row-level parquet shards, which WOULD allow cheap partial reads,
but that repo is gated -- requires an authenticated, terms-accepted Hugging Face account. Not
scripted here since accepting dataset terms on someone's behalf isn't something to automate; if you
want that route, `huggingface-cli login` after accepting the terms on the dataset page, then adjust
this script to pull individual row groups from one test parquet shard.)

Given that, this script does NOT download anything by default -- it just prints the verified
options above. Pass --shard to fetch exactly one lc-col HDF5 test shard (~4.3GB) as the smallest
real, ungated subset available; even that is `data/raw/`-only (gitignored) and meant as a local
smoke-test sample, not training data. The real large-subset pull for actual fine-tuning belongs
inside the Kaggle notebook (see notebooks/), where disk/bandwidth aren't a local constraint.

Usage:
    python data/scripts/download_bigearthnet.py            # prints options, downloads nothing
    python data/scripts/download_bigearthnet.py --shard     # fetch one ~4.3GB HDF5 test shard
"""

import argparse
from pathlib import Path

from common import hf_snapshot, print_summary

DEST = Path(__file__).resolve().parents[1] / "raw" / "bigearthnet"

SHARD_REPO = "lc-col/bigearthnet"
SHARD_FILE = "bigearthnet_test_p0.hdf5.gz"  # smallest coherent single-file subset found (~4.3GB)


def print_options() -> None:
    print(__doc__)


def download_one_shard() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    hf_snapshot(SHARD_REPO, DEST, allow_patterns=[SHARD_FILE, "bigearthnet_hdf5_test.csv"])
    print_summary(DEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", action="store_true", help="download one ~4.3GB lc-col HDF5 test shard")
    args = parser.parse_args()

    if args.shard:
        download_one_shard()
    else:
        print_options()
