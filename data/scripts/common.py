"""Shared helpers for the download_*.py scripts. Each script is idempotent (skips files that
already exist at the expected size) and ends by printing a file-count/size summary so a human can
eyeball success -- silent partial failures (especially Google-Drive quota/interstitial issues) are
the main risk per CLAUDE.md's data-acquisition notes."""

import os
import sys
from pathlib import Path

import requests

CHUNK_SIZE = 1 << 20  # 1MB


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def print_summary(dest_dir: Path) -> None:
    files = [p for p in dest_dir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"\n[summary] {dest_dir}: {len(files)} file(s), {human_size(total)} total", file=sys.stderr)


def download_url(url: str, dest_path: Path, expected_min_bytes: int = 1024) -> bool:
    """Plain streaming HTTP download. Skips if dest_path already exists and looks complete
    (>= expected_min_bytes). Returns True on success.

    Writes to a `.part` sibling and only `os.replace()`s it onto `dest_path` once the whole
    download has streamed through without error -- previously this wrote straight to `dest_path`,
    so a connection drop mid-download left a truncated file sitting at the real destination,
    which the size check above would then treat as "already present" forever after (the
    "exists and is non-empty" check has no way to tell a truncated file from a complete one
    once it's already at the final path). A `.part` file left behind by an interrupted run is
    never at `dest_path`, so the next invocation correctly sees the real file as missing and
    retries -- `os.replace` is atomic on both POSIX and Windows, so a reader can never observe a
    half-written `dest_path` either."""
    if dest_path.exists() and dest_path.stat().st_size >= expected_min_bytes:
        print(f"[skip] {dest_path.name} already present", file=sys.stderr)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    print(f"[get] {url} -> {dest_path}", file=sys.stderr)
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
        os.replace(part_path, dest_path)
        return True
    except requests.RequestException as exc:
        print(f"[error] failed to download {url}: {exc}", file=sys.stderr)
        part_path.unlink(missing_ok=True)  # don't leave a stale partial file behind
        return False


def download_url_resumable(url: str, dest_path: Path, expected_size: int, max_attempts: int = 60) -> bool:
    """For big files on hosts that drop connections (Zenodo did, mid-download, on a 2.5GB zip --
    `download_url` above deletes its `.part` file on any failure, so every drop restarts from zero).
    Keeps the `.part` file between attempts and resumes with an HTTP Range request, backing off
    between tries (and honouring `Retry-After` on a 429). Only `os.replace()`s onto `dest_path`
    once the `.part` is exactly `expected_size` bytes, so a half-finished file is never mistaken
    for a complete one; a failed run leaves the `.part` in place for the next invocation to resume."""
    import time

    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(f"[skip] {dest_path.name} already present", file=sys.stderr)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    last_report = 0
    for attempt in range(1, max_attempts + 1):
        have = part_path.stat().st_size if part_path.exists() else 0
        if have == expected_size:
            break
        if have > expected_size:  # corrupt/oversized leftover -- start over
            part_path.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"[get] attempt {attempt}: {url} from byte {have:,}/{expected_size:,}", file=sys.stderr)
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30))
                    print(f"[429] rate limited -- waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                if have and resp.status_code != 206:
                    print("[warn] server ignored the Range header -- restarting from byte 0", file=sys.stderr)
                    have = 0
                with open(part_path, "ab" if have else "wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        pos = f.tell()
                        if pos - last_report >= 100 * CHUNK_SIZE:
                            last_report = pos
                            print(f"[progress] {human_size(pos)} / {human_size(expected_size)}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"[retry] connection dropped ({type(exc).__name__}) -- will resume", file=sys.stderr)
        time.sleep(min(60, 2 ** min(attempt, 6)))

    if part_path.exists() and part_path.stat().st_size == expected_size:
        os.replace(part_path, dest_path)
        return True
    print(f"[error] gave up on {url} after {max_attempts} attempts; partial file kept at {part_path} for resuming", file=sys.stderr)
    return False


def zenodo_record_files(record_id: str) -> list[dict]:
    """Looks up a Zenodo record's file list via its REST API (avoids hardcoding exact filenames,
    which drift between deposit versions)."""
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_zenodo_record(record_id: str, dest_dir: Path) -> None:
    files = zenodo_record_files(record_id)
    if not files:
        print(f"[error] Zenodo record {record_id} returned no files -- check https://zenodo.org/records/{record_id}", file=sys.stderr)
        return
    for f in files:
        url = f["links"]["self"]
        filename = f.get("key") or f.get("filename") or url.rsplit("/", 1)[-1]
        size = f.get("size", 1024)
        download_url(url, dest_dir / filename, expected_min_bytes=max(1, size - 1))


def hf_snapshot(repo_id: str, dest_dir: Path, repo_type: str = "dataset", allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    print(f"[hf] snapshot_download({repo_id}, allow_patterns={allow_patterns}) -> {dest_dir}", file=sys.stderr)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(dest_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as exc:  # HF hub raises several distinct exception types across versions
        print(f"[error] huggingface_hub snapshot_download failed: {exc}", file=sys.stderr)
        print(f"[fallback] browse/download manually at https://huggingface.co/datasets/{repo_id}", file=sys.stderr)


def gdown_folder(url: str, dest_dir: Path) -> None:
    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gdown] folder {url} -> {dest_dir}", file=sys.stderr)
    try:
        gdown.download_folder(url=url, output=str(dest_dir), quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"[error] gdown folder download failed ({exc}).", file=sys.stderr)
        print(f"[fallback] open {url} in a browser and download manually into {dest_dir}", file=sys.stderr)
