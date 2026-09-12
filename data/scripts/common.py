"""Shared helpers for the download_*.py scripts. Each script is idempotent (skips files that
already exist at the expected size) and ends by printing a file-count/size summary so a human can
eyeball success -- silent partial failures (especially Google-Drive quota/interstitial issues) are
the main risk per CLAUDE.md's data-acquisition notes."""

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
    (>= expected_min_bytes). Returns True on success."""
    if dest_path.exists() and dest_path.stat().st_size >= expected_min_bytes:
        print(f"[skip] {dest_path.name} already present", file=sys.stderr)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {url} -> {dest_path}", file=sys.stderr)
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
        return True
    except requests.RequestException as exc:
        print(f"[error] failed to download {url}: {exc}", file=sys.stderr)
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
