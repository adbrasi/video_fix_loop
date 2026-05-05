from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import httpx
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "AdwolfCzar/looper_v4"
REPO_TYPE = "dataset"
REVISION = "main"

log = logging.getLogger(__name__)


def _raw_tree() -> list[dict]:
    """Fetch the chunks/ tree as plain dicts. Separated for easy mocking."""
    api = HfApi()
    out: list[dict] = []
    for entry in api.list_repo_tree(REPO_ID, path_in_repo="chunks", repo_type=REPO_TYPE,
                                     revision=REVISION, recursive=False):
        is_dir = entry.__class__.__name__.endswith("Folder")
        out.append({
            "type": "directory" if is_dir else "file",
            "path": entry.path,
            "size": getattr(entry, "size", 0) or 0,
        })
    return out


def list_remote_zips() -> list[tuple[str, int]]:
    """Return [(repo_path, size_bytes), ...] for chunk_*.zip only."""
    tree = _raw_tree()
    out: list[tuple[str, int]] = []
    for e in tree:
        if e["type"] != "file":
            continue
        p = e["path"]
        if p.endswith(".zip") and "/chunk_" in p:
            out.append((p, e["size"]))
    return sorted(out)


def download_zip(repo_path: str, dest_dir: Path, *, max_retries: int = 5) -> Path:
    """Download a single chunk zip via huggingface_hub. Returns local path.

    hf_hub_download handles partial resume internally.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type=REPO_TYPE,
                revision=REVISION,
                local_dir=str(dest_dir),
            )
            return Path(cached)
        except (HfHubHTTPError, requests.RequestException,
                httpx.TimeoutException, httpx.NetworkError, OSError) as e:
            last_err = e
            backoff = min(60, 2 ** attempt)
            log.warning("download %s failed (attempt %d/%d): %s — retry in %ss",
                        repo_path, attempt, max_retries, e, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"download failed for {repo_path}: {last_err}")
