from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from src.db import DB, ZipStatus

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def find_video_txt_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    """Walk `root` and return (video_path, txt_sibling_or_None) for every video file."""
    pairs: list[tuple[Path, Path | None]] = []
    for video in sorted(root.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue
        txt = video.with_suffix(".txt")
        pairs.append((video, txt if txt.exists() else None))
    return pairs


def safe_zip_member(name: str) -> bool:
    """Reject zip slip / absolute paths."""
    if name.startswith("/") or ".." in Path(name).parts:
        return False
    return True


def extract_and_register(*, zip_path: Path, zip_name: str, work_dir: Path, db: DB) -> Path:
    """Unzip into `work_dir/<basename_without_ext>/` and register all videos in DB.

    Returns the output dir.
    """
    base = Path(zip_name).stem
    out = work_dir / base
    out.mkdir(parents=True, exist_ok=True)

    db.set_zip_status(zip_name, ZipStatus.EXTRACTING)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not safe_zip_member(member):
                log.warning("skipping unsafe zip member: %s", member)
                continue
            zf.extract(member, out)

    pairs = find_video_txt_pairs(out)
    rels = [(str(v.relative_to(out)),) for v, _ in pairs]
    db.register_videos(zip_name, rels)
    db.set_zip_status(zip_name, ZipStatus.PROCESSING)
    log.info("extracted %s: %d videos", zip_name, len(pairs))
    return out
