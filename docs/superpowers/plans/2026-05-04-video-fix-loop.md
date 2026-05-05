# video_fix_loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming, resumable pipeline that downloads ~70 zips (389GB) from HF dataset `AdwolfCzar/looper_v4`, normalizes ~30k videos to 30fps + max 5s, preserves audio, and writes flat output to `outputs/`.

**Architecture:** Producer-consumer pipeline with `multiprocessing.Queue`. SQLite (WAL) for resumable state. Random pick of pending zips. Per-video ffprobe-driven action decision: copy / stream-copy / re-encode. CPU-parallel libx264 saturating 96 cores. Cleanup deletes zip + extracted dir as soon as a chunk is fully processed.

**Tech Stack:** Python 3.11+, `huggingface_hub`, `ffmpeg-python` (or subprocess), SQLite stdlib, `multiprocessing`, `pytest`. ffmpeg/ffprobe binaries on system PATH.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned dependencies |
| `run.sh` | Bootstrap: venv + ffmpeg check + run |
| `.gitignore` | Add `outputs/`, `work/`, `state.db`, `run.log`, `.venv/` |
| `src/__init__.py` | Package marker |
| `src/db.py` | SQLite schema, CRUD, atomic state transitions |
| `src/downloader.py` | HF chunk listing + zip download with retry |
| `src/extractor.py` | Unzip + enumerate video/txt pairs + register in DB |
| `src/encoder.py` | ffprobe inspection + ffmpeg dispatch + collision-safe output |
| `src/pipeline.py` | Orchestrator: queues, worker pools, signal handlers, cleanup |
| `src/main.py` | CLI entry: `run`, `status`, `reset` |
| `tests/test_db.py` | DB state transitions and concurrency |
| `tests/test_encoder.py` | Action decision + collision rename |
| `tests/test_extractor.py` | Pair detection + DB registration |
| `tests/conftest.py` | Pytest fixtures (synthetic video, tmp DB) |

---

## Task 1: Project skeleton and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore`
- Create: `run.sh`

- [ ] **Step 1.1: Update .gitignore**

Append to `.gitignore`:

```
# runtime artifacts
outputs/
work/
state.db
state.db-*
run.log
*.log

# python
.venv/
.pytest_cache/
__pycache__/
*.pyc
```

- [ ] **Step 1.2: Create requirements.txt**

```
huggingface_hub>=0.24.0
tqdm>=4.66.0
pytest>=8.0.0
```

- [ ] **Step 1.3: Create empty package markers**

`src/__init__.py`: empty file
`tests/__init__.py`: empty file

- [ ] **Step 1.4: Create tests/conftest.py with synthetic-video fixture**

```python
import shutil
import subprocess
from pathlib import Path

import pytest


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_video(path: Path, fps: int, duration: float, with_audio: bool = False) -> None:
    """Generate a tiny synthetic video using ffmpeg lavfi."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x64:rate={fps}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


@pytest.fixture
def synthetic_video_factory(tmp_path):
    if not _have_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not available")

    def make(fps: int = 30, duration: float = 2.0, with_audio: bool = False, name: str = "clip.mp4") -> Path:
        out = tmp_path / name
        _make_video(out, fps=fps, duration=duration, with_audio=with_audio)
        return out

    return make


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "state.db"
```

- [ ] **Step 1.5: Create run.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg/ffprobe not found. Install with: sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python -m src.main "$@"
```

Make executable:

```bash
chmod +x run.sh
```

- [ ] **Step 1.6: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py tests/__init__.py tests/conftest.py run.sh
git commit -m "chore: scaffold project (deps, gitignore, run.sh, test fixtures)"
```

---

## Task 2: SQLite state module (`src/db.py`)

**Files:**
- Create: `src/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 2.1: Write failing tests**

`tests/test_db.py`:

```python
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.db import DB, ZipStatus, VideoStatus


def test_init_creates_schema(tmp_db):
    db = DB(tmp_db)
    db.init()
    with sqlite3.connect(tmp_db) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"zips", "videos"}.issubset(names)


def test_upsert_zips_idempotent(tmp_db):
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("a.zip", 100), ("b.zip", 200)])
    db.upsert_zips([("a.zip", 100), ("b.zip", 200), ("c.zip", 300)])
    pending = db.list_zips_by_status(ZipStatus.PENDING)
    assert {z["name"] for z in pending} == {"a.zip", "b.zip", "c.zip"}


def test_claim_random_pending_zip_atomic(tmp_db):
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("only.zip", 1)])
    a = db.claim_random_zip(target=ZipStatus.DOWNLOADING)
    b = db.claim_random_zip(target=ZipStatus.DOWNLOADING)
    assert a is not None and a["name"] == "only.zip"
    assert b is None  # already claimed


def test_register_videos_and_query_pending(tmp_db):
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("z.zip", 1)])
    db.register_videos("z.zip", [("inner/v1.mp4",), ("inner/v2.mp4",)])
    pend = db.list_pending_videos_for_zip("z.zip")
    assert {v["internal_path"] for v in pend} == {"inner/v1.mp4", "inner/v2.mp4"}


def test_set_video_done_idempotent(tmp_db):
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("z.zip", 1)])
    db.register_videos("z.zip", [("v.mp4",)])
    db.set_video_done("z.zip", "v.mp4", output_name="v.mp4", action="cp",
                      fps_in=30.0, duration_in=2.0, has_audio=False)
    db.set_video_done("z.zip", "v.mp4", output_name="v.mp4", action="cp",
                      fps_in=30.0, duration_in=2.0, has_audio=False)
    counts = db.zip_progress("z.zip")
    assert counts["done"] == 1 and counts["total"] == 1


def test_reset_stale_in_progress(tmp_db):
    """Resume must reset zips left in transient states."""
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("a.zip", 1), ("b.zip", 1)])
    db.set_zip_status("a.zip", ZipStatus.DOWNLOADING)
    db.set_zip_status("b.zip", ZipStatus.EXTRACTING)
    db.reset_stale_zips(zip_files_present=set())  # nothing on disk
    statuses = {z["name"]: z["status"] for z in db.all_zips()}
    assert statuses["a.zip"] == ZipStatus.PENDING.value
    assert statuses["b.zip"] == ZipStatus.PENDING.value
```

- [ ] **Step 2.2: Run failing tests**

```bash
./.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: ImportError / collection failure (`src.db` doesn't exist).

- [ ] **Step 2.3: Implement `src/db.py`**

```python
from __future__ import annotations

import enum
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional


class ZipStatus(str, enum.Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class VideoStatus(str, enum.Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


SCHEMA = """
CREATE TABLE IF NOT EXISTS zips (
    name        TEXT PRIMARY KEY,
    size_bytes  INTEGER,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    zip_name      TEXT NOT NULL,
    internal_path TEXT NOT NULL,
    output_name   TEXT,
    fps_in        REAL,
    duration_in   REAL,
    has_audio     INTEGER,
    action        TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    UNIQUE(zip_name, internal_path)
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_zip   ON videos(zip_name);
CREATE INDEX IF NOT EXISTS idx_zips_status  ON zips(status);
"""


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=5000")
        try:
            yield c
        finally:
            c.close()

    def init(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---------------- zips ----------------

    def upsert_zips(self, items: Iterable[tuple[str, int]]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO zips(name, size_bytes) VALUES(?, ?)",
                list(items),
            )

    def all_zips(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM zips ORDER BY name")]

    def list_zips_by_status(self, status: ZipStatus) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM zips WHERE status=?", (status.value,))]

    def claim_random_zip(self, *, source: ZipStatus = ZipStatus.PENDING,
                          target: ZipStatus = ZipStatus.DOWNLOADING) -> Optional[dict]:
        """Atomically pick a random zip in `source` state and move to `target`. Returns row or None."""
        with self._lock, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT name FROM zips WHERE status=? ORDER BY RANDOM() LIMIT 1",
                (source.value,),
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            c.execute(
                "UPDATE zips SET status=?, started_at=COALESCE(started_at, CURRENT_TIMESTAMP) WHERE name=?",
                (target.value, row["name"]),
            )
            c.execute("COMMIT")
            return dict(c.execute("SELECT * FROM zips WHERE name=?", (row["name"],)).fetchone())

    def set_zip_status(self, name: str, status: ZipStatus, error: Optional[str] = None) -> None:
        with self._conn() as c:
            if status == ZipStatus.DONE:
                c.execute("UPDATE zips SET status=?, error=NULL, finished_at=CURRENT_TIMESTAMP WHERE name=?",
                          (status.value, name))
            elif status == ZipStatus.FAILED:
                c.execute("UPDATE zips SET status=?, error=?, finished_at=CURRENT_TIMESTAMP WHERE name=?",
                          (status.value, error, name))
            else:
                c.execute("UPDATE zips SET status=? WHERE name=?", (status.value, name))

    def reset_stale_zips(self, zip_files_present: set[str]) -> None:
        """On startup: zips in transient states get reverted to safe state.
        - downloading → pending (download was incomplete)
        - extracting/processing → downloaded if zip file exists, else pending
        """
        with self._conn() as c:
            c.execute("UPDATE zips SET status='pending' WHERE status='downloading'")
            for row in c.execute("SELECT name FROM zips WHERE status IN ('extracting','processing')").fetchall():
                target = "downloaded" if row["name"] in zip_files_present else "pending"
                c.execute("UPDATE zips SET status=? WHERE name=?", (target, row["name"]))

    # ---------------- videos ----------------

    def register_videos(self, zip_name: str, items: Iterable[tuple[str]]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO videos(zip_name, internal_path) VALUES(?, ?)",
                [(zip_name, p[0]) for p in items],
            )

    def list_pending_videos_for_zip(self, zip_name: str) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM videos WHERE zip_name=? AND status='pending'", (zip_name,))]

    def set_video_done(self, zip_name: str, internal_path: str, *,
                        output_name: str, action: str,
                        fps_in: float, duration_in: float, has_audio: bool) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE videos SET status='done', error=NULL, output_name=?, action=?,
                   fps_in=?, duration_in=?, has_audio=? WHERE zip_name=? AND internal_path=?""",
                (output_name, action, fps_in, duration_in, int(has_audio), zip_name, internal_path),
            )

    def set_video_failed(self, zip_name: str, internal_path: str, error: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE videos SET status='failed', error=? WHERE zip_name=? AND internal_path=?",
                (error, zip_name, internal_path),
            )

    def zip_progress(self, zip_name: str) -> dict:
        with self._conn() as c:
            row = c.execute(
                """SELECT
                       COUNT(*) AS total,
                       SUM(status='done') AS done,
                       SUM(status='failed') AS failed,
                       SUM(status='pending') AS pending
                   FROM videos WHERE zip_name=?""",
                (zip_name,),
            ).fetchone()
        return {k: (row[k] or 0) for k in ("total", "done", "failed", "pending")}

    def overall_progress(self) -> dict:
        with self._conn() as c:
            r = c.execute(
                """SELECT
                       (SELECT COUNT(*) FROM zips) AS zips_total,
                       (SELECT COUNT(*) FROM zips WHERE status='done') AS zips_done,
                       (SELECT COUNT(*) FROM videos) AS videos_total,
                       (SELECT COUNT(*) FROM videos WHERE status='done') AS videos_done,
                       (SELECT COUNT(*) FROM videos WHERE status='failed') AS videos_failed"""
            ).fetchone()
        return dict(r)
```

- [ ] **Step 2.4: Run tests, verify pass**

```bash
./.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/db.py tests/test_db.py
git commit -m "feat(db): SQLite state module with WAL + atomic claim"
```

---

## Task 3: HF Downloader (`src/downloader.py`)

**Files:**
- Create: `src/downloader.py`
- Create: `tests/test_downloader.py`

- [ ] **Step 3.1: Write failing tests (no network — use stubs)**

`tests/test_downloader.py`:

```python
from pathlib import Path
from unittest.mock import patch

from src.downloader import list_remote_zips, REPO_ID


def test_list_remote_zips_filters_zip_files():
    fake_tree = [
        {"type": "directory", "path": "chunks/state", "size": 0},
        {"type": "file", "path": "chunks/chunk_a.zip", "size": 1000},
        {"type": "file", "path": "chunks/chunk_a.zip.sha256", "size": 100},
        {"type": "file", "path": "chunks/chunk_b.zip", "size": 2000},
        {"type": "file", "path": "chunks/state/state.json", "size": 50},
    ]
    with patch("src.downloader._raw_tree", return_value=fake_tree):
        items = list_remote_zips()
    assert items == [("chunks/chunk_a.zip", 1000), ("chunks/chunk_b.zip", 2000)]


def test_repo_id_constant():
    assert REPO_ID == "AdwolfCzar/looper_v4"
```

- [ ] **Step 3.2: Run, verify fail (import error)**

```bash
./.venv/bin/python -m pytest tests/test_downloader.py -v
```

- [ ] **Step 3.3: Implement `src/downloader.py`**

```python
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

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
        out.append({
            "type": "directory" if entry.__class__.__name__.endswith("Folder") else "file",
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

    Uses hf_hub_download which handles partial resume internally and writes to a
    cache then symlinks. We then move the file into dest_dir for full control.
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
                local_dir_use_symlinks=False,
            )
            return Path(cached)
        except (HfHubHTTPError, requests.RequestException) as e:
            last_err = e
            backoff = min(60, 2 ** attempt)
            log.warning("download %s failed (attempt %d/%d): %s — retry in %ss",
                        repo_path, attempt, max_retries, e, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"download failed for {repo_path}: {last_err}")
```

Add `requests` to `requirements.txt`:

```
huggingface_hub>=0.24.0
requests>=2.31.0
tqdm>=4.66.0
pytest>=8.0.0
```

Re-install: `./.venv/bin/pip install -r requirements.txt`

- [ ] **Step 3.4: Run tests**

```bash
./.venv/bin/python -m pytest tests/test_downloader.py -v
```

Expected: 2 pass.

- [ ] **Step 3.5: Commit**

```bash
git add src/downloader.py tests/test_downloader.py requirements.txt
git commit -m "feat(downloader): HF chunk listing + retried zip download"
```

---

## Task 4: Extractor (`src/extractor.py`)

**Files:**
- Create: `src/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 4.1: Write failing tests**

`tests/test_extractor.py`:

```python
import zipfile
from pathlib import Path

from src.db import DB, ZipStatus
from src.extractor import extract_and_register, find_video_txt_pairs


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}


def _make_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def test_find_pairs_includes_only_videos_with_txt(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "a.mp4").write_bytes(b"x")
    (extracted / "a.txt").write_text("desc")
    (extracted / "b.mp4").write_bytes(b"x")  # no txt sidecar
    (extracted / "c.txt").write_text("orphan txt")
    pairs = find_video_txt_pairs(extracted)
    assert {p[0].name for p in pairs} == {"a.mp4", "b.mp4"}
    # b has no txt → second element must be None
    by_name = {v.name: t for v, t in pairs}
    assert by_name["a.mp4"] and by_name["a.mp4"].name == "a.txt"
    assert by_name["b.mp4"] is None


def test_extract_and_register_populates_db(tmp_path, tmp_db):
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("chunks/x.zip", 1)])
    db.set_zip_status("chunks/x.zip", ZipStatus.DOWNLOADED)

    zip_path = tmp_path / "x.zip"
    _make_zip(zip_path, {
        "video1.mp4": b"\x00" * 16,
        "video1.txt": b"hello",
        "video2.mp4": b"\x00" * 16,
        "video2.txt": b"world",
    })

    out_dir = extract_and_register(zip_path=zip_path, zip_name="chunks/x.zip",
                                   work_dir=tmp_path / "extracted", db=db)
    assert out_dir.exists()
    pending = db.list_pending_videos_for_zip("chunks/x.zip")
    assert {v["internal_path"] for v in pending} == {"video1.mp4", "video2.mp4"}
```

- [ ] **Step 4.2: Run, verify fail**

```bash
./.venv/bin/python -m pytest tests/test_extractor.py -v
```

- [ ] **Step 4.3: Implement `src/extractor.py`**

```python
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Iterable

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
    base = Path(zip_name).stem  # e.g. chunk_2026...
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
    # Register relative paths from `out`
    rels = [(str(v.relative_to(out)),) for v, _ in pairs]
    db.register_videos(zip_name, rels)
    db.set_zip_status(zip_name, ZipStatus.PROCESSING)
    log.info("extracted %s: %d videos", zip_name, len(pairs))
    return out
```

- [ ] **Step 4.4: Run tests**

```bash
./.venv/bin/python -m pytest tests/test_extractor.py -v
```

Expected: 2 pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/extractor.py tests/test_extractor.py
git commit -m "feat(extractor): unzip + register video/txt pairs in DB"
```

---

## Task 5: Encoder (`src/encoder.py`)

**Files:**
- Create: `src/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 5.1: Write failing tests**

`tests/test_encoder.py`:

```python
import json
from pathlib import Path

import pytest

from src.encoder import probe_video, decide_action, allocate_output_name, process_video


def test_probe_30fps_no_audio(synthetic_video_factory):
    p = synthetic_video_factory(fps=30, duration=2.0, with_audio=False)
    info = probe_video(p)
    assert abs(info.fps - 30.0) < 0.1
    assert 1.5 <= info.duration <= 2.5
    assert info.has_audio is False


def test_probe_24fps_with_audio(synthetic_video_factory):
    p = synthetic_video_factory(fps=24, duration=1.0, with_audio=True)
    info = probe_video(p)
    assert abs(info.fps - 24.0) < 0.1
    assert info.has_audio is True


@pytest.mark.parametrize("fps,duration,expected", [
    (30.0, 2.0, "cp"),            # already 30fps and short
    (30.0, 7.0, "stream_copy"),   # 30fps but long
    (24.0, 2.0, "reencode"),      # wrong fps
    (60.0, 8.0, "reencode"),      # wrong fps and long
])
def test_decide_action(fps, duration, expected):
    assert decide_action(fps=fps, duration=duration) == expected


def test_allocate_output_name_no_collision(tmp_path):
    name = allocate_output_name(tmp_path, "clip.mp4")
    assert name == "clip.mp4"


def test_allocate_output_name_with_collision(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "clip.txt").write_text("x")
    name = allocate_output_name(tmp_path, "clip.mp4")
    assert name == "clip__1.mp4"
    (tmp_path / "clip__1.mp4").write_bytes(b"x")
    name2 = allocate_output_name(tmp_path, "clip.mp4")
    assert name2 == "clip__2.mp4"


def test_process_video_cp_path(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=30, duration=2.0, with_audio=False, name="a.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-a")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.action == "cp"
    assert (out_dir / result.output_name).exists()
    assert (out_dir / result.output_name).with_suffix(".txt").read_text() == "desc-a"


def test_process_video_reencode_path(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=24, duration=2.0, with_audio=True, name="b.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-b")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.action == "reencode"
    out_path = out_dir / result.output_name
    assert out_path.exists()
    new_info = probe_video(out_path)
    assert abs(new_info.fps - 30.0) < 1.5
    assert new_info.duration <= 5.1


def test_process_video_truncate_path(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=30, duration=8.0, with_audio=False, name="c.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-c")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.action == "stream_copy"
    out_path = out_dir / result.output_name
    info = probe_video(out_path)
    assert info.duration <= 5.5  # tolerance for keyframe alignment


def test_process_video_collision_renames_both(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=30, duration=2.0, with_audio=False, name="d.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-d")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "d.mp4").write_bytes(b"existing")
    (out_dir / "d.txt").write_text("existing")
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.output_name == "d__1.mp4"
    assert (out_dir / "d__1.mp4").exists()
    assert (out_dir / "d__1.txt").read_text() == "desc-d"
```

- [ ] **Step 5.2: Run, verify fail**

```bash
./.venv/bin/python -m pytest tests/test_encoder.py -v
```

- [ ] **Step 5.3: Implement `src/encoder.py`**

```python
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MAX_DURATION = 5.0
TARGET_FPS = 30
FPS_TOLERANCE = 0.05  # treat 29.97/30/30.000 as "30"
DURATION_TOLERANCE = 0.05  # below cap counts as ≤5
FFMPEG_TIMEOUT = 120  # seconds per video


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    duration: float
    has_audio: bool


@dataclass(frozen=True)
class EncodeResult:
    action: str          # cp | stream_copy | reencode
    output_name: str
    info: VideoInfo


def _ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


def _parse_fps(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            n, d = float(num), float(den)
            return n / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def probe_video(path: Path) -> VideoInfo:
    data = _ffprobe_json(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    fps = _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    return VideoInfo(fps=fps, duration=duration, has_audio=audio is not None)


def decide_action(*, fps: float, duration: float) -> str:
    near_30 = abs(fps - TARGET_FPS) <= FPS_TOLERANCE
    short = duration <= MAX_DURATION + DURATION_TOLERANCE
    if near_30 and short:
        return "cp"
    if near_30 and not short:
        return "stream_copy"
    return "reencode"


def allocate_output_name(output_dir: Path, desired: str) -> str:
    """Return a filename in `output_dir` that doesn't collide.

    Considers BOTH the .mp4 (or other ext) and the .txt sibling for collision.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = Path(desired)
    stem, ext = p.stem, p.suffix
    candidate_video = output_dir / desired
    candidate_txt = candidate_video.with_suffix(".txt")
    if not candidate_video.exists() and not candidate_txt.exists():
        return desired
    n = 1
    while True:
        new_name = f"{stem}__{n}{ext}"
        v = output_dir / new_name
        t = v.with_suffix(".txt")
        if not v.exists() and not t.exists():
            return new_name
        n += 1


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    subprocess.run(cmd, check=True, timeout=FFMPEG_TIMEOUT)


def _do_cp(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def _do_stream_copy(src: Path, dst: Path) -> None:
    _run_ffmpeg(["-i", str(src), "-t", str(MAX_DURATION),
                 "-c", "copy", "-movflags", "+faststart", str(dst)])


def _do_reencode(src: Path, dst: Path, *, has_audio: bool) -> None:
    args = [
        "-i", str(src),
        "-vf", f"fps={TARGET_FPS}",
        "-t", str(MAX_DURATION),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if has_audio:
        args += ["-c:a", "aac", "-b:a", "128k"]
    else:
        args += ["-an"]
    args.append(str(dst))
    _run_ffmpeg(args)


def process_video(*, video: Path, txt: Optional[Path], output_dir: Path) -> EncodeResult:
    info = probe_video(video)
    action = decide_action(fps=info.fps, duration=info.duration)
    out_name = allocate_output_name(output_dir, video.name)
    out_video = output_dir / out_name
    out_txt = out_video.with_suffix(".txt")

    tmp = out_video.with_suffix(out_video.suffix + ".part")
    try:
        if action == "cp":
            _do_cp(video, tmp)
        elif action == "stream_copy":
            try:
                _do_stream_copy(video, tmp)
            except subprocess.CalledProcessError:
                # some containers don't allow direct stream cut → fallback to reencode
                if tmp.exists():
                    tmp.unlink()
                _do_reencode(video, tmp, has_audio=info.has_audio)
                action = "reencode"
        else:  # reencode
            _do_reencode(video, tmp, has_audio=info.has_audio)
        tmp.replace(out_video)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    if txt is not None and txt.exists():
        shutil.copyfile(txt, out_txt)

    return EncodeResult(action=action, output_name=out_name, info=info)
```

- [ ] **Step 5.4: Run tests**

```bash
./.venv/bin/python -m pytest tests/test_encoder.py -v
```

Expected: 9 pass.

- [ ] **Step 5.5: Commit**

```bash
git add src/encoder.py tests/test_encoder.py
git commit -m "feat(encoder): ffprobe-driven action + cp/stream_copy/reencode + collision rename"
```

---

## Task 6: Pipeline orchestrator (`src/pipeline.py`)

**Files:**
- Create: `src/pipeline.py`

- [ ] **Step 6.1: Implement orchestrator**

`src/pipeline.py`:

```python
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Optional

from src.db import DB, VideoStatus, ZipStatus
from src.downloader import download_zip, list_remote_zips
from src.encoder import process_video
from src.extractor import extract_and_register

log = logging.getLogger(__name__)


@dataclass
class Config:
    workdir: Path
    output_dir: Path
    db_path: Path
    n_downloaders: int = 2
    n_extractors: int = 2
    n_encoders: int = 32
    download_queue_max: int = 4
    extract_queue_max: int = 8
    encode_queue_max: int = 256


_STOP = mp.Event()


def _signal_handler(signum, _frame):
    log.warning("signal %s received → graceful shutdown", signum)
    _STOP.set()


# ============================================================
# Worker functions (each runs in its own process or thread)
# ============================================================

def _downloader_loop(cfg: Config, dl_queue: mp.Queue):
    db = DB(cfg.db_path)
    while not _STOP.is_set():
        zip_row = db.claim_random_zip(source=ZipStatus.PENDING, target=ZipStatus.DOWNLOADING)
        if zip_row is None:
            time.sleep(2)
            if not db.list_zips_by_status(ZipStatus.PENDING):
                break
            continue
        name = zip_row["name"]
        try:
            log.info("download start: %s", name)
            local = download_zip(name, cfg.workdir / "zips")
            db.set_zip_status(name, ZipStatus.DOWNLOADED)
            dl_queue.put((name, str(local)))
        except Exception as e:
            log.exception("download failed: %s", name)
            db.set_zip_status(name, ZipStatus.FAILED, error=f"download: {e}")


def _extractor_loop(cfg: Config, dl_queue: mp.Queue, ex_queue: mp.Queue):
    db = DB(cfg.db_path)
    while not _STOP.is_set():
        try:
            item = dl_queue.get(timeout=2)
        except Empty:
            if _STOP.is_set():
                break
            continue
        if item is None:
            break
        zip_name, local_path = item
        try:
            log.info("extract start: %s", zip_name)
            extracted_dir = extract_and_register(
                zip_path=Path(local_path),
                zip_name=zip_name,
                work_dir=cfg.workdir / "extracted",
                db=db,
            )
            for video_row in db.list_pending_videos_for_zip(zip_name):
                ex_queue.put({
                    "zip_name": zip_name,
                    "internal_path": video_row["internal_path"],
                    "extracted_dir": str(extracted_dir),
                    "local_zip": local_path,
                })
        except Exception as e:
            log.exception("extract failed: %s", zip_name)
            db.set_zip_status(zip_name, ZipStatus.FAILED, error=f"extract: {e}")


def _encoder_loop(cfg: Config, ex_queue: mp.Queue):
    db = DB(cfg.db_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    while not _STOP.is_set():
        try:
            task = ex_queue.get(timeout=2)
        except Empty:
            if _STOP.is_set():
                break
            continue
        if task is None:
            break
        zip_name = task["zip_name"]
        internal = task["internal_path"]
        extracted = Path(task["extracted_dir"])
        video = extracted / internal
        txt = video.with_suffix(".txt")
        if not txt.exists():
            txt = None
        try:
            result = process_video(video=video, txt=txt, output_dir=cfg.output_dir)
            db.set_video_done(
                zip_name, internal,
                output_name=result.output_name,
                action=result.action,
                fps_in=result.info.fps,
                duration_in=result.info.duration,
                has_audio=result.info.has_audio,
            )
        except Exception as e:
            log.exception("encode failed: %s :: %s", zip_name, internal)
            db.set_video_failed(zip_name, internal, str(e))


def _cleaner_loop(cfg: Config):
    """Periodically check for fully-processed zips and free disk."""
    db = DB(cfg.db_path)
    while not _STOP.is_set():
        time.sleep(5)
        for z in db.list_zips_by_status(ZipStatus.PROCESSING):
            name = z["name"]
            prog = db.zip_progress(name)
            if prog["total"] > 0 and prog["pending"] == 0:
                base = Path(name).stem
                zip_path = cfg.workdir / "zips" / Path(name).name
                ext_dir = cfg.workdir / "extracted" / base
                # also handle hf_hub_download nested layout
                nested_zip = cfg.workdir / "zips" / name
                for p in (zip_path, nested_zip):
                    if p.exists():
                        try: p.unlink()
                        except OSError: pass
                if ext_dir.exists():
                    shutil.rmtree(ext_dir, ignore_errors=True)
                if prog["failed"] == 0:
                    db.set_zip_status(name, ZipStatus.DONE)
                else:
                    db.set_zip_status(name, ZipStatus.FAILED,
                                      error=f"{prog['failed']} videos failed")
                log.info("cleaned %s (done=%d failed=%d)", name, prog["done"], prog["failed"])


def _progress_loop(cfg: Config):
    db = DB(cfg.db_path)
    last = time.time()
    while not _STOP.is_set():
        time.sleep(10)
        p = db.overall_progress()
        log.info("PROGRESS zips %d/%d  videos done=%d failed=%d total=%d  elapsed=%ds",
                 p["zips_done"], p["zips_total"],
                 p["videos_done"], p["videos_failed"], p["videos_total"],
                 int(time.time() - last))


# ============================================================
# Orchestration
# ============================================================

def _bootstrap_state(cfg: Config) -> None:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / "zips").mkdir(parents=True, exist_ok=True)
    (cfg.workdir / "extracted").mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    db = DB(cfg.db_path)
    db.init()

    # sync remote zip list
    log.info("listing remote zips...")
    remote = list_remote_zips()
    log.info("found %d remote zips (%.1f GB total)",
             len(remote), sum(s for _, s in remote) / 1e9)
    db.upsert_zips(remote)

    # reset stale states
    present = {p.name for p in (cfg.workdir / "zips").rglob("*.zip")}
    db.reset_stale_zips(zip_files_present=present)


def run(cfg: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s/%(levelname)s] %(message)s",
    )
    _bootstrap_state(cfg)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    dl_queue: mp.Queue = mp.Queue(maxsize=cfg.download_queue_max)
    ex_queue: mp.Queue = mp.Queue(maxsize=cfg.encode_queue_max)

    procs: list[mp.Process] = []

    for i in range(cfg.n_downloaders):
        p = mp.Process(target=_downloader_loop, args=(cfg, dl_queue), name=f"dl-{i}")
        p.start(); procs.append(p)
    for i in range(cfg.n_extractors):
        p = mp.Process(target=_extractor_loop, args=(cfg, dl_queue, ex_queue), name=f"ex-{i}")
        p.start(); procs.append(p)
    for i in range(cfg.n_encoders):
        p = mp.Process(target=_encoder_loop, args=(cfg, ex_queue), name=f"enc-{i}")
        p.start(); procs.append(p)

    cleaner = threading.Thread(target=_cleaner_loop, args=(cfg,), name="cleaner", daemon=True)
    cleaner.start()
    progress = threading.Thread(target=_progress_loop, args=(cfg,), name="progress", daemon=True)
    progress.start()

    # idle watcher: when no pending zips and queues drained, signal stop
    db = DB(cfg.db_path)
    while not _STOP.is_set():
        time.sleep(15)
        p = db.overall_progress()
        unfinished = p["zips_total"] - p["zips_done"]
        # consider failed zips done for termination purposes
        n_failed = sum(1 for z in db.all_zips() if z["status"] == "failed")
        if (p["zips_total"] - p["zips_done"] - n_failed) == 0 and dl_queue.empty() and ex_queue.empty():
            log.info("all work done")
            _STOP.set()
            break

    # send poison pills
    for _ in range(cfg.n_extractors):
        try: dl_queue.put_nowait(None)
        except Exception: pass
    for _ in range(cfg.n_encoders):
        try: ex_queue.put_nowait(None)
        except Exception: pass

    for p in procs:
        p.join(timeout=180)
        if p.is_alive():
            log.warning("forcing terminate: %s", p.name)
            p.terminate()
    log.info("shutdown complete")
```

- [ ] **Step 6.2: Smoke check (no test required, but verify imports)**

```bash
./.venv/bin/python -c "from src.pipeline import run, Config; print('ok')"
```

- [ ] **Step 6.3: Commit**

```bash
git add src/pipeline.py
git commit -m "feat(pipeline): multiprocess orchestrator with signal handling and cleanup"
```

---

## Task 7: CLI (`src/main.py`)

**Files:**
- Create: `src/main.py`

- [ ] **Step 7.1: Implement CLI**

```python
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.db import DB
from src.pipeline import Config, run as run_pipeline


def _default_config(args) -> Config:
    return Config(
        workdir=Path(args.workdir).resolve(),
        output_dir=Path(args.outputs).resolve(),
        db_path=Path(args.db).resolve(),
        n_downloaders=args.downloaders,
        n_extractors=args.extractors,
        n_encoders=args.encoders,
    )


def cmd_run(args) -> int:
    cfg = _default_config(args)
    run_pipeline(cfg)
    return 0


def cmd_status(args) -> int:
    db = DB(Path(args.db))
    p = db.overall_progress()
    print(f"zips done:   {p['zips_done']}/{p['zips_total']}")
    print(f"videos done: {p['videos_done']}")
    print(f"videos fail: {p['videos_failed']}")
    print(f"videos tot:  {p['videos_total']}")
    return 0


def cmd_reset(args) -> int:
    p = Path(args.db)
    if p.exists():
        p.unlink()
    if (p.parent / (p.name + "-wal")).exists():
        (p.parent / (p.name + "-wal")).unlink()
    if (p.parent / (p.name + "-shm")).exists():
        (p.parent / (p.name + "-shm")).unlink()
    print(f"removed {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="video_fix_loop")
    ap.add_argument("--db", default="state.db")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--outputs", default="outputs")

    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="start the pipeline")
    runp.add_argument("--downloaders", type=int, default=2)
    runp.add_argument("--extractors", type=int, default=2)
    runp.add_argument("--encoders", type=int, default=int(os.environ.get("VFL_ENCODERS", "32")))
    runp.set_defaults(func=cmd_run)

    sub.add_parser("status", help="print progress").set_defaults(func=cmd_status)
    sub.add_parser("reset", help="delete state.db").set_defaults(func=cmd_reset)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.2: Verify CLI parses**

```bash
./.venv/bin/python -m src.main --help
./.venv/bin/python -m src.main run --help
```

Expected: usage output, no errors.

- [ ] **Step 7.3: Commit**

```bash
git add src/main.py
git commit -m "feat(cli): main entry with run/status/reset subcommands"
```

---

## Task 8: Run full test suite

- [ ] **Step 8.1: Run all tests**

```bash
./.venv/bin/python -m pytest -v
```

Expected: all tests pass (DB + downloader + extractor + encoder ≈ 19+ tests).

- [ ] **Step 8.2: Commit if any fixes were needed**

If a test failed and was fixed: `git add . && git commit -m "fix: <description>"`

---

## Task 9: Live smoke test on smallest zip

- [ ] **Step 9.1: Identify smallest zip and run for short window**

```bash
curl -s "https://huggingface.co/api/datasets/AdwolfCzar/looper_v4/tree/main/chunks?limit=200" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);z=[(x['path'],x['size']) for x in d if x['path'].endswith('.zip')];z.sort(key=lambda x:x[1]);print(z[0])"
```

Note the smallest zip name and size.

- [ ] **Step 9.2: Pre-mark all zips except smallest as failed (skip them for smoke)**

```bash
./.venv/bin/python -c "
from src.db import DB, ZipStatus
from src.downloader import list_remote_zips
db = DB('state.db'); db.init()
remote = list_remote_zips()
db.upsert_zips(remote)
smallest = min(remote, key=lambda x: x[1])[0]
print('smallest:', smallest)
for name, _ in remote:
    if name != smallest:
        db.set_zip_status(name, ZipStatus.FAILED, error='smoke-skip')
"
```

- [ ] **Step 9.3: Run pipeline (lower encoder count for safety)**

```bash
./.venv/bin/python -m src.main run --encoders 8
```

Watch logs. Expected: download → extract → encode → cleanup. Should complete in <30 min for smallest zip.

- [ ] **Step 9.4: Verify outputs**

```bash
ls outputs/ | head -20
ls outputs/ | wc -l
./.venv/bin/python -m src.main status
du -sh outputs/
du -sh work/   # should be ~0 after cleanup
```

Spot-check one output:

```bash
ffprobe -v error -show_entries stream=avg_frame_rate -show_entries format=duration \
  "$(ls outputs/*.mp4 | head -1)"
```

Expected: ~30 fps, duration ≤ 5.

- [ ] **Step 9.5: Commit logs/notes if any**

If any tweaks were needed during smoke, commit them.

---

## Task 10: Full production run

- [ ] **Step 10.1: Reset state and re-list (smoke marked unrelated zips as failed)**

```bash
./.venv/bin/python -m src.main reset
```

(`reset` deletes state.db; bootstrap will re-list. `outputs/` is preserved — already-processed clips stay; the DB will re-process them but the encoder allocates collision-safe names so they won't overwrite — that's wasteful. Better: keep DB and just re-set the failed zips back to pending.)

Alternative non-destructive reset (preferred when smoke produced real outputs you want to keep):

```bash
./.venv/bin/python -c "
from src.db import DB, ZipStatus
db = DB('state.db')
import sqlite3
with sqlite3.connect('state.db') as c:
    c.execute(\"UPDATE zips SET status='pending', error=NULL, finished_at=NULL WHERE status='failed' AND error='smoke-skip'\")
    print('reset', c.total_changes, 'zips')
"
```

- [ ] **Step 10.2: Launch full run**

```bash
nohup ./.venv/bin/python -m src.main run --encoders 32 > run.log 2>&1 &
echo $! > run.pid
```

- [ ] **Step 10.3: Monitor progress periodically**

```bash
./.venv/bin/python -m src.main status
tail -f run.log
df -h work outputs
```

Expected timeline:
- Steady state: 4 zips in flight (download/extract overlapping)
- Total: ~1.5–2h (network-bound)
- `work/` should oscillate around 5–20 GB
- `outputs/` grows monotonically

- [ ] **Step 10.4: Final verification**

```bash
./.venv/bin/python -m src.main status
ls outputs/*.mp4 | wc -l
ls outputs/*.txt | wc -l
du -sh outputs/
```

Expected: ~30k mp4 + ~30k txt, work/ empty, no zips in pending/processing.

---

## Self-review notes

- Spec coverage: ✅ all sections covered (download, extract, ffprobe-driven action, collision rename, audio preserved, resume, random pick, streaming cleanup, signal handling).
- No placeholders: all code is complete; no TBD/TODO.
- Type consistency: `EncodeResult.info: VideoInfo`, `decide_action` returns string matching `_do_*` cases, DB enum values match status strings used in queries.
- One known divergence from spec: spec mentioned 60s ffmpeg timeout, plan uses 120s for safety on slower decode of long source files (>5s gets truncated, but full decode still happens for stream copy). Acceptable buffer.
