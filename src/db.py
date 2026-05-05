from __future__ import annotations

import enum
import sqlite3
import threading
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
        """Atomically pick a random zip in `source` state and move to `target`."""
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
        """On startup, revert in-flight zip states.
        - downloading → pending
        - extracting/processing → downloaded if zip on disk, else pending
        """
        with self._conn() as c:
            c.execute("UPDATE zips SET status='pending' WHERE status='downloading'")
            for row in c.execute("SELECT name FROM zips WHERE status IN ('extracting','processing')").fetchall():
                target = "downloaded" if row["name"] in zip_files_present else "pending"
                c.execute("UPDATE zips SET status=? WHERE name=?", (target, row["name"]))

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
                       SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                       SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                       SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending
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
