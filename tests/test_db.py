import sqlite3

from src.db import DB, ZipStatus


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
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("a.zip", 1), ("b.zip", 1)])
    db.set_zip_status("a.zip", ZipStatus.DOWNLOADING)
    db.set_zip_status("b.zip", ZipStatus.EXTRACTING)
    db.reset_stale_zips(zip_files_present=set())
    statuses = {z["name"]: z["status"] for z in db.all_zips()}
    assert statuses["a.zip"] == ZipStatus.PENDING.value
    assert statuses["b.zip"] == ZipStatus.PENDING.value


def test_reset_stale_unblocks_downloaded_zip(tmp_db):
    """A zip stuck in DOWNLOADED (downloader crashed before queueing) must be re-claimed."""
    db = DB(tmp_db)
    db.init()
    db.upsert_zips([("a.zip", 1)])
    db.set_zip_status("a.zip", ZipStatus.DOWNLOADED)
    db.reset_stale_zips(zip_files_present={"a.zip"})
    statuses = {z["name"]: z["status"] for z in db.all_zips()}
    assert statuses["a.zip"] == ZipStatus.PENDING.value
