import zipfile
from pathlib import Path

from src.db import DB, ZipStatus
from src.extractor import extract_and_register, find_video_txt_pairs


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
    by_name = {v.name: t for v, t in pairs}
    assert set(by_name) == {"a.mp4", "b.mp4"}
    assert by_name["a.mp4"] is not None and by_name["a.mp4"].name == "a.txt"
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
