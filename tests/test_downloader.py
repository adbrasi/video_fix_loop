from unittest.mock import patch

from src.downloader import REPO_ID, list_remote_zips


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
