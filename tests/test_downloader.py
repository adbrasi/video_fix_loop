from unittest.mock import patch

from src.downloader import REPO_ID, list_remote_zips


def test_list_remote_zips_filters_zip_files():
    fake_tree = [
        {"type": "file", "path": ".gitattributes", "size": 2504},
        {"type": "file", "path": "residuals_part01.zip", "size": 8000},
        {"type": "file", "path": "residuals_part02.zip", "size": 9000},
        {"type": "file", "path": "README.md", "size": 100},
    ]
    with patch("src.downloader._raw_tree", return_value=fake_tree):
        items = list_remote_zips()
    assert items == [("residuals_part01.zip", 8000), ("residuals_part02.zip", 9000)]


def test_repo_id_constant():
    assert REPO_ID == "AdwolfCzar/residals_dataset_base_1"
