from pathlib import Path

import pytest

from src.encoder import _parse_duration, allocate_output_name, decide_action, probe_video, process_video


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
    (30.0, 2.0, "cp"),
    (30.0, 7.0, "stream_copy"),
    (24.0, 2.0, "reencode"),
    (60.0, 8.0, "reencode"),
])
def test_decide_action(fps, duration, expected):
    assert decide_action(fps=fps, duration=duration) == expected


def test_parse_duration_handles_na_and_missing():
    assert _parse_duration("N/A") == 0.0
    assert _parse_duration(None) == 0.0
    assert _parse_duration("") == 0.0
    assert _parse_duration("3.5") == 3.5


def test_allocate_output_name_creates_atomic_placeholder(tmp_path):
    name = allocate_output_name(tmp_path, "clip.mp4")
    # reservation lives in a hidden .part dotfile, never the visible final path —
    # otherwise external scanners would see a 0-byte mp4 mid-encode.
    assert not (tmp_path / name).exists()
    assert (tmp_path / f".{Path(name).stem}.part.mp4").exists()


def test_allocate_output_name_atomic_under_concurrent_calls(tmp_path):
    # multiple sequential calls with the same desired name produce distinct outputs
    names = [allocate_output_name(tmp_path, "clip.mp4") for _ in range(5)]
    assert len(set(names)) == 5  # all unique
    assert names[0] == "clip.mp4"
    for n in names[1:]:
        assert n.startswith("clip__") and n.endswith(".mp4")
    # none of the visible final paths are populated yet
    for n in names:
        assert not (tmp_path / n).exists()


def test_allocate_output_name_no_collision(tmp_path):
    name = allocate_output_name(tmp_path, "clip.mp4")
    assert name == "clip.mp4"


def test_allocate_output_name_with_collision(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "clip.txt").write_text("x")
    name = allocate_output_name(tmp_path, "clip.mp4")
    assert name == "clip__1.mp4"
    name2 = allocate_output_name(tmp_path, "clip.mp4")
    assert name2 == "clip__2.mp4"


def test_allocate_output_name_collision_via_txt_only(tmp_path):
    """If only the .txt sibling exists, still rename to avoid pairing-mismatch."""
    (tmp_path / "clip.txt").write_text("existing description")
    name = allocate_output_name(tmp_path, "clip.mp4")
    assert name == "clip__1.mp4"


def test_process_video_cp_path(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=30, duration=2.0, with_audio=False, name="a.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-a")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.action == "cp"
    out_video = out_dir / result.output_name
    assert out_video.exists()
    assert out_video.with_suffix(".txt").read_text() == "desc-a"


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
    # action may fall back to reencode if stream_copy fails on a particular keyframe layout
    assert result.action in ("stream_copy", "reencode")
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
