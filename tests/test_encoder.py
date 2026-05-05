from pathlib import Path

import pytest

from src.encoder import (
    MAX_DURATION,
    MIN_LEFTOVER_KEEP,
    TARGET_FPS,
    _parse_duration,
    allocate_output_name,
    decide_action,
    plan_segments,
    probe_video,
    process_video,
)


def test_constants_match_spec():
    assert TARGET_FPS == 25
    assert MAX_DURATION == 8.0
    assert MIN_LEFTOVER_KEEP == 5.0


def test_probe_target_fps_no_audio(synthetic_video_factory):
    p = synthetic_video_factory(fps=TARGET_FPS, duration=2.0, with_audio=False)
    info = probe_video(p)
    assert abs(info.fps - TARGET_FPS) < 0.1
    assert 1.5 <= info.duration <= 2.5
    assert info.has_audio is False


def test_probe_24fps_with_audio(synthetic_video_factory):
    p = synthetic_video_factory(fps=24, duration=1.0, with_audio=True)
    info = probe_video(p)
    assert abs(info.fps - 24.0) < 0.1
    assert info.has_audio is True


@pytest.mark.parametrize("fps,duration,expected", [
    (25.0, 2.0, "cp"),         # right fps + short → cp
    (25.0, 8.0, "cp"),          # exactly at the limit
    (30.0, 2.0, "reencode"),    # wrong fps
    (24.0, 5.0, "reencode"),    # wrong fps
])
def test_decide_action_single_segment(fps, duration, expected):
    assert decide_action(fps=fps, duration=duration) == expected


@pytest.mark.parametrize("duration,expected_starts,expected_lengths", [
    (0.0,  [],           []),                      # invalid → discard
    (0.5,  [0.0],        [0.5]),                   # tiny but kept (no cut)
    (4.0,  [0.0],        [4.0]),                   # short → 1 segment full
    (8.0,  [0.0],        [8.0]),                   # exactly limit → 1 segment
    (8.04, [0.0],        [8.04]),                  # within tolerance, no cut
    (10.0, [0.0],        [8.0]),                   # cut + leftover 2 < 5 → discard
    (12.9, [0.0],        [8.0]),                   # cut + leftover 4.9 < 5 → discard
    (13.0, [0.0, 8.0],   [8.0, 5.0]),              # cut + leftover 5 ≥ 5 → keep
    (16.0, [0.0, 8.0],   [8.0, 8.0]),              # 2 full chunks
    (20.0, [0.0, 8.0],   [8.0, 8.0]),              # 2 chunks + leftover 4 → discard
    (21.0, [0.0, 8.0, 16.0], [8.0, 8.0, 5.0]),     # 2 chunks + leftover 5 → keep
    (24.0, [0.0, 8.0, 16.0], [8.0, 8.0, 8.0]),     # 3 full chunks
])
def test_plan_segments(duration, expected_starts, expected_lengths):
    segs = plan_segments(duration)
    assert [s.start for s in segs] == pytest.approx(expected_starts, abs=1e-3)
    assert [s.length for s in segs] == pytest.approx(expected_lengths, abs=1e-3)
    # indices are sequential
    assert [s.index for s in segs] == list(range(len(segs)))


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
    names = [allocate_output_name(tmp_path, "clip.mp4") for _ in range(5)]
    assert len(set(names)) == 5
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
    (tmp_path / "clip.txt").write_text("existing description")
    name = allocate_output_name(tmp_path, "clip.mp4")
    assert name == "clip__1.mp4"


def test_process_video_cp_path_short_already_target_fps(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=TARGET_FPS, duration=2.0, with_audio=False, name="a.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-a")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.action == "cp"
    assert len(result.output_names) == 1
    out_video = out_dir / result.output_names[0]
    assert out_video.exists()
    assert out_video.with_suffix(".txt").read_text() == "desc-a"


def test_process_video_reencode_fps_mismatch(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=24, duration=2.0, with_audio=True, name="b.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-b")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.action == "reencode"
    assert len(result.output_names) == 1
    new_info = probe_video(out_dir / result.output_names[0])
    assert abs(new_info.fps - TARGET_FPS) < 1.5
    assert new_info.duration <= MAX_DURATION + 0.5


def test_process_video_chunks_long_video(synthetic_video_factory, tmp_path):
    """13s @ 25fps → 2 outputs: 8s + 5s."""
    src = synthetic_video_factory(fps=TARGET_FPS, duration=13.0, with_audio=False, name="long.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=None, output_dir=out_dir)
    assert result.action == "reencode"
    assert len(result.output_names) == 2
    # naming convention: stem_sNN.ext
    assert result.output_names[0] == "long_s00.mp4"
    assert result.output_names[1] == "long_s01.mp4"
    info0 = probe_video(out_dir / result.output_names[0])
    info1 = probe_video(out_dir / result.output_names[1])
    assert info0.duration == pytest.approx(8.0, abs=0.5)
    assert info1.duration == pytest.approx(5.0, abs=0.5)
    assert abs(info0.fps - TARGET_FPS) < 1.5
    assert abs(info1.fps - TARGET_FPS) < 1.5


def test_process_video_discards_short_leftover(synthetic_video_factory, tmp_path):
    """10s → 1 output of 8s; leftover 2s discarded."""
    src = synthetic_video_factory(fps=TARGET_FPS, duration=10.0, with_audio=False, name="ten.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=None, output_dir=out_dir)
    assert len(result.output_names) == 1
    assert result.output_names[0] == "ten_s00.mp4"
    info = probe_video(out_dir / result.output_names[0])
    assert info.duration == pytest.approx(8.0, abs=0.5)


def test_process_video_chunks_carry_txt_sibling(synthetic_video_factory, tmp_path):
    """Each chunk gets its own .txt copy when source has a sidecar."""
    src = synthetic_video_factory(fps=TARGET_FPS, duration=13.0, with_audio=False, name="cap.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("description for the whole video")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert len(result.output_names) == 2
    for n in result.output_names:
        sib = (out_dir / n).with_suffix(".txt")
        assert sib.read_text() == "description for the whole video"


def test_process_video_zero_duration_returns_discarded(tmp_path, monkeypatch):
    """If ffprobe yields duration 0 (corrupt source), process_video returns discarded."""
    fake_video = tmp_path / "broken.mp4"
    fake_video.write_bytes(b"")  # just to make path exist
    from src import encoder as enc

    def fake_probe(_path):
        return enc.VideoInfo(fps=25.0, duration=0.0, has_audio=False)

    monkeypatch.setattr(enc, "probe_video", fake_probe)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = enc.process_video(video=fake_video, txt=None, output_dir=out_dir)
    assert result.action == "discarded"
    assert result.output_names == []
    # nothing leaked into outputs/
    assert list(out_dir.iterdir()) == []


def test_process_video_without_txt_sibling(synthetic_video_factory, tmp_path):
    """Datasets without .txt sidecars produce only the .mp4."""
    src = synthetic_video_factory(fps=TARGET_FPS, duration=2.0, with_audio=False, name="e.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = process_video(video=src, txt=None, output_dir=out_dir)
    out_video = out_dir / result.output_names[0]
    assert out_video.exists()
    assert not out_video.with_suffix(".txt").exists()
    # no leftover hidden tmp files
    assert list(out_dir.glob(".*.part.*")) == []


def test_process_video_collision_renames_to_suffix(synthetic_video_factory, tmp_path):
    src = synthetic_video_factory(fps=TARGET_FPS, duration=2.0, with_audio=False, name="d.mp4")
    txt = src.with_suffix(".txt")
    txt.write_text("desc-d")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "d.mp4").write_bytes(b"existing")
    (out_dir / "d.txt").write_text("existing")
    result = process_video(video=src, txt=txt, output_dir=out_dir)
    assert result.output_names == ["d__1.mp4"]
    assert (out_dir / "d__1.mp4").exists()
    assert (out_dir / "d__1.txt").read_text() == "desc-d"
