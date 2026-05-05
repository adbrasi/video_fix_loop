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
