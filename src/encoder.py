from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MAX_DURATION = 5.0
TARGET_FPS = 30
FPS_TOLERANCE = 0.05
DURATION_TOLERANCE = 0.05
FFMPEG_TIMEOUT = 300
LIBX264_THREADS = 2


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


def _parse_duration(raw) -> float:
    """ffprobe may return 'N/A' or missing duration."""
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
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
    duration = _parse_duration(fmt.get("duration")) or _parse_duration(video.get("duration"))
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
    """Atomically reserve a non-colliding filename via O_EXCL.

    Creates an empty placeholder so concurrent encoders see it. Caller must overwrite
    via tmp.replace(). Considers .txt sibling for fallback rename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = Path(desired)
    stem, ext = p.stem, p.suffix
    n = 0
    while True:
        candidate = desired if n == 0 else f"{stem}__{n}{ext}"
        v = output_dir / candidate
        t = v.with_suffix(".txt")
        if t.exists():
            n += 1
            continue
        try:
            fd = os.open(str(v), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return candidate


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
        "-vf", f"fps={TARGET_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-t", str(MAX_DURATION),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-threads", str(LIBX264_THREADS),
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

    tmp = out_video.with_name(f".{out_video.stem}.part{out_video.suffix}")
    try:
        if action == "cp":
            _do_cp(video, tmp)
        elif action == "stream_copy":
            try:
                _do_stream_copy(video, tmp)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if tmp.exists():
                    tmp.unlink()
                _do_reencode(video, tmp, has_audio=info.has_audio)
                action = "reencode"
        else:
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
