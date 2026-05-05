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

MAX_DURATION = 8.0
MIN_LEFTOVER_KEEP = 5.0
TARGET_FPS = 25
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
class Segment:
    index: int
    start: float
    length: float


@dataclass(frozen=True)
class EncodeResult:
    action: str          # cp | reencode | discarded
    output_names: list[str]
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


def plan_segments(duration: float) -> list[Segment]:
    """Plan output segments for a source of given duration.

    Rules:
      - duration <= 0: return []  (discard)
      - duration <= MAX_DURATION (+ tol): single segment spanning the whole video
      - duration  > MAX_DURATION: chunk into MAX_DURATION pieces; final leftover
        kept only if >= MIN_LEFTOVER_KEEP, else discarded
    """
    if duration <= 0:
        return []
    if duration <= MAX_DURATION + DURATION_TOLERANCE:
        return [Segment(index=0, start=0.0, length=duration)]
    segs: list[Segment] = []
    pos = 0.0
    idx = 0
    while pos + MAX_DURATION <= duration:
        segs.append(Segment(index=idx, start=pos, length=MAX_DURATION))
        pos += MAX_DURATION
        idx += 1
    leftover = duration - pos
    if leftover >= MIN_LEFTOVER_KEEP:
        segs.append(Segment(index=idx, start=pos, length=leftover))
    return segs


def decide_action(*, fps: float, duration: float) -> str:
    """Action for a single-segment-full-video case.

    For multi-segment videos the encoder always re-encodes (keyframe accuracy on cuts).
    """
    near_target = abs(fps - TARGET_FPS) <= FPS_TOLERANCE
    short = duration <= MAX_DURATION + DURATION_TOLERANCE
    if near_target and short:
        return "cp"
    return "reencode"


def allocate_output_name(output_dir: Path, desired: str) -> str:
    """Atomically reserve a non-colliding filename via O_EXCL on a hidden .part file.

    The reservation is the hidden tmp file `.{stem}.part{ext}` (same path the encoder
    writes ffmpeg/cp output into), so the visible final filename never appears empty
    to external scanners. Caller writes into the .part path then atomically renames
    to the returned candidate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = Path(desired)
    stem, ext = p.stem, p.suffix
    n = 0
    while True:
        candidate = desired if n == 0 else f"{stem}__{n}{ext}"
        cand = Path(candidate)
        v = output_dir / candidate
        t = v.with_suffix(".txt")
        part = output_dir / f".{cand.stem}.part{ext}"
        if v.exists() or t.exists():
            n += 1
            continue
        try:
            fd = os.open(str(part), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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


def _do_reencode_segment(src: Path, dst: Path, *, start: float, length: float, has_audio: bool) -> None:
    """Re-encode a [start, start+length] window of src into dst at TARGET_FPS."""
    args: list[str] = []
    if start > 0.0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(src)]
    args += [
        "-vf", f"fps={TARGET_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-t", f"{length:.3f}",
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


def _emit_one(*, video: Path, txt: Optional[Path], output_dir: Path,
              dest_basename: str, action: str,
              start: float, length: float, full_duration: float,
              has_audio: bool) -> str:
    """Write one output (cp or reencoded segment) atomically. Returns final name."""
    out_name = allocate_output_name(output_dir, dest_basename)
    out_video = output_dir / out_name
    out_txt = out_video.with_suffix(".txt")
    tmp = out_video.with_name(f".{out_video.stem}.part{out_video.suffix}")
    tmp_txt = out_video.with_name(f".{out_video.stem}.part.txt")
    try:
        if action == "cp":
            _do_cp(video, tmp)
        else:
            _do_reencode_segment(video, tmp, start=start, length=length, has_audio=has_audio)
        if txt is not None and txt.exists():
            shutil.copyfile(txt, tmp_txt)
            tmp_txt.replace(out_txt)
        tmp.replace(out_video)
    finally:
        for p in (tmp, tmp_txt):
            if p.exists():
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
    return out_name


def _rollback_outputs(output_dir: Path, names: list[str]) -> None:
    """Delete previously-emitted segment outputs (and their .txt siblings)."""
    for n in names:
        v = output_dir / n
        t = v.with_suffix(".txt")
        for p in (v, t):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


def process_video(*, video: Path, txt: Optional[Path], output_dir: Path) -> EncodeResult:
    info = probe_video(video)
    segments = plan_segments(info.duration)
    if not segments:
        return EncodeResult(action="discarded", output_names=[], info=info)

    # Single segment that spans the full video → may use cp shortcut
    if len(segments) == 1 and abs(segments[0].length - info.duration) <= DURATION_TOLERANCE:
        action = decide_action(fps=info.fps, duration=info.duration)
        out_name = _emit_one(
            video=video, txt=txt, output_dir=output_dir,
            dest_basename=video.name,
            action=action,
            start=0.0, length=info.duration, full_duration=info.duration,
            has_audio=info.has_audio,
        )
        return EncodeResult(action=action, output_names=[out_name], info=info)

    # Multi-segment (or partial-window single segment): always reencode each piece
    src_stem = Path(video.name).stem
    src_ext = Path(video.name).suffix
    names: list[str] = []
    try:
        for seg in segments:
            seg_basename = f"{src_stem}_s{seg.index:02d}{src_ext}"
            name = _emit_one(
                video=video, txt=txt, output_dir=output_dir,
                dest_basename=seg_basename,
                action="reencode",
                start=seg.start, length=seg.length, full_duration=info.duration,
                has_audio=info.has_audio,
            )
            names.append(name)
    except Exception:
        _rollback_outputs(output_dir, names)
        raise
    return EncodeResult(action="reencode", output_names=names, info=info)
