from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

from src.db import DB, ZipStatus
from src.downloader import download_zip, list_remote_zips
from src.encoder import process_video
from src.extractor import extract_and_register

log = logging.getLogger(__name__)


@dataclass
class Config:
    workdir: Path
    output_dir: Path
    db_path: Path
    n_downloaders: int = 2
    n_extractors: int = 2
    n_encoders: int = 32
    download_queue_max: int = 4
    encode_queue_max: int = 256


def _signal_handler(stop_event: mp.Event):
    def handler(signum, _frame):
        log.warning("signal %s received → graceful shutdown", signum)
        stop_event.set()
    return handler


def _downloader_loop(cfg: Config, dl_queue: mp.Queue, stop: mp.Event):
    db = DB(cfg.db_path)
    while not stop.is_set():
        zip_row = db.claim_random_zip(source=ZipStatus.PENDING, target=ZipStatus.DOWNLOADING)
        if zip_row is None:
            time.sleep(2)
            if not db.list_zips_by_status(ZipStatus.PENDING):
                break
            continue
        name = zip_row["name"]
        try:
            log.info("download start: %s", name)
            local = download_zip(name, cfg.workdir / "zips")
            db.set_zip_status(name, ZipStatus.DOWNLOADED)
            dl_queue.put((name, str(local)))
        except Exception as e:
            log.exception("download failed: %s", name)
            db.set_zip_status(name, ZipStatus.FAILED, error=f"download: {e}")


def _extractor_loop(cfg: Config, dl_queue: mp.Queue, ex_queue: mp.Queue, stop: mp.Event):
    db = DB(cfg.db_path)
    while not stop.is_set():
        try:
            item = dl_queue.get(timeout=2)
        except Empty:
            continue
        if item is None:
            break
        zip_name, local_path = item
        try:
            log.info("extract start: %s", zip_name)
            extracted_dir = extract_and_register(
                zip_path=Path(local_path),
                zip_name=zip_name,
                work_dir=cfg.workdir / "extracted",
                db=db,
            )
            for video_row in db.list_pending_videos_for_zip(zip_name):
                ex_queue.put({
                    "zip_name": zip_name,
                    "internal_path": video_row["internal_path"],
                    "extracted_dir": str(extracted_dir),
                    "local_zip": local_path,
                })
        except Exception as e:
            log.exception("extract failed: %s", zip_name)
            db.set_zip_status(zip_name, ZipStatus.FAILED, error=f"extract: {e}")


def _encoder_loop(cfg: Config, ex_queue: mp.Queue, stop: mp.Event):
    db = DB(cfg.db_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        try:
            task = ex_queue.get(timeout=2)
        except Empty:
            continue
        if task is None:
            break
        zip_name = task["zip_name"]
        internal = task["internal_path"]
        extracted = Path(task["extracted_dir"])
        video = extracted / internal
        txt_candidate = video.with_suffix(".txt")
        txt = txt_candidate if txt_candidate.exists() else None
        try:
            result = process_video(video=video, txt=txt, output_dir=cfg.output_dir)
            db.set_video_done(
                zip_name, internal,
                output_name=result.output_name,
                action=result.action,
                fps_in=result.info.fps,
                duration_in=result.info.duration,
                has_audio=result.info.has_audio,
            )
        except Exception as e:
            log.exception("encode failed: %s :: %s", zip_name, internal)
            db.set_video_failed(zip_name, internal, str(e))


def _cleaner_loop(cfg: Config, stop: mp.Event):
    """Periodically delete fully-processed zips and their extracted dirs."""
    db = DB(cfg.db_path)
    while not stop.is_set():
        time.sleep(5)
        for z in db.list_zips_by_status(ZipStatus.PROCESSING):
            name = z["name"]
            prog = db.zip_progress(name)
            if prog["total"] == 0 or prog["pending"] > 0:
                continue
            base = Path(name).stem
            local_zip = cfg.workdir / "zips" / name
            ext_dir = cfg.workdir / "extracted" / base
            if local_zip.exists():
                try:
                    local_zip.unlink()
                except OSError:
                    pass
            if ext_dir.exists():
                shutil.rmtree(ext_dir, ignore_errors=True)
            if prog["failed"] == 0:
                db.set_zip_status(name, ZipStatus.DONE)
            else:
                db.set_zip_status(name, ZipStatus.FAILED,
                                  error=f"{prog['failed']} videos failed")
            log.info("cleaned %s (done=%d failed=%d)", name, prog["done"], prog["failed"])


def _progress_loop(cfg: Config, stop: mp.Event):
    db = DB(cfg.db_path)
    started = time.time()
    while not stop.is_set():
        time.sleep(15)
        p = db.overall_progress()
        elapsed = int(time.time() - started)
        log.info("PROGRESS zips %d/%d  videos done=%d failed=%d total=%d  elapsed=%ds",
                 p["zips_done"], p["zips_total"],
                 p["videos_done"], p["videos_failed"], p["videos_total"],
                 elapsed)


def _bootstrap_state(cfg: Config) -> None:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / "zips").mkdir(parents=True, exist_ok=True)
    (cfg.workdir / "extracted").mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    db = DB(cfg.db_path)
    db.init()

    log.info("listing remote zips...")
    remote = list_remote_zips()
    log.info("found %d remote zips (%.1f GB total)",
             len(remote), sum(s for _, s in remote) / 1e9)
    db.upsert_zips(remote)

    present = {p.name for p in (cfg.workdir / "zips").rglob("*.zip")}
    db.reset_stale_zips(zip_files_present=present)


def run(cfg: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s/%(levelname)s] %(message)s",
    )
    _bootstrap_state(cfg)

    stop: mp.Event = mp.Event()
    handler = _signal_handler(stop)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    dl_queue: mp.Queue = mp.Queue(maxsize=cfg.download_queue_max)
    ex_queue: mp.Queue = mp.Queue(maxsize=cfg.encode_queue_max)

    procs: list[mp.Process] = []

    for i in range(cfg.n_downloaders):
        p = mp.Process(target=_downloader_loop, args=(cfg, dl_queue, stop), name=f"dl-{i}")
        p.start(); procs.append(p)
    for i in range(cfg.n_extractors):
        p = mp.Process(target=_extractor_loop, args=(cfg, dl_queue, ex_queue, stop), name=f"ex-{i}")
        p.start(); procs.append(p)
    for i in range(cfg.n_encoders):
        p = mp.Process(target=_encoder_loop, args=(cfg, ex_queue, stop), name=f"enc-{i}")
        p.start(); procs.append(p)

    cleaner = threading.Thread(target=_cleaner_loop, args=(cfg, stop), name="cleaner", daemon=True)
    cleaner.start()
    progress = threading.Thread(target=_progress_loop, args=(cfg, stop), name="progress", daemon=True)
    progress.start()

    db = DB(cfg.db_path)
    while not stop.is_set():
        time.sleep(15)
        p = db.overall_progress()
        n_failed = sum(1 for z in db.all_zips() if z["status"] == "failed")
        if (p["zips_total"] - p["zips_done"] - n_failed) == 0 and dl_queue.empty() and ex_queue.empty():
            log.info("all work done")
            stop.set()
            break

    for _ in range(cfg.n_extractors):
        try: dl_queue.put_nowait(None)
        except Exception: pass
    for _ in range(cfg.n_encoders):
        try: ex_queue.put_nowait(None)
        except Exception: pass

    for p in procs:
        p.join(timeout=180)
        if p.is_alive():
            log.warning("forcing terminate: %s", p.name)
            p.terminate()
    log.info("shutdown complete")
