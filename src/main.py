from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.db import DB
from src.pipeline import Config, run as run_pipeline


def _default_config(args) -> Config:
    return Config(
        workdir=Path(args.workdir).resolve(),
        output_dir=Path(args.outputs).resolve(),
        db_path=Path(args.db).resolve(),
        n_downloaders=args.downloaders,
        n_extractors=args.extractors,
        n_encoders=args.encoders,
    )


def cmd_run(args) -> int:
    cfg = _default_config(args)
    run_pipeline(cfg)
    return 0


def cmd_status(args) -> int:
    db = DB(Path(args.db))
    p = db.overall_progress()
    print(f"zips done:   {p['zips_done']}/{p['zips_total']}")
    print(f"videos done: {p['videos_done']}")
    print(f"videos fail: {p['videos_failed']}")
    print(f"videos tot:  {p['videos_total']}")
    return 0


def cmd_reset(args) -> int:
    p = Path(args.db)
    for q in (p, Path(str(p) + "-wal"), Path(str(p) + "-shm")):
        if q.exists():
            q.unlink()
    print(f"removed {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="video_fix_loop")
    ap.add_argument("--db", default="state.db")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--outputs", default="outputs")

    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="start the pipeline")
    runp.add_argument("--downloaders", type=int, default=2)
    runp.add_argument("--extractors", type=int, default=2)
    runp.add_argument("--encoders", type=int, default=int(os.environ.get("VFL_ENCODERS", "48")))
    runp.set_defaults(func=cmd_run)

    sub.add_parser("status", help="print progress").set_defaults(func=cmd_status)
    sub.add_parser("reset", help="delete state.db").set_defaults(func=cmd_reset)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
