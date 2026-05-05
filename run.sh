#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg/ffprobe not found. Install with: sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python -m src.main "$@"
