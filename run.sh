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

# XET tuning: saturate 100 MB/s line on RunPod (uses Tokio runtime + parallel range GETs)
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-32}"
export HF_XET_CLIENT_MAX_IDLE_CONNECTIONS="${HF_XET_CLIENT_MAX_IDLE_CONNECTIONS:-64}"
# 10 GB chunk cache for resume across kills (XET deduplicates by chunk)
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-10737418240}"

exec ./.venv/bin/python -m src.main "$@"
