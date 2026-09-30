#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u experiments/run_night.py --suite freedom_align \
  --data-path "${1:?Usage: run_freedom_align.sh DATA_ROOT [GPU] [OUTPUT] [DATASET]}" \
  --gpu-id "${2:-0}" --output "${3:-night_runs/baby-freedom-align-1}" \
  --dataset "${4:-baby}" --seeds 999 --hours 12 --job-minutes 180 --epochs 1000
