#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec python -u experiments/run_night.py --suite freedom_time_anchor_no_ui \
  --data-path "${1:?Usage: run_freedom_time_anchor_no_ui.sh DATA_ROOT GPU OUTPUT [DATASET]}" \
  --gpu-id "${2:-0}" --output "${3:?Choose a fresh output directory}" \
  --dataset "${4:-baby}" --seeds 999 --epochs 1000 --hours 5 --job-minutes 90
