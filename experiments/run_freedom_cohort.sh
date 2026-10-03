#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec python -u experiments/run_night.py --suite freedom_cohort \
  --data-path "${1:?Usage: run_freedom_cohort.sh DATA_ROOT GPU OUTPUT [DATASET]}" \
  --gpu-id "${2:-0}" --output "${3:?Choose an output directory}" \
  --dataset "${4:-baby}" --seeds 999 --epochs 1000 --hours 8 --job-minutes 90
