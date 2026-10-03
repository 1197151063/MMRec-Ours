#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# This is a CPU diagnostic; bound BLAS threads to avoid oversubscription.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec python -u experiments/research_user_cohorts.py \
  --data-path "${1:?Usage: run_user_cohorts.sh DATA_ROOT OUTPUT [DATASET]}" \
  --output "${2:?Choose a fresh output directory}" --dataset "${3:-baby}" \
  --pairs 4000 --queries 2048 --offsets 1 4 16 64 --far-distance 256 \
  --dim 64 --ks 20 80 --mask-seeds 2026 2027 2028
