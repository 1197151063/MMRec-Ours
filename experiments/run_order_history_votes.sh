#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
data_root="${1:?Usage: run_order_history_votes.sh DATA_ROOT OUTPUT_JSON [DATASET]}"
result_path="${2:?Choose a fresh output JSON filename}"
dataset_name="${3:-baby}"
python -u experiments/probe_order_history_votes.py --data-path "$data_root" \
  --dataset "$dataset_name" --output "$result_path" --evaluate-test
exec python -u experiments/verify_order_votes_native.py --data-path "$data_root" \
  --dataset "$dataset_name" --report "$result_path" --output "${result_path%.json}.native.json"
