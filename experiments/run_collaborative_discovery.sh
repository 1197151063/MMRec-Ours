#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
data_root="${1:?Usage: run_collaborative_discovery.sh DATA_ROOT NEW_OUTPUT_DIR [DATASET]}"
output_dir="${2:?Choose a new output directory}"
dataset_name="${3:-baby}"
if [ -e "$output_dir" ]; then
  echo 'Output already exists; choose a fresh directory' >&2
  exit 1
fi
mkdir -p "$output_dir"
for probe in collaborative_neighbors collaborative_anchors untruncated_paths semantic_history_retrieval; do
  echo "Running TRAIN-only probe: $probe"
  python -u "experiments/probe_${probe}.py" --data-path "$data_root" \
    --dataset "$dataset_name" --output "$output_dir/${probe}.json" \
    > "$output_dir/${probe}.log" 2>&1
  tail -n 2 "$output_dir/${probe}.log"
done
echo 'Running TRAIN-only history MLP to validation early stopping'
python -u experiments/probe_history_mlp.py --data-path "$data_root" \
  --dataset "$dataset_name" --hidden 64 --epochs 200 --dropout .3 .6 \
  --threads 2 --patience 6 --max-seconds 290 --output "$output_dir/history_mlp.json" \
  > "$output_dir/history_mlp.log" 2>&1
tail -n 2 "$output_dir/history_mlp.log"
echo 'Running separately labeled POST-HOC order coverage audit (not model construction)'
python -u experiments/audit_order_signal_coverage.py --data-path "$data_root" \
  --dataset "$dataset_name" --output "$output_dir/order_coverage_audit.json" \
  > "$output_dir/order_coverage_audit.log" 2>&1
echo "Finished. Reports and logs: $output_dir"
