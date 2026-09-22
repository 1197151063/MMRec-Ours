#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:?Usage: bash experiments/run_init.sh /absolute/data/root [gpu_id] [output_dir]}"
GPU_ID="${2:-0}"
OUTPUT="${3:-night_runs/baby-init-1}"
python experiments/diagnose_user_neighbors.py --data-path "$DATA_ROOT" --output "${OUTPUT}-neighbors.json"
python -u experiments/run_night.py --suite init --data-path "$DATA_ROOT" --dataset baby \
  --gpu-id "$GPU_ID" --hours 8 --output "$OUTPUT"
