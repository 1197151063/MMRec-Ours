#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:?Usage: bash experiments/run_profile.sh DATA_ROOT [GPU] [OUTPUT] [HOURS] [DATASET]}"
GPU_ID="${2:-0}"
OUTPUT="${3:-night_runs/baby-profile-1}"
HOURS="${4:-48}"
DATASET="${5:-baby}"
python experiments/diagnose_user_neighbors.py --data-path "$DATA_ROOT" --dataset "$DATASET" --output "${OUTPUT}-neighbors.json"
python -u experiments/run_night.py --suite profile --data-path "$DATA_ROOT" --dataset "$DATASET" \
  --gpu-id "$GPU_ID" --hours "$HOURS" --seeds 999 2024 2025 --output "$OUTPUT"
