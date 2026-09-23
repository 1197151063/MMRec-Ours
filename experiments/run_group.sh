#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:?Usage: bash experiments/run_group.sh DATA_ROOT [GPU] [OUTPUT] [HOURS] [DATASET]}"
python -u experiments/run_night.py --suite group --data-path "$DATA_ROOT" \
  --gpu-id "${2:-0}" --output "${3:-night_runs/baby-group-id-1}" \
  --hours "${4:-2}" --dataset "${5:-baby}" --seeds 999
