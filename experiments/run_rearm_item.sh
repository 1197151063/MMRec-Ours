#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
data="${1:?Usage: run_rearm_item.sh DATA_ROOT [GPU] [OUTPUT_ROOT]}"
gpu="${2:-0}"
output="${3:-night_runs/baby-rearm-item-1}"
for variant in original none alignment_0.01 alignment_0.1 alignment_1; do
  mode="${variant%%_*}"
  weight="${variant#*_}"
  if [[ "$mode" != alignment ]]; then weight=0; fi
  echo "Starting $variant"
  python -u experiments/run_rearm.py --data-path "$data" --gpu-id "$gpu" \
    --dataset baby --output "$output/$variant" --item-mode "$mode" --item-alignment-weight "$weight"
done
