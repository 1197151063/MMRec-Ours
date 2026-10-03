#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
data="${1:?Usage: run_rearm_relations.sh DATA_ROOT [GPU] [OUTPUT] [DATASET]}"
gpu="${2:-0}"
dataset="${4:-baby}"
output="${3:-night_runs/${dataset}-rearm-relations-$(date +%Y%m%d-%H%M%S)-$$}"
echo "Output directory: $output"
exec python -u experiments/run_rearm_relations.py --data-path "$data" --gpu-id "$gpu" \
  --dataset "$dataset" --output "$output"
