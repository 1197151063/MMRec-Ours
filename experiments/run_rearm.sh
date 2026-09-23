#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -u experiments/run_rearm.py --data-path "${1:?Usage: run_rearm.sh DATA_ROOT [GPU] [OUTPUT] [DATASET]}" \
  --gpu-id "${2:-0}" --output "${3:-night_runs/baby-rearm-official-1}" --dataset "${4:-baby}"
