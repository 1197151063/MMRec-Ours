#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:?Usage: bash run_order_audit.sh /absolute/path/to/data [gpu_id]}"
GPU_ID="${2:-0}"
# A fresh no-PE control, then 4 coordinate sources x 3 sides. Checkpoints remain disabled.
python main.py --model LightMRecOrder --dataset baby --data-path "$DATA_ROOT" --gpu-id "$GPU_ID" --config configs/order-audit-none.yaml
python main.py --model LightMRecOrder --dataset baby --data-path "$DATA_ROOT" --gpu-id "$GPU_ID"
