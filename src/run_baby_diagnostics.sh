#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:?Usage: bash run_baby_diagnostics.sh /absolute/path/to/data [gpu_id]}"
GPU_ID="${2:-0}"
python main.py --model LightMRecNoPE --dataset baby --data-path "$DATA_ROOT" --gpu-id "$GPU_ID"
python main.py --model SIMMRec --dataset baby --data-path "$DATA_ROOT" --gpu-id "$GPU_ID" --config configs/simmrec-content-diagnostic.yaml
python main.py --model SIMMRec --dataset baby --data-path "$DATA_ROOT" --gpu-id "$GPU_ID" --config configs/simmrec-id-diagnostic.yaml
