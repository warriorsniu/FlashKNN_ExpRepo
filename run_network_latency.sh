#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results/L20}"
OUT="$RESULTS_ROOT/$RUN_ID/network"
mkdir -p "$OUT"
WARMUPS=10; REPEATS=30; SAMPLES=68; LIDAR_SAMPLES=22
if [[ "${SMOKE:-0}" == "1" ]]; then WARMUPS=1; REPEATS=1; SAMPLES=1; LIDAR_SAMPLES=1; fi

PYTHONPATH="$REPO_DIR/FlashKNN" "$PYTHON_BIN" "$REPO_DIR/DeLA/S3DIS/benchmark_latency.py" \
  --data-root "$EXPREPO_S3DIS" --output "$OUT/dela_s3dis.json" --gpu "$GPU" \
  --warmups "$WARMUPS" --repeats "$REPEATS" --max-samples "$SAMPLES"

NETWORK_MODELS="${NETWORK_MODELS:-ptv3 octformer spunet minkunet34c}"
for MODEL in $NETWORK_MODELS; do
  POINTCEPT_ARGS=("$REPO_DIR/Pointcept/benchmark_latency.py" --model "$MODEL"
    --data-root "$EXPREPO_S3DIS" --output "$OUT/${MODEL}_s3dis.json" --gpu "$GPU"
    --warmups "$WARMUPS" --repeats "$REPEATS" --max-samples "$SAMPLES")
  PYTHONPATH="$REPO_DIR/Pointcept" "$PYTHON_BIN" "${POINTCEPT_ARGS[@]}"
done

if [[ -n "${EXPREPO_SEMANTICKITTI:-}" ]]; then
  for MODEL in dela deepla; do
    REPO_ARG="$REPO_DIR/DeLA"; [[ "$MODEL" == deepla ]] && REPO_ARG="$REPO_DIR/DeepLA-Net"
    PYTHONPATH="$REPO_DIR/FlashKNN:$REPO_DIR/Networks" "$PYTHON_BIN" "$REPO_DIR/Networks/benchmark_semantickitti_network.py" \
      --model "$MODEL" --repo "$REPO_ARG" --variant 24 --data-dir "$EXPREPO_SEMANTICKITTI" \
      --gpu "$GPU" --warmups "$WARMUPS" --repeats "$REPEATS" --max-samples "$LIDAR_SAMPLES" \
      --output "$OUT/${MODEL}_semantickitti_backends.json"
  done
  LIDAR_NETWORK_MODELS="${LIDAR_NETWORK_MODELS:-ptv3 octformer spunet minkunet34c}"
  for MODEL in $LIDAR_NETWORK_MODELS; do
    PYTHONPATH="$REPO_DIR/Pointcept:$REPO_DIR/Networks" "$PYTHON_BIN" \
      "$REPO_DIR/Networks/benchmark_semantickitti_pointcept.py" \
      --model "$MODEL" --pointcept-root "$REPO_DIR/Pointcept" \
      --data-dir "$EXPREPO_SEMANTICKITTI" --gpu "$GPU" \
      --warmups "$WARMUPS" --repeats "$REPEATS" --max-samples "$LIDAR_SAMPLES" \
      --output "$OUT/${MODEL}_semantickitti.json"
  done
fi
echo "$OUT"
