#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${SMOKE:-0}" != "1" && "${ALLOW_FORMAL_ADAPTIVE_NEIGHBORHOOD:-0}" != "1" ]]; then
  echo "Formal adaptive-neighborhood ablation is locked." >&2
  echo "Set ALLOW_FORMAL_ADAPTIVE_NEIGHBORHOOD=1 after explicit authorization." >&2
  exit 2
fi

if [[ "${SMOKE:-0}" != "1" ]]; then
  TARGET_GPU_UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | xargs)"
  if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader \
      | grep -F "$TARGET_GPU_UUID,"; then
    echo "Formal adaptive-neighborhood ablation requires an idle target GPU." >&2
    exit 3
  fi
fi

source "$REPO_DIR/scripts/results_env.sh"
OUT="$RESULTS_ROOT/$RUN_ID/adaptive_neighborhood"
mkdir -p "$OUT"
WARMUPS="${ADAPTIVE_WARMUPS:-5}"
REPEATS="${ADAPTIVE_REPEATS:-20}"
EXTRA_ARGS=()
VALIDATE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  WARMUPS="${ADAPTIVE_WARMUPS:-1}"
  REPEATS="${ADAPTIVE_REPEATS:-1}"
  EXTRA_ARGS+=(--max-samples 1)
  VALIDATE_ARGS+=(--allow-partial)
fi

PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$REPO_DIR/Query/benchmark_adaptive_neighborhood.py" \
  --data-root "$EXPREPO_S3DIS" \
  --output "$OUT/s3dis_adaptive_neighborhood.json" \
  --gpu "$GPU" \
  --k 8 16 24 32 48 64 \
  --adaptive-min-factor "${ADAPTIVE_MIN_FACTOR:-2}" \
  --adaptive-max-factor "${ADAPTIVE_MAX_FACTOR:-8}" \
  --warmups "$WARMUPS" \
  --repeats "$REPEATS" \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" "$REPO_DIR/scripts/validate_adaptive_neighborhood.py" \
  "$OUT/s3dis_adaptive_neighborhood.json" "${VALIDATE_ARGS[@]}"

echo "Result: $OUT/s3dis_adaptive_neighborhood.json"
