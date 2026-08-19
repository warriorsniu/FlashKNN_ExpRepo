#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
source "$REPO_DIR/scripts/results_env.sh"
OUT="$RESULTS_ROOT/$RUN_ID/ablation"
mkdir -p "$OUT"

K_VALUES=(8 16 24 32 40 48 56 64)
WARMUPS="${ABLATION_WARMUPS:-5}"
REPEATS="${ABLATION_REPEATS:-20}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  K_VALUES=(8 32 64)
  WARMUPS="${ABLATION_WARMUPS:-1}"
  REPEATS="${ABLATION_REPEATS:-2}"
  EXTRA_ARGS+=(--max-samples 1)
fi

PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$REPO_DIR/Query/benchmark_ablation.py" \
  --data-root "$EXPREPO_S3DIS" \
  --output "$OUT/s3dis_design_ablation.json" \
  --gpu "$GPU" \
  --k "${K_VALUES[@]}" \
  --warmups "$WARMUPS" \
  --repeats "$REPEATS" \
  "${EXTRA_ARGS[@]}"
