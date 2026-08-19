#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${SMOKE:-0}" != "1" && "${ALLOW_FORMAL_THREAD_GROUPING:-0}" != "1" ]]; then
  echo "Formal thread-grouping ablation is locked." >&2
  echo "Set ALLOW_FORMAL_THREAD_GROUPING=1 only after explicit user authorization." >&2
  exit 2
fi

if [[ "${SMOKE:-0}" != "1" ]]; then
  TARGET_GPU_UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid \
    --format=csv,noheader | xargs)"
  if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
      --format=csv,noheader | grep -F "$TARGET_GPU_UUID,"; then
    echo "Formal thread-grouping ablation requires an idle target GPU." >&2
    exit 3
  fi
  unset TARGET_GPU_UUID
fi

source "$REPO_DIR/scripts/results_env.sh"
OUT="$RESULTS_ROOT/$RUN_ID/thread_grouping"
ANALYSIS_OUT="$REPO_DIR/analysis/output/$RUN_ID/thread_grouping"
mkdir -p "$OUT"
WARMUPS="${THREAD_GROUP_WARMUPS:-5}"
REPEATS="${THREAD_GROUP_REPEATS:-20}"
EXTRA_ARGS=()
VALIDATE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  WARMUPS="${THREAD_GROUP_WARMUPS:-1}"
  REPEATS="${THREAD_GROUP_REPEATS:-1}"
  EXTRA_ARGS+=(--max-samples 1)
  VALIDATE_ARGS+=(--allow-partial)
fi

RESULT="$OUT/s3dis_thread_grouping.json"
PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$REPO_DIR/Query/benchmark_thread_grouping.py" \
  --data-root "$EXPREPO_S3DIS" \
  --output "$RESULT" \
  --gpu "$GPU" \
  --k 8 16 24 32 48 64 \
  --warmups "$WARMUPS" \
  --repeats "$REPEATS" \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" "$REPO_DIR/scripts/validate_thread_grouping.py" \
  "$RESULT" "${VALIDATE_ARGS[@]}"
"$PYTHON_BIN" "$REPO_DIR/analysis/analyze_thread_grouping.py" \
  "$RESULT" --output-dir "$ANALYSIS_OUT"
echo "Result: $RESULT"
echo "Analysis: $ANALYSIS_OUT"
