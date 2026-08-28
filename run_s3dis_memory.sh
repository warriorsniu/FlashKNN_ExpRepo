#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"

GPU="${GPU:-0}"
S3DIS_QUERY_ROOT="${EXPREPO_S3DIS_QUERY:-$EXPREPO_S3DIS}"
source "$REPO_DIR/scripts/results_env.sh"
PLATFORM_NAME="$(basename "$RESULTS_ROOT")"
RUN_ID="${RUN_ID:-$(tr '[:upper:]' '[:lower:]' <<<"$PLATFORM_NAME")_s3dis_memory_k32_$(date +%Y%m%d)}"
OUT="$RESULTS_ROOT/$RUN_ID"
METHODS_TEXT="${METHODS:-flashknn cuda_kdtree faiss_flat faiss_ivf}"
read -r -a METHOD_ARGS <<<"$METHODS_TEXT"

CANONICAL_S3DIS="${CANONICAL_S3DIS:-$RESULTS_ROOT/$(
  case "$PLATFORM_NAME" in
    RTX3090) echo rtx3090_final_20260825 ;;
    L20) echo l20_final_20260824 ;;
    *) echo complete ;;
  esac
)/query/s3dis_sample_part.json}"

if [[ ! -f "$CANONICAL_S3DIS" ]]; then
  echo "Canonical S3DIS result not found: $CANONICAL_S3DIS" >&2
  exit 2
fi

TARGET_UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | xargs)"
CO_TENANTS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null | awk -F, -v uuid="$TARGET_UUID" '$1 == uuid {print}')"
if [[ -n "$CO_TENANTS" && "${ALLOW_CO_TENANT:-0}" != "1" ]]; then
  echo "Selected GPU has compute co-tenants; refusing formal memory measurement:" >&2
  echo "$CO_TENANTS" >&2
  exit 2
fi

EXTRA=()
EXPECTED_ROOMS=81
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--max-samples 1)
  EXPECTED_ROOMS=1
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi

mkdir -p "$OUT"
"$PYTHON_BIN" "$REPO_DIR/scripts/collect_system_info.py" "$OUT/system.json"
PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN" \
  "$PYTHON_BIN" "$REPO_DIR/Query/benchmark_s3dis_memory.py" \
  --data-root "$S3DIS_QUERY_ROOT" \
  --canonical-s3dis "$CANONICAL_S3DIS" \
  --output "$OUT/s3dis_memory_k32.json" \
  --gpu "$GPU" --mode pre post --k 32 --methods "${METHOD_ARGS[@]}" \
  "${EXTRA[@]}"
"$PYTHON_BIN" "$REPO_DIR/analysis/analyze_s3dis_memory.py" \
  --input "$OUT/s3dis_memory_k32.json" --output-dir "$OUT/analysis" \
  --expected-rooms "$EXPECTED_ROOMS"

echo "$OUT"
