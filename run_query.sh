#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"
GPU="${GPU:-0}"
S3DIS_QUERY_ROOT="${EXPREPO_S3DIS_QUERY:-$EXPREPO_S3DIS}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$REPO_DIR/results/$RUN_ID/query"
mkdir -p "$OUT"
S3DIS_EXTRA=()
LIDAR_EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "WARNING: SMOKE=1 crops S3DIS sample_part to 1,000 points and uses one timed repeat."
  echo "         These timings validate functionality only and must not be used for speedup claims."
  S3DIS_EXTRA=(--warmups 1 --repeats 1 --max-samples 1 --crop-points 1000)
  LIDAR_EXTRA=(--warmups 1 --repeats 1 --max-samples 1)
fi

python "$REPO_DIR/scripts/collect_system_info.py" "$REPO_DIR/results/$RUN_ID/system.json"
PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN" python "$REPO_DIR/Query/benchmark_s3dis.py" \
  --data-root "$S3DIS_QUERY_ROOT" --output "$OUT/s3dis_sample_part.json" --gpu "$GPU" \
  --scope sample_part --mode pre post --k 8 16 24 32 48 64 "${S3DIS_EXTRA[@]}"
PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN" python "$REPO_DIR/Query/benchmark_s3dis.py" \
  --data-root "$S3DIS_QUERY_ROOT" --output "$OUT/s3dis_full_k32.json" --gpu "$GPU" \
  --scope full --mode pre post --k 32 "${S3DIS_EXTRA[@]}"

if [[ -n "${EXPREPO_SEMANTICKITTI:-}" ]]; then
  PYTHONPATH="$REPO_DIR/FlashKNN" python "$REPO_DIR/Query/benchmark_semantickitti.py" \
    --data-dir "$EXPREPO_SEMANTICKITTI" --output "$OUT/semantickitti.json" --gpu "$GPU" \
    --mode pre post --k 16 24 32 --alpha 4 8 16 32 "${LIDAR_EXTRA[@]}"
fi
echo "$OUT"
