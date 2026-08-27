#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"

GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-rtx3090_pytorch3d_ball_query_$(date +%Y%m%d_%H%M%S)}"
POINTCEPT_RESULT="${POINTCEPT_RESULT:-$REPO_DIR/results/RTX3090/rtx3090_final_corrected_20260824/query/ball_query_s3dis_sample_part.json}"
PYTORCH3D_COMMIT="${PYTORCH3D_COMMIT:-fdaf9bd6fed7977e4c2056e7c77c640781e58fcd}"
OUT="$REPO_DIR/results/RTX3090/$RUN_ID/query/pytorch3d_ball_query_s3dis_sample_part.json"

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA=(--max-samples 1 --mode pre --k 32 --warmups 1 --repeats 1)
fi

PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN" "$PYTHON_BIN" \
  "$REPO_DIR/Query/benchmark_pytorch3d_ball_query.py" \
  --data-root "${EXPREPO_S3DIS_QUERY:-$EXPREPO_S3DIS}" \
  --radii-source "$POINTCEPT_RESULT" \
  --output "$OUT" \
  --pytorch3d-commit "$PYTORCH3D_COMMIT" \
  --gpu "$GPU" --mode pre post --k 24 32 48 --percentile 0.9 \
  --crop-points 250000 --warmups 3 --repeats 10 --resume "${EXTRA[@]}"

echo "$OUT"
