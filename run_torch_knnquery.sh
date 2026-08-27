#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/data/nyc/miniconda3/envs/flashknn-exp-cu118/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"

GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-rtx3090_torch_knnquery_k16_$(date +%Y%m%d_%H%M%S)}"
TORCH_KNNQUERY_ROOT="${TORCH_KNNQUERY_ROOT:-$REPO_DIR/.runtime/torch_knnquery}"
UPSTREAM_COMMIT="947957e45d8ba1b0da6f2b437ee104de2bff6b00"
OUT="$REPO_DIR/results/RTX3090/$RUN_ID"

if [[ ! -d "$TORCH_KNNQUERY_ROOT/.git" ]]; then
  git clone https://github.com/janericlenssen/torch_knnquery.git "$TORCH_KNNQUERY_ROOT"
  git -C "$TORCH_KNNQUERY_ROOT" checkout "$UPSTREAM_COMMIT"
fi
if [[ "$(git -C "$TORCH_KNNQUERY_ROOT" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
  echo "Unexpected torch_knnquery revision at $TORCH_KNNQUERY_ROOT" >&2
  exit 1
fi
if rg -q 'AT_DISPATCH_FLOATING_TYPES\(points\.type\(\)' "$TORCH_KNNQUERY_ROOT/src/knnquery.cu"; then
  git -C "$TORCH_KNNQUERY_ROOT" apply "$REPO_DIR/third_party/torch_knnquery_pytorch27.patch"
fi

mkdir -p "$OUT"
(cd "$TORCH_KNNQUERY_ROOT" && \
  CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" MAX_JOBS="${MAX_JOBS:-4}" \
  "$PYTHON_BIN" setup.py build_ext --inplace)
GPU="$GPU" "$PYTHON_BIN" "$REPO_DIR/scripts/collect_system_info.py" "$OUT/system.json"

EXTRA=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA=(--max-samples "$MAX_SAMPLES")
fi
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$TORCH_KNNQUERY_ROOT:$REPO_DIR/Query:$REPO_DIR/FlashKNN" \
  "$PYTHON_BIN" "$REPO_DIR/Query/benchmark_torch_knnquery.py" \
  --data-root "$EXPREPO_S3DIS" \
  --torch-knnquery-root "$TORCH_KNNQUERY_ROOT" \
  --output "$OUT/torch_knnquery_s3dis_pre250k_k16.json" \
  --gpu 0 --k 16 --warmups 3 --repeats 10 "${EXTRA[@]}"

echo "$OUT"
