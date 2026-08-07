#!/usr/bin/env bash
# Build and run the matched Pointcept ball-query and Arkade RT-core baselines.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"

GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-related_$(date +%Y%m%d_%H%M%S)}"
OUT="$REPO_DIR/results/$RUN_ID/query"
POINTOPS_BUILD="$REPO_DIR/Pointcept/libs/pointops/build/lib.linux-x86_64-cpython-310"
OPTIX_ROOT="${OptiX_INSTALL_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
mkdir -p "$OUT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN; run scripts/install.sh first." >&2
  exit 1
fi

if [[ ! -d "$POINTOPS_BUILD/pointops" ]]; then
  echo "Pointcept pointops extension is missing; run scripts/install.sh first." >&2
  exit 1
fi

COMMON_PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN:$POINTOPS_BUILD:$REPO_DIR/Pointcept/libs/pointops"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$COMMON_PYTHONPATH" "$PYTHON_BIN" \
  "$REPO_DIR/Query/benchmark_ball_query.py" \
  --data-root "$EXPREPO_S3DIS" --output "$OUT/s3dis_ball_query.json" --gpu "$GPU" \
  --mode pre post --k 24 32 48 --percentile 0.9 --crop-points 250000 \
  --warmups 3 --repeats 10

if [[ -z "$OPTIX_ROOT" || ! -f "$OPTIX_ROOT/include/optix.h" ]]; then
  echo "Set OptiX_INSTALL_DIR to an extracted OptiX 8 SDK before running Arkade." >&2
  exit 1
fi

export OptiX_INSTALL_DIR="$OPTIX_ROOT"
for K in 24 32 48; do
  cmake -S "$REPO_DIR/third_party/Arkade" -B "$REPO_DIR/third_party/Arkade/build-k${K}-l2" \
    -G Ninja -DKN="$K" -DNORM=2 -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
  cmake --build "$REPO_DIR/third_party/Arkade/build-k${K}-l2" \
    --target arkade-benchmark -j "$(nproc)"
done

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN" "$PYTHON_BIN" \
  "$REPO_DIR/Query/benchmark_arkade.py" \
  --repo "$REPO_DIR" --data-root "$EXPREPO_S3DIS" \
  --output "$OUT/s3dis_arkade.json" --gpu "$GPU" --mode pre post --k 24 32 48 \
  --crop-points 250000 --initial-radius 0.02 --warmups 3 --repeats 10 \
  --retries 2 --resume

echo "$OUT"
