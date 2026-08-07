#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/scripts/runtime_env.sh"
MAX_JOBS="${MAX_JOBS:-4}"
NVCC_RELEASE="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1)"
if [[ "$NVCC_RELEASE" != "12.8" ]]; then
  echo "Native wheels for torch 2.7.1+cu128 require CUDA toolkit 12.8; CUDA_HOME=$CUDA_HOME provides nvcc $NVCC_RELEASE." >&2
  exit 2
fi

# TARGET_CUDA_ARCH is a dotted compute capability such as 8.9 (L20), 9.0
# (H20), or 8.6 (RTX 3090).  With a visible GPU it is detected automatically;
# setting it explicitly also permits cross-compilation without that GPU.
if [[ -z "${TARGET_CUDA_ARCH:-}" ]]; then
  TARGET_CUDA_ARCH="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
fi
if [[ ! "$TARGET_CUDA_ARCH" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid TARGET_CUDA_ARCH=$TARGET_CUDA_ARCH (examples: 8.9, 9.0, 8.6)." >&2
  exit 2
fi
COMPACT_CUDA_ARCH="${TARGET_CUDA_ARCH//./}"
export MAX_JOBS TORCH_CUDA_ARCH_LIST="$TARGET_CUDA_ARCH"
WHEELHOUSE="${1:-$REPO_DIR/wheelhouse}"

python - <<'PY'
import sys
import torch
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Native wheelhouse requires Python 3.10, got {sys.version.split()[0]}"
    )
if torch.__version__ != "2.7.1+cu128" or torch.version.cuda != "12.8":
    raise SystemExit(
        f"Native wheelhouse requires torch 2.7.1+cu128/CUDA 12.8, "
        f"got {torch.__version__}/CUDA {torch.version.cuda}"
    )
print(f"Wheel build ABI: Python {sys.version.split()[0]}, torch {torch.__version__}")
PY

if [[ -e "$WHEELHOUSE" ]] && find "$WHEELHOUSE" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to mix wheels in non-empty directory: $WHEELHOUSE" >&2
  exit 2
fi
mkdir -p "$WHEELHOUSE"
WHEELHOUSE="$(cd "$WHEELHOUSE" && pwd)"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sm${COMPACT_CUDA_ARCH}-wheelhouse.XXXXXX")"
cleanup() { rm -rf -- "$BUILD_ROOT"; }
trap cleanup EXIT

copy_and_build() {
  local source="$1" name="$2"
  mkdir -p "$BUILD_ROOT/$name"
  rsync -a --exclude build --exclude '*.egg-info' --exclude '*.so' \
    "$source/" "$BUILD_ROOT/$name/"
  (cd "$BUILD_ROOT/$name" && python -m pip wheel -v --no-build-isolation \
    --no-deps --wheel-dir "$WHEELHOUSE" .)
}

copy_and_build "$REPO_DIR/FlashKNN" flashknn
copy_and_build "$REPO_DIR/Query/ThirdParty/cudaKDTree" cukd
copy_and_build "$REPO_DIR/Query/ThirdParty/flann" flann
copy_and_build "$REPO_DIR/Query/ThirdParty/nanoflannkdtree" nanoflann
copy_and_build "$REPO_DIR/Pointcept/libs/pointops" pointops

git clone https://github.com/octree-nn/dwconv.git "$BUILD_ROOT/dwconv"
git -C "$BUILD_ROOT/dwconv" checkout ae53057eaf36dab01aa2727fcc93a749fd995af5
(cd "$BUILD_ROOT/dwconv" && python -m pip wheel -v --no-build-isolation \
  --no-deps --wheel-dir "$WHEELHOUSE" .)

git clone https://github.com/NVIDIA/MinkowskiEngine.git "$BUILD_ROOT/MinkowskiEngine"
git -C "$BUILD_ROOT/MinkowskiEngine" checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
git -C "$BUILD_ROOT/MinkowskiEngine" apply \
  "$REPO_DIR/scripts/patches/minkowski_cuda12_torch27.patch"
(
  cd "$BUILD_ROOT/MinkowskiEngine"
  python setup.py bdist_wheel --dist-dir "$WHEELHOUSE" \
    --force_cuda --blas=openblas \
    --blas_include_dirs="$CONDA_PREFIX/include" \
    --blas_library_dirs="$CONDA_PREFIX/lib"
)

FAISS_BUILD_ONLY=1 FAISS_SKIP_GPU_SMOKE=1 \
  FAISS_CUDA_ARCH="$COMPACT_CUDA_ARCH" FAISS_WHEEL_DIR="$WHEELHOUSE" \
  MAX_JOBS="$MAX_JOBS" bash "$REPO_DIR/scripts/install_faiss_gpu.sh"

for wheel in "$WHEELHOUSE"/*.whl; do
  case "$(basename "$wheel" | tr '[:upper:]' '[:lower:]')" in
    pytorchnanoflann-*) continue ;;
  esac
  python "$REPO_DIR/scripts/verify_wheel_arch.py" \
    --wheel "$wheel" --expected-arch "$COMPACT_CUDA_ARCH" \
    --cuobjdump "$CUDA_HOME/bin/cuobjdump"
done
python "$REPO_DIR/scripts/make_wheelhouse_manifest.py" \
  --wheelhouse "$WHEELHOUSE" --cuda-arch "$TARGET_CUDA_ARCH" \
  --cuda-home "$CUDA_HOME"
echo "sm_${COMPACT_CUDA_ARCH} wheelhouse ready: $WHEELHOUSE"
