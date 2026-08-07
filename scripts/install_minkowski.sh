#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINKOWSKI_COMMIT="${MINKOWSKI_COMMIT:-02fc608bea4c0549b0a7b00ca1bf15dee4a0b228}"
MAX_JOBS="${MAX_JOBS:-8}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="$(python - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME or "")
PY
)"
fi
if [[ -z "$CUDA_HOME" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "A CUDA toolkit with nvcc is required; set CUDA_HOME explicitly." >&2
  exit 2
fi
export CUDA_HOME MAX_JOBS

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
fi
export TORCH_CUDA_ARCH_LIST

ENV_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
if [[ ! -f "$ENV_PREFIX/include/cblas.h" || ! -f "$ENV_PREFIX/lib/libopenblas.so" ]]; then
  echo "OpenBLAS headers/library are missing from $ENV_PREFIX; install them in the active Python environment." >&2
  exit 2
fi
# The pinned MinkowskiEngine setup.py can silently omit --blas_library_dirs
# from the final linker invocation. LIBRARY_PATH is honored by GCC even in
# that case and keeps a user-local uv/venv OpenBLAS installation discoverable.
export LIBRARY_PATH="$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

python -m pip install --upgrade ninja wheel
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/minkowski-torch27.XXXXXX")"
cleanup() { rm -rf -- "$BUILD_ROOT"; }
trap cleanup EXIT

mkdir "$BUILD_ROOT/MinkowskiEngine"
gh api "repos/NVIDIA/MinkowskiEngine/tarball/$MINKOWSKI_COMMIT" \
  > "$BUILD_ROOT/minkowski.tar.gz"
tar -xzf "$BUILD_ROOT/minkowski.tar.gz" --strip-components=1 \
  -C "$BUILD_ROOT/MinkowskiEngine"
git -C "$BUILD_ROOT/MinkowskiEngine" apply \
  "$REPO_DIR/scripts/patches/minkowski_cuda12_torch27.patch"

(
  cd "$BUILD_ROOT/MinkowskiEngine"
  python setup.py bdist_wheel \
    --force_cuda \
    --blas=openblas \
    --blas_include_dirs="$ENV_PREFIX/include" \
    --blas_library_dirs="$ENV_PREFIX/lib"
  python -m pip install --force-reinstall --no-deps dist/minkowskiengine-*.whl
)

python - <<'PY'
import MinkowskiEngine as ME
import torch
print("MinkowskiEngine", ME.__version__)
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
PY
