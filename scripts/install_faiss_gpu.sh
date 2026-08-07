#!/usr/bin/env bash
set -euo pipefail

# Build the official FAISS GPU Python bindings for the GPU that will execute
# the benchmark.  Third-party PyPI wheels may contain only sm_70/sm_80 cubins
# and no suitable PTX, so they cannot be assumed to execute on newer targets
# such as L20 (sm_89) or H20 (sm_90).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_JOBS="${MAX_JOBS:-8}"
FAISS_VERSION="${FAISS_VERSION:-1.12.0}"

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
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

if [[ -n "${FAISS_CUDA_ARCH:-}" ]]; then
  CUDA_ARCH="$FAISS_CUDA_ARCH"
else
  CUDA_ARCH="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
)"
fi
CUDA_ARCH="${CUDA_ARCH//./}"
if [[ ! "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
  echo "Invalid FAISS_CUDA_ARCH=$CUDA_ARCH (for example: 89 for L20, 90 for H20, or 86 for RTX 3090)." >&2
  exit 2
fi

python -m pip install --upgrade cmake swig
if [[ "${FAISS_BUILD_ONLY:-0}" != "1" ]]; then
  python -m pip uninstall -y faiss-gpu-cu12 faiss-cpu faiss || true
fi

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/faiss-sm${CUDA_ARCH}.XXXXXX")"
cleanup() { rm -rf -- "$BUILD_ROOT"; }
trap cleanup EXIT
git clone --depth 1 --branch "v${FAISS_VERSION}" \
  https://github.com/facebookresearch/faiss.git "$BUILD_ROOT/faiss"

PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
if [[ ! -f "$PYTHON_PREFIX/lib/libopenblas.so" ]]; then
  echo "OpenBLAS is missing from the active environment; run scripts/install.sh." >&2
  exit 2
fi
BLAS_ARGS=(-DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH="$PYTHON_PREFIX")
if [[ -f "$PYTHON_PREFIX/lib/libopenblas.so" ]]; then
  BLAS_ARGS+=(
    -DBLAS_LIBRARIES="$PYTHON_PREFIX/lib/libopenblas.so"
    -DBLAS_INCLUDE_DIRS="$PYTHON_PREFIX/include"
    -DLAPACK_LIBRARIES="$PYTHON_PREFIX/lib/libopenblas.so"
  )
fi

cmake -S "$BUILD_ROOT/faiss" -B "$BUILD_ROOT/faiss/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBUILD_TESTING=OFF \
  "${BLAS_ARGS[@]}" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build "$BUILD_ROOT/faiss/build" --target swigfaiss -j "$MAX_JOBS"
(
  cd "$BUILD_ROOT/faiss/build/faiss/python"
  python setup.py bdist_wheel
)
FAISS_WHEEL="$(find "$BUILD_ROOT/faiss/build/faiss/python/dist" -maxdepth 1 \
  -type f -name 'faiss-*.whl' -print -quit)"
if [[ -z "$FAISS_WHEEL" ]]; then
  echo "FAISS wheel was not produced." >&2
  exit 3
fi
# FAISS 1.12's setup.py incorrectly labels a wheel containing _swigfaiss.so as
# py3-none-any. Retag it for the exact CPython ABI and Linux platform so pip
# cannot mistake this native binary for a portable pure-Python wheel.
if [[ "$(basename "$FAISS_WHEEL")" == *-py3-none-any.whl ]]; then
  FAISS_DIST_DIR="$(dirname "$FAISS_WHEEL")"
  PYTHON_ABI_TAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  RETAGGED_NAME="$(
    cd "$FAISS_DIST_DIR"
    python -m wheel tags --remove \
      --python-tag "$PYTHON_ABI_TAG" --abi-tag "$PYTHON_ABI_TAG" \
      --platform-tag linux_x86_64 "$(basename "$FAISS_WHEEL")"
  )"
  FAISS_WHEEL="$FAISS_DIST_DIR/$RETAGGED_NAME"
fi
if [[ -n "${FAISS_WHEEL_DIR:-}" ]]; then
  mkdir -p "$FAISS_WHEEL_DIR"
  cp -f "$FAISS_WHEEL" "$FAISS_WHEEL_DIR/"
fi

if [[ "${FAISS_BUILD_ONLY:-0}" == "1" ]]; then
  python "$REPO_DIR/scripts/verify_wheel_arch.py" \
    --wheel "$FAISS_WHEEL" --expected-arch "$CUDA_ARCH" \
    --cuobjdump "$CUDA_HOME/bin/cuobjdump"
  echo "FAISS_BUILD_ONLY=1: wheel copied without modifying the active environment."
  exit 0
fi

python -m pip install --force-reinstall --no-deps "$FAISS_WHEEL"

FAISS_PACKAGE_DIR="$(python - <<'PY'
import faiss
from pathlib import Path
print(Path(faiss.__file__).resolve().parent)
PY
)"
# Do not put grep -q behind cuobjdump while `set -o pipefail` is active:
# grep exits as soon as it sees the first matching cubin, cuobjdump then gets
# SIGPIPE, and the otherwise successful pipeline is reported as a failure.
FAISS_CUBIN_LISTING="$(
  find "$FAISS_PACKAGE_DIR" -maxdepth 1 -type f -name '*.so' -print0 \
    | xargs -0 "$CUDA_HOME/bin/cuobjdump" --list-elf 2>/dev/null
)"
if ! grep -q "sm_${CUDA_ARCH}" <<<"$FAISS_CUBIN_LISTING"; then
  echo "FAISS was installed but its Python extension does not contain sm_${CUDA_ARCH}." >&2
  exit 3
fi

if [[ "${FAISS_SKIP_GPU_SMOKE:-0}" != "1" ]]; then
  python "$REPO_DIR/scripts/verify_faiss_gpu.py" --expected-arch "$CUDA_ARCH"
else
  echo "FAISS_SKIP_GPU_SMOKE=1: binary architecture checked; runtime smoke skipped."
fi
