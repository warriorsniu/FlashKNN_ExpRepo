#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_JOBS="${MAX_JOBS:-8}"
export MAX_JOBS

# Preserve only an architecture explicitly supplied by the caller.  A cached
# .runtime/cuda_arch.env belongs to the machine on which installation last ran
# and must not make a copied repository compile for the wrong GPU.
REQUESTED_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
source "$REPO_DIR/scripts/runtime_env.sh"
NVCC_RELEASE="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1)"
if [[ "$NVCC_RELEASE" != "12.8" ]]; then
  echo "PyTorch 2.7.1+cu128 experiments require CUDA toolkit 12.8; CUDA_HOME=$CUDA_HOME provides nvcc $NVCC_RELEASE." >&2
  echo "Install CUDA 12.8 (normally /usr/local/cuda-12.8) or set CUDA_HOME to it, then rerun." >&2
  exit 2
fi
python -m pip install 'numpy==1.26.4' scipy pandas openpyxl tabulate matplotlib tqdm pyyaml \
  huggingface_hub 'httpx[socks]' addict einops 'plyfile==1.1.3' termcolor timm tensorboardX yapf \
  torch-geometric 'spconv-cu120==2.3.6' 'ocnn==2.2.6'
python -m pip install --no-build-isolation 'SharedArray==3.2.1'
python -m pip install 'torch-scatter==2.1.2' \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# FLANN-CUDA uses the LZ4 C headers/library. Keep this inside the Conda
# environment so no sudo/system-package step is required on the target host.
# Keep OpenBLAS and its runtime from the same channel/build family. Mixing the
# defaults openblas-devel package with conda-forge OpenBLAS can leave a stale
# libgfortran.so.3 dependency that breaks source-built FAISS at import time.
conda install -y --override-channels -c conda-forge \
  lz4-c \
  'openblas=0.3.31=pthreads_h6ec200e_0' \
  'libopenblas=0.3.31=pthreads_h94d23a6_0'

# Detect the visible GPU instead of assuming a particular model. L20 reports
# 8.9 (sm_89), H20 reports 9.0 (sm_90), and RTX 3090 reports 8.6 (sm_86).
# An explicit caller override is retained for deliberate cross-compilation.
DETECTED_CUDA_ARCH="$(python -c 'import torch; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")')"
export TORCH_CUDA_ARCH_LIST="${REQUESTED_CUDA_ARCH_LIST:-$DETECTED_CUDA_ARCH}"
NORMALIZED_ARCH_LIST=" ${TORCH_CUDA_ARCH_LIST//;/ } "
if [[ "$NORMALIZED_ARCH_LIST" != *" $DETECTED_CUDA_ARCH "* && \
      "$NORMALIZED_ARCH_LIST" != *" $DETECTED_CUDA_ARCH+PTX "* ]]; then
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST does not include visible GPU capability $DETECTED_CUDA_ARCH" >&2
  exit 2
fi
mkdir -p "$REPO_DIR/.runtime"
printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "$TORCH_CUDA_ARCH_LIST" \
  > "$REPO_DIR/.runtime/cuda_arch.env"
echo "Detected CUDA capability: $DETECTED_CUDA_ARCH; native target: $TORCH_CUDA_ARCH_LIST"
USE_BUNDLED_WHEELHOUSE=0
if [[ -f "$REPO_DIR/wheelhouse/manifest.json" ]]; then
  BUNDLED_CUDA_ARCH="$(python -c \
    'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["target_cuda_arch"])' \
    "$REPO_DIR/wheelhouse/manifest.json")"
  if [[ "$BUNDLED_CUDA_ARCH" == "$DETECTED_CUDA_ARCH" ]]; then
    USE_BUNDLED_WHEELHOUSE=1
  else
    echo "Bundled wheelhouse targets sm_${BUNDLED_CUDA_ARCH//./}, but the visible GPU is sm_${DETECTED_CUDA_ARCH//./}; rebuilding native extensions from source."
  fi
fi
if [[ "$USE_BUNDLED_WHEELHOUSE" == "1" ]]; then
  echo "Using bundled, checksummed native wheelhouse."
  python "$REPO_DIR/scripts/install_native_wheelhouse.py" \
    --wheelhouse "$REPO_DIR/wheelhouse" \
    --expected-arch "$DETECTED_CUDA_ARCH" \
    --cuobjdump "$CUDA_HOME/bin/cuobjdump"
else
  # PyPI faiss-gpu-cu12 wheels currently contain only sm_70/sm_80 cubins and
  # no PTX. Build official FAISS explicitly for the visible architecture.
  FAISS_CUDA_ARCH="${DETECTED_CUDA_ARCH//./}" \
    bash "$REPO_DIR/scripts/install_faiss_gpu.sh"
  python -m pip install -v --no-build-isolation "$REPO_DIR/FlashKNN"
  python -m pip install -v --no-build-isolation "$REPO_DIR/Query/ThirdParty/cudaKDTree"
  python -m pip install -v --no-build-isolation "$REPO_DIR/Query/ThirdParty/flann"
  python -m pip install -v --no-build-isolation "$REPO_DIR/Query/ThirdParty/nanoflannkdtree"
  python -m pip install -v --no-build-isolation --no-deps "$REPO_DIR/Pointcept/libs/pointops"

  DWCONV_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dwconv-torch27.XXXXXX")"
  cleanup_dwconv() { rm -rf -- "$DWCONV_ROOT"; }
  trap cleanup_dwconv EXIT
  git clone https://github.com/octree-nn/dwconv.git "$DWCONV_ROOT/dwconv"
  git -C "$DWCONV_ROOT/dwconv" checkout ae53057eaf36dab01aa2727fcc93a749fd995af5
  python -m pip install -v --no-build-isolation "$DWCONV_ROOT/dwconv"
  cleanup_dwconv
  trap - EXIT
  bash "$REPO_DIR/scripts/install_minkowski.sh"
fi
# DeLA and DeepLA use identical native cutils sources. Install one shared
# extension into the environment so benchmark processes never invoke the JIT
# compiler or depend on a persistent source-tree build directory.
python -m pip install -v --force-reinstall --no-deps --no-build-isolation \
  "$REPO_DIR/DeLA/utils/cutils"
python "$REPO_DIR/scripts/verify_network_arch.py" \
  --expected-arch "$DETECTED_CUDA_ARCH" --cuobjdump "$CUDA_HOME/bin/cuobjdump"

# Import both source adapters; each must resolve the installed shared module.
PYTHONPATH="$REPO_DIR/DeLA:$REPO_DIR/Pointcept" python -c \
  'import utils.cutils; import FlashKNN; import Cukd.CuFun; import PyTorchCudaFlann; import PyTorchNanoFlann; import pointops, torch_scatter, SharedArray, spconv, ocnn, dwconv, MinkowskiEngine, pointcept.datasets, pointcept.models'
PYTHONPATH="$REPO_DIR/DeepLA-Net" python -c 'import utils.cutils'
python "$REPO_DIR/scripts/verify_extensions.py" --expected-arch "$DETECTED_CUDA_ARCH"

python -m pip check
echo "Unified query/DeLA/Pointcept installation complete."
