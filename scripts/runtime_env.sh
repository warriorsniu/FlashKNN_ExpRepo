#!/usr/bin/env bash
# Shared CUDA runtime selection for all entry points. This prevents a stale
# CUDA LD_LIBRARY_PATH inherited from another project from taking precedence
# over the CUDA 11.8 or 12.8 toolkit selected for the current platform.

cuda_major() {
  [[ -x "$1/bin/nvcc" ]] || return 1
  "$1/bin/nvcc" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' \
    | tail -1
}

cuda_release() {
  [[ -x "$1/bin/nvcc" ]] || return 1
  "$1/bin/nvcc" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
    | tail -1
}

# Do not trust PATH alone: another project may prepend an unrelated toolkit.
# An explicit CUDA_HOME wins when it points to one of the two supported
# experiment toolkits.  Without an override, prefer CUDA 12.8 (L20/H20), then
# CUDA 11.8 (the historical RTX 3090 platform), and only then the machine's
# canonical /usr/local/cuda symlink.
CURRENT_RELEASE="$(cuda_release "${CUDA_HOME:-/nonexistent}" || true)"
CURRENT_MAJOR="$(cuda_major "${CUDA_HOME:-/nonexistent}" || true)"
if [[ "$CURRENT_RELEASE" != "12.8" && "$CURRENT_RELEASE" != "11.8" ]]; then
  PREFERRED_RELEASE="$(cuda_release /usr/local/cuda-12.8 || true)"
  HISTORICAL_RELEASE="$(cuda_release /usr/local/cuda-11.8 || true)"
  DEFAULT_RELEASE="$(cuda_release /usr/local/cuda || true)"
  if [[ "$PREFERRED_RELEASE" == "12.8" ]]; then
    CUDA_HOME="$(readlink -f /usr/local/cuda-12.8)"
  elif [[ "$HISTORICAL_RELEASE" == "11.8" ]]; then
    CUDA_HOME="$(readlink -f /usr/local/cuda-11.8)"
  elif [[ -n "$DEFAULT_RELEASE" ]]; then
    CUDA_HOME="$(readlink -f /usr/local/cuda)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# PyTorch's cu118/cu128 wheels ship their matching user-space runtime. System
# CUDA lib64 entries in LD_LIBRARY_PATH can override those bundled libraries.
# Keep CUDA_HOME for nvcc and explicit build-time -L paths, but remove inherited
# CUDA toolkit libraries at runtime. Non-CUDA entries are preserved.
CLEAN_LD_LIBRARY_PATH=""
IFS=: read -r -a LD_ENTRIES <<< "${LD_LIBRARY_PATH:-}"
for entry in "${LD_ENTRIES[@]}"; do
  [[ -z "$entry" ]] && continue
  case "$entry" in
    */cuda*/lib|*/cuda*/lib64|*/cuda*/targets/*/lib) continue ;;
  esac
  CLEAN_LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH:+$CLEAN_LD_LIBRARY_PATH:}$entry"
done
if [[ -n "$CLEAN_LD_LIBRARY_PATH" ]]; then
  export LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi

# A uv/venv installation may keep native dependencies such as OpenBLAS and
# LZ4 inside the Python prefix. Preserve those libraries at runtime without
# requiring a Conda-specific activation hook.
PYTHON_PREFIX="${EXPREPO_PYTHON_PREFIX:-${VIRTUAL_ENV:-${CONDA_PREFIX:-}}}"
if [[ -n "$PYTHON_PREFIX" && -d "$PYTHON_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset CLEAN_LD_LIBRARY_PATH LD_ENTRIES entry CURRENT_RELEASE CURRENT_MAJOR \
  PREFERRED_RELEASE HISTORICAL_RELEASE DEFAULT_RELEASE PYTHON_PREFIX

RUNTIME_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.runtime"
if [[ -f "$RUNTIME_ENV_DIR/cuda_arch.env" ]]; then
  source "$RUNTIME_ENV_DIR/cuda_arch.env"
fi
