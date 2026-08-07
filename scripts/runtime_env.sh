#!/usr/bin/env bash
# Shared CUDA runtime selection for all entry points. This prevents a stale
# CUDA 11.x LD_LIBRARY_PATH inherited from another project from taking
# precedence over the toolkit used to compile the unified CUDA 12.x stack.

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

# Do not trust PATH alone: another project may prepend CUDA 11.8. Keep an
# explicit CUDA_HOME only when it points to CUDA 12+, otherwise prefer an
# exact CUDA 12.8 toolkit and then the machine's canonical /usr/local/cuda
# symlink. This makes a host with an old project-level CUDA_HOME work without
# manual path editing.
CURRENT_RELEASE="$(cuda_release "${CUDA_HOME:-/nonexistent}" || true)"
CURRENT_MAJOR="$(cuda_major "${CUDA_HOME:-/nonexistent}" || true)"
if [[ "$CURRENT_RELEASE" != "12.8" ]]; then
  PREFERRED_RELEASE="$(cuda_release /usr/local/cuda-12.8 || true)"
  DEFAULT_MAJOR="$(cuda_major /usr/local/cuda || true)"
  if [[ "$PREFERRED_RELEASE" == "12.8" ]]; then
    CUDA_HOME="$(readlink -f /usr/local/cuda-12.8)"
  elif [[ -z "$CURRENT_MAJOR" || "$CURRENT_MAJOR" -lt 12 ]] && \
       [[ -n "$DEFAULT_MAJOR" && "$DEFAULT_MAJOR" -ge 12 ]]; then
    CUDA_HOME="$(readlink -f /usr/local/cuda)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# PyTorch's cu128 wheel ships a mutually compatible CUDA 12.8 user-space
# runtime. System CUDA lib64 entries in LD_LIBRARY_PATH override those bundled
# libraries (even when CUDA_HOME is only a minor version apart) and can mix,
# for example, cuSPARSE 12.8 with nvJitLink 12.6. Keep CUDA_HOME for nvcc and
# explicit build-time -L paths, but remove inherited CUDA toolkit libraries at
# runtime. Non-CUDA entries are preserved.
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
if [[ -n "${VIRTUAL_ENV:-}" && -d "$VIRTUAL_ENV/lib" ]]; then
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset CLEAN_LD_LIBRARY_PATH LD_ENTRIES entry CURRENT_RELEASE CURRENT_MAJOR \
  PREFERRED_RELEASE DEFAULT_MAJOR

RUNTIME_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.runtime"
if [[ -f "$RUNTIME_ENV_DIR/cuda_arch.env" ]]; then
  source "$RUNTIME_ENV_DIR/cuda_arch.env"
fi
