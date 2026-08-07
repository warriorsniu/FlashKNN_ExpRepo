#!/usr/bin/env bash
# Backward-compatible H20-specific entry point. New packaging should invoke
# build_native_wheelhouse.sh with TARGET_CUDA_ARCH explicitly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_CUDA_ARCH=9.0 exec "$SCRIPT_DIR/build_native_wheelhouse.sh" "$@"
