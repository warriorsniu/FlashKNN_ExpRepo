#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"

PREFLIGHT_ARGS=(--s3dis "$EXPREPO_S3DIS" --semantickitti "$EXPREPO_SEMANTICKITTI")
if [[ -n "${GPU:-}" ]]; then PREFLIGHT_ARGS+=(--gpu "$GPU"); fi
if [[ "${SMOKE:-0}" == "1" ]]; then PREFLIGHT_ARGS+=(--quick); fi
"$PYTHON_BIN" "$REPO_DIR/scripts/preflight.py" "${PREFLIGHT_ARGS[@]}" "$@"

if [[ " $* " != *" --data-only "* ]]; then
  PYTHONPATH="$REPO_DIR/Pointcept" "$PYTHON_BIN" -c \
    'import torch, pointops, torch_scatter, SharedArray, spconv, ocnn, dwconv, MinkowskiEngine, pointcept.datasets, pointcept.models; assert torch.__version__.startswith("2.7.1+"); assert torch.version.cuda == "12.8"; assert torch.cuda.is_available()'
  if [[ "${SMOKE:-0}" != "1" ]]; then
    "$PYTHON_BIN" "$REPO_DIR/scripts/verify_network_models.py" --s3dis "$EXPREPO_S3DIS"
  fi
fi
