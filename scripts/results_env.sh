#!/usr/bin/env bash
# Choose a hardware-specific result root so a local RTX 3090 run cannot be
# written into results/L20 by accident. Call after GPU has been selected.

if [[ -z "${RESULTS_ROOT:-}" ]]; then
  EXPREPO_GPU_NAME="$(nvidia-smi -i "${GPU:-0}" --query-gpu=name \
    --format=csv,noheader 2>/dev/null | head -n 1 | xargs)"
  case "$EXPREPO_GPU_NAME" in
    *"RTX 3090"*) EXPREPO_PLATFORM_DIR="RTX3090" ;;
    *"L20"*) EXPREPO_PLATFORM_DIR="L20" ;;
    *"H20"*) EXPREPO_PLATFORM_DIR="H20" ;;
    *)
      EXPREPO_PLATFORM_DIR="$(tr -cs '[:alnum:]' '_' <<<"$EXPREPO_GPU_NAME" \
        | sed 's/^_//; s/_$//')"
      [[ -n "$EXPREPO_PLATFORM_DIR" ]] || EXPREPO_PLATFORM_DIR="UnknownGPU"
      ;;
  esac
  RESULTS_ROOT="$REPO_DIR/results/$EXPREPO_PLATFORM_DIR"
fi
export RESULTS_ROOT
unset EXPREPO_GPU_NAME EXPREPO_PLATFORM_DIR
