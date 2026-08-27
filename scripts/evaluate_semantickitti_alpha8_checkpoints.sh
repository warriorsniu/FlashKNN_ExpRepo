#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate the three paper seeds with one GPU and a caller-specified
# FlashKNN source tree. This is an accuracy/compatibility check, not a latency
# benchmark, so the CUDA toolkit used to build the extension is recorded but
# is not compared across machines.

PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN}"
FLASHKNN_ROOT="${FLASHKNN_ROOT:?set FLASHKNN_ROOT to the source directory containing functions/}"
NETWORK_ROOT="${NETWORK_ROOT:?set NETWORK_ROOT to experiments/lidar/network}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT to the checkpoint root}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the SemanticKITTI dataset directory}"
DELA_REPO="${DELA_REPO:?set DELA_REPO}"
DEEPLA_REPO="${DEEPLA_REPO:?set DEEPLA_REPO}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-4}"

mkdir -p "$OUTPUT_ROOT"

run_evaluation() {
  local model="$1"
  local repo="$2"
  local checkpoint="$3"
  local output="$4"
  if [[ -s "$output" ]]; then
    echo "Skip completed $output"
    return
  fi
  PYTHONPATH="$FLASHKNN_ROOT:$NETWORK_ROOT" "$PYTHON_BIN" \
    "$NETWORK_ROOT/evaluate.py" \
    --model "$model" --repo "$repo" --variant 24 \
    --checkpoint "$checkpoint" --data-root "$DATA_ROOT" \
    --gpu "$GPU" --alpha 8 --workers "$WORKERS" --output "$output"
}

run_evaluation dela "$DELA_REPO" \
  "$EXPERIMENT_ROOT/formal_bs6/dela/last.pt" "$OUTPUT_ROOT/dela_seed47.json"
run_evaluation dela "$DELA_REPO" \
  "$EXPERIMENT_ROOT/dela_repeats/flashknn/seed48/last.pt" "$OUTPUT_ROOT/dela_seed48.json"
run_evaluation dela "$DELA_REPO" \
  "$EXPERIMENT_ROOT/dela_repeats/flashknn/seed49/last.pt" "$OUTPUT_ROOT/dela_seed49.json"
run_evaluation deepla "$DEEPLA_REPO" \
  "$EXPERIMENT_ROOT/formal_bs6/deepla24/last.pt" "$OUTPUT_ROOT/deepla_seed47.json"
run_evaluation deepla "$DEEPLA_REPO" \
  "$EXPERIMENT_ROOT/deepla24_repeats/flashknn/seed48/last.pt" "$OUTPUT_ROOT/deepla_seed48.json"
run_evaluation deepla "$DEEPLA_REPO" \
  "$EXPERIMENT_ROOT/deepla24_repeats/flashknn/seed49/last.pt" "$OUTPUT_ROOT/deepla_seed49.json"

sha256sum \
  "$FLASHKNN_ROOT/csrc/flash_knn_query_dynamic_load.cu" \
  "$FLASHKNN_ROOT/csrc/flash_knn_bitonic_top_p.cuh" \
  "$FLASHKNN_ROOT/functions/"CuFun*.so \
  "$EXPERIMENT_ROOT/formal_bs6/dela/last.pt" \
  "$EXPERIMENT_ROOT/dela_repeats/flashknn/seed48/last.pt" \
  "$EXPERIMENT_ROOT/dela_repeats/flashknn/seed49/last.pt" \
  "$EXPERIMENT_ROOT/formal_bs6/deepla24/last.pt" \
  "$EXPERIMENT_ROOT/deepla24_repeats/flashknn/seed48/last.pt" \
  "$EXPERIMENT_ROOT/deepla24_repeats/flashknn/seed49/last.pt" \
  > "$OUTPUT_ROOT/provenance.sha256"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" - <<'PY' > "$OUTPUT_ROOT/environment.txt"
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
