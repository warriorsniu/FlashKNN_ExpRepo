#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/data/nyc/miniconda3/envs/flashknn-exp-cu118/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
source "$REPO_DIR/scripts/python_env.sh"
source "$REPO_DIR/scripts/runtime_env.sh"
source "$REPO_DIR/data/paths.env"

GPU="${GPU:-5}"
RUN_ID="${RUN_ID:-rtx3090_gmss_full_k_$(date +%Y%m%d_%H%M%S)}"
if [[ "${ALLOW_FORMAL_GMSS:-0}" != "1" ]]; then
  echo "Formal GMSS ablation is locked." >&2
  echo "Set ALLOW_FORMAL_GMSS=1 only after explicit user authorization." >&2
  exit 2
fi

TARGET_GPU_UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | xargs)"
if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
    --format=csv,noheader | grep -F "$TARGET_GPU_UUID,"; then
  echo "Formal GMSS ablation requires an idle target GPU." >&2
  exit 3
fi
unset TARGET_GPU_UUID

source "$REPO_DIR/scripts/results_env.sh"
OUT="$RESULTS_ROOT/$RUN_ID/ablation"
RESULT="$OUT/s3dis_gmss_ablation.json"
ANALYSIS_OUT="$REPO_DIR/analysis/output/rtx3090_ablation_final_with_gmss_20260820"
BASE_RESULT="$REPO_DIR/results/RTX3090/rtx3090_ablation_final_20260810/ablation/s3dis_design_ablation.json"
mkdir -p "$OUT"

GPU="$GPU" "$PYTHON_BIN" "$REPO_DIR/scripts/collect_system_info.py" \
  "$RESULTS_ROOT/$RUN_ID/system.json"
PYTHONPATH="$REPO_DIR/Query:$REPO_DIR/FlashKNN${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$REPO_DIR/Query/benchmark_ablation.py" \
  --data-root "$EXPREPO_S3DIS" \
  --output "$RESULT" \
  --gpu "$GPU" \
  --k 8 16 24 32 40 48 56 64 \
  --variants gmss \
  --warmups 5 \
  --repeats 20

"$PYTHON_BIN" "$REPO_DIR/scripts/validate_gmss_ablation.py" "$RESULT"
"$PYTHON_BIN" "$REPO_DIR/analysis/analyze_ablation.py" "$BASE_RESULT" \
  --extra-result "$RESULT" \
  --output-dir "$ANALYSIS_OUT" \
  --font-dir /data/nyc/fonts \
  --memory-sorting-asset \
    "$REPO_DIR/../PaperRevise/revision_results_assets/core_memory_sorting" \
  --candidate-skip-asset \
    "$REPO_DIR/../PaperRevise/revision_results_assets/core_candidate_skip"

echo "Result: $RESULT"
echo "Analysis: $ANALYSIS_OUT"
