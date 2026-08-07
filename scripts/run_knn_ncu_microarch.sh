#!/usr/bin/env bash
# Profile all paper-reported kNN microarchitecture metrics with Nsight Compute.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCU_GPU="${NCU_GPU:-1}"
NCU_OUTPUT_DIR="${NCU_OUTPUT_DIR:-/tmp/flashknn_ncu_microarch_l20}"
K="${K:-32}"
source "$REPO_DIR/.venv/bin/activate"
source "$REPO_DIR/scripts/runtime_env.sh"
mkdir -p "$NCU_OUTPUT_DIR"

for backend in cukd flash-smps flash-gmss; do
  output="$NCU_OUTPUT_DIR/${backend}_k${K}"
  CUDA_VISIBLE_DEVICES="$NCU_GPU" "$CUDA_HOME/bin/ncu" \
    --metrics dram__sectors_read.sum,dram__sectors_write.sum,smsp__sass_average_branch_targets_threads_uniform.pct,smsp__thread_inst_executed_per_inst_executed.ratio \
    -k 'regex:(?i).*knn.*' \
    --print-summary per-kernel \
    --csv \
    --log-file "$output.csv" \
    --force-overwrite \
    -o "$output" \
    "$REPO_DIR/.venv/bin/python" "$REPO_DIR/scripts/profile_knn_threads.py" \
    --backend "$backend" --repo "$REPO_DIR" \
    --data-root "$REPO_DIR/data/s3dis" --gpu 0 --k "$K"
done

echo "NCU reports: $NCU_OUTPUT_DIR"
