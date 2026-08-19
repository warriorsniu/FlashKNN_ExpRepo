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

# NCU_GPU is a physical nvidia-smi index.  The profiled process sees only that
# device, remapped to logical CUDA device 0; profile_knn_threads.py must
# therefore always receive --gpu 0.
GPU_IDENTITY="$(nvidia-smi -i "$NCU_GPU" \
  --query-gpu=index,uuid,name --format=csv,noheader | head -n 1)"
echo "NCU physical GPU: $GPU_IDENTITY"

METRICS="dram__sectors_read.sum,dram__sectors_write.sum"
METRICS+=",smsp__sass_average_branch_targets_threads_uniform.pct"
METRICS+=",smsp__thread_inst_executed_per_inst_executed.ratio"
METRICS+=",gpu__time_duration.sum,launch__registers_per_thread"
METRICS+=",launch__shared_mem_per_block,sm__warps_active.avg.pct_of_peak_sustained_active"
METRICS+=",launch__block_size,launch__grid_size,launch__waves_per_multiprocessor"

for backend in cukd flash-smps flash-gmss; do
  output="$NCU_OUTPUT_DIR/${backend}_k${K}"
  CUDA_VISIBLE_DEVICES="$NCU_GPU" "$CUDA_HOME/bin/ncu" \
    --metrics "$METRICS" \
    -k 'regex:(?i).*knn.*' \
    --print-summary per-kernel \
    --csv \
    --log-file "$output.csv" \
    --force-overwrite \
    -o "$output" \
    "$REPO_DIR/.venv/bin/python" "$REPO_DIR/scripts/profile_knn_threads.py" \
    --backend "$backend" --repo "$REPO_DIR" \
    --data-root "$REPO_DIR/data/s3dis" --gpu 0 --k "$K"
  "$CUDA_HOME/bin/ncu" --import "$output.ncu-rep" --page raw --csv \
    --log-file "${output}_raw.csv" --force-overwrite
done

echo "NCU reports: $NCU_OUTPUT_DIR"
