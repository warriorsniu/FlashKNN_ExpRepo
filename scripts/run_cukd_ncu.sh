#!/usr/bin/env bash
# Profile paper-scale cudaKDTree query I/O using the exact metrics from ncu.sh.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCU_GPU="${NCU_GPU:-1}"
NCU_OUTPUT="${NCU_OUTPUT:-/tmp/flashknn_cukd_k32_l20}"
source "$REPO_DIR/.venv/bin/activate"
source "$REPO_DIR/scripts/runtime_env.sh"

CUDA_VISIBLE_DEVICES="$NCU_GPU" "$CUDA_HOME/bin/ncu" \
  --metrics dram__sectors_read.sum,dram__sectors_write.sum \
  -k 'regex:(?i).*knn.*' \
  --print-summary per-kernel \
  --csv \
  --log-file "$NCU_OUTPUT.csv" \
  --force-overwrite \
  -o "$NCU_OUTPUT" \
  "$REPO_DIR/.venv/bin/python" "$REPO_DIR/scripts/profile_cukd_io.py" \
  --repo "$REPO_DIR" --data-root "$REPO_DIR/data/s3dis" --gpu 0

echo "NCU report: $NCU_OUTPUT.ncu-rep"
echo "NCU CSV: $NCU_OUTPUT.csv"
