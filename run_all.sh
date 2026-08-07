#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${GPU+x}" ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2,2n | head -1 | cut -d, -f1 | tr -d ' ')"
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export GPU RUN_ID

echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU"
bash "$REPO_DIR/scripts/preflight.sh"
bash "$REPO_DIR/run_query.sh"
bash "$REPO_DIR/run_network_latency.sh"
VALIDATE_ARGS=(--run-dir "$REPO_DIR/results/$RUN_ID")
if [[ "${SMOKE:-0}" == "1" ]]; then VALIDATE_ARGS+=(--smoke); fi
python "$REPO_DIR/scripts/validate_result_coverage.py" "${VALIDATE_ARGS[@]}"
python "$REPO_DIR/analysis/analyze_results.py" \
  --results "$REPO_DIR/results/$RUN_ID" \
  --output-dir "$REPO_DIR/analysis/output/$RUN_ID"
echo "All outputs: $REPO_DIR/results/$RUN_ID"
echo "Analysis: $REPO_DIR/analysis/output/$RUN_ID"
