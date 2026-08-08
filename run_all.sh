#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/python_env.sh"

if [[ -z "${GPU+x}" ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2,2n | head -1 | cut -d, -f1 | tr -d ' ')"
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export GPU RUN_ID
RUN_DIR="$REPO_DIR/results/$RUN_ID"

echo "Run ID: $RUN_ID"
echo "Physical GPU: $GPU"
echo "Python: $PYTHON_BIN ($EXPREPO_ENV_KIND)"
if [[ -n "${REUSE_RUN_DIRS:-}" ]]; then
  IFS=: read -r -a REUSE_SOURCES <<< "$REUSE_RUN_DIRS"
  MERGE_ARGS=(--destination "$RUN_DIR")
  for source in "${REUSE_SOURCES[@]}"; do MERGE_ARGS+=(--source "$source"); done
  "$PYTHON_BIN" "$REPO_DIR/scripts/merge_run_results.py" "${MERGE_ARGS[@]}"
fi
bash "$REPO_DIR/scripts/preflight.sh"
bash "$REPO_DIR/run_query.sh"
bash "$REPO_DIR/run_network_latency.sh"
VALIDATE_ARGS=(--run-dir "$RUN_DIR")
if [[ "${SMOKE:-0}" == "1" ]]; then VALIDATE_ARGS+=(--smoke); fi
"$PYTHON_BIN" "$REPO_DIR/scripts/validate_result_coverage.py" "${VALIDATE_ARGS[@]}"
"$PYTHON_BIN" "$REPO_DIR/analysis/analyze_results.py" \
  --results "$RUN_DIR" \
  --output-dir "$REPO_DIR/analysis/output/$RUN_ID"
echo "All outputs: $REPO_DIR/results/$RUN_ID"
echo "Analysis: $REPO_DIR/analysis/output/$RUN_ID"
