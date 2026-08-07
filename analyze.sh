#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$REPO_DIR/analysis/analyze_results.py" --results "$REPO_DIR/results" \
  --output-dir "$REPO_DIR/analysis/output"
