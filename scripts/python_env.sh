#!/usr/bin/env bash
# Select the Python interpreter for Conda, uv/venv, or an explicit override.

if [[ -z "${REPO_DIR:-}" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

select_experiment_python() {
  local candidate=""
  local environment_kind=""

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidate="$PYTHON_BIN"
    environment_kind="explicit PYTHON_BIN"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    candidate="$VIRTUAL_ENV/bin/python"
    environment_kind="active venv/uv"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    candidate="$CONDA_PREFIX/bin/python"
    environment_kind="active conda"
  elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    candidate="$REPO_DIR/.venv/bin/python"
    environment_kind="repository uv/venv"
  elif command -v python >/dev/null 2>&1; then
    candidate="$(command -v python)"
    environment_kind="PATH"
  elif command -v python3 >/dev/null 2>&1; then
    candidate="$(command -v python3)"
    environment_kind="PATH"
  else
    echo "No Python interpreter found. Activate Conda, activate a uv venv, or set PYTHON_BIN." >&2
    return 1
  fi

  if [[ ! -x "$candidate" ]]; then
    echo "Selected Python is not executable: $candidate" >&2
    return 1
  fi
  PYTHON_BIN="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
  EXPREPO_PYTHON_PREFIX="$(cd "$(dirname "$candidate")/.." && pwd)"
  EXPREPO_ENV_KIND="$environment_kind"
  export PYTHON_BIN EXPREPO_PYTHON_PREFIX EXPREPO_ENV_KIND
}

select_experiment_python
