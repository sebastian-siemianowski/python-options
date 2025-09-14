#!/usr/bin/env bash
# run.sh — convenience launcher for options.py
# Usage examples:
#   ./run.sh
#   ./run.sh --tickers AAPL,MSFT --min_oi 200 --min_vol 50

set -euo pipefail

# Encourage colored output from Rich even under make or non-TTY
export PYTHONUNBUFFERED=1
# Honor NO_COLOR if user sets it; otherwise force color
if [[ -z "${NO_COLOR:-}" ]]; then
  export FORCE_COLOR=1
fi

# Prefer virtualenv Python if available
if [[ -x ".venv/bin/python" ]]; then
  PY_CMD=".venv/bin/python"
else
  if command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
  elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
  else
    echo "Error: Python not found. Please install Python 3 (see README)." >&2
    exit 1
  fi
fi

exec "$PY_CMD" options.py "$@"
