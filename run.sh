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

# Prefer virtualenv Python if available; otherwise pick a real system Python 3 (avoid docker-wrapped aliases)
if [[ -x ".venv/bin/python" ]]; then
  PY_CMD=".venv/bin/python"
else
  CANDIDATES=(
    "/usr/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "python3"
    "python"
  )
  PY_CMD=""
  for c in "${CANDIDATES[@]}"; do
    RES="$c"
    if [[ "$RES" != /* ]]; then
      RES="$(type -P "$RES" 2>/dev/null || true)"
    fi
    if [[ -n "$RES" && -x "$RES" ]]; then
      if "$RES" -c "import sys; assert sys.version_info >= (3,7); print(sys.executable)" >/dev/null 2>&1; then
        PY_CMD="$RES"
        break
      fi
    fi
  done
  if [[ -z "$PY_CMD" ]]; then
    echo "Error: Could not find a working Python 3 interpreter. Install Python 3 or create the venv with: bash setup_venv.sh" >&2
    exit 1
  fi
fi

exec "$PY_CMD" options.py "$@"
