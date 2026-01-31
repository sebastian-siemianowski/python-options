#!/usr/bin/env bash
# setup_venv.sh
# Create a virtual environment and install requirements without relying on docker-wrapped python/pip.
# Usage: bash src/setup/setup_venv.sh

set -euo pipefail

# Allow user to force a specific interpreter (e.g., export PYTHON=/usr/bin/python3)
if [[ -n "${PYTHON:-}" ]]; then
  CANDIDATES=("$PYTHON")
else
  # Try system interpreters first to avoid shell wrappers that may call Docker
  CANDIDATES=(
    "/usr/bin/python3"
    "/opt/homebrew/bin/python3"   # Apple Silicon Homebrew
    "/usr/local/bin/python3"      # Intel macOS Homebrew
    "python3"
    "python"
  )
fi

PY_CMD=""
for c in "${CANDIDATES[@]}"; do
  RES="$c"
  # If candidate is not an absolute path, resolve to the real executable, bypassing aliases/functions
  if [[ "$RES" != /* ]]; then
    RES="$(type -P "$RES" 2>/dev/null || true)"
  fi
  if [[ -n "$RES" && -x "$RES" ]]; then
    # Verify it is a working Python 3 and not a docker wrapper by running a tiny snippet
    if "$RES" -c "import sys; assert sys.version_info >= (3,7); print(sys.executable)" >/dev/null 2>&1; then
      PY_CMD="$RES"
      break
    fi
  fi
done

if [[ -z "$PY_CMD" ]]; then
  echo "Error: Could not find a working Python 3 interpreter."
  echo "Tips:"
  echo "  - On macOS, install with: brew install python@3"
  echo "  - On Linux, install with your package manager (e.g., apt install python3 python3-venv)"
  echo "  - If your shell wraps 'python' via Docker, run:"
  echo "      export PYTHON=/usr/bin/python3"
  echo "    and then re-run: bash src/setup/setup_venv.sh"
  exit 1
fi

echo "Using Python: $($PY_CMD --version 2>&1) [$PY_CMD]"

# Create venv if missing
if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PY_CMD" -m venv .venv || {
    echo "Failed to create venv with $PY_CMD. If your python is docker-wrapped, set PYTHON to a system interpreter, e.g.:" >&2
    echo "  export PYTHON=/usr/bin/python3" >&2
    exit 1
  }
fi

# Activate venv
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f .venv/Scripts/activate ]]; then
  # Windows Git Bash / Cygwin
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  echo "Error: could not find venv activation script. Tried .venv/bin/activate and .venv/Scripts/activate" >&2
  exit 1
fi

# Resolve the venv's Python explicitly to avoid shell aliases (e.g., docker-wrapped python)
if [[ -x ".venv/bin/python" ]]; then
  VENV_PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  VENV_PY=".venv/Scripts/python.exe"
else
  # Fallback to whatever "python" resolves to in PATH within the venv
  VENV_PY="$(command -v python || echo python)"
fi

echo "Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip

echo "Installing project requirements ..."
"$VENV_PY" -m pip install -r src/setup/requirements.txt

echo "\nAll set! Virtual environment is active. To activate it later:"
if [[ -f .venv/bin/activate ]]; then
  echo "  source .venv/bin/activate"
else
  echo "  .venv\\Scripts\\Activate.ps1   # on Windows PowerShell"
fi

echo "To run the screener:"
echo "  .venv/bin/python src/options/options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50  # or 'python src/options/options.py' after activating the venv"