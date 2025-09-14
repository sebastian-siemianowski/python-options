#!/usr/bin/env bash
# setup_venv.sh
# Create a virtual environment and install requirements without relying on a standalone `pip` in PATH.
# Usage: bash setup_venv.sh

set -euo pipefail

# Prefer python3, fallback to python
PY_CMD="python3"
if ! command -v python3 >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PY_CMD="python"
  else
    echo "Error: python3/python not found. Please install Python 3 first (see README)." >&2
    exit 1
  fi
fi

echo "Using Python: $($PY_CMD --version 2>&1)"

# Create venv if missing
if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  $PY_CMD -m venv .venv
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

echo "Upgrading pip ..."
python -m pip install --upgrade pip

echo "Installing project requirements ..."
python -m pip install -r requirements.txt

echo "\nAll set! Virtual environment is active. To activate it later:"
if [[ -f .venv/bin/activate ]]; then
  echo "  source .venv/bin/activate"
else
  echo "  .venv\\Scripts\\Activate.ps1   # on Windows PowerShell"
fi

echo "To run the screener:"
echo "  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50"