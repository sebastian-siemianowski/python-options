#!/usr/bin/env bash
# backtest.sh — quick command to run ONLY the strategy backtest portion
# It achieves this by setting very high liquidity thresholds so the options screener
# finds no candidates, while the per-ticker strategy backtest still runs and prints
# combined profitability + average per-ticker profitability.
#
# Usage examples:
#   ./backtest.sh                              # use tickers.csv or defaults
#   ./backtest.sh --tickers AAPL,MSFT,NVDA     # override tickers
#   ./backtest.sh --bt_years 3 --bt_dte 7      # tweak backtest params
#   make backtest ARGS="--tickers AAPL,MSFT"

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

# Force extremely high thresholds to skip option chain candidates
MIN_OI=99999999
MIN_VOL=99999999

exec "$PY_CMD" options.py --min_oi "$MIN_OI" --min_vol "$MIN_VOL" "$@"
