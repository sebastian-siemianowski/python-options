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

# Force extremely high thresholds to skip option chain candidates
MIN_OI=99999999
MIN_VOL=99999999

exec "$PY_CMD" options.py --min_oi "$MIN_OI" --min_vol "$MIN_VOL" "$@"
