SHELL := /bin/bash

.PHONY: run backtest doctor clear top50 build-russell bagger50 fx-plnjpy fx-diagnostics fx-diagnostics-lite fx-calibration fx-model-comparison fx-validate-kalman fx-validate-kalman-plots tune show-q clear-q tests report top20 data four purge failed setup
# Usage:
#   make setup                         # full setup: install deps + download all data (runs 3x for reliability)
#   make run                           # runs with defaults (screener + backtest)
#   make run ARGS="--tickers AAPL,MSFT --min_oi 200 --min_vol 50"
#   make backtest                      # runs backtest-only convenience wrapper
#   make backtest ARGS="--tickers AAPL,MSFT --bt_years 3"
#   make doctor                        # (re)installs requirements into the venv
#   make top50                         # runs the revenue growth screener
#   make top50 ARGS=""                 # extra args (reserved)
#   make build-russell                 # builds data/russell2500_tickers.csv from public sources
#   make bagger50                      # ranks by highest 100× Bagger Score (adds 100× Score column)
#   make bagger50 ARGS="--bagger_horizon 15"   # optional flags; also supports --top_n, --plain, --bagger_verbose
#   make fx-plnjpy                     # generate PLN/JPY FX signals (see README)
#   make fx-diagnostics                # full diagnostics: log-likelihood, parameter stability, OOS tests (expensive)
#   make fx-diagnostics-lite           # lightweight diagnostics: log-likelihood and parameter stability (no OOS)
#   make fx-calibration                # PIT calibration verification (tests if probabilities match outcomes)
#   make fx-model-comparison           # structural model comparison via AIC/BIC (GARCH vs EWMA, etc.)
#   make fx-validate-kalman            # Level-7 Kalman validation science (drift, likelihood, PIT, stress)
#   make fx-validate-kalman-plots      # Kalman validation with diagnostic plots saved to plots/kalman_validation/
#   make tune                          # estimate optimal Kalman drift q parameters via MLE (caches results)
#   make tune ARGS="--force"           # re-estimate q for all assets (ignore cache)
#   make show-q                        # display cached q parameter estimates
#   make clear-q                       # clear q parameter cache
#   make tests                         # runs all tests in the tests/ directory
#   make data                          # precaches securities data for faster screening/backtesting
#   make failed                        # list assets that failed processing
#   make purge                          # purge cached data for failed assets
#   make purge ARGS="--all"             # purge cache AND clear the failed assets list

# Ensure virtual environment exists before running commands
.venv/bin/python:
	@bash ./setup_venv.sh

# Dependency stamp to ensure requirements are installed before running tools
.venv/.deps_installed: requirements.txt | .venv/bin/python
	@echo "Installing/updating Python dependencies from requirements.txt ..."
	@.venv/bin/python -m pip install -r requirements.txt
	@touch .venv/.deps_installed

run: .venv/.deps_installed
	@bash ./run.sh $(ARGS)

backtest: .venv/.deps_installed
	@bash ./backtest.sh $(ARGS)

build-russell: .venv/.deps_installed
	@.venv/bin/python scripts/build_russell2500.py --out data/universes/russell2500_tickers.csv

top50: .venv/.deps_installed
	@.venv/bin/python top50_revenue_growth.py $(ARGS)

bagger50: .venv/.deps_installed
	@.venv/bin/python top50_revenue_growth.py --sort_by bagger $(ARGS)

fx-plnjpy: .venv/.deps_installed
	@.venv/bin/python scripts/fx_pln_jpy_signals.py $(ARGS) --cache-json cache/fx_plnjpy.json

# Diagnostics and validation convenience targets for FX signals
fx-diagnostics: .venv/.deps_installed
	@echo "Running full diagnostics (log-likelihood, parameter stability, OOS tests)..."
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --diagnostics $(ARGS)

fx-diagnostics-lite: .venv/.deps_installed
	@echo "Running lightweight diagnostics (no OOS tests)..."
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --diagnostics_lite $(ARGS)

fx-calibration: .venv/.deps_installed
	@echo "Running PIT calibration verification..."
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --pit-calibration $(ARGS)

fx-model-comparison: .venv/.deps_installed
	@echo "Running structural model comparison (AIC/BIC)..."
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --model-comparison $(ARGS)

fx-validate-kalman: .venv/.deps_installed
	@echo "Running Level-7 Kalman validation science..."
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --validate-kalman $(ARGS)

fx-validate-kalman-plots: .venv/.deps_installed
	@echo "Running Kalman validation with diagnostic plots..."
	@mkdir -p plots/kalman_validation
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --validate-kalman --validation-plots $(ARGS)

# Kalman q parameter tuning via MLE (with world-class UX)
tune: .venv/.deps_installed
	@mkdir -p cache
	@.venv/bin/python scripts/tune_pretty.py $(ARGS)

show-q:
	@if [ -f cache/kalman_q_cache.json ]; then \
		echo "=== Cached Kalman q Parameters (JSON) ==="; \
		cat cache/kalman_q_cache.json; \
	else \
		echo "No cache file found. Run 'make tune' first."; \
	fi

clear-q:
	@echo "Clearing Kalman q parameter cache..."
	@rm -f cache/kalman_q_cache.json
	@echo "Cache cleared."

tests: .venv/.deps_installed
	@echo "Running all tests..."
	@.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

# Manually (re)install requirements and refresh the dependency stamp
doctor: .venv/bin/python
	@echo "(Re)installing dependencies into the virtual environment ..."
	@.venv/bin/python -m pip install -r requirements.txt
	@touch .venv/.deps_installed
	@echo "Dependencies installed."

clear:
	@echo "Clearing data cache..."
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf plots/*.png
	@rm -rf data/meta/
	@rm -f data/*.backup
	@echo "Data cache cleared successfully!"

stocks: fx-plnjpy

# Render from cached results only (no network/compute)
report: .venv/.deps_installed
	@.venv/bin/python scripts/fx_pln_jpy_signals.py --from-cache --cache-json cache/fx_plnjpy.json

# Quick smoke: run only the first 20 assets
top20: .venv/.deps_installed
	@ASSETS=$$(PYTHONPATH=$(CURDIR) ./.venv/bin/python -c "import importlib.util, pathlib; fx=pathlib.Path('scripts/fx_data_utils.py'); spec=importlib.util.spec_from_file_location('fx_data_utils', fx); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); assets=sorted(list(getattr(mod,'DEFAULT_ASSET_UNIVERSE'))); print(','.join(assets[:20]))"); \
	if [ -z "$$ASSETS" ]; then echo 'No assets resolved'; exit 1; fi; \
	$(MAKE) fx-plnjpy ARGS="--assets $$ASSETS"

# Precache securities data (full history) - runs 5 download passes for reliability
data: .venv/.deps_installed
	@echo "Downloading price data (5 passes for reliability)..."
	@.venv/bin/python scripts/refresh_data.py --skip-trim --retries 5 --workers 2 --batch-size 16 $(ARGS)

# Refresh data: delete last 5 days from cache, then bulk re-download 5 times
refresh: .venv/.deps_installed
	@echo "Refreshing price data (trim last 5 days, then 5 download passes)..."
	@.venv/bin/python scripts/refresh_data.py --days 5 --retries 5 --workers 2 --batch-size 16 $(ARGS)

four:
	@if [ ! -f cache/kalman_q_cache.json ]; then \
		echo "cache/kalman_q_cache.json not found"; exit 1; \
	fi
	@PYTHONPATH=$(CURDIR) .venv/bin/python -c "from scripts.fx_data_utils import drop_first_k_from_kalman_cache; removed = drop_first_k_from_kalman_cache(4, 'cache/kalman_q_cache.json'); print(f'Removed {len(removed)} entries: {', '.join(removed)}')"

# List failed assets
failed: .venv/.deps_installed
	@.venv/bin/python scripts/purge_failed.py --list

# Purge cached data for failed assets
purge: .venv/.deps_installed
	@.venv/bin/python scripts/purge_failed.py $(ARGS)

# Full setup: create venv, install dependencies, and download all data (runs 3x for reliability)
setup:
	@echo "============================================================"
	@echo "STEP 1/4: Setting up Python virtual environment..."
	@echo "============================================================"
	@bash ./setup_venv.sh
	@echo ""
	@echo "============================================================"
	@echo "STEP 2/4: Installing Python dependencies..."
	@echo "============================================================"
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install -r requirements.txt
	@touch .venv/.deps_installed
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Downloading price data (Pass 1 of 3)..."
	@echo "============================================================"
	@.venv/bin/python scripts/precache_data.py --workers 2 --batch-size 16 || true
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Downloading price data (Pass 2 of 3)..."
	@echo "============================================================"
	@.venv/bin/python scripts/precache_data.py --workers 2 --batch-size 16 || true
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Downloading price data (Pass 3 of 3)..."
	@echo "============================================================"
	@.venv/bin/python scripts/precache_data.py --workers 2 --batch-size 16 || true
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Cleaning cached data (removing empty rows)..."
	@echo "============================================================"
	@.venv/bin/python scripts/clean_cache.py
	@echo ""
	@echo "============================================================"
	@echo "STEP 4/4: Setup complete!"
	@echo "============================================================"
	@echo ""
	@echo "You can now run:"
	@echo "  make fx-plnjpy    - Generate FX/asset signals"
	@echo "  make tune         - Tune Kalman filter parameters"
	@echo "  make failed       - List any assets that failed to download"
	@echo "  make purge        - Purge cache for failed assets"
	@echo "  make clean-cache  - Remove empty rows from cached data"
	@echo ""

# Clean cached price data by removing empty rows (dates before company existed)
clean-cache: .venv/.deps_installed
	@.venv/bin/python scripts/clean_cache.py

colors: .venv/.deps_installed
	@.venv/bin/python scripts/show_colors.py

