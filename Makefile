SHELL := /bin/bash

.PHONY: run backtest doctor clear top50 top100 build-russell russell5000 bagger50 fx-plnjpy fx-diagnostics fx-diagnostics-lite fx-calibration fx-model-comparison fx-validate-kalman fx-validate-kalman-plots tune calibrate show-q clear-q tests report top20 data four purge failed setup temp metals debt

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAKEFILE USAGE                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🚀 SETUP & INSTALLATION                                                     │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make setup              Full setup: venv + deps + download data (3 passes) │
# │  make doctor             (Re)install requirements into virtual environment  │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  📊 OPTIONS SCREENER & BACKTEST                                              │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make run                Run screener + backtest with defaults               │
# │  make run ARGS="--tickers AAPL,MSFT --min_oi 200"                            │
# │  make backtest           Run backtest-only mode                              │
# │  make backtest ARGS="--tickers AAPL,MSFT --bt_years 3"                       │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  📈 FX & ASSET SIGNALS                                                       │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make stocks             Download prices + generate signals for all assets   │
# │  make fx-plnjpy          Generate PLN/JPY FX signals                         │
# │  make report             Render from cached results (no network)             │
# │  make top20              Quick smoke test: first 20 assets only              │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🔬 DIAGNOSTICS & VALIDATION                                                 │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make fx-diagnostics     Full diagnostics (log-LL, stability, OOS tests)    │
# │  make fx-diagnostics-lite Lightweight (no OOS tests)                         │
# │  make fx-calibration     PIT calibration verification                        │
# │  make fx-model-comparison Structural model comparison (AIC/BIC)              │
# │  make fx-validate-kalman Level-7 Kalman validation science                   │
# │  make fx-validate-kalman-plots  Validation with diagnostic plots             │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🎛️  KALMAN TUNING & CALIBRATION                                             │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make tune               Estimate optimal Kalman q parameters via MLE        │
# │  make tune ARGS="--force"  Re-estimate all (ignore cache)                    │
# │  make calibrate          Re-tune only assets with PIT failures (p < 0.05)   │
# │  make calibrate-four     Re-tune 4 random failing assets (for testing)      │
# │  make escalate           Re-tune assets needing escalation (mixture/ν)      │
# │  make show-q             Display cached q parameter estimates                │
# │  make clear-q            Clear q parameter cache                             │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  📂 CACHE MANAGEMENT                                                         │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make cache-stats        Show tuning cache statistics                        │
# │  make cache-list         List all cached symbols                             │
# │  make cache-migrate      Migrate legacy cache to per-asset files             │
# │  make four               Remove first 4 cached assets (for re-tuning)       │
# │  make clear              Clear data cache and temporary files                │
# │  make clean-cache        Remove empty rows from cached price data            │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  📥 DATA DOWNLOAD & MANAGEMENT                                               │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make data               Precache securities data (full history)             │
# │  make refresh            Delete last 5 days + re-download (5 passes)        │
# │  make failed             List assets that failed processing                  │
# │  make purge              Purge cached data for failed assets                 │
# │  make purge ARGS="--all"   Purge cache AND clear failed list                 │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🔍 STOCK SCREENERS                                                          │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make top50              Revenue growth screener (Russell 2500)              │
# │  make bagger50           100× Bagger Score ranking                           │
# │  make bagger50 ARGS="--bagger_horizon 15"                                    │
# │  make top100             Top 100 screener (Russell 5000 universe)            │
# │  make build-russell      Build Russell 2500 tickers CSV                      │
# │  make russell5000        Build Russell 5000 tickers CSV                      │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  💱 DEBT ALLOCATION (Multi-Currency)                                         │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make debt               Compare all currencies: EURJPY, AUDJPY, EURAUD      │
# │  make debt ARGS="--single"  Analyze only EURJPY (skip comparison)            │
# │  make debt ARGS="--aud"     Analyze only EURAUD (AUD debt)                   │
# │  make debt ARGS="--json"    Output as JSON                                   │
# │  make debt ARGS="--no-refresh"  Use cached data only                         │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🧪 TESTING                                                                  │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make tests              Run all tests in src/tests/                         │
# └──────────────────────────────────────────────────────────────────────────────┘

# Ensure virtual environment exists before running commands
.venv/bin/python:
	@bash ./src/setup/setup_venv.sh

# Dependency stamp to ensure requirements are installed before running tools
.venv/.deps_installed: src/setup/requirements.txt | .venv/bin/python
	@echo "Installing/updating Python dependencies from src/setup/requirements.txt ..."
	@.venv/bin/python -m pip install -r src/setup/requirements.txt
	@touch .venv/.deps_installed

run: .venv/.deps_installed
	@bash ./src/setup/run.sh $(ARGS)

backtest: .venv/.deps_installed
	@bash ./src/backtesting/backtest.sh $(ARGS)

build-russell: .venv/.deps_installed
	@.venv/bin/python src/screeners/russell2500.py --out src/data/russell/russell2500_tickers.csv

russell5000: .venv/.deps_installed
	@.venv/bin/python src/screeners/russell5000.py --out src/data/russell/russell5000_tickers.csv $(ARGS)

top50: .venv/.deps_installed
	@.venv/bin/python top50_revenue_growth.py $(ARGS)

bagger50: .venv/.deps_installed
	@.venv/bin/python top50_revenue_growth.py --sort_by bagger $(ARGS)

top100: .venv/.deps_installed
	@.venv/bin/python src/top100.py $(ARGS)

fx-plnjpy: .venv/.deps_installed
	@.venv/bin/python src/decision/signals.py $(ARGS) --cache-json src/data/currencies/fx_plnjpy.json

# Diagnostics and validation convenience targets for FX signals
fx-diagnostics: .venv/.deps_installed
	@echo "Running full diagnostics (log-likelihood, parameter stability, OOS tests)..."
	@.venv/bin/python src/decision/signals.py --diagnostics $(ARGS)

fx-diagnostics-lite: .venv/.deps_installed
	@echo "Running lightweight diagnostics (no OOS tests)..."
	@.venv/bin/python src/decision/signals.py --diagnostics_lite $(ARGS)

fx-calibration: .venv/.deps_installed
	@echo "Running PIT calibration verification..."
	@.venv/bin/python src/decision/signals.py --pit-calibration $(ARGS)

fx-model-comparison: .venv/.deps_installed
	@echo "Running structural model comparison (AIC/BIC)..."
	@.venv/bin/python src/decision/signals.py --model-comparison $(ARGS)

fx-validate-kalman: .venv/.deps_installed
	@echo "Running Level-7 Kalman validation science..."
	@.venv/bin/python src/decision/signals.py --validate-kalman $(ARGS)

fx-validate-kalman-plots: .venv/.deps_installed
	@echo "Running Kalman validation with diagnostic plots..."
	@mkdir -p src/data/plots/kalman_validation
	@.venv/bin/python src/decision/signals.py --validate-kalman --validation-plots $(ARGS)

# Kalman q parameter tuning via MLE (with world-class UX)
tune: .venv/.deps_installed
	@mkdir -p cache
	@.venv/bin/python src/tuning/tune_ux.py $(ARGS)

# Re-tune only assets that failed calibration without escalation attempt
# This targets assets where neither mixture nor ν-refinement was tried
# Use this after implementing new escalation logic to activate it
escalate: .venv/.deps_installed
	@echo "🔧 Re-tuning assets that need escalation (mixture/ν-refinement not attempted)..."
	@.venv/bin/python src/tuning/tune_ux.py --force-escalation $(ARGS)

# Re-tune 4 random assets with calibration failures
# Useful for testing calibration fixes incrementally
calibrate-four: .venv/.deps_installed
	@if [ ! -f src/data/calibration/calibration_failures.json ]; then \
		echo "❌ No calibration_failures.json found. Run 'make tune' first."; \
		exit 1; \
	fi
	@echo "🎲 Selecting 4 random assets with calibration failures..."
	@FAILED_ASSETS=$$(.venv/bin/python -c "import json, random; f=json.load(open('src/data/calibration/calibration_failures.json')); assets=[i['asset'] for i in f['issues']]; random.shuffle(assets); print(','.join(assets[:4]))"); \
	if [ -z "$$FAILED_ASSETS" ]; then \
		echo "✅ No calibration failures found. All assets are well-calibrated!"; \
	else \
		echo "🔧 Re-tuning: $$FAILED_ASSETS"; \
		.venv/bin/python src/tuning/tune_ux.py --assets "$$FAILED_ASSETS" --force $(ARGS); \
	fi

# Re-tune only assets with calibration failures (PIT p-value < 0.05)
# Uses calibration_failures.json from previous tune run
# Options:
#   make calibrate                          # Re-tune all calibration failures
#   make calibrate ARGS="--severity critical"  # Only critical failures
#   make calibrate ARGS="--dry-run"         # Preview what would be re-tuned
calibrate: .venv/.deps_installed
	@if [ ! -f src/data/calibration/calibration_failures.json ]; then \
		echo "❌ No calibration_failures.json found. Run 'make tune' first."; \
		exit 1; \
	fi
	@echo "📊 Extracting assets with calibration failures..."
	@FAILED_ASSETS=$$(.venv/bin/python src/calibration/extract_calibration_failures.py); \
	if [ -z "$$FAILED_ASSETS" ]; then \
		echo "✅ No calibration failures found. All assets are well-calibrated!"; \
	else \
		ASSET_COUNT=$$(echo "$$FAILED_ASSETS" | tr ',' '\n' | wc -l | tr -d ' '); \
		echo "🔧 Re-tuning $$ASSET_COUNT assets with calibration issues..."; \
		.venv/bin/python src/tuning/tune_ux.py --assets "$$FAILED_ASSETS" --force $(ARGS); \
	fi

# FX Debt Allocation Engine - EURJPY balance sheet convexity control
debt: .venv/.deps_installed
	@mkdir -p src/data/debt
	@.venv/bin/python src/debt/debt_allocator.py $(ARGS)

show-q:
	@if [ -d src/data/tune ] && [ "$$(ls -A src/data/tune/*.json 2>/dev/null | head -1)" ]; then \
		echo "=== Cached Kalman q Parameters (per-asset) ==="; \
		echo "Directory: src/data/tune/"; \
		.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from tuning.kalman_cache import list_cached_symbols, get_cache_stats; symbols=list_cached_symbols(); stats=get_cache_stats(); print(f'Total assets: {stats[\"n_assets\"]}'); print(f'Total size: {stats[\"total_size_kb\"]:.1f} KB'); print('First 20 symbols:', ', '.join(symbols[:20]) + ('...' if len(symbols) > 20 else ''))"; \
	else \
		echo "No cache files found. Run 'make tune' first."; \
	fi

clear-q:
	@echo "Clearing Kalman q parameter cache..."
	@rm -f src/data/kalman_q_cache.json
	@rm -f src/data/tune/*.json
	@echo "Cache cleared."

# Cache management utilities
cache-stats: .venv/.deps_installed
	@echo "📊 Kalman tuning cache statistics:"
	@.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from tuning.kalman_cache import get_cache_stats; s=get_cache_stats(); print(f'  Assets:     {s[\"n_assets\"]}'); print(f'  Total Size: {s[\"total_size_kb\"]:.1f} KB'); print(f'  Avg Size:   {s[\"avg_size_kb\"]:.1f} KB'); print(f'  Directory:  {s[\"cache_dir\"]}')"

cache-migrate: .venv/.deps_installed
	@echo "🔄 Migrating legacy cache to per-asset files..."
	@.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from tuning.kalman_cache import migrate_legacy_cache; migrate_legacy_cache()"

cache-migrate-bma: .venv/.deps_installed
	@echo "🔄 Adding has_bma flag to cache files..."
	@.venv/bin/python src/data_ops/migrate_has_bma.py

cache-list: .venv/.deps_installed
	@echo "📋 Cached symbols:"
	@.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from tuning.kalman_cache import list_cached_symbols; symbols=list_cached_symbols(); print(f'  Total: {len(symbols)} assets'); print('  ' + ', '.join(symbols[:20]) + ('...' if len(symbols) > 20 else ''))"

tests: .venv/.deps_installed
	@echo "Running all tests..."
	@.venv/bin/python -m unittest discover -s src/tests -p "test_*.py" -v

# Manually (re)install requirements and refresh the dependency stamp
doctor: .venv/bin/python
	@echo "(Re)installing dependencies into the virtual environment ..."
	@.venv/bin/python -m pip install -r src/setup/requirements.txt
	@touch .venv/.deps_installed
	@echo "Dependencies installed."

clear:
	@echo "Clearing data cache..."
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf src/data/plots/*.png
	@rm -rf src/data/options/meta/
	@rm -f data/*.backup
	@echo "Data cache cleared successfully!"

stocks: .venv/.deps_installed
	@.venv/bin/python src/data_ops/refresh_data.py --skip-trim --retries 5 --workers 12 --batch-size 16 $(ARGS)
	@$(MAKE) fx-plnjpy

# Render from cached results only (no network/compute)
report: .venv/.deps_installed
	@.venv/bin/python src/decision/signals.py --from-cache --cache-json src/data/currencies/fx_plnjpy.json

# Quick smoke: run only the first 20 assets
top20: .venv/.deps_installed
	@ASSETS=$$(PYTHONPATH=$(CURDIR) ./.venv/bin/python -c "import importlib.util, pathlib; fx=pathlib.Path('src/ingestion/data_utils.py'); spec=importlib.util.spec_from_file_location('fx_data_utils', fx); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); assets=sorted(list(getattr(mod,'DEFAULT_ASSET_UNIVERSE'))); print(','.join(assets[:20]))"); \
	if [ -z "$$ASSETS" ]; then echo 'No assets resolved'; exit 1; fi; \
	$(MAKE) fx-plnjpy ARGS="--assets $$ASSETS"

# Precache securities data (full history) - runs 5 download passes for reliability
data: .venv/.deps_installed
	@.venv/bin/python src/data_ops/refresh_data.py --skip-trim --retries 5 --workers 12 --batch-size 16 $(ARGS)

# Refresh data: delete last 5 days from cache, then bulk re-download 5 times
refresh: .venv/.deps_installed
	@.venv/bin/python src/data_ops/refresh_data.py --days 5 --retries 5 --workers 12 --batch-size 16 $(ARGS)

four:
	@if [ ! -d src/data/tune ] || [ -z "$$(ls -A src/data/tune/*.json 2>/dev/null | head -1)" ]; then \
		echo "No per-asset cache files found in src/data/tune/"; exit 1; \
	fi
	@PYTHONPATH=$(CURDIR) .venv/bin/python -c "from src.ingestion.data_utils import drop_first_k_from_kalman_cache; removed = drop_first_k_from_kalman_cache(4, 'src/data/tune'); print(f'Removed {len(removed)} entries: {chr(44).join(removed)}')"

# List failed assets
failed: .venv/.deps_installed
	@.venv/bin/python src/data_ops/purge_failed.py --list

# Purge cached data for failed assets
purge: .venv/.deps_installed
	@.venv/bin/python src/data_ops/purge_failed.py $(ARGS)

# Full setup: create venv, install dependencies, and download all data (runs 3x for reliability)
setup:
	@echo "============================================================"
	@echo "STEP 1/4: Setting up Python virtual environment..."
	@echo "============================================================"
	@bash ./src/setup/setup_venv.sh
	@echo ""
	@echo "============================================================"
	@echo "STEP 2/4: Installing Python dependencies..."
	@echo "============================================================"
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install -r src/setup/requirements.txt
	@touch .venv/.deps_installed
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Downloading price data (Pass 1 of 3)..."
	@echo "============================================================"
	@.venv/bin/python src/data_ops/precache_data.py --workers 2 --batch-size 16 || true
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Downloading price data (Pass 2 of 3)..."
	@echo "============================================================"
	@.venv/bin/python src/data_ops/precache_data.py --workers 2 --batch-size 16 || true
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Downloading price data (Pass 3 of 3)..."
	@echo "============================================================"
	@.venv/bin/python src/data_ops/precache_data.py --workers 2 --batch-size 16 || true
	@echo ""
	@echo "============================================================"
	@echo "STEP 3/4: Cleaning cached data (removing empty rows)..."
	@echo "============================================================"
	@.venv/bin/python src/data_ops/clean_cache.py
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
	@.venv/bin/python src/data_ops/clean_cache.py

colors: .venv/.deps_installed
	@.venv/bin/python src/show_colors.py

# Market Risk Temperature - cross-asset stress indicator
temp: .venv/.deps_installed
	@.venv/bin/python src/decision/risk_temperature.py
# Metals Risk Temperature - cross-metal stress indicator
metals: .venv/.deps_installed
	@.venv/bin/python src/decision/metals_risk_temperature.py
