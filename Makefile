SHELL := /bin/bash

.PHONY: run backtest doctor clear top50 top100 build-russell russell5000 bagger50 fx-plnjpy fx-diagnostics fx-diagnostics-lite fx-calibration fx-model-comparison fx-validate-kalman fx-validate-kalman-plots tune retune calibrate show-q clear-q tests report top20 data four purge failed setup temp metals debt risk market chain chain-force chain-dry stocks options-tune options-tune-force options-tune-dry arena arena-data arena-tune arena-results arena-safe-storage arena-safe pit pit-metals pit-full pit-g metals-diag diag diag-pit diag-debug diag-refine

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
# │  make options-tune       Tune volatility models for high conviction options  │
# │  make chain              Generate options signals using tuned parameters     │
# │  make chain-force        Force re-tune all volatility models                 │
# │  make chain-dry          Preview options pipeline (no processing)            │
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
# │  make retune             Refresh data, backup tune folder, run tune          │
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
# │  🌡️  RISK DASHBOARD (Unified Risk + Market Direction)                        │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make risk               Full dashboard: cross-asset + metals + equity +     │
# │                          market direction (indices, universes, sectors)      │
# │  make risk ARGS="--json" Output as JSON                                      │
# │  make temp               Cross-asset risk temperature only                   │
# │  make metals             Metals risk temperature only                        │
# │  make market             Equity market temperature only                      │
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
# │  make pit                Run PIT calibration test (22 assets, full tuning)   │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🏟️  ARENA — Experimental Model Competition                                  │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make arena              Full competition: data + tune + results             │
# │  make arena-data         Download benchmark data (12 symbols)                │
# │  make arena-tune         Run model competition (standard + experimental)     │
# │  make arena-results      Show latest competition results                     │
# │                                                                              │
# │  Benchmark Universe:                                                         │
# │    Small Cap: UPST, AFRM, IONQ                                              │
# │    Mid Cap:   CRWD, DKNG, SNAP                                              │
# │    Large Cap: AAPL, NVDA, TSLA                                              │
# │    Index:     SPY, QQQ, IWM                                                 │
# │                                                                              │
# │  Experimental models compete against standard momentum models:               │
# │    - momentum_gaussian, momentum_phi_gaussian                                │
# │    - momentum_phi_student_t_nu_{4,6,8,12,20}                                │
# │                                                                              │
# │  Models must beat standard by >5% to qualify for promotion.                  │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  ⚖️  STRUCTURAL BACKTEST ARENA — Behavioral Validation Layer                 │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make arena-backtest-data     Download 50-ticker multi-sector universe       │
# │  make arena-backtest-tune     Tune parameters (MULTIPROCESSING, fast)        │
# │  make arena-backtest-tune ARGS="--workers 8"     Use 8 parallel workers      │
# │  make arena-backtest-tune ARGS="--no-parallel"   Sequential mode             │
# │  make arena-backtest          Run behavioral backtests + safety rules        │
# │  make arena-backtest-results  Show latest backtest results                   │
# │                                                                              │
# │  NON-OPTIMIZATION CONSTITUTION:                                              │
# │    - Financial metrics are OBSERVATIONAL ONLY                                │
# │    - Decisions based on BEHAVIORAL SAFETY, not raw performance               │
# │    - One-way flow: Tuning → Backtest → Integration Trial                     │
# │                                                                              │
# │  Decision Outcomes: APPROVED | RESTRICTED | QUARANTINED | REJECTED           │
# │                                                                              │
# │  Universe Coverage (50 tickers):                                             │
# │    Technology, Finance, Defence, Healthcare, Industrials,                    │
# │    Energy, Materials, Consumer, Communication                                │
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

# Full retune: refresh data, backup existing tune folder, then run tune
# Backup folder is named with timestamp: tune-bak/tune_YYYYMMDD_HHMMSS
retune: .venv/.deps_installed
	@echo "═══════════════════════════════════════════════════════════════════════════"
	@echo "  🔄 RETUNE: Refresh Data → Backup Tune → Run Tune"
	@echo "═══════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "📥 Step 1/3: Refreshing market data..."
	@$(MAKE) refresh
	@echo ""
	@echo "📦 Step 2/3: Backing up existing tune folder..."
	@if [ -d src/data/tune ] && [ -n "$$(ls -A src/data/tune 2>/dev/null)" ]; then \
		BACKUP_NAME="tune_$$(date +%Y%m%d_%H%M%S)"; \
		mkdir -p src/data/tune-bak; \
		mv src/data/tune "src/data/tune-bak/$$BACKUP_NAME"; \
		echo "  ✅ Backed up to: src/data/tune-bak/$$BACKUP_NAME"; \
		mkdir -p src/data/tune; \
	else \
		echo "  ℹ️  No existing tune folder to backup (or empty)"; \
		mkdir -p src/data/tune; \
	fi
	@echo ""
	@echo "🎛️  Step 3/3: Running tune..."
	@$(MAKE) tune $(ARGS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════"
	@echo "  ✅ RETUNE COMPLETE"
	@echo "═══════════════════════════════════════════════════════════════════════════"

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

# =============================================================================
# ONLINE BAYESIAN PARAMETER UPDATES (February 2026)
# =============================================================================
# These targets manage the online parameter estimation system that adapts
# Kalman filter parameters in real-time using Sequential Monte Carlo.
#
# Uses multiprocessing (not threads) for CPU-bound particle filter operations.
#
# WORKFLOW:
#   1. make tune          - Estimate batch parameters (batch priors)
#   2. make online        - Run online updates using cached prices
#   3. make signal        - Generate signals with adaptive parameters
# =============================================================================

# Run online parameter updates for all tuned assets
# Uses cached price data and persists state to src/data/online_update/
# Default: uses all CPU cores for parallel processing
online: .venv/.deps_installed
	@echo "🔄 Running Online Bayesian Parameter Updates (multiprocessing)..."
	@mkdir -p src/data/online_update
	@.venv/bin/python src/tests/run_online_update.py $(ARGS)

# Show online update cache statistics
online-stats: .venv/.deps_installed
	@echo "📊 Online Update Cache Statistics:"
	@.venv/bin/python src/tests/online_stats.py

# Clear online update cache
online-clear: .venv/.deps_installed
	@echo "🗑️  Clearing online update cache..."
	@.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from calibration.online_update import clear_persisted_states; count = clear_persisted_states(); print('  Deleted', count, 'state files')"

# Verify online update is working
online-test: .venv/.deps_installed
	@echo "🧪 Testing Online Bayesian Parameter Updates..."
	@.venv/bin/python src/tests/verify_online_update.py

tests: .venv/.deps_installed
	@echo "Running all tests..."
	@.venv/bin/python -m unittest discover -s src/tests -p "test_*.py" -v

# PIT calibration test for unified Student-t model (failing assets only)
pit: .venv/.deps_installed
	@echo "🎯 Running PIT calibration test (failing assets only)..."
	@.venv/bin/python -B test_unified_pit_failures.py --full $(ARGS)

# PIT calibration test - metals only (GC=F, SI=F, XAGUSD)
pit-metals: .venv/.deps_installed
	@echo "🥇 Running PIT calibration test (metals: GC=F, SI=F, XAGUSD)..."
	@.venv/bin/python -B test_unified_pit_failures.py --metals $(ARGS)

# PIT calibration test - all PIT-failing and CRPS-failing assets
pit-special: .venv/.deps_installed
	@echo "🔬 Running PIT calibration test (special: all PIT+CRPS failing assets)..."
	@.venv/bin/python -B test_unified_pit_failures.py --special $(ARGS)

# PIT calibration test - comprehensive (all assets: failing + passing)
pit-full: .venv/.deps_installed
	@echo "🎯 Running COMPREHENSIVE PIT calibration test (all assets)..."
	@.venv/bin/python -B test_unified_pit_failures.py --all $(ARGS)

# PIT calibration test for Gaussian models (φ-Gaussian and pure Gaussian)
pit-g: .venv/.deps_installed
	@ASSET_COUNT=$$(grep -E "^\s*'[A-Z0-9=.-]+'" test_gaussian_pit_failures.py | head -100 | wc -l | tr -d ' '); \
	echo "🎯 Running Gaussian PIT calibration test ($$ASSET_COUNT assets)..."
	@.venv/bin/python -B test_gaussian_pit_failures.py --full $(ARGS)

# Comprehensive model diagnostics for Gold & Silver (all models, all metrics)
metals-diag: .venv/.deps_installed
	@echo "⚙️  Running comprehensive model diagnostics (GC=F, SI=F)..."
	@.venv/bin/python -B metals_model_diagnostics.py $(ARGS)

# Comprehensive model diagnostics for all low-PIT assets + Gold & Silver
# Options: --critical-only (p<0.01 only), --assets SYM1,SYM2, --no-reference
diag: .venv/.deps_installed
	@echo "⚙️  Running low-PIT model diagnostics (50 assets + GC=F, SI=F)..."
	@.venv/bin/python -B low_pit_diagnostics.py $(ARGS)

# PIT summary table only (no per-asset details)
diag-pit: .venv/.deps_installed
	@.venv/bin/python -B low_pit_diagnostics.py --pit-only $(ARGS)

# Deep PIT calibration debug (chi-squared correction simulation)
# Options: --assets SYM1,SYM2  --models U-t4,U-t8,U-t20
diag-debug: .venv/.deps_installed
	@.venv/bin/python -B debug_pit_calibration.py $(ARGS)

# PIT refinement debug (ν-MLE + location debiasing)
# Options: --assets SYM1,SYM2  --models U-t4,U-t8,U-t20
diag-refine: .venv/.deps_installed
	@.venv/bin/python -B debug_pit_refinement.py $(ARGS)

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

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  🔗 OPTIONS CHAIN ANALYSIS (Hierarchical Bayesian Framework)                 │
# ├──────────────────────────────────────────────────────────────────────────────┤
# │  make options-tune       Tune volatility models for high conviction options  │
# │  make options-tune-force Force re-tune all volatility models                 │
# │  make chain              Generate options signals using tuned parameters     │
# │  make chain-force        Force re-tune + generate signals                    │
# │  make chain-dry          Preview options pipeline (no processing)            │
# └──────────────────────────────────────────────────────────────────────────────┘

# Options volatility tuning - reads from src/data/high_conviction/
# Tunes constant_vol, mean_reverting_vol, regime_vol, regime_skew_vol, variance_swap_anchor
# Saves results to src/data/option_tune/
options-tune: .venv/.deps_installed
	@PYTHONPATH=$(CURDIR) .venv/bin/python -m src.decision.option_tune $(ARGS)

options-tune-force: .venv/.deps_installed
	@PYTHONPATH=$(CURDIR) .venv/bin/python -m src.decision.option_tune --force $(ARGS)

options-tune-dry: .venv/.deps_installed
	@PYTHONPATH=$(CURDIR) .venv/bin/python -m src.decision.option_tune --dry-run $(ARGS)

# Options signal pipeline - reads from src/data/high_conviction/
# Uses tuned volatility models from src/data/option_tune/
chain: .venv/.deps_installed
	@PYTHONPATH=$(CURDIR) .venv/bin/python -m src.decision.option_signal $(ARGS)

chain-force: .venv/.deps_installed
	@PYTHONPATH=$(CURDIR) .venv/bin/python -m src.decision.option_signal --force $(ARGS)

chain-dry: .venv/.deps_installed
	@PYTHONPATH=$(CURDIR) .venv/bin/python -m src.decision.option_signal --dry-run $(ARGS)

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

# ═══════════════════════════════════════════════════════════════════════════════
# ARENA — Experimental Model Competition (February 2026)
# ═══════════════════════════════════════════════════════════════════════════════
# Isolated sandbox for testing experimental models against production baselines.
# Experimental models must beat standard momentum models by >5% to graduate.
#
# Benchmark Universe (12 symbols):
#   Small Cap: UPST, AFRM, IONQ
#   Mid Cap:   CRWD, DKNG, SNAP
#   Large Cap: AAPL, NVDA, TSLA
#   Index:     SPY, QQQ, IWM
#
# Standard Models (baselines):
#   - kalman_gaussian_momentum, kalman_phi_gaussian_momentum
#   - phi_student_t_nu_{4,6,8,12,20}_momentum
#
# Experimental Models (in src/arena/arena_models.py):
#   - momentum_student_t_v2 (adaptive tail coupling)
#   - momentum_student_t_regime_coupled (regime-aware ν)
# ═══════════════════════════════════════════════════════════════════════════════

# Full arena workflow: run competition + show results (use arena-data to refresh data)
arena: .venv/.deps_installed
	@echo ""
	@$(MAKE) arena-tune
	@$(MAKE) arena-results

# Download benchmark data for arena (3 small cap, 3 mid cap, 3 large cap, 3 index)
# Run this explicitly when you need fresh data: make arena-data
arena-data: .venv/.deps_installed
	@echo "Downloading arena benchmark data..."
	@mkdir -p src/arena/data
	@.venv/bin/python src/arena/arena_cli.py data $(ARGS)

# Run arena model competition (standard + experimental models)
arena-tune: .venv/.deps_installed
	@mkdir -p src/arena/data/results
	@mkdir -p src/arena/disabled
	@.venv/bin/python src/arena/arena_cli.py tune $(ARGS)

# Show latest arena competition results
arena-results: .venv/.deps_installed
	@.venv/bin/python src/arena/arena_cli.py results

# Show models in safe storage (archived competition winners)
arena-safe-storage: .venv/.deps_installed
	@.venv/bin/python src/arena/show_safe_storage.py

# Run arena tests on all safe_storage models and update results
arena-safe: .venv/.deps_installed
	@echo ""
	@echo "Running Safe Storage Arena..."
	@mkdir -p src/arena/safe_storage/data
	@.venv/bin/python src/arena/safe_storage/run_safe_arena.py $(ARGS)

# Show disabled experimental models
arena-disabled: .venv/.deps_installed
	@.venv/bin/python src/arena/arena_cli.py disabled

# Re-enable a disabled model (usage: make arena-enable MODEL=model_name)
arena-enable: .venv/.deps_installed
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make arena-enable MODEL=model_name"; \
		echo "Available disabled models:"; \
		.venv/bin/python src/arena/arena_cli.py disabled; \
	else \
		.venv/bin/python src/arena/arena_cli.py disabled --enable $(MODEL); \
	fi

# Re-enable all disabled models
arena-enable-all: .venv/.deps_installed
	@.venv/bin/python src/arena/arena_cli.py disabled --clear

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL BACKTEST ARENA — Behavioral Validation Layer
# ═══════════════════════════════════════════════════════════════════════════════
# A "court of law" for models — behavioral validation with full diagnostics.
# Financial metrics are OBSERVATIONAL ONLY — never used for optimization.
#
# NON-OPTIMIZATION CONSTITUTION:
#   1. Separation of Powers: Backtest tuning isolated from production tuning
#   2. One-Way Flow: Tuning → Backtest → Integration Trial (no reverse)
#   3. Behavior Over Performance: Diagnostics inform safety, not returns
#   4. Representativeness: 50 tickers across sectors/caps/regimes
#
# Decision Outcomes: APPROVED | RESTRICTED | QUARANTINED | REJECTED
# ═══════════════════════════════════════════════════════════════════════════════

# Download backtest data for the 50-ticker canonical universe
arena-backtest-data: .venv/.deps_installed
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════════════"
	@echo "  STRUCTURAL BACKTEST DATA PIPELINE"
	@echo "════════════════════════════════════════════════════════════════════════════"
	@mkdir -p src/arena/data/backtest_data
	@.venv/bin/python src/arena/backtest_cli.py data $(ARGS)

# Tune backtest-specific parameters (for FAIRNESS, not optimization)
arena-backtest-tune: .venv/.deps_installed
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════════════"
	@echo "  BACKTEST PARAMETER TUNING (for fairness, not optimization)"
	@echo "════════════════════════════════════════════════════════════════════════════"
	@mkdir -p src/arena/backtest_tuned_params
	@.venv/bin/python src/arena/backtest_cli.py tune $(ARGS)

# Execute structural backtests and apply behavioral safety rules
arena-backtest: .venv/.deps_installed
	@mkdir -p src/arena/data/backtest_results
	@.venv/bin/python src/arena/backtest_cli.py run $(ARGS)

# Show latest structural backtest results
arena-backtest-results: .venv/.deps_installed
	@.venv/bin/python src/arena/backtest_cli.py results

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
# Combines all risk temperature modules into a single view:
#   - Cross-asset stress indicators (risk_temperature)
#   - Metals crash risk and overnight exposure (metals_risk_temperature)
#   - US equity market momentum and sector rotation (market_temperature)
# ═══════════════════════════════════════════════════════════════════════════════

# Unified Risk Dashboard - combined view of all risk indicators
risk: .venv/.deps_installed
	@.venv/bin/python src/decision/risk_dashboard.py $(ARGS)

# Individual temperature modules (for legacy/debugging)
temp: .venv/.deps_installed
	@.venv/bin/python src/decision/risk_temperature.py

metals: .venv/.deps_installed
	@.venv/bin/python src/decision/metals_risk_temperature.py

market: .venv/.deps_installed
	@.venv/bin/python src/decision/market_temperature.py $(ARGS)
