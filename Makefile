SHELL := /bin/bash

.PHONY: run backtest doctor clear top50 build-russell bagger50 fx-plnjpy tests
# Usage:
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
#   make tests                         # runs all tests in the tests/ directory

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
	@.venv/bin/python scripts/fx_pln_jpy_signals.py $(ARGS)

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
