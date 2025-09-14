SHELL := /bin/bash

.PHONY: run backtest doctor clear
# Usage:
#   make run                           # runs with defaults (screener + backtest)
#   make run ARGS="--tickers AAPL,MSFT --min_oi 200 --min_vol 50"
#   make backtest                      # runs backtest-only convenience wrapper
#   make backtest ARGS="--tickers AAPL,MSFT --bt_years 3"
#   make doctor                        # checks your Python environment and libs
#   make doctor ARGS="--fix"           # auto-install requirements

# Ensure virtual environment exists before running commands
.venv/bin/python:
	@bash ./setup_venv.sh

run: .venv/bin/python
	@bash ./run.sh $(ARGS)

backtest: .venv/bin/python
	@bash ./backtest.sh $(ARGS)

clear:
	@echo "Clearing data cache..."
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf plots/*.png
	@rm -rf data/meta/
	@rm -f data/*.backup
	@echo "Data cache cleared successfully!"
