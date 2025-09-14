SHELL := /bin/bash

.PHONY: run backtest doctor
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
