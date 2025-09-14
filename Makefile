SHELL := /bin/bash

.PHONY: run backtest
# Usage:
#   make run                           # runs with defaults (screener + backtest)
#   make run ARGS="--tickers AAPL,MSFT --min_oi 200 --min_vol 50"
#   make backtest                      # runs backtest-only convenience wrapper
#   make backtest ARGS="--tickers AAPL,MSFT --bt_years 3"
run:
	@bash ./run.sh $(ARGS)

backtest:
	@bash ./backtest.sh $(ARGS)
