SHELL := /bin/bash

.PHONY: run
# Usage:
#   make run                           # runs with defaults
#   make run ARGS="--tickers AAPL,MSFT --min_oi 200 --min_vol 50"
run:
	@bash ./run.sh $(ARGS)
