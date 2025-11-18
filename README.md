# python-options (macOS)

This repo contains a simple options screener (options.py). These instructions are tailored for macOS.

## 1) Install Python 3 (macOS)

You can either use the helper script or install via Homebrew directly.

- Using the helper script (recommended):

  chmod +x install_python.sh
  ./install_python.sh

- Or manually with Homebrew (https://brew.sh):

  brew install python

After installation, verify:

  python3 --version
  pip3 --version

## 2) Create and activate a virtual environment

From the project folder:

  python3 -m venv .venv
  source .venv/bin/activate

## 3) Install project dependencies

With the virtual environment active:

  python3 -m pip install -r requirements.txt

## 4) Run the screener

Example:

  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
- screener_results.csv
- screener_results_backtest.csv
- plots/<TICKER>_support_resistance.png

## Quick start (one command)

If you prefer automation, run:

  bash setup_venv.sh

This will create a virtual environment and install requirements using `python -m pip`.

## Troubleshooting (macOS)

- zsh: command not found: python
  - Use `python3` instead. Example: `python3 -m pip install -r requirements.txt`
  - If `python3` is not found, install via `./install_python.sh` or `brew install python`.
  - Optional (zsh): add an alias so `python` maps to `python3`:
    - echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc

- zsh: command not found: pip
  - Use the module form: `python3 -m pip install -r requirements.txt`
  - Ensure your virtual environment is activated: `source .venv/bin/activate`
  - If you recently installed Homebrew/Python, open a new terminal so PATH updates, or run `hash -r` in zsh.

- Apple Silicon build tools
  - If you encounter build errors for scipy or numpy on Apple Silicon, ensure Command Line Tools are installed:

    xcode-select --install

Notes:
- These instructions focus on macOS. Linux/Windows setup has been omitted intentionally to keep this README mac-specific.

# python-options (macOS)

This repo contains a simple options screener (options.py). These instructions are tailored for macOS.

## 1) Install Python 3 (macOS)

You can either use the helper script or install via Homebrew directly.

- Using the helper script (recommended):

  chmod +x install_python.sh
  ./install_python.sh

- Or manually with Homebrew (https://brew.sh):

  brew install python

After installation, verify:

  python3 --version
  pip3 --version

## 2) Create and activate a virtual environment

From the project folder:

  python3 -m venv .venv
  source .venv/bin/activate

## 3) Install project dependencies

With the virtual environment active:

  python3 -m pip install -r requirements.txt

## 4) Run the screener

Example:

  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
- screener_results.csv
- screener_results_backtest.csv
- plots/<TICKER>_support_resistance.png

## Quick start (one command)

If you prefer automation, run:

  bash setup_venv.sh

This will create a virtual environment and install requirements using `python -m pip`.

## Troubleshooting (macOS)

- zsh: command not found: python
  - Use `python3` instead. Example: `python3 -m pip install -r requirements.txt`
  - If `python3` is not found, install via `./install_python.sh` or `brew install python`.
  - Optional (zsh): add an alias so `python` maps to `python3`:
    - echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc

- zsh: command not found: pip
  - Use the module form: `python3 -m pip install -r requirements.txt`
  - Ensure your virtual environment is activated: `source .venv/bin/activate`
  - If you recently installed Homebrew/Python, open a new terminal so PATH updates, or run `hash -r` in zsh.

- Apple Silicon build tools
  - If you encounter build errors for scipy or numpy on Apple Silicon, ensure Command Line Tools are installed:

    xcode-select --install

Notes:
- These instructions focus on macOS. Linux/Windows setup has been omitted intentionally to keep this README mac-specific.

## Run command shortcuts

After setting up the environment (see steps above), you can run the screener with a simple command:

- Using the helper script:

  chmod +x run.sh
  ./run.sh   # no parameters needed; uses tickers.csv if present or built-in defaults

- Or using Make (pass extra args via ARGS):

  make run   # runs with defaults
  make run ARGS="--tickers_csv tickers.csv --min_oi 200 --min_vol 50"

### Backtest-only shortcut (quick way to see profitability)

If you only want to run the multi-year strategy backtest (skip the options screener) and print the total profitability and the average per-ticker profitability in the console, use:

- Using the helper script:

  chmod +x backtest.sh
  ./backtest.sh            # uses tickers.csv if present, otherwise built-in defaults
  ./backtest.sh --tickers AAPL,MSFT --bt_years 3 --bt_dte 7

- Or using Make:

  make backtest
  make backtest ARGS="--tickers AAPL,MSFT --bt_years 3"

What you’ll see in the console at the end (now with world‑class formatting):
- A colored banner and a table of Top Option Candidates (if any)
- A rich, colored Strategy Backtest Summary table (per ticker)
- "Combined total profitability of all strategy trades: ...% (equal stake per trade)" in green/red depending on performance
- "Average per-ticker total trade profitability: ...%" in green/red

Notes:
- Pretty console output uses the Rich library and our scripts now force colors even when running under `make` or other non‑TTY contexts.
- To disable colors, set NO_COLOR=1 (e.g., `NO_COLOR=1 make backtest`).

Additional outputs remain the same:
- backtests/<TICKER>_equity.csv
- screener_results_backtest.csv (contains per-ticker strategy metrics)


## Using a CSV for tickers

You can now provide tickers via a CSV file. By default, the script looks for a file named `tickers.csv` in the project root.

- CSV format: one ticker per line is recommended. A header like `ticker` is supported. Comma/space/semicolon-separated values are also accepted.

Examples:

  # default (uses tickers.csv if it exists; otherwise built-in defaults)
  ./run.sh

  # explicit CSV path and optional thresholds
  ./run.sh --tickers_csv tickers.csv --min_oi 200 --min_vol 50

Backward compatibility: you can still pass a comma-separated list via `--tickers`, which is used if a CSV isn’t provided/found.



## Improved backtesting (multi-year strategy simulation)

The screener now includes a more realistic multi-year backtest that simulates buying short-dated calls on breakout signals.
It prices options via Black–Scholes using historical realized volatility and supports take-profit/stop-loss exits. Outputs include
per-ticker equity curves under backtests/ and an aggregate summary in screener_results_backtest.csv.

Key CLI flags:
- --bt_years N          # years of underlying history to load for backtesting (default: 3)
- --bt_dte D            # option DTE in days for simulated trades (default: 7)
- --bt_moneyness PCT    # OTM percent for strike; K = S * (1 + PCT), e.g., 0.05 = 5% OTM (default: 0.05)
- --bt_tp_x X           # optional take-profit multiple of entry premium, e.g., 3.0 for +200%
- --bt_sl_x X           # optional stop-loss multiple of entry premium, e.g., 0.5 for -50%
- --bt_alloc_frac F     # fraction of equity per trade, 0..1 (default: 0.1)
- --bt_trend_filter T   # true/false; require uptrend (default: true)
- --bt_vol_filter V     # true/false; require rv5<rv21<rv63 (default: true)
- --bt_time_stop_frac Q # fraction of DTE for time-based check (default: 0.5)
- --bt_time_stop_mult M # min multiple at time stop to remain (default: 1.2)
- --bt_use_target_delta B # true/false; use target delta for strike (default: false)
- --bt_target_delta D   # target delta value when enabled (default: 0.25)
- --bt_trail_start_mult X # start trailing when option >= X * entry (default: 1.5)
- --bt_trail_back B     # trailing drawback from peak fraction (default: 0.5)
- --bt_protect_mult P   # protective floor vs entry (default: 0.7)
- --bt_cooldown_days N  # cooldown days after losing trade (default: 0)
- --bt_entry_weekdays W # comma-separated weekdays 0=Mon..6=Sun to allow entries (e.g., 0,1,2)
- --bt_skip_earnings E  # true/false; skip entries near earnings (auto-fetched from yfinance)
- --bt_use_underlying_atr_exits B # true/false; use ATR-based exits on underlying (default: true)
- --bt_tp_atr_mult X   # underlying ATR take-profit multiple (default: 2.0)
- --bt_sl_atr_mult X   # underlying ATR stop-loss multiple (default: 1.0)

Examples:

  # Run with defaults (uses tickers.csv if present)
  ./run.sh

  # 5-year backtest, 7DTE, 5% OTM, with TP=3x and SL=0.5x
  ./run.sh --bt_years 5 --bt_dte 7 --bt_moneyness 0.05 --bt_tp_x 3 --bt_sl_x 0.5

Outputs (in addition to the existing CSVs and plots):
- backtests/<TICKER>_equity.csv   # per-ticker equity curve over time
- screener_results_backtest.csv   # now includes strategy metrics (CAGR, Sharpe, max drawdown, win rate)


## Diagnose and fix missing Python libraries (Environment Doctor)

If you suspect the wrong or missing libraries, use the built-in doctor:

- Quick check:

      make doctor

- Auto-fix (installs/repairs packages from requirements.txt):

      make doctor ARGS="--fix"

You can also run the script directly:

    chmod +x doctor.sh
    ./doctor.sh           # check only
    ./doctor.sh --fix     # attempt to install requirements automatically

Tips:
- It prefers the project virtualenv at .venv if present; otherwise it uses your system python3.
- If no virtualenv is active, consider creating one first:

      python3 -m venv .venv && source .venv/bin/activate

- On macOS/Apple Silicon, ensure you have Command Line Tools installed if you hit build errors:

      xcode-select --install


## Price data cache

To avoid re-downloading historical prices every run, the script caches daily OHLCV per ticker in CSV files.
- Default cache directory: data/
- Cache file naming: data/<TICKER>_1d.csv
- On each run, the cache is incrementally updated by fetching only missing dates. If any required columns are missing for existing rows, those rows are re-fetched and filled.

Controls:
- --data_dir PATH       Use a custom directory for the cache (defaults to data or env PRICE_DATA_DIR)
- --cache_refresh       Force refresh for the requested window (re-downloads and overwrites cache for that range)
- Environment: set PRICE_DATA_DIR to override the default cache directory

# python-options (macOS)

This repo contains a simple options screener (options.py). These instructions are tailored for macOS.

## 1) Install Python 3 (macOS)

You can either use the helper script or install via Homebrew directly.

- Using the helper script (recommended):

  chmod +x install_python.sh
  ./install_python.sh

- Or manually with Homebrew (https://brew.sh):

  brew install python

After installation, verify:

  python3 --version
  pip3 --version

## 2) Create and activate a virtual environment

From the project folder:

  python3 -m venv .venv
  source .venv/bin/activate

## 3) Install project dependencies

With the virtual environment active:

  python3 -m pip install -r requirements.txt

## 4) Run the screener

Example:

  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
- screener_results.csv
- screener_results_backtest.csv
- plots/<TICKER>_support_resistance.png

## Quick start (one command)

If you prefer automation, run:

  bash setup_venv.sh

This will create a virtual environment and install requirements using `python -m pip`.

## Troubleshooting (macOS)

- zsh: command not found: python
  - Use `python3` instead. Example: `python3 -m pip install -r requirements.txt`
  - If `python3` is not found, install via `./install_python.sh` or `brew install python`.
  - Optional (zsh): add an alias so `python` maps to `python3`:
    - echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc

- zsh: command not found: pip
  - Use the module form: `python3 -m pip install -r requirements.txt`
  - Ensure your virtual environment is activated: `source .venv/bin/activate`
  - If you recently installed Homebrew/Python, open a new terminal so PATH updates, or run `hash -r` in zsh.

- Apple Silicon build tools
  - If you encounter build errors for scipy or numpy on Apple Silicon, ensure Command Line Tools are installed:

    xcode-select --install

Notes:
- These instructions focus on macOS. Linux/Windows setup has been omitted intentionally to keep this README mac-specific.

# python-options (macOS)

This repo contains a simple options screener (options.py). These instructions are tailored for macOS.

## 1) Install Python 3 (macOS)

You can either use the helper script or install via Homebrew directly.

- Using the helper script (recommended):

  chmod +x install_python.sh
  ./install_python.sh

- Or manually with Homebrew (https://brew.sh):

  brew install python

After installation, verify:

  python3 --version
  pip3 --version

## 2) Create and activate a virtual environment

From the project folder:

  python3 -m venv .venv
  source .venv/bin/activate

## 3) Install project dependencies

With the virtual environment active:

  python3 -m pip install -r requirements.txt

## 4) Run the screener

Example:

  python options.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
- screener_results.csv
- screener_results_backtest.csv
- plots/<TICKER>_support_resistance.png

## Quick start (one command)

If you prefer automation, run:

  bash setup_venv.sh

This will create a virtual environment and install requirements using `python -m pip`.

## Troubleshooting (macOS)

- zsh: command not found: python
  - Use `python3` instead. Example: `python3 -m pip install -r requirements.txt`
  - If `python3` is not found, install via `./install_python.sh` or `brew install python`.
  - Optional (zsh): add an alias so `python` maps to `python3`:
    - echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc

- zsh: command not found: pip
  - Use the module form: `python3 -m pip install -r requirements.txt`
  - Ensure your virtual environment is activated: `source .venv/bin/activate`
  - If you recently installed Homebrew/Python, open a new terminal so PATH updates, or run `hash -r` in zsh.

- Apple Silicon build tools
  - If you encounter build errors for scipy or numpy on Apple Silicon, ensure Command Line Tools are installed:

    xcode-select --install

Notes:
- These instructions focus on macOS. Linux/Windows setup has been omitted intentionally to keep this README mac-specific.

## Run command shortcuts

After setting up the environment (see steps above), you can run the screener with a simple command:

- Using the helper script:

  chmod +x run.sh
  ./run.sh   # no parameters needed; uses tickers.csv if present or built-in defaults

- Or using Make (pass extra args via ARGS):

  make run   # runs with defaults
  make run ARGS="--tickers_csv tickers.csv --min_oi 200 --min_vol 50"

### Backtest-only shortcut (quick way to see profitability)

If you only want to run the multi-year strategy backtest (skip the options screener) and print the total profitability and the average per-ticker profitability in the console, use:

- Using the helper script:

  chmod +x backtest.sh
  ./backtest.sh            # uses tickers.csv if present, otherwise built-in defaults
  ./backtest.sh --tickers AAPL,MSFT --bt_years 3 --bt_dte 7

- Or using Make:

  make backtest
  make backtest ARGS="--tickers AAPL,MSFT --bt_years 3"

What you’ll see in the console at the end (now with world‑class formatting):
- A colored banner and a table of Top Option Candidates (if any)
- A rich, colored Strategy Backtest Summary table (per ticker)
- "Combined total profitability of all strategy trades: ...% (equal stake per trade)" in green/red depending on performance
- "Average per-ticker total trade profitability: ...%" in green/red

Notes:
- Pretty console output uses the Rich library and our scripts now force colors even when running under `make` or other non‑TTY contexts.
- To disable colors, set NO_COLOR=1 (e.g., `NO_COLOR=1 make backtest`).

Additional outputs remain the same:
- backtests/<TICKER>_equity.csv
- screener_results_backtest.csv (contains per-ticker strategy metrics)


## Using a CSV for tickers

You can now provide tickers via a CSV file. By default, the script looks for a file named `tickers.csv` in the project root.

- CSV format: one ticker per line is recommended. A header like `ticker` is supported. Comma/space/semicolon-separated values are also accepted.

Examples:

  # default (uses tickers.csv if it exists; otherwise built-in defaults)
  ./run.sh

  # explicit CSV path and optional thresholds
  ./run.sh --tickers_csv tickers.csv --min_oi 200 --min_vol 50

Backward compatibility: you can still pass a comma-separated list via `--tickers`, which is used if a CSV isn’t provided/found.



## Improved backtesting (multi-year strategy simulation)

The screener now includes a more realistic multi-year backtest that simulates buying short-dated calls on breakout signals.
It prices options via Black–Scholes using historical realized volatility and supports take-profit/stop-loss exits. Outputs include
per-ticker equity curves under backtests/ and an aggregate summary in screener_results_backtest.csv.

Key CLI flags:
- --bt_years N          # years of underlying history to load for backtesting (default: 3)
- --bt_dte D            # option DTE in days for simulated trades (default: 7)
- --bt_moneyness PCT    # OTM percent for strike; K = S * (1 + PCT), e.g., 0.05 = 5% OTM (default: 0.05)
- --bt_tp_x X           # optional take-profit multiple of entry premium, e.g., 3.0 for +200%
- --bt_sl_x X           # optional stop-loss multiple of entry premium, e.g., 0.5 for -50%
- --bt_alloc_frac F     # fraction of equity per trade, 0..1 (default: 0.1)
- --bt_trend_filter T   # true/false; require uptrend (default: true)
- --bt_vol_filter V     # true/false; require rv5<rv21<rv63 (default: true)
- --bt_time_stop_frac Q # fraction of DTE for time-based check (default: 0.5)
- --bt_time_stop_mult M # min multiple at time stop to remain (default: 1.2)
- --bt_use_target_delta B # true/false; use target delta for strike (default: false)
- --bt_target_delta D   # target delta value when enabled (default: 0.25)
- --bt_trail_start_mult X # start trailing when option >= X * entry (default: 1.5)
- --bt_trail_back B     # trailing drawback from peak fraction (default: 0.5)
- --bt_protect_mult P   # protective floor vs entry (default: 0.7)
- --bt_cooldown_days N  # cooldown days after losing trade (default: 0)
- --bt_entry_weekdays W # comma-separated weekdays 0=Mon..6=Sun to allow entries (e.g., 0,1,2)
- --bt_skip_earnings E  # true/false; skip entries near earnings (auto-fetched from yfinance)
- --bt_use_underlying_atr_exits B # true/false; use ATR-based exits on underlying (default: true)
- --bt_tp_atr_mult X   # underlying ATR take-profit multiple (default: 2.0)
- --bt_sl_atr_mult X   # underlying ATR stop-loss multiple (default: 1.0)

Examples:

  # Run with defaults (uses tickers.csv if present)
  ./run.sh

  # 5-year backtest, 7DTE, 5% OTM, with TP=3x and SL=0.5x
  ./run.sh --bt_years 5 --bt_dte 7 --bt_moneyness 0.05 --bt_tp_x 3 --bt_sl_x 0.5

Outputs (in addition to the existing CSVs and plots):
- backtests/<TICKER>_equity.csv   # per-ticker equity curve over time
- screener_results_backtest.csv   # now includes strategy metrics (CAGR, Sharpe, max drawdown, win rate)


## Price data cache

To avoid re-downloading historical prices every run, the script caches daily OHLCV per ticker in CSV files.
- Default cache directory: data/
- Cache file naming: data/<TICKER>_1d.csv
- On each run, the cache is incrementally updated by fetching only missing dates. If any required columns are missing for existing rows, those rows are re-fetched and filled.

Controls:
- --data_dir PATH       Use a custom directory for the cache (defaults to data or env PRICE_DATA_DIR)
- --cache_refresh       Force refresh of cached price data for requested window (re-downloads and overwrites cache for that range)
- Environment: set PRICE_DATA_DIR to override the default cache directory

### Additional caches (to avoid re-fetching options metadata)

Besides price history, the following are cached automatically under data/ to reduce repeated downloads:
- Expiration dates list (yfinance Ticker.options) → data/meta/<TICKER>_meta.json (TTL ~ 12 hours)
- Earnings dates (get_earnings_dates / calendar fallback) → data/meta/<TICKER>_meta.json (TTL ~ 3 days)
- Option chains (calls) per expiry (Ticker.option_chain) → data/options/<TICKER>/<EXPIRY>_calls.csv (TTL ~ 60 minutes)
- Option chains (puts) per expiry (Ticker.option_chain) → data/options/<TICKER>/<EXPIRY>_puts.csv (TTL ~ 60 minutes)
- Ticker info/fast_info snapshot → data/meta/<TICKER>_meta.json (TTL ~ 1 day)
- Dividends series → data/meta/<TICKER>_dividends.csv (TTL ~ 7 days)
- Splits series → data/meta/<TICKER>_splits.csv (TTL ~ 30 days)

You can override TTLs via environment variables:
- EXPIRATIONS_TTL_HOURS (default 12)
- EARNINGS_TTL_DAYS (default 3)
- OPTION_CHAIN_TTL_MIN (default 60)
- INFO_TTL_DAYS (default 1)
- DIVIDENDS_TTL_DAYS (default 7)
- SPLITS_TTL_DAYS (default 30)

Notes:
- On cache read errors or stale TTL, the script transparently re-fetches and overwrites the cache.
- To force a clean refresh, delete specific cache files in data/meta or data/options for the affected ticker/expiry.



## Revenue Growth Screener (Top 50)

This project includes a script to rank small/mid caps by 3-year revenue CAGR using yfinance.

How to run (uses the project virtualenv via Make):

  make top50

Defaults:
- Universe source: CSV
- CSV path: data/universes/russell2500_tickers.csv
- Output: top50_small_mid_revenue_cagr.csv

Providing the universe:
- Place a file at data/universes/russell2500_tickers.csv with a header `ticker` and one symbol per row.
- Example:

  ticker
  AAPL
  MSFT

Overriding defaults:

  make top50 ARGS="--csv path/to/your_list.csv --min_mkt_cap 1e8 --max_mkt_cap 2e10 --top_n 100"

Notes:
- The script attempts multiple sources in yfinance to construct at least 4 annual revenue values. Some tickers may be skipped if data is missing.
- Market cap filter is read from yfinance fast_info/info and used to restrict to small/mid caps.
