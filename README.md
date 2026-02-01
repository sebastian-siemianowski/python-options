<h1 align="center">Quantitative Signal Engine</h1>

<p align="center">
  <strong>Where Bayesian Model Averaging meets Kalman Filtering</strong><br>
  <sub>Multi-asset signal generation with calibrated uncertainty</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/platform-macOS-lightgrey.svg" alt="macOS">
  <img src="https://img.shields.io/badge/assets-100+-green.svg" alt="100+ Assets">
  <img src="https://img.shields.io/badge/models-7_per_regime-orange.svg" alt="7 Models">
</p>

<p align="center">
  <a href="#the-system">The System</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#daily-workflow">Daily Workflow</a> •
  <a href="#command-reference">Commands</a> •
  <a href="#the-mathematics">Mathematics</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## Why This System Exists

Most trading systems choose a single model and pretend it's correct. This system doesn't.

Instead, it maintains **7 competing models** across **5 market regimes**, letting Bayesian inference continuously update which models are most credible given recent data. Signals emerge from the **full posterior predictive distribution**—not from any single "best guess."

The result: **calibrated uncertainty**. When the system says "62% probability of positive return," it means that historically, 62% of such predictions were correct.

> *"The goal is not to be right. The goal is to know how confident you should be."*

---

## The System

This is a **belief evolution engine**, not a rule engine.

At its core, the system maintains a population of competing models—each representing a different hypothesis about market dynamics. These models evolve in probability over time through Bayesian updating, and signals emerge from the full predictive distribution, not from point estimates.

### The Pipeline

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║   ┌─────────────┐                                                                     ║
║   │ Yahoo       │                                                                     ║
║   │ Finance API │                                                                     ║
║   └──────┬──────┘                                                                     ║
║          │                                                                            ║
║          ▼                                                                            ║
║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
║   │                         DATA ENGINE  (make data)                            │     ║
║   │                                                                             │     ║
║   │   • Fetch 10 years OHLCV for 100+ symbols                                   │     ║
║   │   • Multi-pass retry (Yahoo is flaky)                                       │     ║
║   │   • Incremental cache updates                                               │     ║
║   │   • Currency conversion to PLN base                                         │     ║
║   │                                                                             │     ║
║   │   Output: src/data/options/stock_prices/{SYMBOL}_1d.csv                                              │     ║
║   └──────────────────────────────────┬──────────────────────────────────────────┘     ║
║                                      │                                                ║
║                                      ▼                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
║   │                        TUNING ENGINE  (make tune)                           │     ║
║   │                                                                             │     ║
║   │   For each asset:                                                           │     ║
║   │   ┌─────────────────────────────────────────────────────────────────────┐   │     ║
║   │   │  For each regime r ∈ {LOW_VOL_TREND, HIGH_VOL_TREND,                │   │     ║
║   │   │                       LOW_VOL_RANGE, HIGH_VOL_RANGE, CRISIS_JUMP}:  │   │     ║
║   │   │                                                                     │   │     ║
║   │   │    For each model m ∈ {kalman_gaussian,                             │   │     ║
║   │   │                        kalman_phi_gaussian,                         │   │     ║
║   │   │                        phi_student_t_nu_4,  phi_student_t_nu_6,     │   │     ║
║   │   │                        phi_student_t_nu_8,  phi_student_t_nu_12,    │   │     ║
║   │   │                        phi_student_t_nu_20}:                        │   │     ║
║   │   │                                                                     │   │     ║
║   │   │      1. Fit θ = {q, c, φ} via MLE with regularization prior         │   │     ║
║   │   │      2. Compute log-likelihood ℓ(θ)                                 │   │     ║
║   │   │      3. Compute BIC = -2ℓ + k·log(n)                                │   │     ║
║   │   │      4. Compute Hyvärinen score (robust to misspecification)        │   │     ║
║   │   │      5. Run PIT calibration diagnostics                             │   │     ║
║   │   │                                                                     │   │     ║
║   │   │    Aggregate across models:                                         │   │     ║
║   │   │      • w(m|r) = exp(-½ · ΔBIC) · hyv_weight^(1-α)                   │   │     ║
║   │   │      • Apply temporal smoothing: w ← w_prev^α · w_raw               │   │     ║
║   │   │      • Apply hierarchical shrinkage toward global                   │   │     ║
║   │   │      • Normalize: p(m|r) = w(m|r) / Σw                              │   │     ║
║   │   └─────────────────────────────────────────────────────────────────────┘   │     ║
║   │                                                                             │     ║
║   │   Output: src/data/kalman_q_cache.json                           │     ║
║   │           {asset: {regime: {model: {q, φ, ν, BIC, p(m|r), ...}}}}           │     ║
║   └──────────────────────────────────┬──────────────────────────────────────────┘     ║
║                                      │                                                ║
║                                      ▼                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────────┐     ║
║   │                       SIGNAL ENGINE  (make stocks)                          │     ║
║   │                                                                             │     ║
║   │   For each asset:                                                           │     ║
║   │   ┌─────────────────────────────────────────────────────────────────────┐   │     ║
║   │   │  1. REGIME DETECTION                                                │   │     ║
║   │   │     • Compute rolling volatility (EWMA fast/slow blend)             │   │     ║
║   │   │     • Compute drift magnitude                                       │   │     ║
║   │   │     • Classify: r_t ∈ {0,1,2,3,4}                                   │   │     ║
║   │   │                                                                     │   │     ║
║   │   │  2. LOAD BELIEFS                                                    │   │     ║
║   │   │     • Retrieve p(m|r_t) and θ_{r_t,m} from cache                    │   │     ║
║   │   │     • If regime sparse → borrow from global (hierarchical)          │   │     ║
║   │   │                                                                     │   │     ║
║   │   │  3. POSTERIOR PREDICTIVE MONTE CARLO                                │   │     ║
║   │   │     samples = []                                                    │   │     ║
║   │   │     for m, weight in p(m|r_t):                                      │   │     ║
║   │   │         n_samples = weight × N_total                                │   │     ║
║   │   │         for each sample:                                            │   │     ║
║   │   │             μ = kalman_drift_estimate                               │   │     ║
║   │   │             for t in 1..horizon:                                    │   │     ║
║   │   │                 μ ← φ·μ + η,  η ~ N(0, q)                           │   │     ║
║   │   │                 r_t ← μ + ε,  ε ~ model_distribution(σ)             │   │     ║
║   │   │             samples.append(Σ r_t)                                   │   │     ║
║   │   │                                                                     │   │     ║
║   │   │  4. DECISION LAYER                                                  │   │     ║
║   │   │     • P(return > 0) = count(samples > 0) / N                        │   │     ║
║   │   │     • E[return] = mean(samples)                                     │   │     ║
║   │   │     • Apply exhaustion dampening (UE↑/UE↓)                          │   │     ║
║   │   │     • Map: P > 58% → BUY, P < 42% → SELL, else → HOLD               │   │     ║
║   │   └─────────────────────────────────────────────────────────────────────┘   │     ║
║   │                                                                             │     ║
║   │   Output: Console tables + cached JSON                                      │     ║
║   └──────────────────────────────────┬──────────────────────────────────────────┘     ║
║                                      │                                                ║
║                                      ▼                                                ║
║                        ┌───────────────────────────┐                                  ║
║                        │   BUY  │  HOLD  │  SELL   │                                  ║
║                        │   🟢   │   ⚪   │   🔴   │                                   ║
║                        └───────────────────────────┘                                  ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

### Quick Reference

| Engine | Command | Input | Output | Time |
|--------|---------|-------|--------|------|
| **Data** | `make data` | Yahoo Finance API | `data/*.csv` | 5-15 min |
| **Tuning** | `make tune` | Price CSVs | `kalman_q_cache.json` | 2-10 min |
| **Signal** | `make stocks` | Cache + fresh prices | Console + JSON | 1-3 min |

### Asset Universe

The system tracks **100+ assets** across multiple asset classes:

| Class | Examples | Count |
|-------|----------|-------|
| **Equities** | AAPL, MSFT, NVDA, TSLA, JPM, GS, UNH, LLY... | ~80 |
| **Defense** | LMT, RTX, NOC, GD, BA, HII, AVAV, PLTR... | ~40 |
| **ETFs** | SPY, VOO, GLD, SLV, SMH | 5 |
| **Commodities** | GC=F (Gold), SI=F (Silver) | 2 |
| **Crypto** | BTC-USD, MSTR | 2 |
| **FX** | PLNJPY=X | 1 |

All prices are converted to a common base currency (PLN) for portfolio-level analysis.

### Model Universe

The Tuning Engine fits **7 model classes** per regime:

| Model | Parameters | Use Case |
|-------|------------|----------|
| `kalman_gaussian` | q, c | Baseline Gaussian innovations |
| `kalman_phi_gaussian` | q, c, φ | AR(1) drift with Gaussian |
| `phi_student_t_nu_4` | q, c, φ | Heavy tails (ν=4) |
| `phi_student_t_nu_6` | q, c, φ | Moderate tails (ν=6) |
| `phi_student_t_nu_8` | q, c, φ | Light tails (ν=8) |
| `phi_student_t_nu_12` | q, c, φ | Near-Gaussian (ν=12) |
| `phi_student_t_nu_20` | q, c, φ | Almost Gaussian (ν=20) |

Student-t models use a **discrete ν grid** (not continuous optimization). Each ν is a separate sub-model in BMA, allowing the posterior to express uncertainty about tail thickness.

### Regime Classification

Markets are classified into **5 regimes** based on volatility and drift:

| Regime | Condition |
|--------|-----------|
| `LOW_VOL_TREND` | vol < 0.85×median, \|drift\| > threshold |
| `HIGH_VOL_TREND` | vol > 1.3×median, \|drift\| > threshold |
| `LOW_VOL_RANGE` | vol < 0.85×median, \|drift\| ≤ threshold |
| `HIGH_VOL_RANGE` | vol > 1.3×median, \|drift\| ≤ threshold |
| `CRISIS_JUMP` | vol > 2×median OR tail_indicator > 4 |

Regime assignment is **deterministic and consistent** between tuning and inference.

---

## Quick Start

### Prerequisites

- macOS (Intel or Apple Silicon)
- Python 3.7+
- ~10GB disk space for price cache

### Installation (One Command)

```bash
make setup
```

This will:
1. Create `.venv/` virtual environment
2. Install dependencies from `src/setup/requirements.txt`
3. Download 10 years of price data (3 passes for reliability)
4. Clean cached data

**Time:** 5-15 minutes depending on network.

### Generate Your First Signals

```bash
make stocks
```

### What You'll See

The system outputs beautifully formatted Rich console tables with Apple-quality UX:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    ▲ NVDA  NVIDIA Corporation                                 ┃
┃                    142.58  │  LOW_VOL_TREND  │  2025-01-27  │  Student-t      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  Horizon   │  P(r>0) │ E[return] │      CI 68%       │   Profit │   Signal   │  Strength   
 ───────────┼─────────┼───────────┼───────────────────┼──────────┼────────────┼─────────────
  1 day     │   54.2% │    +0.08% │ [ -0.6%,  +0.7%]  │      +5k │   — HOLD   │ ──────────  
  1 week    │   58.7% │    +0.42% │ [ -1.3%,  +2.3%]  │     +49k │   ↑ BUY    │ ██░░░░░░░░  
  1 month   │   63.1% │    +1.84% │ [ -2.1%,  +5.3%]  │    +170k │   ↑ BUY    │ ████░░░░░░  
  3 months  │   71.2% │    +5.62% │ [ -7.9%, +17.5%]  │    +618k │   ↑ BUY    │ ███████░░░  
  6 months  │   68.4% │    +9.93% │ [-24.4%, +44.3%]  │    +1.7M │   ↑ BUY    │ ██████░░░░  
  12 months │   72.8% │   +19.80% │ [-75.5%,+115.1%]  │    +6.2M │  ▲▲ BUY    │ ████████░░  

         P(r>0) prob of positive return    E[return] expected return    Profit on 1M PLN notional

                    ↑ BUY  P ≥ 58%         — HOLD  42% < P < 58%         ↓ SELL  P ≤ 42%
```

**Design Features:**
- **Perfect alignment** — Rich Table ensures columns line up precisely
- **Alternating row colors** — Subtle grey bands improve scannability
- **Strength bars** — Visual confidence indicator (█░ for signal, ─ for neutral)
- **Color hierarchy** — Bright green (strong buy), green (buy), red (sell), dim (hold)
- **Signal badges** — ▲▲ BUY, ↑ BUY, — HOLD, ↓ SELL, ▼▼ SELL
- **Header panel** — Bordered card with asset identity and regime

Signals are color-coded:
- 🟢 **BUY** (green): P(r>0) ≥ 58%
- ⚪ **HOLD** (dim): P(r>0) ∈ (42%, 58%)
- 🔴 **SELL** (red): P(r>0) ≤ 42%

### Understanding the Columns

| Column | Meaning |
|--------|---------|
| **Horizon** | Forecast period (trading days) |
| **P(r>0)** | Probability that return will be positive |
| **E[return]** | Expected log return from posterior mean |
| **Signal** | Decision derived from probability threshold |
| **Confidence** | Visual indicator of probability magnitude |

### Understanding the Regime

Each asset is classified into one of 5 regimes:

| Regime | What It Means | Typical Behavior |
|--------|---------------|------------------|
| `LOW_VOL_TREND` | Quiet trending market | Smooth, directional moves |
| `HIGH_VOL_TREND` | Volatile trending market | Sharp moves with direction |
| `LOW_VOL_RANGE` | Quiet range-bound | Mean-reverting, choppy |
| `HIGH_VOL_RANGE` | Volatile range-bound | Whipsaw, no clear direction |
| `CRISIS_JUMP` | Extreme stress | Tail events, correlations spike |

The regime affects which model receives the most weight in the BMA mixture.

---

## Daily Workflow

### The 30-Second Morning Routine

```bash
make stocks
```

That's it. This single command:
1. Refreshes the last 5 days of price data
2. Loads cached Kalman parameters
3. Generates signals for all assets
4. Displays formatted output

### When to Re-Tune

The Tuning Engine should be run:
- **Weekly** during normal markets
- **After major regime shifts** (VIX spike, Fed announcement)
- **When signals feel stale** or miscalibrated

```bash
# Weekly calibration
make tune

# Force complete re-estimation (ignore cache)
make tune ARGS="--force"
```

### Offline Mode

Already have cached data? Work without network:

```bash
# Render from cache only
make report

# Or set environment variable
OFFLINE_MODE=1 make stocks
```

### Quick Validation

Before trusting signals, validate calibration:

```bash
# Check if probabilities match historical outcomes
make fx-calibration

# Quick smoke test with 20 assets
make top20
```

---

## Command Reference

All interaction happens through `make`. The Makefile orchestrates Python scripts, manages the virtual environment, and handles caching transparently.

---

### 🚀 Setup & Installation

#### `make setup`

**The one command to rule them all.** Run this once after cloning.

```bash
make setup
```

**What happens internally:**
1. Creates `.venv/` via `src/setup/setup_venv.sh`
2. Upgrades pip and installs `src/setup/requirements.txt`
3. Runs `ingestion/precache_data.py` **3 times** (Yahoo Finance is flaky)
4. Runs `ingestion/clean_cache.py` to remove empty rows

**Time:** 5-15 minutes  
**Disk:** ~10GB for full price cache

#### `make doctor`

Reinstalls all dependencies. Use when imports fail or packages are corrupted.

```bash
make doctor
```

---

### 📊 Data Engine

The Data Engine fetches OHLCV (Open, High, Low, Close, Volume) from Yahoo Finance and caches locally as CSV files.

#### `make data`

Downloads full price history for all assets in the universe.

```bash
make data                              # Standard run
make data ARGS="--workers 4"           # Reduce parallelism
make data ARGS="--batch-size 8"        # Smaller batches
```

**Internals:**
- Runs `ingestion/refresh_data.py --skip-trim --retries 5 --workers 12 --batch-size 16`
- 5 retry passes (Yahoo Finance rate-limits aggressively)
- 12 parallel workers, 16 assets per batch
- Output: `src/data/options/stock_prices/<SYMBOL>_1d.csv`

#### `make refresh`

Updates only the last 5 days. Fast daily refresh.

```bash
make refresh
```

**Internals:**
- Deletes last 5 rows from each cache file
- Re-downloads with 5 retry passes
- Typical time: 2-5 minutes

#### `make clean-cache`

Removes rows with all-NaN values (dates before asset existed).

```bash
make clean-cache
```

#### `make failed`

Lists assets that failed to download (stored in `src/fx_failures.json`).

```bash
make failed
```

#### `make purge`

Deletes cache files for failed assets so they can be re-downloaded.

```bash
make purge                    # Clear cache for failed assets
make purge ARGS="--all"       # Also clear the failures list
```

---

### 🔧 Tuning Engine

The Tuning Engine estimates Kalman filter parameters via Maximum Likelihood Estimation, then applies Bayesian Model Averaging across model classes.

#### `make tune`

The heart of the calibration system.

```bash
make tune                              # Uses cache, skips already-tuned assets
make tune ARGS="--force"               # Re-estimate everything
make tune ARGS="--max-assets 10"       # Test with subset
make tune ARGS="--dry-run"             # Preview without executing
```

**What happens internally:**
1. Loads asset universe from `ingestion/data_utils.py`
2. For each asset, for each of 5 regimes:
   - Fits 7 model classes (Gaussian, AR(1)-Gaussian, Student-t with ν ∈ {4,6,8,12,20})
   - Computes BIC, AIC, Hyvärinen score, PIT diagnostics
   - Converts scores to posterior weights
   - Applies temporal smoothing against previous run
   - Applies hierarchical shrinkage toward global
3. Saves to `src/data/kalman_q_cache.json`

**Key ARGS:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--force` | Ignore cache, re-estimate all | False |
| `--max-assets N` | Process only first N assets | All |
| `--dry-run` | Preview without processing | False |
| `--prior-mean X` | Prior mean for log₁₀(q) | -6.0 |
| `--prior-lambda X` | Regularization strength | 1.0 |
| `--lambda-regime X` | Hierarchical shrinkage | 0.05 |
| `--debug` | Show stack traces on errors | False |

#### `make show-q`

Displays the cached Kalman parameters as raw JSON.

```bash
make show-q
```

#### `make clear-q`

Deletes the parameter cache. Next `make tune` will re-estimate everything.

```bash
make clear-q
```

---

### 📈 Signal Engine

The Signal Engine consumes tuned parameters and generates Buy/Hold/Sell signals via posterior predictive Monte Carlo.

#### `make stocks`

**The main command for daily use.** Refreshes data, then generates signals.

```bash
make stocks                            # Full pipeline
make stocks ARGS="--assets AAPL,MSFT"  # Specific assets only
```

**What happens internally:**
1. Runs `ingestion/refresh_data.py` (updates last 5 days)
2. Runs `signals.py` with caching enabled
3. For each asset:
   - Determines current regime r_t
   - Loads model posterior p(m|r_t) from cache
   - Runs posterior predictive Monte Carlo
   - Computes expected utility across horizons
   - Maps to BUY/HOLD/SELL

**Output:** Beautiful Rich console tables showing:
- Signal per horizon (1d, 3d, 7d, 21d, 63d, 126d, 252d)
- Probability of positive return
- Expected log return
- Confidence indicators

#### `make report`

Renders signals from cache without network calls. Use when offline.

```bash
make report
```

#### `make top20`

Quick smoke test with first 20 assets. Good for testing changes.

```bash
make top20
```

---

### 🔬 Diagnostic Commands

These commands validate model quality and calibration.

#### `make fx-diagnostics`

Full diagnostic suite. **Expensive** (runs out-of-sample tests).

```bash
make fx-diagnostics
```

**Includes:**
- Log-likelihood analysis
- Parameter stability across time windows
- Out-of-sample predictive tests

#### `make fx-diagnostics-lite`

Lightweight diagnostics. Skips OOS tests.

```bash
make fx-diagnostics-lite
```

#### `make fx-calibration`

Probability Integral Transform (PIT) calibration check.

```bash
make fx-calibration
```

**What it tests:** If your 60% confidence intervals contain outcomes 60% of the time.

#### `make fx-model-comparison`

Compares model classes via AIC/BIC. Shows which models the posterior favors.

```bash
make fx-model-comparison
```

#### `make fx-validate-kalman`

Level-7 Kalman validation science:
- Drift estimation accuracy
- Likelihood surface analysis
- PIT histogram uniformity
- Stress regime behavior

```bash
make fx-validate-kalman                        # Console output only
make fx-validate-kalman-plots                  # Also saves plots to src/data/plots/kalman_validation/
```

#### `make tests`

Runs the unit test suite.

```bash
make tests
```

---

### 💰 Debt Allocation Engine

A specialized decision engine for balance-sheet currency risk.

#### `make debt`

Determines the optimal day to switch JPY-denominated debt to EUR-denominated debt.

```bash
make debt
```

**This is NOT a trade signal.** It's a corporate treasury tool for:
- Balance-sheet convexity control
- Latent state inference (NORMAL → COMPRESSED → PRE_POLICY → POLICY)
- Auditable, causal decision logic

Output: `src/data/debt/`

---

### 📋 Options Screener & Backtesting

Legacy modules for equity options analysis.

#### `make run`

Runs the options screener with support/resistance analysis.

```bash
make run                                       # Uses tickers.csv
make run ARGS="--tickers AAPL,MSFT,NVDA"       # Explicit tickers
make run ARGS="--min_oi 200 --min_vol 50"      # Filter thresholds
```

**Output:**
- `screener_results.csv`
- `src/data/plots/<TICKER>_support_resistance.png`

#### `make backtest`

Multi-year strategy simulation with Black-Scholes pricing.

```bash
make backtest                                  # Uses tickers.csv
make backtest ARGS="--tickers AAPL --bt_years 5"
```

**Key ARGS:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--bt_years` | Years of history | 3 |
| `--bt_dte` | Days to expiration | 7 |
| `--bt_moneyness` | OTM percent | 0.05 |
| `--bt_tp_x` | Take-profit multiple | None |
| `--bt_sl_x` | Stop-loss multiple | None |
| `--bt_alloc_frac` | Equity fraction per trade | 0.1 |

**Output:**
- `src/backtesting/equity_curves/<TICKER>_equity.csv`
- `screener_results_backtest.csv`

---

### 📊 Fundamental Screeners

#### `make top50`

Ranks small/mid caps by 3-year revenue CAGR.

```bash
make top50
make top50 ARGS="--csv path/to/universe.csv"
```

#### `make bagger50`

Ranks by 100× Bagger Score (probability-weighted growth potential).

```bash
make bagger50
make bagger50 ARGS="--bagger_horizon 15"       # 15-year horizon
make bagger50 ARGS="--bagger_verbose"          # Show sub-scores
```

#### `make top100`

Top 100 screener using Russell 5000 universe.

```bash
make top100
```

#### `make build-russell`

Builds `src/data/russell/russell2500_tickers.csv` from public sources.

```bash
make build-russell
```

#### `make russell5000`

Builds the larger Russell 5000 universe.

```bash
make russell5000
```

---

### 🧹 Utility Commands

#### `make clear`

Nuclear option. Clears all caches, plots, and temp files.

```bash
make clear
```

**Deletes:**
- `__pycache__/`
- `src/data/plots/*.png`
- `src/data/options/meta/`
- `data/*.backup`

#### `make colors`

Displays color palette test. Useful for terminal configuration.

```bash
make colors
```

---

## Architecture

```
python-options/
│
├── Makefile                    # Command interface (start here)
│
├── src/
│   ├── tune.py           # TUNING ENGINE: MLE + BMA
│   ├── tune_ux.py          # Tuning UX wrapper
│   ├── signals.py   # SIGNAL ENGINE: Posterior predictive
│   ├── signals_ux.py  # Rich console output
│   ├── ingestion/refresh_data.py         # DATA ENGINE: Bulk download
│   ├── ingestion/data_utils.py        # Data utilities + caching
│   ├── debt_allocator.py       # Debt switch decision engine
│   └── quant/
│       └── cache/
│           ├── tune/           # Tuned parameters (per-asset)
│           │   ├── AAPL.json
│           │   ├── MSFT.json
│           │   └── ...
│           └── calibration/    # Calibration diagnostics
│
├── data/                       # Price cache (CSV per symbol)
├── options.py                  # Options screener
├── src/backtesting/            # Backtesting module
└── src/data/plots/                      # Generated charts
```

### Design Principles

1. **Separation of concerns**
   - Tuning engine knows nothing about decisions
   - Signal engine acts on beliefs, doesn't create them
   - Presentation layer is fully decoupled

2. **Bayesian integrity**
   - When evidence is weak, the system becomes more ignorant, not more confident
   - Fallback is hierarchical: `p(m|r, weak data) → p(m|global)`
   - Never synthesize beliefs that weren't learned

3. **Auditability**
   - All parameters cached and versioned
   - No hidden state mutations
   - Deterministic regime assignment

---

## Risk Temperature Governance (February 2026)

The system implements institutional-grade risk temperature governance based on the Chinese Quantitative Systems Professor's control theory approach. This layer modulates position sizing based on cross-asset stress indicators without touching distributional beliefs.

### Governance Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Hysteresis Bands** | Asymmetric thresholds for regime transitions | Prevents oscillation at regime boundaries |
| **Conservative Imputation** | Missing data at 75th percentile | Defensive degradation when data quality deteriorates |
| **Rate Limiting** | Maximum temperature change of 0.3/day | Prevents whipsawing from single-day movements |
| **Dynamic Gap Risk** | 95th percentile of trailing 60-day gaps | Adaptive overnight budget constraint |
| **Complete Audit Trail** | Full attribution for reconstruction | Regulatory compliance and post-incident analysis |

### Regime State Machine

```
States: Calm → Elevated → Stressed → Extreme

Upward Thresholds (always allowed):
    Calm → Elevated:     temp > 0.5
    Elevated → Stressed: temp > 1.0
    Stressed → Extreme:  temp > 1.5

Downward Thresholds (with hysteresis gap):
    Extreme → Stressed:  temp < 1.2   (gap of 0.3)
    Stressed → Elevated: temp < 0.7   (gap of 0.3)
    Elevated → Calm:     temp < 0.3   (gap of 0.2)
```

### Scale Factors by Regime

| Regime | Scale Factor | Position Effect |
|--------|-------------|-----------------|
| Calm | 100% | Full allocation |
| Elevated | 75% | Reduced exposure |
| Stressed | 45% | Significantly reduced |
| Extreme | 20% | Minimal / defensive |

### Usage

```python
from decision.metals_risk_temperature import compute_governed_metals_risk_temperature

# Compute with full governance
result = compute_governed_metals_risk_temperature(start_date="2020-01-01")

# Access governance information
print(f"Temperature: {result.temperature:.2f}")
print(f"Regime State: {result.regime_state}")
print(f"Scale Factor: {result.scale_factor:.2%}")

# Get complete audit trail
audit_json = result.get_audit_json()
human_readable = result.render_audit_trail()
```

### Key Files

- `src/decision/regime_governance.py` — Core governance module
- `src/decision/metals_risk_temperature.py` — Governed metals temperature
- `src/decision/risk_temperature.py` — Main risk temperature with governance integration
- `src/tests/test_regime_governance.py` — Comprehensive test suite

---

## The Mathematics

> *"The math always emerges from the underlying system—not the other way around."*

This section documents the mathematical foundations that govern each engine. The code implements these equations; understanding them illuminates why the system behaves as it does.

### Master Symbol Glossary

Before diving in, here's a complete reference of all mathematical symbols used:

#### Prices & Returns

| Symbol | Name | Meaning |
|--------|------|---------|
| Pₜ | Price at time t | The asset price at time step t |
| rₜ | Return at time t | Log return: ln(Pₜ/Pₜ₋₁) |
| h | Horizon | Forecast period in trading days |

#### Volatility

| Symbol | Name | Meaning |
|--------|------|---------|
| σ | Sigma | Standard deviation (volatility) |
| σₜ² | Sigma squared | Variance at time t |
| λ | Lambda | Decay factor in EWMA (0.94-0.97) |

#### Kalman Filter

| Symbol | Name | Meaning |
|--------|------|---------|
| μₜ | Mu | Latent (hidden) drift at time t |
| q | Process noise | How much drift can change per step |
| ηₜ | Eta | Random shock to drift ~ N(0, q) |
| εₜ | Epsilon | Observation noise ~ N(0, σ²) |
| K | Kalman gain | Weight given to new observation (0-1) |
| P | State variance | Uncertainty in drift estimate |
| m | Posterior mean | Best estimate of drift after update |

#### AR(1) Model

| Symbol | Name | Meaning |
|--------|------|---------|
| φ | Phi | Mean-reversion coefficient (-1 to 1) |
| τ | Tau | Prior standard deviation for φ |

#### Student-t Distribution

| Symbol | Name | Meaning |
|--------|------|---------|
| ν | Nu | Degrees of freedom (tail thickness) |
| t_ν | Student-t | t-distribution with ν degrees of freedom |

#### Bayesian Inference

| Symbol | Name | Meaning |
|--------|------|---------|
| p(·) | Probability | Probability or density function |
| p(m\|r) | Model posterior | Probability of model m given regime r |
| θ | Theta | Model parameters (q, φ, etc.) |
| ℓ | Log-likelihood | Sum of log probabilities |

#### Model Selection

| Symbol | Name | Meaning |
|--------|------|---------|
| BIC | Bayesian Info Criterion | Penalized likelihood for model comparison |
| k | Parameter count | Number of free parameters in model |
| n | Sample size | Number of observations |
| w | Weight | Unnormalized model weight |
| α | Alpha | Smoothing/blending coefficient |

#### Decision Theory

| Symbol | Name | Meaning |
|--------|------|---------|
| E[·] | Expectation | Average value |
| P(·) | Probability | Likelihood of event |
| EU | Expected Utility | Risk-adjusted expected value |
| f* | Optimal fraction | Kelly criterion bet size |
| z | Z-score | Standardized edge metric |

---

### Data Engine: Returns and Volatility

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| rₜ | "r sub t" | The return at time t |
| Pₜ | "P sub t" | The price at time t |
| Pₜ₋₁ | "P sub t minus 1" | The price at the previous time step |
| log | Natural logarithm | ln(x), the inverse of eˣ |
| σₜ² | "sigma squared sub t" | Variance (volatility squared) at time t |
| λ | "lambda" | Decay factor controlling how fast old data fades |
| σ | "sigma" | Standard deviation (square root of variance) |

</details>

**Log Returns**

The system works with log returns, not simple returns:

```
rₜ = log(Pₜ / Pₜ₋₁)
```

**In plain English:** *"Today's return equals the natural log of today's price divided by yesterday's price."*

Log returns are additive over time and approximately normal for small values, which simplifies the probabilistic machinery.

**Realized Volatility**

Volatility is estimated via exponentially-weighted moving average (EWMA):

```
σₜ² = λ · σₜ₋₁² + (1 - λ) · rₜ²
```

**In plain English:** *"Today's variance equals lambda times yesterday's variance, plus (1 - lambda) times today's squared return."*

**What this means:**
- When λ = 0.94: Yesterday's variance gets 94% weight, today's return gets 6%
- Higher λ = slower adaptation to new information
- Lower λ = faster adaptation, more reactive

Where λ ∈ (0,1) controls decay. We use multiple speeds:
- **Fast** (λ = 0.94): Responsive to recent moves
- **Slow** (λ = 0.97): Smoother, less reactive

The final volatility blends both for robustness.

**Winsorization**

Extreme returns are clipped to reduce outlier influence:

```
rₜ → clip(rₜ, -3σ, +3σ)
```

**In plain English:** *"If the return is more extreme than 3 standard deviations, cap it at 3 standard deviations."*

This makes parameter estimation more stable without discarding information entirely.

---

### Tuning Engine: Kalman Filter + MLE

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| μₜ | "mu sub t" | Hidden (latent) drift at time t — the "true" trend we're trying to estimate |
| μₜ₋₁ | "mu sub t minus 1" | Hidden drift at previous time step |
| ηₜ | "eta sub t" | Random shock to the drift (process noise) |
| εₜ | "epsilon sub t" | Observation noise (market randomness) |
| q | "q" | Process noise variance — how much drift can change per step |
| σₜ² | "sigma squared" | Observation noise variance (market volatility) |
| N(0, q) | Normal distribution | Gaussian with mean 0 and variance q |
| K | Kalman gain | How much weight to give new observations (0 to 1) |
| P | State variance | Our uncertainty about the drift estimate |
| m | Posterior mean | Our best estimate of drift after seeing data |
| mₜ | "m sub t" | Posterior mean at time t |
| ℓ(q) | Log-likelihood | How well parameters explain the observed data |
| vₜ | Predictive variance | Total uncertainty before seeing observation |
| φ | "phi" | Mean-reversion coefficient in AR(1) model |
| τ | "tau" | Prior standard deviation for φ |
| ν | "nu" | Degrees of freedom in Student-t distribution |
| t_ν | Student-t | Heavy-tailed distribution with ν degrees of freedom |

</details>

**The State-Space Model**

We model latent drift μₜ as a random walk observed through noisy returns:

```
State equation:     μₜ = μₜ₋₁ + ηₜ,     ηₜ ~ N(0, q)
Observation:        rₜ = μₜ + εₜ,       εₜ ~ N(0, σₜ²)
```

**In plain English:**
- *"The true drift today equals yesterday's drift plus a random shock."*
- *"The observed return equals the true drift plus market noise."*

**What this means:**
- We never see μₜ directly — it's hidden (latent)
- We only see rₜ (the actual return)
- The Kalman filter infers μₜ from the noisy observations

Here:
- **μₜ** is the unobserved "true" drift (what we're trying to estimate)
- **q** is the **process noise variance** (how much drift can change per step)
- **σₜ²** is the observation noise (market volatility)

**Kalman Filter Recursion**

Given prior μₜ₋₁|ₜ₋₁ ~ N(m, P), the Kalman filter updates:

```
Predict:    μₜ|ₜ₋₁ ~ N(m, P + q)

Update:     K = (P + q) / (P + q + σₜ²)           # Kalman gain
            mₜ = m + K · (rₜ - m)                  # Posterior mean
            Pₜ = (1 - K) · (P + q)                 # Posterior variance
```

**In plain English:**
1. **Predict:** *"Before seeing today's return, our uncertainty grows by q."*
2. **Kalman gain:** *"K measures how much to trust the new observation vs our prior belief."*
3. **Update mean:** *"New estimate = old estimate + K × (surprise)."*
4. **Update variance:** *"Our uncertainty shrinks after seeing data."*

**Intuition for Kalman gain K:**
- K close to 1: Trust the new observation heavily (high signal-to-noise)
- K close to 0: Stick with prior belief (low signal-to-noise)

The Kalman gain K ∈ (0,1) balances prior belief against new evidence.

**Maximum Likelihood Estimation**

We find q by maximizing the log-likelihood:

```
ℓ(q) = Σₜ log p(rₜ | r₁:ₜ₋₁, q)
     = -½ Σₜ [ log(2π · vₜ) + (rₜ - mₜ)² / vₜ ]
```

**In plain English:** *"Find the value of q that makes the observed returns most probable."*

Where vₜ = P + q + σₜ² is the predictive variance (total uncertainty before observation).

**Regularization Prior**

To prevent overfitting, we add a Gaussian prior on log₁₀(q):

```
log₁₀(q) ~ N(μ_prior, 1/λ)
```

Default: μ_prior = -6 (q ≈ 10⁻⁶), λ = 1.0

**In plain English:** *"We believe q is probably around 0.000001, and penalize values far from this."*

The penalized objective becomes:

```
ℓ_penalized(q) = ℓ(q) - λ/2 · (log₁₀(q) - μ_prior)²
```

**AR(1) Extension (φ-models)**

For mean-reverting drift, we extend the state equation:

```
μₜ = φ · μₜ₋₁ + ηₜ,     φ ∈ (-1, 1)
```

**In plain English:** *"Today's drift equals phi times yesterday's drift, plus noise."*

**What φ values mean:**
- φ = 0: Drift has no memory (fully mean-reverting)
- φ = 0.9: Drift is very persistent (slow mean-reversion)
- φ = 1: Random walk (no mean-reversion) — **unstable, we avoid this**
- φ < 0: Drift oscillates (rare in financial data)

When |φ| < 1, drift reverts toward zero. We apply a shrinkage prior:

```
φ ~ N(0, τ²)
```

This prevents unit-root instability (φ → 1).

**Student-t Innovations**

To capture fat tails, we replace Gaussian innovations with Student-t:

```
εₜ ~ t_ν(0, σₜ)
```

**In plain English:** *"Market noise follows a Student-t distribution instead of Gaussian, allowing for rare extreme moves."*

The degrees-of-freedom ν controls tail thickness:
- ν = 4: Very heavy tails (frequent extreme moves)
- ν = 20: Nearly Gaussian (rare extreme moves)
- ν → ∞: Gaussian limit

We use a discrete grid ν ∈ {4, 6, 8, 12, 20} and let BMA select the mixture.

---

### Tuning Engine: Bayesian Model Averaging

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| p(·) | Probability | Probability or probability density |
| p(m\|r) | "p of m given r" | Probability of model m, given we're in regime r |
| rₜ₊ₕ | "r sub t plus h" | Return h days from now |
| r | Regime | Market state (e.g., LOW_VOL_TREND) |
| m | Model | A specific model class (e.g., kalman_gaussian) |
| θ | "theta" | All parameters of a model (q, φ, etc.) |
| θᵣ,ₘ | "theta r,m" | Parameters of model m in regime r |
| Σₘ | "sum over m" | Add up across all models |
| BIC | Bayesian Info Criterion | Score balancing fit vs complexity |
| ℓ | Log-likelihood | How well model explains data |
| k | Parameter count | Number of free parameters |
| n | Sample size | Number of data points |
| w | Weight | Unnormalized probability |
| exp(·) | Exponential | e raised to the power of (·) |
| α | "alpha" | Blending coefficient (0 to 1) |
| λ | "lambda" | Shrinkage coefficient (0 to 1) |
| H(m) | Hyvärinen score | Robust model comparison metric |
| ∂ | Partial derivative | Rate of change with respect to one variable |

</details>

**The BMA Equation**

Given regime r and model class m with parameters θ, the posterior predictive is:

```
p(rₜ₊ₕ | r) = Σₘ p(rₜ₊ₕ | r, m, θᵣ,ₘ) · p(m | r)
```

**In plain English:** *"The probability of a future return equals the weighted average of each model's prediction, where weights are how much we trust each model."*

**Breaking it down:**
- p(rₜ₊ₕ | r, m, θ) = "What does model m predict for the return?"
- p(m | r) = "How much do we trust model m in this regime?"
- Σₘ = "Add up across all 7 models"

This is the **core equation** of the system. Signals emerge from this mixture, not from any single "best" model.

**Model Weights via BIC**

For each model m in regime r, we compute BIC:

```
BIC_m,r = -2 · ℓ_m,r + k_m · log(n_r)
```

**In plain English:** *"BIC = (how well it fits) minus (penalty for complexity)."*

**Breaking it down:**
- -2·ℓ = Negative log-likelihood (lower is better fit)
- k·log(n) = Penalty for having more parameters
- Models with more parameters must fit much better to justify the complexity

Where:
- ℓ_m,r = maximized log-likelihood (how well model fits data)
- k_m = number of parameters (complexity penalty)
- n_r = sample size in regime r

Weights are softmax over negative BIC:

```
w_raw(m|r) = exp(-½ · (BIC_m,r - BIC_min,r))
p(m|r) = w_raw(m|r) / Σₘ' w_raw(m'|r)
```

**In plain English:** *"Convert BIC differences to probabilities using softmax. Lower BIC → higher probability."*

**Hyvärinen Score (Robust Alternative)**

BIC assumes the true model is in the candidate set. When misspecified, the **Hyvärinen score** is more robust:

```
H(m) = Σₜ [ ∂²log p / ∂r² + ½(∂log p / ∂r)² ]
```

**In plain English:** *"A scoring rule based on the curvature and slope of the log-density. Rewards models that are confident where they should be."*

**Why use it:**
- Works even when no model is "true"
- Naturally rewards accurate tail predictions
- Doesn't require computing normalizing constants

This is a **proper scoring rule** that doesn't require normalizing constants and naturally rewards tail accuracy.

We blend BIC and Hyvärinen:

```
w_combined(m) = w_bic(m)^α · w_hyvarinen(m)^(1-α)
```

**In plain English:** *"Final weight is the geometric mean of BIC weight and Hyvärinen weight."*

Default α = 0.5 (equal weighting).

**Temporal Smoothing**

To prevent erratic model switching, we smooth weights over time:

```
w_smooth(m|r) ∝ w_prev(m|r)^α · w_raw(m|r)
```

**In plain English:** *"New weight = (yesterday's weight)^α × (today's raw weight). This makes weights change gradually."*

With α ≈ 0.85, this creates "sticky" posteriors that adapt gradually.

**Hierarchical Shrinkage**

When regime r has few samples, we shrink toward the global posterior:

```
p(m|r) = (1 - λ) · p_local(m|r) + λ · p(m|global)
```

**In plain English:** *"When data is scarce, borrow strength from the overall (global) model weights."*

**What λ controls:**
- λ = 0: Use only local regime data
- λ = 1: Ignore local data, use global weights entirely
- Default λ = 0.05: Slight shrinkage toward global

Default λ = 0.05. When samples < threshold, we set λ = 1 (full borrowing) and mark `borrowed_from_global = True`.

---

### Signal Engine: Posterior Predictive Monte Carlo

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| p(rₜ₊ₕ \| r_t) | Predictive distribution | Probability of future return given current regime |
| rₜ₊ₕ | "r sub t plus h" | Return h days from now |
| r_t | Current regime | Which of the 5 regimes we're in now |
| N_total | Total samples | Number of Monte Carlo samples (e.g., 10,000) |
| n_m | Samples for model m | Number of samples allocated to model m |
| w | Weight | Model probability p(m\|r) |
| μ | "mu" | Current drift estimate |
| h | Horizon | Forecast period in days |
| q_m | Process noise for model m | Model-specific q parameter |
| σ | "sigma" | Volatility |
| P(r > 0) | Probability positive | Chance that return exceeds zero |
| E[r] | Expected return | Average (mean) return |

</details>

**Monte Carlo Sampling**

We approximate p(rₜ₊ₕ | r_t) via simulation:

```python
samples = []
for m, w in model_posterior.items():
    n_m = int(w * N_total)  # samples proportional to weight
    for _ in range(n_m):
        # Simulate Kalman path for h steps
        μ = current_drift_estimate
        for step in range(h):
            μ += sample_from(N(0, q_m))
            r_step = μ + sample_from(distribution_m(σ))
        samples.append(sum_of_r_steps)
```

**In plain English:**
1. *"For each model, draw samples proportional to how much we trust it."*
2. *"For each sample, simulate the drift evolving over h days."*
3. *"Add up all the daily returns to get the h-day return."*
4. *"Collect all samples into one big distribution."*

**Why this works:**
- Models we trust more contribute more samples
- The final distribution automatically reflects model uncertainty
- We never pick a "best" model — uncertainty is preserved

This produces samples from the full BMA mixture, not from any single model.

**Probability of Positive Return**

From the sample distribution:

```
P(rₜ₊ₕ > 0) = (# samples > 0) / N_total
```

**In plain English:** *"Count how many samples are positive, divide by total samples."*

This is the key quantity for BUY/HOLD/SELL decisions.

**Expected Log Return**

```
E[rₜ₊ₕ] = mean(samples)
```

**In plain English:** *"Average all the samples to get expected return."*

Used for position sizing and expected utility calculations.

**Signal Mapping**

Signals map from probability:

```
P(r > 0) ≥ 0.58  →  BUY
P(r > 0) ∈ (0.42, 0.58)  →  HOLD
P(r > 0) ≤ 0.42  →  SELL
```

**In plain English:**
- *"If there's a 58%+ chance of positive return → BUY"*
- *"If there's a 42% or less chance → SELL"*
- *"Otherwise → HOLD (not enough edge)"*

The 58%/42% thresholds derive from expected utility theory with symmetric loss.

---

### Signal Engine: Expected Utility

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| EU | Expected Utility | Risk-adjusted expected value of a decision |
| p | Probability | Chance of winning |
| U(·) | Utility function | How much we value an outcome |
| f* | "f star" | Optimal bet fraction (Kelly criterion) |
| b | Win/loss ratio | How much we win vs lose |
| z | Z-score | Standardized edge (like Sharpe ratio) |
| μ | "mu" | Expected return (drift) |
| σ | "sigma" | Volatility (standard deviation) |
| h | Horizon | Forecast period in days |
| √h | Square root of h | Scaling factor for multi-day returns |
| z_adj | Adjusted z-score | Z-score after volatility dampening |
| σ_median | Median volatility | "Normal" volatility level |

</details>

**The EU Framework**

Decisions maximize expected utility, not expected return:

```
EU = p · U(gain) + (1-p) · U(loss)
```

**In plain English:** *"Expected utility = (chance of winning × value of winning) + (chance of losing × value of losing)."*

For Kelly-style sizing with log utility U(x) = log(1 + x):

```
f* = p - (1-p)/b
```

**In plain English:** *"Optimal bet size = probability of winning minus (probability of losing divided by win/loss ratio)."*

Where:
- f* = optimal fraction of capital to bet
- p = probability of win
- b = win/loss ratio (how much you win vs lose)

**Example:**
- p = 60%, b = 1.5 (win $1.50 for every $1 risked)
- f* = 0.60 - 0.40/1.5 = 0.60 - 0.27 = 0.33
- *"Bet 33% of capital"*

**Risk-Adjusted Edge**

We compute a Sharpe-style z-score:

```
z = (μ / σ) · √h
```

**In plain English:** *"Edge = (expected return / volatility) × square root of horizon."*

**Why √h?**
- Returns scale linearly with time: μ × h
- Volatility scales with square root: σ × √h
- So Sharpe scales with √h

Where h is the horizon in days. This normalizes edge across timeframes.

**Volatility Regime Dampening**

In high-volatility regimes, we reduce conviction:

```
z_adj = z · (1 - vol_penalty)
vol_penalty = max(0, (σ / σ_median - 1.5) · 0.3)
```

**In plain English:** *"If volatility is 1.5× higher than normal, start reducing our edge estimate."*

**What this means:**
- σ/σ_median = 1.0 (normal vol): No penalty
- σ/σ_median = 1.5 (50% above normal): No penalty yet
- σ/σ_median = 2.0 (double normal): 15% penalty
- σ/σ_median = 3.0 (triple normal): 45% penalty

This prevents overconfidence when uncertainty is elevated.

---

### Debt Engine: Latent State Model

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| S | State | Current latent (hidden) policy state |
| Sₜ | "S sub t" | State at time t |
| Sₜ₋₁ | "S sub t minus 1" | State at previous time step |
| Y | Observation vector | The 5 features we can measure |
| C | Convex loss | Asymmetric penalty for adverse moves |
| P | Tail mass | Probability of extreme outcomes |
| D | Disagreement | How much models disagree (entropy) |
| dD | Disagreement momentum | Rate of change in disagreement |
| V | Vol compression | Volatility relative to recent history |
| P(S\|Y) | State posterior | Probability of state given observations |
| α | "alpha" | Decision threshold (e.g., 0.60) |
| → | Transition arrow | Allowed state transition |

</details>

**State Space**

The debt allocator models policy stress via 4 latent states:

```
S ∈ {NORMAL, COMPRESSED, PRE_POLICY, POLICY}
```

**In plain English:** *"The market is always in one of 4 hidden stress states."*

**What each state means:**

| State | Meaning | Typical Duration |
|-------|---------|------------------|
| NORMAL | Business as usual | Months |
| COMPRESSED | Vol suppressed, pressure building | Weeks |
| PRE_POLICY | Stress emerging, policy imminent | Days |
| POLICY | Active policy intervention | Days to weeks |

States are **partially ordered**: NORMAL → COMPRESSED → PRE_POLICY → POLICY. Backward transitions are forbidden except via explicit reset.

**Observation Model**

We observe a 5-dimensional feature vector:

```
Y = (C, P, D, dD, V)

C  = Convex loss functional (asymmetric penalty for adverse moves)
P  = Tail mass (probability beyond threshold)
D  = Epistemic disagreement (entropy of model posterior)
dD = Disagreement momentum (rate of change in D)
V  = Volatility compression ratio (current vol / recent vol)
```

**In plain English:** *"We measure 5 things about the market that give clues about the hidden state."*

**Transition Dynamics**

State transitions follow a constrained Markov process:

```
P(Sₜ | Sₜ₋₁, Y) ∝ P(Y | Sₜ) · P(Sₜ | Sₜ₋₁)
```

**In plain English:** *"The probability of being in a state depends on what we observe AND where we were before."*

With diagonal dominance (persistence ≈ 0.85) and forward-only transitions.

**Decision Rule**

Switch debt when:

```
P(PRE_POLICY | Y) > α
```

**In plain English:** *"If the probability of being in PRE_POLICY state exceeds our threshold, trigger the switch."*

Default α = 0.60. The decision is **irreversible** (once triggered, done).

---

### Calibration: PIT Test

<details>
<summary><strong>📖 Symbols used in this section</strong></summary>

| Symbol | Name | What it represents |
|--------|------|-------------------|
| u | Uniform value | Transformed probability (should be 0-1 uniform) |
| F | CDF | Cumulative Distribution Function (predicted) |
| F(x) | "F of x" | Probability that outcome ≤ x |
| r_actual | Actual return | The return that actually occurred |
| KS | Kolmogorov-Smirnov | Test statistic measuring calibration |
| sup | Supremum | Maximum value |
| F_empirical | Empirical CDF | CDF estimated from actual data |
| Uniform(0,1) | Uniform distribution | Every value between 0 and 1 equally likely |

</details>

**Probability Integral Transform**

If predictions are well-calibrated:

```
u = F(r_actual)  should be  ~ Uniform(0, 1)
```

**In plain English:** *"If we plug actual outcomes into our predicted CDF, the results should be uniformly distributed."*

**Why this works:**
- If you predict "30% chance of rain" and it rains 30% of the time when you say that, you're calibrated
- PIT is the formal version of this for continuous distributions
- If u values cluster near 0 or 1, predictions are systematically wrong

Where F is the predicted CDF (cumulative distribution function).

**KS Test**

We compute Kolmogorov-Smirnov statistic:

```
KS = sup_u | F_empirical(u) - u |
```

**In plain English:** *"Find the maximum gap between the empirical distribution of u values and the uniform line."*

p-value > 0.05 indicates calibration is acceptable.

**Interpretation**

| Pattern | KS Value | Meaning |
|---------|----------|---------|
| KS ≈ 0 | < 0.05 | Perfect calibration ✓ |
| KS moderate | 0.05-0.10 | Minor miscalibration |
| KS > 0.1 | > 0.10 | Significant miscalibration ✗ |

**Visual patterns in PIT histogram:**
- **U-shape** (values cluster at 0 and 1): Overconfidence — predictions are too narrow
- **∩-shape** (values cluster in middle): Underconfidence — predictions are too wide
- **Flat** (uniform distribution): Well-calibrated ✓

---

### K=2 Mixture Model for Calibration Improvement

When single models fail PIT calibration (p-value < 0.05), the system automatically attempts a **K=2 mixture of symmetric φ-t models** to capture latent regime heterogeneity.

<details>
<summary><strong>📖 Key Insight</strong></summary>

Calibration failures often occur not because the model has wrong parameters, but because markets alternate between **calm** and **stress** regimes within the estimation window. A single symmetric distribution cannot express this asymmetry.

The K=2 mixture solves this by allowing the predictive distribution to allocate mass asymmetrically **without breaking symmetry locally**.

</details>

**Model Definition**

```
p(rₜ | Fₜ₋₁) = w · Tᵥ(rₜ; μₜ, σ_A) + (1-w) · Tᵥ(rₜ; μₜ, σ_B)
```

Where:
- `φ` is **shared** across components (same drift dynamics)
- `ν` is **shared** (same tail thickness)
- `σ_A` = calm regime scale
- `σ_B` = stress regime scale, constrained: `σ_B ≥ 1.5 × σ_A`
- `w ∈ [0.1, 0.9]` = weight on calm component

**Interpretation**

| Component | σ | Role |
|-----------|---|------|
| A (calm) | σ_A (smaller) | Normal market conditions |
| B (stress) | σ_B (larger) | Crisis / tail events |

**Selection Logic**

The mixture model is only selected if:
1. Single model has calibration warning (PIT p < 0.05)
2. Mixture fitting succeeds
3. Mixture BIC < single model BIC - threshold

**Design Principles**

✓ Asymmetry emerges from geometry (σ dispersion), not parameters
✓ K=2 only (no K>2, prevents overfitting)
✓ Shared φ and ν (maintains interpretability)
✓ Static weights (no HMM complexity)
✓ BIC-controlled selection (simpler model preferred)

---

### PIT-Driven Distribution Escalation (PDDE)

The system implements a **hierarchical model escalation** mechanism that automatically upgrades model complexity when diagnostics demand it.

<details>
<summary><strong>📖 Core Principle</strong></summary>

> **Escalate model complexity only when diagnostics demand it.**
> Treat PIT failure as information — not error.

Do NOT expand the global model grid blindly.
Refine locally, conditionally, and reversibly.

</details>

**Escalation Chain**

```
Level 0: φ-Gaussian
    ↓ (PIT p < 0.05)
Level 1: φ-Student-t (coarse ν grid: 4, 6, 8, 12, 20)
    ↓ (PIT fail at boundary ν)
Level 2: Adaptive ν Refinement (local grid expansion)
    ↓ (ν-refinement fails)
Level 3: K=2 Scale Mixture (σ dispersion for regime heterogeneity)
    ↓ (mixture fails, extreme kurtosis)
Level 4: EVT Tail Splice (GPD beyond threshold, rare)
```

**Escalation Triggers**

| Level | Trigger Condition | What It Does |
|-------|-------------------|--------------|
| 0 → 1 | PIT p < 0.05 | Try heavier tails (Student-t) |
| 1 → 2 | Best ν at boundary (12 or 20) | Refine ν locally |
| 2 → 3 | ν-refinement fails | Try regime mixture |
| 3 → 4 | Kurtosis > 10, mixture fails | Apply EVT tail splice |

**Output Contract**

Each asset records its escalation history:

```json
{
  "final_model": "phi-t | phi-t-refined | mixture | evt",
  "escalation_level": 0-4,
  "pit_ks_pvalue": 0.0823,
  "escalation_path": ["baseline_fit", "student_t_selected", "nu_refinement_attempted"],
  "justification": "diagnostic-driven"
}
```

**Files**

| File | Purpose |
|------|---------|
| `src/pit_driven_escalation.py` | Orchestration logic |
| `src/calibration/adaptive_nu_refinement.py` | Level 2: ν refinement |
| `src/calibration/phi_t_mixture_k2.py` | Level 3: K=2 mixture |
| `src/data/calibration/calibration_failures.json` | Diagnostic output |

**View Escalation Summary**

After running `make tune`, the summary shows escalation statistics:

```
📈  MODEL SELECTION

    ○ Gaussian       ████████████░░░░░░░░░   42  ( 35.0%)
    ● Student-t      ████████████████████░   78  ( 65.0%)

    ◆ K=2 Mixture Fallback
      Attempted: 25  →  Selected: 8  (32% success)

    ◇ Adaptive ν Refinement
      Attempted: 15  →  Improved: 6  (40% success)
```

---

### Summary: The Mathematical Contract

This box summarizes the entire system in symbols. Refer to the Master Symbol Glossary at the top of this section for definitions.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   DATA:     rₜ = log(Pₜ/Pₜ₋₁)                               │
│             σₜ² = EWMA(rₜ²)                                 │
│                                                             │
│   TUNING:   μₜ = φμₜ₋₁ + ηₜ        (state equation)         │
│             rₜ = μₜ + εₜ           (observation)            │
│             q* = argmax ℓ(q)       (MLE)                    │
│             p(m|r) ∝ exp(-BIC/2)   (BMA weights)            │
│                                                             │
│   SIGNAL:   p(r|data) = Σₘ p(r|m,θ) · p(m|r)   (mixture)    │
│             P(r>0) = ∫₀^∞ p(r) dr              (probability)│
│             signal = map(P(r>0))               (decision)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Line-by-line translation:**

| Line | Symbols | Plain English |
|------|---------|---------------|
| `rₜ = log(Pₜ/Pₜ₋₁)` | Return = log(today's price / yesterday's price) | "Compute log returns" |
| `σₜ² = EWMA(rₜ²)` | Variance = exponentially-weighted average of squared returns | "Estimate volatility" |
| `μₜ = φμₜ₋₁ + ηₜ` | Drift = phi × yesterday's drift + noise | "Drift evolves with mean-reversion" |
| `rₜ = μₜ + εₜ` | Return = drift + noise | "Observed return is noisy drift" |
| `q* = argmax ℓ(q)` | Optimal q = value that maximizes likelihood | "Find best-fit process noise" |
| `p(m\|r) ∝ exp(-BIC/2)` | Model probability proportional to exp(-BIC/2) | "Weight models by BIC" |
| `p(r\|data) = Σₘ p(r\|m,θ)·p(m\|r)` | Prediction = weighted sum across models | "Average all model predictions" |
| `P(r>0) = ∫₀^∞ p(r) dr` | Probability positive = integral from 0 to ∞ | "Count positive samples" |
| `signal = map(P(r>0))` | Signal = function of probability | "Convert probability to BUY/HOLD/SELL" |

**The math is the system. The code merely implements it.**

---

## Cheat Sheet

### First Time Setup
```bash
make setup              # Install everything, download data
```

### Daily Use
```bash
make stocks             # The one command you need
```

### Weekly Maintenance
```bash
make tune               # Re-calibrate parameters
make stocks             # Generate fresh signals
```

### When Things Break
```bash
make doctor             # Reinstall dependencies
make failed             # See what failed
make purge              # Clear failed cache
make data               # Re-download everything
```

### Quick Reference Table

| I want to... | Command |
|--------------|---------|
| Generate signals | `make stocks` |
| Just see cached signals | `make report` |
| Re-tune all parameters | `make tune ARGS="--force"` |
| Test with few assets | `make top20` |
| Validate calibration | `make fx-calibration` |
| Clear everything | `make clear` |
| See what failed | `make failed` |
| Work offline | `OFFLINE_MODE=1 make stocks` |

---

## Troubleshooting

### Common Issues

**`zsh: command not found: python`**
```bash
echo 'alias python=python3' >> ~/.zshrc && source ~/.zshrc
```

**Build errors on Apple Silicon**
```bash
xcode-select --install
```

**Assets failing to download**
```bash
make failed    # See which assets failed
make purge     # Clear their cache
make data      # Re-download
```

**Stale parameters**
```bash
make clear-q   # Clear parameter cache
make tune      # Re-estimate
make stocks    # Fresh signals
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PRICE_DATA_DIR` | Override data cache location |
| `NO_COLOR=1` | Disable colored output |
| `PYTHON` | Force specific interpreter |
| `OFFLINE_MODE=1` | Use cached data only, no network calls |

---

## Philosophy

### The Core Principle

> *"Act only on beliefs that were actually learned."*

This system is a **belief evolution engine**. It maintains competing hypotheses about market dynamics and lets Bayesian inference arbitrate between them.

### What Makes This Different

| Traditional Systems | This System |
|---------------------|-------------|
| Pick the "best" model | Maintain model uncertainty |
| Point estimates | Full distributions |
| Fixed parameters | Continuously re-calibrated |
| Confidence from conviction | Confidence from calibration |
| Fail silently when wrong | Know when you don't know |

### The Three Laws

1. **Never invent beliefs.** When evidence is weak, become more ignorant—not more confident. Fallback is always hierarchical (regime → global), never fabricated.

2. **Preserve distributional integrity.** Decisions come from distributions, not point estimates. The signal layer sees samples, not parameters.

3. **Separate epistemology from agency.** The Tuning Engine learns beliefs. The Signal Engine acts on them. They never mix.

### The Goal

**Calibrated uncertainty**, not false precision.

When the system says "62% probability," it should be right 62% of the time. Not 70%. Not 55%. Exactly 62%.

That's what the PIT calibration tests verify. That's what makes this system trustworthy.

---

<h1 align="center">🇵🇱 Wersja Polska / Polish Version</h1>

<p align="center">
  <strong>Pełne tłumaczenie dokumentacji na język polski</strong>
</p>

---

## Dlaczego Ten System Istnieje

Większość systemów tradingowych wybiera jeden model i udaje, że jest poprawny. Ten system tak nie działa.

Zamiast tego utrzymuje **7 konkurujących modeli** w **5 reżimach rynkowych**, pozwalając bayesowskiemu wnioskowaniu ciągle aktualizować, które modele są najbardziej wiarygodne w świetle ostatnich danych. Sygnały wyłaniają się z **pełnego rozkładu predykcyjnego a posteriori** — nie z pojedynczego "najlepszego przypuszczenia."

Rezultat: **skalibrowana niepewność**. Kiedy system mówi "62% prawdopodobieństwa dodatniego zwrotu," oznacza to, że historycznie 62% takich prognoz okazało się trafnych.

> *"Celem nie jest mieć rację. Celem jest wiedzieć, jak bardzo powinieneś być pewny."*

---

## System

To jest **silnik ewolucji przekonań**, nie silnik reguł.

W swojej istocie system utrzymuje populację konkurujących modeli — każdy reprezentuje inną hipotezę o dynamice rynku. Te modele ewoluują w prawdopodobieństwie w czasie poprzez bayesowską aktualizację, a sygnały wyłaniają się z pełnego rozkładu predykcyjnego, nie z punktowych estymacji.

### Trzy Silniki

| Silnik | Komenda | Co Robi |
|--------|---------|---------|
| **Silnik Danych** | `make data` | Pobiera OHLCV dla 100+ aktywów, cachuje jako CSV |
| **Silnik Strojenia** | `make tune` | Dopasowuje parametry Kalmana przez MLE, oblicza wagi BMA |
| **Silnik Sygnałów** | `make stocks` | Próbkuje rozkład predykcyjny, mapuje na sygnały |

### Uniwersum Aktywów

System śledzi **100+ aktywów** w wielu klasach:

| Klasa | Przykłady | Liczba |
|-------|-----------|--------|
| **Akcje** | AAPL, MSFT, NVDA, TSLA, JPM, GS, UNH, LLY... | ~80 |
| **Obronność** | LMT, RTX, NOC, GD, BA, HII, AVAV, PLTR... | ~40 |
| **ETF-y** | SPY, VOO, GLD, SLV, SMH | 5 |
| **Towary** | GC=F (Złoto), SI=F (Srebro) | 2 |
| **Krypto** | BTC-USD, MSTR | 2 |
| **FX** | PLNJPY=X | 1 |

Wszystkie ceny są przeliczane na wspólną walutę bazową (PLN) dla analizy na poziomie portfela.

### Uniwersum Modeli

Silnik Strojenia dopasowuje **7 klas modeli** na reżim:

| Model | Parametry | Zastosowanie |
|-------|-----------|--------------|
| `kalman_gaussian` | q, c | Bazowe innowacje gaussowskie |
| `kalman_phi_gaussian` | q, c, φ | AR(1) dryft z gaussowskim |
| `phi_student_t_nu_4` | q, c, φ | Grube ogony (ν=4) |
| `phi_student_t_nu_6` | q, c, φ | Umiarkowane ogony (ν=6) |
| `phi_student_t_nu_8` | q, c, φ | Lekkie ogony (ν=8) |
| `phi_student_t_nu_12` | q, c, φ | Prawie gaussowski (ν=12) |
| `phi_student_t_nu_20` | q, c, φ | Niemal gaussowski (ν=20) |

Modele Student-t używają **dyskretnej siatki ν** (nie ciągłej optymalizacji). Każde ν jest osobnym podmodelem w BMA, pozwalając posteriorowi wyrażać niepewność co do grubości ogonów.

### Klasyfikacja Reżimów

Rynki są klasyfikowane do **5 reżimów** na podstawie zmienności i dryftu:

| Reżim | Warunek |
|-------|---------|
| `LOW_VOL_TREND` (niski vol, trend) | vol < 0.85×mediana, \|dryft\| > próg |
| `HIGH_VOL_TREND` (wysoki vol, trend) | vol > 1.3×mediana, \|dryft\| > próg |
| `LOW_VOL_RANGE` (niski vol, zakres) | vol < 0.85×mediana, \|dryft\| ≤ próg |
| `HIGH_VOL_RANGE` (wysoki vol, zakres) | vol > 1.3×mediana, \|dryft\| ≤ próg |
| `CRISIS_JUMP` (skok kryzysowy) | vol > 2×mediana LUB wskaźnik_ogona > 4 |

Przypisanie reżimu jest **deterministyczne i spójne** między strojeniem a wnioskowaniem.

---

## Szybki Start

### Wymagania

- macOS (Intel lub Apple Silicon)
- Python 3.7+
- ~10GB miejsca na dysku dla cache cen

### Instalacja (Jedna Komenda)

```bash
make setup
```

To wykona:
1. Utworzenie środowiska wirtualnego `.venv/`
2. Instalację zależności z `src/setup/requirements.txt`
3. Pobranie 10 lat danych cenowych (3 przebiegi dla niezawodności)
4. Wyczyszczenie danych w cache

**Czas:** 5-15 minut w zależności od sieci.

### Wygeneruj Swoje Pierwsze Sygnały

```bash
make stocks
```

### Co Zobaczysz

System wyświetla pięknie sformatowane tabele Rich z jakością UX Apple:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    ▲ NVDA  NVIDIA Corporation                                 ┃
┃                    142.58  │  LOW_VOL_TREND  │  2025-01-27  │  Student-t      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  Horyzont  │  P(r>0) │  E[zwrot] │      CI 68%       │    Zysk  │   Sygnał   │    Siła     
 ───────────┼─────────┼───────────┼───────────────────┼──────────┼────────────┼─────────────
  1 dzień   │   54.2% │    +0.08% │ [ -0.6%,  +0.7%]  │      +5k │  — CZEKAJ  │ ──────────  
  1 tydzień │   58.7% │    +0.42% │ [ -1.3%,  +2.3%]  │     +49k │   ↑ KUP    │ ██░░░░░░░░  
  1 miesiąc │   63.1% │    +1.84% │ [ -2.1%,  +5.3%]  │    +170k │   ↑ KUP    │ ████░░░░░░  
  3 miesiące│   71.2% │    +5.62% │ [ -7.9%, +17.5%]  │    +618k │   ↑ KUP    │ ███████░░░  
  6 miesięcy│   68.4% │    +9.93% │ [-24.4%, +44.3%]  │    +1.7M │   ↑ KUP    │ ██████░░░░  
  12 mies.  │   72.8% │   +19.80% │ [-75.5%,+115.1%]  │    +6.2M │  ▲▲ KUP    │ ████████░░  

        P(r>0) prawdop. dodatniego zwrotu    E[zwrot] oczekiwany zwrot    Zysk na 1M PLN

                   ↑ KUP  P ≥ 58%         — CZEKAJ  42% < P < 58%        ↓ SPRZEDAJ  P ≤ 42%
```

**Cechy projektu:**
- **Idealne wyrównanie** — Rich Table zapewnia precyzyjne wyrównanie kolumn
- **Naprzemienne kolory wierszy** — Subtelne szare pasy poprawiają czytelność
- **Paski siły** — Wizualny wskaźnik pewności (█░ dla sygnału, ─ dla neutralnego)
- **Hierarchia kolorów** — Jasny zielony (silny kup), zielony (kup), czerwony (sprzedaj), przyciemniony (czekaj)
- **Odznaki sygnałów** — ▲▲ KUP, ↑ KUP, — CZEKAJ, ↓ SPRZEDAJ, ▼▼ SPRZEDAJ
- **Panel nagłówka** — Karta z obramowaniem z identyfikacją aktywa i reżimem

Sygnały są oznaczone kolorami:
- 🟢 **KUP** (zielony): P(r>0) ≥ 58%
- ⚪ **CZEKAJ** (przygaszony): P(r>0) ∈ (42%, 58%)
- 🔴 **SPRZEDAJ** (czerwony): P(r>0) ≤ 42%

### Zrozumienie Kolumn

| Kolumna | Znaczenie |
|---------|-----------|
| **Horyzont** | Okres prognozy (dni handlowe) |
| **P(r>0)** | Prawdopodobieństwo, że zwrot będzie dodatni |
| **E[zwrot]** | Oczekiwany log-zwrot ze średniej posteriori |
| **Sygnał** | Decyzja wynikająca z progu prawdopodobieństwa |
| **Pewność** | Wizualny wskaźnik wielkości prawdopodobieństwa |

### Zrozumienie Reżimu

Każdy aktyw jest klasyfikowany do jednego z 5 reżimów:

| Reżim | Co Oznacza | Typowe Zachowanie |
|-------|------------|-------------------|
| `LOW_VOL_TREND` | Cichy rynek trendujący | Gładkie, kierunkowe ruchy |
| `HIGH_VOL_TREND` | Zmienny rynek trendujący | Ostre ruchy z kierunkiem |
| `LOW_VOL_RANGE` | Cichy rynek boczny | Powracający do średniej, szarpany |
| `HIGH_VOL_RANGE` | Zmienny rynek boczny | Whipsaw, brak jasnego kierunku |
| `CRISIS_JUMP` | Ekstremalny stres | Zdarzenia ogonowe, korelacje rosną |

Reżim wpływa na to, który model otrzymuje największą wagę w mieszance BMA.

---

## Dzienny Przepływ Pracy

### 30-Sekundowa Poranna Rutyna

```bash
make stocks
```

To wszystko. Ta pojedyncza komenda:
1. Odświeża ostatnie 5 dni danych cenowych
2. Ładuje zapisane parametry Kalmana
3. Generuje sygnały dla wszystkich aktywów
4. Wyświetla sformatowane wyjście

### Kiedy Ponownie Stroić

Silnik Strojenia powinien być uruchamiany:
- **Co tydzień** podczas normalnych rynków
- **Po dużych zmianach reżimu** (skok VIX, ogłoszenie Fed)
- **Gdy sygnały wydają się nieaktualne** lub źle skalibrowane

```bash
# Cotygodniowa kalibracja
make tune

# Wymuś pełną re-estymację (ignoruj cache)
make tune ARGS="--force"
```

### Tryb Offline

Masz już dane w cache? Pracuj bez sieci:

```bash
# Renderuj tylko z cache
make report

# Lub ustaw zmienną środowiskową
OFFLINE_MODE=1 make stocks
```

---

## Referencja Komend

### Główne Komendy

| Komenda | Opis |
|---------|------|
| `make setup` | Pełna konfiguracja: venv + zależności + dane (uruchom raz) |
| `make data` | Pobierz wszystkie dane cenowe (5 prób) |
| `make refresh` | Odśwież ostatnie 5 dni danych |
| `make tune` | Kalibruj parametry Kalmana |
| `make stocks` | **Główna komenda:** odśwież + sygnały |
| `make report` | Renderuj sygnały z cache (offline) |

### Komendy Strojenia

| Komenda | Opis |
|---------|------|
| `make tune` | Strój wszystkie aktywa (używa cache) |
| `make tune ARGS="--force"` | Wymuś re-estymację |
| `make show-q` | Wyświetl zapisane parametry |
| `make clear-q` | Wyczyść cache parametrów |

### Komendy Diagnostyczne

| Komenda | Opis |
|---------|------|
| `make fx-diagnostics` | Pełna diagnostyka (kosztowna) |
| `make fx-diagnostics-lite` | Lekka diagnostyka |
| `make fx-calibration` | Sprawdzenie kalibracji PIT |
| `make fx-model-comparison` | Porównanie modeli AIC/BIC |
| `make fx-validate-kalman` | Walidacja filtru Kalmana |
| `make tests` | Uruchom testy jednostkowe |

### Komendy Narzędziowe

| Komenda | Opis |
|---------|------|
| `make doctor` | Przeinstaluj zależności |
| `make failed` | Lista nieudanych aktywów |
| `make purge` | Wyczyść cache dla nieudanych aktywów |
| `make clear` | Wyczyść wszystkie cache |
| `make clean-cache` | Usuń puste wiersze |
| `make top20` | Szybki test z 20 aktywami |

---

## Matematyka

> *"Matematyka zawsze wyłania się z bazowego systemu — nie odwrotnie."*

Ta sekcja dokumentuje fundamenty matematyczne rządzące każdym silnikiem.

### Główny Słownik Symboli

#### Ceny i Zwroty

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| Pₜ | Cena w czasie t | Cena aktywa w kroku czasowym t |
| rₜ | Zwrot w czasie t | Log-zwrot: ln(Pₜ/Pₜ₋₁) |
| h | Horyzont | Okres prognozy w dniach handlowych |

#### Zmienność

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| σ | Sigma | Odchylenie standardowe (zmienność) |
| σₜ² | Sigma kwadrat | Wariancja w czasie t |
| λ | Lambda | Współczynnik zaniku w EWMA (0.94-0.97) |

#### Filtr Kalmana

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| μₜ | Mu | Ukryty (latentny) dryft w czasie t |
| q | Szum procesu | O ile dryft może zmienić się na krok |
| ηₜ | Eta | Losowy szok do dryftu ~ N(0, q) |
| εₜ | Epsilon | Szum obserwacji ~ N(0, σ²) |
| K | Wzmocnienie Kalmana | Waga nadana nowej obserwacji (0-1) |
| P | Wariancja stanu | Niepewność w estymacji dryftu |
| m | Średnia posteriori | Najlepsza estymacja dryftu po aktualizacji |

#### Model AR(1)

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| φ | Phi | Współczynnik powrotu do średniej (-1 do 1) |
| τ | Tau | Priorytetowe odchylenie standardowe dla φ |

#### Rozkład Studenta-t

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| ν | Nu | Stopnie swobody (grubość ogonów) |
| t_ν | Student-t | Rozkład t z ν stopniami swobody |

#### Wnioskowanie Bayesowskie

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| p(·) | Prawdopodobieństwo | Funkcja prawdopodobieństwa lub gęstości |
| p(m\|r) | Posterior modelu | Prawdopodobieństwo modelu m przy reżimie r |
| θ | Theta | Parametry modelu (q, φ, itd.) |
| ℓ | Log-wiarygodność | Suma logarytmów prawdopodobieństw |

#### Selekcja Modeli

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| BIC | Bayesowskie Kryterium Inf. | Penalizowana wiarygodność do porównań |
| k | Liczba parametrów | Liczba wolnych parametrów w modelu |
| n | Wielkość próby | Liczba obserwacji |
| w | Waga | Nieznormalizowana waga modelu |
| α | Alfa | Współczynnik wygładzania/mieszania |

#### Teoria Decyzji

| Symbol | Nazwa | Znaczenie |
|--------|-------|-----------|
| E[·] | Wartość oczekiwana | Średnia wartość |
| P(·) | Prawdopodobieństwo | Szansa zdarzenia |
| EU | Oczekiwana Użyteczność | Skorygowana o ryzyko wartość oczekiwana |
| f* | Optymalna frakcja | Wielkość zakładu wg kryterium Kelly'ego |
| z | Z-score | Standaryzowana metryka przewagi |

---

### Silnik Danych: Zwroty i Zmienność

**Log-Zwroty**

System pracuje z log-zwrotami, nie prostymi zwrotami:

```
rₜ = log(Pₜ / Pₜ₋₁)
```

**Po polsku:** *"Dzisiejszy zwrot równa się logarytmowi naturalnemu z dzisiejszej ceny podzielonej przez wczorajszą cenę."*

**Zrealizowana Zmienność**

Zmienność jest estymowana przez wykładniczo-ważoną średnią ruchomą (EWMA):

```
σₜ² = λ · σₜ₋₁² + (1 - λ) · rₜ²
```

**Po polsku:** *"Dzisiejsza wariancja równa się lambda razy wczorajsza wariancja, plus (1 - lambda) razy dzisiejszy zwrot do kwadratu."*

**Co to oznacza:**
- Gdy λ = 0.94: Wczorajsza wariancja dostaje 94% wagi, dzisiejszy zwrot 6%
- Wyższe λ = wolniejsza adaptacja do nowych informacji
- Niższe λ = szybsza adaptacja, bardziej reaktywne

**Winsoryzacja**

Ekstremalne zwroty są przycinane, aby zmniejszyć wpływ wartości odstających:

```
rₜ → clip(rₜ, -3σ, +3σ)
```

**Po polsku:** *"Jeśli zwrot jest bardziej ekstremalny niż 3 odchylenia standardowe, ogranicz go do 3 odchyleń standardowych."*

---

### Silnik Strojenia: Filtr Kalmana + MLE

**Model Przestrzeni Stanów**

Modelujemy latentny dryft μₜ jako błądzenie losowe obserwowane przez zaszumione zwroty:

```
Równanie stanu:      μₜ = μₜ₋₁ + ηₜ,     ηₜ ~ N(0, q)
Obserwacja:          rₜ = μₜ + εₜ,       εₜ ~ N(0, σₜ²)
```

**Po polsku:**
- *"Prawdziwy dryft dzisiaj równa się wczorajszemu dryftowi plus losowy szok."*
- *"Obserwowany zwrot równa się prawdziwemu dryftowi plus szum rynkowy."*

**Rekurencja Filtru Kalmana**

Przy danym priorze μₜ₋₁|ₜ₋₁ ~ N(m, P), filtr Kalmana aktualizuje:

```
Predykcja:  μₜ|ₜ₋₁ ~ N(m, P + q)

Aktualizacja: K = (P + q) / (P + q + σₜ²)     # Wzmocnienie Kalmana
              mₜ = m + K · (rₜ - m)            # Średnia posteriori
              Pₜ = (1 - K) · (P + q)           # Wariancja posteriori
```

**Po polsku:**
1. **Predykcja:** *"Przed zobaczeniem dzisiejszego zwrotu, nasza niepewność rośnie o q."*
2. **Wzmocnienie Kalmana:** *"K mierzy, ile ufać nowej obserwacji vs. naszemu priorowi."*
3. **Aktualizacja średniej:** *"Nowa estymacja = stara estymacja + K × (niespodzianka)."*
4. **Aktualizacja wariancji:** *"Nasza niepewność maleje po zobaczeniu danych."*

**Estymacja Największej Wiarygodności (MLE)**

Znajdujemy q maksymalizując log-wiarygodność:

```
ℓ(q) = Σₜ log p(rₜ | r₁:ₜ₋₁, q)
```

**Po polsku:** *"Znajdź wartość q, która sprawia, że obserwowane zwroty są najbardziej prawdopodobne."*

**Rozszerzenie AR(1) (modele φ)**

Dla dryftu powracającego do średniej, rozszerzamy równanie stanu:

```
μₜ = φ · μₜ₋₁ + ηₜ,     φ ∈ (-1, 1)
```

**Po polsku:** *"Dzisiejszy dryft równa się phi razy wczorajszy dryft, plus szum."*

**Co oznaczają wartości φ:**
- φ = 0: Dryft nie ma pamięci (pełny powrót do średniej)
- φ = 0.9: Dryft jest bardzo trwały (wolny powrót do średniej)
- φ = 1: Błądzenie losowe (brak powrotu do średniej) — **niestabilne, unikamy**
- φ < 0: Dryft oscyluje (rzadkie w danych finansowych)

**Innowacje Studenta-t**

Aby uchwycić grube ogony, zastępujemy gaussowskie innowacje rozkładem Studenta-t:

```
εₜ ~ t_ν(0, σₜ)
```

**Po polsku:** *"Szum rynkowy podąża za rozkładem Studenta-t zamiast gaussowskiego, pozwalając na rzadkie ekstremalne ruchy."*

---

### Silnik Strojenia: Bayesowskie Uśrednianie Modeli

**Równanie BMA**

Przy danym reżimie r i klasie modelu m z parametrami θ, rozkład predykcyjny posteriori to:

```
p(rₜ₊ₕ | r) = Σₘ p(rₜ₊ₕ | r, m, θᵣ,ₘ) · p(m | r)
```

**Po polsku:** *"Prawdopodobieństwo przyszłego zwrotu równa się ważonej średniej prognoz każdego modelu, gdzie wagi to ile ufamy każdemu modelowi."*

To jest **główne równanie** systemu. Sygnały wyłaniają się z tej mieszanki, nie z żadnego pojedynczego "najlepszego" modelu.

**Wagi Modeli przez BIC**

Dla każdego modelu m w reżimie r, obliczamy BIC:

```
BIC_m,r = -2 · ℓ_m,r + k_m · log(n_r)
```

**Po polsku:** *"BIC = (jak dobrze pasuje) minus (kara za złożoność)."*

**Wygładzanie Czasowe**

Aby zapobiec gwałtownemu przełączaniu modeli, wygładzamy wagi w czasie:

```
w_smooth(m|r) ∝ w_prev(m|r)^α · w_raw(m|r)
```

**Po polsku:** *"Nowa waga = (wczorajsza waga)^α × (dzisiejsza surowa waga). To sprawia, że wagi zmieniają się stopniowo."*

**Hierarchiczne Kurczenie**

Gdy reżim r ma mało próbek, kurczymy w kierunku globalnego posterioru:

```
p(m|r) = (1 - λ) · p_local(m|r) + λ · p(m|global)
```

**Po polsku:** *"Gdy danych jest mało, pożyczaj siłę z ogólnych (globalnych) wag modeli."*

---

### Silnik Sygnałów: Monte Carlo Predykcyjne Posteriori

**Próbkowanie Monte Carlo**

Przybliżamy p(rₜ₊ₕ | r_t) przez symulację:

```python
samples = []
for m, w in model_posterior.items():
    n_m = int(w * N_total)  # próbki proporcjonalne do wagi
    for _ in range(n_m):
        μ = current_drift_estimate
        for step in range(h):
            μ += sample_from(N(0, q_m))
            r_step = μ + sample_from(distribution_m(σ))
        samples.append(sum_of_r_steps)
```

**Po polsku:**
1. *"Dla każdego modelu, losuj próbki proporcjonalnie do tego, jak bardzo mu ufamy."*
2. *"Dla każdej próbki, symuluj ewolucję dryftu przez h dni."*
3. *"Zsumuj wszystkie dzienne zwroty, aby uzyskać zwrot h-dniowy."*
4. *"Zbierz wszystkie próbki w jeden duży rozkład."*

**Prawdopodobieństwo Dodatniego Zwrotu**

Z rozkładu próbek:

```
P(rₜ₊ₕ > 0) = (# próbek > 0) / N_total
```

**Po polsku:** *"Policz ile próbek jest dodatnich, podziel przez całkowitą liczbę próbek."*

**Mapowanie Sygnałów**

Sygnały mapują z prawdopodobieństwa:

```
P(r > 0) ≥ 0.58  →  KUP
P(r > 0) ∈ (0.42, 0.58)  →  CZEKAJ
P(r > 0) ≤ 0.42  →  SPRZEDAJ
```

**Po polsku:**
- *"Jeśli jest 58%+ szans na dodatni zwrot → KUP"*
- *"Jeśli jest 42% lub mniej szans → SPRZEDAJ"*
- *"W przeciwnym razie → CZEKAJ (niewystarczająca przewaga)"*

---

### Silnik Sygnałów: Oczekiwana Użyteczność

**Ramy EU**

Decyzje maksymalizują oczekiwaną użyteczność, nie oczekiwany zwrot:

```
EU = p · U(zysk) + (1-p) · U(strata)
```

**Po polsku:** *"Oczekiwana użyteczność = (szansa wygranej × wartość wygranej) + (szansa przegranej × wartość przegranej)."*

Dla rozmiaru pozycji w stylu Kelly'ego z logarytmiczną użytecznością U(x) = log(1 + x):

```
f* = p - (1-p)/b
```

**Po polsku:** *"Optymalna wielkość zakładu = prawdopodobieństwo wygranej minus (prawdopodobieństwo przegranej podzielone przez stosunek wygrana/przegrana)."*

**Przykład:**
- p = 60%, b = 1.5 (wygrana $1.50 za każdy zaryzykowany $1)
- f* = 0.60 - 0.40/1.5 = 0.60 - 0.27 = 0.33
- *"Postaw 33% kapitału"*

---

### Kalibracja: Test PIT

**Transformata Całkowa Prawdopodobieństwa**

Jeśli prognozy są dobrze skalibrowane:

```
u = F(r_actual)  powinno być  ~ Uniform(0, 1)
```

**Po polsku:** *"Jeśli podstawimy rzeczywiste wyniki do naszej prognozowanej dystrybuanty, wyniki powinny być równomiernie rozłożone."*

**Test KS**

Obliczamy statystykę Kołmogorowa-Smirnowa:

```
KS = sup_u | F_empirical(u) - u |
```

**Po polsku:** *"Znajdź maksymalną lukę między empirycznym rozkładem wartości u a linią równomierną."*

p-value > 0.05 wskazuje, że kalibracja jest akceptowalna.

**Interpretacja**

| Wzorzec | Wartość KS | Znaczenie |
|---------|------------|-----------|
| KS ≈ 0 | < 0.05 | Idealna kalibracja ✓ |
| KS umiarkowane | 0.05-0.10 | Mniejsza błędna kalibracja |
| KS > 0.1 | > 0.10 | Znacząca błędna kalibracja ✗ |

**Wzorce wizualne w histogramie PIT:**
- **Kształt U** (wartości grupują się przy 0 i 1): Nadmierna pewność — prognozy są zbyt wąskie
- **Kształt ∩** (wartości grupują się w środku): Niedostateczna pewność — prognozy są zbyt szerokie
- **Płaski** (rozkład równomierny): Dobrze skalibrowany ✓

---

### Podsumowanie: Kontrakt Matematyczny

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   DANE:     rₜ = log(Pₜ/Pₜ₋₁)                               │
│             σₜ² = EWMA(rₜ²)                                 │
│                                                             │
│   STROJENIE: μₜ = φμₜ₋₁ + ηₜ       (równanie stanu)         │
│              rₜ = μₜ + εₜ          (obserwacja)             │
│              q* = argmax ℓ(q)      (MLE)                    │
│              p(m|r) ∝ exp(-BIC/2)  (wagi BMA)               │
│                                                             │
│   SYGNAŁ:   p(r|dane) = Σₘ p(r|m,θ) · p(m|r)   (mieszanka) │
│             P(r>0) = ∫₀^∞ p(r) dr    (prawdopodobieństwo)  │
│             sygnał = map(P(r>0))     (decyzja)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Matematyka jest systemem. Kod jedynie ją implementuje.**

---

## Ściągawka

### Pierwsza Konfiguracja
```bash
make setup              # Zainstaluj wszystko, pobierz dane
```

### Codzienne Użycie
```bash
make stocks             # Jedna komenda, której potrzebujesz
```

### Cotygodniowa Konserwacja
```bash
make tune               # Przekalibruj parametry
make stocks             # Wygeneruj świeże sygnały
```

### Gdy Coś Nie Działa
```bash
make doctor             # Przeinstaluj zależności
make failed             # Zobacz co się nie udało
make purge              # Wyczyść cache nieudanych
make data               # Pobierz wszystko ponownie
```

### Tabela Szybkiej Referencji

| Chcę... | Komenda |
|---------|---------|
| Wygenerować sygnały | `make stocks` |
| Tylko zobaczyć zapisane sygnały | `make report` |
| Przekalibrować wszystkie parametry | `make tune ARGS="--force"` |
| Przetestować z kilkoma aktywami | `make top20` |
| Zwalidować kalibrację | `make fx-calibration` |
| Wyczyścić wszystko | `make clear` |
| Zobaczyć co się nie udało | `make failed` |
| Pracować offline | `OFFLINE_MODE=1 make stocks` |

---

## Filozofia

### Główna Zasada

> *"Działaj tylko na podstawie przekonań, które faktycznie zostały wyuczone."*

Ten system to **silnik ewolucji przekonań**. Utrzymuje konkurujące hipotezy o dynamice rynku i pozwala bayesowskiemu wnioskowaniu arbitrować między nimi.

### Co Czyni To Innym

| Tradycyjne Systemy | Ten System |
|--------------------|------------|
| Wybierz "najlepszy" model | Utrzymuj niepewność modelu |
| Punktowe estymacje | Pełne rozkłady |
| Stałe parametry | Ciągle przekalibrowywane |
| Pewność z przekonania | Pewność z kalibracji |
| Cicho zawodzą gdy błędne | Wiedzą, kiedy nie wiedzą |

### Trzy Prawa

1. **Nigdy nie wymyślaj przekonań.** Gdy dowody są słabe, stań się bardziej niewiedzący — nie bardziej pewny. Fallback jest zawsze hierarchiczny (reżim → globalny), nigdy sfabrykowany.

2. **Zachowaj integralność rozkładową.** Decyzje pochodzą z rozkładów, nie punktowych estymacji. Warstwa sygnałów widzi próbki, nie parametry.

3. **Oddziel epistemologię od agencji.** Silnik Strojenia uczy się przekonań. Silnik Sygnałów działa na ich podstawie. Nigdy się nie mieszają.

### Cel

**Skalibrowana niepewność**, nie fałszywa precyzja.

Gdy system mówi "62% prawdopodobieństwa," powinien mieć rację w 62% przypadków. Nie 70%. Nie 55%. Dokładnie 62%.

To właśnie weryfikują testy kalibracji PIT. To sprawia, że ten system jest godny zaufania.

---

## Licencja / License

This project is for educational and research purposes. See individual dependencies for their respective licenses.

Ten projekt służy celom edukacyjnym i badawczym. Zobacz poszczególne zależności dla ich odpowiednich licencji.

---

<p align="center">
  <sub>Built with scientific rigor and engineering craftsmanship.</sub>
</p>

<p align="center">
  <sub>Zbudowany z naukową rygorem i rzemieślniczym kunsztem.</sub>
</p>

<p align="center">
  <sub>The math is the system. The code merely implements it.</sub>
</p>

<p align="center">
  <sub>Matematyka jest systemem. Kod jedynie ją implementuje.</sub>
</p>