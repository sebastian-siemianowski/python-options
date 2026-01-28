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
║   │   Output: data/{SYMBOL}_1d.csv                                              │     ║
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
║   │   Output: scripts/quant/cache/kalman_q_cache.json                           │     ║
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
2. Install dependencies from `requirements.txt`
3. Download 10 years of price data (3 passes for reliability)
4. Clean cached data

**Time:** 5-15 minutes depending on network.

### Generate Your First Signals

```bash
make stocks
```

### What You'll See

The system outputs beautifully formatted Rich console tables:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│                           NVDA — NVIDIA Corporation                       │
│                      Regime: LOW_VOL_TREND │ Current: $142.58             │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

 Horizon     P(r>0)    E[return]    Signal     Confidence
─────────────────────────────────────────────────────────────
 1 day       54.2%      +0.08%      HOLD       ██░░░░░░░░
 1 week      58.7%      +0.42%      BUY        ████░░░░░░
 1 month     63.1%      +1.84%      BUY        ██████░░░░
 3 months    71.2%      +5.62%      BUY        ████████░░
 12 months   78.4%     +18.41%      BUY        █████████░
```

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
1. Creates `.venv/` via `setup_venv.sh`
2. Upgrades pip and installs `requirements.txt`
3. Runs `precache_data.py` **3 times** (Yahoo Finance is flaky)
4. Runs `clean_cache.py` to remove empty rows

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
- Runs `refresh_data.py --skip-trim --retries 5 --workers 12 --batch-size 16`
- 5 retry passes (Yahoo Finance rate-limits aggressively)
- 12 parallel workers, 16 assets per batch
- Output: `data/<SYMBOL>_1d.csv`

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

Lists assets that failed to download (stored in `scripts/fx_failures.json`).

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
1. Loads asset universe from `fx_data_utils.py`
2. For each asset, for each of 5 regimes:
   - Fits 7 model classes (Gaussian, AR(1)-Gaussian, Student-t with ν ∈ {4,6,8,12,20})
   - Computes BIC, AIC, Hyvärinen score, PIT diagnostics
   - Converts scores to posterior weights
   - Applies temporal smoothing against previous run
   - Applies hierarchical shrinkage toward global
3. Saves to `scripts/quant/cache/kalman_q_cache.json`

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
1. Runs `refresh_data.py` (updates last 5 days)
2. Runs `fx_pln_jpy_signals.py` with caching enabled
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
make fx-validate-kalman-plots                  # Also saves plots to plots/kalman_validation/
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

Output: `scripts/quant/cache/debt/`

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
- `plots/<TICKER>_support_resistance.png`

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
- `backtests/<TICKER>_equity.csv`
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

Builds `data/universes/russell2500_tickers.csv` from public sources.

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
- `plots/*.png`
- `data/meta/`
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
├── scripts/
│   ├── tune_q_mle.py           # TUNING ENGINE: MLE + BMA
│   ├── tune_pretty.py          # Tuning UX wrapper
│   ├── fx_pln_jpy_signals.py   # SIGNAL ENGINE: Posterior predictive
│   ├── fx_signals_presentation.py  # Rich console output
│   ├── refresh_data.py         # DATA ENGINE: Bulk download
│   ├── fx_data_utils.py        # Data utilities + caching
│   ├── debt_allocator.py       # Debt switch decision engine
│   └── quant/
│       └── cache/
│           └── kalman_q_cache.json  # Tuned parameters
│
├── data/                       # Price cache (CSV per symbol)
├── options.py                  # Options screener
├── backtests/                  # Equity curves
└── plots/                      # Generated charts
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

## The Mathematics

> *"The math always emerges from the underlying system—not the other way around."*

This section documents the mathematical foundations that govern each engine. The code implements these equations; understanding them illuminates why the system behaves as it does.

---

### Data Engine: Returns and Volatility

**Log Returns**

The system works with log returns, not simple returns:

```
rₜ = log(Pₜ / Pₜ₋₁)
```

Log returns are additive over time and approximately normal for small values, which simplifies the probabilistic machinery.

**Realized Volatility**

Volatility is estimated via exponentially-weighted moving average (EWMA):

```
σₜ² = λ · σₜ₋₁² + (1 - λ) · rₜ²
```

Where λ ∈ (0,1) controls decay. We use multiple speeds:
- **Fast** (λ = 0.94): Responsive to recent moves
- **Slow** (λ = 0.97): Smoother, less reactive

The final volatility blends both for robustness.

**Winsorization**

Extreme returns are clipped to reduce outlier influence:

```
rₜ → clip(rₜ, -3σ, +3σ)
```

This makes parameter estimation more stable without discarding information entirely.

---

### Tuning Engine: Kalman Filter + MLE

**The State-Space Model**

We model latent drift μₜ as a random walk observed through noisy returns:

```
State equation:     μₜ = μₜ₋₁ + ηₜ,     ηₜ ~ N(0, q)
Observation:        rₜ = μₜ + εₜ,       εₜ ~ N(0, σₜ²)
```

Here:
- **μₜ** is the unobserved "true" drift
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

The Kalman gain K ∈ (0,1) balances prior belief against new evidence.

**Maximum Likelihood Estimation**

We find q by maximizing the log-likelihood:

```
ℓ(q) = Σₜ log p(rₜ | r₁:ₜ₋₁, q)
     = -½ Σₜ [ log(2π · vₜ) + (rₜ - mₜ)² / vₜ ]
```

Where vₜ = P + q + σₜ² is the predictive variance.

**Regularization Prior**

To prevent overfitting, we add a Gaussian prior on log₁₀(q):

```
log₁₀(q) ~ N(μ_prior, 1/λ)
```

Default: μ_prior = -6 (q ≈ 10⁻⁶), λ = 1.0

The penalized objective becomes:

```
ℓ_penalized(q) = ℓ(q) - λ/2 · (log₁₀(q) - μ_prior)²
```

**AR(1) Extension (φ-models)**

For mean-reverting drift, we extend the state equation:

```
μₜ = φ · μₜ₋₁ + ηₜ,     φ ∈ (-1, 1)
```

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

The degrees-of-freedom ν controls tail thickness:
- ν = 4: Very heavy tails
- ν = 20: Nearly Gaussian
- ν → ∞: Gaussian limit

We use a discrete grid ν ∈ {4, 6, 8, 12, 20} and let BMA select the mixture.

---

### Tuning Engine: Bayesian Model Averaging

**The BMA Equation**

Given regime r and model class m with parameters θ, the posterior predictive is:

```
p(rₜ₊ₕ | r) = Σₘ p(rₜ₊ₕ | r, m, θᵣ,ₘ) · p(m | r)
```

This is the **core equation** of the system. Signals emerge from this mixture, not from any single "best" model.

**Model Weights via BIC**

For each model m in regime r, we compute BIC:

```
BIC_m,r = -2 · ℓ_m,r + k_m · log(n_r)
```

Where:
- ℓ_m,r = maximized log-likelihood
- k_m = number of parameters
- n_r = sample size in regime r

Weights are softmax over negative BIC:

```
w_raw(m|r) = exp(-½ · (BIC_m,r - BIC_min,r))
p(m|r) = w_raw(m|r) / Σₘ' w_raw(m'|r)
```

**Hyvärinen Score (Robust Alternative)**

BIC assumes the true model is in the candidate set. When misspecified, the **Hyvärinen score** is more robust:

```
H(m) = Σₜ [ ∂²log p / ∂r² + ½(∂log p / ∂r)² ]
```

This is a **proper scoring rule** that doesn't require normalizing constants and naturally rewards tail accuracy.

We blend BIC and Hyvärinen:

```
w_combined(m) = w_bic(m)^α · w_hyvarinen(m)^(1-α)
```

Default α = 0.5.

**Temporal Smoothing**

To prevent erratic model switching, we smooth weights over time:

```
w_smooth(m|r) ∝ w_prev(m|r)^α · w_raw(m|r)
```

With α ≈ 0.85, this creates "sticky" posteriors that adapt gradually.

**Hierarchical Shrinkage**

When regime r has few samples, we shrink toward the global posterior:

```
p(m|r) = (1 - λ) · p_local(m|r) + λ · p(m|global)
```

Default λ = 0.05. When samples < threshold, we set λ = 1 (full borrowing) and mark `borrowed_from_global = True`.

---

### Signal Engine: Posterior Predictive Monte Carlo

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

This produces samples from the full BMA mixture, not from any single model.

**Probability of Positive Return**

From the sample distribution:

```
P(rₜ₊ₕ > 0) = (# samples > 0) / N_total
```

This is the key quantity for BUY/HOLD/SELL decisions.

**Expected Log Return**

```
E[rₜ₊ₕ] = mean(samples)
```

Used for position sizing and expected utility calculations.

**Signal Mapping**

Signals map from probability:

```
P(r > 0) ≥ 0.58  →  BUY
P(r > 0) ∈ (0.42, 0.58)  →  HOLD
P(r > 0) ≤ 0.42  →  SELL
```

The 58%/42% thresholds derive from expected utility theory with symmetric loss.

---

### Signal Engine: Expected Utility

**The EU Framework**

Decisions maximize expected utility, not expected return:

```
EU = p · U(gain) + (1-p) · U(loss)
```

For Kelly-style sizing with log utility U(x) = log(1 + x):

```
f* = p - (1-p)/b
```

Where:
- f* = optimal fraction of capital
- p = probability of win
- b = win/loss ratio

**Risk-Adjusted Edge**

We compute a Sharpe-style z-score:

```
z = (μ / σ) · √h
```

Where h is the horizon in days. This normalizes edge across timeframes.

**Volatility Regime Dampening**

In high-volatility regimes, we reduce conviction:

```
z_adj = z · (1 - vol_penalty)
vol_penalty = max(0, (σ / σ_median - 1.5) · 0.3)
```

This prevents overconfidence when uncertainty is elevated.

---

### Debt Engine: Latent State Model

**State Space**

The debt allocator models policy stress via 4 latent states:

```
S ∈ {NORMAL, COMPRESSED, PRE_POLICY, POLICY}
```

States are **partially ordered**: NORMAL → COMPRESSED → PRE_POLICY → POLICY. Backward transitions are forbidden except via explicit reset.

**Observation Model**

We observe a 5-dimensional feature vector:

```
Y = (C, P, D, dD, V)

C  = Convex loss functional (asymmetric penalty for adverse moves)
P  = Tail mass (probability beyond threshold)
D  = Epistemic disagreement (entropy of model posterior)
dD = Disagreement momentum (rate of change)
V  = Volatility compression ratio
```

**Transition Dynamics**

State transitions follow a constrained Markov process:

```
P(Sₜ | Sₜ₋₁, Y) ∝ P(Y | Sₜ) · P(Sₜ | Sₜ₋₁)
```

With diagonal dominance (persistence ≈ 0.85) and forward-only transitions.

**Decision Rule**

Switch debt when:

```
P(PRE_POLICY | Y) > α
```

Default α = 0.60. The decision is **irreversible** (once triggered, done).

---

### Calibration: PIT Test

**Probability Integral Transform**

If predictions are well-calibrated:

```
u = F(r_actual)  should be  ~ Uniform(0, 1)
```

Where F is the predicted CDF.

**KS Test**

We compute Kolmogorov-Smirnov statistic:

```
KS = sup_u | F_empirical(u) - u |
```

p-value > 0.05 indicates calibration is acceptable.

**Interpretation**

- KS ≈ 0: Perfect calibration
- KS > 0.1: Miscalibration detected
- Systematic U-shape in PIT histogram: Overconfidence
- Systematic ∩-shape: Underconfidence

---

### Summary: The Mathematical Contract

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   DATA:     rₜ = log(Pₜ/Pₜ₋₁)                                 │
│             σₜ² = EWMA(rₜ²)                                  │
│                                                             │
│   TUNING:   μₜ = φμₜ₋₁ + ηₜ        (state equation)           │
│             rₜ = μₜ + εₜ           (observation)              │
│             q* = argmax ℓ(q)       (MLE)                    │
│             p(m|r) ∝ exp(-BIC/2)   (BMA weights)            │
│                                                             │
│   SIGNAL:   p(r|data) = Σₘ p(r|m,θ) · p(m|r)   (mixture)    │
│             P(r>0) = ∫₀^∞ p(r) dr              (probability)│
│             signal = map(P(r>0))               (decision)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The math is the system. The code merely implements it.

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

## License

This project is for educational and research purposes. See individual dependencies for their respective licenses.

---

<p align="center">
  <sub>Built with scientific rigor and engineering craftsmanship.</sub>
</p>

<p align="center">
  <sub>The math is the system. The code merely implements it.</sub>
</p>