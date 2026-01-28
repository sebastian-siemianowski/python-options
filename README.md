<h1 align="center">Quantitative Signal Engine</h1>

<p align="center">
  <strong>Bayesian Model Averaging meets Kalman Filtering for multi-asset signal generation</strong>
</p>

<p align="center">
  <a href="#the-system">The System</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#daily-workflow">Daily Workflow</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#command-reference">Commands</a>
</p>

<p align="center">
  <sub>Built for practitioners who want rigorous probabilistic inference without the academic overhead.</sub>
</p>

---

## The System

This is a **belief evolution engine**, not a rule engine.

At its core, the system maintains a population of competing models—each representing a different hypothesis about market dynamics. These models evolve in probability over time through Bayesian updating, and signals emerge from the full predictive distribution, not from point estimates.

### Three Engines

| Engine | Command | Purpose |
|--------|---------|---------|
| **Data Engine** | `make data` | Fetches and caches OHLCV for 50+ assets |
| **Tuning Engine** | `make tune` | Calibrates Kalman parameters via MLE + BMA |
| **Signal Engine** | `make stocks` | Generates Buy/Hold/Sell from posterior predictive |

The engines form a pipeline: **Data → Tune → Signal**

```
Price Data (Yahoo Finance)
       ↓
   make data
       ↓
┌──────────────────────────────────────────┐
│  TUNING ENGINE (make tune)               │
│                                          │
│  For each regime r ∈ {5 regimes}:        │
│    For each model m ∈ {7 models}:        │
│      • Fit θ_{r,m} via MLE               │
│      • Compute BIC, Hyvärinen score      │
│    → p(m|r) via BIC-weighted posterior   │
│    → Apply temporal smoothing            │
│    → Hierarchical shrinkage to global    │
└──────────────────────────────────────────┘
       ↓
   kalman_q_cache.json
       ↓
┌──────────────────────────────────────────┐
│  SIGNAL ENGINE (make stocks)             │
│                                          │
│  For current regime r_t:                 │
│    p(x|r_t) = Σ_m p(x|r_t,m,θ) · p(m|r_t)│
│                                          │
│  → Posterior predictive Monte Carlo      │
│  → Expected utility calculation          │
│  → Position sizing via Kelly geometry    │
└──────────────────────────────────────────┘
       ↓
   BUY / HOLD / SELL signals
```

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
- 10GB disk space for price cache

### One-Command Setup

```bash
make setup
```

This will:
1. Create `.venv/` virtual environment
2. Install dependencies from `requirements.txt`
3. Download price data (3 passes for reliability)
4. Clean cached data

**Time:** 5-15 minutes depending on network.

### First Run

After setup, generate your first signals:

```bash
make stocks
```

You'll see a beautiful Rich console output with:
- Per-asset signal tables (1d → 252d horizons)
- Probability estimates with confidence intervals
- Color-coded Buy/Hold/Sell recommendations

---

## Daily Workflow

### Morning Routine

```bash
# 1. Refresh price data (last 5 days)
make refresh

# 2. Generate signals
make stocks
```

### Weekly Calibration

```bash
# Re-estimate Kalman parameters
make tune

# Then generate signals with fresh parameters
make stocks
```

### When Parameters Feel Stale

```bash
# Force full re-estimation (ignore cache)
make tune ARGS="--force"
```

---

## Command Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `make setup` | Full setup: venv + deps + data (run once) |
| `make data` | Download all price data (5 retries) |
| `make refresh` | Refresh last 5 days of data |
| `make tune` | Calibrate Kalman parameters |
| `make stocks` | **Main command:** refresh + signals |
| `make report` | Render signals from cache (offline) |

### Tuning Commands

| Command | Description |
|---------|-------------|
| `make tune` | Tune all assets (uses cache) |
| `make tune ARGS="--force"` | Force re-estimation |
| `make show-q` | Display cached parameters |
| `make clear-q` | Clear parameter cache |

### Diagnostic Commands

| Command | Description |
|---------|-------------|
| `make fx-diagnostics` | Full diagnostics (expensive) |
| `make fx-diagnostics-lite` | Lightweight diagnostics |
| `make fx-calibration` | PIT calibration check |
| `make fx-model-comparison` | AIC/BIC model comparison |
| `make fx-validate-kalman` | Kalman filter validation |
| `make tests` | Run unit tests |

### Utility Commands

| Command | Description |
|---------|-------------|
| `make doctor` | Reinstall dependencies |
| `make failed` | List failed assets |
| `make purge` | Clear cache for failed assets |
| `make clear` | Clear all caches |
| `make clean-cache` | Remove empty rows |
| `make top20` | Quick smoke test (20 assets) |

### Options & Backtesting

| Command | Description |
|---------|-------------|
| `make run` | Options screener (uses tickers.csv) |
| `make backtest` | Run strategy backtest |
| `make top50` | Rank by 3Y revenue CAGR |
| `make bagger50` | Rank by 100× Bagger Score |
| `make top100` | Top 100 screener |

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

## Scientific Foundation

### Posterior Predictive Distribution

For current regime r_t, the predictive distribution is:

```
p(x | r_t) = Σ_m p(x | r_t, m, θ_{r_t,m}) · p(m | r_t)
```

This is computed via **Monte Carlo**:
1. Sample models proportional to posterior weights
2. For each model, simulate forward paths
3. Concatenate samples → full mixture distribution

### Model Selection

Weights are computed from **BIC** with optional **Hyvärinen score** blending:

```
w_raw(m|r) = exp(-0.5 × (BIC_{m,r} - BIC_min))
w_combined(m) = w_bic(m)^α × w_hyvarinen(m)^(1-α)
```

Hyvärinen score is Fisher-consistent under misspecification and naturally rewards tail accuracy.

### Temporal Smoothing

Model posteriors evolve smoothly over time:

```
w_smooth(m|r) = w_prev(m|r)^α × w_raw(m|r)
```

This prevents erratic model switching while allowing adaptation.

### Hierarchical Shrinkage

When regime r has insufficient samples:

```
p(m|r) → p(m|global)
θ_{r,m} → θ_{global,m}
```

Marked as `borrowed_from_global = True` for transparency.

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

---

## Philosophy

> "Act only on beliefs that were actually learned."

This system is a **belief evolution engine**. It maintains competing hypotheses about market dynamics and lets Bayesian inference arbitrate between them.

When evidence is weak:
- The system becomes more ignorant, not more confident
- It reverts to higher-level posteriors, not point estimates
- It never invents beliefs

The goal is **calibrated uncertainty**, not false precision.

---

<p align="center">
  <sub>Built with scientific rigor and engineering craftsmanship.</sub>
</p>
