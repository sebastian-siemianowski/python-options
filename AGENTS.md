# AGENTS.md - AI Coding Agent Guidelines

## Project Overview

A **quantitative signal engine** using Bayesian Model Averaging (BMA) with Kalman filtering. Generates trading signals for 100+ assets with calibrated uncertainty estimates.

**Core philosophy**: "Heavy tails, asymmetry, and momentum are *hypotheses*, not certainties." All models compete via BIC weights.

## Architecture

```
DATA (Yahoo) → TUNING (MLE + BMA) → SIGNALS (Posterior Predictive Monte Carlo)
```

### Key Components

| Directory | Purpose | Entry Point |
|-----------|---------|-------------|
| `src/tuning/` | Fit 14 models per regime, compute BMA weights | `tune.py`, `tune_ux.py` |
| `src/decision/` | Generate signals from fitted models | `signals.py`, `signals_ux.py` |
| `src/models/` | Distribution implementations (Student-t, Skew-t, NIG, GMM) | `model_registry.py` |
| `src/calibration/` | PIT calibration, online updates, escalation | `pit_calibration.py` |
| `src/ingestion/` | Yahoo Finance data fetching + caching | `data_utils.py` |

### Data Flow

1. **Prices**: `src/data/prices/{SYMBOL}_1d.csv` (OHLCV from Yahoo)
2. **Tuning cache**: `src/data/tune/{SYMBOL}.json` (per-asset model parameters)
3. **Signals**: `src/data/high_conviction/buy/` and `sell/` (JSON output)

## Critical Patterns

### Model Registry (Single Source of Truth)
**Location**: `src/models/model_registry.py`

Both `tune.py` and `signals.py` MUST use the registry. This prevents the #1 failure mode: model name mismatch → dropped from BMA silently.

```python
from models.model_registry import get_active_models, ModelSpec
```

### Regime Classification (5 regimes)
Must be **identical** in tuning and inference:
- `LOW_VOL_TREND`, `HIGH_VOL_TREND`, `LOW_VOL_RANGE`, `HIGH_VOL_RANGE`, `CRISIS_JUMP`

Defined in both `tune.py` and `signals.py` via `assign_regime_labels()`.

### Momentum Augmentation
Enabled by default since Feb 2026 (94.9% selection rate). Models compete via BMA:
- Base: `kalman_gaussian`, `phi_student_t_nu_8`, etc.
- Augmented: `kalman_gaussian_momentum`, `phi_student_t_nu_8_momentum`, etc.

Use `--disable-momentum` flag for ablation testing.

## Developer Workflow

### Daily Commands
```bash
make setup     # First-time setup (venv + deps + data)
make stocks    # Refresh prices + generate signals
make tune      # Re-estimate model parameters (weekly)
make tests     # Run test suite
```

### Testing Patterns
```bash
make top20              # Quick smoke test (20 assets)
make calibrate          # Re-tune only failing assets
make calibrate-four     # Test 4 random failing assets
```

Tests are in `src/tests/test_*.py`. Run single test:
```bash
.venv/bin/python -m unittest src.tests.test_momentum_augmented -v
```

### Cache Management
```bash
make show-q        # Display cached parameters
make clear-q       # Clear tuning cache
make cache-stats   # Cache statistics
```

### Arena Commands (Experimental Model Competition)
```bash
make arena              # Full workflow: data + tune + results
make arena-data         # Download benchmark data (12 symbols)
make arena-tune         # Run competition (standard vs experimental)
make arena-results      # Show latest competition results
make arena-safe-storage # Show archived models in safe storage
```

## Arena System (Experimental Models)

**Location**: `src/arena/`

Isolated sandbox for testing experimental models against production baselines.

### Benchmark Universe (12 symbols)
- Small Cap: UPST, AFRM, IONQ
- Mid Cap: CRWD, DKNG, SNAP  
- Large Cap: AAPL, NVDA, TSLA
- Index: SPY, QQQ, IWM

### Standard Models (Baselines)
`kalman_gaussian_momentum`, `kalman_phi_gaussian_momentum`, `phi_student_t_nu_{4,6,8,12,20}_momentum`

### Active Experimental Models (Generations 13-17 - Feb 2026)
60 models organized into multiple generations with **Hyvärinen-aware density control**:

**Key Mathematical Techniques (Family A - Hyvärinen Control):**
- Score-gradient clipping: `s_clip = s / max(1, |s|/c)`
- GAS-style variance: `σ² = ω + β σ²_{t-1} + α s²_{t-1}`
- Curvature-adaptive inflation: `σ² = σ²_{t-1}(1 + α max(0, ∂²ℓ))`
- Local Hyv minimization: `θ = argmin Σ H_i(θ)`
- Entropy-preserving inflation: `σ² ← σ² exp(αz), s.t. ΔH ≈ 0`

| Generation | Models | Focus | Description |
|------------|--------|-------|-------------|
| Gen13 | optimal_hyv_*, css_neghyv_* | Optimal Hyv | Q-shift filters + deflation regime |
| Gen14 | ultimate_* | Low Hyv | DISCRETE regime multipliers (Hyv ~1700-2000) |
| Gen15 | merged_elite_* | CSS/FEC | Memory-smoothed CSS + FEC factors |
| Gen16 | exact_clone_* | Clone | Exact dtcwt_vol_regime clone with CSS/FEC |
| **Gen17** | **hyv_aware_*** | **Elite Hyv** | **Score-function control (Hyv < 1000)** |

**Current Leaders:**
- `hyv_aware_eta` (Final: 62.94, Hyv: **294**, CSS: 0.62, FEC: 0.78, +4.6 vs STD) - **BEST Hyv!**
- `hyv_aware_gamma` (Final: 63.39, Hyv: **1193**, CSS: 0.54, FEC: 0.80, +4.6 vs STD)
- `optimal_hyv_iota` (Final: 72.68, Hyv: 4645, CSS: 0.84, FEC: 0.88, +13.9 vs STD) - Best FINAL

**Best Balanced (Safe Storage)**: `dtcwt_vol_regime` 
- Passes ALL hard gates: Hyv: -2392, CSS: 0.66, FEC: 0.80, FINAL: 61.44

### Scoring System (`src/arena/scoring/`)
Combined score using proper scoring rules:

- **BIC**: Bayesian Information Criterion (complexity penalty)
- **CRPS**: Continuous Ranked Probability Score (calibration + sharpness)
- **Hyvärinen**: H = 0.5 s² - 1/σ² (detects variance collapse, elite target: <1000)
- **PIT**: Probability Integral Transform (calibration quality)
- **CSS**: Calibration Stability Under Stress (stress-period calibration)
- **FEC**: Forecast Entropy Consistency (uncertainty coherence)
- **DIG**: Directional Information Gain (sign prediction value)

### Hard Gates (Non-Negotiable)
```
CSS >= 0.65    # Calibration must hold during stress
FEC >= 0.75    # Entropy must track market uncertainty
Hyv < 1000     # Elite: prevent variance collapse (target <500 for safe storage)
vs STD >= 3    # Must beat best standard by 3+ points
PIT >= 75%     # Distributional correctness
```

### Multiprocessing Support
Arena uses `ProcessPoolExecutor` for parallel model fitting:
```bash
make arena-tune ARGS="--symbols SPY"          # Parallel (default)
make arena-tune ARGS="--symbols SPY --no-parallel"  # Sequential
make arena-tune ARGS="--workers 4"            # Custom worker count
```

### Promotion Gate
Experimental model graduates if ALL criteria pass:
1. Final Score > best standard by >3 points
2. CSS >= 0.65 (calibration stability under stress)
3. FEC >= 0.75 (forecast entropy consistency)
4. PIT pass rate >= 75%
5. Not last in any category

## Code Conventions

### Import Structure
```python
# Scripts add paths explicitly for imports
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
```

### Configuration via Environment
- `OFFLINE_MODE=1`: Use cached data only (no Yahoo API calls)
- `TUNING_QUIET=1`: Suppress verbose output during tuning

### Rich Console for UX
All user-facing output uses Rich tables and panels. Follow patterns in `signals_ux.py` and `tune_ux.py`.

## Key Files to Reference

| When working on... | Reference these files |
|--------------------|----------------------|
| Adding a new distribution | `models/model_registry.py`, `models/phi_student_t.py` |
| Modifying signal logic | `decision/signals.py` (8600+ lines, search for section) |
| Calibration/PIT tests | `calibration/pit_calibration.py`, `calibration/pit_penalty.py` |
| Data fetching | `ingestion/data_utils.py` |
| New Make target | `Makefile` (fully documented with sections) |
| New experimental model | `arena/safe_storage/*.py` (standalone model files) |

## Technical Details

### Dependencies
Minimal set in `src/setup/requirements.txt`: yfinance, numpy, pandas, scipy, rich, numba, hmmlearn

### Performance
- Numba kernels in `models/numba_kernels.py` for hot paths
- Multiprocessing (not threads) for CPU-bound tuning
- Filter cache in `models/filter_cache.py` for repeated Kalman runs

### PIT Calibration
**Critical for Kelly sizing**: If P(r>0)=62% → 62% of outcomes should be positive.
- `ECE < 0.05` = well-calibrated
- `pit_ks_pvalue < 0.05` = needs recalibration

# Important
- Avoid using cmdand heredoc in the terminal there are frequent errors when using heredoc
- Do not put raw multiline code in terminal - NEVER - NEVER
