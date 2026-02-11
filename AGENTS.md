# AGENTS.md - AI Coding Agent Guidelines

# ⚠️ TERMINAL RULES - CRITICAL
# 1. NEVER use heredoc (<<EOF or <<'EOF') in terminal - causes garbled output
# 2. NEVER put multiline code directly in terminal commands
# 3. For multiline code: CREATE A FILE first, then run the file
# 4. For shell scripts: Write to a .sh file, then execute it

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

### Active Experimental Models (Generation 18 - Feb 2026)
All experimental models have been evaluated and promoted to safe storage.
No active experimental models - ready for new generation development.

**Best Archived (Safe Storage - 10 models):**
- `elite_hybrid_omega2` (Final: 68.82, CSS: 0.74, FEC: 0.87, +10.7 vs STD) - **BEST Gen18**
- `optimal_hyv_iota` (Final: 72.27, CSS: 0.84, FEC: 0.88, +13.9 vs STD) - **BEST CSS/FEC**
- `dtcwt_qshift` (Final: 63.98, BIC: -33700, +7.2 vs STD) - **Q-shift champion**
- `dtcwt_magnitude_threshold` (Final: 63.94, CSS: 0.73, FEC: 0.85, +7.1 vs STD)
- `dualtree_complex_wavelet` (Final: 63.90, CSS: 0.77, +7.1 vs STD) - **Core DTCWT**
- `hyv_aware_eta` (Final: 62.94, Hyv: **294**, +4.6 vs STD) - **BEST Hyvärinen**
- `elite_hybrid_eta` (Final: 62.29, CSS: 0.69, FEC: 0.84, +5.5 vs STD)
- `dtcwt_adaptive_levels` (Final: 62.24, +5.4 vs STD)
- `dtcwt_vol_regime` (Final: 61.44, CSS: 0.66, FEC: 0.80, +4.6 vs STD) - **Passes ALL hard gates**
- `stress_adaptive_inflation` (Final: 59.78, Hyv: 902, +3.0 vs STD)

**Key Mathematical Techniques:**
1. Q-shift filters: Near-shift-invariance with quarter-sample delay
2. Memory-smoothed deflation: `defl_t = 0.85 * defl_{t-1} + 0.15 * instant_t`
3. Hyvärinen control: `H = 0.5 s² - 1/σ²`, target H ≈ -500
4. Magnitude thresholding: Soft threshold at `median + k*σ`
5. Hierarchical stress: Multi-horizon weighted stress aggregation

**Hard Gates Target:**
- Final > 70, BIC < -29000, CRPS < 0.020, Hyv < 1000
- PIT: PASS, CSS > 0.65, FEC > 0.80, vs STD > +8%

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

## Structural Backtest Arena (Behavioral Validation)

**Location**: `src/arena/backtest_*.py`

A "court of law" for models — behavioral validation with full diagnostics.
Financial metrics are OBSERVATIONAL ONLY, never used for optimization.

### Non-Optimization Constitution
1. **Separation of Powers**: Backtest tuning isolated from production tuning
2. **One-Way Flow**: Tuning → Backtest → Integration Trial (no reverse dependency)
3. **Behavior Over Performance**: Diagnostics inform safety decisions
4. **Representativeness**: 50 tickers across sectors, caps, and regimes

### Backtest Universe (50 tickers)
Covers all major sectors and market caps:
- Technology: AAPL, MSFT, NVDA, GOOGL, CRM, ADBE, CRWD, NET
- Finance: JPM, BAC, GS, MS, SCHW, AFRM
- Defence: LMT, RTX, NOC, GD
- Healthcare: JNJ, UNH, PFE, ABBV, MRNA
- Industrials: CAT, DE, BA, UPS, GE
- Energy: XOM, CVX, COP, SLB, OXY
- Materials: LIN, FCX, NEM, NUE
- Consumer: AMZN, TSLA, HD, NKE, SBUX, PG, KO, PEP, COST
- Communication: META, NFLX, DIS, SNAP

### Diagnostics (Read-Only)
**Financial** (observational):
- Cumulative PnL, CAGR, Sharpe, Sortino
- Max drawdown, drawdown duration
- Profit factor, hit rate

**Behavioral** (primary for decisions):
- Equity curve convexity
- Tail loss clustering
- Regime stability
- Leverage sensitivity
- Turnover distribution

**Cross-Asset**:
- Performance dispersion across tickers
- Drawdown correlation
- Sector-specific fragility
- Crisis-period amplification

### Decision Outcomes
- `APPROVED` — Passed all behavioral safety gates
- `RESTRICTED` — Passed with caveats (sector/regime limits)
- `QUARANTINED` — Needs observation period before retry
- `REJECTED` — Failed behavioral gates, no promotion

### Key Files
| File | Purpose |
|------|---------|
| `backtest_config.py` | Universe definition, thresholds, constitution |
| `backtest_data.py` | Data pipeline for 64-ticker universe |
| `backtest_tune.py` | Parameter tuning for fairness (not optimization) |
| `backtest_engine.py` | Backtest execution with SignalFields → Geometry flow |
| `backtest_cli.py` | Command-line interface |
| `signal_fields.py` | Epistemic fields contract (direction, confidence, stability, etc.) |
| `signal_geometry.py` | Converts SignalFields to trading actions with proper sizing |
| `backtest_models/` | Models for backtest arena (copy from safe_storage) |

### Signal Geometry System (Critical Architecture)

**Location**: `src/arena/signal_fields.py`, `src/arena/signal_geometry.py`

Models are NOT traders. They are **distributional geometry estimators**.
The Signal Geometry layer properly interprets epistemic fields.

**Key Principles (Professor Wang, Chen, Liu):**
1. **Direction is SYNTHESIZED, not read**: `direction_score = f(asymmetry, momentum, stability)`
2. **Size comes from CONFIDENCE, not direction**: Capital authority ≠ Polarity authority
3. **Stability gates everything**: Unstable environments → no trade

**SignalFields Contract** (`signal_fields.py`):
```python
@dataclass
class SignalFields:
    direction: float      # Price momentum validated by Kalman (-1 to +1)
    asymmetry: float      # Distribution skewness
    belief_momentum: float # Rate of belief change
    confidence: float     # Model's certainty (0 to +1)
    stability: float      # Regime stability (-1 to +1)
    regime_fit: float     # How well data fits model assumptions
    tail_risk_left: float # Left tail (downside) risk
    tail_risk_right: float # Right tail (upside) potential
```

**GeometryDecision Output** (`signal_geometry.py`):
```python
@dataclass
class GeometryDecision:
    action: TradeAction   # ALLOW_LONG, ALLOW_SHORT, DENY_ENTRY, HOLD, etc.
    direction: int        # +1 or -1
    position_size: float  # From confidence × regime_fit, NOT direction
    direction_score: float # Synthesized direction strength
    size_authority: float # Capital authority from confidence
```

**Current Configuration (Long-Only Mode):**
- Markets have positive drift → shorts destroy CAGR
- Trend strength filter requires 2+ confirming factors
- Direction threshold: 0.10, Confidence threshold: 0.30
- Base position: 50%, Max position: 100%

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

### Signal Geometry Layer

**Location**: `src/arena/signal_fields.py`, `src/arena/signal_geometry.py`

The SignalFields → SignalGeometry architecture separates model outputs from trading decisions:

1. **Model.filter()** → mu, sigma (distributional estimates)
2. **SignalFields** → epistemic measures (direction, confidence, stability)
3. **SignalGeometry** → trading actions (entry, sizing, exit)

**Critical Insight (Feb 2026)**:
- Use **mu MOMENTUM** (change in mu), NOT mu level for direction
- Analysis showed: mu momentum Sharpe = 1.293 vs mu level Sharpe = 0.294
- Strong filtering destroys edge — be selective but not too aggressive

**Optimal Thresholds (Feb 2026)**:
```
min_direction_strength: 0.50  # Only strong momentum changes
min_confidence: 0.60          # High consistency required
Result: Sortino +4.63, Max DD 3.8%, near breakeven PnL
```

**Key Files**:
| File | Purpose |
|------|---------|
| `signal_fields.py` | Convert Kalman outputs to epistemic fields |
| `signal_geometry.py` | Gate + size trades based on field quality |

## Code Conventions