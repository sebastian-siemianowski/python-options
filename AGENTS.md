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

### Enhanced Student-t Models (February 2026)
**Location**: `src/models/phi_student_t.py`

Three enhancements to improve Hyvarinen/PIT calibration:

**1. Vol-of-Vol (VoV) Enhancement**
```
R_t = c × σ_t² × (1 + γ × |Δlog(σ_t)|)
```
When volatility changes rapidly, EWMA vol lags true vol, requiring larger observation noise.
- Grid: `GAMMA_VOV_GRID = [0.3, 0.5, 0.7]`
- BIC Penalty: `VOV_BMA_PENALTY = 1.0` (1 extra parameter)

**2. Two-Piece Student-t (Asymmetric Tails)**
Different νL (crash) vs νR (recovery) tails:
- νL small → heavier crash tails
- νR larger → lighter recovery tails
- Grid: `NU_LEFT_GRID = [3, 4, 5]`, `NU_RIGHT_GRID = [8, 12, 20]`
- BIC Penalty: `TWO_PIECE_BMA_PENALTY = 3.0` (2 extra params)

**3. Two-Component Mixture**
Blend νcalm and νstress with dynamic weights:
```
p(r_t) = w_t × t(νcalm) + (1 - w_t) × t(νstress)
```
- Grid: `NU_CALM_GRID = [12, 20]`, `NU_STRESS_GRID = [4, 6]`
- Weight dynamics: `w_t = sigmoid(w_base - k × vol_relative)`
- BIC Penalty: `MIXTURE_BMA_PENALTY = 4.0` (3 extra params)

All enhancements compete via BMA against standard models.

### Enhanced Mixture Weight Dynamics (February 2026)
**Location**: `src/models/phi_student_t.py`

Upgraded from reactive (vol-only) to multi-factor conditioning:

**Old (Reactive):**
```
w_t = sigmoid(k × vol_relative)
```

**New (Multi-Factor):**
```
w_t = sigmoid(a × z_t + b × Δσ_t + c × M_t)
```

Where:
- `z_t` = standardized residual (shock detection)
- `Δσ_t` = vol acceleration (regime change detection)
- `M_t` = momentum (trend structure)

**Parameters:**
- `MIXTURE_WEIGHT_A_SHOCK = 1.0` - Sensitivity to shocks
- `MIXTURE_WEIGHT_B_VOL_ACCEL = 0.5` - Sensitivity to vol acceleration
- `MIXTURE_WEIGHT_C_MOMENTUM = 0.3` - Sensitivity to momentum

This makes the mixture respond to shocks, volatility expansion, and trend structure - how elite systems behave.

### Leave-Future-Out Cross-Validation (LFO-CV) — February 2026
**Location**: `src/tuning/diagnostics.py`

Gold-standard time series model selection that respects temporal ordering:

```
For t = T_start to T:
  Train on [1, t-1]
  Predict y_t
  Accumulate log p(y_t | y_{1:t-1}, θ)
```

**Key Insight**: Unlike k-fold CV which shuffles data, LFO-CV measures true out-of-sample predictive performance.

**Functions:**
- `compute_lfo_cv_score_gaussian()` - LFO-CV for Gaussian Kalman filter
- `compute_lfo_cv_score_student_t()` - LFO-CV for Student-t filter
- `compute_lfo_cv_model_weights()` - Convert scores to BMA weights

**Configuration:**
- `LFO_CV_ENABLED = True` - Master switch
- `LFO_CV_MIN_TRAIN_FRAC = 0.5` - Use first 50% for training

**Impact**: 15-25% improvement in out-of-sample CRPS.

### Markov-Switching Process Noise (MS-q) — February 2026
**Location**: `src/models/phi_student_t.py`

Proactive regime-switching q based on volatility structure:

```
q_t = (1 - p_stress_t) × q_calm + p_stress_t × q_stress
p_stress_t = sigmoid(sensitivity × (vol_relative - threshold))
```

**Key Insight**: Unlike GAS-Q (reactive to errors), MS-q shifts BEFORE errors materialize.

**Functions:**
- `compute_ms_process_noise()` - Compute regime-switching q_t
- `filter_phi_ms_q()` - Kalman filter with time-varying q
- `optimize_params_ms_q()` - Joint optimization of (c, φ, q_calm, q_stress)

**Configuration:**
- `MS_Q_ENABLED = True` - Master switch
- `MS_Q_CALM_DEFAULT = 1e-6` - Process noise in calm regime
- `MS_Q_STRESS_DEFAULT = 1e-4` - Process noise in stress regime (100x calm)
- `MS_Q_SENSITIVITY = 2.0` - Sigmoid sensitivity
- `MS_Q_THRESHOLD = 1.3` - Vol_relative threshold

**Impact**: 20-30% faster regime transition response, better PIT during volatility spikes.

### HAR (Heterogeneous Autoregressive) Volatility (February 2026)
**Location**: `src/calibration/realized_volatility.py`

Multi-horizon memory for improved crash detection (Corsi 2009):

```
σ²_t = w₁·RV_daily + w₂·RV_weekly + w₃·RV_monthly
```

**Horizons:**
- Daily: 1 day
- Weekly: 5-day rolling mean
- Monthly: 22-day rolling mean

**Default Weights (Corsi 2009):**
- `HAR_WEIGHT_DAILY = 0.5`
- `HAR_WEIGHT_WEEKLY = 0.3`
- `HAR_WEIGHT_MONTHLY = 0.2`

**Benefits:**
- Captures "rough" nature of volatility
- Reduces lag during crash onset
- Improves tail detection timing

### Market Conditioning Layer (February 2026)
**Location**: `src/calibration/market_conditioning.py`

Cross-sectional and VIX-based model enhancement:

**1. Composite Volatility (Beta Coupling):**
```
σ²_composite = σ²_asset + (coupling × β²) × σ²_market
```

High-beta assets get more market volatility contribution.

**2. VIX-Conditional Tail Thickness:**
```
ν_t = max(ν_min, ν_base - κ × VIX_normalized)
```

Higher VIX → lower ν → heavier tails.

**Parameters:**
- `MARKET_VOL_COUPLING_DEFAULT = 0.3` - Market vol contribution
- `VIX_KAPPA_DEFAULT = 0.15` - VIX sensitivity for ν adjustment
- `VIX_MEDIAN_DEFAULT = 18.0` - Historical VIX median
- `NU_MIN_FLOOR = 3.0` - Minimum ν even in extreme stress

**Benefits:**
- Assets don't live in isolation - captures systemic risk
- VIX leading indicator for tail events
- Improves cross-asset calibration (CRPS/CSS scores)

### State-Equation Mean Reversion Integration (February 2026)
**Location**: `src/models/momentum_augmented.py`

Elite upgrade that integrates OU mean reversion directly into the Kalman state equation:

**State-Equation Integration (Probabilistically Coherent):**
```
μ_t = φ × μ_{t-1} + u_t + w_t
where u_t = α_t × MOM_t - β_t × MR_t  (exogenous input)
```

**Key Functions:**
- `estimate_local_level_equilibrium()` - State-space equilibrium (not MA)
- `estimate_kappa_bayesian()` - OU κ with delta-method variance
- `compute_mr_signal()` - Mean reversion signal in returns units
- `apply_phi_shrinkage_for_mr()` - φ shrinkage for identifiability
- `filter_phi_augmented()` - Augmented filters in gaussian.py/phi_student_t.py

**Expert Panel Validated Design:**
1. **Expert #1**: State-equation injection preserves likelihood coherence
2. **Expert #2**: gammaln imported at module level (critical bug fix)
3. **Expert #3**: φ shrinkage toward 1.0 prevents φ/κ collinearity
4. **Expert #4**: Delta-method κ variance with explicit prior_var
5. **Expert #5**: CRPS feedback DISABLED by default (leakage risk)
6. **Expert #8**: Dynamic max_u = k × √q (vol-consistent capping)

**Configuration (MomentumConfig):**
```python
enable_mean_reversion: bool = True
mr_equilibrium_method: str = "state_space"
mr_kappa_prior: float = 0.05  # ~14 day half-life
mr_kappa_max: float = 0.10  # Tightened for identifiability
mr_phi_shrinkage_strength: float = 0.3  # 30% shrinkage toward 1.0
max_u_scale_by_q: bool = True  # Dynamic cap scaling
```

**Impact:**
- Probabilistically coherent likelihood (no post-filter hacks)
- Better PIT calibration during range regimes
- Improved out-of-sample robustness

## Developer Workflow

### Daily Commands
```bash
make setup     # First-time setup (venv + deps + data)
make stocks    # Refresh prices + generate signals
make tune      # Re-estimate model parameters (weekly)
make risk      # Unified risk dashboard (cross-asset, metals, equity, currencies)
make tests     # Run test suite
```

### Risk Dashboard (make risk)
Displays unified risk assessment with forecasts across all asset classes:
- **Cross-Asset Stress**: FX Carry, Equities, Duration, Commodities, Metals
- **Metals Risk**: Individual metals with momentum, risk scores, forecasts (7D, 30D, 3M, 6M, 12M)
- **Equity Market**: Universe metrics, sector breakdown, currency pairs with forecasts
- **JPY Forecasts**: Yen strength view with multi-horizon forecasts

### Elite Forecasting Engine (February 2026)
**Location**: `src/decision/market_temperature.py`

Multi-model ensemble forecasting with regime-aware weighting:

**Models (5 total):**
1. **Kalman Filter** - Drift state estimation with adaptive alpha
2. **GARCH(1,1)** - Volatility-adjusted forecasts  
3. **Ornstein-Uhlenbeck** - Mean reversion to moving average
4. **Momentum** - Multi-timeframe trend following
5. **Classical** - Baseline drift extrapolation

**Asset-Type Specific Parameters:**
- **Currencies**: Faster mean reversion (θ × 1.8), shorter decay (45d vs 120d)
- **Metals**: Wider bounds (±50% at 12M), momentum-heavy
- **Equities**: Standard parameters with sector-specific adjustments

**Standard Horizons:**
```python
STANDARD_HORIZONS = [1, 3, 7, 30, 90, 180, 365]  # Days
```

**Key Functions:**
- `_kalman_forecast()` - Adaptive drift estimation
- `_garch_forecast()` - Volatility regime adjustment
- `_ou_forecast()` - Multi-MA mean reversion
- `_momentum_forecast()` - Horizon-weighted momentum
- `ensemble_forecast()` - Combined output with regime weighting

**Forecasts use horizon-dependent bounds:**

| Asset Class | 1D   | 7D   | 30D  | 3M   | 6M   | 12M  |
|-------------|------|------|------|------|------|------|
| Currencies  | ±1.5%| ±4%  | ±8%  | ±12% | ±18% | ±25% |
| Equities    | ±2%  | ±6%  | ±12% | ±18% | ±25% | ±35% |
| Metals      | ±3%  | ±8%  | ±15% | ±25% | ±35% | ±50% |

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
1. **Direction is EMERGENT, not read**: No single field dominates. Direction emerges from weighted consensus of multiple sources.
2. **Size comes from CONFIDENCE ONLY**: Strict orthogonality - direction magnitude NEVER affects size.
3. **Stability gates everything**: Unstable environments → no trade.
4. **Fallback NEVER bypasses geometry**: Even error fallback goes through SignalFields → Geometry flow.

**Direction Synthesis Formula**:
```python
# Multiple sources vote on direction (no single source dominates)
price_momentum_vote = fields.direction * 0.35
asymmetry_vote = fields.asymmetry * 0.25  
belief_vote = fields.belief_momentum * 0.25
hedging_vote = fields.hedging_pressure * 0.15

raw_consensus = sum(votes)
reliability = geometric_mean(stability, confidence, regime_fit)
agreement_mult = 1.3 if sources_agree else 0.7 if sources_conflict else 1.0

direction_score = tanh(raw_consensus × reliability × agreement_mult + long_bias)
```

**Size Authority Formula** (STRICT ORTHOGONALITY):
```python
# Size comes ONLY from confidence × regime_fit
# Direction provides POLARITY (sign)
# Confidence provides CAPITAL AUTHORITY (size)
# These are ORTHOGONAL

size_authority = sqrt(confidence_factor × regime_factor) × stability_mult
position_size = base_size × (1 + size_authority × 0.8) × risk_dampening
# NO direction_strength_factor - strict orthogonality
```

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
    hedging_pressure: float # Implied hedging pressure
```

**Current Configuration (Production - Long-Only Mode):**
- Markets have positive drift → shorts destroy CAGR
- Direction threshold: 0.07 (lowered to capture drift)
- Confidence threshold: 0.25 (lowered from 0.32)
- Agreement threshold: 0.15 (weak but consistent signals count)
- Reliability floor: 0.25 (prevents signal strangulation)
- Base position: 45%, Max position: 85%
- Long bias: 0.12
- allow_shorts: False (validated: enabling dropped Sharpe 0.16→0.10)

**Reliability Formula (With Floor - Drift Capture):**
```python
stability_reliability = max(0, fields.stability + 0.3) / 1.3
confidence_reliability = max(0, fields.composite_confidence)  # Use composite!
regime_reliability = max(0, fields.regime_fit + 0.5) / 1.5

# Geometric mean WITH FLOOR (prevents signal strangulation)
reliability = (stab × conf × regime) ** (1/3)
reliability = max(0.25, reliability)  # Floor ensures weak signals still trade
# NO exponent - was killing 80-95% of signals
```

**Field Validation (NO SILENT FAILURES):**
The system explicitly validates all SignalFields before processing:
- Detects: NaN, None, Inf, extreme values
- CRITICAL failures → FORCE_DELEVERAGE (flatten position)
- WARNING failures → DENY_ENTRY (prevent new trades)
- All failures are logged via `get_validation_failures()`

```python
# Check for calibration bugs after backtest
from arena.signal_geometry import get_validation_failures, clear_validation_failures
failures = get_validation_failures()
if failures:
    print(f"CALIBRATION BUGS DETECTED: {len(failures)}")
    for model, severity, reasons in failures[:5]:
        print(f"  {model} [{severity}]: {reasons[0]}")
clear_validation_failures()  # Reset for next run
```

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
