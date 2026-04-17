# Architecture.md -- Modular Refactoring of tune.py and signals.py

## Problem Statement

`tune.py` (~7,450 lines) and `signals.py` (~11,900 lines) are monolithic files that violate
the Single Responsibility Principle. Each contains 12+ logically distinct subsystems entangled
through shared mutable state, feature flags, and deeply nested function calls.

**Consequences of the current state:**
- Merge conflicts on every feature branch (two engineers cannot touch tuning independently)
- 680 lines of feature-flag imports make cold-start understanding impossible
- Testing requires loading the entire 7,450/11,900-line module even for unit-level assertions
- Circular reasoning between sections (e.g., PIT calibration defined alongside GARCH fitting)
- No clear ownership boundaries -- a "calibration fix" touches the same file as a "regime change"

**What this refactoring is NOT:**
- NOT a rewrite. Every function body stays identical; only its address changes.
- NOT a feature change. BMA weights, PIT scores, MC paths must be bit-identical.
- NOT an optimization. Performance is unchanged (Numba kernels are already separate).

## Validation Universe (50 Assets)

Every epic must pass end-to-end validation on these 50 assets, confirming that tuning
parameters and generated signals are identical before and after the refactoring.

| Category | Symbols |
|----------|---------|
| Large Cap Tech | MSFT, GOOGL, AAPL, NVDA, AMZN, META |
| Large Cap Finance | JPM, GS, BAC |
| Large Cap Industrial | CAT, DE, BA |
| Large Cap Health | JNJ, UNH, PFE |
| Mid Cap Growth | CRWD, DKNG, PLTR, SQ, RKLB |
| Mid Cap Tech | SNOW, NET, SHOP |
| Small Cap Speculative | UPST, AFRM, IONQ, SOFI |
| High Vol / Meme | MSTR, TSLA |
| Broad Index | SPY, QQQ, IWM |
| Defence | LMT, RTX, NOC |
| Energy | XOM, CVX, COP |
| Consumer | HD, COST, PG |
| Precious Metals | GC=F, SI=F |
| Cryptocurrency | BTC-USD, ETH-USD |

## Equivalence Contract

For every epic, the acceptance criteria include an **Equivalence Gate**:

```
For each asset in the 50-asset universe:
  1. Run tuning BEFORE refactor, capture: BMA weights, model params, LFO-CV scores
  2. Run tuning AFTER refactor, capture: same outputs
  3. Assert: max |weight_before - weight_after| < 1e-10
  4. Assert: max |param_before - param_after| < 1e-10
  5. Run signal generation BEFORE, capture: Signal objects (all fields)
  6. Run signal generation AFTER, capture: Signal objects
  7. Assert: all numeric fields identical within 1e-8
  8. Assert: all categorical fields (label, regime) identical
```

This contract is non-negotiable. Any deviation means a bug was introduced.

## Target Architecture

### tune.py decomposition -> src/tuning/tune_modules/

```
src/tuning/tune_modules/
    __init__.py                  # Re-exports for backward compat
    config.py                    # Feature flags, constants, conditional imports (~680 lines)
    utilities.py                 # Logging, model classification, config retrieval (~500 lines)
    calibration_pipeline.py      # EMOS, DIG, calibration orchestration (~460 lines)
    process_noise.py             # tune_asset_q, cache management, complexity estimation (~730 lines)
    volatility_fitting.py        # GARCH MLE, OU fitting, cache I/O (~290 lines)
    kalman_wrappers.py           # Filter wrappers, PIT computation (~275 lines)
    pit_diagnostics.py           # Extended PIT metrics, predictive reconstruction (~275 lines)
    model_fitting.py             # fit_all_models_for_regime, elite diagnostics (~1,418 lines)
    regime_bma.py                # fit_regime_model_posterior, hierarchical shrinkage (~925 lines)
    bma_pipeline.py              # tune_regime_model_averaging, full orchestration (~398 lines)
    asset_tuning.py              # tune_asset_with_bma, multiprocessing dispatch (~692 lines)
    cli.py                       # Argument parsing, main() entry point (~46 lines)
```

### signals.py decomposition -> src/decision/signal_modules/

```
src/decision/signal_modules/
    __init__.py                  # Re-exports for backward compat
    config.py                    # Feature flags, conditional imports (~490 lines)
    volatility_imports.py        # Vol framework imports, calibration guards (~560 lines)
    regime_classification.py     # Asset type, regime inference, price utilities (~340 lines)
    momentum_features.py         # Momentum scoring, directional exhaustion (~460 lines)
    signal_dataclass.py          # Signal frozen dataclass definition (~95 lines)
    data_fetching.py             # fetch_px_asset, GARCH MLE, Student-t nu fitting (~380 lines)
    kalman_diagnostics.py        # Innovation whiteness, LL computation, regime priors (~265 lines)
    parameter_loading.py         # Load tuned params, select regime params (~570 lines)
    kalman_filtering.py          # _kalman_filter_drift core + gain monitoring (~760 lines)
    feature_pipeline.py          # compute_features orchestration (~1,136 lines)
    hmm_regimes.py               # HMM fitting, parameter stability tracking (~125 lines)
    walk_forward.py              # Walk-forward validation engine (~500 lines)
    signal_state.py              # Signal state persistence (~35 lines)
    pnl_attribution.py           # PnL decomposition (~80 lines)
    comprehensive_diagnostics.py # Unified diagnostic summary (~170 lines)
    monte_carlo.py               # Unified MC engine + path simulation (~1,100 lines)
    bma_engine.py                # bayesian_model_average_mc (~1,230 lines)
    threshold_calibration.py     # EU mapping, edge floors, confirmation logic (~270 lines)
    probability_mapping.py       # PIT calibration, EMOS correction, label mapping (~355 lines)
    signal_generation.py         # latest_signals main engine (~1,434 lines)
    asset_processing.py          # process_single_asset worker (~615 lines)
    cli.py                       # Argument parsing, main(), rendering dispatch (~380 lines)
```

### Backward Compatibility Layer

Both `tune.py` and `signals.py` remain as thin re-export shims:

```python
# tune.py (after refactoring)
"""Backward compatibility shim. All implementation in tune_modules/."""
from tuning.tune_modules import *  # noqa: F401,F403
```

This ensures that every external import (`from tuning.tune import tune_asset_with_bma`)
continues to work without modification.

---

## Epic 1: Extract tune.py Configuration & Utilities

**Goal**: Move the 680-line feature-flag import block and 500-line utility function block
into dedicated modules. This is the safest first step because these sections have no
mutable state and are pure functions / constants.

**Source**: tune.py Lines 321-1000 (config), Lines 1301-1795 (utilities)
**Target**: `src/tuning/tune_modules/config.py`, `src/tuning/tune_modules/utilities.py`

### Story 1.1: Create tune_modules package with config.py

**As an** architect maintaining the tuning pipeline,
**I want** all feature flags and conditional imports in a single config module,
**so that** I can understand what features are enabled without reading 680 lines of tune.py.

**File**: `src/tuning/tune_modules/__init__.py`, `src/tuning/tune_modules/config.py`

**Scope**:
- Create `src/tuning/tune_modules/` directory with `__init__.py`
- Move ALL feature flag constants: `MOMENTUM_AUGMENTATION_ENABLED`, `UNIFIED_STUDENT_T_ONLY`,
  `GAS_Q_ENABLED`, `ISOTONIC_RECALIBRATION_ENABLED`, `MARKET_CONDITIONING_ENABLED`,
  `ELITE_TUNING_ENABLED`, `RV_FEEDBACK_ENABLED`, `GH_DISTRIBUTION_ENABLED`,
  `TVVM_ENABLED`, `FILTER_RESULT_CACHE_ENABLED`, `EPIC7_ENABLED`
- Move ALL conditional try/except import blocks (30+ modules)
- Move ALL associated dataclasses: `GASQFitResult`
- Preserve the exact try/except pattern for graceful degradation

**Acceptance Criteria**:
- [x] `src/tuning/tune_modules/__init__.py` exists and re-exports all public symbols
- [x] `config.py` contains every feature flag previously in tune.py L321-1000
- [x] Every `try: from X import Y; FEATURE=True; except: FEATURE=False` block is preserved
- [x] `tune.py` imports from `tune_modules.config` instead of inline definitions
- [x] `GASQFitResult` dataclass is importable from `tune_modules.config`
- [x] No feature flag value changes (all booleans identical)
- [x] `grep -c "ENABLED" src/tuning/tune.py` decreases by at least 15
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 5 assets (SPY, NVDA, UPST, GC=F, BTC-USD) produce identical tune output

### Story 1.2: Extract utility functions to utilities.py

**As an** engineer debugging model classification,
**I want** utility functions in a dedicated module,
**so that** I can test `is_student_t_model()` without importing the entire tuning pipeline.

**File**: `src/tuning/tune_modules/utilities.py`

**Scope**:
- Move: `_is_quiet()`, `_log()`, `is_student_t_model()`, `is_heavy_tailed_model()`
- Move: `get_gh_config()`, `get_tvvm_config()`, `get_recalibration_config()`,
  `get_adaptive_nu_config()`
- Move: `compute_vol_proportional_q_floor()`, `apply_cross_asset_phi_pooling()`
- Each function depends only on config flags and numpy/scipy -- no circular deps

**Acceptance Criteria**:
- [x] All 10 utility functions importable from `tune_modules.utilities`
- [x] `_log()` still respects `TUNING_QUIET` env var
- [x] `is_student_t_model("phi_student_t_nu_8_momentum")` returns `True`
- [x] `compute_vol_proportional_q_floor()` produces identical output for all 5 regimes
- [x] Unit tests for each function pass in isolation (no tune.py import required)
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 5 assets produce identical tune output

### Story 1.3: Update tune.py to import from tune_modules

**As an** engineer working on the tuning pipeline,
**I want** tune.py to delegate to tune_modules for config and utilities,
**so that** tune.py shrinks by ~1,180 lines and the extraction is transparent.

**File**: `src/tuning/tune.py`

**Scope**:
- Replace L321-1000 with: `from tuning.tune_modules.config import *`
- Replace L1301-1795 with: `from tuning.tune_modules.utilities import *`
- Preserve line-level backward compatibility for all `from tuning.tune import X` usage

**Acceptance Criteria**:
- [x] tune.py shrinks by at least 1,100 lines
- [x] `from tuning.tune import MOMENTUM_AUGMENTATION_ENABLED` still works
- [x] `from tuning.tune import is_student_t_model` still works
- [x] `from tuning.tune import GASQFitResult` still works
- [x] No new imports added to any file outside tune.py and tune_modules/
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: All 50 assets produce identical tune output (full validation)

### Story 1.4: Write unit tests for extracted modules

**As an** engineer validating the refactoring,
**I want** focused unit tests for config.py and utilities.py,
**so that** future changes to feature flags or utilities are caught immediately.

**File**: `src/tests/test_tune_modules_config.py`, `src/tests/test_tune_modules_utilities.py`

**Scope**:
- Test that all feature flags are boolean
- Test that all conditional imports produce correct availability flags
- Test `is_student_t_model()` with 10+ model name variants
- Test `is_heavy_tailed_model()` boundary cases
- Test `compute_vol_proportional_q_floor()` with synthetic data
- Test `_log()` quiet/verbose modes

**Acceptance Criteria**:
- [x] At least 20 test cases across both test files
- [x] Tests run in < 5s (no model fitting, no data fetching)
- [x] Tests are self-contained (no dependency on price data or cache files)
- [x] 100% branch coverage for `is_student_t_model()` and `is_heavy_tailed_model()`
- [x] All tests pass on clean checkout

---

## Epic 2: Extract tune.py Calibration, Process Noise & Volatility Fitting

**Goal**: Move the middle layers of tune.py -- calibration pipeline (EMOS, DIG), process noise
optimization (tune_asset_q), and volatility fitting (GARCH, OU) -- into dedicated modules.
These are self-contained computational blocks with clean input/output contracts.

**Source**: tune.py Lines 1949-2536 (calibration), Lines 2538-3310 (process noise),
Lines 3313-3749 (volatility + cache I/O)
**Target**: `tune_modules/calibration_pipeline.py`, `tune_modules/process_noise.py`,
`tune_modules/volatility_fitting.py`

### Story 2.1: Extract calibration pipeline to calibration_pipeline.py

**As an** engineer working on EMOS or DIG calibration,
**I want** calibration logic in its own module,
**so that** I can iterate on calibration without risk of breaking model fitting.

**File**: `src/tuning/tune_modules/calibration_pipeline.py`

**Scope**:
- Move: `_crps_normal_single()`, `_emos_crps_objective()`, `train_emos_parameters()`
- Move: `compute_vol_calibration_ratios()`, `compute_dig_per_model()`,
  `compute_dig_weight()`, `adjust_bma_weights_with_dig()`
- Move: `run_calibration_pipeline()`, `save_calibration_report()`
- Move: `apply_regime_q_floor()` (regime-conditional process noise floor)
- Dependencies: numpy, scipy.optimize.minimize_scalar, config flags

**Acceptance Criteria**:
- [x] All 10 functions importable from `tune_modules.calibration_pipeline`
- [x] `train_emos_parameters()` produces identical (alpha, beta) for synthetic data
- [x] `compute_dig_per_model()` returns same DIG scores for fixed model predictions
- [x] `adjust_bma_weights_with_dig()` does not change weight sum (still normalizes to 1.0)
- [x] `run_calibration_pipeline()` orchestration unchanged
- [x] `apply_regime_q_floor()` returns identical q floors for all 5 regimes
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: 10 assets (SPY, AAPL, NVDA, JPM, TSLA, UPST, GC=F, BTC-USD, QQQ, MSTR)

### Story 2.2: Extract process noise tuning to process_noise.py

**As an** engineer debugging q optimization,
**I want** process noise logic isolated,
**so that** I can trace tune_asset_q without wading through GARCH or calibration code.

**File**: `src/tuning/tune_modules/process_noise.py`

**Scope**:
- Move: `tune_asset_q()` (main q optimization, ~560 lines)
- Move: `load_asset_list()`, `compute_price_data_hash()`, `get_last_price_date()`
- Move: `needs_retune()`, `stamp_tune_result()`
- Move: `estimate_tuning_complexity()`, `sort_assets_by_complexity()`,
  `get_optimal_worker_count()`
- Dependencies: os, hashlib, numpy, config flags (GAS_Q, RV_Q)

**Acceptance Criteria**:
- [x] All 9 functions importable from `tune_modules.process_noise`
- [x] `tune_asset_q()` returns identical q_optimal for 5 test assets
- [x] `compute_price_data_hash()` produces identical SHA256 for same price data
- [x] `needs_retune()` correctly detects stale caches (hash mismatch)
- [x] `sort_assets_by_complexity()` returns same ordering for fixed complexity scores
- [x] `get_optimal_worker_count()` respects CPU count
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: 10 assets produce identical q values

### Story 2.3: Extract volatility fitting to volatility_fitting.py

**As an** engineer working on GARCH or OU models,
**I want** volatility fitting in its own module,
**so that** I can add GJR-GARCH variants without touching the model fitting pipeline.

**File**: `src/tuning/tune_modules/volatility_fitting.py`

**Scope**:
- Move: `gjr_garch_log_likelihood()`, `garch_log_likelihood()`, `fit_garch_mle()`
- Move: `fit_ou_params()` (OU mean-reversion speed estimation)
- Move: `load_cache()`, `load_single_asset_cache()`, `save_cache_json()`
- Dependencies: scipy.optimize.minimize, numpy, json, os

**Acceptance Criteria**:
- [x] All 7 functions importable from `tune_modules.volatility_fitting`
- [x] `fit_garch_mle()` returns identical (omega, alpha, beta, gamma) for SPY returns
- [x] `fit_ou_params()` returns identical kappa for synthetic OU process
- [x] `load_cache()` / `save_cache_json()` round-trip preserves all keys and values
- [x] `load_single_asset_cache()` returns None for missing assets (not crash)
- [x] GARCH barrier constraint (lambda=5.0) preserved in optimization
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: 10 assets produce identical GARCH and OU parameters

### Story 2.4: Update tune.py imports for Epic 2 modules

**As an** engineer,
**I want** tune.py to delegate to the three new modules,
**so that** tune.py shrinks by another ~1,500 lines.

**File**: `src/tuning/tune.py`

**Scope**:
- Replace L1949-2536 with imports from `tune_modules.calibration_pipeline`
- Replace L2538-3310 with imports from `tune_modules.process_noise`
- Replace L3313-3749 with imports from `tune_modules.volatility_fitting`
- Verify all call sites in remaining tune.py code resolve correctly

**Acceptance Criteria**:
- [x] tune.py shrinks by at least 1,400 additional lines (cumulative 3,391 from original 7977 -> 4586)
- [x] `from tuning.tune import tune_asset_q` still works
- [x] `from tuning.tune import fit_garch_mle` still works
- [x] `from tuning.tune import train_emos_parameters` still works
- [x] No circular imports between tune_modules submodules
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: All 50 assets produce identical tune output

---

## Epic 3: Extract tune.py Kalman Wrappers & PIT Diagnostics

**Goal**: Move the filter wrapper layer and extended PIT diagnostic computation into
dedicated modules. These are the "measurement" functions that assess model quality --
logically distinct from the model fitting that produces the parameters.

**Source**: tune.py Lines 3751-3921 (Kalman wrappers), Lines 3923-4332 (PIT diagnostics)
**Target**: `tune_modules/kalman_wrappers.py`, `tune_modules/pit_diagnostics.py`

### Story 3.1: Extract Kalman filter wrappers to kalman_wrappers.py

**As an** engineer adding a new filter variant,
**I want** filter wrappers in a dedicated module,
**so that** I can add a new Kalman variant without risking the BMA pipeline.

**File**: `src/tuning/tune_modules/kalman_wrappers.py`

**Scope**:
- Move: `kalman_filter_drift()`, `kalman_filter_drift_phi()` (legacy stubs)
- Move: `compute_pit_ks_pvalue()`, `compute_pit_ks_pvalue_student_t()` (legacy stubs)
- Move: `optimize_q_mle()` (MLE for q with fixed c, phi)
- Move: `compute_predictive_pit_student_t()`, `compute_predictive_pit_gaussian()`
- Move: `compute_predictive_scores_student_t()`, `compute_predictive_scores_gaussian()`
- Dependencies: models/gaussian.py, models/phi_student_t.py, scipy.stats

**Acceptance Criteria**:
- [x] All 10 functions importable from `tune_modules.kalman_wrappers`
- [x] `optimize_q_mle()` returns identical q for fixed (c, phi) on synthetic data
- [x] `compute_predictive_pit_student_t()` returns PIT values in [0, 1]
- [x] `compute_predictive_scores_gaussian()` returns finite CRPS, log-loss, DIG
- [x] Legacy stubs still delegate correctly to models/* implementations
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: 5 assets produce identical PIT scores

### Story 3.2: Extract PIT diagnostics to pit_diagnostics.py

**As an** engineer improving calibration metrics,
**I want** PIT diagnostic computation in its own module,
**so that** I can add new calibration metrics without touching model fitting.

**File**: `src/tuning/tune_modules/pit_diagnostics.py`

**Scope**:
- Move: `reconstruct_predictive_from_filtered_gaussian()`
- Move: `compute_pit_from_filtered_gaussian()`
- Move: `compute_extended_pit_metrics_gaussian()` (~120 lines)
- Move: `compute_extended_pit_metrics_student_t()` (~200 lines)
- Dependencies: scipy.stats, numpy

**Acceptance Criteria**:
- [x] All 4 functions importable from `tune_modules.pit_diagnostics` (plus _fast_ks_uniform)
- [x] `compute_extended_pit_metrics_gaussian()` returns all 10+ diagnostic keys
- [x] `compute_extended_pit_metrics_student_t()` returns all 12+ diagnostic keys
- [x] `pit_mean_absolute_error` (ECE) identical to 1e-10 for fixed filtered states
- [x] `pit_ks_statistic` identical for fixed PIT arrays
- [x] Tail counts (`pit_extreme_low_count`, `pit_extreme_high_count`) identical
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: 10 assets produce identical extended PIT metrics

### Story 3.3: Update tune.py imports for Epic 3 modules

**As an** engineer,
**I want** tune.py to delegate Kalman wrappers and PIT diagnostics,
**so that** tune.py shrinks by another ~580 lines.

**File**: `src/tuning/tune.py`

**Scope**:
- Replace L3751-3921 with imports from `tune_modules.kalman_wrappers`
- Replace L3923-4332 with imports from `tune_modules.pit_diagnostics`
- Verify that `fit_all_models_for_regime()` still resolves all wrapper calls

**Acceptance Criteria**:
- [x] tune.py shrinks by at least 500 additional lines (cumulative 3,999 from original 7977 -> 3978)
- [x] `from tuning.tune import compute_extended_pit_metrics_student_t` still works
- [x] `from tuning.tune import compute_predictive_pit_gaussian` still works
- [x] No import cycles between kalman_wrappers.py and pit_diagnostics.py
- [x] All 4652+ existing tests pass (4720 passed, 18 skipped)
- [x] **Equivalence Gate**: All 50 assets produce identical tune output

---

## Epic 4: Extract tune.py Core Model Fitting & BMA Pipeline

**Goal**: Extract the computational heart of tune.py -- the 1,418-line model fitting
orchestrator and the 925-line regime BMA engine -- into dedicated modules. These are the
most complex and highest-risk extractions because they contain the deepest call chains
and most shared state. Epic 4 must be done AFTER Epics 1-3 because the extracted core
depends on config, utilities, kalman_wrappers, and pit_diagnostics already being modular.

**Source**: tune.py Lines 4334-5662 (model fitting), Lines 5664-6589 (regime BMA),
Lines 6191-6589 (BMA pipeline), Lines 6590-7450 (asset tuning + CLI)
**Target**: `tune_modules/model_fitting.py`, `tune_modules/regime_bma.py`,
`tune_modules/bma_pipeline.py`, `tune_modules/asset_tuning.py`, `tune_modules/cli.py`

### Story 4.1: Extract model fitting to model_fitting.py

**As an** engineer adding a new model to the BMA ensemble,
**I want** model fitting isolated from BMA weighting,
**so that** adding a model variant requires editing only the fitting module.

**File**: `src/tuning/tune_modules/model_fitting.py`

**Scope**:
- Move: `compute_elite_diagnostics()` (~190 lines of Hyvarinen + combined scoring)
- Move: `format_elite_status()` (~50 lines of diagnostic formatting)
- Move: `fit_all_models_for_regime()` (~1,040 lines -- the main model fitting loop)
- Move: Internal helpers `_cache_key()`, `_get_cached_or_filter()` (~100 lines)
- Dependencies: config flags, kalman_wrappers, models/*, numba_kernels, diagnostics

**Critical Constraint**: `fit_all_models_for_regime()` is the most complex function in the
codebase. It must be moved AS-IS with zero logic changes. The filter cache dict must remain
a local variable (not module-level mutable state).

**Acceptance Criteria**:
- [x] `fit_all_models_for_regime()` importable from `tune_modules.model_fitting`
- [x] Function signature unchanged: `(returns, vol, regime_label, n_samples, ...)`
- [x] Filter cache is local to each call (not leaking between assets)
- [x] `_get_cached_or_filter()` still uses Numba fast path for Gaussian + Student-t LFO-CV
- [x] `compute_elite_diagnostics()` returns identical combined scores
- [x] GAS-Q, RV-Q, momentum augmentation all activate correctly based on config flags
- [x] Model registry integration unchanged (all 14+ models compete)
- [x] All 4652+ existing tests pass (4720 passed, 18-19 skipped)
- [x] **Equivalence Gate**: 15 assets produce identical per-regime model fits
  (SPY, AAPL, NVDA, MSFT, TSLA, JPM, GS, UPST, MSTR, GC=F, BTC-USD, QQQ, CRWD, XOM, JNJ)

### Story 4.2: Extract regime BMA to regime_bma.py

**As an** engineer tuning the model selection method (BIC vs Hyvarinen vs combined),
**I want** the BMA weighting logic in its own module,
**so that** I can experiment with selection methods without touching model fitting.

**File**: `src/tuning/tune_modules/regime_bma.py`

**Scope**:
- Move: `fit_regime_model_posterior()` (~525 lines)
  - Includes: BIC weight computation, Hyvarinen weight computation, combined weighting
  - Includes: Temporal smoothing of posteriors
  - Includes: Hierarchical fallback to global when n_samples < threshold
  - Includes: Asymmetric PIT penalty integration
  - Includes: Elite tuning integration
- Move: `tune_regime_model_averaging()` (~398 lines)
  - Includes: Global model fitting orchestration
  - Includes: Per-regime fitting loop
  - Includes: Hierarchical shrinkage toward global
  - Includes: Result packaging
- Dependencies: model_fitting.fit_all_models_for_regime, config flags, diagnostics

**Acceptance Criteria**:
- [x] `fit_regime_model_posterior()` importable from `tune_modules.regime_bma`
- [x] `tune_regime_model_averaging()` importable from `tune_modules.regime_bma`
- [x] BMA weights sum to 1.0 for every regime (normalization preserved)
- [x] Temporal smoothing alpha parameter works identically
- [x] Hierarchical shrinkage factor `sf = 1 / (1 + lambda * min_samples / n)` preserved
- [x] Fallback to global posterior triggers at same n_samples threshold
- [x] Model selection method ('bic', 'hyvarinen', 'combined') all produce identical weights
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 15 assets produce identical BMA posteriors across all regimes

### Story 4.3: Extract asset tuning to asset_tuning.py

**As an** engineer configuring the multiprocessing dispatch,
**I want** per-asset tuning and worker management in a dedicated module,
**so that** I can adjust parallelism without touching the BMA math.

**File**: `src/tuning/tune_modules/asset_tuning.py`

**Scope**:
- Move: `tune_asset_with_bma()` (~690 lines -- per-asset entry point)
  - Includes: Price data loading, volatility computation, regime labeling
  - Includes: Cache load/save, isotonic recalibration integration
  - Includes: Error handling with traceback capture
- Move: `_tune_worker()` (~80 lines -- multiprocessing worker wrapper)
- Dependencies: regime_bma.tune_regime_model_averaging, volatility_fitting,
  process_noise, ingestion.data_utils

**Acceptance Criteria**:
- [x] `tune_asset_with_bma()` importable from `tune_modules.asset_tuning`
- [x] `_tune_worker()` importable from `tune_modules.asset_tuning`
- [x] Price fetching via `fetch_px` unchanged
- [x] Cache JSON path (`src/data/tune/{SYMBOL}.json`) unchanged
- [x] `--force` flag bypasses cache correctly
- [x] Isotonic recalibration applied when available
- [x] Error dict format unchanged: `{"asset": str, "error": str, "traceback": str}`
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: All 50 assets produce identical cache JSON files

### Story 4.4: Extract CLI to cli.py

**As an** engineer adding a new CLI flag,
**I want** argument parsing in a small dedicated module,
**so that** I can add flags without reading 7,000 lines.

**File**: `src/tuning/tune_modules/cli.py`

**Scope**:
- Move: CLI argument parsing (argparse setup)
- Move: `main()` function (orchestration: load assets, dispatch workers, report stats)
- Dependencies: asset_tuning._tune_worker, process_noise.load_asset_list

**Acceptance Criteria**:
- [x] `main()` importable from `tune_modules.cli`
- [x] All CLI flags work: `--assets`, `--force`, `--dry-run`, `--debug`, `--calibrate`
- [x] ProcessPoolExecutor dispatch unchanged
- [x] Summary statistics (tuned/failed/elapsed) identical
- [x] `python -m tuning.tune --help` still works
- [x] All 4652+ existing tests pass

### Story 4.5: Create tune.py backward-compatible shim

**As an** engineer with existing imports from tune.py,
**I want** tune.py to remain importable with all public symbols,
**so that** no external code breaks.

**File**: `src/tuning/tune.py`

**Scope**:
- Replace entire file with thin re-export shim (~50 lines)
- Import and re-export all public symbols from tune_modules subpackages
- Preserve `if __name__ == "__main__": main()` entry point

**Acceptance Criteria**:
- [x] tune.py is < 100 lines (down from ~7,450)
- [x] `from tuning.tune import tune_asset_with_bma` works
- [x] `from tuning.tune import fit_all_models_for_regime` works
- [x] `from tuning.tune import tune_regime_model_averaging` works
- [x] `from tuning.tune import LFO_CV_ENABLED` works
- [x] `from tuning.tune import MOMENTUM_AUGMENTATION_ENABLED` works
- [x] `python src/tuning/tune.py --assets SPY --force` works
- [x] `python src/tuning/tune_ux.py --assets SPY --force` works
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: All 50 assets produce bit-identical output

---

## Epic 5: Extract signals.py Configuration & Data Layer

**Goal**: Begin the signals.py decomposition by extracting the 490-line import/config block
and the 560-line volatility framework import block, plus the data fetching layer. These are
the safest sections to extract first -- pure configuration and stateless I/O.

**Source**: signals.py Lines 1-503 (config), Lines 510-1070 (vol imports),
Lines 1640-1760 (regime classification), Lines 2317-2694 (data fetching)
**Target**: `signal_modules/config.py`, `signal_modules/volatility_imports.py`,
`signal_modules/regime_classification.py`, `signal_modules/data_fetching.py`

### Story 5.1: Create signal_modules package with config.py

**As an** architect reviewing the signal pipeline,
**I want** all feature flags and conditional imports for signals in a single config module,
**so that** understanding the pipeline does not require scanning 490 lines of try/except blocks.

**File**: `src/decision/signal_modules/__init__.py`, `src/decision/signal_modules/config.py`

**Scope**:
- Create `src/decision/signal_modules/` directory with `__init__.py`
- Move: All standard library and basic dependency imports
- Move: Filter cache system conditional imports
- Move: Presentation layer (signals_ux) conditional imports
- Move: High-conviction storage path definitions
- Move: Signal chart generation imports
- Move: Risk temperature module imports
- Move: Data utility imports
- Move: Cache-only enforcement (`OFFLINE_MODE` logic)
- Preserve exact import guard patterns (try/except with stub fallbacks)

**Acceptance Criteria**:
- [x] `src/decision/signal_modules/__init__.py` exists
- [x] `config.py` contains every feature flag and guarded import from signals.py L1-503
- [x] Every `try: from X import Y; AVAILABLE=True; except: AVAILABLE=False` preserved
- [x] `OFFLINE_MODE` enforcement logic works identically
- [x] signals.py imports from `signal_modules.config` instead of inline
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 5 assets (SPY, NVDA, UPST, GC=F, BTC-USD) produce identical signals

### Story 5.2: Extract volatility framework imports to volatility_imports.py

**As an** engineer adding a new volatility model,
**I want** vol framework imports organized in one place,
**so that** I know exactly what vol models are available.

**File**: `src/decision/signal_modules/volatility_imports.py`

**Scope**:
- Move: Garman-Klass realized vol imports (HAR multi-horizon)
- Move: CRPS model selection imports (LFO-CV)
- Move: Enhanced Student-t imports (VoV, Two-Piece, Mixture)
- Move: EVT tail modeling imports (GPD)
- Move: Contaminated Student-t imports
- Move: Hansen Skew-t imports
- Move: Enhanced mixture weight imports
- Move: Markov-Switching q imports
- Move: Model registry imports
- Move: Risk temperature imports
- Move: GAS-Q and RV-Q imports
- Move: Unified risk context and crash risk imports
- Move: Epic 8 signal enrichment imports
- Preserve all stub fallback functions for unavailable modules

**Acceptance Criteria**:
- [x] All ~20 conditional import blocks moved with stubs intact
- [x] `HANSEN_SKEW_T_AVAILABLE`, `EVT_AVAILABLE`, `GAS_Q_AVAILABLE` etc. all correct
- [x] Stub functions (e.g., `hansen_skew_t_rvs` fallback) produce same output
- [x] No module becomes unavailable that was previously available
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 5 assets produce identical signals

### Story 5.3: Extract regime classification to regime_classification.py

**As an** engineer customizing asset-type behavior,
**I want** asset classification and price smoothing in a dedicated module,
**so that** adding a new asset class is a one-file change.

**File**: `src/decision/signal_modules/regime_classification.py`

**Scope**:
- Move: `classify_asset_type()` (symbol -> "equity"/"currency"/"metal"/"crypto")
- Move: `_compute_sig_h_cap()` (horizon-dependent return caps by asset class)
- Move: `_smooth_display_price()` (EMA-smoothed price display)
- Move: `clear_display_price_cache()` (reset intra-run smoothing)
- Move: `infer_current_regime()` (map features -> regime label)
- Move: `SECTOR_MAP`, `COMPANY_NAMES` constants (if defined in signals.py)
- Dependencies: numpy (pure functions)

**Acceptance Criteria**:
- [x] All 5 functions importable from `signal_modules.regime_classification`
- [x] `classify_asset_type("BTC-USD")` returns "crypto"
- [x] `classify_asset_type("GC=F")` returns "metal"
- [x] `classify_asset_type("AAPL")` returns "equity"
- [x] `_compute_sig_h_cap("equity", 30)` returns same cap as before
- [x] `infer_current_regime()` maps identical feature dicts to identical regime labels
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 10 assets produce identical regime assignments

### Story 5.4: Extract data fetching to data_fetching.py

**As an** engineer debugging price data issues,
**I want** data fetching logic isolated,
**so that** I can trace a "missing price" error without reading 11,900 lines.

**File**: `src/decision/signal_modules/data_fetching.py`

**Scope**:
- Move: `fetch_px_asset()` (Yahoo Finance + PLN conversion)
- Move: `_garch11_mle()` (GARCH MLE for signals context)
- Move: `_fit_student_nu_mle()` (Student-t nu estimation)
- Dependencies: yfinance, scipy.optimize, ingestion.data_utils

**Acceptance Criteria**:
- [x] All 3 functions importable from `signal_modules.data_fetching`
- [x] `fetch_px_asset("SPY")` returns identical price DataFrame
- [x] PLN conversion path works for currency pairs
- [x] `_garch11_mle()` returns identical (omega, alpha, beta) for fixed returns
- [x] `_fit_student_nu_mle()` returns identical nu for fixed returns
- [x] Error handling (missing data, API failures) unchanged
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 10 assets produce identical price data and GARCH fits

---

## Epic 6: Extract signals.py Kalman Filtering & Feature Pipeline

**Goal**: Extract the Kalman drift estimation core (~760 lines), feature computation pipeline
(~1,136 lines), and supporting modules (momentum, HMM, diagnostics). These form the
"inference engine" of signal generation -- the heaviest computational layer.

**Source**: signals.py Lines 1765-2220 (momentum), Lines 2696-2960 (Kalman diagnostics),
Lines 2961-3530 (param loading), Lines 3531-4462 (Kalman filtering),
Lines 4463-5598 (feature pipeline), Lines 5599-5804 (HMM + stability)
**Target**: `signal_modules/momentum_features.py`, `signal_modules/kalman_diagnostics.py`,
`signal_modules/parameter_loading.py`, `signal_modules/kalman_filtering.py`,
`signal_modules/feature_pipeline.py`, `signal_modules/hmm_regimes.py`

### Story 6.1: Extract momentum features to momentum_features.py

**As an** engineer tuning momentum timeframes,
**I want** momentum scoring and exhaustion detection in a dedicated module,
**so that** I can adjust momentum horizons without touching the Kalman filter.

**File**: `src/decision/signal_modules/momentum_features.py`

**Scope**:
- Move: `compute_momentum_score()` (~90 lines -- weighted 5d/21d/63d momentum)
- Move: `compute_directional_exhaustion_from_features()` (~300 lines -- multi-horizon EMA deviations)
- Move: `_compute_simple_exhaustion()` (~65 lines -- single-horizon exhaustion)
- Dependencies: numpy (pure numerical functions)

**Acceptance Criteria**:
- [x] All 3 functions importable from `signal_modules.momentum_features`
- [x] `compute_momentum_score()` returns identical scores for fixed price series
- [x] Exhaustion is dual-sided: ue_up and ue_down both computed correctly
- [x] `_compute_simple_exhaustion()` boundary: returns 0.0 for flat price series
- [x] EMA windows (5, 10, 21, 63) unchanged
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 10 assets produce identical momentum scores and exhaustion values

### Story 6.2: Extract Kalman diagnostics to kalman_diagnostics.py

**As an** engineer investigating filter quality,
**I want** innovation whiteness tests and log-likelihood computation in a dedicated module,
**so that** I can add new diagnostics without touching the filter loop.

**File**: `src/decision/signal_modules/kalman_diagnostics.py`

**Scope**:
- Move: `_test_innovation_whiteness()` (~90 lines -- Ljung-Box + Q-Q diagnostics)
- Move: `_compute_kalman_log_likelihood()` (~50 lines -- Gaussian LL)
- Move: `_compute_kalman_log_likelihood_heteroskedastic()` (~50 lines -- vol-adjusted LL)
- Move: `_estimate_regime_drift_priors()` (~70 lines -- Bayesian regime priors)
- Dependencies: scipy.stats, numpy

**Acceptance Criteria**:
- [x] All 4 functions importable from `signal_modules.kalman_diagnostics`
- [x] Ljung-Box test returns identical p-values for fixed innovation sequences
- [x] Log-likelihood values identical to 1e-10 for fixed filtered states
- [x] Regime drift priors identical for fixed regime returns
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 5 assets produce identical diagnostic dicts

### Story 6.3: Extract parameter loading to parameter_loading.py

**As an** engineer debugging "wrong regime params" issues,
**I want** parameter loading and regime selection in a dedicated module,
**so that** I can trace which model params were selected for which regime.

**File**: `src/decision/signal_modules/parameter_loading.py`

**Scope**:
- Move: `_safe_get_nested()` (~20 lines -- safe dict navigation)
- Move: `_load_tuned_kalman_params()` (~120 lines -- JSON cache loading)
- Move: `_select_regime_params()` (~200 lines -- regime-specific param selection)
- Move: Helper methods: `_is_student_t()`, `_is_momentum_augmented()`,
  `_get_base_model_name()`, `_is_valid_model()`
- Move: `_extract_best_model_params()` (~30 lines -- best single model fallback)
- Dependencies: json, os, model_registry

**Acceptance Criteria**:
- [x] All 7 functions importable from `signal_modules.parameter_loading`
- [x] `_load_tuned_kalman_params("SPY")` returns identical param dict
- [x] `_select_regime_params()` picks identical model for each regime
- [x] Hierarchical fallback to global works when regime has insufficient samples
- [x] `_is_student_t("phi_student_t_nu_8_momentum")` returns True
- [x] `_safe_get_nested()` returns default for missing keys (not crash)
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 10 assets produce identical regime param selections

### Story 6.4: Extract Kalman filtering core to kalman_filtering.py

**As an** engineer improving the Kalman filter,
**I want** the filter implementation in its own module,
**so that** I can modify filter logic without risk to MC sampling or signal generation.

**File**: `src/decision/signal_modules/kalman_filtering.py`

**Scope**:
- Move: `innovation_weight()` (~20 lines -- asymmetric down-weighting)
- Move: `_compute_kalman_gain_from_filtered()` (~40 lines)
- Move: `_apply_gain_monitoring_reset()` (~115 lines -- gain stall detection)
- Move: `_kalman_filter_drift()` (~760 lines -- main filter implementation)
  - Includes: initialization, forward pass, gain monitoring, model selection,
    RV-Q/GAS-Q adaptive q, Rauch-Tippett backward smoothing, diagnostics collection
- Dependencies: kalman_diagnostics, parameter_loading, models/*,
  calibration modules (GAS-Q, RV-Q)

**Critical Constraint**: `_kalman_filter_drift()` is the second most complex function
after `fit_all_models_for_regime()`. Must be moved AS-IS. All internal state must remain
local to the function call.

**Acceptance Criteria**:
- [x] `_kalman_filter_drift()` importable from `signal_modules.kalman_filtering`
- [x] Function signature unchanged
- [x] Outputs: mu_kf, var_kf, kalman_metadata all identical to 1e-10
- [x] Gain monitoring reset triggers at identical conditions
- [x] RV-Q and GAS-Q adaptive process noise paths work correctly
- [x] Student-t robust weighting path unchanged
- [x] Enhanced model selection (Gaussian/Student-t/enhanced) unchanged
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 15 assets produce identical filtered state trajectories

### Story 6.5: Extract feature pipeline to feature_pipeline.py

**As an** engineer adding a new feature to the pipeline,
**I want** feature computation in its own module,
**so that** adding a feature does not require reading 11,900 lines.

**File**: `src/decision/signal_modules/feature_pipeline.py`

**Scope**:
- Move: `compute_features()` (~1,136 lines -- comprehensive feature engineering)
  - Includes: data validation, EWMA drift/vol, HAR-GK volatility estimation
  - Includes: vol flooring, model-based drift loading, regime detection
  - Includes: Kalman filtering invocation, post-filter features (skew, kurtosis, momentum)
  - Includes: nu estimation, HMM regime fitting
- Dependencies: kalman_filtering._kalman_filter_drift, parameter_loading,
  momentum_features, hmm_regimes, volatility_imports

**Acceptance Criteria**:
- [x] `compute_features()` importable from `signal_modules.feature_pipeline`
- [x] Function signature unchanged: `(px, asset_symbol, ohlc_df, ...)`
- [x] Returns dict with all ~50 keys (px, ret, mu, vol, vol_regime, trend_z, nu_hat, etc.)
- [x] HAR-GK ONLY enforcement preserved (raises error if OHLC unavailable)
- [x] Vol flooring with lagged expanding quantile unchanged
- [x] HMM regime detection produces identical state sequences
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 15 assets produce identical feature dicts (all 50+ keys)

### Story 6.6: Extract HMM regimes to hmm_regimes.py

**As an** engineer improving regime detection,
**I want** HMM fitting and parameter stability tracking in a small module,
**so that** I can swap HMM for a different regime model without touching features.

**File**: `src/decision/signal_modules/hmm_regimes.py`

**Scope**:
- Move: `fit_hmm_regimes()` (~120 lines -- Gaussian HMM 3-state fitting)
- Move: `track_parameter_stability()` (~80 lines -- rolling GARCH drift tracking)
- Dependencies: hmmlearn (optional), numpy, scipy

**Acceptance Criteria**:
- [x] Both functions importable from `signal_modules.hmm_regimes`
- [x] HMM returns identical regime_series for fixed input data (seeded)
- [x] Parameter stability tracking returns identical drift metrics
- [x] Graceful fallback when hmmlearn unavailable
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 10 assets produce identical HMM regime assignments

---

## Epic 7: Extract signals.py Monte Carlo & BMA Engine

**Goal**: Extract the Monte Carlo simulation engine (~1,100 lines), BMA engine (~1,230 lines),
and supporting diagnostics. This is the highest-value extraction for signals.py because
the MC engine is the most frequently modified section and benefits most from isolation.

**Source**: signals.py Lines 6593-7376 (MC orchestration), Lines 7377-7585 (MC diagnostics),
Lines 7586-8814 (BMA engine), Lines 8455-8814 (forward path simulation)
**Target**: `signal_modules/monte_carlo.py`, `signal_modules/bma_engine.py`

### Story 7.1: Extract Monte Carlo engine to monte_carlo.py

**As an** engineer tuning MC path generation,
**I want** the MC simulation in its own module,
**so that** I can modify path dynamics without risk to probability calibration.

**File**: `src/decision/signal_modules/monte_carlo.py`

**Scope**:
- Move: `run_unified_mc()` (~485 lines -- main MC orchestration)
- Move: `run_regime_specific_mc()` (~205 lines -- deprecated per-regime MC)
- Move: `MCDiagnostics` dataclass (~35 lines)
- Move: `diagnose_mc_paths()` (~85 lines -- QA on path quality)
- Move: `compute_model_posteriors_from_combined_score()` (~90 lines -- score -> BMA weights)
- Move: `_simulate_forward_paths()` (~300 lines -- core path simulation)
  - Includes: GJR-GARCH with leverage, vol-of-vol, regime switching on observation variance,
    Markov-switching process noise, dual-frequency drift, per-step return capping
- Move: `shift_features()`, `make_features_views()` (utility functions)
- Dependencies: numpy, config flags, model parameters

**Critical Constraint**: `_simulate_forward_paths()` contains the most intricate numerical
code in the system. GJR-GARCH leverage effects, vol-of-vol noise, and Markov-switching q
must produce bit-identical paths. Seed handling must be preserved.

**Acceptance Criteria**:
- [x] `run_unified_mc()` importable from `signal_modules.monte_carlo`
- [x] `_simulate_forward_paths()` importable from `signal_modules.monte_carlo`
- [x] MC paths are bit-identical for fixed seed (seed=42 produces same cum_out matrix)
- [x] GJR-GARCH leverage asymmetry preserved (negative returns increase vol)
- [x] Vol-of-vol noise tier activated at correct thresholds
- [x] Markov-switching q transitions at same sigmoid sensitivity
- [x] Per-step return capping uses correct asset-class bounds
- [x] `MCDiagnostics` reports identical path statistics
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 15 assets produce identical MC path distributions (same seed)

### Story 7.2: Extract BMA engine to bma_engine.py

**As an** engineer improving the BMA mixture,
**I want** the Bayesian model averaging MC engine in its own module,
**so that** I can modify mixture composition without touching path simulation.

**File**: `src/decision/signal_modules/bma_engine.py`

**Scope**:
- Move: `bayesian_model_average_mc()` (~1,228 lines -- the full BMA MC engine)
  - Includes: Model selection and weight extraction
  - Includes: Per-model sample generation (z_drift, z_obs)
  - Includes: MC stepping loop with all enhancements
  - Includes: Anomaly detection and trimming
  - Includes: Quantile extraction (percentiles, E[gain]/E[loss])
  - Includes: Result packaging and caching
- Dependencies: monte_carlo._simulate_forward_paths, config flags, model parameters

**Acceptance Criteria**:
- [x] `bayesian_model_average_mc()` importable from `signal_modules.bma_engine`
- [x] Function signature unchanged
- [x] Returns identical `(cum_out, results_dict)` for fixed inputs and seed
- [x] BMA entropy floor prevents belief collapse (same threshold)
- [x] Model weight extraction from tuned_params unchanged
- [x] Quantile computation (CI bands) identical to 1e-10
- [x] E[gain] and E[loss] asymmetric computations identical
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 15 assets produce identical BMA distributions

### Story 7.3: Extract walk-forward validation to walk_forward.py

**As an** engineer running out-of-sample validation,
**I want** walk-forward logic in its own module,
**so that** I can run validation independently of signal generation.

**File**: `src/decision/signal_modules/walk_forward.py`

**Scope**:
- Move: `WalkForwardRecord` dataclass (~15 lines)
- Move: `WalkForwardResult` dataclass (~12 lines)
- Move: `run_walk_forward_backtest()` (~180 lines -- core expanding-window engine)
- Move: `walkforward_result_to_dataframe()` (~20 lines)
- Move: `save_walkforward_csv()` (~12 lines)
- Move: `_WFFeatureCache` class (~40 lines -- feature caching for efficiency)
- Move: `run_walk_forward_parallel()` (~70 lines -- multiprocessing wrapper)
- Dependencies: feature_pipeline.compute_features, bma_engine, numpy, pandas

**Acceptance Criteria**:
- [x] All classes and functions importable from `signal_modules.walk_forward`
- [x] Walk-forward engine produces identical CRPS, hit rate, coverage for fixed data
- [x] `_WFFeatureCache` reduces computation (no redundant feature calc)
- [x] Parallel wrapper dispatches correctly via ProcessPoolExecutor
- [x] CSV output format unchanged
- [x] All 4652+ existing tests pass (4719 passed, 19 skipped)
- [x] **Equivalence Gate**: 5 assets produce identical walk-forward results

---

## Epic 8: Extract signals.py Signal Generation & Output Layer

**Goal**: Extract the remaining signals.py sections -- the Signal dataclass, threshold
calibration, probability mapping, the main `latest_signals()` engine, asset processing
worker, small utility modules, and CLI. After this epic, signals.py becomes a thin shim.

**Source**: signals.py Lines 2222-2316 (Signal dataclass), Lines 6307-6592 (state + diagnostics),
Lines 8816-9437 (threshold + probability), Lines 9439-10873 (latest_signals),
Lines 10874-11897 (CLI + worker + rendering)
**Target**: `signal_modules/signal_dataclass.py`, `signal_modules/threshold_calibration.py`,
`signal_modules/probability_mapping.py`, `signal_modules/signal_generation.py`,
`signal_modules/signal_state.py`, `signal_modules/pnl_attribution.py`,
`signal_modules/comprehensive_diagnostics.py`, `signal_modules/asset_processing.py`,
`signal_modules/cli.py`

### Story 8.1: Extract Signal dataclass to signal_dataclass.py

**As an** engineer adding a new field to the Signal record,
**I want** the dataclass definition in a small, dedicated file,
**so that** the schema is easy to review and version-control.

**File**: `src/decision/signal_modules/signal_dataclass.py`

**Scope**:
- Move: `Signal` frozen dataclass (~95 lines, 80+ fields)
- Move: `StudentTDriftModel` helper class (static logpdf method, ~10 lines)
- Dependencies: dataclasses, math, scipy.special.gammaln

**Acceptance Criteria**:
- [x] `Signal` importable from `signal_modules.signal_dataclass`
- [x] `StudentTDriftModel` importable from same module (already in volatility_imports.py)
- [x] All 80+ fields present with identical types and defaults
- [x] `Signal` is frozen (immutable after construction)
- [x] `StudentTDriftModel.logpdf()` returns identical values for fixed inputs
- [x] All 4652+ existing tests pass (4719 passed, 19 skipped)
- [x] `from decision.signals import Signal` still works via shim

### Story 8.2: Extract threshold calibration to threshold_calibration.py

**As an** engineer tuning signal thresholds,
**I want** EU mapping and confirmation logic in a dedicated module,
**so that** I can adjust thresholds without touching probability calibration.

**File**: `src/decision/signal_modules/threshold_calibration.py`

**Scope**:
- Move: `composite_edge()` (~35 lines -- combine price momentum + confidence)
- Move: `optimal_threshold()` (~35 lines -- find Sharpe-maximizing threshold)
- Move: `compute_calibrated_thresholds()` (~20 lines -- static thresholds from tuned params)
- Move: `compute_dynamic_thresholds()` (~95 lines -- regime-adaptive adjustment)
- Move: `compute_adaptive_edge_floor()` (~25 lines -- prevent signal strangulation)
- Move: `apply_confirmation_logic()` (~60 lines -- multi-day confirmation + hysteresis)
- Dependencies: numpy, config flags

**Acceptance Criteria**:
- [x] All 6 functions importable from `signal_modules.threshold_calibration`
- [x] `composite_edge()` returns identical scores for fixed inputs
- [x] `compute_dynamic_thresholds()` adjusts correctly per regime
- [x] `apply_confirmation_logic()` hysteresis prevents flip-flopping
- [x] `compute_adaptive_edge_floor()` prevents signal strangulation (non-zero floor)
- [x] All 4652+ existing tests pass (4719 passed, 19 skipped)
- [x] **Equivalence Gate**: 10 assets produce identical threshold values

### Story 8.3: Extract probability mapping to probability_mapping.py

**As an** engineer improving calibration transport,
**I want** probability calibration in a dedicated module,
**so that** I can iterate on EMOS or isotonic maps without touching signal construction.

**File**: `src/decision/signal_modules/probability_mapping.py`

**Scope**:
- Move: `_load_signals_calibration()` (~15 lines -- load PIT calibration maps)
- Move: `_apply_single_p_map()` (~45 lines -- isotonic transport map)
- Move: `_apply_p_up_calibration()` (~105 lines -- full p_up calibration pipeline)
- Move: `_apply_emos_correction()` (~105 lines -- EMOS variance + location bias)
- Move: `_apply_magnitude_bias_correction()` (~25 lines -- sign-dependent correction)
- Move: `_get_calibrated_label_thresholds()` (~40 lines -- horizon-specific thresholds)
- Move: `label_from_probability()` (~20 lines -- map p_up to BUY/SELL/HOLD label)
- Dependencies: numpy, scipy.stats, config flags

**Acceptance Criteria**:
- [x] All 7 functions importable from `signal_modules.probability_mapping`
- [x] `_apply_p_up_calibration()` applies EMOS then isotonic in correct order
- [x] `label_from_probability()` maps identical (p_up, strength) to identical labels
- [x] EMOS correction preserves probability bounds [0, 1]
- [x] Isotonic transport map is monotonic (calibrated p_up order preserved)
- [x] `_get_calibrated_label_thresholds()` returns correct per-horizon thresholds
- [x] All 4652+ existing tests pass (4719 passed, 19 skipped)
- [x] **Equivalence Gate**: 10 assets produce identical calibrated probabilities and labels

### Story 8.4: Extract signal generation to signal_generation.py

**As an** engineer debugging signal construction,
**I want** the `latest_signals()` engine in its own module,
**so that** I can trace signal generation without reading 11,900 lines.

**File**: `src/decision/signal_modules/signal_generation.py`

**Scope**:
- Move: `latest_signals()` (~1,434 lines -- THE main signal generation engine)
  - Extract latest values (mu, vol, trend, z5 with 2-day stability)
  - Regime detection from vol + drift
  - Risk temperature (cross-asset stress scaling)
  - Unified MC invocation via `bayesian_model_average_mc()`
  - Probability mapping via calibration pipeline
  - Position sizing via EU framework
  - Exhaustion computation (ue_up/ue_down)
  - Signal object construction (all 80+ fields)
  - Constraint validation (CI ordering, vol monotonicity, coherent probabilities)
- Dependencies: bma_engine, monte_carlo, probability_mapping, threshold_calibration,
  momentum_features, signal_dataclass, config flags

**Critical Constraint**: `latest_signals()` is the longest function in the codebase. It
must be moved AS-IS with zero logic changes. All intermediate variables must remain local.

**Acceptance Criteria**:
- [x] `latest_signals()` importable from `signal_modules.signal_generation`
- [x] Function signature unchanged
- [x] Returns identical `(List[Signal], Dict[int, Dict])` for fixed inputs
- [x] All Signal fields are finite, monotonic variance, coherent probabilities
- [x] PIT violations trigger identical EXIT signals
- [x] Exhaustion computation produces identical ue_up/ue_down
- [x] All horizons returned in requested order
- [x] All 4652+ existing tests pass (4719 passed, 19 skipped)
- [x] **Equivalence Gate**: 15 assets produce identical Signal objects (all 80+ fields)

### Story 8.5: Extract small utility modules

**As an** engineer,
**I want** signal state persistence, PnL attribution, and comprehensive diagnostics
in their own tiny modules,
**so that** they can evolve independently.

**Files**: `src/decision/signal_modules/signal_state.py`,
`src/decision/signal_modules/pnl_attribution.py`,
`src/decision/signal_modules/comprehensive_diagnostics.py`

**Scope**:
- `signal_state.py`: Move `load_signal_state()`, `save_signal_state()` (~35 lines)
- `pnl_attribution.py`: Move `compute_pnl_attribution()` (~80 lines)
- `comprehensive_diagnostics.py`: Move `compute_all_diagnostics()` (~170 lines)
- Dependencies: json, os, numpy, feature_pipeline, walk_forward

**Acceptance Criteria**:
- [x] All 4 functions importable from their respective modules
- [x] Signal state round-trip (save then load) preserves all keys
- [x] PnL attribution decomposition sums correctly (drift + vol = total)
- [x] Comprehensive diagnostics includes HMM, parameter stability, walk-forward, PIT, model comparison
- [x] All 4652+ existing tests pass

### Story 8.6: Extract asset processing to asset_processing.py

**As an** engineer configuring per-asset processing,
**I want** the `process_single_asset()` worker in its own module,
**so that** I can modify enrichment logic without touching signal math.

**File**: `src/decision/signal_modules/asset_processing.py`

**Scope**:
- Move: `process_single_asset()` (~615 lines -- worker function)
  - Fetch price data (px + OHLC for GK vol)
  - `compute_features()` invocation
  - `latest_signals()` invocation
  - `_enrich_signal_with_epic8()` (conviction, Kelly, signal TTL)
  - Render output tables
  - Return results dict
- Dependencies: signal_generation, feature_pipeline, data_fetching, config flags

**Acceptance Criteria**:
- [x] `process_single_asset()` importable from `signal_modules.asset_processing`
- [x] Function signature unchanged
- [x] Returns identical results dict for fixed price data
- [x] Epic 8 signal enrichment (conviction, Kelly sizing, signal decay) unchanged
- [x] Error handling with graceful retry logic preserved
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: 10 assets produce identical enriched results

### Story 8.7: Extract CLI to cli.py

**As an** engineer adding a new CLI flag to signal generation,
**I want** argument parsing and orchestration in a small module,
**so that** adding a flag is a 5-line change.

**File**: `src/decision/signal_modules/cli.py`

**Scope**:
- Move: `parse_args()` (~30 lines)
- Move: `main()` (~116 lines -- orchestration, parallel dispatch, rendering)
- Move: Rendering function calls (sector summary, high-conviction storage,
  below-SMA50 analysis, risk dashboard, regime model summary)
- Move: `render_regime_model_summary()` (~120 lines)
- Dependencies: asset_processing, config flags, signals_ux

**Acceptance Criteria**:
- [x] `main()` importable from `signal_modules.cli`
- [x] All CLI flags work: `--horizons`, `--assets`, `--diagnostics`, `--from-cache`
- [x] ProcessPoolExecutor dispatch unchanged
- [x] All rendering functions produce identical output
- [x] `python src/decision/signals.py --assets SPY` still works via shim
- [x] All 4652+ existing tests pass

### Story 8.8: Create signals.py backward-compatible shim

**As an** engineer with existing imports from signals.py,
**I want** signals.py to remain importable with all public symbols,
**so that** no external code breaks.

**File**: `src/decision/signals.py`

**Scope**:
- Replace entire file with thin re-export shim (~50 lines)
- Import and re-export all public symbols from signal_modules subpackages
- Preserve `if __name__ == "__main__": main()` entry point

**Acceptance Criteria**:
- [x] signals.py is < 100 lines (down from ~11,900)
- [x] `from decision.signals import Signal` works
- [x] `from decision.signals import latest_signals` works
- [x] `from decision.signals import compute_features` works
- [x] `from decision.signals import bayesian_model_average_mc` works
- [x] `from decision.signals import process_single_asset` works
- [x] `python src/decision/signals.py --assets SPY` works
- [x] `python src/decision/signals_ux.py --assets SPY` works
- [x] All 4652+ existing tests pass
- [x] **Equivalence Gate**: All 50 assets produce bit-identical output

---

## Epic 9: End-to-End Validation & Integration Testing

**Goal**: Build a comprehensive test framework that validates the entire refactoring across
50 assets, ensuring tuning-to-signal bit-identical output, backward import compatibility,
and no performance regression.

### Story 9.1: Create E2E equivalence test framework

**As an** engineer validating the refactoring,
**I want** an automated framework that captures before/after snapshots of all pipeline outputs,
**so that** any deviation is caught before merge.

**File**: `src/tests/test_architecture_e2e.py`

**Scope**:
- Create `capture_tune_output(asset)` -> dict with BMA weights, model params, LFO-CV scores
- Create `capture_signal_output(asset)` -> dict with all Signal fields
- Create `assert_tune_equivalence(before, after, tolerance=1e-10)`
- Create `assert_signal_equivalence(before, after, tolerance=1e-8)`
- Use JSON serialization for snapshot storage
- Support `--update-snapshots` flag for baseline regeneration

**Acceptance Criteria**:
- [x] Framework can capture and compare tune output for any asset
- [x] Framework can capture and compare signal output for any asset
- [x] Tolerance is configurable (default 1e-10 for tune, 1e-8 for signal)
- [x] Categorical fields (label, regime) compared with exact equality
- [x] Framework reports WHICH field diverged and by how much
- [x] Snapshot files stored in `src/tests/snapshots/` (gitignored for size)

### Story 9.2: Tune pipeline E2E test (50 assets)

**As an** engineer,
**I want** to verify that all 50 assets produce identical tune output after refactoring,
**so that** I have confidence the extraction introduced no bugs.

**File**: `src/tests/test_architecture_e2e.py::TestTunePipelineEquivalence`

**Scope**:
- Run `tune_asset_with_bma()` for all 50 assets via both old and new import paths
- Compare: BMA weights per regime, model parameters (q, c, phi, nu),
  LFO-CV scores, PIT metrics, Hyvarinen scores
- Parameterized test: one test case per asset

**Acceptance Criteria**:
- [x] All 50 assets produce identical BMA weights (max diff < 1e-10)
- [x] All 50 assets produce identical model parameters (max diff < 1e-10)
- [x] All 50 assets produce identical LFO-CV scores (max diff < 1e-10)
- [x] All 50 assets produce identical PIT KS p-values (max diff < 1e-8)
- [x] Test runs in < 30 minutes (parallel)
- [x] Zero failures

### Story 9.3: Signal pipeline E2E test (50 assets)

**As an** engineer,
**I want** to verify that all 50 assets produce identical signals after refactoring,
**so that** I have confidence the signal math is preserved.

**File**: `src/tests/test_architecture_e2e.py::TestSignalPipelineEquivalence`

**Scope**:
- Run `process_single_asset()` for all 50 assets via both old and new import paths
- Compare: All 80+ Signal fields for all horizons
- Compare: Feature dict keys and values
- Parameterized test: one test case per asset

**Acceptance Criteria**:
- [x] All 50 assets produce identical Signal objects (all numeric fields within 1e-8)
- [x] All 50 assets produce identical labels (BUY/SELL/HOLD)
- [x] All 50 assets produce identical feature dicts (50+ keys)
- [x] MC paths are bit-identical when seeded
- [x] Test runs in < 45 minutes (parallel)
- [x] Zero failures

### Story 9.4: Cross-pipeline E2E test (tune -> signal chain)

**As an** engineer,
**I want** to verify the full chain (tune then signal) produces identical end-to-end output,
**so that** I can confirm the modules interact correctly.

**File**: `src/tests/test_architecture_e2e.py::TestCrossPipelineEquivalence`

**Scope**:
- For 10 representative assets: tune, then generate signals using tuned params
- Compare final Signal objects
- This catches integration bugs between tune_modules and signal_modules

**Acceptance Criteria**:
- [x] 10 assets (SPY, AAPL, NVDA, TSLA, JPM, UPST, GC=F, BTC-USD, QQQ, MSTR) pass
- [x] Tuned params from tune_modules feed correctly into signal_modules
- [x] No import errors when crossing module boundaries
- [x] Zero failures

### Story 9.5: Import compatibility test

**As an** engineer with existing code that imports from tune.py and signals.py,
**I want** automated tests that verify all public symbols are still importable,
**so that** no external code breaks.

**File**: `src/tests/test_architecture_imports.py`

**Scope**:
- Test every public symbol that was importable from `tuning.tune`
  (at least 30 symbols: functions, classes, constants)
- Test every public symbol that was importable from `decision.signals`
  (at least 40 symbols: functions, classes, constants)
- Test that direct module imports work (`from tuning.tune_modules.config import X`)
- Test that shim imports work (`from tuning.tune import X`)

**Acceptance Criteria**:
- [x] At least 70 import assertions (30 tune + 40 signals)
- [x] Every function, class, and constant that was public before is still importable
- [x] Both shim path and direct module path work for each symbol
- [x] Test runs in < 10 seconds (import-only, no computation)
- [x] Zero failures

### Story 9.6: Performance regression test

**As an** engineer,
**I want** to verify that the modular architecture does not introduce performance regression,
**so that** the refactoring is purely structural with no runtime cost.

**File**: `src/tests/test_architecture_performance.py`

**Scope**:
- Benchmark `tune_asset_with_bma("SPY")` via old and new paths (5 runs each)
- Benchmark `process_single_asset("SPY")` via old and new paths (5 runs each)
- Assert: new path is within 5% of old path (no significant slowdown)
- Measure import time for new module structure vs monolithic file

**Acceptance Criteria**:
- [x] Tune pipeline: new path within 5% of old path wall time
- [x] Signal pipeline: new path within 5% of old path wall time
- [x] Import time: new modules load in < 3 seconds (cold start)
- [x] No memory regression (RSS within 10% of baseline)
- [x] Results logged to `src/tests/performance_baseline.json`

---

## Execution Order

The epics must be executed in strict order due to dependencies:

```
Phase 1: tune.py decomposition (Epics 1-4)
  Epic 1 (config + utilities)     -- no deps, safest first
  Epic 2 (calibration + process noise + vol) -- depends on Epic 1 config
  Epic 3 (Kalman wrappers + PIT)  -- depends on Epic 1 utilities
  Epic 4 (core fitting + BMA + CLI) -- depends on Epics 1-3

Phase 2: signals.py decomposition (Epics 5-8)
  Epic 5 (config + data layer)    -- no deps on tune_modules
  Epic 6 (Kalman + features)      -- depends on Epic 5 config
  Epic 7 (MC + BMA engine)        -- depends on Epic 6 features
  Epic 8 (signal gen + CLI)       -- depends on Epics 5-7

Phase 3: Validation (Epic 9)
  Epic 9 (E2E tests)              -- depends on all epics complete
```

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Circular imports between tune_modules | Enforce DAG: config <- utilities <- calibration <- model_fitting <- regime_bma <- asset_tuning <- cli |
| Circular imports between signal_modules | Enforce DAG: config <- data_fetching <- parameter_loading <- kalman_filtering <- feature_pipeline <- monte_carlo <- bma_engine <- signal_generation <- cli |
| Mutable module-level state leaks | Audit: filter caches, display price caches must be function-local or explicitly reset between assets |
| Import performance regression | Lazy imports: heavy modules (numba, scipy) imported at call time, not module load time |
| External code breaks | Shim files (tune.py, signals.py) re-export all public symbols with `from X import *` |

### Definition of Done (All Epics)

The refactoring is complete when ALL of the following are true:

1. `tune.py` is < 100 lines (shim only)
2. `signals.py` is < 100 lines (shim only)
3. `src/tuning/tune_modules/` contains 12 modules totaling ~7,400 lines
4. `src/decision/signal_modules/` contains 22 modules totaling ~11,800 lines
5. All 4,652+ existing tests pass with zero modifications to test files
6. All 50 validation assets produce bit-identical output
7. No performance regression beyond 5%
8. No external import breaks (shim re-exports verified)
9. No circular imports (verified by `python -c "import tuning.tune_modules"`)
10. No mutable module-level state (verified by running 2 assets sequentially and confirming independence)
