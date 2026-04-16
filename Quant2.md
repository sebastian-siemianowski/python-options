# Quant2.md -- Profitability & Accuracy Transformation Backlog
# Version: 1.0 | Author: Product Owner (Quant) | Date: 2 April 2026
# Scope: tune.py, signals.py, frontend synchronization, validation harness

---

# EXECUTIVE SUMMARY

The current system produces flat, non-directional forecasts across all horizons:
`+0.0% | +0.0% | +0.1% | +0.7% | +0.6% | +2.2% | +2.9%` (1d through 12m).

Root cause analysis reveals **seven systemic failures** in the tune-to-signal pipeline:

1. **Drift signal is microscopic**: Kalman-extracted mu_t is ~0.04%/day for SPY. At H=1d,
   this rounds to +0.0%. The system treats daily drift as the *only* directional signal,
   ignoring momentum, mean-reversion, and cross-asset information.

2. **Phi decays drift instantly**: Standard models have phi=0.19-0.23, meaning drift halves
   every ~3 days. By H=21d, the AR(1) propagation has killed 99.9% of the starting drift.
   The forecast converges to the unconditional mean (zero) within a week.

3. **BMA concentrates on wrong models**: Unified models with 14+ parameters dominate BMA
   weights via lower BIC, but their GARCH + Student-t(nu=3) combination produced explosive
   MC paths (fixed in v7.9). The model selection never validated on out-of-sample forecasts.

4. **No walk-forward validation**: All models are fit and scored on the same data window.
   There is no genuine out-of-sample test for the BMA ensemble as a predictive system.

5. **EMOS calibration is a no-op**: The EMOS correction (a + b*mu, c + d*sig) was trained
   on empty data, producing identity transforms. The system has never been calibrated against
   realized returns.

6. **Regime classification is brittle**: Hard thresholds with no uncertainty produce abrupt
   regime switches that cascade into parameter discontinuities.

7. **Frontend shows different data than terminal**: Horizon labels, profit columns, exhaustion
   metrics, and conviction scores flow to the frontend but are not rendered or are misaligned.

This backlog contains **8 Epics** with **42 Stories** that address these failures systematically.
Every story includes acceptance criteria, test specifications, and targeted profitability metrics.

**Validation universe** (12 symbols, 3 tiers):
- Large Cap: AAPL, NVDA, MSFT, GOOGL
- Mid Cap/High Vol: TSLA, CRWD, DKNG, COIN
- Index/ETF: SPY, QQQ, GLD, TLT

**Success criteria** (measured on validation universe, 2-year backtest):
- Directional accuracy: >55% hit rate at H=7d (currently ~50%)
- Signal differentiation: >15% of signals are BUY/SELL (currently <5%)
- Sharpe ratio: >0.5 on equal-weight signal-following portfolio (currently ~0)
- CRPS improvement: >20% reduction vs current system
- Calibration: ECE < 0.03 across all horizons

---

# EPIC 1: DRIFT SIGNAL AMPLIFICATION & PERSISTENCE
# Priority: P0 (Critical) | Estimated Effort: 8 Stories | Files: tune.py, signals.py

## Problem Statement

The Kalman filter extracts a latent drift mu_t that is economically microscopic:
SPY mu_t ~ 0.04%/day, AAPL ~ 0.36%/day, NVDA ~ 1.28%/day. For the majority of
assets, mu_t at 1d rounds to +0.0% in the display.

Worse, the AR(1) drift propagation `mu_{t+k} = phi^k * mu_t` with phi in [0.19, 0.23]
for standard models (or phi=0.0 for floored unified models) means drift is effectively
zero beyond H=3d. The forecast is unconditional for all horizons >= 1 week.

The system currently uses *only* the Kalman posterior drift as directional signal.
It ignores:
- Multi-horizon momentum (computed in `compute_features()` but never used in MC)
- Mean-reversion signals (OU kappa estimated in tune.py but not propagated)
- Cross-asset momentum (no inter-asset information flow)
- Regime drift priors (estimated in `_estimate_regime_drift_priors()` but unused)

## Impact

Without a directional signal, the system cannot differentiate BUY from HOLD.
Every asset converges to the same +0.0% to +2.9% drift-free random walk.
This makes the system **incapable of generating trading alpha**.

---

### Story 1.1: Momentum-Augmented Drift Initialization [DONE]
**Priority**: P0 | **Points**: 8 | **File**: signals.py (lines 8269-9360, latest_signals)

#### Context
`compute_features()` (line 3775) already computes `momentum_score` -- a multi-horizon
(21/63/126/252d) composite. This score has Sharpe ~1.3 for direction prediction (validated
in arena analysis). But `latest_signals()` never uses it to initialize or augment mu_t_mc.

Currently, the MC starting drift is:
```python
mu_t_mc = feats["mu_post"].iloc[-1]  # Pure Kalman posterior
```

#### Requirements
1. Compute a momentum-derived drift adjustment:
   ```python
   # In latest_signals(), before MC call:
   mom_raw = feats.get("momentum_score", 0.0)
   vol_now = feats["vol"].iloc[-1]
   # Scale momentum to return units: mom_score * vol * scaling
   mom_drift = mom_raw * vol_now * MOM_DRIFT_SCALE  # MOM_DRIFT_SCALE ~ 0.1
   ```
2. Blend with Kalman drift using horizon-dependent weighting:
   ```python
   # Short horizons: trust Kalman (recent drift). Long horizons: trust momentum.
   w_mom = min(1.0, H / MOM_CROSSOVER_HORIZON)  # MOM_CROSSOVER_HORIZON ~ 63
   mu_t_mc_augmented = (1 - w_mom) * mu_t_mc + w_mom * mom_drift
   ```
3. Pass `mu_t_mc_augmented` to `bayesian_model_average_mc()` instead of raw `mu_t_mc`.
4. The blending weight must be configurable via constants at module top (not CLI args).
5. The momentum augmentation must be disabled when `momentum_score` is NaN or when
   fewer than 126 observations exist.

#### Acceptance Criteria
- [x] For NVDA (strong uptrend), H=7d forecast is > +1.0% (currently +0.14%)
- [x] For SPY (mild uptrend), H=7d forecast is in [+0.1%, +0.5%] (currently +0.04%)
- [x] For a flat/declining asset (e.g., MSFT at time of writing), H=7d forecast < SPY
- [x] momentum_score NaN does not crash -- falls back to pure Kalman drift
- [x] All existing tests pass without modification (1289 passed, 2 skipped)
- [x] New test: `test_momentum_drift_augmentation()` validates blending formula (20 tests pass)

#### Test Specification
```python
class TestMomentumDriftAugmentation(unittest.TestCase):
    def test_short_horizon_kalman_dominated(self):
        """At H=1, momentum weight should be < 0.02."""
        w_mom = min(1.0, 1 / 63)
        self.assertLess(w_mom, 0.02)

    def test_long_horizon_momentum_dominated(self):
        """At H=252, momentum weight should be 1.0."""
        w_mom = min(1.0, 252 / 63)
        self.assertEqual(w_mom, 1.0)

    def test_nan_momentum_fallback(self):
        """NaN momentum_score should produce pure Kalman drift."""
        # Mock feats with mom_score=NaN, verify mu_t_mc unchanged

    def test_directional_improvement_nvda(self):
        """NVDA augmented drift should exceed pure Kalman by >= 50%."""
        # Run with and without augmentation, compare magnitudes
```

---

### Story 1.2: Regime Drift Prior Integration [DONE]
**Priority**: P0 | **Points**: 5 | **File**: signals.py (lines 2285-2350, 8269-9360)

#### Context
`_estimate_regime_drift_priors()` (line 2285) computes per-regime historical mean returns.
For LOW_VOL_TREND (regime 0), this prior captures the average daily return during trending
periods (~0.08%/day for SPY). This information is computed but never flows into the MC
initialization.

The tune.py cache already stores per-regime model parameters, but the *drift level*
information is lost -- only (phi, q, c, nu) are stored per model, not a regime-specific
drift prior.

#### Requirements
1. In `_estimate_regime_drift_priors()`, compute regime-specific drift:
   ```python
   regime_drift_prior[r] = np.mean(returns[regime_labels == r])
   regime_drift_std[r] = np.std(returns[regime_labels == r]) / sqrt(n_r)
   ```
2. Store these in a `regime_drift_priors` dict within the features.
3. In `latest_signals()`, after regime assignment, blend the prior:
   ```python
   current_regime = assign_current_regime(...)
   prior_drift = regime_drift_priors.get(current_regime, 0.0)
   # Bayesian shrinkage: weight by relative precision
   tau_kalman = 1.0 / max(P_t, 1e-10)
   tau_prior = 1.0 / max(regime_drift_std[current_regime]**2, 1e-10)
   mu_t_shrunk = (tau_kalman * mu_t_mc + tau_prior * prior_drift) / (tau_kalman + tau_prior)
   ```
4. This shrinkage pulls extreme Kalman drifts toward regime-appropriate levels.
5. The shrinkage operates *before* momentum augmentation (Story 1.1).

#### Acceptance Criteria
- [x] SPY in LOW_VOL_TREND regime gets positive drift prior (~+0.05%/day)
- [x] SPY in CRISIS_JUMP regime gets negative drift prior (~-0.10%/day)
- [x] Shrinkage reduces drift variance across assets by >= 30%
- [x] Assets with very uncertain Kalman drift (large P_t) shrink more toward prior
- [x] New test: `test_regime_drift_prior_shrinkage()` validates Bayesian update (11 tests pass)

---

### Story 1.3: Adaptive Phi Estimation with Cross-Asset Pooling [DONE]
**Priority**: P1 | **Points**: 8 | **File**: tune.py (lines 3378-4514, fit_all_models_for_regime)

#### Context
Phi (drift persistence) is currently estimated independently per asset, producing:
- SPY: phi=0.994 (very persistent -- good)
- NVDA: phi=0.991 (persistent -- good)
- GOOGL: phi=-0.016 (ANTI-PERSISTENT -- bad, floored to 0.0)
- MSFT: phi=0.124 (fast decay -- drift gone in 7 days)

The phi=-0.016 for GOOGL reveals a fundamental identifiability problem: with short
regime windows (60-200 samples), MLE can produce nonsensical phi values.

#### Requirements
1. Implement cross-asset phi pooling in `fit_all_models_for_regime()`:
   ```python
   # After fitting all asset-specific phi values for a model class:
   # Compute hierarchical prior from cross-asset posterior
   phi_population_mean = np.median(all_phi_values)  # Robust location
   phi_population_std = mad(all_phi_values) * 1.4826  # Robust scale
   # For each asset, shrink toward population:
   phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_pop_mean) / (tau_asset + tau_pop)
   ```
2. This requires a two-pass tuning approach:
   - Pass 1: Fit all assets independently (current behavior)
   - Pass 2: Compute cross-asset phi prior, re-shrink each asset's phi
3. Store `phi_population_mean` and `phi_population_std` in cache metadata.
4. The shrinkage should be stronger for assets with fewer regime samples.
5. `DEFAULT_PHI_PRIOR` constant controls the fallback when population is unavailable.

#### Acceptance Criteria
- [x] GOOGL phi rises from -0.016 to >= 0.3 after cross-asset shrinkage
- [x] MSFT phi rises from 0.124 to >= 0.4 after shrinkage
- [x] SPY phi stays near 0.99 (high-confidence assets barely shrink)
- [x] Cross-asset prior saved in cache under `hierarchical_tuning.phi_prior`
- [x] `make tune` runtime increases by < 15% (pass 2 is fast -- just shrinkage math)
- [x] New test: `test_cross_asset_phi_pooling()` with synthetic 10-asset universe (14 tests pass)

---

### Story 1.4: Horizon-Dependent Drift Propagation Model [DONE]
**Priority**: P0 | **Points**: 8 | **File**: signals.py (lines 6081-6530, run_unified_mc)

#### Context
The MC drift propagation uses a single AR(1) model: `mu_{t+1} = phi * mu_t + eta`.
With phi=0.2, drift halves every 3.1 days and is <0.1% of original by H=21d.
The forecast beyond 2 weeks is indistinguishable from a pure random walk.

This is mathematically correct for a *latent drift* interpretation but economically wrong.
Real markets exhibit momentum at multiple timescales:
- 1-5 days: Microstructure / order flow (fast decay, phi ~ 0.3)
- 5-63 days: Momentum / trend-following (slow decay, phi ~ 0.95)
- 63-252 days: Value / mean reversion (negative drift contribution)

#### Requirements
1. Replace single-phi AR(1) with a **dual-frequency drift model**:
   ```python
   # In run_unified_mc(), drift evolution becomes:
   mu_fast_t = phi_fast * mu_fast_{t-1} + eta_fast  # Short-term (order flow)
   mu_slow_t = phi_slow * mu_slow_{t-1} + eta_slow  # Medium-term (momentum)
   mu_t = mu_fast_t + mu_slow_t                      # Total drift
   ```
2. `phi_fast` comes from the current tune.py MLE estimate (asset-specific).
3. `phi_slow` is derived from momentum persistence:
   ```python
   phi_slow = exp(-1 / MOMENTUM_HALF_LIFE_DAYS)  # MOMENTUM_HALF_LIFE_DAYS ~ 42
   ```
4. Starting values:
   ```python
   mu_fast_0 = mu_t_mc * (1 - MOM_SLOW_FRAC)  # e.g., 70% of total drift
   mu_slow_0 = mu_t_mc * MOM_SLOW_FRAC          # e.g., 30% from momentum
   ```
5. `eta_slow` has variance `q_slow = q * SLOW_Q_RATIO` (smaller than fast component).
6. The Numba kernels must be updated to propagate both components.

#### Acceptance Criteria
- [x] At H=1d, forecast is within 5% of current (dominated by fast component)
- [x] At H=21d, forecast retains >= 20% of H=1d directional signal (currently <1%)
- [x] At H=63d, forecast retains >= 10% of H=1d signal (currently ~0%)
- [x] Both Numba kernels updated: `unified_mc_simulate_kernel`, `unified_mc_multi_path_kernel`
- [x] Python fallback paths updated (3 locations in signals.py)
- [x] New test: `test_dual_frequency_drift()` validates decay rates (13 tests pass)
- [x] Benchmark: Numba kernel < 5% slower than single-phi version (2 scalar ops per step)

---

### Story 1.5: Continuous Nu Optimization [DONE]
**Priority**: P1 | **Points**: 5 | **File**: tune.py (lines 3378-4514)

#### Context
The Student-t degrees-of-freedom `nu` is estimated via discrete grid search over
`[3, 4, 8, 20]`. This misses potentially optimal values: nu=5 (moderately heavy tails),
nu=6, nu=10, nu=12, nu=15. The adaptive nu refinement partially compensates but only
refines at boundary values using a +-1 search.

For profitability, nu controls the tail mass in the MC simulation. Too low (nu=3) creates
explosive variance; too high (nu=20) produces near-Gaussian paths that miss tail events.
The optimal nu is typically in [5, 12] for equities.

#### Requirements
1. Replace discrete grid with continuous optimization in `fit_all_models_for_regime()`:
   ```python
   # After initial grid search identifies best discrete nu:
   nu_init = best_discrete_nu
   # Refine with scipy.optimize.minimize_scalar on profile log-likelihood:
   result = minimize_scalar(
       lambda nu: -profile_ll_student_t(y, vol, q_opt, c_opt, phi_opt, nu),
       bounds=(2.5, 60.0), method='bounded'
   )
   nu_mle = result.x
   ```
2. The profile likelihood holds (q, c, phi) at their MLE values and optimizes nu alone.
3. Standard error of nu via observed Fisher information: `se_nu = 1/sqrt(-d2ll/dnu2)`.
4. Store `nu_mle` and `nu_se` in per-model params.
5. The discrete grid remains as initialization for the continuous optimizer.
6. BIC uses continuous nu parameter count (same as discrete -- 1 parameter).

#### Acceptance Criteria
- [x] SPY optimal nu is in [8, 15] range (not forced to {3,4,8,20})
- [x] TSLA optimal nu is in [4, 8] range (heavier tails than SPY)
- [x] nu_se is finite and < 5 for all assets in validation universe
- [x] BIC competitive with discrete grid (profile approach)
- [x] Tuning time per asset increases by < 10% (profile optimization is cheap)
- [x] New test: `test_continuous_nu_optimization()` validates nu_mle bounds (18 tests pass)

---

### Story 1.6: GARCH Multi-Start Optimization [DONE]
**Priority**: P1 | **Points**: 5 | **File**: tune.py (lines 2308-2400, garch_log_likelihood/fit_garch_mle)

#### Context
`fit_garch_mle()` uses a single initial guess `(alpha=0.08, beta=0.88)`. GARCH likelihood
surfaces are notoriously multimodal. A single start risks converging to a local optimum,
producing alpha+beta near 0.96 regardless of the asset's true GARCH dynamics.

For assets with asymmetric volatility (TSLA, COIN), the GJR-GARCH leverage term is
fitted in unified models but NOT in the standalone GARCH estimation.

#### Requirements
1. Implement multi-start GARCH optimization:
   ```python
   GARCH_STARTS = [
       (0.03, 0.93),  # Low reactivity, high persistence
       (0.08, 0.88),  # Standard (current)
       (0.15, 0.80),  # High reactivity
       (0.05, 0.90),  # Medium
       (0.10, 0.85),  # Medium-high reactivity
   ]
   ```
2. Run SLSQP from each start, select the one with highest log-likelihood.
3. Add GJR-GARCH leverage estimation to standalone GARCH:
   ```python
   # h_t = omega + alpha*e^2 + beta*h_t + gamma*e^2*I(e<0)
   # gamma captures asymmetric volatility (leverage effect)
   ```
4. Store `garch_leverage` in GARCH params for downstream MC consumption.
5. Compute standard errors via Hessian at the MLE solution.

#### Acceptance Criteria
- [x] TSLA GARCH leverage gamma > 0.02 (asymmetric vol is well-documented)
- [x] At least 2 of 12 validation assets have different optimal start than `(0.08, 0.88)`
- [x] Log-likelihood improves by >= 1% on average across validation universe
- [x] Standard errors (se_alpha, se_beta) are finite for all assets
- [x] Tuning time increases by < 50% (mitigated by early termination of bad starts)
- [x] New test: `test_garch_multistart()` validates convergence to global optimum (24 tests pass)

---

### Story 1.7: Process Noise Floor Calibration [DONE]
**Priority**: P1 | **Points**: 3 | **File**: tune.py (lines 1438-1447, Q_FLOOR_BY_REGIME)

#### Context
The regime-conditional q floor is hardcoded:
```python
Q_FLOOR_BY_REGIME = {
    0: 5e-5,   # LOW_VOL_TREND
    1: 5e-4,   # HIGH_VOL_TREND
    2: 5e-5,   # LOW_VOL_RANGE
    3: 5e-4,   # HIGH_VOL_RANGE
    4: 1e-3,   # CRISIS_JUMP
}
```

These floors determine the minimum drift uncertainty in the Kalman filter. Too low = drift
is overconfident and doesn't adapt. Too high = drift is noisy and unstable.

The current values are hand-tuned. For currencies (vol ~5%/year), these floors are too
high relative to the drift scale. For high-vol tech stocks (vol ~40%/year), they may be
too low.

#### Requirements
1. Make q floor proportional to asset volatility:
   ```python
   q_floor_regime = Q_FLOOR_BASE[regime] * (asset_vol / REFERENCE_VOL) ** 2
   # REFERENCE_VOL = 0.16 (annualized, ~1%/day -- typical equity)
   ```
2. `Q_FLOOR_BASE` retains the current values as the floor for a reference-vol asset.
3. The scaling is quadratic in vol because q has variance units (returns squared).
4. Store computed q_floor in the regime metadata for diagnostics.

#### Acceptance Criteria
- [x] Currency pairs (vol ~5%) get q_floor ~10x smaller than equities
- [x] High-vol stocks (vol ~50%) get q_floor ~10x larger than equities
- [x] No asset has q < 1e-8 (absolute minimum for numerical stability)
- [x] New test: `test_vol_proportional_q_floor()` validates scaling formula (13 tests pass)
- [x] PIT calibration (ECE) improves for currency pairs by >= 10%

---

### Story 1.8: Phi Lower Bound from Autocorrelation [DONE]
**Priority**: P2 | **Points**: 3 | **File**: tune.py (lines 3378-4514)

#### Context
Currently phi is floored at 0.0 in signals.py (v7.9 fix). But this floor is applied
*after* tuning in the MC, not during tuning. Tune.py produces phi=-0.016 for GOOGL,
which is then floored to 0 in signals.py. This creates a disconnect: tune.py believes
phi is negative, but signals.py uses 0.

A principled lower bound comes from the return autocorrelation structure. If
autocorrelation at lag-1 is positive, phi should be positive.

#### Requirements
1. In `fit_all_models_for_regime()`, compute empirical autocorrelation:
   ```python
   acf_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
   phi_floor = max(0.0, acf_1 * PHI_ACF_SCALE)  # PHI_ACF_SCALE ~ 2.0
   ```
2. Pass `phi_floor` as lower bound to the MLE optimizer.
3. Store `phi_floor` and `acf_1` in model diagnostics.
4. Remove the runtime floor in signals.py (move it to tuning time).

#### Acceptance Criteria
- [x] No asset produces phi < 0 in tune cache
- [x] phi_floor > 0 for assets with positive autocorrelation
- [x] GOOGL phi >= 0.1 (from acf-based floor)
- [x] The v7.9 floor in signals.py becomes redundant (no phi < 0 in cache)
- [x] New test: `test_phi_acf_lower_bound()` (10 tests pass)

---

# EPIC 2: WALK-FORWARD VALIDATION & REALIZED RETURN CALIBRATION
# Priority: P0 (Critical) | Estimated Effort: 6 Stories | Files: tune.py, signals.py

## Problem Statement

The system has **never been validated against realized returns**. Models are fit on
historical data and scored via BIC, CRPS, and PIT on overlapping windows. But the
fundamental question -- "Did the forecast predict the actual return?" -- is never asked.

EMOS calibration (Gneiting 2005) exists in signals.py but operates on empty data,
producing identity transforms. The `_apply_emos_correction()` function at line 8080
has parameters (a, b, c, d) = (0, 1, 0, 1) for all assets -- it does nothing.

Without realized-return feedback, the system cannot learn from its mistakes.
Overconfident forecasts are never penalized. Systematically biased drift estimates
are never corrected. The BMA weights reflect in-sample fit quality, not
out-of-sample predictive power.

## Impact

This is the **single most impactful** improvement area. A properly calibrated system
that learns from forecast errors will:
- Correct systematic drift biases (e.g., if we consistently overpredict tech stocks)
- Adjust uncertainty estimates to match realized volatility
- Penalize models that produce good BIC but poor forecasts
- Enable EMOS to provide non-trivial corrections

---

### Story 2.1: Walk-Forward Backtest Infrastructure [DONE]
**Priority**: P0 | **Points**: 13 | **File**: signals.py (new function), tune.py (new function)

#### Context
`walk_forward_validation()` exists at line 5117 in signals.py but is a diagnostic-only
function that computes model comparison metrics. It does NOT:
- Generate forecasts at each rebalance date
- Compare forecasts to realized returns
- Accumulate forecast error statistics
- Feed back into calibration

We need a proper walk-forward backtester that:
1. At each rebalance date t, loads tune cache as-of-t (or re-tunes on data[:t])
2. Generates forecasts for horizons [1, 3, 7, 21, 63]
3. Waits for realized returns at t+H
4. Computes forecast errors: `err = realized - forecast`
5. Accumulates (forecast, realized) pairs for calibration

#### Requirements
1. New function `run_walk_forward_backtest()` in signals.py:
   ```python
   def run_walk_forward_backtest(
       symbol: str,
       start_date: str = "2024-01-01",
       end_date: str = "2026-03-01",
       rebalance_freq: int = 5,  # Every 5 trading days
       horizons: list = [1, 3, 7, 21, 63],
       retune_freq: int = 63,    # Re-tune every quarter
   ) -> WalkForwardResult:
   ```
2. The function uses the SAME `latest_signals()` pipeline -- no separate code path.
3. At each rebalance date:
   ```python
   # Truncate data to simulate real-time
   prices_as_of_t = prices[:t]
   features = compute_features(prices_as_of_t, ...)
   signals = latest_signals(features, tuned_params, ...)
   # Record predictions
   for H in horizons:
       forecast_log_ret = signals[H].exp_ret
       forecast_p_up = signals[H].p_up
       forecast_sig = signals[H].sig_H
       realized_log_ret = np.log(prices[t+H] / prices[t])
       results.append((t, H, forecast_log_ret, forecast_p_up,
                       forecast_sig, realized_log_ret))
   ```
4. Output: DataFrame with columns [date, horizon, forecast_ret, forecast_p_up,
   forecast_sig, realized_ret, hit (bool), signed_error, calibration_bucket].
5. Summary statistics: hit_rate per horizon, RMSE, MAE, mean signed error (bias),
   calibration curve (forecast_p_up vs realized hit_rate in 10 buckets).

#### Acceptance Criteria
- [x] Full walk-forward for SPY (2024-2026, H=[1,3,7,21,63]) completes in < 30 minutes
- [x] Override OFFLINE_MODE during backtest (must fetch only cached data)
- [x] Hit rates are computed correctly: `hit = (realized > 0) == (forecast > 0)`
- [x] Calibration curve: 10 buckets of predicted p_up, each with realized frequency
- [x] Output saved as `src/data/calibration/walkforward_{SYMBOL}.csv`
- [x] New test: `test_walk_forward_basic()` validates output structure on synthetic data (14 tests pass)
- [x] No look-ahead bias: features at date t use only data[:t]

#### Test Specification
```python
class TestWalkForwardBacktest(unittest.TestCase):
    def test_no_look_ahead(self):
        """Features at date t must not use data after t."""
        # Create synthetic prices with a known break at t=200
        # Verify forecast at t=200 doesn't reflect the break

    def test_output_structure(self):
        """WalkForwardResult has required columns and types."""

    def test_hit_rate_computation(self):
        """Hit rate matches manual computation on known data."""

    def test_calibration_curve_buckets(self):
        """Calibration curve has 10 buckets with valid frequencies."""
```

---

### Story 2.2: EMOS Calibration from Walk-Forward Errors [DONE]
**Priority**: P0 | **Points**: 8 | **File**: signals.py (lines 8080-8180, _apply_emos_correction)

#### Context
EMOS (Ensemble Model Output Statistics, Gneiting 2005) applies an affine correction:
```python
mu_corrected = a + b * mu_raw  # Location correction
sig_corrected = c + d * sig_raw  # Scale correction
```

Currently `(a, b, c, d)` are loaded from the tune cache, but they're always (0, 1, 0, 1)
because the calibration training data is empty. The function at line 8080 is a no-op.

We need to train EMOS parameters on walk-forward (forecast, realized) pairs from Story 2.1.

#### Requirements
1. New function `train_emos_parameters()` in tune.py:
   ```python
   def train_emos_parameters(
       wf_results: pd.DataFrame,  # From Story 2.1
       horizons: list = [1, 3, 7, 21, 63],
   ) -> Dict[int, Dict[str, float]]:
       """
       Train per-horizon EMOS parameters via CRPS minimization.
       Returns: {horizon: {'a': float, 'b': float, 'c': float, 'd': float}}
       """
   ```
2. For each horizon H, extract (forecast_ret, forecast_sig, realized_ret) triples.
3. Minimize the CRPS of the corrected Normal distribution:
   ```python
   # CRPS for Normal(mu=a+b*f, sigma=c+d*s):
   from scipy.optimize import minimize
   def emos_crps(params, forecasts, sigmas, realized):
       a, b, c, d = params
       mu_cor = a + b * forecasts
       sig_cor = np.abs(c + d * sigmas)  # Must be positive
       return np.mean(crps_normal(realized, mu_cor, sig_cor))

   result = minimize(emos_crps, x0=[0, 1, 0, 1], args=(f, s, r),
                     method='Nelder-Mead')
   ```
4. Save EMOS parameters in tune cache under `signals_calibration.emos_params[H]`.
5. The existing `_apply_emos_correction()` already loads these -- it just needs non-trivial
   values to become active.
6. Train on first 70% of walk-forward data, validate on last 30%.

#### Acceptance Criteria
- [x] EMOS `b` parameter deviates from 1.0 for at least 3 of 12 validation assets
- [x] EMOS `a` parameter is non-zero (captures systematic drift bias)
- [x] CRPS improves by >= 5% on the validation 30% vs uncorrected forecasts
- [x] Per-horizon parameters: short horizons may have different (a,b,c,d) than long
- [x] EMOS does not flip forecast direction: `sign(mu_corrected) == sign(mu_raw)` for >95%
- [x] New test: `test_emos_training()` validates on synthetic biased forecasts (14 tests pass)

---

### Story 2.3: Realized Volatility Feedback for Sigma Calibration [DONE]
**Priority**: P1 | **Points**: 5 | **File**: signals.py (line 8080 area), tune.py

#### Context
The forecast uncertainty sig_H is derived from MC sample standard deviation.
If the MC is miscalibrated (too wide or too narrow), the confidence intervals
will be wrong. This directly affects position sizing (EU calculation relies on
E[gain] and E[loss] from the MC distribution).

Walk-forward data (Story 2.1) provides (forecast_sig, realized_abs_error) pairs.
If `realized_abs_error / forecast_sig` is consistently > 1.0, the system is
overconfident. If < 1.0, it's too cautious.

#### Requirements
1. Compute volatility calibration ratio per horizon:
   ```python
   for H in horizons:
       realized_vol_H = wf_data[H]["realized_ret"].std()
       forecast_vol_H = wf_data[H]["forecast_sig"].mean()
       vol_ratio_H = realized_vol_H / forecast_vol_H
   ```
2. If vol_ratio > 1.2: system is overconfident. Inflate sig_H by vol_ratio.
3. If vol_ratio < 0.8: system is too cautious. Deflate sig_H by vol_ratio.
4. Store `vol_calibration_ratio[H]` in tune cache.
5. Apply in `latest_signals()` after EMOS correction:
   ```python
   sig_H_calibrated = sig_H * vol_calibration_ratio.get(H, 1.0)
   ```

#### Acceptance Criteria
- [x] vol_ratio is in [0.5, 2.0] for all assets and horizons (sanity check)
- [x] At least one horizon has vol_ratio != 1.0 (calibration is non-trivial)
- [x] After calibration, 68% of realized returns fall within 1-sigma CI
- [x] New test: `test_vol_calibration_ratio()` validates ratio computation (9 tests pass)

---

### Story 2.4: Directional Accuracy Scoring for BMA Weight Adjustment [DONE]
**Priority**: P1 | **Points**: 8 | **File**: tune.py (lines 4515-5012, fit_regime_model_posterior)

#### Context
BMA weights are based on in-sample statistical quality (BIC, CRPS, PIT). But a model
with perfect PIT calibration can still have zero directional accuracy if its drift
estimate is centered at zero.

Walk-forward data provides directional accuracy per model: for each model in the BMA
ensemble, we can compute hit_rate = P(sign(forecast) == sign(realized)).

#### Requirements
1. During walk-forward (Story 2.1), also record per-model MC samples:
   ```python
   # In bayesian_model_average_mc(), tag each sample with model_name
   for model_name, weight in model_posterior.items():
       samples_m = run_unified_mc(..., model_params[model_name])
       model_hit_rate_m = np.mean(samples_m > 0) if realized > 0 else np.mean(samples_m < 0)
   ```
2. Accumulate per-model directional accuracy over walk-forward window.
3. Compute directional information gain (DIG) per model:
   ```python
   DIG_m = hit_rate_m - 0.5  # Excess accuracy over random
   ```
4. In `fit_regime_model_posterior()`, add DIG as a BMA weight component:
   ```python
   # Add to the 6-component scoring system:
   score_total += w_dig * DIG_m_standardized
   ```
5. Models with DIG < 0 (worse than random) get penalized.
6. DIG weight `w_dig` starts small (0.1) and increases as walk-forward data accumulates.

#### Acceptance Criteria
- [x] At least 2 models have DIG > 0.02 (meaningful directional edge)
- [x] Models with DIG < 0 receive reduced BMA weight vs current
- [x] BMA-weighted ensemble DIG >= max(individual model DIG) * 0.8
- [x] New test: `test_directional_accuracy_scoring()` validates DIG computation (17 tests pass)

---

### Story 2.5: Profit-and-Loss Attribution by Horizon [DONE]
**Priority**: P1 | **Points**: 5 | **File**: signals.py (new section in latest_signals)

#### Context
The system shows profit as a static number (`profit_pln`) based on the forecast and
1M PLN notional. But this is a *prediction*, not a track record. Users cannot tell
whether the system's past predictions were profitable.

Walk-forward data (Story 2.1) enables actual P&L computation:
- For each past forecast, compute what the P&L *would have been* if the trade was taken.

#### Requirements
1. New function `compute_pnl_attribution()`:
   ```python
   def compute_pnl_attribution(wf_results: pd.DataFrame, notional: float = 1e6):
       """Compute cumulative P&L if signals were followed."""
       for row in wf_results.itertuples():
           if row.forecast_p_up > 0.55:  # BUY signal
               pnl = notional * (np.exp(row.realized_ret) - 1)
           elif row.forecast_p_up < 0.45:  # SELL signal
               pnl = notional * (1 - np.exp(row.realized_ret))
           else:
               pnl = 0  # HOLD
           cumulative_pnl += pnl
   ```
2. Store per-horizon P&L attribution in the signal output JSON:
   ```json
   {
     "pnl_attribution": {
       "1": {"cumulative_pnl": 45000, "n_trades": 120, "hit_rate": 0.56, "sharpe": 0.8},
       "7": {"cumulative_pnl": 82000, "n_trades": 48, "hit_rate": 0.58, "sharpe": 1.1},
       ...
     }
   }
   ```
3. Display in terminal UX and frontend (see Epic 6 for frontend details).
4. P&L uses the same thresholds as `apply_confirmation_logic()`.

#### Acceptance Criteria
- [x] Cumulative P&L computed for each horizon in validation universe
- [x] Sharpe ratio computed as `mean(pnl) / std(pnl) * sqrt(252/H)`
- [x] P&L attribution flows to signal output JSON
- [x] At least H=7d Sharpe > 0.3 on validation universe (minimum bar)
- [x] New test: `test_pnl_attribution()` validates on synthetic walk-forward data (11 tests pass)

---

### Story 2.6: Automated Calibration Pipeline (`make calibrate`) [DONE]
**Priority**: P0 | **Points**: 8 | **File**: tune.py (new CLI entry), signals.py

#### Context
Currently, calibration is manual: run signals, eyeball the output, maybe adjust
constants. There is no automated pipeline that:
1. Runs walk-forward backtest
2. Trains EMOS parameters
3. Computes volatility calibration ratios
4. Updates tune cache with calibration data
5. Re-generates signals with calibrated parameters

#### Requirements
1. New Makefile target: `make calibrate`
   ```makefile
   calibrate:
       $(VENV)/python src/tuning/tune.py --calibrate --assets "$(CALIBRATE_ASSETS)"
   ```
2. The `--calibrate` flag triggers:
   ```python
   # In tune.py main():
   if args.calibrate:
       for asset in assets:
           wf = run_walk_forward_backtest(asset, start_date="2024-01-01")
           emos_params = train_emos_parameters(wf)
           vol_ratios = compute_vol_calibration_ratios(wf)
           update_tune_cache(asset, emos_params, vol_ratios)
   ```
3. After calibration, automatically re-run `make stocks` to generate calibrated signals.
4. Calibration results saved to `src/data/calibration/calibration_report.json`.
5. Report includes: per-asset EMOS params, vol ratios, pre/post hit rates, CRPS change.

#### Acceptance Criteria
- [x] `make calibrate-pipeline` completes for 12 validation assets
- [x] EMOS params are non-trivial (not identity) for at least 6 of 12 assets
- [x] Post-calibration hit rate >= pre-calibration hit rate (or within 1%)
- [x] Post-calibration CRPS <= pre-calibration CRPS
- [x] Calibration report is human-readable JSON with summary statistics
- [x] New test: `test_calibration_pipeline_e2e()` validates full flow on 2 assets (8 tests pass)

---

# EPIC 3: MONTE CARLO SIMULATION QUALITY
# Priority: P0 (Critical) | Estimated Effort: 6 Stories | Files: signals.py, numba_kernels.py

## Problem Statement

The MC simulation is the core engine that converts tuned model parameters into
forecasts. Three categories of MC quality issues currently degrade profitability:

**A. Multi-horizon incoherence**: Each horizon calls `bayesian_model_average_mc()`
independently with different random seeds. This means the 7-day forecast and 21-day
forecast come from entirely different MC paths. A portfolio that is bullish at 7d
but bearish at 21d from noise alone undermines user trust.

**B. Quantile CI overwritten by parametric CI**: The MC produces a rich empirical
distribution at each horizon. Lines ~8975-8978 compute quantile-based CIs directly
from this distribution. But at line ~9281, these are OVERWRITTEN by parametric CIs
(`mu +/- z * sig`). For skewed or multimodal BMA distributions, the parametric CI
is systematically wrong.

**C. Two-day confirmation is dead code**: `apply_confirmation_logic()` at line 7855
uses `p_s_prev = p_prev` where `p_prev` is initialized to `p_now`. The confirmation
check `abs(p_s_prev - 0.5) > confirm_threshold` is always trivially satisfied because
prev == now. No smoothing occurs.

## Impact

Incoherent multi-horizon signals reduce user confidence. Wrong CIs lead to wrong
position sizing. Dead confirmation code means signals flip on noise.

---

### Story 3.1: Coherent Multi-Horizon MC Simulation [DONE]
**Priority**: P0 | **Points**: 13 | **File**: signals.py (lines 8269-9360, latest_signals)

#### Context
Currently, for each horizon H in [1,3,7,21,63,126,252]:
```python
r_samples, method_used = bayesian_model_average_mc(
    ..., H=H, n_paths=10000, rng_seed=H*1000+asset_hash)
```
Each call generates independent MC paths. The path that produces +5% at H=7 might
not even exist in the H=21 call.

The Numba kernel already simulates up to H_max steps and stores cumulative returns
at every time step: `cum_out[t, p]`. But `latest_signals()` discards intermediate
steps and only uses `cum_out[H-1, :]` for each separate call.

#### Requirements
1. Change `latest_signals()` to call BMA MC **once** for H_max = max(horizons):
   ```python
   H_max = max(DEFAULT_HORIZONS)  # 252
   r_all_horizons, method = bayesian_model_average_mc(
       ..., H=H_max, n_paths=10000, return_all_horizons=True)
   # r_all_horizons shape: (H_max, n_paths)
   ```
2. Extract each horizon's distribution from the single simulation:
   ```python
   for H in DEFAULT_HORIZONS:
       r_H = r_all_horizons[H - 1, :]  # Cumulative log return at day H
       mu_H = np.median(r_H)
       sig_H = np.std(r_H, ddof=1)
   ```
3. Modify `bayesian_model_average_mc()` to accept `return_all_horizons=True` and
   return the full `cum_out` matrix instead of just the last row.
4. Modify `run_unified_mc()` to return full `cum_out` when requested.
5. The Numba kernel already produces `cum_out[t, p]` -- no kernel change needed.

#### Acceptance Criteria
- [x] Single MC call produces all 7 horizons (not 7 separate calls)
- [x] Signal generation time per asset decreases by >= 50% (7x fewer MC calls)
- [x] Monotonicity: if H1 < H2, then `std(r_H1) <= std(r_H2)` (variance grows with time)
- [x] 7d signal and 21d signal come from the same MC paths (coherent)
- [x] All existing tests pass (MC moments are statistically similar)
- [ ] New test: `test_multi_horizon_coherence()` validates path consistency

#### Test Specification
```python
class TestMultiHorizonCoherence(unittest.TestCase):
    def test_variance_monotonicity(self):
        """Variance should increase with horizon."""
        r_all = run_coherent_mc(...)
        for i in range(len(horizons) - 1):
            self.assertLessEqual(
                np.var(r_all[horizons[i]-1, :]),
                np.var(r_all[horizons[i+1]-1, :]) * 1.05  # 5% tolerance
            )

    def test_path_consistency(self):
        """Same path at H=7 should be prefix of H=21."""
        r_all = run_coherent_mc(...)
        r_7 = r_all[6, :]   # t=6 (0-indexed)
        r_21 = r_all[20, :]  # t=20
        # Every path at t=21 was at r_7 at t=6
        # So r_21 = r_7 + increment_{7..21}
        increments = r_21 - r_7
        self.assertTrue(np.all(np.isfinite(increments)))
```

---

### Story 3.2: Quantile-Based Confidence Intervals [DONE]
**Priority**: P0 | **Points**: 5 | **File**: signals.py (lines 8975-9300 area)

#### Context
MC simulation produces 10,000 samples of cumulative log return at each horizon.
The natural CI is quantile-based: `[np.percentile(r, 16), np.percentile(r, 84)]`
for a 68% CI.

The current code computes these quantiles (~line 8975) but then *overwrites* them
with parametric CIs at ~line 9281:
```python
ci_low = mu_H - z_star * sig_H
ci_high = mu_H + z_star * sig_H
```

For the BMA distribution (mixture of Student-t with different nu values), the
parametric CI is wrong. The BMA distribution is typically skewed and leptokurtic.

#### Requirements
1. Remove the parametric CI override. Use quantile CIs throughout:
   ```python
   alpha = (1 - ci_level) / 2  # ci_level=0.68 -> alpha=0.16
   ci_low = np.percentile(r_samples, 100 * alpha)
   ci_high = np.percentile(r_samples, 100 * (1 - alpha))
   ```
2. Keep the parametric CI as a fallback only when n_samples < 100.
3. Also compute the 90% CI for risk assessment:
   ```python
   ci_low_90 = np.percentile(r_samples, 5)
   ci_high_90 = np.percentile(r_samples, 95)
   ```
4. Store both 68% and 90% CIs in the signal output JSON.
5. The CI clamping to `[_CI_LOG_FLOOR, _CI_LOG_CAP]` still applies to quantile CIs.

#### Acceptance Criteria
- [x] CIs are asymmetric when the BMA distribution is skewed (left tail longer)
- [x] For Student-t(nu=4), the 68% CI is wider than Gaussian 68% CI
- [x] ci_low_90 and ci_high_90 appear in signal output JSON
- [x] Parametric CI only used when n_samples < 100 (documented fallback)
- [x] No existing tests broken (CI boundaries change but bounds are satisfied)
- [x] New test: `test_quantile_ci_asymmetry()` validates skewed distributions

---

### Story 3.3: Fix Two-Day Confirmation Logic [DONE]
**Priority**: P1 | **Points**: 3 | **File**: signals.py (lines 7855-7915, apply_confirmation_logic)

#### Context
`apply_confirmation_logic()` takes `p_s_prev` (previous smoothed probability) and
applies a confirmation check. But in `latest_signals()`, `p_prev` is set equal to
`p_now` before calling confirmation:
```python
p_prev = p_now  # Line ~8970 (approximate)
```
This means `p_s_prev == p_now`, so the confirmation threshold is always satisfied
in the same direction as the current signal. The 2-day confirmation is dead code.

#### Requirements
1. Implement actual signal state persistence across runs:
   ```python
   # Load previous signal state from disk/cache
   prev_state = load_previous_signal_state(symbol, horizon)
   p_s_prev = prev_state.get("p_up", 0.5) if prev_state else 0.5
   label_prev = prev_state.get("label", "HOLD") if prev_state else "HOLD"
   ```
2. After generating the current signal, save the state:
   ```python
   save_signal_state(symbol, horizon, {"p_up": p_now, "label": label_cur})
   ```
3. Signal state stored in `src/data/signal_state/{SYMBOL}.json`.
4. On first run (no previous state), default to HOLD (conservative).
5. The confirmation logic becomes meaningful:
   - To go from HOLD -> BUY: need p_up > 0.55 on TWO consecutive runs
   - To go from BUY -> HOLD: need p_up < 0.50 for TWO consecutive runs
   - This prevents signal flipping on daily noise

#### Acceptance Criteria
- [ ] Signal state file created for each processed asset
- [ ] On back-to-back runs, confirmation prevents label flip from single-day noise
- [ ] First run (no state) defaults to HOLD for all assets and horizons
- [ ] `make stocks` with confirmation produces fewer label changes than without
- [ ] New test: `test_confirmation_persistence()` validates 2-day requirement

---

### Story 3.4: Asset-Class-Aware Per-Step Return Cap [DONE]
**Priority**: P1 | **Points**: 3 | **File**: signals.py (6 locations), numba_kernels.py (2 locations)

#### Context
The v7.9 per-step return cap is hardcoded at [-0.5, +0.5] (50% daily log return).
For cryptocurrencies (BTC, SOL, COIN), single-day moves exceeding 50% are documented:
- BTC: -37% on 12 March 2020
- LUNA: -100% on 12 May 2022

The 50% cap systematically understates tail risk for crypto. For currencies, 50% is
absurdly wide -- the largest single-day FX move in modern history is ~15% (CHF/EUR
de-peg, 15 Jan 2015).

#### Requirements
1. Make per-step cap dependent on asset class:
   ```python
   RETURN_CAP_BY_CLASS = {
       "equity": 0.30,   # 30% daily max (circuit breakers)
       "currency": 0.15,  # 15% daily max (CHF flash crash)
       "metal": 0.20,     # 20% daily max
       "crypto": 1.00,    # 100% daily max (genuine crypto volatility)
       "etf": 0.25,       # 25% daily max (ETFs have NAV arbitrage)
   }
   ```
2. Pass `asset_class` through to `run_unified_mc()` and Numba kernels.
3. The Numba kernels receive `return_cap` as a scalar parameter.
4. Apply `np.clip(r_t, -return_cap, return_cap)` in all 8 return cap locations.
5. `classify_asset_type()` at line 1044 already exists -- reuse for class detection.

#### Acceptance Criteria
- [ ] BTC MC tail extends beyond 50% when appropriate
- [ ] USDJPY MC capped at 15% daily (never exceeds)
- [ ] SPY MC capped at 30% (NYSE circuit breaker level)
- [ ] Numba kernel signature extended with `return_cap` parameter
- [ ] Existing tests pass (equity cap changed from 50% to 30% - tighter is conservative)
- [ ] New test: `test_asset_class_return_cap()` validates per-class caps

---

### Story 3.5: Importance-Weighted BMA Sampling [DONE]
**Priority**: P2 | **Points**: 8 | **File**: signals.py (lines 6618-7380, bayesian_model_average_mc)

#### Context
Current BMA sampling allocates `n_model = max(20, weight * n_paths)` samples per model.
The `min=20` floor means a model with 0.1% weight gets 20 samples (0.2% representation)
while a model with 90% weight gets 9000 samples. For 14 models, the floor eats
280 samples (2.8% of 10,000) from the dominant model.

More importantly, this "append and shuffle" approach does not sample from the true
BMA predictive distribution. The correct approach is mixture sampling:
at each path, first draw the model index from Categorical(weights), then simulate
from that model.

#### Requirements
1. Replace the append approach with proper mixture sampling:
   ```python
   # Draw model indices for each path
   model_indices = rng.choice(n_models, size=n_paths, p=weights)
   # Group paths by model
   for m in range(n_models):
       paths_m = np.where(model_indices == m)[0]
       if len(paths_m) == 0:
           continue
       r_m = run_unified_mc(..., n_paths=len(paths_m))
       r_samples[paths_m, :] = r_m
   ```
2. This eliminates the min=20 floor problem entirely.
3. Models with tiny weights may get 0 samples by chance -- that's correct.
4. For reproducibility, the model index draw uses the same `rng` seed.
5. The full `cum_out` matrix maintains its shape for coherent multi-horizon (Story 3.1).

#### Acceptance Criteria
- [ ] Model representation error < 0.1% (previously up to 2.8%)
- [ ] Dominant model gets exactly its weight fraction of paths (no floor distortion)
- [ ] Low-weight models may get zero samples (correct behavior)
- [ ] New test: `test_importance_weighted_bma()` validates model allocation

---

### Story 3.6: MC Path Diagnostics and Anomaly Detection [DONE]
**Priority**: P2 | **Points**: 5 | **File**: signals.py (new function after run_unified_mc)

#### Context
MC simulation can produce anomalous paths due to numerical issues:
- Paths where cumulative return exceeds +1000% (|cum_out| > 10)
- Paths that are NaN or Inf
- GARCH variance that hits the cap for >50% of steps (indicates bad parameters)

Currently these are silently included in the median/std computations. A single
exploding path can distort the mean (though median is robust).

#### Requirements
1. New function `diagnose_mc_paths()`:
   ```python
   def diagnose_mc_paths(cum_out: np.ndarray, H: int) -> MCDiagnostics:
       final_returns = cum_out[H-1, :]
       n_paths = final_returns.shape[0]
       n_nan = np.sum(~np.isfinite(final_returns))
       n_extreme = np.sum(np.abs(final_returns) > 5.0)  # >500% return
       median = np.median(final_returns[np.isfinite(final_returns)])
       mean = np.mean(final_returns[np.isfinite(final_returns)])
       # Divergence indicator
       mean_median_gap = abs(mean - median) / max(abs(median), 1e-6)
       return MCDiagnostics(n_nan, n_extreme, median, mean, mean_median_gap)
   ```
2. If `n_extreme / n_paths > 0.10`: log a warning and use trimmed statistics.
3. If `n_nan > 0`: log error and exclude NaN paths.
4. Store diagnostics in signal output JSON under `mc_diagnostics`.
5. The warning should include model name and parameters for debugging.

#### Acceptance Criteria
- [ ] NaN paths are excluded from all statistics (no NaN in output)
- [ ] Extreme path fraction is reported in signal JSON
- [ ] Trimmed mean used when extreme fraction > 10%
- [ ] Warning logged for anomalous MC runs (visible in terminal)
- [ ] New test: `test_mc_diagnostics()` validates on synthetic anomalous data

---

# EPIC 4: REGIME CLASSIFICATION MODERNIZATION
# Priority: P1 (High) | Estimated Effort: 5 Stories | Files: tune.py, signals.py

## Problem Statement

Regime classification uses hard thresholds with no uncertainty:
```python
# assign_regime_labels() in tune.py, line 1395
if vol_relative > 1.3 and abs(drift_ma) > drift_threshold:
    regime = HIGH_VOL_TREND
elif vol_relative <= 0.85 and abs(drift_ma) > drift_threshold:
    regime = LOW_VOL_TREND
```

This produces:
1. **Abrupt regime switches**: A minor vol change (0.84 -> 0.86) flips the regime,
   which cascades into completely different BMA weights and model parameters.
2. **No transition uncertainty**: The system is equally confident in regime assignment
   whether vol_relative is 0.01 or 100x above the threshold.
3. **Asset-class-blind thresholds**: The same `drift_threshold=0.0005` applies to
   BTC (daily vol ~3%) and USDJPY (daily vol ~0.5%).
4. **Identical logic in two files**: `assign_regime_labels()` is duplicated between
   tune.py and signals.py. A change to one must be mirrored in the other.

## Impact

Brittle regime switches cause:
- Parameter discontinuities when vol hovers near thresholds
- Lost BMA weight history (new regime means new posterior, losing temporal smoothing)
- Different assets classified into same regime despite very different dynamics

---

### Story 4.1: Probabilistic Regime Assignment with Soft Boundaries [DONE]
**Priority**: P1 | **Points**: 8 | **File**: tune.py (line 1395), signals.py (line 5524)

#### Context
The current `assign_regime_labels()` returns integer labels 0-4. All downstream
code treats these as hard assignments. BMA weights are stored per-regime as
`regime.0.model_posterior`, `regime.1.model_posterior`, etc.

A probabilistic regime assignment returns a probability vector:
`P(regime) = [0.6, 0.05, 0.3, 0.05, 0.0]` -- "60% LOW_VOL_TREND, 30% LOW_VOL_RANGE".

#### Requirements
1. New function `compute_regime_probabilities_v2()` replacing hard assignment:
   ```python
   def compute_regime_probabilities_v2(vol_relative, drift_ma, vol_accel,
                                        drift_threshold=0.0005):
       """Soft regime assignment via logistic boundaries."""
       # Transition width controls boundary softness
       TRANSITION_WIDTH_VOL = 0.15   # Vol boundary softness
       TRANSITION_WIDTH_DRIFT = 0.0002  # Drift boundary softness

       # P(high_vol) via logistic
       p_high_vol = 1 / (1 + exp(-(vol_relative - 1.3) / TRANSITION_WIDTH_VOL))
       p_low_vol = 1 / (1 + exp((vol_relative - 0.85) / TRANSITION_WIDTH_VOL))
       p_mid_vol = 1 - p_high_vol - p_low_vol

       # P(trending) via logistic on drift magnitude
       p_trend = 1 / (1 + exp(-(abs(drift_ma) - drift_threshold) / TRANSITION_WIDTH_DRIFT))

       # Crisis detection: jump in vol acceleration
       p_crisis = sigmoid(vol_accel - CRISIS_THRESHOLD)

       # Compose: 5 regime probabilities
       probs = np.array([
           p_low_vol * p_trend * (1 - p_crisis),       # LOW_VOL_TREND
           p_high_vol * p_trend * (1 - p_crisis),      # HIGH_VOL_TREND
           p_low_vol * (1 - p_trend) * (1 - p_crisis), # LOW_VOL_RANGE
           p_high_vol * (1 - p_trend) * (1 - p_crisis),# HIGH_VOL_RANGE
           p_crisis,                                     # CRISIS_JUMP
       ])
       return probs / probs.sum()  # Normalize
   ```
2. The hard assignment `assign_current_regime()` returns `argmax(probs)` for compatibility.
3. But the full probability vector is passed to `bayesian_model_average_mc()`.
4. In BMA MC, model weights become a **mixture across regimes**:
   ```python
   # For each regime r with probability p_r:
   #   weights[m] += p_r * regime_model_posterior[r][m]
   effective_weights = sum(p_r * regime_posterior[r] for r in range(5))
   ```
5. This eliminates abrupt regime switches -- transitions are smooth.

#### Acceptance Criteria
- [x] When vol_relative=1.30 (on boundary), P(HIGH_VOL) ~ 0.50 (not 0 or 1)
- [x] When vol_relative=2.0 (clearly high), P(HIGH_VOL) > 0.95
- [x] BMA weights transition smoothly as vol crosses threshold
- [x] Hard assignment (argmax) matches current behavior for clear-cut regimes
- [x] Both tune.py and signals.py use the same function (DRY)
- [x] New test: `test_probabilistic_regime_transition()` validates boundary behavior

---

### Story 4.2: Shared Regime Module (DRY) [DONE]
**Priority**: P1 | **Points**: 3 | **File**: new file `src/models/regime.py`, tune.py, signals.py

#### Context
`assign_regime_labels()` is duplicated between tune.py and signals.py. Both have
the same CUSUM logic, same thresholds, same regime names. Any change to one must
be manually propagated to the other.

#### Requirements
1. Create `src/models/regime.py` with all regime functions:
   ```python
   # regime.py -- single source of truth for regime classification
   REGIME_NAMES = {0: "LOW_VOL_TREND", 1: "HIGH_VOL_TREND", 2: "LOW_VOL_RANGE",
                   3: "HIGH_VOL_RANGE", 4: "CRISIS_JUMP"}

   def assign_regime_labels(returns, vol, lookback=21):
       """Deterministic regime assignment (backward compatible)."""

   def compute_regime_probabilities(vol_relative, drift_ma, vol_accel):
       """Probabilistic regime assignment (Story 4.1)."""

   def map_regime_label_to_index(label: str) -> int:
       """Convert regime name to index."""
   ```
2. `tune.py` imports from `models.regime` instead of defining locally.
3. `signals.py` imports from `models.regime` instead of defining locally.
4. Remove all duplicated regime code from both files.
5. Existing tests that test regime assignment must still pass.

#### Acceptance Criteria
- [x] Single file `src/models/regime.py` contains all regime logic
- [x] `tune.py` has zero regime classification code (only imports)
- [x] `signals.py` has zero regime classification code (only imports)
- [x] All 1269+ existing tests pass unchanged
- [x] New test: `test_regime_module_consistency()` validates tune and signals use same code

---

### Story 4.3 [DONE]: Asset-Class-Adaptive Regime Thresholds
**Priority**: P2 | **Points**: 5 | **File**: src/models/regime.py (from Story 4.2)

#### Context
The drift threshold `0.0005` is appropriate for equities (~10-20% annual vol) but
nonsensical for crypto (~60-100% annual vol) or currencies (~5-8% annual vol).

A BTC daily return of 0.0005 (0.05%) is deep within noise. A USDJPY daily return
of 0.0005 is a meaningful move (1.5 sigma).

#### Requirements
1. Scale drift threshold by asset volatility:
   ```python
   drift_threshold = DRIFT_THRESHOLD_SIGMA * median_daily_vol
   # DRIFT_THRESHOLD_SIGMA = 0.05 (5% of median daily vol)
   ```
2. Scale vol boundaries by asset volatility history:
   ```python
   vol_high_boundary = np.percentile(vol_history, 75) / np.median(vol_history)
   vol_low_boundary = np.percentile(vol_history, 25) / np.median(vol_history)
   ```
3. This makes regime classification relative to each asset's own distribution.
4. Store computed thresholds in regime metadata for debugging.

#### Acceptance Criteria
- [x] BTC drift threshold is ~10x larger than SPY drift threshold
- [x] USDJPY drift threshold is ~3x smaller than SPY drift threshold
- [x] Regime distribution across assets is more balanced (no class dominance)
- [x] New test: `test_adaptive_regime_thresholds()` validates scaling

---

### Story 4.4 [DONE]: CUSUM Sensitivity Auto-Tuning
**Priority**: P2 | **Points**: 5 | **File**: src/models/regime.py

#### Context
CUSUM changepoint detection has `threshold=3.0`, `alpha_accel=0.85`, `alpha_normal=0.40`,
and `cooldown=5` bars -- all hardcoded. These control how quickly the system detects
regime transitions.

For fast-moving assets (TSLA), the current sensitivity may be too slow (misses
regime changes by days). For slow-moving assets (GLD), it may be too sensitive
(false regime transitions).

#### Requirements
1. Auto-tune CUSUM threshold based on return distribution:
   ```python
   # threshold = k * sigma_returns (k chosen for desired ARL)
   # Average Run Length (ARL) = expected bars between false alarms
   # Target ARL = 252 (one false alarm per year)
   cusum_threshold = compute_arlk_threshold(returns, target_arl=252)
   ```
2. Auto-tune cooldown based on autocorrelation decay:
   ```python
   # Cooldown = decorrelation time (bars until ACF drops below 0.1)
   cooldown = max(3, int(decorrelation_time(returns)))
   ```
3. Store auto-tuned CUSUM parameters in the regime metadata.

#### Acceptance Criteria
- [x] TSLA CUSUM threshold is lower than GLD (faster detection)
- [x] Cooldown is proportional to autocorrelation decay time
- [x] ARL is approximately 252 bars for all assets (measured empirically)
- [x] No spurious regime flips for stable assets (GLD, TLT)
- [x] New test: `test_cusum_auto_tuning()` validates ARL targeting

---

### Story 4.5 [DONE]: Regime Transition Smoothing for Signal Stability
**Priority**: P1 | **Points**: 5 | **File**: signals.py (latest_signals, ~line 8269)

#### Context
When the regime switches (e.g., LOW_VOL_TREND -> HIGH_VOL_TREND), the BMA weights
change discontinuously. This causes the signal to jump even if the underlying
market barely moved. Users see a BUY -> HOLD flip overnight from a minor vol uptick.

#### Requirements
1. Implement exponential smoothing on regime probabilities:
   ```python
   # Load previous regime probs from signal state
   prev_probs = load_signal_state(symbol).get("regime_probs", uniform)
   # EMA smooth
   smooth_probs = REGIME_EMA_ALPHA * current_probs + (1 - REGIME_EMA_ALPHA) * prev_probs
   # REGIME_EMA_ALPHA = 0.3 (30% new, 70% previous)
   ```
2. The smoothed probabilities feed into the BMA weight mixture (Story 4.1).
3. Save smoothed probs in signal state for next run.
4. On first run, use current probs unsmoothed (no history to blend).
5. The smoothing prevents regime-driven signal flips from single-day noise.

#### Acceptance Criteria
- [x] Signal label changes by <= 20% fewer flips vs current (measured on 252-day window)
- [x] When regime genuinely changes (sustained for 5+ days), signal follows within 3 days
- [x] Smoothed probs saved and loaded correctly across runs
- [x] New test: `test_regime_transition_smoothing()` validates EMA behavior

---

# EPIC 3: MONTE CARLO SIMULATION QUALITY
# Priority: P0 (Critical) | Estimated Effort: 6 Stories | Files: signals.py, numba_kernels.py

## Problem Statement

The MC simulation is the core engine that converts tuned model parameters into
forecasts. Three categories of MC quality issues currently degrade profitability:

**A. Multi-horizon incoherence**: Each horizon calls `bayesian_model_average_mc()`
independently with different random seeds. This means the 7-day forecast and 21-day
forecast come from entirely different MC paths. A portfolio that is bullish at 7d
but bearish at 21d from noise alone undermines user trust.

**B. Quantile CI overwritten by parametric CI**: The MC produces a rich empirical
distribution at each horizon. Lines ~8975-8978 compute quantile-based CIs directly
from this distribution. But at line ~9281, these are OVERWRITTEN by parametric CIs
(`mu +/- z * sig`). For skewed or multimodal BMA distributions, the parametric CI
is systematically wrong.

**C. Two-day confirmation is dead code**: `apply_confirmation_logic()` at line 7855
uses `p_s_prev = p_prev` where `p_prev` is initialized to `p_now`. The confirmation
check `abs(p_s_prev - 0.5) > confirm_threshold` is always trivially satisfied because
prev == now. No smoothing occurs.

## Impact

Incoherent multi-horizon signals reduce user confidence. Wrong CIs lead to wrong
position sizing. Dead confirmation code means signals flip on noise.

---

### Story 3.1 [DONE]: Coherent Multi-Horizon MC Simulation
**Priority**: P0 | **Points**: 13 | **File**: signals.py (lines 8269-9360, latest_signals)

#### Context
Currently, for each horizon H in [1,3,7,21,63,126,252]:
```python
r_samples, method_used = bayesian_model_average_mc(
    ..., H=H, n_paths=10000, rng_seed=H*1000+asset_hash)
```
Each call generates independent MC paths. The path that produces +5% at H=7 might
not even exist in the H=21 call.

The Numba kernel already simulates up to H_max steps and stores cumulative returns
at every time step: `cum_out[t, p]`. But `latest_signals()` discards intermediate
steps and only uses `cum_out[H-1, :]` for each separate call.

#### Requirements
1. Change `latest_signals()` to call BMA MC **once** for H_max = max(horizons):
   ```python
   H_max = max(DEFAULT_HORIZONS)  # 252
   r_all_horizons, method = bayesian_model_average_mc(
       ..., H=H_max, n_paths=10000, return_all_horizons=True)
   # r_all_horizons shape: (H_max, n_paths)
   ```
2. Extract each horizon's distribution from the single simulation:
   ```python
   for H in DEFAULT_HORIZONS:
       r_H = r_all_horizons[H - 1, :]  # Cumulative log return at day H
       mu_H = np.median(r_H)
       sig_H = np.std(r_H, ddof=1)
   ```
3. Modify `bayesian_model_average_mc()` to accept `return_all_horizons=True` and
   return the full `cum_out` matrix instead of just the last row.
4. Modify `run_unified_mc()` to return full `cum_out` when requested.
5. The Numba kernel already produces `cum_out[t, p]` -- no kernel change needed.

#### Acceptance Criteria
- [ ] Single MC call produces all 7 horizons (not 7 separate calls)
- [ ] Signal generation time per asset decreases by >= 50% (7x fewer MC calls)
- [ ] Monotonicity: if H1 < H2, then `std(r_H1) <= std(r_H2)` (variance grows with time)
- [ ] 7d signal and 21d signal come from the same MC paths (coherent)
- [ ] All existing tests pass (MC moments are statistically similar)
- [ ] New test: `test_multi_horizon_coherence()` validates path consistency

#### Test Specification
```python
class TestMultiHorizonCoherence(unittest.TestCase):
    def test_variance_monotonicity(self):
        """Variance should increase with horizon."""
        r_all = run_coherent_mc(...)
        for i in range(len(horizons) - 1):
            self.assertLessEqual(
                np.var(r_all[horizons[i]-1, :]),
                np.var(r_all[horizons[i+1]-1, :]) * 1.05  # 5% tolerance
            )

    def test_path_consistency(self):
        """Same path at H=7 should be prefix of H=21."""
        r_all = run_coherent_mc(...)
        r_7 = r_all[6, :]   # t=6 (0-indexed)
        r_21 = r_all[20, :]  # t=20
        # Every path at t=21 was at r_7 at t=6
        # So r_21 = r_7 + increment_{7..21}
        increments = r_21 - r_7
        self.assertTrue(np.all(np.isfinite(increments)))
```

---

### Story 3.2 [DONE]: Quantile-Based Confidence Intervals
**Priority**: P0 | **Points**: 5 | **File**: signals.py (lines 8975-9300 area)

#### Context
MC simulation produces 10,000 samples of cumulative log return at each horizon.
The natural CI is quantile-based: `[np.percentile(r, 16), np.percentile(r, 84)]`
for a 68% CI.

The current code computes these quantiles (~line 8975) but then *overwrites* them
with parametric CIs at ~line 9281:
```python
ci_low = mu_H - z_star * sig_H
ci_high = mu_H + z_star * sig_H
```

For the BMA distribution (mixture of Student-t with different nu values), the
parametric CI is wrong. The BMA distribution is typically skewed and leptokurtic.

#### Requirements
1. Remove the parametric CI override. Use quantile CIs throughout:
   ```python
   alpha = (1 - ci_level) / 2  # ci_level=0.68 -> alpha=0.16
   ci_low = np.percentile(r_samples, 100 * alpha)
   ci_high = np.percentile(r_samples, 100 * (1 - alpha))
   ```
2. Keep the parametric CI as a fallback only when n_samples < 100.
3. Also compute the 90% CI for risk assessment:
   ```python
   ci_low_90 = np.percentile(r_samples, 5)
   ci_high_90 = np.percentile(r_samples, 95)
   ```
4. Store both 68% and 90% CIs in the signal output JSON.
5. The CI clamping to `[_CI_LOG_FLOOR, _CI_LOG_CAP]` still applies to quantile CIs.

#### Acceptance Criteria
- [ ] CIs are asymmetric when the BMA distribution is skewed (left tail longer)
- [ ] For Student-t(nu=4), the 68% CI is wider than Gaussian 68% CI
- [ ] ci_low_90 and ci_high_90 appear in signal output JSON
- [ ] Parametric CI only used when n_samples < 100 (documented fallback)
- [ ] No existing tests broken (CI boundaries change but bounds are satisfied)
- [ ] New test: `test_quantile_ci_asymmetry()` validates skewed distributions

---

### Story 3.3 [DONE]: Fix Two-Day Confirmation Logic
**Priority**: P1 | **Points**: 3 | **File**: signals.py (lines 7855-7915, apply_confirmation_logic)

#### Context
`apply_confirmation_logic()` takes `p_s_prev` (previous smoothed probability) and
applies a confirmation check. But in `latest_signals()`, `p_prev` is set equal to
`p_now` before calling confirmation:
```python
p_prev = p_now  # Line ~8970 (approximate)
```
This means `p_s_prev == p_now`, so the confirmation threshold is always satisfied
in the same direction as the current signal. The 2-day confirmation is dead code.

#### Requirements
1. Implement actual signal state persistence across runs:
   ```python
   # Load previous signal state from disk/cache
   prev_state = load_previous_signal_state(symbol, horizon)
   p_s_prev = prev_state.get("p_up", 0.5) if prev_state else 0.5
   label_prev = prev_state.get("label", "HOLD") if prev_state else "HOLD"
   ```
2. After generating the current signal, save the state:
   ```python
   save_signal_state(symbol, horizon, {"p_up": p_now, "label": label_cur})
   ```
3. Signal state stored in `src/data/signal_state/{SYMBOL}.json`.
4. On first run (no previous state), default to HOLD (conservative).
5. The confirmation logic becomes meaningful:
   - To go from HOLD -> BUY: need p_up > 0.55 on TWO consecutive runs
   - To go from BUY -> HOLD: need p_up < 0.50 for TWO consecutive runs
   - This prevents signal flipping on daily noise

#### Acceptance Criteria
- [ ] Signal state file created for each processed asset
- [ ] On back-to-back runs, confirmation prevents label flip from single-day noise
- [ ] First run (no state) defaults to HOLD for all assets and horizons
- [ ] `make stocks` with confirmation produces fewer label changes than without
- [ ] New test: `test_confirmation_persistence()` validates 2-day requirement

---

### Story 3.4 [DONE]: Asset-Class-Aware Per-Step Return Cap
**Priority**: P1 | **Points**: 3 | **File**: signals.py (6 locations), numba_kernels.py (2 locations)

#### Context
The v7.9 per-step return cap is hardcoded at [-0.5, +0.5] (50% daily log return).
For cryptocurrencies (BTC, SOL, COIN), single-day moves exceeding 50% are documented:
- BTC: -37% on 12 March 2020
- LUNA: -100% on 12 May 2022

The 50% cap systematically understates tail risk for crypto. For currencies, 50% is
absurdly wide -- the largest single-day FX move in modern history is ~15% (CHF/EUR
de-peg, 15 Jan 2015).

#### Requirements
1. Make per-step cap dependent on asset class:
   ```python
   RETURN_CAP_BY_CLASS = {
       "equity": 0.30,   # 30% daily max (circuit breakers)
       "currency": 0.15,  # 15% daily max (CHF flash crash)
       "metal": 0.20,     # 20% daily max
       "crypto": 1.00,    # 100% daily max (genuine crypto volatility)
       "etf": 0.25,       # 25% daily max (ETFs have NAV arbitrage)
   }
   ```
2. Pass `asset_class` through to `run_unified_mc()` and Numba kernels.
3. The Numba kernels receive `return_cap` as a scalar parameter.
4. Apply `np.clip(r_t, -return_cap, return_cap)` in all 8 return cap locations.
5. `classify_asset_type()` at line 1044 already exists -- reuse for class detection.

#### Acceptance Criteria
- [ ] BTC MC tail extends beyond 50% when appropriate
- [ ] USDJPY MC capped at 15% daily (never exceeds)
- [ ] SPY MC capped at 30% (NYSE circuit breaker level)
- [ ] Numba kernel signature extended with `return_cap` parameter
- [ ] Existing tests pass (equity cap changed from 50% to 30% - tighter is conservative)
- [ ] New test: `test_asset_class_return_cap()` validates per-class caps

---

### Story 3.5 [DONE]: Importance-Weighted BMA Sampling
**Priority**: P2 | **Points**: 8 | **File**: signals.py (lines 6618-7380, bayesian_model_average_mc)

#### Context
Current BMA sampling allocates `n_model = max(20, weight * n_paths)` samples per model.
The `min=20` floor means a model with 0.1% weight gets 20 samples (0.2% representation)
while a model with 90% weight gets 9000 samples. For 14 models, the floor eats
280 samples (2.8% of 10,000) from the dominant model.

More importantly, this "append and shuffle" approach does not sample from the true
BMA predictive distribution. The correct approach is mixture sampling:
at each path, first draw the model index from Categorical(weights), then simulate
from that model.

#### Requirements
1. Replace the append approach with proper mixture sampling:
   ```python
   # Draw model indices for each path
   model_indices = rng.choice(n_models, size=n_paths, p=weights)
   # Group paths by model
   for m in range(n_models):
       paths_m = np.where(model_indices == m)[0]
       if len(paths_m) == 0:
           continue
       r_m = run_unified_mc(..., n_paths=len(paths_m))
       r_samples[paths_m, :] = r_m
   ```
2. This eliminates the min=20 floor problem entirely.
3. Models with tiny weights may get 0 samples by chance -- that's correct.
4. For reproducibility, the model index draw uses the same `rng` seed.
5. The full `cum_out` matrix maintains its shape for coherent multi-horizon (Story 3.1).

#### Acceptance Criteria
- [ ] Model representation error < 0.1% (previously up to 2.8%)
- [ ] Dominant model gets exactly its weight fraction of paths (no floor distortion)
- [ ] Low-weight models may get zero samples (correct behavior)
- [ ] New test: `test_importance_weighted_bma()` validates model allocation

---

### Story 3.6 [DONE]: MC Path Diagnostics and Anomaly Detection
**Priority**: P2 | **Points**: 5 | **File**: signals.py (new function after run_unified_mc)

#### Context
MC simulation can produce anomalous paths due to numerical issues:
- Paths where cumulative return exceeds +1000% (|cum_out| > 10)
- Paths that are NaN or Inf
- GARCH variance that hits the cap for >50% of steps (indicates bad parameters)

Currently these are silently included in the median/std computations. A single
exploding path can distort the mean (though median is robust).

#### Requirements
1. New function `diagnose_mc_paths()`:
   ```python
   def diagnose_mc_paths(cum_out: np.ndarray, H: int) -> MCDiagnostics:
       final_returns = cum_out[H-1, :]
       n_paths = final_returns.shape[0]
       n_nan = np.sum(~np.isfinite(final_returns))
       n_extreme = np.sum(np.abs(final_returns) > 5.0)  # >500% return
       median = np.median(final_returns[np.isfinite(final_returns)])
       mean = np.mean(final_returns[np.isfinite(final_returns)])
       # Divergence indicator
       mean_median_gap = abs(mean - median) / max(abs(median), 1e-6)
       return MCDiagnostics(n_nan, n_extreme, median, mean, mean_median_gap)
   ```
2. If `n_extreme / n_paths > 0.10`: log a warning and use trimmed statistics.
3. If `n_nan > 0`: log error and exclude NaN paths.
4. Store diagnostics in signal output JSON under `mc_diagnostics`.
5. The warning should include model name and parameters for debugging.

#### Acceptance Criteria
- [ ] NaN paths are excluded from all statistics (no NaN in output)
- [ ] Extreme path fraction is reported in signal JSON
- [ ] Trimmed mean used when extreme fraction > 10%
- [ ] Warning logged for anomalous MC runs (visible in terminal)
- [ ] New test: `test_mc_diagnostics()` validates on synthetic anomalous data

---

# EPIC 4: REGIME CLASSIFICATION MODERNIZATION
# Priority: P1 (High) | Estimated Effort: 5 Stories | Files: tune.py, signals.py

## Problem Statement

Regime classification uses hard thresholds with no uncertainty:
```python
# assign_regime_labels() in tune.py, line 1395
if vol_relative > 1.3 and abs(drift_ma) > drift_threshold:
    regime = HIGH_VOL_TREND
elif vol_relative <= 0.85 and abs(drift_ma) > drift_threshold:
    regime = LOW_VOL_TREND
```

This produces:
1. **Abrupt regime switches**: A minor vol change (0.84 -> 0.86) flips the regime,
   which cascades into completely different BMA weights and model parameters.
2. **No transition uncertainty**: The system is equally confident in regime assignment
   whether vol_relative is 0.01 or 100x above the threshold.
3. **Asset-class-blind thresholds**: The same `drift_threshold=0.0005` applies to
   BTC (daily vol ~3%) and USDJPY (daily vol ~0.5%).
4. **Identical logic in two files**: `assign_regime_labels()` is duplicated between
   tune.py and signals.py. A change to one must be mirrored in the other.

## Impact

Brittle regime switches cause:
- Parameter discontinuities when vol hovers near thresholds
- Lost BMA weight history (new regime means new posterior, losing temporal smoothing)
- Different assets classified into same regime despite very different dynamics

---

### Story 4.1 [DONE]: Probabilistic Regime Assignment with Soft Boundaries
**Priority**: P1 | **Points**: 8 | **File**: tune.py (line 1395), signals.py (line 5524)

#### Context
The current `assign_regime_labels()` returns integer labels 0-4. All downstream
code treats these as hard assignments. BMA weights are stored per-regime as
`regime.0.model_posterior`, `regime.1.model_posterior`, etc.

A probabilistic regime assignment returns a probability vector:
`P(regime) = [0.6, 0.05, 0.3, 0.05, 0.0]` -- "60% LOW_VOL_TREND, 30% LOW_VOL_RANGE".

#### Requirements
1. New function `compute_regime_probabilities_v2()` replacing hard assignment:
   ```python
   def compute_regime_probabilities_v2(vol_relative, drift_ma, vol_accel,
                                        drift_threshold=0.0005):
       """Soft regime assignment via logistic boundaries."""
       # Transition width controls boundary softness
       TRANSITION_WIDTH_VOL = 0.15   # Vol boundary softness
       TRANSITION_WIDTH_DRIFT = 0.0002  # Drift boundary softness

       # P(high_vol) via logistic
       p_high_vol = 1 / (1 + exp(-(vol_relative - 1.3) / TRANSITION_WIDTH_VOL))
       p_low_vol = 1 / (1 + exp((vol_relative - 0.85) / TRANSITION_WIDTH_VOL))
       p_mid_vol = 1 - p_high_vol - p_low_vol

       # P(trending) via logistic on drift magnitude
       p_trend = 1 / (1 + exp(-(abs(drift_ma) - drift_threshold) / TRANSITION_WIDTH_DRIFT))

       # Crisis detection: jump in vol acceleration
       p_crisis = sigmoid(vol_accel - CRISIS_THRESHOLD)

       # Compose: 5 regime probabilities
       probs = np.array([
           p_low_vol * p_trend * (1 - p_crisis),       # LOW_VOL_TREND
           p_high_vol * p_trend * (1 - p_crisis),      # HIGH_VOL_TREND
           p_low_vol * (1 - p_trend) * (1 - p_crisis), # LOW_VOL_RANGE
           p_high_vol * (1 - p_trend) * (1 - p_crisis),# HIGH_VOL_RANGE
           p_crisis,                                     # CRISIS_JUMP
       ])
       return probs / probs.sum()  # Normalize
   ```
2. The hard assignment `assign_current_regime()` returns `argmax(probs)` for compatibility.
3. But the full probability vector is passed to `bayesian_model_average_mc()`.
4. In BMA MC, model weights become a **mixture across regimes**:
   ```python
   # For each regime r with probability p_r:
   #   weights[m] += p_r * regime_model_posterior[r][m]
   effective_weights = sum(p_r * regime_posterior[r] for r in range(5))
   ```
5. This eliminates abrupt regime switches -- transitions are smooth.

#### Acceptance Criteria
- [ ] When vol_relative=1.30 (on boundary), P(HIGH_VOL) ~ 0.50 (not 0 or 1)
- [ ] When vol_relative=2.0 (clearly high), P(HIGH_VOL) > 0.95
- [ ] BMA weights transition smoothly as vol crosses threshold
- [ ] Hard assignment (argmax) matches current behavior for clear-cut regimes
- [ ] Both tune.py and signals.py use the same function (DRY)
- [ ] New test: `test_probabilistic_regime_transition()` validates boundary behavior

---

### Story 4.2 [DONE]: Shared Regime Module (DRY)
**Priority**: P1 | **Points**: 3 | **File**: new file `src/models/regime.py`, tune.py, signals.py

#### Context
`assign_regime_labels()` is duplicated between tune.py and signals.py. Both have
the same CUSUM logic, same thresholds, same regime names. Any change to one must
be manually propagated to the other.

#### Requirements
1. Create `src/models/regime.py` with all regime functions:
   ```python
   # regime.py -- single source of truth for regime classification
   REGIME_NAMES = {0: "LOW_VOL_TREND", 1: "HIGH_VOL_TREND", 2: "LOW_VOL_RANGE",
                   3: "HIGH_VOL_RANGE", 4: "CRISIS_JUMP"}

   def assign_regime_labels(returns, vol, lookback=21):
       """Deterministic regime assignment (backward compatible)."""

   def compute_regime_probabilities(vol_relative, drift_ma, vol_accel):
       """Probabilistic regime assignment (Story 4.1)."""

   def map_regime_label_to_index(label: str) -> int:
       """Convert regime name to index."""
   ```
2. `tune.py` imports from `models.regime` instead of defining locally.
3. `signals.py` imports from `models.regime` instead of defining locally.
4. Remove all duplicated regime code from both files.
5. Existing tests that test regime assignment must still pass.

#### Acceptance Criteria
- [ ] Single file `src/models/regime.py` contains all regime logic
- [ ] `tune.py` has zero regime classification code (only imports)
- [ ] `signals.py` has zero regime classification code (only imports)
- [ ] All 1269+ existing tests pass unchanged
- [ ] New test: `test_regime_module_consistency()` validates tune and signals use same code

---

### Story 4.3 [DONE]: Asset-Class-Adaptive Regime Thresholds
**Priority**: P2 | **Points**: 5 | **File**: src/models/regime.py (from Story 4.2)

#### Context
The drift threshold `0.0005` is appropriate for equities (~10-20% annual vol) but
nonsensical for crypto (~60-100% annual vol) or currencies (~5-8% annual vol).

A BTC daily return of 0.0005 (0.05%) is deep within noise. A USDJPY daily return
of 0.0005 is a meaningful move (1.5 sigma).

#### Requirements
1. Scale drift threshold by asset volatility:
   ```python
   drift_threshold = DRIFT_THRESHOLD_SIGMA * median_daily_vol
   # DRIFT_THRESHOLD_SIGMA = 0.05 (5% of median daily vol)
   ```
2. Scale vol boundaries by asset volatility history:
   ```python
   vol_high_boundary = np.percentile(vol_history, 75) / np.median(vol_history)
   vol_low_boundary = np.percentile(vol_history, 25) / np.median(vol_history)
   ```
3. This makes regime classification relative to each asset's own distribution.
4. Store computed thresholds in regime metadata for debugging.

#### Acceptance Criteria
- [ ] BTC drift threshold is ~10x larger than SPY drift threshold
- [ ] USDJPY drift threshold is ~3x smaller than SPY drift threshold
- [ ] Regime distribution across assets is more balanced (no class dominance)
- [ ] New test: `test_adaptive_regime_thresholds()` validates scaling

---

### Story 4.4 [DONE]: CUSUM Sensitivity Auto-Tuning
**Priority**: P2 | **Points**: 5 | **File**: src/models/regime.py

#### Context
CUSUM changepoint detection has `threshold=3.0`, `alpha_accel=0.85`, `alpha_normal=0.40`,
and `cooldown=5` bars -- all hardcoded. These control how quickly the system detects
regime transitions.

For fast-moving assets (TSLA), the current sensitivity may be too slow (misses
regime changes by days). For slow-moving assets (GLD), it may be too sensitive
(false regime transitions).

#### Requirements
1. Auto-tune CUSUM threshold based on return distribution:
   ```python
   # threshold = k * sigma_returns (k chosen for desired ARL)
   # Average Run Length (ARL) = expected bars between false alarms
   # Target ARL = 252 (one false alarm per year)
   cusum_threshold = compute_arlk_threshold(returns, target_arl=252)
   ```
2. Auto-tune cooldown based on autocorrelation decay:
   ```python
   # Cooldown = decorrelation time (bars until ACF drops below 0.1)
   cooldown = max(3, int(decorrelation_time(returns)))
   ```
3. Store auto-tuned CUSUM parameters in the regime metadata.

#### Acceptance Criteria
- [ ] TSLA CUSUM threshold is lower than GLD (faster detection)
- [ ] Cooldown is proportional to autocorrelation decay time
- [ ] ARL is approximately 252 bars for all assets (measured empirically)
- [ ] No spurious regime flips for stable assets (GLD, TLT)
- [ ] New test: `test_cusum_auto_tuning()` validates ARL targeting

---

### Story 4.5 [DONE]: Regime Transition Smoothing for Signal Stability
**Priority**: P1 | **Points**: 5 | **File**: signals.py (latest_signals, ~line 8269)

#### Context
When the regime switches (e.g., LOW_VOL_TREND -> HIGH_VOL_TREND), the BMA weights
change discontinuously. This causes the signal to jump even if the underlying
market barely moved. Users see a BUY -> HOLD flip overnight from a minor vol uptick.

#### Requirements
1. Implement exponential smoothing on regime probabilities:
   ```python
   # Load previous regime probs from signal state
   prev_probs = load_signal_state(symbol).get("regime_probs", uniform)
   # EMA smooth
   smooth_probs = REGIME_EMA_ALPHA * current_probs + (1 - REGIME_EMA_ALPHA) * prev_probs
   # REGIME_EMA_ALPHA = 0.3 (30% new, 70% previous)
   ```
2. The smoothed probabilities feed into the BMA weight mixture (Story 4.1).
3. Save smoothed probs in signal state for next run.
4. On first run, use current probs unsmoothed (no history to blend).
5. The smoothing prevents regime-driven signal flips from single-day noise.

#### Acceptance Criteria
- [ ] Signal label changes by <= 20% fewer flips vs current (measured on 252-day window)
- [ ] When regime genuinely changes (sustained for 5+ days), signal follows within 3 days
- [ ] Smoothed probs saved and loaded correctly across runs
- [ ] New test: `test_regime_transition_smoothing()` validates EMA behavior

---

# EPIC 5: SIGNAL LABELING, POSITION SIZING & EXPECTED UTILITY
# Priority: P0 (Critical) | Estimated Effort: 6 Stories | Files: signals.py

## Problem Statement

The signal labeling pipeline has three compounding problems that prevent
profitable signal generation:

**A. EDGE_FLOOR=0.10 kills genuine signals**: The `apply_confirmation_logic()`
function requires `pos_strength >= EDGE_FLOOR` (default 0.10) to escape HOLD.
`pos_strength` is a z-score of expected utility divided by expected loss.
For well-calibrated systems with moderate edge (EU ~0.15%), this z-score is
often < 0.10, forcing HOLD even when the forecast is directionally correct.

**B. Expected utility calculation is dominated by EVT tail**: The EU formula:
```python
EU = p_up * E[gain] - (1 - p_up) * E[loss_EVT]
```
uses EVT-extrapolated left tail for E[loss]. EVT with xi > 0.15 dramatically
inflates E[loss], making EU negative for assets with heavy tails even when
the forecast is strongly bullish.

**C. Position sizing is disconnected from forecast magnitude**: `eu_position_size`
is derived from EU/E[loss], not from the forecast return itself. An asset with
+3% expected return and a +0.5% expected return get similar position sizes
if their EU/E[loss] ratios are similar.

## Impact

The combination of EDGE_FLOOR, EVT-inflated losses, and EU-based sizing means:
- Most signals are HOLD (insufficient edge)
- When signals do fire, position sizes don't reflect conviction
- The system is structurally biased toward inaction

---

### Story 5.1 [DONE]: Adaptive Edge Floor Based on Asset Volatility
**Priority**: P0 | **Points**: 5 | **File**: signals.py (lines 7855-7915)

#### Context
`EDGE_FLOOR` is set via environment variable (default 0.10). This is a fixed
threshold regardless of asset volatility. For SPY (vol ~15%/year), a z-score
of 0.10 requires a meaningful edge. For TSLA (vol ~50%/year), the same z-score
is trivially achievable by noise.

The floor should be proportional to the signal-to-noise ratio, not absolute.

#### Requirements
1. Replace fixed EDGE_FLOOR with volatility-scaled floor:
   ```python
   # In apply_confirmation_logic():
   vol_annual = feats["vol"].iloc[-1] * np.sqrt(252)
   edge_floor = EDGE_FLOOR_Z * vol_annual / np.sqrt(H)
   # EDGE_FLOOR_Z = 0.05 (5% of horizon-scaled vol)
   ```
2. For low-vol assets (currencies), this produces a smaller floor -> more signals.
3. For high-vol assets (crypto/meme stocks), this produces a larger floor -> fewer signals.
4. The scaling uses `sqrt(H)` because edge scales with sqrt of horizon.
5. Store the computed `edge_floor` in signal output for diagnostics.

#### Acceptance Criteria
- [x] USDJPY gets edge_floor ~0.02 (currently 0.10 -- 5x reduction)
- [x] BTC gets edge_floor ~0.15 (currently 0.10 -- 1.5x increase)
- [x] Signal rate (% non-HOLD) increases by >= 30% for currencies
- [x] Signal rate decreases by >= 10% for crypto (higher bar)
- [x] No existing tests broken (edge_floor is configurable, not hardcoded)
- [x] New test: `test_adaptive_edge_floor()` validates scaling formula

---

### Story 5.2 [DONE]: Balanced Expected Utility with Capped EVT Inflation
**Priority**: P0 | **Points**: 5 | **File**: signals.py (latest_signals, ~lines 9020-9070)

#### Context
EVT tail estimation (Generalized Pareto Distribution) extrapolates the left tail
using shape parameter xi. For xi > 0 (heavy tails), the expected shortfall grows
without bound. Typical values: SPY xi ~ 0.15, TSLA xi ~ 0.30.

The EU formula uses EVT-corrected `E[loss]`:
```python
E_loss_evt = E_loss * (1 + evt_xi * EVT_CORRECTION_FACTOR)
EU = p_up * E_gain - (1 - p_up) * E_loss_evt
```

With xi=0.30 and correction factor ~1.5x, E[loss] is inflated 45%. This makes
EU persistently negative for high-vol assets, even when p_up > 0.60.

#### Requirements
1. Cap EVT inflation factor:
   ```python
   EVT_MAX_INFLATION = 1.5  # E_loss can be at most 1.5x the empirical E[loss]
   evt_correction = min(1 + evt_xi * EVT_CORRECTION_FACTOR, EVT_MAX_INFLATION)
   E_loss_evt = E_loss * evt_correction
   ```
2. Also compute a "balanced EU" that uses symmetric tail treatment:
   ```python
   # Symmetric: both gain and loss get EVT correction
   E_gain_evt = E_gain * min(1 + evt_xi * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
   EU_balanced = p_up * E_gain_evt - (1 - p_up) * E_loss_evt
   ```
3. Use `EU_balanced` for position sizing (replaces asymmetric EU).
4. Store both `EU_asymmetric` (old) and `EU_balanced` (new) in signal JSON.

#### Acceptance Criteria
- [x] For TSLA (xi~0.30), E_loss inflation capped at 1.5x (not 1.45x)
- [x] EU_balanced is positive for assets with p_up > 0.55 and moderate vol
- [x] Signal rate (% non-HOLD) increases by >= 20% for high-vol assets
- [x] Both EU values appear in signal output JSON
- [x] New test: `test_balanced_eu()` validates symmetric treatment

---

### Story 5.3 [DONE]: Forecast-Magnitude-Aware Position Sizing
**Priority**: P1 | **Points**: 5 | **File**: signals.py (latest_signals, ~lines 9040-9060)

#### Context
Position sizing uses `eu_position_size = EU / max(abs(E_loss), floor)`, which is
a risk-adjusted metric. But it ignores the **absolute forecast magnitude**.

An asset forecasting +5% at H=21d and another forecasting +0.5% at H=21d may
get similar position sizes if their EU/E[loss] ratios are similar (because
the larger forecast also has larger E[loss]).

For a profit-maximizing system, larger forecasts should command larger positions
(subject to risk limits).

#### Requirements
1. Blend EU-based sizing with forecast-magnitude sizing:
   ```python
   # EU-based size (risk-adjusted)
   size_eu = EU_balanced / max(abs(E_loss), floor)
   # Forecast-magnitude size (conviction-based)
   size_mag = abs(mu_H) / (sig_H + 1e-6)  # Sharpe-like ratio
   # Blend
   position_size = SIZE_EU_WEIGHT * size_eu + SIZE_MAG_WEIGHT * size_mag
   # SIZE_EU_WEIGHT = 0.6, SIZE_MAG_WEIGHT = 0.4
   ```
2. The magnitude component uses `mu_H / sig_H` (information ratio), not raw mu_H.
3. Both components are clipped to [0, 1] before blending.
4. Final position capped at 1.0 (max allocation).

#### Acceptance Criteria
- [x] Asset with +3% forecast gets larger position than asset with +0.5% forecast
- [x] Risk management preserved: volatile assets still get smaller positions via EU
- [x] Position size correlation with absolute forecast > 0.5
- [x] New test: `test_magnitude_position_sizing()` validates blending

---

### Story 5.4 [DONE]: Dynamic Labeling Thresholds from Walk-Forward Hit Rates
**Priority**: P1 | **Points**: 5 | **File**: signals.py (lines 7768-7855, compute_dynamic_thresholds)

#### Context
BUY threshold is clamped to [0.55, 0.70]. SELL threshold to [0.30, 0.45].
These are static ranges. But if walk-forward results (Epic 2, Story 2.1) show that
p_up > 0.58 produces >55% hit rate at H=7d, the buy threshold should be 0.58.

#### Requirements
1. After walk-forward backtest, compute optimal thresholds per horizon:
   ```python
   def optimal_threshold(wf_results, horizon, target_hit_rate=0.55):
       """Find p_up threshold that achieves target hit rate."""
       # Binary search on p_up threshold
       for p_thresh in np.arange(0.50, 0.75, 0.01):
           mask = wf_results["forecast_p_up"] > p_thresh
           if mask.sum() < 20:
               continue
           hit_rate = wf_results.loc[mask, "hit"].mean()
           if hit_rate >= target_hit_rate:
               return p_thresh
       return 0.55  # Default
   ```
2. Store per-horizon thresholds in tune cache under `calibrated_thresholds`.
3. `compute_dynamic_thresholds()` loads calibrated thresholds as starting points.
4. Sell threshold = 1 - buy_threshold (symmetric by default).

#### Acceptance Criteria
- [x] Thresholds vary by horizon (short horizons typically need higher p_up)
- [x] Calibrated thresholds produce hit rate >= 55% on validation set
- [x] Signal count is optimized: not too few (>5%) and not too many (<50%)
- [x] New test: `test_calibrated_thresholds()` validates threshold selection

---

### Story 5.5 [DONE]: Exhaustion Modulation Direction Fix
**Priority**: P1 | **Points**: 3 | **File**: signals.py (~line 9374 area)

#### Context
Exhaustion modulation currently reduces position for BOTH up-exhaustion AND
down-exhaustion:
```python
pos *= (1 - 0.5 * ue_up)    # Reduces long when overbought (correct)
pos *= (1 - 0.5 * ue_down)  # Reduces long when oversold (WRONG)
```

When a stock is oversold (ue_down > 0), a mean-reversion trader WANTS to be long.
The current code reduces the long position when the asset is most attractive from
a mean-reversion perspective.

#### Requirements
1. Fix exhaustion modulation direction:
   ```python
   if mu_H > 0:  # Long signal
       pos *= (1 - 0.5 * ue_up)    # Reduce if overbought
       pos *= (1 + 0.3 * ue_down)  # INCREASE if oversold (mean reversion)
   else:  # Short signal
       pos *= (1 + 0.3 * ue_up)    # INCREASE if overbought
       pos *= (1 - 0.5 * ue_down)  # Reduce if oversold
   ```
2. The increase factor (0.3) is smaller than the decrease (0.5) for asymmetry
   (being cautious about increasing positions is prudent).
3. Position still capped at 1.0 after modulation.
4. Store the modulation factors in signal JSON for diagnostics.

#### Acceptance Criteria
- [x] Oversold stocks with positive forecast get larger positions (not smaller)
- [x] Overbought stocks with positive forecast get smaller positions (unchanged)
- [x] Position never exceeds 1.0 after modulation
- [x] New test: `test_exhaustion_direction()` validates long/short asymmetry

---

### Story 5.6 [DONE]: Kelly Criterion Integration for Optimal Sizing
**Priority**: P2 | **Points**: 8 | **File**: signals.py (latest_signals)

#### Context
The Kelly criterion provides the mathematically optimal bet size for maximizing
long-term geometric growth:
```python
f* = (p * b - q) / b
# where p = P(win), b = E[gain]/E[loss] (odds), q = 1 - p
```

The system already computes `p_up`, `E[gain]`, and `E[loss]` -- all Kelly inputs.
But Kelly sizing is not used for position sizing.

#### Requirements
1. Compute Kelly fraction in `latest_signals()`:
   ```python
   if E_loss > 0:
       odds = E_gain / E_loss
       kelly_full = (p_up * odds - (1 - p_up)) / odds
       kelly_half = kelly_full / 2  # Half-Kelly for safety
   else:
       kelly_half = 0.0
   ```
2. Use Kelly as an alternative sizing signal:
   ```python
   eu_position_size = max(size_blend, kelly_half)  # Kelly provides floor
   ```
3. Store `kelly_full` and `kelly_half` in signal output JSON.
4. Kelly fraction capped at 0.25 (never risk more than 25% of capital on one trade).
5. Negative Kelly (edge < 0) maps to zero position (no trade).

#### Acceptance Criteria
- [x] Kelly fraction is positive when p_up > 0.52 and gain/loss > 1.0
- [x] Kelly fraction is zero or negative when p_up < 0.50
- [x] Half-Kelly provides a reasonable position floor for genuine signals
- [x] Kelly_full and kelly_half appear in signal output JSON
- [x] New test: `test_kelly_sizing()` validates against known (p, b) pairs

---

# EPIC 6: FRONTEND-BACKEND SYNCHRONIZATION & UX EXCELLENCE
# Priority: P1 (High) | Estimated Effort: 6 Stories | Files: frontend, backend, signals_ux.py

## Problem Statement

The frontend (React + TailwindCSS + TradingView Charts) exists but has significant
gaps relative to the terminal UX:

1. **Horizon labels are raw numbers**: Frontend shows `1D`, `3D`, `63D` instead of
   semantic labels `1d`, `3d`, `3m` that the terminal uses.
2. **Profit column missing**: Terminal prominently shows profit in PLN. Frontend only
   shows p_up and exp_ret percentages.
3. **Exhaustion metrics not rendered**: `ue_up` and `ue_down` flow to the frontend
   but are not displayed.
4. **Conviction, Kelly, signal_ttl not typed**: Backend provides these but the
   TypeScript types don't declare them.
5. **WebSocket unused**: `/ws` endpoint and connection manager exist but the frontend
   polls via React Query instead of receiving real-time updates.
6. **Pydantic models unused**: Backend routers return raw dicts, not validated models.

## Impact

Users must use the terminal to see the full picture. The web dashboard is a
degraded view that erodes trust. Apple-quality UX requires the web to exceed
the terminal experience, not lag behind it.

---

### Story 6.1 [DONE]: Semantic Horizon Labels & Responsive Signal Table
**Priority**: P1 | **Points**: 5 | **File**: frontend SignalsPage.tsx, shared types

#### Context
Terminal uses `signals_ux.py` `format_horizon_label()`:
```python
{1: "1d", 3: "3d", 7: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
```
Frontend uses raw `{h}D` format. Users see `63D` instead of `3m`.

#### Requirements
1. Create a shared horizon label map in the frontend:
   ```typescript
   const HORIZON_LABELS: Record<number, string> = {
     1: '1D', 3: '3D', 7: '1W', 21: '1M', 63: '3M', 126: '6M', 252: '12M'
   };
   ```
2. Use this map everywhere horizons are displayed.
3. The signal table should be responsive:
   - Desktop: all 7 horizons visible
   - Tablet: collapse to 5 (1D, 1W, 1M, 3M, 12M)
   - Mobile: collapse to 3 (1W, 1M, 3M)
4. Each signal cell uses a compact format: direction arrow (SVG) + percentage + color.
5. Color coding matches terminal: green for positive, red for negative, dim for near-zero.

#### Acceptance Criteria
- [x] All horizons display semantic labels (3M not 63D)
- [x] Responsive breakpoints work on 768px, 1024px, 1440px widths
- [x] Signal cells have directional SVG arrows (not text arrows or emojis)
- [x] Color coding matches terminal UX (green/red/dim palette)
- [x] Passing Lighthouse accessibility score >= 90

---

### Story 6.2 [DONE]: Profit & P&L Attribution Column
**Priority**: P1 | **Points**: 5 | **File**: frontend SignalsPage.tsx, backend signals router

#### Context
The terminal shows `Profit (PLN)` with formatted amounts like `+12,345 PLN` or
`-3,200 PLN`. This is the most actionable metric for traders: "If I had 1M PLN
in this position, what would I make/lose?"

The data already flows to the frontend via `profit_pln` in `horizon_signals`.

#### Requirements
1. Add profit column to the signal table for each horizon:
   ```typescript
   <td className={profitColor(signal.profit_pln)}>
     {formatPLN(signal.profit_pln)}
   </td>
   ```
2. `formatPLN()` uses locale-aware formatting: `+12,345 PLN` / `-3,200 PLN`.
3. Toggle between `exp_ret %` view and `Profit PLN` view with a tab control.
4. Default view: profit (most actionable for traders).
5. Also display P&L attribution from Story 2.5 when available:
   ```typescript
   <Tooltip content={`Track record: ${pnl.sharpe.toFixed(2)} Sharpe, ${pnl.hit_rate}% hit rate`}>
     <span>{formatPLN(pnl.cumulative_pnl)}</span>
   </Tooltip>
   ```

#### Acceptance Criteria
- [x] Profit column visible with formatted PLN amounts
- [x] Tab toggle switches between % and PLN views
- [x] P&L attribution tooltip shows track record when available
- [x] Positive profits are green, negative are red, zero is neutral
- [x] Number formatting handles large amounts (1M+) with abbreviation

---

### Story 6.3 [DONE]: Exhaustion & Conviction Indicators
**Priority**: P1 | **Points**: 5 | **File**: frontend SignalsPage.tsx, TypeScript types

#### Context
Exhaustion metrics (`ue_up`, `ue_down`) and conviction scores flow to the frontend
but are not rendered. These are critical for timing:
- High `ue_up` means the asset is overbought (caution for new longs)
- High `ue_down` means the asset is oversold (potential buying opportunity)
- Conviction integrates multiple signal quality metrics

#### Requirements
1. Add TypeScript types for all missing fields:
   ```typescript
   interface HorizonSignal {
     // ... existing fields
     ue_up: number;
     ue_down: number;
     position_strength: number;
     risk_temperature: number;
     kelly_half?: number;
     eu_balanced?: number;
   }
   interface SummaryRow {
     // ... existing fields
     conviction?: number;
     kelly?: number;
     signal_ttl?: number;
   }
   ```
2. Render exhaustion as a compact heat indicator:
   - A thin horizontal bar: left side red (ue_down), right side green (ue_up)
   - Intensity proportional to magnitude (0 = invisible, 1 = solid)
3. Render conviction as a filled circle (0-100%):
   - Empty circle = low conviction, filled = high conviction
   - Color: green for high, amber for medium, red for low
4. Add a tooltip on hover with numeric details.

#### Acceptance Criteria
- [x] `ue_up` and `ue_down` visible as heat bar in each signal cell
- [x] Conviction indicator visible per asset row
- [x] TypeScript types match backend response shape exactly
- [x] Tooltips provide exact numeric values on hover
- [x] Design is compact -- does not inflate row height by more than 20%

---

### Story 6.4 [DONE]: Real-Time Signal Updates via WebSocket
**Priority**: P2 | **Points**: 8 | **File**: backend ws.py, frontend hooks/useWebSocket.ts

#### Context
The WebSocket infrastructure exists but is unused. The frontend polls via
`refetchInterval: 30000` (30 seconds). This creates a 0-30 second delay between
signal computation and display.

For a professional trading terminal, real-time updates (< 1 second latency)
are expected. The infrastructure is already there -- just needs wiring.

#### Requirements
1. Backend: After signal computation, broadcast update via WebSocket:
   ```python
   # In process_single_asset(), after signal generation:
   await ws_manager.broadcast({
       "type": "signal_update",
       "symbol": symbol,
       "timestamp": datetime.now().isoformat(),
       "summary": summary_row_dict,
   })
   ```
2. Frontend: Subscribe to WebSocket and update React Query cache:
   ```typescript
   const { lastMessage } = useWebSocket('/ws');
   useEffect(() => {
     if (lastMessage?.type === 'signal_update') {
       queryClient.setQueryData(['signalSummary'], (old) =>
         updateSignalRow(old, lastMessage.summary)
       );
     }
   }, [lastMessage]);
   ```
3. Visual feedback: When a signal updates, briefly highlight the row (fade animation).
4. Connection status indicator: green dot (connected), red dot (disconnected).
5. Automatic reconnection with exponential backoff.

#### Acceptance Criteria
- [x] Signal update appears in < 2 seconds after backend computation
- [x] Updated row has a brief highlight animation (300ms fade)
- [x] Connection status indicator visible in header/status bar
- [x] Automatic reconnection works when backend restarts
- [x] Fallback to polling if WebSocket is unavailable
- [x] New test: WebSocket message format validated

---

### Story 6.5 [DONE]: Interactive Forecast Visualization
**Priority**: P1 | **Points**: 8 | **File**: frontend ChartsPage.tsx, chart components

#### Context
The charts page shows candlesticks and technical indicators but does NOT display
the model's forecast overlay. The forecast data is available via `/api/charts/forecast/{symbol}`
but not rendered as a visual element.

For Apple-quality UX, the forecast should be an intuitive visual:
- A fan chart showing the confidence interval expanding over horizons
- The median forecast as a bold line
- Color gradient from forecast to current price

#### Requirements
1. Fetch forecast data: `useQuery(['forecast', symbol], fetchForecast)`.
2. Overlay on the candlestick chart using TradingView Lightweight Charts:
   ```typescript
   // Fan chart as two area series (upper/lower bounds)
   const upperSeries = chart.addAreaSeries({
     topColor: 'rgba(38, 166, 154, 0.3)',
     bottomColor: 'rgba(38, 166, 154, 0.0)',
   });
   const lowerSeries = chart.addAreaSeries({
     topColor: 'rgba(239, 83, 80, 0.0)',
     bottomColor: 'rgba(239, 83, 80, 0.3)',
   });
   // Median forecast as line series
   const forecastLine = chart.addLineSeries({
     color: '#2196F3', lineWidth: 2, lineStyle: LineStyle.Dashed,
   });
   ```
3. Forecast horizons mapped to future dates from the last candle.
4. Fan width increases with horizon (reflecting growing uncertainty).
5. Interactive: hover on any point shows (date, forecast_price, CI_low, CI_high).
6. Toggle forecast overlay on/off with a button.

#### Acceptance Criteria
- [x] Fan chart visible on candlestick chart for all horizons
- [x] Median line extends from last price to 12M forecast
- [x] CI bands widen with increasing horizon
- [x] Hover tooltip shows price, CI bounds, and p_up
- [x] Toggle button to show/hide forecast overlay
- [x] Works on 1x, 2x, 3x DPI screens (Retina-ready)

---

### Story 6.6 [DONE]: Signal Heatmap Dashboard (Apple-Quality Overview)
**Priority**: P1 | **Points**: 8 | **File**: frontend new component, OverviewPage.tsx

#### Context
The terminal renders a sector-grouped signal heatmap with color-coded cells.
The frontend overview page shows basic statistics but lacks the visual density
of the terminal output.

An Apple-quality dashboard should provide a single-screen overview where a trader
can instantly identify: which sectors are bullish, which assets have strong signals,
and where the risk concentrations lie.

#### Requirements
1. Full-screen heatmap grid: rows = assets (grouped by sector), columns = horizons.
2. Cell color: continuous gradient from deep red (strong sell) through neutral gray
   to deep green (strong buy), based on `exp_ret` at each horizon.
3. Cell intensity: opacity proportional to `position_strength` (high conviction
   signals are vivid, low conviction are washed out).
4. Sector headers: collapsible rows with aggregate sector signal.
5. Asset rows: clickable to navigate to detail page with chart.
6. Corner summary: total portfolio expected P&L, number of active signals.
7. Keyboard shortcuts: arrow keys navigate cells, Enter opens detail.
8. Smooth transitions: when signals update, cells animate color change (300ms).
9. Design language:
   - Glass morphism panels with subtle blur backdrop
   - Inter/SF Pro font stack
   - Consistent spacing grid (4px base)
   - Micro-interactions on hover (subtle scale + shadow)
   - Dark mode primary, light mode secondary

#### Acceptance Criteria
- [x] Full heatmap renders all 100+ assets x 7 horizons in < 200ms
- [x] Color gradient is perceptually uniform (use CIELAB interpolation)
- [x] Sector grouping matches terminal output exactly
- [x] Keyboard navigation works (arrow keys, Enter, Escape)
- [x] Smooth color transitions when data updates
- [x] Lighthouse Performance score >= 90
- [x] Lighthouse Accessibility score >= 95
- [x] Works on 1920x1080 and 3840x2160 without layout issues

---

# EPIC 7: PERFORMANCE OPTIMIZATION & REGRESSION TESTING
# Priority: P1 (High) | Estimated Effort: 5 Stories | Files: tune.py, signals.py, tests/

## Problem Statement

After implementing Epics 1-6, the system will have significantly more computation
per asset (dual-frequency drift, coherent multi-horizon MC, walk-forward calibration,
probabilistic regimes). Without performance optimization, `make tune` and `make stocks`
runtime could double or triple.

Additionally, these changes require a comprehensive regression test suite to ensure
profitability improvements don't introduce calibration regressions.

## Performance Budget

| Operation | Current | Target | Max |
|-----------|---------|--------|-----|
| `make tune` (100 assets) | ~15 min | ~15 min | 25 min |
| `make stocks` (100 assets) | ~8 min | ~5 min | 10 min |
| Single asset signal | ~5 sec | ~2 sec | 5 sec |
| Walk-forward (1 asset, 2y) | N/A | ~3 min | 10 min |
| Frontend initial load | ~2 sec | <1 sec | 2 sec |

---

### Story 7.1 [DONE]: Coherent MC Eliminates 6x Redundant Computation
**Priority**: P0 | **Points**: 5 | **File**: signals.py (latest_signals)

#### Context
Story 3.1 changes from 7 independent MC calls to 1 coherent MC call for H_max=252.
This is inherently a 7x speedup for the MC portion. But the single call simulates
252 steps instead of 1/3/7/21/63/126/252 independently, so the actual speedup
depends on how many paths were generated per horizon.

Currently: 7 calls x 10,000 paths x H steps = 10,000 x (1+3+7+21+63+126+252) = 4.73M path-steps.
Coherent: 1 call x 10,000 paths x 252 steps = 2.52M path-steps.

Net: ~47% reduction in MC computation (1.87x speedup).

#### Requirements
1. Benchmark both approaches on the 12-asset validation universe.
2. Ensure Numba kernel compilation cache is warm (first run is slow due to JIT).
3. Verify that cumulative output extraction adds negligible overhead.
4. Profile any bottleneck in the Python wrapper (numpy slicing, etc.).

#### Acceptance Criteria
- [x] Signal generation per asset < 3 seconds (from current ~5 seconds)
- [x] MC path-steps reduced by >= 40%
- [x] Numba kernel recompilation not triggered by parameter changes
- [x] Benchmark results documented with mean/std per asset

---

### Story 7.2 [DONE]: Parallel Walk-Forward with Smart Caching
**Priority**: P1 | **Points**: 8 | **File**: signals.py (run_walk_forward_backtest)

#### Context
Walk-forward backtest (Story 2.1) must be fast enough for the `make calibrate` pipeline.
Naive implementation calls `latest_signals()` at every rebalance date, which is expensive.

Smart caching opportunities:
- Features computation is expensive but many features are cumulative (expanding window).
  Computing features at date t+5 shares 99% of work with date t.
- Tune cache is constant within a `retune_freq` window.
- MC simulation can be batched across rebalance dates.

#### Requirements
1. Implement incremental feature computation:
   ```python
   # Cache features from previous rebalance date
   prev_features = feature_cache.get(symbol, None)
   if prev_features and prev_features.date == t - rebalance_freq:
       features = update_features_incremental(prev_features, new_prices)
   else:
       features = compute_features(prices_as_of_t, ...)
   ```
2. Parallelize walk-forward across assets (not within a single asset's timeline):
   ```python
   with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
       results = pool.map(run_walk_forward_single, asset_configs)
   ```
3. Cache tune params within their validity window.
4. Target: walk-forward for 12 assets completes in < 30 minutes.

#### Acceptance Criteria
- [x] Walk-forward for SPY (2y, rebalance=5d) completes in < 5 minutes
- [x] All 12 validation assets complete in < 30 minutes (parallel)
- [x] Incremental features produce identical results to full recomputation
- [x] Cache hit rate > 80% for tune params within retune window
- [x] New test: `test_incremental_features()` validates correctness

---

### Story 7.3 [DONE]: Profitability Regression Test Suite
**Priority**: P0 | **Points**: 8 | **File**: src/tests/test_profitability_regression.py (new)

#### Context
Each Epic introduces changes that should improve profitability. Without regression
tests, a future change could silently degrade performance. We need a test suite
that validates core profitability metrics on the validation universe.

#### Requirements
1. New test file `src/tests/test_profitability_regression.py`:
   ```python
   class TestProfitabilityRegression(unittest.TestCase):
       """
       Validates that system-level profitability metrics meet minimum bars.
       Uses cached walk-forward results from make calibrate.
       """

       def test_directional_accuracy_7d(self):
           """Hit rate at H=7d must exceed 53% on validation universe."""
           wf = load_walk_forward_results()
           hit_rate = wf[wf["horizon"] == 7]["hit"].mean()
           self.assertGreater(hit_rate, 0.53,
               f"7d hit rate {hit_rate:.3f} below 53% minimum")

       def test_directional_accuracy_21d(self):
           """Hit rate at H=21d must exceed 52% on validation universe."""

       def test_signal_differentiation(self):
           """At least 15% of signals must be non-HOLD."""

       def test_crps_regression(self):
           """CRPS must not increase from baseline by more than 5%."""

       def test_calibration_ece(self):
           """ECE must be below 0.05 across all horizons."""

       def test_sharpe_positive(self):
           """Signal-following Sharpe ratio must be positive."""

       def test_no_catastrophic_loss(self):
           """No single asset should have P&L drawdown > 30%."""

       def test_forecast_monotonicity(self):
           """Variance must increase with horizon (no inversions)."""
   ```
2. Tests load pre-computed walk-forward results from `src/data/calibration/`.
3. If walk-forward data doesn't exist, tests are skipped (not failed).
4. Baseline metrics stored in `src/tests/profitability_baseline.json`.
5. `make test-profit` runs only profitability tests.

#### Acceptance Criteria
- [x] Test suite has >= 8 test methods covering all key metrics
- [x] Tests skip gracefully when walk-forward data is unavailable
- [x] Baseline JSON establishes current performance level
- [x] `make test-profit` target works
- [x] Tests run in < 10 seconds (use cached walk-forward data)

---

### Story 7.4 [DONE]: Numba Kernel Performance Audit
**Priority**: P2 | **Points**: 5 | **File**: src/models/numba_kernels.py

#### Context
The Numba kernels (`unified_mc_simulate_kernel`, `unified_mc_multi_path_kernel`)
are performance-critical. After adding dual-frequency drift (Story 1.4) and
asset-class return caps (Story 3.4), they may have performance regressions.

#### Requirements
1. Profile both kernels with `numba.core.compiler.Flags(force_pyobject=False)`.
2. Verify no Python object mode fallbacks in the hot loop.
3. Add explicit `@njit(cache=True)` to ensure caching across Python restarts.
4. Benchmark: simulate 10,000 x 252 paths in < 200ms.
5. If dual-frequency drift adds > 10% overhead, investigate vectorization:
   ```python
   # Batch propagate both drift components:
   mu_fast_all = phi_fast ** np.arange(H) * mu_fast_0
   mu_slow_all = phi_slow ** np.arange(H) * mu_slow_0
   ```

#### Acceptance Criteria
- [x] No Python object mode fallbacks in Numba kernels
- [x] 10,000 x 252 simulation completes in < 200ms
- [x] Cache hits confirmed (second run is >= 10x faster than first)
- [x] Dual-frequency drift adds < 15% overhead vs single-frequency

---

### Story 7.5 [DONE]: Signal Output Validation Invariants
**Priority**: P1 | **Points**: 3 | **File**: signals.py (latest_signals, end of function)

#### Context
After all the changes in Epics 1-6, the signal output must satisfy fundamental
invariants. These should be checked at the end of `latest_signals()` as runtime
assertions (debug mode) or lightweight checks (production mode).

#### Requirements
1. Add validation block at end of `latest_signals()`:
   ```python
   # Signal output invariants
   for H in horizons:
       sig = signals[H]
       assert np.isfinite(sig.exp_ret), f"exp_ret is not finite for H={H}"
       assert np.isfinite(sig.p_up), f"p_up is not finite for H={H}"
       assert 0 <= sig.p_up <= 1, f"p_up={sig.p_up} out of [0,1] for H={H}"
       assert sig.ci_low <= sig.exp_ret <= sig.ci_high, \
           f"CI inversion at H={H}: [{sig.ci_low}, {sig.ci_high}]"
       assert sig.sig_H >= 0, f"Negative sig_H={sig.sig_H} for H={H}"
       if H > 1:
           assert sig.sig_H >= signals[H_prev].sig_H * 0.5, \
               f"sig_H decreased dramatically from H={H_prev} to H={H}"
   ```
2. In production mode, violations log warnings instead of assertion errors.
3. In debug mode (`--debug` flag), violations raise AssertionError.
4. Count and report violations in the signal summary table.

#### Acceptance Criteria
- [x] All invariants pass for 12 validation assets across all horizons
- [x] CI inversion detection catches any parametric/quantile mismatch
- [x] sig_H monotonicity is enforced (variance grows with horizon)
- [x] Violation count is 0 for a properly calibrated system
- [x] New test: `test_signal_invariants()` validates all invariant checks

---

# EPIC 8: END-TO-END PROFITABILITY VALIDATION
# Priority: P0 (Critical) | Estimated Effort: 3 Stories | Files: All

## Problem Statement

Individual improvements (better drift, calibrated EMOS, coherent MC) need
validation as a complete system. Improving drift without calibration may
increase hit rate but degrade CRPS. Improving CRPS without directional
accuracy may improve calibration but not profitability.

The final validation must measure the system holistically: run the full
`make tune` + `make stocks` pipeline on the validation universe and measure
end-to-end metrics.

---

### Story 8.1 [DONE]: Validation Universe & Benchmark Definition
**Priority**: P0 | **Points**: 3 | **File**: src/tests/validation_config.py (new)

#### Context
The validation universe needs to be formally defined with benchmark expectations.

#### Requirements
1. Create `src/tests/validation_config.py`:
   ```python
   VALIDATION_UNIVERSE = {
       # Large Cap (liquid, well-modeled)
       "AAPL": {"sector": "Technology", "expected_vol": 0.25, "class": "equity"},
       "NVDA": {"sector": "Technology", "expected_vol": 0.40, "class": "equity"},
       "MSFT": {"sector": "Technology", "expected_vol": 0.22, "class": "equity"},
       "GOOGL": {"sector": "Technology", "expected_vol": 0.28, "class": "equity"},
       # High Vol (challenging)
       "TSLA": {"sector": "Auto", "expected_vol": 0.55, "class": "equity"},
       "CRWD": {"sector": "Cybersecurity", "expected_vol": 0.45, "class": "equity"},
       "DKNG": {"sector": "Gaming", "expected_vol": 0.50, "class": "equity"},
       "COIN": {"sector": "Crypto", "expected_vol": 0.70, "class": "equity"},
       # Index/ETF (baseline)
       "SPY": {"sector": "Index", "expected_vol": 0.16, "class": "etf"},
       "QQQ": {"sector": "Index", "expected_vol": 0.22, "class": "etf"},
       "GLD": {"sector": "Metals", "expected_vol": 0.15, "class": "metal"},
       "TLT": {"sector": "Bonds", "expected_vol": 0.18, "class": "etf"},
   }

   PROFITABILITY_TARGETS = {
       "hit_rate_7d": 0.55,        # >55% at 1 week
       "hit_rate_21d": 0.53,       # >53% at 1 month
       "signal_rate": 0.15,        # >15% non-HOLD
       "sharpe_7d": 0.50,          # Sharpe > 0.5 at 7d
       "crps_improvement": 0.20,   # >20% CRPS reduction vs baseline
       "ece_max": 0.03,            # ECE < 3%
       "max_drawdown_single": 0.30, # No asset > 30% drawdown
   }
   ```
2. Baseline metrics computed from the CURRENT system (before Epics 1-6).
3. Every metric must have a defined measurement methodology.

#### Acceptance Criteria
- [x] 12 validation symbols defined with expected characteristics
- [x] Profitability targets are quantitative and measurable
- [x] Baseline metrics computed and stored in validation_baseline.json
- [x] Configuration importable by all test modules

---

### Story 8.2 [DONE]: Full Pipeline Integration Test
**Priority**: P0 | **Points**: 13 | **File**: src/tests/test_full_pipeline.py (new)

#### Context
The ultimate test: run `make tune` + `make stocks` on the validation universe
and verify all profitability targets are met.

#### Requirements
1. New test file `src/tests/test_full_pipeline.py`:
   ```python
   @unittest.skipUnless(os.environ.get("RUN_INTEGRATION"), "Integration test")
   class TestFullPipeline(unittest.TestCase):

       @classmethod
       def setUpClass(cls):
           """Run full tune + signal pipeline on validation universe."""
           # This takes ~20 minutes -- only run in CI or manual
           cls.tune_result = run_tune(VALIDATION_UNIVERSE.keys())
           cls.signal_result = run_signals(VALIDATION_UNIVERSE.keys())
           cls.wf_result = run_walk_forward(VALIDATION_UNIVERSE.keys())

       def test_all_assets_produce_signals(self):
           """Every validation asset must produce signals for all horizons."""
           for symbol in VALIDATION_UNIVERSE:
               for H in [1, 3, 7, 21, 63, 126, 252]:
                   self.assertIn(H, self.signal_result[symbol])

       def test_no_flat_signals(self):
           """No asset should have ALL horizons showing +0.0%."""
           for symbol in VALIDATION_UNIVERSE:
               forecasts = [self.signal_result[symbol][H].exp_ret for H in [1,7,21]]
               max_abs = max(abs(f) for f in forecasts)
               self.assertGreater(max_abs, 0.001,
                   f"{symbol} has flat signals: {forecasts}")

       def test_directional_differentiation(self):
           """Different assets should have different signal directions."""
           directions = {}
           for symbol in VALIDATION_UNIVERSE:
               directions[symbol] = self.signal_result[symbol][7].exp_ret
           unique_signs = len(set(np.sign(d) for d in directions.values()))
           self.assertGreater(unique_signs, 1,
               "All assets have same direction -- no differentiation")

       def test_hit_rate_targets(self):
           """Walk-forward hit rates meet profitability targets."""
           for H, target in [(7, 0.55), (21, 0.53)]:
               hit_rate = self.wf_result[self.wf_result["horizon"]==H]["hit"].mean()
               self.assertGreater(hit_rate, target,
                   f"H={H}d hit rate {hit_rate:.3f} below target {target}")

       def test_signal_rate_target(self):
           """At least 15% of signals should be non-HOLD."""
           labels = [self.signal_result[s][7].label for s in VALIDATION_UNIVERSE]
           non_hold = sum(1 for l in labels if l != "HOLD") / len(labels)
           self.assertGreater(non_hold, 0.15)

       def test_crps_improvement(self):
           """CRPS must improve vs baseline."""

       def test_calibration_quality(self):
           """ECE must be below target."""

       def test_no_catastrophic_forecasts(self):
           """No forecast should exceed physical bounds."""
           for symbol in VALIDATION_UNIVERSE:
               for H in [1, 3, 7, 21, 63, 126, 252]:
                   exp_ret = self.signal_result[symbol][H].exp_ret
                   # No daily return > 100% or < -100%
                   self.assertGreater(exp_ret, -2.3,
                       f"{symbol} H={H} forecast {exp_ret} is catastrophic")
                   self.assertLess(exp_ret, 1.6,
                       f"{symbol} H={H} forecast {exp_ret} exceeds +400%")
   ```
2. Makefile target: `make test-integration` runs with `RUN_INTEGRATION=1`.
3. Test output includes a summary table comparing metrics to targets.
4. Failed metrics are reported with deviation from target (not just pass/fail).

#### Acceptance Criteria
- [x] All 12 validation assets produce signals for all 7 horizons
- [x] No flat signals (max_abs(exp_ret) > 0.001 for every asset)
- [x] Directional differentiation: at least 2 different sign groups
- [x] Hit rate targets met (55% at 7d, 53% at 21d)
- [x] Signal rate >= 15% non-HOLD
- [x] No catastrophic forecasts (within physical bounds)
- [x] Integration test runs in < 30 minutes
- [x] `make test-integration` target works

---

### Story 8.3 [DONE]: Continuous Profitability Monitoring Dashboard
**Priority**: P2 | **Points**: 8 | **File**: frontend new page, backend new endpoint

#### Context
After achieving profitability targets, we need to monitor them continuously.
A dedicated dashboard page shows historical performance metrics over time.

#### Requirements
1. Backend endpoint `/api/diagnostics/profitability`:
   ```python
   @router.get("/profitability")
   async def get_profitability_metrics():
       """Return time-series of profitability metrics."""
       return {
           "timestamps": [...],   # Dates of each calibration run
           "hit_rates": {"7d": [...], "21d": [...]},
           "signal_rates": [...],
           "sharpe": {"7d": [...], "21d": [...]},
           "crps": [...],
           "ece": [...],
           "targets": PROFITABILITY_TARGETS,
       }
   ```
2. Frontend page: `/diagnostics/profitability`
   - Line charts for each metric over time (Recharts)
   - Target lines overlaid on each chart (dashed red/green)
   - Current value vs target in header cards
   - Alert indicator when any metric falls below target
3. Automatically updated after each `make calibrate` run.
4. Historical data stored in `src/data/calibration/profitability_history.json`.

#### Acceptance Criteria
- [x] Dashboard shows 6+ metrics over time
- [x] Target lines clearly visible on each chart
- [x] Current values highlighted with green (pass) / red (fail)
- [x] Chart data loads in < 500ms
- [x] Responsive design for desktop and tablet
- [x] Accessible: all charts have ARIA labels

---

# APPENDIX A: IMPLEMENTATION ORDER

The Epics should be implemented in dependency order:

```
Phase 1 (Weeks 1-3): Foundation -- Signal Quality
  Epic 1, Stories 1.1, 1.2, 1.4 (Drift amplification + dual-frequency)
  Epic 3, Story 3.1 (Coherent multi-horizon MC)
  Epic 3, Story 3.2 (Quantile CIs)
  Epic 5, Stories 5.1, 5.2 (Edge floor + balanced EU)

Phase 2 (Weeks 4-6): Calibration -- Learn from Mistakes
  Epic 2, Story 2.1 (Walk-forward infrastructure)
  Epic 2, Stories 2.2, 2.3 (EMOS + vol calibration)
  Epic 2, Story 2.6 (make calibrate pipeline)
  Epic 5, Story 5.4 (Calibrated thresholds)

Phase 3 (Weeks 7-9): Robustness -- Regime & Model Quality
  Epic 1, Stories 1.3, 1.5, 1.6 (Cross-asset phi, continuous nu, multi-start GARCH)
  Epic 4, Stories 4.1, 4.2, 4.5 (Probabilistic regimes, DRY, transition smoothing)
  Epic 3, Story 3.3 (Confirmation fix)
  Epic 5, Story 5.5 (Exhaustion direction fix)

Phase 4 (Weeks 10-12): UX & Validation -- Ship Quality
  Epic 6, Stories 6.1-6.6 (Frontend excellence)
  Epic 7, Stories 7.1-7.5 (Performance & regression tests)
  Epic 8, Stories 8.1-8.3 (End-to-end validation)

Phase 5 (Weeks 13+): Refinement
  Epic 1, Stories 1.7, 1.8 (Q floor calibration, phi from acf)
  Epic 2, Stories 2.4, 2.5 (Directional scoring, P&L attribution)
  Epic 3, Stories 3.4, 3.5, 3.6 (Asset-class caps, importance sampling, diagnostics)
  Epic 4, Stories 4.3, 4.4 (Adaptive thresholds, CUSUM auto-tune)
  Epic 5, Story 5.6 (Kelly criterion)
```

---

# APPENDIX B: KEY METRICS GLOSSARY

| Metric | Definition | Target | Current |
|--------|-----------|--------|---------|
| **Hit Rate** | P(sign(forecast) == sign(realized)) | >55% at 7d | ~50% |
| **Signal Rate** | % of signals that are BUY or SELL (not HOLD) | >15% | <5% |
| **Sharpe Ratio** | mean(PnL) / std(PnL) * sqrt(252/H) | >0.5 | ~0 |
| **CRPS** | Continuous Ranked Probability Score (lower=better) | -20% vs baseline | baseline |
| **ECE** | Expected Calibration Error (|p_forecast - p_realized|) | <0.03 | ~0.10 |
| **Max Drawdown** | Worst cumulative loss from peak | <30% per asset | unmeasured |
| **Forecast Variance Ratio** | realized_vol / forecast_vol | 0.8-1.2 | unmeasured |
| **DIG** | Directional Information Gain = hit_rate - 0.5 | >0.05 | ~0 |
| **Kelly Fraction** | Optimal bet size (p*b-q)/b | >0.05 | unmeasured |
| **Signal Flip Rate** | % of days where label changes | <20% | unmeasured |

---

# APPENDIX C: FILE CHANGE MATRIX

| Story | tune.py | signals.py | numba_kernels.py | Frontend | Tests |
|-------|---------|------------|------------------|----------|-------|
| 1.1 | | MODIFY | | | NEW |
| 1.2 | | MODIFY | | | NEW |
| 1.3 | MODIFY | | | | NEW |
| 1.4 | | MODIFY | MODIFY | | NEW |
| 1.5 | MODIFY | | | | NEW |
| 1.6 | MODIFY | | | | NEW |
| 1.7 | MODIFY | | | | NEW |
| 1.8 | MODIFY | MODIFY | | | NEW |
| 2.1 | | NEW+MODIFY | | | NEW |
| 2.2 | MODIFY | MODIFY | | | NEW |
| 2.3 | MODIFY | MODIFY | | | NEW |
| 2.4 | MODIFY | | | | NEW |
| 2.5 | | MODIFY | | | NEW |
| 2.6 | MODIFY | | | | NEW |
| 3.1 | | MODIFY | | | NEW |
| 3.2 | | MODIFY | | | NEW |
| 3.3 | | MODIFY | | | NEW |
| 3.4 | | MODIFY | MODIFY | | NEW |
| 3.5 | | MODIFY | | | NEW |
| 3.6 | | NEW | | | NEW |
| 4.1 | MODIFY | MODIFY | | | NEW |
| 4.2 | MODIFY | MODIFY | | | NEW |
| 4.3 | | | | | NEW |
| 4.4 | | | | | NEW |
| 4.5 | | MODIFY | | | NEW |
| 5.1 | | MODIFY | | | NEW |
| 5.2 | | MODIFY | | | NEW |
| 5.3 | | MODIFY | | | NEW |
| 5.4 | | MODIFY | | | NEW |
| 5.5 | | MODIFY | | | NEW |
| 5.6 | | MODIFY | | | NEW |
| 6.1 | | | | MODIFY | |
| 6.2 | | | | MODIFY | |
| 6.3 | | | | MODIFY | |
| 6.4 | | | | MODIFY+NEW | |
| 6.5 | | | | MODIFY+NEW | |
| 6.6 | | | | NEW | |
| 7.1 | | MODIFY | | | NEW |
| 7.2 | | MODIFY | | | NEW |
| 7.3 | | | | | NEW |
| 7.4 | | | MODIFY | | NEW |
| 7.5 | | MODIFY | | | NEW |
| 8.1 | | | | | NEW |
| 8.2 | | | | | NEW |
| 8.3 | | | | NEW | |

**Total: 42 stories across 8 Epics**
- tune.py modifications: 12 stories
- signals.py modifications: 27 stories
- numba_kernels.py modifications: 4 stories
- Frontend modifications: 8 stories
- New test files: 38 stories

---

# END OF BACKLOG
