# Quant.md -- Signal Engine Transformation Program
# ================================================================
# Product Backlog: From Near-Zero Forecasts to Massively Profitable,
# Accurate, and Performant Quantitative Signal Engine
# ================================================================
#
# Author: Quant Product Owner
# Date: 2 April 2026
# Version: 1.0
#
# CONTEXT:
# The current system produces calibrated but perceptually useless forecasts.
# Typical output: +0.0%, +0.0%, +0.1%, +0.7%, +0.6%, +2.2%, +2.9%
# across 1d/3d/1w/1m/3m/6m/12m horizons.
#
# ROOT CAUSES IDENTIFIED:
# 1. Kalman drift collapse: q ~ 1e-6 locks filter to zero-drift prior (60%)
# 2. Ensemble over-averaging: 5 models cancel directional signals (25%)
# 3. Hard caps binding in low-vol regimes: vol-bound < display threshold (10%)
# 4. Display rounding: 0.047% renders as +0.0% (5%)
#
# THIS PROGRAM ADDRESSES:
# 1. Forecast accuracy -- directional, calibrated, and actionable
# 2. Backend/frontend synchronization -- real-time, coherent data flow
# 3. World-class UX -- Apple-caliber design that makes engineers envious
# 4. Performance -- sub-second tuning feedback, parallel signal generation
# 5. Validation -- tested on benchmark universe before production rollout
#
# BENCHMARK VALIDATION UNIVERSE (12 symbols):
# Small Cap:  UPST, AFRM, IONQ
# Mid Cap:    CRWD, DKNG, SNAP
# Large Cap:  AAPL, NVDA, TSLA
# Index:      SPY, QQQ, IWM
#
# All stories must pass validation on this universe before merge.
# ================================================================

---

## STORY QUALITY STANDARDS

Every story in this document adheres to these non-negotiable standards:

1. **Measurable Acceptance Criteria**: Every AC specifies a testable condition with
   explicit thresholds, not qualitative language ("improves" -> "improves by >= 3%").

2. **Anti-Leakage Discipline**: Any story that uses historical data for optimization
   MUST separate train/test temporally. No future data may leak into any computation.
   Stories that touch calibration feedback explicitly call out the anti-leakage boundary.

3. **Mathematical Correctness**: All formulas are complete with defined variables,
   boundary conditions, and edge case handling. No hand-waving.

4. **Benchmark Gating**: No story ships without validation on the 12-symbol benchmark
   universe. "Works on SPY" is insufficient -- it must work across cap sizes, sectors,
   and volatility regimes.

5. **Calibration Preservation**: Forecast accuracy improvements MUST NOT degrade PIT
   calibration or ECE beyond stated tolerances. Accuracy at the cost of calibration
   is a false improvement.

6. **Graceful Degradation**: Every new feature has a fallback path. Missing data,
   failed computation, or unavailable dependency must produce a warning, not a crash.

7. **Idempotency**: Running any pipeline step twice with the same inputs produces
   identical outputs. No side effects, no randomness without seeded RNG.

---

## CROSS-CUTTING CONCERNS (APPLY TO ALL STORIES)

**Definition of Done (per story):**
- [ ] Code reviewed and merged to main
- [ ] All acceptance criteria verified with automated tests
- [ ] Benchmark validation run on 12-symbol universe (results in validation JSON)
- [ ] No PIT calibration regression (ECE delta < 0.02)
- [ ] Terminal and frontend display identical values (parity test passes)
- [ ] Error paths tested: invalid input, missing data, NaN propagation
- [ ] Performance: no regression in tune or signal generation time (< 10% increase)

**Anti-Patterns to Avoid:**
- Amplifying forecasts without checking calibration (creates overconfident signals)
- Using lookahead data in any computation (destroys backtest validity)
- Hard-coding parameters that vary across assets or regimes (use the tune cache)
- Displaying raw model output without formatting (users see numbers, not distributions)
- Silent fallbacks (every fallback must log a WARNING with context)

---

## EPIC 1: KALMAN DRIFT RESCUE -- STOP THE ZERO-FORECAST COLLAPSE

**Epic Owner:** Quant Engine Lead
**Priority:** P0 -- Critical Path
**Estimated Scope:** 14 stories, ~45 tasks

**Problem Statement:**
The system's primary forecast engine (Kalman filter with BMA) produces near-zero
drift estimates because the tuned process noise q converges to ~1e-6. At this level,
the Kalman gain K_t approaches zero, causing the filter to trust its prior (zero drift)
over observations. The result: mu_t decays toward zero exponentially, and all horizon
forecasts collapse.

**Mathematical Root Cause:**
```
Given: q = 1e-6, c = 1.0, phi = 0.95
Kalman gain: K_t = P_t|t-1 / (P_t|t-1 + c)
Steady-state P: P_ss = (-c + sqrt(c^2 + 4*q*c)) / 2 ~ sqrt(q*c) ~ 0.001
Therefore: K_ss ~ 0.001 / 1.001 ~ 0.001
Innovation impact: K_ss * (y_t - phi*mu_t) ~ 0.001 * innovation
Result: 99.9% of each observation is DISCARDED
```

**Success Criteria:**
- Benchmark universe 1-day directional accuracy > 55% (currently ~50.5%)
- 7-day directional accuracy > 58%
- 30-day directional accuracy > 60%
- No forecast displays as +0.0% when the underlying signal exceeds 0.05%
- PIT calibration ECE < 0.05 maintained (no accuracy at cost of calibration)
- Tuning time does not increase by more than 20%

---

### Story 1.1: Adaptive Process Noise Floor with Regime Conditioning

**As a** signal consumer,
**I want** the Kalman filter to maintain meaningful drift sensitivity even in low-volatility regimes,
**so that** forecasts reflect genuine directional information instead of collapsing to zero.

**Background:**
The current tuning procedure optimizes q via MLE/BIC, which correctly identifies that
a tiny q explains the data best (returns are mostly noise). However, "best statistical fit"
and "most useful forecast" are different objectives. We need a floor on q that preserves
forecast utility without sacrificing too much calibration quality.

**Dependencies:** None (foundational change, no prerequisites)

**Risk:** If q_floor is too high, the filter becomes excessively reactive, tracking noise
as if it were signal. This degrades calibration (ECE increases). Mitigation: regime-
conditioning ensures floors are proportional to expected drift magnitude, and BIC
adjustment penalizes overly aggressive floors.

**Acceptance Criteria:**
1. A regime-conditional q floor is implemented in `tune.py`:
   - LOW_VOL_TREND: q_floor = 5e-5 (drift matters here)
   - HIGH_VOL_TREND: q_floor = 1e-4 (fast-moving drift)
   - LOW_VOL_RANGE: q_floor = 2e-5 (small but nonzero)
   - HIGH_VOL_RANGE: q_floor = 5e-5
   - CRISIS_JUMP: q_floor = 5e-4 (maximum adaptivity)
2. Optimized q is clipped: q_final = max(q_mle, q_floor)
3. BIC penalty is adjusted: BIC_adj = BIC + lambda * log(q_floor / q_mle) when floor binds
4. Benchmark validation: directional accuracy improves by >= 3% on 7-day horizon
5. PIT calibration ECE remains < 0.06 (slight relaxation allowed)
6. Unit tests cover all 5 regime floors with synthetic data

**Tasks:**
- [ ] 1.1.1: Add Q_FLOOR_BY_REGIME dict to `src/tuning/tune.py` with 5 regime entries
- [ ] 1.1.2: Modify `optimize_params_gaussian()` to apply floor after MLE optimization
- [ ] 1.1.3: Modify `optimize_params_phi_gaussian()` with same floor logic
- [ ] 1.1.4: Modify `optimize_params_phi_student_t()` with same floor logic
- [ ] 1.1.5: Add BIC adjustment term when floor binds (penalizes overly aggressive floor)
- [ ] 1.1.6: Log floor-binding events: count and percentage per asset per regime
- [ ] 1.1.7: Add tune output field `q_floor_applied: bool` to JSON cache
- [ ] 1.1.8: Write unit test: synthetic trending series, verify q >= q_floor
- [ ] 1.1.9: Write unit test: synthetic ranging series, verify q >= q_floor but smaller
- [ ] 1.1.10: Run benchmark validation on 12-symbol universe, report directional accuracy
- [ ] 1.1.11: Update `signals.py` to read `q_floor_applied` flag from tuned params

---

### Story 1.2: Kalman Gain Monitoring and Adaptive Reset

**As a** quant researcher,
**I want** the Kalman filter to detect when its gain has collapsed below a useful threshold
and automatically reset its state estimate,
**so that** the filter does not become permanently "stuck" on a stale drift estimate.

**Background:**
Once K_t < 0.01, the filter ignores >99% of new observations. If the market regime
changes (e.g., from range-bound to trending), the filter cannot adapt for 50-100+ bars.
An adaptive reset mechanism detects this condition and injects a controlled perturbation.

**Dependencies:** Story 1.1 (Adaptive Process Noise Floor -- provides meaningful K_t baseline)

**Risk:** Overly aggressive resets (low threshold or high inflation) cause filter oscillation
where P_t never converges. Mitigation: maximum 3 resets per 252-bar window and cool-down
period of 20 bars after each reset.

**Acceptance Criteria:**
1. Kalman gain K_t is tracked during filtering in `signals.py`
2. When K_t < K_MIN_THRESHOLD (default 0.005) for GAIN_STALL_WINDOW consecutive bars:
   - State variance P is inflated: P_new = P_old * RESET_INFLATION_FACTOR (default 10.0)
   - This increases K_t, allowing the filter to relearn
3. Reset events are counted and logged per asset
4. Maximum 3 resets per 252-bar window (prevents oscillation)
5. Benchmark: filter recovers to K_t > 0.02 within 5 bars after reset
6. Directional accuracy improvement of >= 2% on crisis-period subsample

**Tasks:**
- [ ] 1.2.1: Add K_t tracking to Kalman filter loop in `signals.py`
- [ ] 1.2.2: Implement stall detection: rolling window of K_t < threshold
- [ ] 1.2.3: Implement P-inflation reset mechanism with configurable factor
- [ ] 1.2.4: Add reset counter with per-window maximum (anti-oscillation)
- [ ] 1.2.5: Add reset event logging with timestamp and K_t values
- [ ] 1.2.6: Write test: synthetic regime change at bar 200, verify reset triggers
- [ ] 1.2.7: Write test: verify max 3 resets per window constraint
- [ ] 1.2.8: Benchmark on crisis subsamples (COVID Mar 2020, Oct 2023 correction)

---

### Story 1.3: GAS-Q Effectiveness Audit and Propagation to Market Temperature

**As a** signal consumer,
**I want** the existing GAS-Q integration in `signals.py` verified for effectiveness
and the same time-varying q_t propagated into `market_temperature.py` ensemble forecasts,
**so that** the adaptive process noise benefits the user-facing forecast display.

**Background:**
GAS-Q is already fitted during tuning (omega, alpha_gasq, beta_gasq stored per asset)
and `signals.py` already loads these params when `gas_q_augmented: true`. The filter
uses time-varying q_t via `gas_q_filter_gaussian_kernel` and `gas_q_filter_student_t`.
However, `market_temperature.py` (the actual forecast display engine) does NOT use
GAS-Q -- it uses static EMA-based drift. This is the real gap: the tuning effort is
captured in signals.py but NOT propagated to the forecasts users actually see.

**Current State (already implemented):**
```
tune.py -> fits GAS-Q params -> stores gas_q_params in JSON
signals.py -> loads gas_q_params -> runs gas_q_filter_*_kernel -> BMA signals
market_temperature.py -> IGNORES all tuned params -> uses hardcoded EMA drift
```

**Target State:**
```
market_temperature._kalman_forecast() -> loads gas_q_params -> uses time-varying q_t
```

**Dependencies:** Story 1.7 (Market Temperature uses tuned params)

**Acceptance Criteria:**
1. Effectiveness audit: measure directional accuracy WITH vs WITHOUT GAS-Q on signals.py path
2. If GAS-Q improves accuracy by < 1%: investigate parameterization, bounds, initialization
3. GAS-Q q_t bounds verified: q_min = 1e-7, q_max = 1e-2 (already in code -- verify)
4. `_kalman_forecast()` in market_temperature.py accepts and uses GAS-Q params
5. Time-varying q_t feeds into Kalman filter within ensemble forecast path
6. Fallback to static q when GAS-Q params missing (graceful degradation)
7. Benchmark: forecast display accuracy improves by >= 1.5% at 7-day horizon
8. GAS-Q q_t trajectory logged for diagnostics (mean, std, max per asset)

**Risk:** GAS-Q with poorly tuned alpha/beta can cause q_t oscillation, leading to
filter instability. Mitigation: exponential smoothing on q_t trajectory with damping.

**Tasks:**
- [ ] 1.3.1: Audit existing GAS-Q integration: run A/B on 12 symbols with/without
- [ ] 1.3.2: Log q_t trajectory statistics per asset (mean, std, min, max, % at bounds)
- [ ] 1.3.3: If accuracy gain < 1%: diagnose -- check q_t bounds, alpha/beta magnitudes
- [ ] 1.3.4: Propagate GAS-Q params into `_kalman_forecast()` in market_temperature.py
- [ ] 1.3.5: Add time-varying q_t computation inside ensemble Kalman path
- [ ] 1.3.6: Add q_t damping: q_t_smooth = 0.8 * q_t_smooth + 0.2 * q_t_raw
- [ ] 1.3.7: Add fallback: if GAS-Q params missing, use static q with log warning
- [ ] 1.3.8: Write test: verify q_t increases after large innovation
- [ ] 1.3.9: Write test: verify q_t trajectory stays within bounds
- [ ] 1.3.10: Benchmark: forecast display accuracy before/after GAS-Q propagation

---

### Story 1.4: Multi-Scale Drift Estimation with Horizon-Matched Filters

**As a** portfolio manager,
**I want** forecasts at each horizon to use a drift estimate calibrated to that specific
time scale,
**so that** 1-day forecasts capture intraday momentum while 90-day forecasts capture
quarterly trends.

**Background:**
Currently, one Kalman filter produces one mu_t estimate used for all horizons. This is
suboptimal: short-horizon forecasts need fast-reacting drift (high K_t), while
long-horizon forecasts need smooth drift (low K_t). Multi-scale filtering runs parallel
filters with different q values, selecting the appropriate one per horizon.

**Dependencies:** Story 1.1 (Adaptive Process Noise Floor -- q_fast needs a floor too)

**Risk:** Three parallel filters per model per asset is expensive: for 14 models x 3 scales
= 42 filter passes vs current 14. Runtime triples unless mitigated by sharing BMA weights
across scales (not re-estimating -- this is the key efficiency insight). Also, the
assumption that BMA weights transfer across time scales may not hold for all models.

**Acceptance Criteria:**
1. Three parallel filters run per model:
   - Fast filter: q_fast = 10 * q_tuned (1-3 day horizons)
   - Medium filter: q_med = q_tuned (7-30 day horizons)
   - Slow filter: q_slow = 0.1 * q_tuned (90-365 day horizons)
2. Horizon-to-filter mapping is configurable
3. Forecasts at each horizon use the matched filter's mu_t and P_t
4. BMA weights are shared across filters (not re-estimated per scale)
5. Runtime increase < 3x per asset (three filters vs one)
6. Benchmark: per-horizon directional accuracy improves for all horizons

**Tasks:**
- [ ] 1.4.1: Define HORIZON_FILTER_MAP: {1: 'fast', 3: 'fast', 7: 'medium', ...}
- [ ] 1.4.2: Add q_multiplier parameter to filter functions
- [ ] 1.4.3: Run three parallel filter passes in `signals.py` signal generation
- [ ] 1.4.4: Store per-filter (mu_t, P_t) tuples in signal output
- [ ] 1.4.5: Modify forecast assembly to use horizon-matched (mu, P)
- [ ] 1.4.6: Write test: fast filter responds faster to regime change than slow
- [ ] 1.4.7: Write test: slow filter has lower variance in calm regime
- [ ] 1.4.8: Benchmark: per-horizon accuracy table for 12-symbol universe
- [ ] 1.4.9: Profile runtime: verify < 3x increase

---

### Story 1.5: Innovation-Weighted Drift Aggregation

**As a** quant researcher,
**I want** the drift estimate to weight recent observations by their information content
(innovation magnitude relative to expected volatility),
**so that** large surprises (earnings, macro events) update the drift faster than noise.

**Background:**
Standard Kalman filtering weights all observations equally through the gain K_t. But
a 3-sigma innovation (earnings surprise) carries far more information than a 0.1-sigma
tick. Innovation-weighted aggregation increases the effective K_t for surprising
observations while decreasing it for noise:
```
K_effective_t = K_t * w(z_t)
w(z_t) = min(1 + alpha_iw * (|z_t| - 1)^+, w_max)
```

**IMPORTANT CAVEAT:** Modifying K_t based on z_t breaks the optimality guarantees of
the standard Kalman filter. This is intentional -- we are trading theoretical optimality
(under Gaussian assumptions that don't hold) for practical utility (faster response to
regime changes). However, this creates a risk of filter divergence if alpha_iw is too
large. The mitigation is conservative defaults and mandatory PIT monitoring.

**Dependencies:** Story 1.2 (Kalman Gain Monitoring -- detects if this causes divergence)

**Acceptance Criteria:**
1. Innovation weighting function `w(z_t)` implemented with configurable alpha and cap
2. Applied to Kalman update step: mu_t += K_effective * innovation_t
3. Weighting is asymmetric-capable: downside surprises can have higher weight
4. Default: alpha_iw = 0.3, w_max = 2.0 (conservative, not 3.0 -- divergence risk)
5. P_t (state covariance) update also adjusted to maintain filter consistency:
   P_t|t = (1 - K_effective * H) * P_t|t-1 (Joseph form for numerical stability)
6. Benchmark: post-earnings drift captured 40% faster (measured on AAPL, NVDA)
7. No increase in false positive rate for non-event days
8. PIT calibration ECE monitored: if ECE > 0.07 with innovation weighting, reduce alpha_iw

**Risk:** Innovation weighting with alpha_iw > 0.5 causes filter P_t to grow unbounded
in volatile regimes (positive feedback: large z -> large K -> large P -> large K).
Mitigation: cap w_max at 2.0 and add P_max bound.

**Tasks:**
- [ ] 1.5.1: Implement `innovation_weight(z_t, alpha, w_max)` utility function
- [ ] 1.5.2: Add asymmetric option: alpha_down vs alpha_up
- [ ] 1.5.3: Integrate into Gaussian Kalman update step with Joseph-form P update
- [ ] 1.5.4: Integrate into phi-Student-t update step
- [ ] 1.5.5: Add P_max bound: P_t = min(P_t, P_MAX) where P_MAX = 10 * P_steady_state
- [ ] 1.5.6: Write test: 3-sigma innovation gets w ~ 1.6, 0.5-sigma gets w ~ 1.0
- [ ] 1.5.7: Write test: P_t does not grow unbounded under repeated large innovations
- [ ] 1.5.8: Benchmark on AAPL/NVDA earnings dates: drift recovery speed
- [ ] 1.5.9: Benchmark on noise days: verify no false positive increase
- [ ] 1.5.10: PIT regression test: ECE with innovation weighting < ECE + 0.02

---

### Story 1.6: Ensemble Forecast De-correlation and Signal Extraction

**As a** signal consumer,
**I want** the 5-model ensemble to extract directional signal rather than averaging it away,
**so that** when 3+ models agree on direction, the forecast reflects their consensus
strength instead of being diluted by dissenting models.

**Background:**
Current ensemble uses fixed weights and simple weighted average. When momentum says +1.2%
and OU says -0.8%, the average is +0.2% -- losing the strong directional signal. A smarter
ensemble should:
1. Detect agreement/disagreement among models
2. Weight agreeing models higher (consensus amplification)
3. Report disagreement as uncertainty, not as cancellation

The approach uses **sign-agreement-weighted averaging** rather than ad-hoc amplification:
```
agreement = fraction_of_models_with_sign_equal_to_weighted_median
agreement_contrast = (agreement - 0.5) * 2    # maps [0.5, 1.0] -> [0, 1]

For each model i:
  sign_i = sign(forecast_i)
  sign_median = sign(weighted_median)
  if sign_i == sign_median:
    adjusted_weight_i = base_weight_i * (1 + contrast_boost * agreement_contrast)
  else:
    adjusted_weight_i = base_weight_i * (1 - contrast_boost * agreement_contrast)
  normalize all adjusted weights to sum to 1.0
```
With contrast_boost = 0.6: unanimous agreement (agreement=1.0) gives agreeing models
1.6x their base weight and dissenters 0.4x. This is smoother than step-function
amplification and degrades gracefully.

**Dependencies:** None (works with existing ensemble infrastructure)

**Acceptance Criteria:**
1. Model agreement score: fraction of models (by weight) with same sign as weighted median
2. Sign-agreement-weighted averaging applied with configurable contrast_boost (default 0.6)
3. Disagreement quantified: `forecast_dispersion = std(model_forecasts) / abs(mean)`
4. Dispersion reported as separate `forecast_uncertainty` field (0-1 scale, normalized)
5. When dispersion > 1.5: "CONTESTED" label, intervals widened by 50%
6. Benchmark: 7-day absolute forecast magnitude increases by > 40%
7. Directional accuracy does not decrease (must verify!)
8. New fields propagated to frontend display

**Risk:** Over-amplification of consensus creates false confidence. When all 5 models
agree but are ALL wrong (e.g., trend reversal), the amplified signal is worse than
unamplified. Mitigation: contrast_boost capped at 0.6, and Story 1.13 feedback loop
will dampen assets where consensus historically underperforms.

**Tasks:**
- [ ] 1.6.1: Compute per-horizon agreement score in `market_temperature.py`
- [ ] 1.6.2: Implement consensus amplification logic with configurable thresholds
- [ ] 1.6.3: Implement disagreement dampening with uncertainty field
- [ ] 1.6.4: Add `forecast_uncertainty` to ensemble output tuple
- [ ] 1.6.5: Update `signals_ux.py` to display uncertainty indicator
- [ ] 1.6.6: Update backend `/api/charts/forecast/{symbol}` to include uncertainty
- [ ] 1.6.7: Write test: 5 models all positive -> amplified
- [ ] 1.6.8: Write test: 2 positive, 3 negative -> dampened with high uncertainty
- [ ] 1.6.9: Benchmark on 12-symbol universe: magnitude and accuracy

---

### Story 1.7: Market Temperature Kalman to Use Tuned Parameters

**As a** system architect,
**I want** `market_temperature.py` to load per-asset tuned parameters (q, c, phi, nu)
instead of using hardcoded EMA-based drift estimation,
**so that** the forecasting engine benefits from the expensive tuning phase.

**Background:**
`market_temperature.py` `_kalman_forecast()` currently computes drift via simple EMA
blending (70/30, 50/40/10, etc.) and applies ad-hoc persistence factors. Meanwhile,
the tuning phase carefully optimizes q, c, phi per asset per regime. This disconnect
means tuning effort is wasted for the forecast display the user actually sees.

This is arguably the SINGLE HIGHEST IMPACT story in the entire program. Every other
improvement flows through market_temperature.py -- if it ignores tuned params, nothing
else matters for the user-facing forecast.

**Dependencies:** Stories 3.3 and 3.4 (GARCH and OU params fitted in tuning)

**Risk:** Switching from EMA to Kalman-filter-based drift could produce dramatically
different forecasts for some assets. Users may be alarmed by sudden changes. Mitigation:
A/B comparison report showing EMA vs Kalman forecasts for all assets, with highlighted
differences, BEFORE the switch goes live.

**Acceptance Criteria:**
1. `_kalman_forecast()` accepts optional tuned_params dict
2. When params available: runs actual Kalman filter with (q, c, phi) from tune cache
3. Drift estimate = filter's mu_t (not EMA)
4. Persistence = phi^H (not hardcoded decay tables)
5. Fallback to current EMA logic when tuned params unavailable
6. Benchmark: forecast accuracy improves by >= 5% across all horizons

**Tasks:**
- [ ] 1.7.1: Add tuned_params loading in `ensemble_forecast()` entry point
- [ ] 1.7.2: Refactor `_kalman_forecast()` to accept (q, c, phi) parameters
- [ ] 1.7.3: Implement proper Kalman filter pass within `_kalman_forecast()`
- [ ] 1.7.4: Replace persistence tables with phi^H decay
- [ ] 1.7.5: Maintain EMA fallback path for assets without tune cache
- [ ] 1.7.6: Write test: tuned SPY params produce different forecast than EMA
- [ ] 1.7.7: Benchmark: accuracy table with/without tuned params

---

### Story 1.8: Display Precision and Semantic Formatting

**As a** user reading the signal table,
**I want** forecasts to display with sufficient precision and semantic context
so that I can distinguish between "no signal" and "small but real signal."

**Background:**
Current formatting uses `.1f` (one decimal place), so 0.047% displays as +0.0%.
This is technically correct but informationally useless. We need:
- Adaptive precision: show more decimals for small numbers
- Semantic labels: "FLAT" vs "+0.05% (weak bullish)" vs "+2.3% (strong bullish)"
- Color intensity proportional to signal strength

**Acceptance Criteria:**
1. Forecasts < 0.1% display with 2 decimal places (e.g., +0.05%)
2. Forecasts >= 0.1% and < 1.0% display with 1 decimal place (e.g., +0.3%)
3. Forecasts >= 1.0% display with 1 decimal place (e.g., +2.3%)
4. Semantic labels added: FLAT (< 0.02%), SLIGHT (0.02-0.1%), LEAN (0.1-0.5%),
   MODERATE (0.5-2%), STRONG (2-5%), EXTREME (>5%)
5. Color saturation scales with absolute forecast magnitude
6. Both terminal (Rich) and frontend display updated consistently

**Tasks:**
- [ ] 1.8.1: Implement `adaptive_format_pct(value)` utility function
- [ ] 1.8.2: Implement `semantic_signal_label(value)` with 6 tiers
- [ ] 1.8.3: Update `signals_ux.py` `format_profit_with_signal()` to use new formatting
- [ ] 1.8.4: Update frontend SignalsPage cell rendering with same logic
- [ ] 1.8.5: Implement color intensity gradient (low alpha for weak, high for strong)
- [ ] 1.8.6: Write test: 0.047% -> "+0.05% SLIGHT", 2.3% -> "+2.3% STRONG"
- [ ] 1.8.7: Screenshot comparison: before/after for terminal and frontend

---

### Story 1.9: Directional Accuracy Tracking and Scorecard

**As a** quant researcher,
**I want** an automated system that tracks realized directional accuracy of forecasts
against actual returns at each horizon,
**so that** we can measure whether changes improve or degrade forecast quality.

**Background:**
There is currently no systematic way to measure "did the +0.3% 7-day forecast for AAPL
actually result in a positive 7-day return?" This scorecard is essential for validating
every improvement in this epic.

**Acceptance Criteria:**
1. Daily job records forecasts: {symbol, date, horizon, forecast_pct, direction}
2. After H days, realized return is looked up and compared
3. Metrics computed: hit_rate, mean_absolute_error, directional_accuracy, information_coefficient
4. Scorecard stored as JSON: `src/data/calibration/forecast_scorecard.json`
5. CLI command: `make scorecard` displays summary table
6. Historical backfill for last 90 days using cached forecasts
   NOTE: This requires signal cache history to exist. If cache was recently cleared,
   backfill starts from the first available cached forecast (may be < 90 days).
   The scorecard must handle partial history gracefully.

**Tasks:**
- [ ] 1.9.1: Create `src/calibration/forecast_scorecard.py` with ForecastRecord dataclass
- [ ] 1.9.2: Implement `record_forecasts()` -- snapshot current forecasts to scorecard
- [ ] 1.9.3: Implement `evaluate_forecasts()` -- match predictions to realized returns
- [ ] 1.9.4: Implement `compute_scorecard_metrics()` -- hit_rate, MAE, IC per horizon
- [ ] 1.9.5: Add `make scorecard` target to Makefile
- [ ] 1.9.6: Add scorecard display to `signals_ux.py` as optional section
- [ ] 1.9.7: Implement 90-day historical backfill using cached tune/price data
- [ ] 1.9.8: Write test: synthetic perfect forecaster -> 100% hit rate
- [ ] 1.9.9: Write test: random forecaster -> ~50% hit rate

---

### Story 1.10: Momentum Integration into Kalman State Equation

**As a** quant researcher,
**I want** multi-timeframe momentum signals (5d, 21d, 63d) integrated as exogenous
inputs to the Kalman state equation rather than as post-filter overlays,
**so that** momentum information improves the drift estimate coherently within the
probabilistic framework.

**Background:**
The current momentum_augmented.py framework injects mean reversion as u_t into the
state equation. The same mechanism should handle pure momentum signals, creating a
unified "drift prior" that combines:
- Filter's own learning (Kalman gain * innovation)
- Momentum signal (trend following)
- Mean reversion signal (equilibrium pull)

**Acceptance Criteria:**
1. Momentum signal computed: weighted blend of 5d/21d/63d returns
2. Injected as exogenous input u_t alongside mean reversion signal
3. Relative weights between momentum and mean reversion are regime-dependent:
   - TREND regimes: momentum weight 0.7, MR weight 0.3
   - RANGE regimes: momentum weight 0.3, MR weight 0.7
4. Total u_t capped by dynamic max_u = k * sqrt(q) (existing framework)
5. Benchmark: trending period accuracy improves by >= 5%
6. Mean-reverting period accuracy does not degrade

**Tasks:**
- [ ] 1.10.1: Implement `compute_momentum_signal(returns, weights=[0.5, 0.3, 0.2])`
- [ ] 1.10.2: Add regime-dependent weighting in `filter_phi_augmented()`
- [ ] 1.10.3: Blend momentum + mean reversion into single u_t
- [ ] 1.10.4: Verify max_u capping applies to blended signal
- [ ] 1.10.5: Write test: strong trend -> momentum dominates u_t
- [ ] 1.10.6: Write test: range-bound -> mean reversion dominates u_t
- [ ] 1.10.7: Benchmark on trending subsample (NVDA 2024, TSLA rallies)
- [ ] 1.10.8: Benchmark on ranging subsample (SPY Q3 2024)

---

### Story 1.11: Volatility-Adjusted Forecast Scaling

**As a** signal consumer,
**I want** forecasts to be scaled by current volatility regime relative to historical norms,
**so that** a +0.5% forecast in a 10% vol environment and a +0.5% forecast in a 40% vol
environment are distinguished in terms of signal strength and confidence.

**Background:**
In a low-vol environment, a +0.5% forecast is a strong signal (high SNR). In a high-vol
environment, the same +0.5% is noise (low SNR). Vol-adjustment normalizes the informational
content of forecasts:
```
vol_ratio = current_vol / median_vol  (rolling 252-day median)
signal_quality = 1 / max(vol_ratio, 0.5)  # High vol -> quality DOWN
```
IMPORTANT: This scales the CONFIDENCE/QUALITY indicator, not the forecast magnitude.
The forecast itself stays unchanged (it represents expected return). The quality factor
tells the user whether to trust it.

**Dependencies:** None (standalone scaling layer)

**Acceptance Criteria:**
1. Vol-ratio computed: current_ewma_vol / rolling_252d_median_vol
2. Signal quality factor: quality = clip(1.0 / vol_ratio, 0.3, 2.0)
   - vol_ratio = 0.5 (calm): quality = 2.0 (high SNR -- trust this forecast)
   - vol_ratio = 1.0 (normal): quality = 1.0 (baseline)
   - vol_ratio = 2.0 (crisis): quality = 0.5 (low SNR -- discount this forecast)
3. Quality factor displayed alongside forecast, NOT multiplied into forecast value
4. Confidence indicator reflects vol regime: high vol = lower displayed confidence
5. Vol regime label: CALM (< 0.7), NORMAL (0.7-1.3), ELEVATED (1.3-2.0), EXTREME (> 2.0)
6. Benchmark: risk-adjusted directional accuracy (Sharpe of forecast direction) improves

**Risk:** If quality factor is accidentally MULTIPLIED into forecasts (instead of displayed
alongside), low-vol forecasts get amplified and high-vol get dampened -- the OPPOSITE of
desired behavior for position sizing.

**Tasks:**
- [ ] 1.11.1: Compute rolling vol-ratio in `market_temperature.py`
- [ ] 1.11.2: Implement `compute_signal_quality(vol_ratio)` with bounds
- [ ] 1.11.3: Add quality factor to ensemble output (separate from forecast value)
- [ ] 1.11.4: Add vol_regime label to forecast output (CALM/NORMAL/ELEVATED/EXTREME)
- [ ] 1.11.5: Update `signals_ux.py` to display vol regime and quality badge
- [ ] 1.11.6: Update frontend to display vol regime indicator
- [ ] 1.11.7: Write test: vol_ratio=0.5 -> quality=2.0, vol_ratio=2.0 -> quality=0.5
- [ ] 1.11.8: Write test: forecast VALUE unchanged by scaling (quality is separate)
- [ ] 1.11.9: Benchmark: Sharpe of directional bets weighted by quality vs unweighted

---

### Story 1.12: Regime Transition Speed Enhancement

**As a** signal consumer,
**I want** the regime classifier to detect transitions within 1-2 days instead of 4+,
**so that** forecasts adapt to new market conditions before the move is over.

**Background:**
Current regime classification uses smoothing alpha=0.40, creating a 4+ day lag for
crisis detection. A CUSUM (Cumulative Sum) change-point detector can identify
regime transitions in 1-2 bars by accumulating deviations from expected values.

**Dependencies:** None (standalone regime improvement)

**Risk:** False positive regime changes cause whipsaw: the system switches to crisis mode,
then back to trend mode within 2 days. This burns transaction costs and creates confusing
signals. Mitigation: 5-bar cool-down after any alpha acceleration (no second transition
allowed during cool-down), and CUSUM threshold set for < 5% false positive rate.

**Acceptance Criteria:**
1. CUSUM-based change-point detector added as regime transition accelerator
2. When CUSUM triggers, smoothing alpha temporarily increases to 0.85 for 5 bars
3. Cool-down period: 5 bars after acceleration ends before another can trigger
4. Regime transition lag reduced to <= 2 days for crisis onset
5. False positive rate for regime changes < 5% (calibrated on 2020-2025 data)
6. Benchmark: COVID crash (Feb 20-Mar 23, 2020) detected by day 2

**Tasks:**
- [ ] 1.12.1: Implement CUSUM detector in `tune.py` `assign_regime_labels()`
- [ ] 1.12.2: Add adaptive smoothing alpha with temporary acceleration
- [ ] 1.12.3: Mirror implementation in `signals.py` regime classification
- [ ] 1.12.4: Write test: synthetic regime change at bar 100, verify detection by bar 102
- [ ] 1.12.5: Write test: no regime change scenario, verify no false triggers
- [ ] 1.12.6: Benchmark on historical crisis dates: detection lag table

---

### Story 1.13: Per-Asset Forecast Calibration Feedback Loop

**As a** quant researcher,
**I want** a feedback mechanism that adjusts forecast amplification based on historical
accuracy for each specific asset,
**so that** assets where the model is reliably directional get amplified forecasts while
assets where it fails get dampened ones.

**Background:**
Raw amplification (amp = 0.5 + accuracy) is dangerous because it ignores sample size.
An asset with 3/5 = 60% accuracy over 5 days should NOT be amplified as much as one
with 150/250 = 60% over 250 days. Bayesian shrinkage toward a neutral prior (amp=1.0)
handles this naturally:
```
amp = prior_amp + (accuracy - prior_accuracy) * shrinkage_weight(n)
where:
  prior_amp = 1.0 (neutral -- no amplification before evidence)
  prior_accuracy = 0.50 (coin-flip prior)
  shrinkage_weight(n) = n / (n + n_prior)   # n_prior = 60 (pseudocount)
```
With n=10 observations: weight = 10/70 = 0.14 -> amp barely moves from 1.0
With n=250 observations: weight = 250/310 = 0.81 -> amp converges to true accuracy

**Dependencies:** Story 1.9 (Directional Accuracy Scorecard provides accuracy data)

**Acceptance Criteria:**
1. Rolling 60-day directional accuracy tracked per asset (from Story 1.9 scorecard)
2. Bayesian shrinkage formula applied:
   ```
   n = number of evaluated forecasts (60-day window)
   shrinkage = n / (n + 60)
   amp = 1.0 + (accuracy - 0.50) * shrinkage * sensitivity
   where sensitivity = 2.0 (maps 60% accuracy -> 1.16x at full shrinkage)
   ```
3. Amplification bounds: amp in [0.5, 1.5] (never more than 50% adjustment)
4. Minimum sample size: n >= 20 before any amplification (below: amp = 1.0)
5. Applied at the ensemble forecast output stage
6. Updated daily, stored in forecast scorecard JSON
7. No leakage: uses only past data (lookback, not lookahead)
8. Benchmark: portfolio-level Sharpe improvement >= 0.05

**Risk:** Feedback loops can create self-reinforcing amplification spirals (good asset
gets amplified, looks even better, gets amplified more). Mitigation: hard bounds [0.5, 1.5]
and decay toward 1.0 when recent accuracy data is unavailable (> 10 days stale).

**Tasks:**
- [ ] 1.13.1: Extend forecast_scorecard.py with per-asset rolling accuracy + sample count
- [ ] 1.13.2: Implement Bayesian `compute_asset_amplification(accuracy, n, prior_n=60)`
- [ ] 1.13.3: Integrate amplification into `ensemble_forecast()` output
- [ ] 1.13.4: Add anti-leakage assertion: amplification uses only t-1 data
- [ ] 1.13.5: Add minimum sample guard: n < 20 -> amp = 1.0
- [ ] 1.13.6: Add staleness decay: if last accuracy > 10 days old, amp decays toward 1.0
- [ ] 1.13.7: Write test: n=10, accuracy=60% -> amp ~ 1.03 (heavily shrunk toward 1.0)
- [ ] 1.13.8: Write test: n=250, accuracy=60% -> amp ~ 1.16 (converged)
- [ ] 1.13.9: Write test: n=250, accuracy=45% -> amp ~ 0.92 (dampened)
- [ ] 1.13.10: Benchmark: Sharpe of amplified vs neutral (amp=1.0) forecast portfolio

---

### Story 1.14: Hard Cap Relaxation with Confidence-Gated Bounds

**As a** signal consumer,
**I want** hard caps on forecasts to be relaxed proportionally to model confidence,
**so that** high-conviction signals are not artificially truncated.

**Background:**
Current hard caps (e.g., +/-3% for 1-day equity) bind frequently in low-vol regimes
where vol_bound < cap. When the ensemble has high confidence (4+ models agree with
strong signals), the cap should widen to let the signal through.

**Dependencies:** Story 1.6 (Ensemble De-correlation -- provides the agreement/confidence score)

**Risk:** Relaxing caps without validation creates tail risk -- the few times caps bind
are often the times they are most needed (extreme events). Mitigation: vol-bound remains
as a hard constraint independent of confidence, and caps can at most double (never removed).

**Acceptance Criteria:**
1. Confidence score from ensemble agreement (Story 1.6) feeds into cap calculation
2. Cap multiplier: 1.0 + 0.5 * confidence (so 80% confidence -> 1.4x cap)
3. Maximum cap multiplier: 2.0 (caps can at most double)
4. Vol-bound remains as separate constraint (safety first)
5. Benchmark: fewer cap-binding events, forecast range widens by 30%

**Tasks:**
- [ ] 1.14.1: Add confidence parameter to cap calculation in `ensemble_forecast()`
- [ ] 1.14.2: Implement graduated cap multiplier with max bound
- [ ] 1.14.3: Log cap-binding events with confidence context
- [ ] 1.14.4: Write test: high confidence widens cap, low confidence does not
- [ ] 1.14.5: Benchmark: cap-binding frequency before/after

---

## EPIC 2: ENSEMBLE FORECASTING ENGINE OVERHAUL

**Epic Owner:** Quant Engine Lead
**Priority:** P0 -- Critical Path
**Estimated Scope:** 10 stories, ~35 tasks

**Problem Statement:**
`market_temperature.py` is the engine that generates the 7-horizon forecasts displayed
to users. It currently uses 5 models with hardcoded parameters, no connection to tuning
output, and simple weighted averaging that destroys directional signal. This epic
transforms it into an institution-grade forecasting engine.

**Success Criteria:**
- Forecasts are non-trivially directional (median absolute forecast > 0.3% at 7-day)
- Directional hit rate > 57% at 7-day across benchmark universe
- Ensemble Sharpe > 0.5 (annualized, long/short based on forecast sign)
- All 5 models use asset-specific calibrated parameters
- Regime-aware weighting adapts automatically to market conditions
- Frontend and terminal display identical forecast values

---

### Story 2.1: GARCH Parameter Loading from Tuned Cache

**As a** system architect,
**I want** `_garch_forecast()` to use per-asset GARCH(1,1) parameters from the tuning
cache instead of hardcoded defaults (omega=1e-5, alpha=0.08, beta=0.88),
**so that** volatility forecasts reflect each asset's specific dynamics.

**Background:**
GARCH parameters vary dramatically across assets: UPST (alpha~0.15, beta~0.80) vs
SPY (alpha~0.04, beta~0.94). Using SPY-like defaults for small caps understates
volatility clustering and produces weak forecasts during their volatile episodes.

GARCH fitting should use Kalman filter RESIDUALS (not raw returns) as input. Raw returns
contain the drift component which GARCH is trying to ignore. By fitting on residuals
from the best Kalman model, we get cleaner volatility dynamics.

**Dependencies:** Story 3.3 (GARCH Fitting in Tuning Pipeline -- produces the params to load)

**Acceptance Criteria:**
1. Tuning phase fits GARCH(1,1) per asset: (omega, alpha, beta) via MLE
2. Parameters stored in tune JSON: `garch_params: {omega, alpha, beta, persistence}`
3. `_garch_forecast()` loads these parameters when available
4. Fallback to current defaults when tuned params missing
5. GARCH persistence (alpha+beta) validated < 1.0 (stationarity)
6. Benchmark: vol forecast RMSE improves by >= 15% for small-cap universe

**Tasks:**
- [ ] 2.1.1: Add GARCH(1,1) MLE fitting to `tune.py` tuning pipeline
- [ ] 2.1.2: Store GARCH params in tune JSON output
- [ ] 2.1.3: Load GARCH params in `_garch_forecast()` from tune cache
- [ ] 2.1.4: Add stationarity check: alpha + beta < 1.0
- [ ] 2.1.5: Implement fallback to defaults with logging
- [ ] 2.1.6: Write test: UPST has higher alpha than SPY (more reactive)
- [ ] 2.1.7: Benchmark: vol forecast accuracy table per asset

---

### Story 2.2: Ornstein-Uhlenbeck Calibration with Asset-Specific Mean Reversion

**As a** quant researcher,
**I want** `_ou_forecast()` to use calibrated OU parameters (theta, kappa, sigma_ou)
per asset instead of generic MA-based mean reversion,
**so that** the model correctly estimates how fast each asset reverts to its local mean.

**Background:**
OU parameters define two critical quantities:
- kappa (speed): half-life = ln(2)/kappa. SPY might have 60-day half-life, UPST 15-day.
- theta (level): local equilibrium. Using wrong theta miscalibrates reversion magnitude.

Current implementation uses MA-50/100/200 as theta targets with base kappa=0.025 adjusted
by autocorrelation. This is an approximation. Proper OU MLE on the Euler-Maruyama
discretization (AR(1) on demeaned log prices) is more principled but also more fragile
(kappa estimation has high variance for non-stationary series). The AR(1) approach is
a reasonable first step with acknowledged limitations.

**Dependencies:** Story 3.4 (OU Estimation in Tuning Pipeline -- produces the params to load)

**Acceptance Criteria:**
1. OU parameter estimation added to tuning: kappa via AR(1) regression on demeaned returns
2. theta estimated as exponentially-weighted moving average level  
3. Stored in tune JSON: `ou_params: {kappa, theta, sigma_ou, half_life_days}`
4. `_ou_forecast()` uses calibrated params: fc = (theta - price) * (1 - exp(-kappa*H))
5. Half-life validated: 5 < half_life < 252 (reasonable range)
6. Benchmark: mean-reverting period MAE improves by >= 20%

**Tasks:**
- [ ] 2.2.1: Implement AR(1) kappa estimation in tuning pipeline
- [ ] 2.2.2: Implement EWMA theta estimation with span = half_life
- [ ] 2.2.3: Store OU params in tune JSON
- [ ] 2.2.4: Load OU params in `_ou_forecast()`
- [ ] 2.2.5: Replace MA-based reversion with proper OU projection
- [ ] 2.2.6: Add half-life bounds validation
- [ ] 2.2.7: Write test: fast kappa -> rapid reversion forecast
- [ ] 2.2.8: Benchmark on range-bound subsample

---

### Story 2.3: Momentum Model with Regime-Adaptive Timeframe Selection

**As a** signal consumer,
**I want** `_momentum_forecast()` to dynamically select momentum timeframes based on
which timeframes have been most predictive in the current regime,
**so that** the model does not use 252-day momentum in a crisis or 5-day momentum in
a slow trend.

**Acceptance Criteria:**
1. Per-regime momentum timeframe performance tracked: {5d, 10d, 21d, 63d, 126d, 252d}
2. Timeframe weights computed from rolling 60-day hit rate per timeframe:
   ```
   raw_weight_i = hit_rate_i - 0.50  (excess accuracy above coin flip)
   raw_weight_i = max(raw_weight_i, 0.01)  (floor: no timeframe goes to zero)
   weights = softmax(raw_weights / temperature)  where temperature = 0.1
   ```
3. Entropy-based diversity constraint: H(weights) >= 0.5 * H_uniform
   where H(w) = -sum(w_i * log(w_i)) and H_uniform = log(6) for 6 timeframes.
   If entropy is too low (concentration on 1-2 timeframes): increase temperature
   until constraint is met. This ensures at least 3 timeframes contribute.
4. Regime transitions trigger weight recalculation within 2 bars
5. Benchmark: momentum model standalone hit rate > 55%

**Tasks:**
- [ ] 2.3.1: Compute rolling hit rate per momentum timeframe per regime
- [ ] 2.3.2: Convert hit rates to softmax weights with temperature parameter
- [ ] 2.3.3: Add diversity constraint (min 3 active timeframes)
- [ ] 2.3.4: Implement fast weight recalculation on regime transition
- [ ] 2.3.5: Integrate into `_momentum_forecast()`
- [ ] 2.3.6: Write test: trending regime -> longer timeframes weighted higher
- [ ] 2.3.7: Write test: crisis -> short timeframes dominate
- [ ] 2.3.8: Benchmark: standalone momentum model hit rate table

---

### Story 2.4: Bayesian Model Combination Replacing Fixed Weights

**As a** quant researcher,
**I want** ensemble model weights determined by Bayesian posterior probabilities based
on recent forecasting performance instead of fixed regime-dependent weights,
**so that** models that are currently performing well get higher weight automatically.

**Background:**
Current fixed weights (e.g., [0.30, 0.10, 0.10, 0.35, 0.15] for trending) cannot adapt
to which models are actually performing well. Bayesian Model Combination (BMC) updates
weights based on predictive likelihood:
```
w_i,t+1 = w_i,t * p(y_t | model_i) / sum_j(w_j,t * p(y_t | model_j))
```

The 30-day lookback is implemented via a forgetting factor (exponential discounting of
past likelihood contributions), not a hard window. This avoids the "cliff effect" where
a model's weight changes sharply when a single observation exits the window.

**Dependencies:** Stories 2.1 and 2.2 (calibrated model params -- BMC needs well-specified models)

**Risk:** BMC with forgetting can chase noise: a model gets lucky for 10 days, gains weight,
then reverts. The weight floor (min 0.05) prevents model extinction, and the 45-day
half-life provides sufficient smoothing. Monitor weight volatility: if any model's weight
changes by > 0.2 in a single day, the forgetting factor is too aggressive.

**Acceptance Criteria:**
1. Per-model rolling predictive likelihood tracked with 30-day lookback
2. BMC weight update applied after each observation
3. Weight floor: min 0.05 per model (prevents extinction)
4. Forgetting factor: weights half-life = 45 days (adapts to regime changes)
5. Stored in forecast cache: `ensemble_weights: {kalman: 0.28, garch: 0.12, ...}`
6. Benchmark: ensemble Sharpe improves by >= 0.1 vs fixed weights

**Tasks:**
- [ ] 2.4.1: Implement predictive likelihood computation per model per bar
- [ ] 2.4.2: Implement BMC weight update rule with forgetting factor
- [ ] 2.4.3: Add weight floor constraint (min 0.05)
- [ ] 2.4.4: Track weight history for debugging: `ensemble_weight_history.json`
- [ ] 2.4.5: Integrate BMC weights into `ensemble_forecast()`
- [ ] 2.4.6: Add weight visualization endpoint for frontend
- [ ] 2.4.7: Write test: model that predicts well gains weight
- [ ] 2.4.8: Write test: model that fails loses weight but stays above floor
- [ ] 2.4.9: Benchmark: Sharpe and hit rate with BMC vs fixed weights

---

### Story 2.5: Forecast Confidence Intervals and Fan Charts

**As a** signal consumer,
**I want** each horizon forecast to include confidence intervals (10th, 25th, 50th, 75th,
90th percentiles) derived from the BMA posterior predictive distribution,
**so that** I understand not just the point forecast but the range of likely outcomes.

**Background:**
The current system outputs a single point forecast per horizon. But a +1.5% forecast
with 10th percentile at -3% is very different from one with 10th percentile at +0.5%.
The BMA posterior predictive already computes these quantiles internally -- they just
need to be surfaced.

**Acceptance Criteria:**
1. Each horizon forecast includes: {p10, p25, p50, p75, p90} in percentage terms
2. Fan chart width reflects model uncertainty (wider in high-vol, narrower in consensus)
3. Output included in signal cache JSON for frontend consumption
4. Backend endpoint returns quantiles with forecast data
5. Terminal display shows range: "+1.5% [+0.3, +2.8]" (median [p25, p75])
6. Frontend fan chart visualization (see Epic 5 for full UX story)

**Risk:** Quantiles from Monte Carlo are noisy with few samples. With 1000 draws,
the 10th percentile estimate has ~3% standard error. Mitigation: use at least 5000
draws for stable quantile estimates, or use analytical quantiles when available.

**Tasks:**
- [ ] 2.5.1: Extract quantiles from BMA Monte Carlo samples in `signals.py` (min 5000 draws)
- [ ] 2.5.2: Add quantiles to per-horizon forecast output structure
- [ ] 2.5.3: Store quantiles in signal cache JSON
- [ ] 2.5.4: Update `signals_ux.py` to display range notation
- [ ] 2.5.5: Add quantiles to `/api/signals/summary` response
- [ ] 2.5.6: Add quantiles to `/api/charts/forecast/{symbol}` response
- [ ] 2.5.7: Write test: high-vol asset has wider intervals than low-vol
- [ ] 2.5.8: Write test: quantile ordering: p10 < p25 < p50 < p75 < p90
- [ ] 2.5.9: **Quantile calibration test: verify ~10% of outcomes fall below p10 estimate**
  (Run on historical data: if p10 captures 15% or 5% of outcomes, intervals are miscalibrated)
- [ ] 2.5.10: Write test: 5000 vs 1000 draws -- quantile stability check (std < 0.5%)

---

### Story 2.6: Classical Model Upgrade to Adaptive Drift with Regime Switching

**As a** quant researcher,
**I want** the classical drift extrapolation model replaced with an adaptive drift model
that uses regime-dependent persistence and VIX conditioning,
**so that** the 5th ensemble member contributes meaningful information.

**Background:**
The current classical model is a simple drift extrapolation (basically a constant).
The original plan was BSTS (Bayesian Structural Time Series), but BSTS requires MCMC
sampling (Stan, PyMC, or custom) which adds heavy dependencies, is slow (~1s/asset),
and is over-engineered for daily return forecasting where signal-to-noise is tiny.

A pragmatic upgrade: adaptive drift with regime switching. Drift persistence decays
faster in high-vol regimes (mean reversion dominates) and slower in low-vol trends
(momentum dominates). VIX level modulates confidence intervals. This captures 80%
of BSTS benefit at 1% of the implementation cost.

**Dependencies:** None (standalone model improvement)

**Acceptance Criteria:**
1. Adaptive drift model replaces classical in ensemble:
   - TREND regimes: persistence = 0.97^H (slow decay, trust the trend)
   - RANGE regimes: persistence = 0.90^H (faster decay, drift unreliable)
   - CRISIS: persistence = 0.80^H (very fast decay, all bets off)
2. VIX conditioning: when VIX > 25, confidence intervals widen by factor 1 + 0.03*(VIX-25)
3. Drift computed from exponentially-weighted returns with half-life = 21 days
4. Forecast includes trend persistence estimate per horizon
5. Runtime: < 5ms per asset (no MCMC -- pure analytical computation)
6. Benchmark: adaptive model standalone accuracy > classical by >= 3%
7. No new dependencies (only numpy, scipy already available)

**Risk:** Regime-dependent persistence introduces a regime classification dependency.
If regime classification is wrong, persistence is wrong. Mitigation: blend persistence
across regimes weighted by regime probability (soft switching, not hard).

**Tasks:**
- [ ] 2.6.1: Implement adaptive drift with EW half-life of 21 days
- [ ] 2.6.2: Implement regime-dependent persistence decay tables
- [ ] 2.6.3: Implement soft regime switching (blend persistence by regime probability)
- [ ] 2.6.4: Add VIX conditioning for confidence interval widening
- [ ] 2.6.5: Generate multi-horizon forecasts via persistence projection
- [ ] 2.6.6: Replace `_classical_forecast()` with `_adaptive_drift_forecast()`
- [ ] 2.6.7: Write test: trending regime produces slower-decaying forecast
- [ ] 2.6.8: Write test: crisis regime produces rapid decay toward zero
- [ ] 2.6.9: Benchmark: classical vs adaptive drift standalone accuracy
- [ ] 2.6.10: Verify runtime < 5ms per asset (no regressions)

---

### Story 2.7: Cross-Asset Signal Propagation

**As a** portfolio manager,
**I want** the ensemble to incorporate cross-asset signals (e.g., rising VIX as bear signal
for equities, USD strength as headwind for metals),
**so that** forecasts account for systematic macro factors.

**Acceptance Criteria:**
1. Cross-asset signal computation: VIX level/change, DXY level/change, SPY breadth
2. Signals injected as drift adjustment in ensemble: adj = beta * cross_signal
3. Beta estimated from 120-day rolling regression per asset
4. Maximum adjustment: 30% of standalone forecast magnitude (safety limit)
5. Benchmark: cross-asset-adjusted forecasts improve hit rate by >= 2%

**Tasks:**
- [ ] 2.7.1: Implement cross-asset signal extraction (VIX, DXY, SPY returns)
- [ ] 2.7.2: Compute rolling beta per asset to each cross-asset signal
- [ ] 2.7.3: Apply drift adjustment with safety cap in ensemble
- [ ] 2.7.4: Store cross-asset betas in tune cache
- [ ] 2.7.5: Write test: rising VIX produces negative equity adjustment
- [ ] 2.7.6: Write test: safety cap prevents over-adjustment
- [ ] 2.7.7: Benchmark: hit rate with/without cross-asset signals

---

### Story 2.8: Ensemble Forecast Caching and Staleness Detection

**As a** system operator,
**I want** ensemble forecasts to be cached with timestamps and automatically flagged
when stale (> 4 hours since market data update),
**so that** users never see forecasts based on yesterday's data without explicit warning.

**Acceptance Criteria:**
1. Forecast output includes: `generated_at`, `data_through`, `staleness_hours`
2. Staleness alert when data_through is > 4 hours behind current time (during market hours)
3. Cache invalidation triggered by new price data arrival
4. Frontend displays "Last updated: 2h ago" with color coding (green < 1h, amber < 4h, red > 4h)
5. Backend health endpoint includes forecast freshness check

**Tasks:**
- [ ] 2.8.1: Add timestamps to forecast output in `market_temperature.py`
- [ ] 2.8.2: Implement staleness computation in signal_service.py
- [ ] 2.8.3: Add staleness to `/api/signals/summary` response
- [ ] 2.8.4: Update frontend to display freshness indicator
- [ ] 2.8.5: Add freshness to health check endpoint
- [ ] 2.8.6: Write test: 5-hour-old forecast flagged as stale

---

### Story 2.9: Ensemble Model Explainability Output

**As a** signal consumer,
**I want** each forecast to include a breakdown of which models contributed what,
**so that** I understand why the system is bullish or bearish on a given asset.

**Acceptance Criteria:**
1. Per-horizon: individual model forecasts + weights + weighted contributions
2. Top contributor identified with structured template:
   "[MODEL_NAME] ([FORECAST]%, weight [WEIGHT]): [VERB] [DIRECTION] signal"
   Example: "Momentum (+1.2%, weight 0.35): driving bullish signal"
3. Disagreement detected with structured template:
   "[MODEL_NAME] ([FORECAST]%) opposes consensus -- [IMPACT] uncertainty"
   Example: "GARCH (-0.8%) opposes consensus -- adding uncertainty"
4. Natural-language templates maintained as constants (not generated dynamically
   via LLM or string interpolation from unvalidated sources)
4. Stored in signal cache and available via API
5. Frontend "Why?" tooltip or expandable row shows model breakdown

**Tasks:**
- [ ] 2.9.1: Track per-model forecasts before ensemble averaging
- [ ] 2.9.2: Compute weighted contributions: weight_i * forecast_i
- [ ] 2.9.3: Identify top contributor and largest dissenter
- [ ] 2.9.4: Generate natural-language reason string
- [ ] 2.9.5: Add to signal cache JSON
- [ ] 2.9.6: Add to `/api/signals/summary` response
- [ ] 2.9.7: Write test: 5 positive models -> unanimous bullish explanation
- [ ] 2.9.8: Write test: 3 bullish + 2 bearish -> contested explanation

---

### Story 2.10: Benchmark Validation Harness for Ensemble Changes

**As a** quant researcher,
**I want** an automated validation harness that runs before/after any ensemble change,
comparing directional accuracy, Sharpe, and calibration on the 12-symbol benchmark
universe,
**so that** no change degrades performance without explicit awareness.

**Acceptance Criteria:**
1. `make validate-ensemble` target runs full evaluation pipeline
2. Compares current vs baseline on: hit_rate, Sharpe, CRPS, ECE per horizon
3. Traffic-light summary with EXPLICIT thresholds:
   - GREEN: ALL of (hit_rate_delta >= 0, sharpe_delta >= 0, crps_delta <= 0, ece_delta <= 0)
   - AMBER: ANY metric degrades but by less than tolerance:
     (hit_rate drops < 1%, sharpe drops < 0.03, crps rises < 0.002, ece rises < 0.01)
   - RED: ANY metric degrades beyond tolerance
4. Baseline stored in `src/data/calibration/ensemble_baseline.json`
5. CI-friendly: returns exit code 0 for GREEN/AMBER, 1 for RED
6. Tolerances are configurable via environment variables (CI may use stricter thresholds)
7. Results displayed in both terminal (Rich table) and frontend diagnostics

**Tasks:**
- [ ] 2.10.1: Create `src/calibration/ensemble_validator.py` with evaluation pipeline
- [ ] 2.10.2: Implement metric comparison with traffic-light logic
- [ ] 2.10.3: Store/load baseline JSON
- [ ] 2.10.4: Add `make validate-ensemble` Makefile target
- [ ] 2.10.5: Add results to diagnostics API endpoint
- [ ] 2.10.6: Write test: synthetic improvement -> GREEN
- [ ] 2.10.7: Write test: synthetic regression -> RED

---

## EPIC 3: TUNING PIPELINE HARDENING AND ACCELERATION

**Epic Owner:** Platform Engineering Lead
**Priority:** P1 -- High
**Estimated Scope:** 8 stories, ~30 tasks

**Problem Statement:**
The tuning pipeline (`make tune`) is the foundation of forecast quality. Current issues:
- Runtime: 15-25 minutes for full universe (100+ assets)
- Silent failures: models can fail to converge without warning
- No incremental tuning: re-tunes everything even if only 3 assets need it
- GARCH/OU parameters not fitted (used only in market_temperature with defaults)
- No validation gate: bad tune params propagate to signals silently

**Success Criteria:**
- Full tune time < 5 minutes (3x speedup)
- Incremental tune < 30 seconds for changed assets
- Zero silent failures: all convergence issues logged and surfaced
- GARCH and OU parameters fitted alongside Kalman models
- Automatic validation gate: bad params blocked from cache
- Frontend shows live tuning progress with per-asset status

---

### Story 3.1: Incremental Tuning with Change Detection

**As a** system operator,
**I want** `make tune` to detect which assets have new price data since last tune
and only re-tune those assets,
**so that** daily re-tuning takes seconds instead of minutes.

**Acceptance Criteria:**
1. Tune cache includes `last_price_date` and `price_data_hash` per asset
2. Change detection: compare price CSV content hash vs stored hash in tune JSON
   (NOT file mtime -- mtime is unreliable across git checkout, rsync, backup restore)
3. Hash computation: fast hash of last 20 OHLCV rows (sufficient to detect new data)
4. `make tune` default mode: incremental (only changed assets)
5. `make tune-full` forces full re-tune of all assets
6. `make tune-asset ARGS="SPY AAPL"` tunes specific assets only
7. Incremental tune of 5 changed assets completes in < 30 seconds
8. Log output: "Tuning 5/142 assets (incremental mode, 137 unchanged)"

**Tasks:**
- [ ] 3.1.1: Add `last_price_date` and `price_data_hash` to tune JSON output
- [ ] 3.1.2: Implement content-based change detection: hash last 20 rows of price CSV
- [ ] 3.1.3: Compare stored hash vs current hash (not mtime -- git-safe)
- [ ] 3.1.3: Add `--incremental` flag to `tune_ux.py` (default: on)
- [ ] 3.1.4: Add `--full` flag to force full re-tune
- [ ] 3.1.5: Add `--assets` flag for specific asset selection
- [ ] 3.1.6: Update Makefile targets: tune (incremental), tune-full, tune-asset
- [ ] 3.1.7: Write test: unchanged asset skipped, changed asset re-tuned
- [ ] 3.1.8: Profile: 5-asset incremental tune < 30s

---

### Story 3.2: Parallel Tuning with Process Pool Optimization

**As a** system operator,
**I want** tuning to use all available CPU cores with optimized work distribution,
**so that** full tune completes 3x faster.

**Background:**
Current tuning uses ProcessPoolExecutor but with default chunk size and no work-stealing.
Assets vary in tuning complexity: UPST (small-cap, volatile) takes 5x longer than SPY.

**Acceptance Criteria:**
1. Work distribution sorted by estimated complexity (volatile assets first)
2. Chunk size = 1 (enables work-stealing across uneven tasks)
3. Worker count = min(cpu_count - 1, 8) (leave 1 core for OS)
4. Progress bar shows: "[47/142] NVDA... 3.2s | ETA: 2m15s"
5. Failed assets do not block others (isolated process per asset)
6. Full tune < 5 minutes on 8-core machine

**Tasks:**
- [ ] 3.2.1: Sort asset tuning queue by historical tuning time (slowest first)
- [ ] 3.2.2: Set chunk_size=1 in ProcessPoolExecutor.map()
- [ ] 3.2.3: Implement progress callback with Rich progress bar
- [ ] 3.2.4: Add per-asset timing to tune output
- [ ] 3.2.5: Isolate failures: catch exceptions per-asset, continue queue
- [ ] 3.2.6: Write test: 12-symbol benchmark completes within timeout
- [ ] 3.2.7: Profile on 8-core machine: full tune time

---

### Story 3.3: GARCH(1,1) Parameter Fitting in Tuning Pipeline

**As a** quant researcher,
**I want** GARCH(1,1) parameters (omega, alpha, beta) fitted per asset during tuning
and stored in the tune cache,
**so that** the forecast engine uses calibrated volatility dynamics.

**Acceptance Criteria:**
1. GARCH(1,1) MLE fitting added after Kalman model fitting
2. Parameters: omega, alpha, beta, long_run_var = omega / (1 - alpha - beta)
3. Stationarity enforced: alpha + beta < 0.999, alpha > 0, beta > 0
4. Stored in tune JSON: `garch_params: {omega, alpha, beta, persistence, long_run_vol}`
5. Fitting uses returns residuals from best Kalman model (cleaner signal)
6. Runtime < 0.5s per asset (scipy.optimize with analytic gradient)

**Tasks:**
- [ ] 3.3.1: Implement GARCH(1,1) log-likelihood function with Gaussian innovations
- [ ] 3.3.2: Implement MLE optimizer with stationarity constraints
- [ ] 3.3.3: Extract residuals from best Kalman model for GARCH input
- [ ] 3.3.4: Add GARCH params to tune JSON output schema
- [ ] 3.3.5: Integrate into per-asset tuning pipeline
- [ ] 3.3.6: Write test: known GARCH params recovered from simulation
- [ ] 3.3.7: Write test: stationarity constraint binds for non-stationary series

---

### Story 3.4: OU Parameter Estimation in Tuning Pipeline

**As a** quant researcher,
**I want** Ornstein-Uhlenbeck parameters (kappa, theta, sigma_ou) estimated per asset
during tuning,
**so that** mean reversion forecasts use calibrated speed and level.

**Acceptance Criteria:**
1. kappa estimated via AR(1) regression: log_price_t = phi * log_price_{t-1} + c + eps
   kappa = -log(phi) / dt
2. theta estimated as EWMA of log prices with half-life = ln(2)/kappa
3. sigma_ou from residual standard deviation
4. Stored in tune JSON: `ou_params: {kappa, theta, sigma_ou, half_life_days}`
5. Half-life bounds: 5 < half_life < 252 days
6. Boundary cases handled: kappa near zero (random walk) -> half_life = 252 (cap)

**Tasks:**
- [ ] 3.4.1: Implement AR(1) regression for kappa estimation
- [ ] 3.4.2: Implement EWMA theta estimation
- [ ] 3.4.3: Compute sigma_ou from regression residuals
- [ ] 3.4.4: Add OU params to tune JSON output schema
- [ ] 3.4.5: Integrate into per-asset tuning pipeline
- [ ] 3.4.6: Write test: known OU params recovered from simulation
- [ ] 3.4.7: Write test: random walk produces half_life near cap

---

### Story 3.5: Tuning Validation Gate (Quality Control)

**As a** system architect,
**I want** a validation gate that checks tuned parameters for quality before writing
to the cache,
**so that** bad parameters (convergence failures, extreme values, NaN) never reach
signal generation.

**Acceptance Criteria:**
1. Validation checks per model:
   - All parameters are finite (no NaN, no Inf)
   - q in [1e-8, 1e-1] (reasonable process noise range)
   - c in [1e-4, 100.0] (reasonable observation noise range)
   - phi in [0.5, 1.0] (reasonable persistence range)
   - nu in [2.5, 50.0] (reasonable tail thickness range)
   - BIC is finite and negative (valid log-likelihood)
2. Failed models excluded from BMA weights (set weight to 0)
3. If all models fail: use conservative prior with warning
4. Validation results logged: "AAPL: 12/14 models passed validation"
5. Summary written to tune JSON: `validation: {passed: 12, failed: 2, warnings: [...]}`

**Tasks:**
- [ ] 3.5.1: Implement `validate_model_params(model_name, params)` function
- [ ] 3.5.2: Define per-parameter bounds as configuration dict
- [ ] 3.5.3: Integrate validation after MLE fitting, before BMA weight computation
- [ ] 3.5.4: Set failed model weights to 0 and renormalize surviving weights
- [ ] 3.5.5: Implement conservative prior fallback for complete failure
- [ ] 3.5.6: Add validation summary to tune JSON output
- [ ] 3.5.7: Write test: NaN parameter caught and excluded
- [ ] 3.5.8: Write test: extreme q (1e-12) flagged and bumped to floor
- [ ] 3.5.9: Write test: all models fail -> conservative prior used

---

### Story 3.6: Numba Coverage Gap Analysis and New Kernel Development

**As a** platform engineer,
**I want** the remaining non-Numba hot paths identified and compiled, and existing Numba
kernels benchmarked against Python reference implementations,
**so that** all performance-critical code runs at native speed.

**Background:**
`models/numba_kernels.py` is already extensive: 56 @njit(cache=True) functions covering
Kalman filters, Student-t distributions, Hansen Skew-t, GAS-Q, Monte Carlo, and scoring.
However, there may be remaining hot paths in `market_temperature.py` (ensemble forecasting),
`tune.py` (BMA weight computation), and `signals.py` (feature computation) that are still
Python-level loops. This story identifies and fills those gaps.

**Current Numba Coverage (already implemented):**
- Gaussian/phi-Gaussian/phi-Student-t filter kernels
- Hansen Skew-t and CST filter kernels
- Momentum-augmented filter kernels
- MS-q (Markov-switching process noise) kernels
- LFO-CV kernels, GARCH variance kernel
- CRPS, PIT, Anderson-Darling scoring kernels
- Monte Carlo simulation kernels
- GAS-Q filter kernel

**Dependencies:** None (standalone performance work)

**Acceptance Criteria:**
1. Hot path audit: profile `make stocks` with cProfile, identify top 20 functions by time
2. Any non-Numba function in top 10 by cumulative time: candidate for Numba kernel
3. Specific candidates likely identified:
   - Ensemble weight computation in market_temperature.py
   - Feature extraction loops (momentum, vol, skew computations)
   - BMA weight normalization across models
4. New kernels match Python reference to 1e-12 precision
5. Fallback to Python when Numba unavailable (graceful degradation -- already exists)
6. Benchmark: signal generation speedup documented per new kernel

**Risk:** Numba @njit has restrictions (no Python objects, no scipy inside @njit, no
dynamic memory allocation). Some functions may not be Numba-compatible and must remain
as Python. Profile first, then target only the ones that matter.

**Tasks:**
- [ ] 3.6.1: Profile `make stocks` with cProfile, export top 50 functions by cum_time
- [ ] 3.6.2: Cross-reference top functions with numba_kernels.py coverage
- [ ] 3.6.3: Identify >3 non-Numba hot paths in top 20
- [ ] 3.6.4: Implement Numba kernels for identified gaps (est. 2-4 new kernels)
- [ ] 3.6.5: Add precision tests: new Numba vs Python reference match to 1e-12
- [ ] 3.6.6: Benchmark: per-function speedup before/after Numba
- [ ] 3.6.7: Update models/numba_kernels.py with new kernels
- [ ] 3.6.8: Document Numba coverage in code comments (what's covered, what's not)

---

### Story 3.7: Live Tuning Progress in Frontend via SSE Enhancement

**As a** user monitoring tuning,
**I want** the live SSE retune stream to show per-asset progress with model-level detail
and estimated time remaining,
**so that** I can monitor the tuning process in real time from the web UI.

**Background:**
Current SSE stream (`/api/tune/retune/stream`) shows basic "Tuning AAPL [3/142]" messages.
This should be enhanced to show:
- Per-model convergence status within each asset
- BIC values as they are computed
- Validation gate pass/fail per model
- ETA based on rolling average of per-asset tune time

**Acceptance Criteria:**
1. SSE events include: asset, model, status, bic, time_elapsed, eta
2. Frontend TuningPage shows:
   - Overall progress bar with ETA
   - Currently-tuning asset with model-level status grid
   - Completed assets with pass/fail summary
3. Event types: `tune_start`, `model_complete`, `asset_complete`, `tune_done`
4. Frontend handles reconnection gracefully (resume from last known state)

**Tasks:**
- [ ] 3.7.1: Enhance SSE event format with structured JSON payloads
- [ ] 3.7.2: Add per-model status events during tuning
- [ ] 3.7.3: Compute rolling ETA from per-asset timing
- [ ] 3.7.4: Update frontend TuningPage with enhanced progress display
- [ ] 3.7.5: Implement model-level status grid component
- [ ] 3.7.6: Add SSE reconnection logic with state recovery
- [ ] 3.7.7: Write test: SSE events are valid JSON and properly sequenced

---

### Story 3.8: Tuning Cache Versioning and Migration

**As a** system architect,
**I want** tune cache files to include a version number and automatic migration logic,
**so that** format changes (new parameters, schema updates) do not break existing caches.

**Acceptance Criteria:**
1. Tune JSON includes `cache_version: 2` field (current implicit version is 1)
2. Migration function: v1 -> v2 adds GARCH and OU params with defaults
3. Future migrations: v2 -> v3 handled by chained migration functions
4. On load: version checked, migration applied if needed, migrated file saved
5. `make cache-migrate` target upgrades all cache files
6. Backup created before migration: `{SYMBOL}.json.bak`

**Tasks:**
- [ ] 3.8.1: Add `cache_version` field to tune JSON output
- [ ] 3.8.2: Implement version detection for existing caches (default to v1)
- [ ] 3.8.3: Implement v1->v2 migration (add garch_params, ou_params with defaults)
- [ ] 3.8.4: Implement migration chain runner (applies all needed migrations)
- [ ] 3.8.5: Add automatic migration on cache load in `signals.py`
- [ ] 3.8.6: Add `make cache-migrate` Makefile target
- [ ] 3.8.7: Add backup creation before migration
- [ ] 3.8.8: Write test: v1 cache migrated to v2 with correct defaults

---

## EPIC 4: BACKEND-FRONTEND SYNCHRONIZATION AND REAL-TIME DATA FLOW

**Epic Owner:** Full-Stack Lead
**Priority:** P1 -- High
**Estimated Scope:** 10 stories, ~38 tasks

**Problem Statement:**
Terminal output (`make stocks`) and frontend display currently show different data.
The terminal reads from `signals_ux.py` which formats from raw signal objects, while
the frontend reads from cached JSON via the backend API. Timestamps, staleness,
and data versions are not tracked. Users cannot trust that what they see in the
browser matches what the terminal just computed.

**Success Criteria:**
- Terminal and frontend show identical forecast values (byte-for-byte)
- Data freshness visible in both terminal and frontend
- WebSocket push for real-time updates (no polling for critical data)
- Cache invalidation is event-driven (new data triggers refresh)
- Version field in all cached data enables consistency checking
- Backend health endpoint detects stale/inconsistent data

---

### Story 4.1: Unified Signal Output Contract

**As a** system architect,
**I want** a single Python dataclass that defines the exact output structure for signals,
used by both terminal rendering and API serialization,
**so that** there is one source of truth for what a signal looks like.

**Background:**
Currently, `signals_ux.py` formats directly from internal signal objects, while
`signal_service.py` reads from JSON cache. Schema drift between these two paths
causes inconsistencies. A unified dataclass, serializable to both Rich tables and
Pydantic models, eliminates this risk.

The Pydantic v2 integration uses `from_attributes=True` (formerly `orm_mode`) to
create Pydantic models directly from the dataclass instances without manual mapping.

**Dependencies:** Story 1.8 (Display Precision -- formatting rules applied inside SignalOutput)

**Acceptance Criteria:**
1. `SignalOutput` dataclass defined in `src/decision/signal_output.py`
2. Fields: symbol, sector, crash_risk, momentum, horizon_forecasts (dict per horizon),
   confidence, regime, model_explanation, generated_at, data_version
3. Each horizon_forecast includes: point_forecast_pct, p10, p25, p50, p75, p90,
   direction_label, confidence_score, model_breakdown
4. `signals_ux.py` renders from `SignalOutput` objects
5. Signal cache JSON is serialization of `List[SignalOutput]`
6. Backend Pydantic model auto-generated from `SignalOutput` fields
7. Guaranteed: terminal and frontend display identical numbers

**Tasks:**
- [ ] 4.1.1: Create `src/decision/signal_output.py` with SignalOutput dataclass
- [ ] 4.1.2: Define HorizonForecast sub-dataclass with all quantile fields
- [ ] 4.1.3: Add `to_json()` and `from_json()` serialization methods
- [ ] 4.1.4: Add `to_rich_row()` method for terminal rendering
- [ ] 4.1.5: Refactor `signals_ux.py` to render from SignalOutput
- [ ] 4.1.6: Refactor `signals.py` to produce SignalOutput objects
- [ ] 4.1.7: Update signal cache JSON format to use SignalOutput serialization
- [ ] 4.1.8: Update backend Pydantic models to match SignalOutput fields
- [ ] 4.1.9: Write test: round-trip serialization (dataclass -> JSON -> dataclass)
- [ ] 4.1.10: Write test: terminal and API output match for same signal

---

### Story 4.2: Data Version Tracking and Consistency Checksums

**As a** system operator,
**I want** all cached data files to include a version hash that chains from price data
through tuning to signals,
**so that** I can verify that signals were generated from the latest tuned parameters
which were generated from the latest price data.

**Background:**
Data flows: prices -> tune -> signals. If prices update but tune does not re-run,
signals are stale. Version hashes create an auditable chain:
```
price_hash = sha256(prices_csv)[-8:]
tune_hash = sha256(price_hash + tune_params)[-8:]
signal_hash = sha256(tune_hash + signal_output)[-8:]
```

**Acceptance Criteria:**
1. Price files: `price_data_hash` computed from last 20 rows of OHLCV
2. Tune cache: includes `price_data_hash` (input) and `tune_hash` (output)
3. Signal cache: includes `tune_hash` (input) and `signal_hash` (output)
4. Backend health check: verifies hash chain consistency
5. Inconsistency flagged: "Signals generated from stale tune (price data updated)"
6. Frontend displays chain status: green (consistent), red (broken chain)

**Tasks:**
- [ ] 4.2.1: Implement price_data_hash computation (fast: last 20 rows only)
- [ ] 4.2.2: Add price_data_hash to tune JSON output
- [ ] 4.2.3: Add tune_hash to signal cache JSON output
- [ ] 4.2.4: Implement hash chain verification in health_service.py
- [ ] 4.2.5: Add chain status to `/api/services/health` response
- [ ] 4.2.6: Update frontend ServicesPage to display chain status
- [ ] 4.2.7: Write test: consistent chain passes verification
- [ ] 4.2.8: Write test: broken chain (stale tune) detected

---

### Story 4.3: WebSocket Push for Signal Updates

**As a** frontend user,
**I want** the dashboard to update automatically when new signals are generated
without polling,
**so that** I see fresh data within seconds of computation completing.

**Background:**
The backend already has WebSocket infrastructure (`ws.py`) but it is unused. When
signal generation completes, a WebSocket message should push the update to all
connected clients.

**Acceptance Criteria:**
1. WebSocket endpoint `ws://localhost:8000/ws` sends update events
2. Event types: `signals_updated`, `tune_completed`, `data_refreshed`, `error`
3. Signal update event includes: timestamp, changed_assets, data_version
4. Frontend auto-reconnects WebSocket with exponential backoff (1s, 2s, 4s, 8s, max 30s)
5. On `signals_updated`: relevant queries invalidated automatically
6. Connection status shown in UI: "Live" (green) / "Reconnecting..." (amber)

**Tasks:**
- [ ] 4.3.1: Implement WebSocket event broadcasting in `ws.py`
- [ ] 4.3.2: Add `signals_updated` event emission after signal generation
- [ ] 4.3.3: Add `tune_completed` event emission after tuning
- [ ] 4.3.4: Frontend WebSocket client with auto-reconnect
- [ ] 4.3.5: Integrate with React Query: invalidate stale queries on update event
- [ ] 4.3.6: Add connection status indicator to Layout component
- [ ] 4.3.7: Write test: WebSocket message triggers query invalidation
- [ ] 4.3.8: Write test: reconnection after disconnect

---

### Story 4.4: Task Pipeline Orchestration (prices -> tune -> signals)

**As a** system operator,
**I want** a single command that orchestrates the full pipeline (download prices,
incremental tune, generate signals) with proper dependency ordering,
**so that** `make stocks` produces fresh, consistent output end-to-end.

**Background:**
Currently `make stocks` downloads prices and generates signals, but may skip tuning.
If model parameters are stale, signals are suboptimal. A pipeline orchestrator
manages the full chain with incremental logic.

**Acceptance Criteria:**
1. `make pipeline` runs: prices -> incremental tune -> signals -> cache update
2. Each phase reports: time taken, assets processed, errors encountered
3. Failure handling per phase:
   - Prices fail: STOP (nothing downstream is valid without fresh prices)
   - Tune fails on <5% of assets: CONTINUE with warnings (most signals still valid)
   - Tune fails on >50% of assets: STOP (tune cache is unreliable)
   - Signals fail: STOP and report (user-facing output is broken)
4. `make pipeline --phase tune` runs only the tune phase
5. Pipeline status written to `src/data/pipeline_status.json`
6. Frontend can trigger pipeline via POST `/api/tasks/pipeline`
7. SSE stream for live pipeline progress

**Tasks:**
- [ ] 4.4.1: Create `src/pipeline.py` orchestrator with phase ordering
- [ ] 4.4.2: Implement per-phase execution with timing and error collection
- [ ] 4.4.3: Implement partial-failure handling (continue remaining phases)
- [ ] 4.4.4: Write pipeline_status.json with per-phase results
- [ ] 4.4.5: Add `make pipeline` Makefile target
- [ ] 4.4.6: Add `POST /api/tasks/pipeline` endpoint
- [ ] 4.4.7: Add SSE stream for pipeline progress
- [ ] 4.4.8: Update frontend with pipeline trigger and progress UI
- [ ] 4.4.9: Write test: full pipeline completes end-to-end

---

### Story 4.5: Backend API Response Optimization

**As a** frontend user,
**I want** API responses for signal data to load in < 200ms,
**so that** page transitions feel instant and the app feels responsive.

**Background:**
Current `/api/signals/summary` reads a large JSON file (potentially 500KB+) and
returns it wholesale. For the signal table, only a subset of fields is needed.
Pagination, field selection, and response compression can dramatically reduce latency.

**Acceptance Criteria:**
1. Support field selection: `?fields=symbol,sector,horizon_forecasts`
2. Support pagination: `?page=1&size=50` (default: all)
3. Support sector filter: `?sector=Technology`
4. Response compression: gzip enabled for responses > 1KB
5. In-memory cache: signal summary cached with 30s TTL (avoid file re-reads)
6. Benchmark: p95 response time < 150ms for paginated request

**Tasks:**
- [ ] 4.5.1: Add field selection query parameter to `/api/signals/summary`
- [ ] 4.5.2: Add pagination parameters with total count in response
- [ ] 4.5.3: Add sector filter parameter
- [ ] 4.5.4: Enable gzip middleware in FastAPI
- [ ] 4.5.5: Add in-memory TTL cache for signal summary
- [ ] 4.5.6: Profile response times and optimize hot paths
- [ ] 4.5.7: Write test: paginated response returns correct page and count
- [ ] 4.5.8: Write test: field selection reduces response size

---

### Story 4.6: Terminal-Frontend Display Parity Validation

**As a** developer,
**I want** an automated test that verifies terminal and frontend display the same
forecast values for a given signal snapshot,
**so that** display inconsistencies are caught before they reach users.

**Acceptance Criteria:**
1. Test fixture: known signal output with specific forecast values
2. Terminal rendering: extract formatted text via Rich console capture
3. API rendering: serialize through Pydantic and extract displayed values
4. Comparison: every forecast value matches to display precision
   (tolerance: abs(terminal_value - api_value) < 0.005% for all forecasts)
5. Test runs as part of `make tests`
6. Test covers: positive, negative, near-zero, and extreme forecasts

**Tasks:**
- [ ] 4.6.1: Create test fixture with known signal values
- [ ] 4.6.2: Implement terminal rendering capture using Rich console
- [ ] 4.6.3: Implement API response value extraction
- [ ] 4.6.4: Implement comparison logic with display-precision tolerance
- [ ] 4.6.5: Cover edge cases: near-zero, large positive, large negative
- [ ] 4.6.6: Add to test suite in `src/tests/test_display_parity.py`

---

### Story 4.7: Forecast Comparison Mode (Previous vs Current)

**As a** signal consumer,
**I want** to see how today's forecasts compare to yesterday's for each asset,
**so that** I can identify which assets have changed direction or magnitude.

**Acceptance Criteria:**
1. Previous signal snapshot stored: `src/data/signal_history/signals_{date}.json`
2. Comparison computed: delta per horizon per asset
3. Terminal display: arrow indicators (up/down) next to current forecast
4. Frontend display: change column or tooltip showing previous value and delta
5. Large changes highlighted: |delta| > abs(previous) * 0.5 flagged
6. History retained for 30 days, then pruned automatically

**Tasks:**
- [ ] 4.7.1: Implement signal snapshot saving with date-stamped filename
- [ ] 4.7.2: Implement comparison loading: find most recent previous snapshot
- [ ] 4.7.3: Compute per-asset per-horizon deltas
- [ ] 4.7.4: Update `signals_ux.py` with delta arrows in terminal display
- [ ] 4.7.5: Add delta data to API response
- [ ] 4.7.6: Update frontend signal table with change indicators
- [ ] 4.7.7: Implement 30-day history pruning
- [ ] 4.7.8: Write test: positive -> negative change detected
- [ ] 4.7.9: Write test: large change highlighted

---

### Story 4.8: Unified Error Reporting Pipeline

**As a** system operator,
**I want** all errors from tuning, signal generation, and data fetching to flow through
a single error reporting pipeline that is accessible via both terminal and frontend,
**so that** I have one place to diagnose issues.

**Acceptance Criteria:**
1. Error record structure: {timestamp, source, severity, asset, message, stack_trace}
2. Errors written to `src/data/errors/errors_{date}.json` (rolling 7-day)
3. Terminal: `make errors` displays recent errors in Rich table
4. Backend: `/api/services/errors` returns paginated error list
5. Frontend ServicesPage: error log tab with severity filtering
6. Severity levels: INFO, WARNING, ERROR, CRITICAL
7. CRITICAL errors trigger WebSocket push notification

**Tasks:**
- [ ] 4.8.1: Create `src/decision/error_reporter.py` with error record structure
- [ ] 4.8.2: Integrate error reporting into tuning pipeline (convergence failures)
- [ ] 4.8.3: Integrate into signal generation (fallback activations, NaN detections)
- [ ] 4.8.4: Integrate into data fetching (API failures, cache corruption)
- [ ] 4.8.5: Add `make errors` Makefile target
- [ ] 4.8.6: Add `/api/services/errors` endpoint with pagination
- [ ] 4.8.7: Update ServicesPage with error log component
- [ ] 4.8.8: Add WebSocket push for CRITICAL errors
- [ ] 4.8.9: Implement 7-day rolling cleanup
- [ ] 4.8.10: Write test: error recorded and retrievable

---

### Story 4.9: Configurable Refresh Intervals and Smart Polling

**As a** frontend user,
**I want** the dashboard to poll for updates intelligently based on market hours
and data freshness,
**so that** it does not waste resources during off-hours but is responsive during trading.

**Acceptance Criteria:**
1. Market hours detection: US market open 9:30-16:00 ET on weekdays
   NOTE: This is US-centric by design (assets are US-listed). If international assets
   are added later, market hours detection must become per-exchange.
2. During market hours: poll every 60s for signals, 30s for health
3. After hours: poll every 5 minutes (data unlikely to change)
4. Weekends: poll every 15 minutes
5. User can override with manual refresh button
6. Poll interval shown in UI: "Refreshing every 60s (market open)"

**Tasks:**
- [ ] 4.9.1: Implement market hours detection utility in frontend
- [ ] 4.9.2: Configure React Query refetch intervals based on market state
- [ ] 4.9.3: Add manual refresh button to each page
- [ ] 4.9.4: Display current poll interval in footer
- [ ] 4.9.5: Write test: market open returns 60s interval
- [ ] 4.9.6: Write test: weekend returns 900s interval

---

### Story 4.10: API Health Dashboard with Dependency Graph

**As a** system operator,
**I want** the health dashboard to show a dependency graph of all system components
(prices, tuning, signals, API, frontend) with live status,
**so that** I can quickly identify which component is causing issues.

**Acceptance Criteria:**
1. Dependency graph: prices -> tune -> signals -> API -> frontend
2. Each node shows: status (green/amber/red), last update time, data version
3. Edges show: data hash consistency (solid = consistent, dashed = stale)
4. Graph rendered in frontend using a simple directed graph visualization
5. Click on node: shows detail card with metrics and errors
6. Auto-refresh every 30 seconds

**Tasks:**
- [ ] 4.10.1: Design dependency graph data structure in backend
- [ ] 4.10.2: Compute per-node status from health checks
- [ ] 4.10.3: Compute edge consistency from hash chain verification
- [ ] 4.10.4: Add `/api/services/dependency-graph` endpoint
- [ ] 4.10.5: Implement frontend dependency graph component
- [ ] 4.10.6: Add click-to-detail interaction
- [ ] 4.10.7: Write test: all-green graph for consistent state
- [ ] 4.10.8: Write test: broken edge detected for stale tune

---

## EPIC 5: WORLD-CLASS FRONTEND UX -- APPLE-CALIBER DESIGN

**Epic Owner:** UX/Frontend Lead
**Priority:** P1 -- High
**Estimated Scope:** 14 stories, ~55 tasks

**Problem Statement:**
The current frontend is functional but not exceptional. To make Apple engineers envious,
we need: fluid micro-animations, information density without clutter, immediate feedback,
spatial reasoning support (charts that tell stories), and a design language so cohesive
that every pixel feels intentional.

**Design Principles:**
1. **Information at a glance**: The most important data (direction, magnitude, confidence)
   visible without scrolling or clicking
2. **Progressive revelation**: Details available on demand, never forced
3. **Spatial memory**: Consistent layouts so users build muscle memory
4. **Micro-feedback**: Every interaction has visible response within 16ms
5. **Dark-first luxury**: Deep blues and subtle glows, not "dark gray everywhere"

**Success Criteria:**
- Signal table loads and renders in < 300ms
- Every table cell change animates smoothly (no visual jumps)
- Forecast fan charts render with sub-frame performance
- Touch/mobile experience is first-class (not an afterthought)
- Zero layout shift (CLS = 0)
- Accessibility: WCAG 2.1 AA compliance

---

### Story 5.1: Signal Table Redesign -- Heat Map Matrix with Sparklines

**As a** signal consumer,
**I want** the signal table to display forecasts as a heat map with embedded sparklines,
**so that** I can instantly identify the strongest opportunities across assets and horizons.

**Background:**
The current table shows text values (+0.0%, +0.3%, etc.) which are hard to compare
visually across 100+ rows. A heat map where color intensity represents forecast
strength, combined with tiny sparklines showing recent trajectory, transforms the
table into a visual intelligence surface.

**Dependencies:** Story 4.1 (Unified Signal Output Contract -- sparkline data included in API)

**Risk:** Color mapping that is not perceptually uniform (e.g., RGB interpolation) creates
misleading visual comparisons. Green and red are problematic for colorblind users (~8% of
male population). Mitigation: use a blue-orange diverging palette as default, with
red-green as a selectable option for non-colorblind users.

**Acceptance Criteria:**
1. Each horizon cell contains:
   - Background color: diverging palette (deep red -> neutral -> deep green)
   - Foreground text: forecast percentage with adaptive precision (Story 1.8)
   - Micro-sparkline (7px height): last 5 values for this horizon
2. Color mapping: linear scale from -max_abs to +max_abs (symmetric)
3. Row hover: entire row highlights with subtle glow, tooltip shows full detail
4. Column sort: click header to sort by any horizon
5. Sticky headers: symbol column and horizon headers stick on scroll
6. Performance: 100+ rows render in < 100ms (virtualized with @tanstack/react-virtual)
   Note: react-window is legacy; @tanstack/react-virtual integrates better with existing
   TanStack ecosystem (React Query, Router).

**Tasks:**
- [ ] 5.1.1: Design diverging color palette: 11-step red-to-green with neutral dead zone
- [ ] 5.1.2: Implement heat map cell component with background color
- [ ] 5.1.3: Implement micro-sparkline component (canvas-based, 7px height)
- [ ] 5.1.4: Add sparkline data to API response (last 5 forecasts per horizon)
- [ ] 5.1.5: Implement column sorting with sort indicators
- [ ] 5.1.6: Implement sticky headers (symbol column + horizon row)
- [ ] 5.1.7: Implement row virtualization (react-window or @tanstack/virtual)
- [ ] 5.1.8: Add row hover glow effect with tooltip
- [ ] 5.1.9: Performance test: 150 rows render < 100ms
- [ ] 5.1.10: Accessibility: contrast ratio check for all heat map colors

---

### Story 5.2: Forecast Fan Chart Visualization

**As a** signal consumer,
**I want** a fan chart for each asset showing the point forecast and confidence intervals
across all horizons,
**so that** I can visually assess both direction and uncertainty.

**Background:**
Fan charts show the forecast as a cone expanding into the future:
- Median line (bold): point forecast trajectory
- Dark band: 25th-75th percentile
- Light band: 10th-90th percentile
The width of the cone communicates uncertainty visually.

**Dependencies:** Story 2.5 (Forecast Confidence Intervals -- provides the quantile data)

**Acceptance Criteria:**
1. Fan chart component: SVG-based, responsive, dark theme
2. X-axis: horizons (1d, 3d, 1w, 1m, 3m, 6m, 12m) -- log scale
3. Y-axis: forecast return percentage
4. Median line: smooth Bezier curve through point forecasts
5. Bands: filled areas between quantile curves with graduated opacity
6. Current price: horizontal dashed line at 0%
7. Mouse hover: crosshair shows exact values at cursor position
8. Loads in ChartsPage and in signal detail modal

**Tasks:**
- [ ] 5.2.1: Design fan chart component with SVG/Canvas rendering
- [ ] 5.2.2: Implement quantile data loading from API
- [ ] 5.2.3: Implement Bezier curve fitting through discrete points
- [ ] 5.2.4: Implement graduated opacity bands (outer lighter, inner darker)
- [ ] 5.2.5: Add crosshair interaction with value tooltip
- [ ] 5.2.6: Implement responsive sizing (fills container)
- [ ] 5.2.7: Add to ChartsPage forecast section
- [ ] 5.2.8: Add to signal row detail expansion
- [ ] 5.2.9: Performance test: renders in < 50ms for single asset

---

### Story 5.3: Regime Indicator and Context Bar

**As a** signal consumer,
**I want** a persistent context bar at the top of every page showing the current
market regime, VIX level, and overall market direction,
**so that** I always have macro context for interpreting individual asset signals.

**Acceptance Criteria:**
1. Context bar: full width, 40px height, fixed at top of content area
2. Left section: regime badge (LOW_VOL_TREND = blue, CRISIS_JUMP = red, etc.)
3. Center section: VIX level with mini-chart (last 20 days)
4. Right section: SPY/QQQ/IWM today's returns with colored arrows
5. Animated transitions: regime change slides in new badge with 300ms transition
6. Data source: `/api/risk/summary` with 60s refresh

**Tasks:**
- [ ] 5.3.1: Design context bar layout with 3 sections
- [ ] 5.3.2: Implement regime badge component with color per regime
- [ ] 5.3.3: Implement VIX mini-chart (20 bars, canvas, 30px height)
- [ ] 5.3.4: Implement index returns display with color coding
- [ ] 5.3.5: Add animated regime transition (CSS transition)
- [ ] 5.3.6: Add to Layout.tsx above Outlet
- [ ] 5.3.7: Connect to risk summary API with 60s polling
- [ ] 5.3.8: Write test: regime badge renders correctly for each regime

---

### Story 5.4: Asset Detail Modal with Deep Analytics

**As a** signal consumer,
**I want** to click on any asset row and get a modal showing comprehensive analytics:
price chart, fan chart, model weights, regime history, and trade reasoning,
**so that** I can make informed decisions without navigating away from the signal table.

**Acceptance Criteria:**
1. Modal: 80% viewport width, slide-in from right, dark glass background
2. Header: Symbol, sector, current price, today's change
3. Tab 1 - Overview: Fan chart + signal summary + regime indicator
4. Tab 2 - Price History: Lightweight-charts OHLCV with volume
5. Tab 3 - Model Detail: BMA weight pie chart + per-model parameters table
6. Tab 4 - Forecast Breakdown: per-model contribution bars per horizon
7. Close: ESC key, click outside, or X button
8. URL hash: `#detail/AAPL` enables direct linking

**Tasks:**
- [ ] 5.4.1: Implement slide-in modal container with glass backdrop
- [ ] 5.4.2: Design header with price data from API
- [ ] 5.4.3: Implement Overview tab with fan chart + summary
- [ ] 5.4.4: Implement Price History tab with Lightweight-charts
- [ ] 5.4.5: Implement Model Detail tab with weight pie + params table
- [ ] 5.4.6: Implement Forecast Breakdown tab with contribution bars
- [ ] 5.4.7: Add keyboard navigation (ESC to close, Tab between tabs)
- [ ] 5.4.8: Add URL hash for direct linking
- [ ] 5.4.9: Performance: modal opens in < 100ms, data loads in < 300ms
- [ ] 5.4.10: Mobile: full screen on < 768px

---

### Story 5.5: Micro-Animations and State Transitions

**As a** frontend user,
**I want** every data change to animate smoothly -- numbers counting up/down,
cells flashing on update, loading states with skeleton screens,
**so that** the interface feels alive and responsive.

**Acceptance Criteria:**
1. Number transitions: forecast values animate from old to new (300ms ease-out)
2. Cell flash: updated cells briefly glow green (up) or red (down) for 500ms
3. Loading states: skeleton screens match layout of loaded content (no layout shift)
4. Page transitions: fade-in content (200ms) after route change
5. Table sort: rows animate to new positions (250ms)
6. All animations respect `prefers-reduced-motion` media query
7. 60fps during all animations (measured via Chrome DevTools)

**Tasks:**
- [ ] 5.5.1: Implement number transition component (animates between values)
- [ ] 5.5.2: Implement cell flash effect (CSS keyframe with value-based color)
- [ ] 5.5.3: Design skeleton screens for each page layout
- [ ] 5.5.4: Implement route transition animations (Framer Motion or CSS)
- [ ] 5.5.5: Implement animated table row reordering
- [ ] 5.5.6: Add `prefers-reduced-motion` respect throughout
- [ ] 5.5.7: Performance audit: verify 60fps during animations
- [ ] 5.5.8: Write visual regression tests for key animations

---

### Story 5.6: Advanced Filtering and Search with Command Palette

**As a** power user,
**I want** a command palette (Cmd+K) that lets me quickly filter assets, jump to pages,
and execute actions,
**so that** I can navigate the system as fast as I think.

**Acceptance Criteria:**
1. Cmd+K (Mac) / Ctrl+K (Win): opens command palette
2. Search modes:
   - Type symbol: jumps to asset detail (e.g., "AAPL")
   - Type sector: filters to sector (e.g., "tech")
   - Type command: executes action (e.g., "refresh signals")
   - Type page: navigates (e.g., "risk dashboard")
3. Fuzzy matching with highlighted matched characters
4. Recent items: last 5 commands shown on open
5. Keyboard navigation: up/down arrows, Enter to select, ESC to close
6. Renders < 16ms per keystroke (instant feel)

**Tasks:**
- [ ] 5.6.1: Implement command palette overlay component
- [ ] 5.6.2: Build asset index for fuzzy search (symbol + name + sector)
- [ ] 5.6.3: Implement fuzzy matcher with score-based ranking
- [ ] 5.6.4: Add command registry: refresh, navigate, filter actions
- [ ] 5.6.5: Implement keyboard handling (Cmd+K, arrows, Enter, ESC)
- [ ] 5.6.6: Add recent items persistence (localStorage)
- [ ] 5.6.7: Style with glass morphism matching app theme
- [ ] 5.6.8: Performance: render < 16ms per keystroke

---

### Story 5.7: Portfolio Impact Visualization

**As a** portfolio manager,
**I want** to see how signals translate to portfolio-level impact -- expected PnL,
risk contribution, and sector exposure -- on a dedicated portfolio view,
**so that** I understand the aggregate effect of acting on all signals.

**Acceptance Criteria:**
1. Portfolio view: assumes equal-weight or user-defined allocation
2. Displays: total expected return, portfolio vol, Sharpe estimate, sector exposure pie
3. Risk decomposition: which assets contribute most to portfolio risk
4. Scenario analysis: "if all forecasts are correct, portfolio returns X%"
5. Sector concentration warning if > 40% in single sector
6. Interactive: click sector slice to see constituent assets

**Tasks:**
- [ ] 5.7.1: Implement portfolio aggregation logic in backend
- [ ] 5.7.2: Add `/api/signals/portfolio-impact` endpoint
- [ ] 5.7.3: Design portfolio view layout with key metrics
- [ ] 5.7.4: Implement sector exposure pie chart (Recharts)
- [ ] 5.7.5: Implement risk decomposition bar chart
- [ ] 5.7.6: Add concentration warning logic
- [ ] 5.7.7: Add interactive click-through from sector to assets
- [ ] 5.7.8: Add to navigation as "Portfolio" page

---

### Story 5.8: Responsive Mobile-First Layout Overhaul

**As a** mobile user,
**I want** the signal dashboard to be fully usable on my phone with touch-optimized
interactions and information-dense but readable layouts,
**so that** I can check signals on the go.

**Acceptance Criteria:**
1. Signal table: horizontal swipe for horizons, sticky symbol column
2. Touch targets: minimum 44x44px for all interactive elements
3. Bottom navigation bar on mobile (replaces sidebar)
4. Swipe gestures: left/right between pages
5. Font sizes: minimum 14px on mobile, optimal readability
6. Charts: touch-to-inspect (long press shows crosshair)
7. No horizontal scrolling needed for primary content
8. Tested on: iPhone 14 (375px), iPad (768px), Android (360px)

**Tasks:**
- [ ] 5.8.1: Implement bottom navigation bar for mobile (< 768px)
- [ ] 5.8.2: Redesign signal table for horizontal swipe paradigm
- [ ] 5.8.3: Add touch-to-inspect on charts (long press = crosshair)
- [ ] 5.8.4: Audit and fix all touch targets (min 44x44px)
- [ ] 5.8.5: Implement swipe page navigation
- [ ] 5.8.6: Test on iPhone 14 Safari viewport (375 x 812)
- [ ] 5.8.7: Test on iPad viewport (768 x 1024)
- [ ] 5.8.8: Performance: Lighthouse mobile score > 90

---

### Story 5.9: Model Explainability Tooltips

**As a** signal consumer,
**I want** every forecast cell to have a tooltip explaining why the system predicts
that value, showing model contributions and confidence factors,
**so that** I trust the system's reasoning.

**Acceptance Criteria:**
1. Hover on forecast cell (desktop) or tap (mobile): shows explainability tooltip
2. Tooltip content:
   - Top contributor: "Momentum: +1.2% (weight 0.35)"
   - Confidence: "High -- 4/5 models agree on direction"
   - Regime: "LOW_VOL_TREND -- favors momentum signals"
   - Risk factor: "VIX at 14.2 (calm) -- tighter intervals"
3. Tooltip appears within 200ms of hover start
4. Tooltip dismisses on mouse leave or tap elsewhere
5. Max tooltip width: 320px, dark glass background
6. Data sourced from model_breakdown field (Story 2.9)

**Tasks:**
- [ ] 5.9.1: Design tooltip layout with 4 sections
- [ ] 5.9.2: Implement tooltip container with positioning logic
- [ ] 5.9.3: Load model breakdown data from API response
- [ ] 5.9.4: Implement delay logic: 200ms hover before show
- [ ] 5.9.5: Add mobile tap variant (tap to toggle)
- [ ] 5.9.6: Style with glass morphism, max 320px width
- [ ] 5.9.7: Write test: tooltip shows correct model as top contributor

---

### Story 5.10: Export and Sharing Capabilities

**As a** signal consumer,
**I want** to export signal data as CSV, PDF, or clipboard-friendly format,
and share specific views via URL,
**so that** I can use the data in external tools and share findings with colleagues.

**Acceptance Criteria:**
1. Export button on every data table: CSV, PDF, Clipboard
2. CSV: includes all visible columns with proper formatting
3. PDF: styled report with charts, tables, generation timestamp
4. Clipboard: tab-separated values for pasting into Excel/Sheets
5. Shareable URLs: every filter/sort/view state encoded in URL params
6. "Copy link" button copies shareable URL to clipboard

**Tasks:**
- [ ] 5.10.1: Implement CSV export utility (client-side generation)
- [ ] 5.10.2: Implement PDF export using html2canvas + jsPDF
- [ ] 5.10.3: Implement clipboard export (tab-separated)
- [ ] 5.10.4: Add export dropdown button to table components
- [ ] 5.10.5: Implement URL state encoding (filters, sort, page)
- [ ] 5.10.6: Add "Copy link" button to each page
- [ ] 5.10.7: Write test: CSV export contains correct data
- [ ] 5.10.8: Write test: URL state round-trips correctly

---

### Story 5.11: Keyboard-First Navigation

**As a** power user,
**I want** comprehensive keyboard shortcuts for all common actions,
**so that** I never need to reach for the mouse during my workflow.

**Acceptance Criteria:**
1. Global shortcuts:
   - `Cmd+K`: Command palette (Story 5.6)
   - `1-9`: Navigate to page (1=Overview, 2=Signals, etc.)
   - `R`: Refresh current page data
   - `?`: Show shortcut help overlay
2. Table shortcuts:
   - `j/k`: Move selection up/down
   - `Enter`: Open detail modal for selected row
   - `ESC`: Close modal / clear selection
   - `/`: Focus search input
3. Help overlay: two-column layout showing all shortcuts
4. Shortcuts disabled when typing in inputs

**Tasks:**
- [ ] 5.11.1: Implement global shortcut handler with input-aware disabling
- [ ] 5.11.2: Add page navigation shortcuts (1-9)
- [ ] 5.11.3: Implement table row selection with j/k navigation
- [ ] 5.11.4: Implement Enter to open detail, ESC to close
- [ ] 5.11.5: Design and implement shortcut help overlay
- [ ] 5.11.6: Add `?` trigger for help overlay
- [ ] 5.11.7: Write test: shortcuts respond correctly, disabled during input

---

### Story 5.12: Signal Watchlist with Custom Alerts

**As a** signal consumer,
**I want** to create a personal watchlist of assets and receive visual alerts
when their signals change significantly,
**so that** I stay informed about the assets I care about most.

**Acceptance Criteria:**
1. Watchlist: user-defined list of symbols stored in localStorage
2. Add/remove from watchlist: star icon on each asset row
3. Watchlist view: filtered signal table showing only starred assets
4. Alert conditions: forecast direction change, magnitude > threshold, regime change
5. Visual alerts: badge count on watchlist nav item + notification card
6. Alert threshold configurable per asset

**Tasks:**
- [ ] 5.12.1: Implement watchlist store in localStorage
- [ ] 5.12.2: Add star toggle icon to asset rows
- [ ] 5.12.3: Implement watchlist filter view
- [ ] 5.12.4: Implement alert detection logic (direction change, magnitude threshold)
- [ ] 5.12.5: Add notification badge to nav item
- [ ] 5.12.6: Implement notification card dropdown
- [ ] 5.12.7: Add per-asset alert threshold configuration
- [ ] 5.12.8: Write test: direction change triggers alert

---

### Story 5.13: Performance Budget and Lighthouse Optimization

**As a** frontend developer,
**I want** the application to maintain Lighthouse scores > 90 across all categories,
**so that** the app is fast, accessible, and follows best practices.

**Acceptance Criteria:**
1. Lighthouse Performance > 90 (LCP < 2.5s, FID < 100ms, CLS = 0)
2. Lighthouse Accessibility > 90 (WCAG 2.1 AA compliance)
3. Lighthouse Best Practices > 90
4. Bundle size < 300KB gzipped (excluding chart libraries)
5. Code splitting: each page lazy-loaded
6. Image optimization: all icons as SVG, no raster images
7. Budget enforced in CI/build step

**Tasks:**
- [ ] 5.13.1: Run initial Lighthouse audit and record baseline
- [ ] 5.13.2: Implement code splitting with React.lazy()
- [ ] 5.13.3: Bundle size analysis: identify large dependencies
- [ ] 5.13.4: Optimize Lightweight-charts import (tree-shake)
- [ ] 5.13.5: Add font subsetting (only used characters)
- [ ] 5.13.6: Accessibility audit: fix all WCAG 2.1 AA issues
- [ ] 5.13.7: Add CLS prevention: reserved height for dynamic content
- [ ] 5.13.8: Write performance budget check script

---

### Story 5.14: Design System Documentation with Component Storybook

**As a** frontend developer,
**I want** a living design system documenting all components, colors, typography,
and interaction patterns,
**so that** new features maintain visual consistency.

**Acceptance Criteria:**
1. Component library documented: StatCard, HeatMapCell, Sparkline, FanChart, etc.
2. Color system: all 12 themed variables with usage guidelines
3. Typography scale: font sizes, weights, line heights
4. Spacing scale: consistent spacing tokens (4px, 8px, 12px, 16px, 24px, 32px)
5. Interactive component gallery (Storybook or custom page)
6. Dark theme guidelines: color contrast ratios documented

**Tasks:**
- [ ] 5.14.1: Document color system with hex values and usage guidelines
- [ ] 5.14.2: Document typography scale with examples
- [ ] 5.14.3: Document spacing tokens
- [ ] 5.14.4: Create component gallery page in app (/design-system route)
- [ ] 5.14.5: Add component props documentation
- [ ] 5.14.6: Document interaction patterns (hover, click, drag)
- [ ] 5.14.7: Document accessible color combinations

---

## EPIC 6: VALIDATION, BACKTESTING, AND PROFITABILITY PROOF

**Epic Owner:** Quant Research Lead
**Priority:** P0 -- Critical Path
**Estimated Scope:** 10 stories, ~40 tasks

**Problem Statement:**
No change should be merged without proof that it improves profitability. Currently,
there is no automated way to measure whether a forecast change makes money. We need
a backtesting framework that simulates trading on historical forecasts, measures PnL,
and gates changes that regress profitability.

**Success Criteria:**
- Walk-forward backtest on 12-symbol universe with 2-year history
- Positive Sharpe > 0.5 for long-only strategy based on forecasts
- Positive Sharpe > 0.3 for long/short strategy
- Maximum drawdown < 20%
- Hit rate > 55% at 7-day horizon
- All metrics tracked automatically and gated in CI

---

### Story 6.1: Walk-Forward Backtest Engine for Forecast Signals

**As a** quant researcher,
**I want** a walk-forward backtesting engine that evaluates forecast profitability
on out-of-sample data by simulating paper trading on historical signals,
**so that** I can measure whether forecast improvements translate to real PnL.

**Background:**
Walk-forward testing avoids look-ahead bias by:
1. Training models on data up to time T
2. Generating forecasts for T+1 to T+H
3. Measuring realized returns at T+H
4. Rolling window forward and repeating

This is more rigorous than the existing Arena backtest because it tests the full
pipeline (tune -> forecast -> trade decision) in temporal order.

**Dependencies:** None (foundational -- many stories depend on this)
**NOTE:** A MINIMAL version of this story (daily step, Sharpe+hit_rate only) must be
implemented in Phase 1 to serve as the validation gate. Full version with all metrics,
per-horizon analysis, and configurable parameters is Phase 5.

**Risk:** Walk-forward with daily re-tuning is computationally expensive (252 tune runs
per year). Mitigation: use weekly step (step=5) as default, daily as option. Also,
incremental tuning (Story 3.1) makes each re-tune fast by only updating changed assets.

**Acceptance Criteria:**
1. Walk-forward engine with configurable:
   - Training window: 252 days (default)
   - Step size: 1 day (daily rebalance) or 5 days (weekly)
   - Horizons evaluated: [1, 3, 7, 30] days
2. Per-step output: {date, symbol, forecast_pct, realized_return, direction_correct}
3. Summary metrics: Sharpe, Sortino, max_drawdown, hit_rate, information_coefficient
4. Per-horizon metrics: accuracy at each forecast horizon
5. Runs on benchmark 12-symbol universe in < 10 minutes
6. Results stored in `src/data/backtest/walkforward_results.json`

**Tasks:**
- [ ] 6.1.1: Create `src/calibration/walkforward_backtest.py` engine
- [ ] 6.1.2: Implement training window management with rolling update
- [ ] 6.1.3: Implement per-step tune + forecast generation
- [ ] 6.1.4: Implement realized return comparison
- [ ] 6.1.5: Compute aggregate metrics (Sharpe, Sortino, max_dd)
- [ ] 6.1.6: Compute per-horizon accuracy metrics
- [ ] 6.1.7: Store results in structured JSON
- [ ] 6.1.8: Add `make walkforward` Makefile target
- [ ] 6.1.9: Write test: known profitable series -> positive Sharpe
- [ ] 6.1.10: Write test: random series -> near-zero Sharpe

---

### Story 6.2: Signal-Based Trading Strategy Simulator

**As a** portfolio manager,
**I want** a trading strategy simulator that converts forecasts into position sizes
and tracks cumulative PnL,
**so that** I can see what following these signals would have returned.

**Background:**
Trading rules:
- Forecast > +0.3%: go long with size proportional to forecast magnitude
- Forecast < -0.3%: go short (if enabled) or stay flat (long-only mode)
- Position size: min(1.0, abs(forecast) / vol_adjusted_cap) * max_position
  where vol_adjusted_cap = asset_252d_vol * sqrt(horizon/252) * 3
  (3-sigma move as maximum expected forecast magnitude, per-asset)
- Transaction costs: 5bps per trade

**Acceptance Criteria:**
1. Long-only mode: buy when forecast > threshold, hold until forecast reverses
2. Long/short mode: position reflects forecast sign and magnitude
3. Position sizing: proportional to forecast strength with risk cap
4. Transaction costs: configurable bps per trade
5. Metrics: cumulative PnL curve, Sharpe, max drawdown, turnover
6. Equity curve exportable for plotting
7. Benchmark: long-only Sharpe > 0.5 on 12-symbol universe

**Tasks:**
- [ ] 6.2.1: Implement position sizing logic based on forecast magnitude
- [ ] 6.2.2: Implement long-only and long/short execution modes
- [ ] 6.2.3: Add transaction cost model (configurable bps)
- [ ] 6.2.4: Track cumulative PnL with daily granularity
- [ ] 6.2.5: Compute Sharpe, Sortino, max drawdown, turnover
- [ ] 6.2.6: Export equity curve as CSV for external analysis
- [ ] 6.2.7: Write test: always-long on uptrend -> positive PnL
- [ ] 6.2.8: Write test: random positions -> near-zero PnL minus costs

---

### Story 6.3: Per-Asset Forecast Quality Report

**As a** quant researcher,
**I want** a detailed per-asset report showing forecast quality metrics
(hit rate, MAE, IC, calibration) at each horizon,
**so that** I can identify which assets the system forecasts well and which need attention.

**Acceptance Criteria:**
1. Report includes for each asset x horizon:
   - Directional hit rate (% forecasts with correct sign)
   - Mean absolute error (forecast vs realized)
   - Information coefficient (rank correlation of forecast vs realized)
   - Calibration score (ECE of probability estimates)
   - Sample size (number of evaluated forecasts)
2. Color coding: green (>60% hit), amber (50-60%), red (<50%)
3. Terminal display: Rich table sorted by overall quality
4. Frontend: sortable table on DiagnosticsPage
5. "Problem assets" list: assets with hit rate < 50% at any horizon

**Tasks:**
- [ ] 6.3.1: Implement per-asset per-horizon metric computation
- [ ] 6.3.2: Implement IC (Spearman rank correlation) calculation
- [ ] 6.3.3: Generate quality report from walkforward results
- [ ] 6.3.4: Add terminal display with Rich color-coded table
- [ ] 6.3.5: Add `/api/diagnostics/forecast-quality` endpoint
- [ ] 6.3.6: Add forecast quality tab to DiagnosticsPage
- [ ] 6.3.7: Implement "problem assets" detection logic
- [ ] 6.3.8: Write test: perfect forecaster -> all metrics at 100%

---

### Story 6.4: Profitability Gate for Code Changes

**As a** developer,
**I want** an automated profitability check that runs as part of the validation
pipeline and rejects changes that degrade forecast profitability,
**so that** every merge makes the system better.

**Background:**
Before any change to tune.py, signals.py, or market_temperature.py is merged:
1. Run walkforward backtest on 12-symbol universe
2. Compare metrics to stored baseline
3. If Sharpe drops by > 0.05 or hit rate drops by > 2%: FAIL
4. If both improve: PASS with summary

**Acceptance Criteria:**
1. `make validate-profitability` target runs full validation
2. Compares: Sharpe, hit_rate, max_drawdown, CRPS against baseline
3. RED flag: any metric degrades beyond threshold
4. GREEN flag: all metrics improve or maintain
5. Baseline stored in `src/data/calibration/profitability_baseline.json`
6. Results logged with timestamp for historical tracking
7. Exit code 0 for PASS, 1 for FAIL (CI compatible)

**Tasks:**
- [ ] 6.4.1: Create `src/calibration/profitability_gate.py`
- [ ] 6.4.2: Implement baseline comparison logic with thresholds
- [ ] 6.4.3: Implement traffic light classification (GREEN, AMBER, RED)
- [ ] 6.4.4: Store and load baseline JSON with versioning
- [ ] 6.4.5: Add `make validate-profitability` Makefile target
- [ ] 6.4.6: Generate human-readable summary report
- [ ] 6.4.7: Write test: improved metrics -> GREEN
- [ ] 6.4.8: Write test: degraded Sharpe -> RED

---

### Story 6.5: Regime-Specific Profitability Analysis

**As a** quant researcher,
**I want** profitability metrics broken down by market regime,
**so that** I can identify if the system is profitable in trending markets but losing
in range-bound ones (or vice versa).

**Acceptance Criteria:**
1. Walkforward results tagged with regime at each step
2. Per-regime metrics: Sharpe, hit_rate, avg_return, max_drawdown
3. Regime transition analysis: profitability around regime changes
4. Report highlights: "Best regime: LOW_VOL_TREND (Sharpe 1.2)"
5. Report highlights: "Worst regime: HIGH_VOL_RANGE (Sharpe -0.3)"
6. Frontend: regime profitability breakdown chart

**Tasks:**
- [ ] 6.5.1: Tag each walkforward step with current regime
- [ ] 6.5.2: Compute per-regime profitability metrics
- [ ] 6.5.3: Analyze transition period performance (5 bars before/after change)
- [ ] 6.5.4: Generate regime profitability summary
- [ ] 6.5.5: Add to terminal report output
- [ ] 6.5.6: Add `/api/diagnostics/regime-profitability` endpoint
- [ ] 6.5.7: Add regime profitability chart to DiagnosticsPage
- [ ] 6.5.8: Write test: trending regime has better metrics than range

---

### Story 6.6: Sector and Cap-Weighted Performance Attribution

**As a** portfolio manager,
**I want** performance attributed by sector and market cap bucket,
**so that** I know if the system works better for tech vs energy, or large-cap vs small-cap.

**Acceptance Criteria:**
1. Sectors: Technology, Finance, Healthcare, Energy, Consumer, Industrials, Index
2. Cap buckets: Small (< $5B), Mid ($5-50B), Large (> $50B), Index
3. Per-group metrics: Sharpe, hit_rate, max_drawdown, avg_position_size
4. Performance attribution: which groups contribute most to portfolio PnL
5. Frontier chart: Sharpe vs drawdown by group
6. Frontend: interactive scatter plot or grouped bar chart

**Tasks:**
- [ ] 6.6.1: Classify assets by sector and cap bucket
- [ ] 6.6.2: Compute per-group profitability metrics from walkforward results
- [ ] 6.6.3: Implement performance attribution (PnL contribution by group)
- [ ] 6.6.4: Generate terminal summary with Rich table
- [ ] 6.6.5: Add `/api/diagnostics/performance-attribution` endpoint
- [ ] 6.6.6: Implement frontend scatter/bar chart for group comparison
- [ ] 6.6.7: Write test: diversified universe has lower drawdown than single sector

---

### Story 6.7: Monte Carlo Confidence on Backtest Metrics

**As a** quant researcher,
**I want** confidence intervals on backtest metrics via block bootstrap Monte Carlo,
**so that** I can distinguish genuine skill from lucky sample.

**Background:**
A Sharpe of 0.6 from 252 daily observations has wide uncertainty:
- Bootstrap 95% CI might be [0.1, 1.1]
- If CI includes 0, we cannot reject "no skill" hypothesis

CRITICAL: Daily returns are serially correlated (momentum, volatility clustering).
Standard iid bootstrap (resample individual days) destroys this dependence and produces
artificially tight confidence intervals. **Block bootstrap** (Politis & Romano 1994)
resamples contiguous blocks of returns, preserving temporal structure:
```
Block size: b = ceil(n^(1/3)) ~ 6 days for 252 observations
Resample: draw n/b blocks with replacement, concatenate
```

**Dependencies:** Story 6.1 (Walk-Forward Backtest -- provides daily return series)

**Acceptance Criteria:**
1. Block bootstrap: 1000 resamples with block size = ceil(n^(1/3))
2. Stationary block bootstrap variant: geometric block length distribution (mean = b)
3. Per-resample: compute Sharpe, hit_rate, max_drawdown, Sortino
4. Output: median metric + [2.5%, 97.5%] confidence interval (95% CI)
5. "Skill significance": reject Sharpe = 0 at 95% confidence if CI excludes 0
6. Comparison: also compute iid bootstrap CI to show the bias of ignoring dependence
7. Displayed in backtest report: "Sharpe: 0.62 [0.18, 1.08] -- significant at 95%"
8. Block size justified in output: "Block length: 6 days (autocorrelation-preserving)"

**Risk:** Block bootstrap with very small block size (b=1) degenerates to iid bootstrap.
With very large block size (b=n) degenerates to 1 sample. The n^(1/3) rule is robust.

**Tasks:**
- [ ] 6.7.1: Implement stationary block bootstrap resampler with geometric block length
- [ ] 6.7.2: Implement block size selection: b = ceil(n^(1/3)) with optional override
- [ ] 6.7.3: Compute metrics per bootstrap sample (vectorized for speed)
- [ ] 6.7.4: Compute confidence intervals (percentile method: 2.5th and 97.5th)
- [ ] 6.7.5: Implement skill significance test (CI lower bound > 0)
- [ ] 6.7.6: Add iid bootstrap comparison (shows how much dependence matters)
- [ ] 6.7.7: Add to backtest report output with block size documentation
- [ ] 6.7.8: Write test: long series with known Sharpe -> CI contains true value
- [ ] 6.7.9: Write test: block CI is wider than iid CI (demonstrates bias correction)

---

### Story 6.8: Transaction Cost Sensitivity Analysis

**As a** portfolio manager,
**I want** to see how profitability changes across different transaction cost assumptions,
**so that** I know the breakeven cost level and can evaluate execution quality requirements.

**Acceptance Criteria:**
1. Run backtest at cost levels: [0, 1, 3, 5, 10, 15, 20] bps
2. Report: Sharpe and PnL at each cost level
3. Breakeven cost: the cost level where Sharpe = 0
4. Chart: Sharpe as function of transaction costs (declining line)
5. Turnover analysis: average trades per day per asset

**Tasks:**
- [ ] 6.8.1: Parameterize backtest for multiple cost levels
- [ ] 6.8.2: Run grid of backtests across cost levels
- [ ] 6.8.3: Compute breakeven cost via interpolation
- [ ] 6.8.4: Generate cost sensitivity chart data
- [ ] 6.8.5: Add to terminal and frontend reports
- [ ] 6.8.6: Write test: zero cost always better than positive cost

---

### Story 6.9: Drawdown Analysis and Risk Budgeting

**As a** risk manager,
**I want** detailed drawdown analysis including drawdown duration, recovery time,
and risk budgets per asset,
**so that** I can set appropriate stop-loss levels and position limits.

**Acceptance Criteria:**
1. Drawdown curve: daily underwater equity graph
2. Top 5 drawdown events: start date, trough date, recovery date, magnitude, duration
3. Drawdown distribution: histogram of drawdown magnitudes
4. Risk budget: maximum contribution any single asset can make to drawdown
5. Frontend: drawdown chart with event annotations

**Tasks:**
- [ ] 6.9.1: Implement drawdown computation from equity curve
- [ ] 6.9.2: Identify top-N drawdown events with dates and magnitudes
- [ ] 6.9.3: Compute drawdown duration and recovery time
- [ ] 6.9.4: Implement per-asset risk contribution during drawdowns
- [ ] 6.9.5: Generate drawdown histogram data
- [ ] 6.9.6: Add to terminal report
- [ ] 6.9.7: Add `/api/diagnostics/drawdown-analysis` endpoint
- [ ] 6.9.8: Implement frontend drawdown chart with annotations

---

### Story 6.10: Automated Backtest Report with Executive Summary

**As a** decision maker,
**I want** a one-page executive summary of backtest results that highlights
key metrics, regime performance, and risk measures,
**so that** I can quickly assess system quality without reading detailed reports.

**Acceptance Criteria:**
1. Executive summary includes:
   - Overall Sharpe [with CI], Hit Rate, Max Drawdown, CAGR
   - Best/Worst asset, Best/Worst regime
   - Transaction cost breakeven
   - Skill significance (YES/NO)
   - Recommendation: DEPLOY / REVIEW / REJECT
2. PDF export with charts embedded
3. Terminal: single-screen Rich panel with key numbers
4. Frontend: summary card on DiagnosticsPage
5. Auto-generated after each walkforward run

**Tasks:**
- [ ] 6.10.1: Design executive summary data structure
- [ ] 6.10.2: Implement metric aggregation into summary
- [ ] 6.10.3: Generate recommendation logic (Sharpe threshold + significance)
- [ ] 6.10.4: Terminal rendering with Rich panel
- [ ] 6.10.5: PDF export with embedded charts
- [ ] 6.10.6: Frontend summary card component
- [ ] 6.10.7: Auto-trigger after walkforward completion

---

## EPIC 7: PERFORMANCE OPTIMIZATION AND SCALABILITY

**Epic Owner:** Platform Engineering Lead
**Priority:** P2 -- Medium
**Estimated Scope:** 6 stories, ~22 tasks

**Problem Statement:**
As the asset universe grows and models become more complex, performance becomes
a bottleneck. Current pain points: full tune is 15-25 minutes, signal generation
for 140+ assets takes 3-5 minutes, frontend renders large tables slowly.

**Success Criteria:**
- Full tune: < 5 minutes on 8-core machine
- Incremental tune: < 30 seconds for typical daily run
- Signal generation: < 60 seconds for full universe
- Frontend table render: < 100ms for 200 rows
- API response time: p95 < 200ms for all endpoints

---

### Story 7.1: Vectorized Filter Operations with NumPy

**As a** platform engineer,
**I want** Kalman filter operations vectorized using NumPy array operations
where possible (batch assets, batch horizons),
**so that** we avoid Python loop overhead for embarrassingly parallel computations.

**Acceptance Criteria:**
1. Multi-horizon forecast: vectorized phi^H computation for all H simultaneously
2. BMA weight computation: vectorized softmax across all models
3. Monte Carlo sampling: batched random draws instead of per-draw loop
4. Benchmark: 2x speedup for signal generation with 7 horizons

**Tasks:**
- [ ] 7.1.1: Vectorize phi^H computation: np.power(phi, np.array(horizons))
- [ ] 7.1.2: Vectorize BMA softmax: scipy.special.softmax over model BICs
- [ ] 7.1.3: Batch Monte Carlo draws: np.random.multivariate_normal
- [ ] 7.1.4: Profile before/after: signal generation time per asset
- [ ] 7.1.5: Write numerical precision test: vectorized matches loop results

---

### Story 7.2: Caching Layer for Repeated Computations

**As a** platform engineer,
**I want** expensive intermediate results (regime classification, volatility estimates,
feature computation) cached across the signal generation pipeline,
**so that** repeated calls do not re-compute identical results.

**Acceptance Criteria:**
1. LRU cache for regime classification: (returns_hash -> regime_label)
2. Volatility estimate cache: EWMA vol computed once per asset per run
3. Feature cache: momentum, skewness, etc. computed once and reused
4. Cache hit rate > 80% during typical signal generation run
5. Memory overhead < 100MB for 200-asset universe

**Tasks:**
- [ ] 7.2.1: Implement returns-hash computation (fast hash of last 30 values)
- [ ] 7.2.2: Add LRU cache decorator to regime classification
- [ ] 7.2.3: Cache EWMA volatility computation with mtime-based invalidation
- [ ] 7.2.4: Cache feature computation with similar invalidation
- [ ] 7.2.5: Add cache hit/miss counters for monitoring
- [ ] 7.2.6: Profile memory usage with full universe

---

### Story 7.3: Lazy Loading and Code Splitting for Frontend

**As a** frontend user,
**I want** pages and heavy components (charts, tables) to load on demand,
**so that** initial page load is fast and subsequent navigation feels instant.

**Acceptance Criteria:**
1. React.lazy() for all page components
2. Lightweight-charts loaded only on ChartsPage
3. Recharts loaded only on pages that use it
4. Initial bundle < 150KB gzipped (app shell only)
5. Per-page chunks < 100KB gzipped each
6. Loading states: skeleton screens during chunk load

**Tasks:**
- [ ] 7.3.1: Wrap all page components with React.lazy()
- [ ] 7.3.2: Add Suspense boundaries with skeleton fallbacks
- [ ] 7.3.3: Analyze current bundle composition with vite-bundle-visualizer
- [ ] 7.3.4: Implement dynamic imports for chart libraries
- [ ] 7.3.5: Verify chunk sizes meet budget
- [ ] 7.3.6: Test loading experience on simulated slow connection

---

### Story 7.4: Database Layer for Historical Data (Optional)

**As a** system architect,
**I want** the option to use SQLite for storing historical forecasts, backtest results,
and scorecard data instead of JSON files,
**so that** queries are faster and data management is more robust.

**Background:**
Current file-based storage works for operational data but becomes unwieldy for
historical analysis (querying 90 days of forecasts across 140 assets). SQLite
provides proper indexing and query capability without infrastructure overhead.

**Acceptance Criteria:**
1. SQLite database: `src/data/quant.db`
2. Tables: forecasts, backtest_results, scorecard_entries, errors
3. Indexes: (symbol, date), (date, horizon), (symbol, regime)
4. Migration from JSON to SQLite: one-time import script
5. Read performance: 10x faster than JSON file scanning for historical queries
6. Write performance: comparable to JSON (batched inserts)
7. Backward compatible: JSON files still generated for current consumers

**Tasks:**
- [ ] 7.4.1: Design SQLite schema with proper normalization
- [ ] 7.4.2: Implement database connection manager with WAL mode
- [ ] 7.4.3: Implement forecast table: insert, query by (symbol, date, horizon)
- [ ] 7.4.4: Implement backtest_results table
- [ ] 7.4.5: Implement JSON-to-SQLite migration script
- [ ] 7.4.6: Add SQLite reader to backend services (optional feature flag)
- [ ] 7.4.7: Benchmark: query performance JSON vs SQLite

---

### Story 7.5: Background Task Queue Enhancement

**As a** system operator,
**I want** long-running tasks (full tune, walkforward backtest) to run in background
with progress tracking and cancellation support,
**so that** the system remains responsive during heavy computation.

**Acceptance Criteria:**
1. Celery tasks enhanced with progress callbacks
2. Task status includes: progress_pct, current_step, eta, errors_so_far
3. Cancellation support: POST `/api/tasks/cancel/{taskId}`
4. Task history: last 50 tasks with results (success/failure/cancelled)
5. Frontend: task manager panel showing active and recent tasks

**Tasks:**
- [ ] 7.5.1: Enhance Celery task wrappers with progress callbacks
- [ ] 7.5.2: Implement task cancellation via Celery revoke
- [ ] 7.5.3: Add task history storage (in-memory ring buffer or SQLite)
- [ ] 7.5.4: Add `/api/tasks/cancel/{taskId}` endpoint
- [ ] 7.5.5: Add `/api/tasks/history` endpoint
- [ ] 7.5.6: Implement task manager panel in frontend

---

### Story 7.6: Continuous Integration Pipeline

**As a** developer,
**I want** a CI pipeline that runs on every commit: unit tests, display parity tests,
benchmark validation, and profitability gate,
**so that** quality is enforced automatically.

**Acceptance Criteria:**
1. Test stages:
   - Unit tests: `make tests` (< 2 minutes)
   - Display parity: test_display_parity.py
   - Benchmark: 12-symbol validation (< 5 minutes)
   - Profitability gate: walkforward on cached data (< 10 minutes)
2. Stage progression: each stage runs only if previous passes
3. Results summary: inline annotations on PR
4. Cache: benchmark data cached between CI runs
5. Total CI time < 15 minutes

**Tasks:**
- [ ] 7.6.1: Create CI configuration file (GitHub Actions or equivalent)
- [ ] 7.6.2: Configure test stages with proper ordering
- [ ] 7.6.3: Add benchmark data caching between runs
- [ ] 7.6.4: Implement results summary output
- [ ] 7.6.5: Add profitability gate as final stage
- [ ] 7.6.6: Optimize for < 15 minute total time

---

## EPIC 8: ADVANCED SIGNAL INTELLIGENCE AND ALPHA GENERATION

**Epic Owner:** Quant Research Lead
**Priority:** P1 -- High
**Estimated Scope:** 8 stories, ~30 tasks

**Problem Statement:**
The system produces probabilistically correct forecasts, but "correct" and "profitable"
are different. This epic adds features that increase alpha -- the excess return above
a benchmark. It focuses on: signal timing, event detection, cross-asset intelligence,
and conviction weighting.

**Success Criteria:**
- Portfolio alpha > 3% annualized above SPY
- Information ratio > 0.4
- Signal hit rate at optimal threshold > 60%
- Event-driven signals (earnings, FOMC) significantly outperform non-event signals
- Cross-asset signals add >= 1% annualized alpha above single-asset signals

---

### Story 8.1: Earnings Event Signal Augmentation

**As a** signal consumer,
**I want** the system to detect upcoming earnings dates and adjust forecast confidence
and magnitude around earnings events,
**so that** I can position appropriately for the highest-volatility events.

**Background:**
Earnings events cause 5-15% moves in small caps. The standard Kalman filter treats
these as regular observations. An earnings-aware signal would:
- Widen confidence intervals by 2-3x in the 3 days before earnings
- Increase process noise q temporarily (filter becomes more adaptive)
- Use historical earnings-day volatility as a prior for the current move
- Detect post-earnings drift and amplify momentum signals

**Dependencies:** Story 1.3 (GAS-Q -- provides time-varying q mechanism for pre-earnings q boost)

**Risk:** yfinance earnings calendar data is unreliable -- dates can be wrong, missing,
or change at the last minute. Mitigation: cross-reference with a secondary source
(manual calendar from earnings whispers, or cached verified dates). Never let a wrong
earnings date REMOVE a valid signal -- only let it WIDEN uncertainty.

**Acceptance Criteria:**
1. Earnings calendar loaded from cached data (yfinance provides next_earnings_date)
2. Pre-earnings signals (T-3 to T-0): confidence reduced, intervals widened
3. Post-earnings signals (T+1 to T+5): innovation-weighted drift amplified
4. Earnings volatility prior: historical abs move on earnings day stored per asset
5. Signal label includes "PRE-EARNINGS" or "POST-EARNINGS" context
6. Benchmark: post-earnings drift capture improves PnL by >= 2% annualized

**Tasks:**
- [ ] 8.1.1: Implement earnings date loading from yfinance/cached calendar
- [ ] 8.1.2: Add pre-earnings detection (T-3 to T-0) flag to features
- [ ] 8.1.3: Implement confidence reduction and interval widening for pre-earnings
- [ ] 8.1.4: Implement post-earnings amplification with innovation weighting
- [ ] 8.1.5: Compute historical earnings-day volatility per asset
- [ ] 8.1.6: Add earnings context label to signal output
- [ ] 8.1.7: Update frontend to display earnings context badge
- [ ] 8.1.8: Benchmark: PnL with/without earnings augmentation

---

### Story 8.2: FOMC and Macro Event Calendar Integration

**As a** signal consumer,
**I want** the system to anticipate FOMC meetings, CPI releases, and other macro events
and adjust forecasts for the pre/post event period,
**so that** systemic risk events are reflected in forecast uncertainty.

**Acceptance Criteria:**
1. Macro event calendar: FOMC dates, CPI/PPI releases, NFP dates
2. Pre-event (T-1): all equity forecasts widen intervals by 1.5x
3. Post-event (T+0): rapid regime reassessment (faster smoothing alpha)
4. Event type conditioning: FOMC affects bonds+equities, CPI affects all
5. Calendar stored in `src/data/macro_calendar.json` (manually maintained)
6. Benchmark: forecast calibration improves during event weeks

**Tasks:**
- [ ] 8.2.1: Create macro event calendar data structure and JSON file
- [ ] 8.2.2: Implement event proximity detection (T-N to T+N)
- [ ] 8.2.3: Apply uncertainty amplification for pre-event periods
- [ ] 8.2.4: Apply accelerated regime assessment for post-event
- [ ] 8.2.5: Add event type conditioning (which assets affected)
- [ ] 8.2.6: Update frontend with macro event indicators
- [ ] 8.2.7: Benchmark: event-week calibration improvement

---

### Story 8.3: Conviction-Weighted Signal Ranking System

**As a** signal consumer,
**I want** signals ranked not just by forecast magnitude but by a composite conviction
score that incorporates model agreement, regime confidence, historical accuracy,
and forecast stability,
**so that** I focus on the signals most likely to be profitable.

**Background:**
A +2% forecast from 5 agreeing models in a well-fit regime with 65% historical
accuracy is very different from a +2% forecast from 3 conflicting models in a
transitioning regime with 48% historical accuracy. Conviction scoring captures this.

**Acceptance Criteria:**
1. Conviction score formula:
   ```
   conviction = (
     0.30 * model_agreement +    # 0-1: fraction of models agreeing on sign
     0.25 * regime_confidence +   # 0-1: regime classification probability
     0.25 * historical_accuracy + # 0-1: rolling 60-day hit rate for this asset
     0.20 * forecast_stability    # 0-1: 1 - (std of last 5 forecasts / mean forecast)
   )
   ```
2. Conviction categories: HIGH (>0.7), MEDIUM (0.5-0.7), LOW (<0.5)
3. Signal table sortable by conviction (default sort: conviction descending)
4. Terminal display: conviction badge next to forecast
5. Frontend: conviction bar with gradient fill
6. Benchmark: HIGH conviction signals have > 65% hit rate

**Tasks:**
- [ ] 8.3.1: Implement conviction score computation with 4 factors
- [ ] 8.3.2: Add conviction to signal output structure
- [ ] 8.3.3: Implement conviction category classification
- [ ] 8.3.4: Add conviction sorting to terminal display
- [ ] 8.3.5: Add conviction bar component to frontend
- [ ] 8.3.6: Default sort: conviction descending
- [ ] 8.3.7: Benchmark: hit rate by conviction category
- [ ] 8.3.8: Write test: high agreement + high accuracy -> HIGH conviction

---

### Story 8.4: Pair Trading Signal Generation

**As a** portfolio manager,
**I want** the system to identify profitable pairs (e.g., AAPL-MSFT, SPY-QQQ)
and generate spread-based signals when pairs diverge from equilibrium,
**so that** I can capture mean-reversion alpha independent of market direction.

**Dependencies:** Story 2.2 (OU Calibration -- provides OU parameter estimation framework for spreads)

**Risk:** Cointegration is not stationary -- pairs that cointegrated historically can
decouple (structural break). Mitigation: re-test cointegration every 30 days and remove
pairs where p-value > 0.10. Also use Johansen test as confirmation (more robust than
Engle-Granger for multi-cointegration).

**Acceptance Criteria:**
1. Pair universe: top 20 pairs by historical cointegration score
2. Spread computation: log price ratio or OLS residual
3. Signal: when spread > 2 sigma from mean, signal convergence
4. Half-life estimated via OU model on spread (from Story 2.2 framework)
5. Position sizing: inversely proportional to spread volatility
6. Benchmark: pair strategy Sharpe > 0.3 (market neutral)

**Tasks:**
- [ ] 8.4.1: Implement cointegration testing (Engle-Granger) for pair screening
- [ ] 8.4.2: Select top 20 pairs by cointegration strength
- [ ] 8.4.3: Compute log price ratio spreads
- [ ] 8.4.4: Estimate OU half-life on spread
- [ ] 8.4.5: Generate convergence signals when spread > threshold
- [ ] 8.4.6: Add pair signals to signal output
- [ ] 8.4.7: Update frontend with pairs tab on SignalsPage
- [ ] 8.4.8: Benchmark: pair strategy returns and Sharpe

---

### Story 8.5: Sector Rotation Signal Generation

**As a** portfolio manager,
**I want** sector-level signals that identify which sectors are strengthening or
weakening relative to the market,
**so that** I can rotate capital toward the strongest sectors.

**Acceptance Criteria:**
1. Sector strength: relative momentum of sector ETF vs SPY (21d, 63d, 126d)
2. Sector breadth: fraction of sector constituents with positive signals
3. Sector signal: composite of strength + breadth, normalized to [-1, +1]
4. Rotation recommendation: "Overweight Technology, Underweight Energy"
5. Display: sector heat map with strength and breadth dimensions
6. Benchmark: sector rotation adds >= 1% annualized alpha

**Tasks:**
- [ ] 8.5.1: Compute relative sector momentum vs SPY
- [ ] 8.5.2: Compute sector breadth (positive signal fraction)
- [ ] 8.5.3: Combine into composite sector signal
- [ ] 8.5.4: Generate rotation recommendation
- [ ] 8.5.5: Add sector signals to `/api/signals/sectors` endpoint
- [ ] 8.5.6: Implement sector heat map component in frontend
- [ ] 8.5.7: Benchmark: sector rotation alpha analysis

---

### Story 8.6: Volatility Surface Signal Integration

**As a** options trader,
**I want** implied volatility term structure and skew analyzed as supplementary
signals for directional trading,
**so that** options market intelligence enhances spot forecasts.

**Background:**
Options-implied signals carry forward-looking information:
- Steep put skew = market expects crash = bearish signal
- Inverted term structure = near-term fear = high uncertainty signal
- Low IV rank = cheap protection = low risk environment

**Acceptance Criteria:**
1. IV data: from existing options data pipeline (src/options/)
2. Put/call skew ratio: 25-delta put IV / 25-delta call IV
3. Term structure slope: 1-month IV / 3-month IV
4. IV rank: current IV percentile over 252 days
5. Signals adjusted: high skew -> widen downside intervals
6. Display: IV context in asset detail modal

**Tasks:**
- [ ] 8.6.1: Load IV data from options pipeline
- [ ] 8.6.2: Compute skew ratio and term structure slope
- [ ] 8.6.3: Compute IV rank (percentile)
- [ ] 8.6.4: Integrate IV signals into forecast uncertainty adjustment
- [ ] 8.6.5: Add IV context to asset detail API response
- [ ] 8.6.6: Display IV indicators in asset detail modal
- [ ] 8.6.7: Benchmark: IV-augmented forecasts vs baseline

---

### Story 8.7: Dynamic Position Sizing Based on Kelly Criterion

**As a** portfolio manager,
**I want** the system to recommend position sizes for each signal based on Kelly
criterion adjusted for forecast uncertainty,
**so that** bet sizes are mathematically optimal given our edge and risk tolerance.

**Background:**
Full Kelly: f* = (p*b - q) / b, where p = win probability, b = win/loss ratio,
q = 1-p. Half-Kelly (f*/2) is standard for robustness. With our forecast confidence
intervals, we can compute p and b per asset per horizon.

CRITICAL: Kelly assumes edge estimates are accurate. If our p=0.60 estimate is really
p=0.52, full Kelly dramatically over-bets and can cause ruin. This is why Half-Kelly is
the minimum requirement, and fractional Kelly (f*/3 or f*/4) should be available as
a conservative option.

**Dependencies:** Story 2.5 (Confidence Intervals -- p and b derived from quantiles)
               Story 1.9 (Scorecard -- historical accuracy validates p estimates)

**Acceptance Criteria:**
1. Kelly fraction computed per asset per horizon from forecast quantiles
2. Half-Kelly used as default (full Kelly too aggressive)
3. Position cap: max 10% per asset (risk limit)
4. Portfolio-level: Kelly fractions sum-adjusted to max 100% total exposure
5. Display: recommended position size next to forecast
6. Benchmark: Kelly-sized portfolio Sharpe > uniform-sized portfolio Sharpe

**Tasks:**
- [ ] 8.7.1: Implement Kelly fraction from forecast quantiles (p_up, E[win], E[loss])
- [ ] 8.7.2: Apply half-Kelly scaling
- [ ] 8.7.3: Apply per-asset position cap (10%)
- [ ] 8.7.4: Implement portfolio-level exposure normalization
- [ ] 8.7.5: Add position size recommendation to signal output
- [ ] 8.7.6: Display in terminal and frontend
- [ ] 8.7.7: Benchmark: Kelly vs uniform sizing Sharpe comparison

---

### Story 8.8: Signal Decay and Time-to-Live Tracking

**As a** signal consumer,
**I want** each signal to have a "time to live" based on forecast horizon and
historical decay rate,
**so that** I know when a signal is still valid and when it has expired.

**Acceptance Criteria:**
1. Signal TTL: computed from forecast horizon (7-day forecast valid for ~3 days)
2. Decay tracking: signal strength diminishes as time passes
3. Expired signals clearly marked in display (grayed out, strikethrough)
4. Auto-refresh recommendation: "Signal expires in 2h, refresh recommended"
5. Historical decay measurement: how quickly do forecasts lose predictive power?
6. Display: countdown timer or progress bar for signal freshness

**Tasks:**
- [ ] 8.8.1: Compute signal TTL based on horizon and generation time
- [ ] 8.8.2: Implement signal decay model (exponential with half-life)
- [ ] 8.8.3: Add TTL and decay to signal output structure
- [ ] 8.8.4: Implement expired signal display (terminal + frontend)
- [ ] 8.8.5: Add refresh recommendation logic
- [ ] 8.8.6: Historical analysis: forecast predictive power decay rate
- [ ] 8.8.7: Write test: 3-day-old 7-day forecast has reduced strength

---

## EPIC 9: IMPLEMENTATION PRIORITY AND SEQUENCING

**Epic Owner:** Program Lead
**Priority:** N/A -- Meta-epic

**Purpose:** Define execution order considering dependencies and impact.

### Phase 1: Foundation (Weeks 1-3) -- Highest Impact, Lowest Risk

**Stories in priority order:**
1. Story 1.8:  Display Precision and Semantic Formatting (quick win, visible impact)
2. Story 1.1:  Adaptive Process Noise Floor (core forecast improvement)
3. Story 1.3:  GAS-Q Integration (use what we already trained)
4. Story 1.7:  Market Temperature Uses Tuned Params (connect tune to forecast)
5. Story 1.9:  Directional Accuracy Scorecard (measurement before optimization)
6. Story 4.1:  Unified Signal Output Contract (backend/frontend sync foundation)

**Validation gate after Phase 1:**
- Run Story 6.1 walkforward on benchmark universe (Story 6.1 is ACCELERATED to Phase 1
  specifically for this gating purpose -- a minimal version runs before Phase 2 starts)
- Establish baseline: Sharpe, hit_rate, max_drawdown
- Store as `profitability_baseline.json`
- NOTE: This requires a MINIMAL Story 6.1 (walk-forward engine with daily step) to be
  implemented as part of Phase 1. Full simulator (6.2) and Monte Carlo (6.7) wait for Phase 5.

### Phase 2: Ensemble Upgrade (Weeks 4-6) -- Model Quality

**Stories in priority order:**
7. Story 1.6:  Ensemble De-correlation and Signal Extraction
8. Story 2.1:  GARCH Parameter Loading
9. Story 2.2:  OU Calibration with Asset-Specific Parameters
10. Story 2.4: Bayesian Model Combination
11. Story 2.5: Forecast Confidence Intervals
12. Story 1.4: Multi-Scale Drift Estimation

**Validation gate after Phase 2:**
- Sharpe improvement >= 0.1 over Phase 1 baseline
- Hit rate improvement >= 3%
- All forecasts include confidence intervals

### Phase 3: Infrastructure (Weeks 5-7, parallel with Phase 2)

**Stories in priority order:**
13. Story 3.1: Incremental Tuning
14. Story 3.2: Parallel Tuning Optimization
15. Story 3.5: Tuning Validation Gate
16. Story 4.3: WebSocket Push
17. Story 4.4: Pipeline Orchestration
18. Story 4.8: Unified Error Reporting

### Phase 4: UX Excellence (Weeks 7-10)

**Stories in priority order:**
19. Story 5.1: Signal Table Redesign (heat map + sparklines)
20. Story 5.2: Forecast Fan Chart
21. Story 5.3: Regime Context Bar
22. Story 5.4: Asset Detail Modal
23. Story 5.5: Micro-Animations
24. Story 5.6: Command Palette

### Phase 5: Profitability Proof (Weeks 8-10, parallel with Phase 4)

**Stories in priority order:**
25. Story 6.1: Walk-Forward Backtest (FULL version -- minimal version in Phase 1)
26. Story 6.2: Trading Strategy Simulator
27. Story 6.4: Profitability Gate
28. Story 6.7: Monte Carlo Confidence

### Phase 6: Alpha Generation (Weeks 10-14)

**Stories in priority order:**
29. Story 8.1: Earnings Event Augmentation
30. Story 8.3: Conviction-Weighted Ranking
31. Story 8.5: Sector Rotation Signals
32. Story 8.7: Kelly Position Sizing
33. Story 1.10: Momentum State Equation Integration
34. Story 1.12: Regime Transition Speed

### Phase 7: Polish and Scale (Weeks 14-16)

**Stories in priority order:**
35. Story 5.8: Mobile-First Layout
36. Story 5.10: Export and Sharing
37. Story 7.1: Vectorized Filter Operations
38. Story 7.3: Lazy Loading Frontend
39. Story 7.6: CI Pipeline
40. Story 5.14: Design System Documentation

---

## DEPENDENCY MAP

```
Story 1.8 (Display) -----> Story 4.1 (Contract) -----> Story 4.6 (Parity Test)
                                  |
Story 1.1 (q Floor) -----> Story 1.3 (GAS-Q) -----> Story 1.2 (Gain Monitor)
                                  |
Story 1.9 (Scorecard) -----> Story 1.13 (Feedback) -----> Story 6.4 (Gate)
                                  |
Story 2.1 (GARCH) --+            |
Story 2.2 (OU) -----+---> Story 1.7 (Use Tuned) ---> Story 2.4 (BMC)
Story 3.3 (Fit GARCH)-+
Story 3.4 (Fit OU) ---+
                                  |
Story 2.5 (Intervals) -----> Story 5.2 (Fan Chart)
                                  |
Story 1.6 (De-corr) ------> Story 2.9 (Explain) -----> Story 5.9 (Tooltips)
                                  |
Story 4.3 (WebSocket) -----> Story 4.4 (Pipeline) -----> Story 4.9 (Polling)
                                  |
Story 6.1 (Walkforward) ---> Story 6.2 (Simulator) ---> Story 6.10 (Report)
                                  |
Story 8.3 (Conviction) -----> Story 8.7 (Kelly) -----> Story 5.7 (Portfolio)
```

---

## SUCCESS METRICS -- PROGRAM LEVEL

| Metric | Current | Phase 1 Target | Phase 3 Target | Final Target |
|--------|---------|----------------|----------------|--------------|
| 7-Day Hit Rate | ~50.5% | 55% | 58% | 62% |
| Annualized Sharpe (L/O) | ~0.1 | 0.3 | 0.5 | 0.7 |
| Annualized Alpha vs SPY | ~0% | 1.5% | 3% | 5% |
| Max Drawdown | unmeasured | < 25% | < 20% | < 15% |
| Info Coefficient (7d) | ~0.02 | 0.08 | 0.12 | 0.18 |
| Full Tune Time | 15-25 min | 15 min | 5 min | 3 min |
| Signal Gen Time | 3-5 min | 3 min | 1 min | 30s |
| API p95 Latency | ~500ms | 300ms | 200ms | 100ms |
| Frontend LCP | ~3s | 2.5s | 1.5s | 1.0s |
| Display +0.0% Rate | ~60% | 10% | 2% | 0% |

---

## RISK REGISTER

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| Overfitting to benchmark universe | HIGH | MEDIUM | Walk-forward testing, out-of-sample validation, cross-validation |
| q floor degrades calibration | MEDIUM | LOW | BIC adjustment penalty, PIT monitoring, automatic rollback |
| Ensemble changes reduce Sharpe | HIGH | MEDIUM | Profitability gate, A/B comparison, gradual rollout |
| Frontend performance degrades with features | MEDIUM | MEDIUM | Performance budgets, Lighthouse CI, lazy loading |
| GAS-Q parameters stale after market regime change | MEDIUM | LOW | Incremental tuning with regime change detection |
| WebSocket connection instability | LOW | MEDIUM | Exponential backoff, graceful degradation to polling |
| Data quality issues (Yahoo Finance gaps) | MEDIUM | MEDIUM | Data validation checks, duplicate detection, fill logic |
| Over-engineering: too many features, too few profitable | HIGH | MEDIUM | Phase gates, profitability validation before each new phase |

---

## GLOSSARY

| Term | Definition |
|------|-----------|
| **BMA** | Bayesian Model Averaging -- weights model outputs by posterior probability |
| **BIC** | Bayesian Information Criterion -- model selection criterion penalizing complexity |
| **CRPS** | Continuous Ranked Probability Score -- measures calibration + sharpness |
| **ECE** | Expected Calibration Error -- gap between predicted and actual probabilities |
| **GAS-Q** | Generalized Autoregressive Score - Q -- score-driven time-varying process noise |
| **IC** | Information Coefficient -- rank correlation between forecast and realized return |
| **Kelly** | Kelly Criterion -- optimal bet sizing given edge and odds |
| **LFO-CV** | Leave-Future-Out Cross-Validation -- time-series model selection |
| **MLE** | Maximum Likelihood Estimation -- parameter fitting method |
| **OU** | Ornstein-Uhlenbeck -- mean-reverting stochastic process |
| **PIT** | Probability Integral Transform -- tests if model is calibrated |
| **Sharpe** | Risk-adjusted return: (return - risk_free) / volatility |
| **Sortino** | Like Sharpe but only penalizes downside volatility |
| **TTL** | Time To Live -- validity period for a signal |
| **Walk-forward** | Out-of-sample testing using rolling training windows |

---

## APPENDIX A: BENCHMARK UNIVERSE CHARACTERISTICS

| Symbol | Sector | Cap | Avg Daily Vol | Beta | Notes |
|--------|--------|-----|---------------|------|-------|
| UPST | Fintech | Small | $80M | 2.1 | High vol, earnings-driven |
| AFRM | Fintech | Small | $120M | 1.9 | Credit-sensitive |
| IONQ | Quantum | Small | $60M | 2.5 | Speculative, momentum-driven |
| CRWD | Cybersec | Mid | $200M | 1.4 | Growth, high PE |
| DKNG | Gaming | Mid | $150M | 1.6 | Event-driven (sports calendar) |
| SNAP | Social | Mid | $180M | 1.7 | Advertising cycle |
| AAPL | Tech | Large | $5B | 1.1 | Bellwether, liquid |
| NVDA | Semis | Large | $8B | 1.8 | AI cycle, high momentum |
| TSLA | Auto | Large | $6B | 1.5 | Meme energy, volatile |
| SPY | Index | Index | $30B | 1.0 | The benchmark itself |
| QQQ | Index | Index | $15B | 1.1 | Tech-heavy index |
| IWM | Index | Index | $4B | 1.2 | Small-cap index, breadth signal |

This universe covers: 3 cap sizes x 4 sectors + 3 indices = maximum diversity for
12-symbol validation. If the system is profitable here, it generalizes well.

---

## APPENDIX B: KEY FILE MODIFICATION MAP

| Story | Files Modified | Files Created |
|-------|---------------|---------------|
| 1.1 | tune.py, signals.py | - |
| 1.2 | signals.py | - |
| 1.3 | signals.py | - |
| 1.4 | signals.py, market_temperature.py | - |
| 1.6 | market_temperature.py | - |
| 1.7 | market_temperature.py | - |
| 1.8 | signals_ux.py, frontend SignalsPage.tsx | signal_formatting.py |
| 1.9 | - | forecast_scorecard.py |
| 2.1 | tune.py, market_temperature.py | - |
| 2.4 | market_temperature.py | - |
| 2.5 | signals.py, signals_ux.py, signal_service.py | - |
| 3.1 | tune_ux.py, Makefile | - |
| 3.3 | tune.py | - |
| 3.5 | tune.py | - |
| 4.1 | signals.py, signals_ux.py, signal_service.py, models.py | signal_output.py |
| 4.3 | ws.py, Layout.tsx | - |
| 5.1 | SignalsPage.tsx | HeatMapCell.tsx, Sparkline.tsx |
| 5.2 | ChartsPage.tsx | FanChart.tsx |
| 5.3 | Layout.tsx | ContextBar.tsx |
| 5.4 | SignalsPage.tsx | AssetDetailModal.tsx |
| 6.1 | - | walkforward_backtest.py |
| 6.4 | Makefile | profitability_gate.py |

---

# END OF QUANT.md
# Total: 8 Epics, 80 Stories, ~295 Tasks
# Estimated Program Duration: 14-16 weeks with 2-person team
