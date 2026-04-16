# Stocks.md -- Elite Quant Model Accuracy Improvement Program

**Author**: Professor of Quantitative Finance & Stochastic Filtering
**Date**: April 2026
**Scope**: 7 core model files, 25 test assets across all capitalizations
**Philosophy**: Every model is a hypothesis. Accuracy comes from honest scoring, not parameter inflation.

---

## Test Universe

All epics MUST be validated against the following representative universe:

| Category | Symbols | Rationale |
|----------|---------|-----------|
| Large Cap Equity | MSFT, GOOGL, NFLX, AAPL, NVDA, AMZN, META | Liquid, well-behaved, sector-diverse Gaussian baseline |
| Mid Cap Equity | CRWD, DKNG, PLTR, SQ, RKLB | Growth-stage vol clustering, regime shifts, sector spread |
| Small Cap Equity | UPST, AFRM, IONQ, SOFI | Thin liquidity, extreme kurtosis, gap risk, earnings vol |
| High Vol / Thematic | MSTR, TSLA | BTC-correlated leverage (MSTR), meme/momentum (TSLA) |
| Broad Index | SPY, QQQ, IWM | Mean-reversion anchor, low idiosyncratic risk |
| Precious Metals | GC=F (Gold), SI=F (Silver) | Macro-driven, slow regimes, jump processes |
| Cryptocurrency | BTC-USD, ETH-USD | Non-Gaussian, 24/7 trading, structural breaks |

**Minimum Sample**: 2 years daily returns (504+ observations).
**Crisis Periods**: Must include COVID crash (Mar 2020), SVB crisis (Mar 2023), Oct 2023 rates shock.

---

## Scoring Protocol

Every improvement must report BEFORE and AFTER on these proper scoring rules:

| Metric | Formula | Elite Target | Failure Threshold |
|--------|---------|--------------|-------------------|
| BIC | $-2\ell + k\log(n)$ | < -30,000 | > -25,000 |
| CRPS | $\text{CRPS}(F, y) = \mathbb{E}[\|X - y\|] - 0.5\mathbb{E}[\|X - X'\|]$ | < 0.018 | > 0.025 |
| PIT KS | $\sup_x \|F_n(x) - x\|$ | p > 0.20 | p < 0.05 |
| Hyvarinen | $H = 0.5 s^2 - 1/\sigma^2$ | < 500 | > 1000 |
| CSS | Calibration Stability Under Stress | > 0.70 | < 0.60 |
| FEC | Forecast Entropy Consistency | > 0.80 | < 0.70 |

---

# PART I: NUMBA COMPUTATIONAL KERNEL ACCURACY

## Epic 1: Gamma Function Precision Enhancement in Numba Kernels

**File**: `src/models/numba_kernels.py`
**Priority**: CRITICAL -- all Student-t likelihoods depend on gammaln accuracy

### Background

The current Stirling approximation for $\log\Gamma(x)$ achieves ~1e-6 relative error for $x > 2$,
but the Student-t log-PDF requires evaluation at $\nu/2$ and $(\nu+1)/2$. When $\nu = 3$:

$$\log\Gamma(1.5) = \log(\sqrt{\pi}/2) \approx -0.1208$$
$$\log\Gamma(2.0) = 0.0$$

At $\nu/2 = 1.5$, the Stirling series has non-trivial error because the correction terms
$\frac{1}{12x} - \frac{1}{360x^3}$ are large relative to the true value. This error propagates
directly into BIC rankings, where a 0.5 nats difference can swap model selection.

The Lanczos approximation (g=7, n=9 coefficients) achieves ~1e-12 relative error for all $x > 0$
but costs ~3x more FLOPs. Since gammaln is evaluated O(1) per timestep (precomputed outside the
loop), the additional cost is negligible relative to the Kalman update.

### Story 1.1: Replace Stirling with Lanczos for Low-nu Regimes

**As a** quantitative researcher tuning Student-t models on heavy-tailed assets (MSTR, BTC-USD),
**I need** gammaln accuracy better than 1e-10 for $\nu \in [2.1, 6]$,
**So that** BIC model rankings between $\nu=3$ and $\nu=4$ are not corrupted by numerical error.

**Acceptance Criteria**:
- [ ] `_lanczos_gammaln(x)` matches `scipy.special.gammaln(x)` to within 1e-12 relative error
      for $x \in \{1.5, 2.0, 2.5, 3.0, 4.0, 10.0, 50.0\}$
- [ ] BIC scores for $\nu=3$ vs $\nu=4$ on MSTR daily returns change by < 0.1 nats
      after switching from Stirling to Lanczos
- [ ] No regression in filter throughput (< 5% slowdown on 1000-step series)
- [ ] Validated on: MSTR, BTC-USD, ETH-USD, SI=F (silver), SPY, CRWD

**Tasks**:
1. Implement Lanczos g=7 with Spouge coefficients in `numba_kernels.py`
2. Add unit test comparing against scipy for $\nu \in [2.1, 50]$ grid
3. Run BIC comparison on MSTR (2y daily): record $\Delta$BIC for each $\nu$ candidate
4. Benchmark filter_kernel throughput: measure ns/timestep before and after
5. Validate PIT uniformity unchanged on SPY, GOOGL, GC=F

### Story 1.2: Incomplete Beta Function Accuracy at Extreme Quantiles

**As a** risk manager computing tail probabilities for VaR/CVaR,
**I need** Student-t CDF accuracy better than 1e-8 at $|z| > 5$,
**So that** PIT values near 0 and 1 are not clipped by numerical noise.

**Acceptance Criteria**:
- [ ] `_betainc(a, b, x)` matches `scipy.special.betainc` to within 1e-8
      for $a = \nu/2$, $b = 0.5$, $x \in [0.001, 0.01, 0.99, 0.999]$
- [ ] Student-t CDF at $z = \pm 6$ (standardized) matches scipy to 1e-8
- [ ] PIT histogram for BTC-USD shows no accumulation at 0.0 or 1.0 bins
- [ ] Validated on: BTC-USD, ETH-USD, MSTR, NFLX, AFRM (earnings gaps)

**Tasks**:
1. Increase continued fraction iterations from 200 to 300 in `_betacf`
2. Add Lentz's modified algorithm for better convergence at extreme $x$
3. Test CDF inversion consistency: $F^{-1}(F(z)) = z \pm 1e{-6}$ for $|z| \leq 8$
4. Run PIT analysis on BTC-USD 3-year daily: verify no bin < 0.01 or > 0.99 anomalies
5. Benchmark: measure CDF evaluation cost per 1000 calls

### Story 1.3: Student-t Quantile Function (CDF Inverse) Implementation

**As a** Monte Carlo simulation engine generating Student-t samples,
**I need** a numerically stable quantile function $F^{-1}(p; \nu)$,
**So that** posterior predictive sampling does not require scipy at inference time.

**Acceptance Criteria**:
- [ ] `student_t_ppf_scalar(p, nu)` matches `scipy.stats.t.ppf(p, nu)` to 1e-6
      for $p \in \{0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999\}$
- [ ] Works for $\nu \in [2.1, 50]$ without NaN or Inf
- [ ] Can be used inside Numba `@njit` loops
- [ ] Validated on: SPY, MSFT, AMZN, GC=F (gold)

**Tasks**:
1. Implement Newton-Raphson inversion: $p_{n+1} = p_n - (F(p_n) - target) / f(p_n)$
2. Use rational approximation for initial guess (Abramowitz & Stegun 26.7.5)
3. Add convergence check: $|F(x_n) - p| < 10^{-10}$ within 15 iterations
4. Unit test against scipy ppf on $\nu \times p$ grid (40 combinations)
5. Integrate into `batch_monte_carlo_sample()` in vectorized_ops.py

---

## Epic 2: Vectorized Operations Numerical Stability

**File**: `src/models/vectorized_ops.py`
**Priority**: HIGH -- multi-horizon forecasts used in all signal generation

### Background

The multi-horizon forecast variance formula:

$$\text{Var}(y_{t+H}) = \phi^{2H} P_t + q \cdot \frac{1 - \phi^{2H}}{1 - \phi^2} + R$$

has three numerical hazards:
1. When $|\phi| \to 1$: denominator $1 - \phi^2 \to 0$, causing division overflow
2. When $H$ is large and $|\phi| < 1$: $\phi^{2H} \to 0$, losing precision
3. When $|\phi| = 1$ exactly: random walk variance $= q \cdot H$ (special case)

Current implementation handles case 3 but not cases 1-2 with full precision.

### Story 2.1: Numerically Stable Geometric Series for Near-Unit-Root Phi

**As a** signal generator producing 365-day forecasts for metals (GC=F),
**I need** forecast variance that degrades gracefully when $\phi \in [0.98, 1.00]$,
**So that** long-horizon confidence intervals are neither collapsed nor exploded.

**Acceptance Criteria**:
- [ ] For $\phi = 0.999$, $H = 365$, $q = 10^{-6}$: variance matches
      double-precision reference to within 1e-10
- [ ] For $\phi = 1.0 - 10^{-8}$: no NaN, no Inf, smooth transition to random walk
- [ ] Forecast intervals for GC=F at 12M horizon are within $\pm 25\%$ of realized
- [ ] Validated on: GC=F, SI=F, SPY, BTC-USD, ETH-USD, CRWD

**Tasks**:
1. Implement Taylor expansion for $|\phi^2 - 1| < 10^{-6}$:
   $\frac{1 - \phi^{2H}}{1 - \phi^2} \approx H + \frac{H(H-1)}{2}(1 - \phi^2) + ...$
2. Add unit test: sweep $\phi \in [0.990, 0.991, ..., 1.000]$ and verify monotonicity
3. Compare against mpmath (arbitrary precision) reference values
4. Run 12M forecast intervals on GC=F and measure coverage probability
5. Stress test: $H = 1000$, $\phi = 0.9999$ -- should be finite and positive

### Story 2.2: Log-Sum-Exp Stability in BMA Weight Computation

**As a** model averaging engine combining 14+ models per regime,
**I need** BMA weights that sum to exactly 1.0 without underflow,
**So that** no model is silently dropped due to numerical precision loss.

**Acceptance Criteria**:
- [ ] $\sum_i w_i = 1.0 \pm 10^{-14}$ for all test assets
- [ ] When best BIC differs from worst by > 500 nats: worst model gets $w > 10^{-100}$ (not 0.0)
- [ ] No model receives exactly $w = 0.0$ (all models maintain non-zero weight)
- [ ] Validated on: MSTR (large BIC spread), UPST (small cap), SPY (small BIC spread)

**Tasks**:
1. Verify current log-sum-exp uses max-subtraction: $\log\sum e^{x_i} = m + \log\sum e^{x_i - m}$
2. Add assertion: `assert np.all(weights > 0)` after BMA computation
3. Add test: 20 models with BIC spread of 1000 nats -- verify no zero weights
4. Profile: measure BMA weight computation time for 14 models x 5 regimes
5. Document numerical guarantees in vectorized_ops.py header comment

### Story 2.3: Batch Monte Carlo with Antithetic Variates

**As a** signal engine generating posterior predictive samples,
**I need** variance-reduced Monte Carlo sampling,
**So that** signal stability improves (fewer samples needed for same precision).

**Acceptance Criteria**:
- [ ] Antithetic sampling reduces MC standard error by > 30% at fixed sample count
- [ ] Signal direction agreement (sign of mean) between runs > 99% for n=1000
- [ ] No bias introduced: mean of antithetic pairs matches standard MC mean
- [ ] Validated on: RKLB (high vol), UPST (small cap), MSFT (low vol), BTC-USD, ETH-USD (fat tails)

**Tasks**:
1. For each sample $z_i$, also use $-z_i$ (antithetic pair)
2. Modify `batch_monte_carlo_sample()` to accept `antithetic=True` flag
3. Test: generate 10,000 samples with and without -- compare mean/std estimates
4. Measure signal stability: run 100 signal generations, count direction flips
5. Benchmark: antithetic with n=500 vs standard with n=1000

---

## Epic 3: GARCH Variance Kernel Improvements

**File**: `src/models/numba_kernels.py` (garch_variance_kernel)
**Priority**: HIGH -- variance dynamics directly affect observation noise R_t

### Background

The GJR-GARCH(1,1) specification:

$$h_t = \omega + \alpha \epsilon^2_{t-1} + \gamma_{lev} \epsilon^2_{t-1} I(\epsilon_{t-1} < 0) + \beta h_{t-1}$$

has known limitations:
1. **Persistence**: $\alpha + \beta + 0.5\gamma_{lev}$ must be < 1 for stationarity, but
   near-unit persistence is common in equity markets (typically 0.95-0.99)
2. **Initialization**: $h_0 = \omega / (1 - \alpha - \beta - 0.5\gamma_{lev})$ is the
   unconditional variance, but this diverges for near-integrated processes
3. **Short samples**: With N < 200, the 5-parameter model overfits severely
4. **Metals/Crypto**: Structural breaks in volatility violate stationarity assumption

### Story 3.1: GARCH Stationarity Enforcement with Soft Constraints

**As a** parameter optimizer fitting GARCH on MSTR (high vol equity),
**I need** persistence constraints that prevent $\alpha + \beta + 0.5\gamma > 1$,
**So that** variance forecasts do not explode to infinity at long horizons.

**Acceptance Criteria**:
- [ ] Persistence $p = \alpha + \beta + 0.5\gamma_{lev} \in [0.80, 0.999]$ for all test assets
- [ ] Log-penalty: $-\lambda \log(1 - p)$ with $\lambda = 5.0$ added to negative LL
- [ ] MSTR: 30-day variance forecast finite and within 3x of realized
- [ ] Silver (SI=F): no GARCH explosion during March 2020 vol spike
- [ ] Validated on: MSTR, SI=F, BTC-USD, NFLX, SPY, DKNG, SOFI

**Tasks**:
1. Add log-barrier penalty to GARCH objective: $-5 \log(1 - (\alpha + \beta + 0.5\gamma))$
2. Implement warm-start: initialize from previous day's GARCH fit
3. Add persistence diagnostic to tuning output (print $p$ per asset)
4. Test on MSTR 3-year: verify no $h_t > 100 \times$ unconditional variance
5. Compare BIC before/after constraint: should improve for N < 300

### Story 3.2: Regime-Aware GARCH Initialization

**As a** Kalman filter using GARCH-enhanced observation noise,
**I need** $h_0$ initialization that reflects current market regime,
**So that** the filter does not spend 20+ days converging from a wrong initial state.

**Acceptance Criteria**:
- [ ] $h_0$ uses trailing 20-day realized variance (not unconditional)
- [ ] Filter convergence to steady state within 10 timesteps (measured by $|h_t - h_{t-1}|/h_t$)
- [ ] PIT uniformity at $t \in [1, 50]$ (early window) passes KS at p > 0.10
- [ ] Validated on: GC=F (regime changes), RKLB (IPO-era short history), IONQ (short history), IWM

**Tasks**:
1. Compute $h_0 = \text{median}(\epsilon^2_{1:20})$ as robust initializer
2. Add "burn-in" diagnostic: measure timesteps until $|h_t / h_{t-1} - 1| < 0.05$
3. Compare PIT for first 50 observations: old vs new initialization
4. Test on RKLB (short history post-IPO): verify GARCH does not diverge
5. Add fallback: if realized var < $10^{-10}$, use unconditional estimate

### Story 3.3: CRPS-Optimal GARCH Residual Scaling

**As a** calibration system producing probabilistic forecasts,
**I need** GARCH residuals scaled so that CRPS is minimized (not just likelihood),
**So that** the forecast distribution is sharp AND calibrated simultaneously.

**Acceptance Criteria**:
- [ ] CRPS improves by > 5% on average across test universe after GARCH residual rescaling
- [ ] PIT uniformity not degraded (KS p-value remains > 0.10)
- [ ] Scaling factor $\alpha_{GARCH} \in [0.85, 1.15]$ for all test assets
- [ ] Gold (GC=F) and Silver (SI=F): CRPS < 0.020
- [ ] Validated on: GC=F, SI=F, MSFT, GOOGL, META, BTC-USD, ETH-USD, MSTR

**Tasks**:
1. After GARCH fit, compute optimal scaling: $\hat{\alpha} = \arg\min_\alpha \text{CRPS}(\alpha h_t)$
2. Grid search $\alpha \in [0.80, 0.85, ..., 1.20]$ on training set
3. Validate on hold-out (last 20% of sample) to prevent overfitting
4. Add CRPS-GARCH diagnostic to tuning output
5. Compare against raw GARCH: $\Delta$CRPS per asset

---

# PART II: PHI-GAUSSIAN FILTER ACCURACY

## Epic 4: Phi-Gaussian Kalman Filter Precision

**File**: `src/models/phi_gaussian.py`
**Priority**: HIGH -- baseline model for BMA; if Gaussian is wrong, everything built on it is wrong

### Background

The phi-Gaussian Kalman filter is the computational backbone:

$$\mu_{t|t-1} = \phi \mu_{t-1|t-1}$$
$$P_{t|t-1} = \phi^2 P_{t-1|t-1} + q$$
$$S_t = P_{t|t-1} + R_t$$
$$K_t = P_{t|t-1} / S_t$$
$$\mu_{t|t} = \mu_{t|t-1} + K_t (r_t - \mu_{t|t-1})$$
$$P_{t|t} = (1 - K_t) P_{t|t-1}$$

Known precision issues:
- **Joseph form**: The update $P_{t|t} = (1 - K_t) P_{t|t-1}$ can lose positive-definiteness
  when $K_t \approx 1$. Joseph's numerically stable form is:
  $P_{t|t} = (1 - K_t) P_{t|t-1} (1 - K_t) + K_t^2 R_t$
- **Innovation variance floor**: $S_t$ must be > 0 always, but for very smooth series
  (e.g., GC=F in low-vol regimes), $P + R$ can be numerically zero
- **Log-likelihood clipping**: Current $|\ell_t| \leq 50$ nats clips crisis ticks,
  but also clips informative ticks in calm regimes where $\ell_t > 50$ legitimately

### Story 4.1: Joseph Form Covariance Update

**As a** Kalman filter processing 2000+ daily returns,
**I need** guaranteed positive-definite state covariance $P_{t|t} > 0$ at every step,
**So that** the filter never produces negative variance (which causes NaN propagation).

**Acceptance Criteria**:
- [ ] $P_{t|t} > 0$ for all $t$ across all test assets (assert in debug mode)
- [ ] On MSTR (extreme innovations), no NaN in mu or P trajectories
- [ ] Joseph form matches standard form to within 1e-14 for well-conditioned cases
- [ ] No regression in BIC/CRPS on any test asset
- [ ] Validated on: MSTR, BTC-USD, NFLX, GC=F, SI=F, RKLB, PLTR, UPST

**Tasks**:
1. Replace `P_filt = (1 - K) * P_pred` with Joseph form in `_kalman_filter_phi()`
2. Add `assert P_filt > _MIN_VARIANCE` in debug mode
3. Run full test universe: verify no NaN in any mu/P trajectory
4. Measure numerical difference: $|P_{joseph} - P_{standard}|$ across 1000 steps
5. Benchmark: Joseph form adds 2 multiplications per step -- measure impact

### Story 4.2: Adaptive Log-Likelihood Clipping

**As a** model selection system using BIC for ranking,
**I need** log-likelihood contributions that preserve signal without crisis domination,
**So that** BIC ranking reflects true model fit, not outlier sensitivity.

**Acceptance Criteria**:
- [ ] Adaptive clip: $|\ell_t| \leq c_{clip} \times \text{MAD}(\ell_{1:t-1})$ with $c_{clip} = 5$
- [ ] On SPY (well-behaved): < 0.1% of timesteps are clipped
- [ ] On MSTR (crisis-prone): < 2% of timesteps are clipped
- [ ] BIC model ranking unchanged for SPY (Gaussian vs Student-t order preserved)
- [ ] Validated on: SPY, MSTR, BTC-USD, NFLX, GOOGL, CRWD, SOFI

**Tasks**:
1. Compute running MAD of log-likelihood: $\text{MAD}_t = \text{median}(|\ell_{1:t} - \text{median}(\ell)|)$
2. Set adaptive clip: $c_{max} = \max(50, 5 \times \text{MAD}_t)$
3. Track clip frequency per asset in diagnostics
4. Compare BIC rankings before/after on all 25 test assets
5. Add `clip_mode` parameter: `'fixed'` (current), `'adaptive'` (new), `'none'` (for debugging)

### Story 4.3: Cross-Validated Phi Shrinkage with Asset-Class Priors

**As a** Bayesian filter fitting $\phi$ (drift persistence) across diverse assets,
**I need** asset-class-specific $\phi$ priors that reflect structural behavior,
**So that** metals get mean-reverting priors and equities get trending priors.

**Acceptance Criteria**:
- [ ] Metals (GC=F, SI=F): $\phi_{prior} = 0.0$ (mean reversion, $\tau = 0.15$)
- [ ] Large cap equity (MSFT, GOOGL): $\phi_{prior} = 0.3$ (mild persistence, $\tau = 0.20$)
- [ ] Small cap / high vol (MSTR, RKLB): $\phi_{prior} = 0.0$ (agnostic, $\tau = 0.30$)
- [ ] Crypto (BTC-USD): $\phi_{prior} = 0.2$ (momentum, $\tau = 0.25$)
- [ ] Prior pulls $\phi$ by at most 0.15 from MLE estimate
- [ ] PIT improves (lower KS statistic) for 17+ of 25 test assets
- [ ] Validated on: full 25-asset test universe

**Tasks**:
1. Extend `_phi_shrinkage_log_prior()` with asset-class dispatch
2. Add `_detect_asset_class()` function (reuse from phi_student_t.py)
3. Set class-specific $(\phi_0, \tau)$ pairs per the acceptance criteria
4. Run before/after comparison: PIT KS, BIC, CRPS on all 25 assets
5. Add diagnostic: $\Delta\phi = \phi_{MLE} - \phi_{shrunk}$ per asset to tuning log

### Story 4.4: Scale-Aware Process Noise Floor with Regime Detection

**As a** Kalman filter that must maintain meaningful state uncertainty,
**I need** $q_{min}$ that adapts to the current volatility regime,
**So that** the state does not collapse to a deterministic path during calm markets.

**Acceptance Criteria**:
- [ ] $q_{min} = \max(10^{-8}, 0.001 \times \text{Var}(\sigma^2_{ewma}), 0.002 \times \text{Var}(r))$
- [ ] On GC=F (low vol periods): state uncertainty $P_t$ never drops below $10^{-10}$
- [ ] On MSTR (high vol): $q_{min}$ does not over-regularize (BIC not worse than unconstrained)
- [ ] PIT U-shape (sign of over-confidence) eliminated for GC=F
- [ ] Validated on: GC=F, SI=F, SPY, MSFT, AMZN, BTC-USD, ETH-USD

**Tasks**:
1. Compute `vol_var_median = np.median(np.diff(ewma_vol ** 2) ** 2)` at tuning time
2. Compute `ret_var = np.var(returns)` as scale reference
3. Set `q_min = max(1e-8, 0.001 * vol_var_median, 0.002 * ret_var)`
4. Add `q_floor_active` flag to diagnostics (True when q was floored)
5. Run PIT comparison: count assets with KS improvement vs degradation

---

## Epic 5: Gaussian Unified Model Pipeline Accuracy

**File**: `src/models/gaussian.py`
**Priority**: HIGH -- 5-stage pipeline; errors compound across stages

### Background

The unified Gaussian optimization is a 5-stage pipeline:
1. **Stage 1**: Base $(q, c, \phi)$ via L-BFGS-B
2. **Stage 2**: Variance inflation $\beta$ via PIT MAD grid
3. **Stage 3**: GJR-GARCH on residuals
4. **Stage 4**: Causal EWM location + CRPS shrinkage
5. **Stage 5**: Walk-forward CV calibration

Each stage builds on the previous. Errors in Stage 1 propagate through all subsequent stages.
The critical question: **does the pipeline converge to a global optimum, or get trapped in
local minima from Stage 1?**

### Story 5.1: Multi-Start Optimization for Stage 1

**As a** model fitter seeking the global maximum likelihood,
**I need** multiple initialization points for $(q, c, \phi)$,
**So that** the L-BFGS-B optimizer does not settle in a local minimum.

**Acceptance Criteria**:
- [ ] At least 3 starting points: $(q_0, c_0, \phi_0) \in \{(10^{-6}, 1, 0), (10^{-4}, 0.5, 0.5), (10^{-5}, 2, -0.3)\}$
- [ ] Best start selected by cross-validated log-likelihood (not training LL)
- [ ] On RKLB: multi-start finds better optimum than single-start in > 30% of runs
- [ ] Average BIC improvement > 2 nats across test universe
- [ ] Runtime increase < 3x (3 starts vs 1 start, each bounded by maxiter=200)
- [ ] Validated on: RKLB, MSTR, NFLX, TSLA, IWM, DKNG, AFRM

**Tasks**:
1. Define 3 initial points spanning the parameter space
2. Run L-BFGS-B from each start, record final LL and parameters
3. Select winner by CV log-likelihood (not training LL)
4. Add `n_starts` parameter to `optimize_params()` (default=3)
5. Profile: measure wall-clock time for 3-start vs 1-start on 25 assets

### Story 5.2: Variance Inflation Stage with Quantile Matching

**As a** calibration system needing well-calibrated prediction intervals,
**I need** variance inflation $\beta$ chosen to match empirical coverage,
**So that** 95% prediction intervals actually contain 95% of outcomes.

**Acceptance Criteria**:
- [ ] 95% prediction interval coverage: $94\% \leq \hat{p}_{95} \leq 96\%$ for all test assets
- [ ] 80% prediction interval coverage: $79\% \leq \hat{p}_{80} \leq 81\%$ for all test assets
- [ ] $\beta$ selected by minimizing: $\sum_q (coverage_q - target_q)^2$ over $q \in \{0.50, 0.80, 0.95\}$
- [ ] On BTC-USD: $\beta > 1.0$ (expected -- fat tails need wider intervals under Gaussian)
- [ ] Validated on: SPY, BTC-USD, GC=F, MSTR, GOOGL, META, SQ

**Tasks**:
1. Compute empirical coverage at quantiles $q \in \{0.50, 0.80, 0.90, 0.95, 0.99\}$
2. Define objective: $L(\beta) = \sum_q (ecov(q, \beta) - q)^2$
3. Grid search $\beta \in [0.8, 0.85, ..., 2.0]$ on training fold
4. Cross-validate: measure coverage on test fold
5. Report coverage table: asset x quantile, before and after

### Story 5.3: Causal EWM Location Correction with Bias Prevention

**As a** filter correcting systematic location bias,
**I need** EWM correction that is strictly causal (no look-ahead),
**So that** PIT mean is centered at 0.5 without introducing future information.

**Acceptance Criteria**:
- [ ] EWM uses only $\{r_1, ..., r_{t-1}\}$ to correct $\mu_{pred,t}$ (strict causality)
- [ ] PIT mean on training set: $0.48 \leq \bar{u} \leq 0.52$
- [ ] PIT mean on TEST set (walk-forward): $0.47 \leq \bar{u} \leq 0.53$
- [ ] No degradation in CRPS on any test asset vs no-correction baseline
- [ ] Validated on: full 25-asset universe

**Tasks**:
1. Audit current EWM: verify lag-1 indexing ($ewm_t$ uses $r_{t-1} - \mu_{t-1}$)
2. Add unit test: inject known bias, verify EWM corrects it within 30 steps
3. Run walk-forward PIT analysis: training vs test PIT mean
4. Add `ewm_bias_diagnostic` to tuning output: mean correction magnitude per asset
5. Test edge case: first 10 observations (EWM not yet stable) -- verify no NaN

### Story 5.4: Walk-Forward Cross-Validation with Purged Gaps

**As a** model selector using walk-forward CV,
**I need** purged gaps between train and test folds to prevent information leakage,
**So that** CV scores are unbiased estimates of true out-of-sample performance.

**Acceptance Criteria**:
- [ ] Gap of $g = 5$ days between train and test fold (purging autocorrelation)
- [ ] At least 5 folds with expanding window (min train size = 50% of data)
- [ ] CV score is average of test-fold log-likelihoods
- [ ] For SPY: CV-selected model matches true out-of-sample best in > 80% of cases
- [ ] Validated on: SPY, QQQ, IWM, AAPL, NVDA, AMZN, PLTR

**Tasks**:
1. Add `purge_gap` parameter to walk-forward CV (default=5)
2. Implement: test fold starts at `train_end + purge_gap + 1`
3. Validate: run with gap=0 vs gap=5, measure CV score difference
4. Compare: CV-selected model vs oracle (true OOS best) agreement rate
5. Document: add fold diagram to gaussian.py docstring

---

# PART III: PHI-STUDENT-T HEAVY TAIL MODELING

## Epic 6: Student-t Degrees of Freedom Optimization

**File**: `src/models/phi_student_t.py`
**Priority**: CRITICAL -- nu selection determines tail behavior for all heavy-tailed assets

### Background

The current approach uses a discrete grid $\nu \in \{3, 4, 8, 20\}$ and selects via BIC.
This is computationally efficient but leaves significant accuracy on the table:

1. **Grid gaps**: The jump from $\nu = 4$ to $\nu = 8$ is enormous in tail weight.
   For MSTR, the true optimal might be $\nu^* = 5.3$. Current grid forces $\nu = 4$ (too heavy)
   or $\nu = 8$ (too light).
2. **Asset heterogeneity**: BTC-USD might need $\nu = 3.5$, while MSFT needs $\nu = 15$.
   The grid lacks resolution in both extremes.
3. **Regime dependence**: Bull markets have lighter tails than bear markets, but
   $\nu$ is fixed across the entire sample.

### Story 6.1: Continuous Nu Optimization via Profile Likelihood

**As a** model fitter seeking the optimal tail parameter,
**I need** continuous $\nu$ optimization (not grid search),
**So that** tail weight matches the true data-generating process.

**Acceptance Criteria**:
- [ ] $\nu^*$ optimized over $[2.1, 50]$ via profile likelihood
- [ ] Profile: fix $(c, \phi, q)$ at grid-selected values, optimize $\nu$ via golden section
- [ ] For MSTR: $\nu^*$ differs from grid-selected by > 0.5 in at least 30% of regimes
- [ ] BIC improvement: > 3 nats on average for heavy-tailed assets (MSTR, BTC-USD, RKLB)
- [ ] No BIC degradation for thin-tailed assets (SPY, MSFT)
- [ ] Validated on: MSTR, BTC-USD, RKLB, SI=F, SPY, MSFT, GOOGL, CRWD, UPST

**Tasks**:
1. After grid search selects best $\nu_{grid}$, refine in $[\nu_{grid} - 2, \nu_{grid} + 4]$
2. Use golden section search (scipy.optimize.minimize_scalar) on profile LL
3. Evaluate BIC at refined $\nu^*$ and compare to grid $\nu$
4. Run on full test universe: report $\Delta$BIC and $\Delta\nu$ per asset
5. Add `refine_nu` flag to `optimize_params_fixed_nu()` (default=True)

### Story 6.2: Regime-Dependent Nu via Markov Switching

**As a** model that encounters both calm and crisis periods,
**I need** $\nu$ that adapts to the current volatility regime,
**So that** crisis periods get heavier tails and calm periods get lighter tails.

**Acceptance Criteria**:
- [ ] Two-regime $\nu$: $\nu_{calm} \geq 8$, $\nu_{crisis} \leq 6$
- [ ] Regime assignment via volatility z-score (consistent with MS-q)
- [ ] On MSTR: crisis-period PIT improves (less U-shaped) with low $\nu_{crisis}$
- [ ] On MSFT: calm-period PIT improves (less peaked) with high $\nu_{calm}$
- [ ] Overall CRPS improves for 17+ of 25 test assets
- [ ] Validated on: MSTR, MSFT, NFLX, GC=F, BTC-USD, SPY, RKLB, DKNG, IONQ

**Tasks**:
1. Use MS-q stress probability: $p_{stress}(t) = \sigma(sens \times z_t)$
2. Effective nu: $\nu_{eff}(t) = (1 - p_s)\nu_{calm} + p_s \nu_{crisis}$
3. Feed $\nu_{eff}(t)$ to `_student_t_logpdf_dynamic_nu` kernel
4. Optimize $(\nu_{calm}, \nu_{crisis})$ jointly via grid + profile LL
5. Report: regime assignment accuracy vs realized tail behavior

### Story 6.3: Smooth Asymmetric Nu Calibration per Asset Class

**As a** unified Student-t model with asymmetric tail behavior,
**I need** calibrated $(\alpha_{asym}, k_{asym})$ parameters per asset class,
**So that** crash tails are heavier than recovery tails (empirical fact).

**Acceptance Criteria**:
- [ ] Equities: $\alpha_{asym} < 0$ (heavier left tail), calibrated from empirical skewness
- [ ] Gold: $\alpha_{asym} \approx 0$ (symmetric, safe haven dynamics)
- [ ] Crypto: $\alpha_{asym}$ unconstrained (can be positive -- bubble dynamics)
- [ ] PIT left tail (u < 0.10) improvement for equity assets
- [ ] PIT right tail (u > 0.90) improvement for crypto assets
- [ ] Validated on: MSTR, NFLX, TSLA, GC=F, BTC-USD, AAPL, META, SOFI

**Tasks**:
1. Compute empirical tail ratio: $\hat{\alpha} = (\text{kurtosis}_{left} - \text{kurtosis}_{right}) / \text{kurtosis}_{total}$
2. Use as warm start for $\alpha_{asym}$ optimization
3. Grid search $k_{asym} \in [0.5, 1.0, 1.5, 2.0]$ with profile LL
4. Compare PIT histogram: left vs right tail bins before and after
5. Add `asymmetry_diagnostic` to tuning output: $\alpha$, left/right tail KS

---

## Epic 7: Markov-Switching Process Noise (MS-q) Precision

**File**: `src/models/phi_student_t.py`
**Priority**: CRITICAL -- MS-q is the proactive regime detector; inaccuracy delays transitions

### Background

MS-q switches process noise between calm and stress regimes:

$$q_t = (1 - p_{stress}(t)) \cdot q_{calm} + p_{stress}(t) \cdot q_{stress}$$
$$p_{stress}(t) = \frac{1}{1 + \exp(-\text{sens} \times (\text{vol\_relative} - \text{threshold}))}$$

The critical accuracy dimension is **transition speed**: how fast does $p_{stress}$ respond
to volatility spikes? Current sigmoid sensitivity is $\text{sens} = 2.0$, which means:
- At vol_relative = threshold + 1: $p_{stress} = 0.88$
- At vol_relative = threshold + 0.5: $p_{stress} = 0.73$
- At vol_relative = threshold: $p_{stress} = 0.50$

For extreme events (COVID crash, SVB crisis), $p_{stress}$ should reach 0.95+ within 3 days.

### Story 7.1: Calibrated Sensitivity per Asset Class

**As a** MS-q module that must detect regime transitions promptly,
**I need** sensitivity parameters calibrated to each asset class's volatility dynamics,
**So that** gold detects slow macro shifts and MSTR detects sudden gaps.

**Acceptance Criteria**:
- [ ] Gold (GC=F): sensitivity = 4.0, threshold = 1.5 (slow, macro-driven)
- [ ] Silver (SI=F): sensitivity = 4.5, threshold = 1.2 (faster, gap-prone)
- [ ] Large cap (MSFT, GOOGL): sensitivity = 2.5, threshold = 1.3 (standard)
- [ ] High vol (MSTR, RKLB): sensitivity = 3.0, threshold = 1.0 (earlier switching)
- [ ] Crypto (BTC-USD): sensitivity = 3.5, threshold = 1.1 (structural breaks)
- [ ] Transition speed: $p_{stress}$ reaches 0.90 within 5 days of vol spike for all assets
- [ ] Validated on: GC=F, SI=F, MSFT, MSTR, RKLB, BTC-USD, SQ, UPST

**Tasks**:
1. Extend `_detect_asset_class()` to return MS-q parameter overrides
2. Calibrate sensitivity on historical vol spikes (March 2020, Jan 2022, Mar 2023)
3. Measure transition latency: days from vol spike to $p_{stress} > 0.90$
4. Compare PIT during crisis windows: before vs after calibrated sensitivity
5. Add `ms_transition_speed` diagnostic to tuning output

### Story 7.2: EWM vs Expanding Window Z-Score Selection

**As a** volatility z-score estimator feeding the MS-q module,
**I need** the optimal baseline estimator (EWM or expanding window) per asset,
**So that** the z-score responds at the right speed for each asset class.

**Acceptance Criteria**:
- [ ] EWM with $\lambda = 0.97$ preferred for metals (slow regimes, long memory)
- [ ] EWM with $\lambda = 0.94$ preferred for high vol equity (fast regimes)
- [ ] Expanding window preferred for indices (stable, long-run mean is informative)
- [ ] Selection via in-sample BIC comparison (EWM-based MS-q vs expanding-window MS-q)
- [ ] Convergence: expanding window requires $n > 100$ for stable mean estimate
- [ ] Validated on: GC=F, SI=F, SPY, MSTR, BTC-USD, IWM, PLTR, AFRM

**Tasks**:
1. Implement both z-score methods in `compute_ms_process_noise_smooth()`
2. Add `ms_z_method` parameter: `'ewm'`, `'expanding'`, `'auto'` (BIC-selected)
3. Run BIC comparison on all 25 test assets: report which method wins per asset
4. Verify convergence: expanding window z-score stable after $t = 100$
5. Add `z_score_baseline` diagnostic to tuning output

### Story 7.3: Joint q_calm / q_stress Optimization with Identifiability Constraints

**As a** parameter estimator fitting two process noise levels,
**I need** identifiability constraints that prevent $q_{calm} \approx q_{stress}$,
**So that** the two-regime model is genuinely bimodal (not a single-regime collapse).

**Acceptance Criteria**:
- [ ] Minimum ratio: $q_{stress} / q_{calm} \geq 5$ (enforced as hard constraint)
- [ ] Maximum ratio: $q_{stress} / q_{calm} \leq 1000$ (prevent numerical issues)
- [ ] On SPY: ratio $\approx 50$-$100$ (moderate regime difference)
- [ ] On BTC-USD: ratio $\approx 200$-$500$ (extreme regime difference)
- [ ] BIC of MS-q model must beat single-q model by > 5 nats to be selected
- [ ] Validated on: SPY, BTC-USD, MSTR, GC=F, SI=F, NVDA, CRWD, IONQ

**Tasks**:
1. Add constraint: $\log(q_{stress}/q_{calm}) \in [\log 5, \log 1000]$ to optimizer bounds
2. Implement LRT (likelihood ratio test): MS-q vs single-q, threshold $\chi^2_1(0.05) = 3.84$
3. Report: fraction of assets where MS-q is selected over single-q
4. Track regime assignments: fraction of time in stress per asset (sanity check)
5. Visualize: $q_t$ trajectory overlay with price chart for MSTR (crisis validation)

---

## Epic 8: Two-Piece and Mixture Student-t Enhancements

**File**: `src/models/phi_student_t.py`, `src/models/numba_kernels.py`
**Priority**: HIGH -- these are the advanced tail models competing in BMA

### Background

Two competing approaches for tail asymmetry:

**Two-Piece Student-t**: Different $\nu_L$ (crash) and $\nu_R$ (recovery):
$$f(r | \nu_L, \nu_R, \mu, \sigma) = \begin{cases}
  c \cdot t(r; \nu_L, \mu, \sigma) & r < \mu \\
  c \cdot t(r; \nu_R, \mu, \sigma) & r \geq \mu
\end{cases}$$

**Two-Component Mixture**: Blend calm and stress Student-t:
$$f(r) = w_t \cdot t(r; \nu_{calm}) + (1 - w_t) \cdot t(r; \nu_{stress})$$

The mixture has richer dynamics (weight changes over time) but more parameters (3 extra vs 2).
BIC naturally penalizes the mixture, so it should only win when dynamics truly matter.

### Story 8.1: Two-Piece Student-t with Continuous Nu_L and Nu_R

**As a** two-piece model fitting asymmetric tails,
**I need** continuous optimization of $(\nu_L, \nu_R)$ instead of grid search,
**So that** the left and right tail parameters are precisely calibrated.

**Acceptance Criteria**:
- [ ] Profile likelihood optimization: fix $(c, \phi, q)$, optimize $(\nu_L, \nu_R)$ jointly
- [ ] Equities: $\nu_L < \nu_R$ (heavier crash tails) for 80%+ of equity assets
- [ ] Gold: $\nu_L \approx \nu_R$ (symmetric) -- two-piece should not be selected
- [ ] BIC improvement > 2 nats over discrete grid for assets where two-piece wins
- [ ] No BIC degradation when two-piece loses (falls back to symmetric Student-t)
- [ ] Validated on: TSLA, NFLX, MSTR, GC=F, BTC-USD, AAPL, NVDA, DKNG, UPST

**Tasks**:
1. After grid search selects best $(\nu_L, \nu_R)$, refine via 2D golden section
2. Search domain: $\nu_L \in [2.1, \nu_L^{grid} + 3]$, $\nu_R \in [\nu_R^{grid} - 3, 30]$
3. Evaluate: two-piece BIC vs symmetric BIC; select winner per regime
4. Report: $(\nu_L^*, \nu_R^*)$ per asset with asymmetry diagnostic
5. Cross-validate: refinement on train fold, BIC on test fold

### Story 8.2: Multi-Factor Mixture Weight Dynamics Validation

**As a** mixture model with time-varying weights,
**I need** validated multi-factor conditioning: shocks, vol acceleration, momentum,
**So that** the mixture weight responds to the correct market signals.

**Acceptance Criteria**:
- [ ] Mixture weight $w_t$ responds within 2 days of a 3-sigma shock
- [ ] Vol acceleration signal: $w_t$ drops (more stress) when vol is accelerating
- [ ] Momentum signal: $w_t$ increases (more calm) during sustained trends
- [ ] Factor loadings $(a, b, c)$ calibrated per asset class
- [ ] On MSTR during COVID crash: $w_{stress}$ peaks within 3 days of March 12, 2020
- [ ] Validated on: MSTR, SPY, BTC-USD, NFLX, GC=F, SQ, AFRM

**Tasks**:
1. Compute factor signals: $z_t$ (shock), $\Delta\sigma_t$ (vol accel), $M_t$ (momentum)
2. Log mixture weight trajectory for each test asset
3. Overlay with known crisis dates: verify $w_{stress}$ peak timing
4. Tune $(a, b, c)$ via cross-sectional calibration across 25 assets
5. Report: average response latency (days from event to $w_{stress} > 0.7$)

### Story 8.3: Mixture vs Two-Piece Model Selection Gate

**As a** BMA system choosing between mixture and two-piece models,
**I need** a clear selection criterion beyond BIC,
**So that** the more appropriate model is selected based on data characteristics.

**Acceptance Criteria**:
- [ ] Selection heuristic: mixture preferred when vol-of-vol > 0.5 (dynamic regimes)
- [ ] Two-piece preferred when empirical skewness > |0.3| (static asymmetry)
- [ ] Heuristic agrees with BIC selection in > 80% of cases
- [ ] When both models have similar BIC (within 3 nats): select simpler (two-piece)
- [ ] No asset receives both mixture AND two-piece weights > 0.1 in BMA
- [ ] Validated on: all 25 test assets

**Tasks**:
1. Compute data diagnostics: empirical skewness, vol-of-vol, regime count
2. Implement soft gate: if VoV > 0.5, upweight mixture candidates by 2 BIC nats
3. Implement Occam gate: if $|\text{BIC}_{mixture} - \text{BIC}_{two-piece}| < 3$, prefer two-piece
4. Run BMA with and without gates: measure model selection stability
5. Report: selection frequency table (asset x model type)

---

# PART IV: UNIFIED STUDENT-T ARCHITECTURE

## Epic 9: Unified Phi-Student-t Model Coherence

**File**: `src/models/phi_student_t_unified.py`
**Priority**: CRITICAL -- this is the most complex model; coherence across 60+ parameters

### Background

The unified model has 4 tiers of complexity:
- **Tier 1**: Core Kalman + smooth asymmetric $\nu$ + MS-q
- **Tier 2**: GJR-GARCH + rough volatility + variance mean reversion
- **Tier 3**: Risk premium + GAS skewness
- **Tier 4**: Dynamic leverage + liquidity feedback + PIT stabilizer

Each tier adds parameters. The risk: **parameter interactions** across tiers can cause
unexpected behavior. For example:
- GJR-GARCH (Tier 2) + MS-q (Tier 1) both affect variance, potentially double-counting
- Risk premium (Tier 3) + phi drift (Tier 1) both affect location, causing collinearity
- VoV damping (Tier 1) attempts to resolve GARCH/MS-q interaction, but the damping
  coefficient itself needs calibration

### Story 9.1: Tier Interaction Diagnostics

**As a** unified model with 60+ interacting parameters,
**I need** diagnostics that detect harmful tier interactions,
**So that** parameter estimates are stable and interpretable.

**Acceptance Criteria**:
- [ ] Detect GARCH + MS-q double-counting: if both active, total variance inflation < 2x base
- [ ] Detect phi + risk premium collinearity: correlation $|\rho(\phi, \lambda_1)| < 0.7$
- [ ] Detect VoV + GARCH redundancy: if both active, VoV damping > 0.2
- [ ] Diagnostic flags raised for > 0 assets in test universe (expect some interactions)
- [ ] Flagged assets receive parameter adjustment (damping increased)
- [ ] Validated on: all 25 test assets, with special attention to MSTR, BTC-USD

**Tasks**:
1. After Stage 3 (GARCH), compute: $\text{var\_ratio} = \text{Var}(h_t^{GARCH}) / \text{Var}(q_t^{MS-q})$
2. If var_ratio < 0.2 or > 5.0: flag "GARCH/MS-q imbalance"
3. After Stage 1, compute: $\hat{\rho}(\phi, \lambda_1)$ via bootstrap
4. If $|\hat{\rho}| > 0.7$: disable risk premium (set $\lambda_1 = 0$)
5. Report interaction matrix: (tier_i, tier_j) interaction strength per asset

### Story 9.2: Hierarchical Parameter Optimization

**As a** multi-stage optimizer fitting 60+ parameters,
**I need** a hierarchical strategy that prevents early-stage errors from propagating,
**So that** later stages refine (not corrupt) earlier stage estimates.

**Acceptance Criteria**:
- [ ] Stage 1 parameters $(q, c, \phi)$ locked during Stages 3-5 (no re-optimization)
- [ ] Stage 2 ($\beta$) allowed to update in Stage 5 (calibration adjustment)
- [ ] Stage 3 (GARCH) parameters bounded by 2x Stage 1 variance scale
- [ ] Final model BIC within 5 nats of oracle (all parameters jointly optimized)
- [ ] Runtime < 30 seconds per asset on 2-year daily data
- [ ] Validated on: MSFT, RKLB, BTC-USD, GC=F, IWM, AMZN, SOFI

**Tasks**:
1. Add parameter locks: `locked_params = {'q', 'c', 'phi'}` after Stage 1
2. Stage 3 bounds: GARCH persistence $< 0.999$, leverage $< 0.5$
3. Stage 5 refinement: only $\beta$, `crps_sigma_shrinkage` are free
4. Benchmark: staged vs joint optimization BIC comparison on 5 assets
5. Profile: wall-clock time per stage on MSTR (longest series)

### Story 9.3: Asset-Class Profile Validation Suite

**As a** system with 5 predefined asset-class profiles,
**I need** empirical validation that each profile outperforms generic defaults,
**So that** asset-class specialization genuinely improves accuracy.

**Acceptance Criteria**:
- [ ] Gold profile: CRPS < 0.018 on GC=F (vs generic default CRPS)
- [ ] Silver profile: PIT KS p > 0.15 on SI=F during vol spikes
- [ ] High-vol equity profile: CSS > 0.70 on MSTR
- [ ] Forex profile: FEC > 0.80 on USDJPY (or proxy)
- [ ] Profile vs generic improvement: > 3% CRPS reduction for 4/5 asset classes
- [ ] Validated on: GC=F, SI=F, MSTR, BTC-USD, ETH-USD, SPY, CRWD, UPST (proxy for each class)

**Tasks**:
1. Run each test asset twice: with asset-class profile and with generic defaults
2. Record: BIC, CRPS, PIT KS, CSS, FEC for both runs
3. Compute: $\Delta\text{CRPS} = \text{CRPS}_{generic} - \text{CRPS}_{profile}$ (positive = improvement)
4. Statistical test: paired t-test across 5 assets, $H_0$: profile = generic
5. If any profile underperforms: adjust profile parameters and re-validate

---

## Epic 10: GJR-GARCH Integration in Unified Pipeline

**File**: `src/models/phi_student_t_unified.py`, `src/models/numba_kernels.py`
**Priority**: HIGH -- Tier 2 variance dynamics affect all downstream scoring

### Background

The unified model's GARCH integration must satisfy three constraints:
1. **Consistency**: GARCH $h_t$ and Kalman $S_t$ must be coherent (no double-counting)
2. **Stationarity**: GARCH persistence must be bounded for finite forecasts
3. **Interaction**: VoV damping coefficient must prevent GARCH + VoV + MS-q triple-counting

The current architecture feeds GARCH $h_t$ into the observation noise:
$$R_t = c \cdot h_t \cdot (1 + \gamma_{VoV} \cdot (1 - \text{damp}) \cdot |\Delta\log\sigma_t|) \cdot \frac{\nu - 2}{\nu}$$

This is multiplicative, meaning errors in $h_t$ are amplified by subsequent factors.

### Story 10.1: GARCH-Kalman Variance Coherence Check

**As a** unified model combining GARCH variance with Kalman innovation variance,
**I need** a coherence diagnostic that detects when $h_t$ and $S_t$ diverge,
**So that** the observation noise accurately reflects true return uncertainty.

**Acceptance Criteria**:
- [ ] Coherence ratio: $\rho_t = h_t / (S_t - P_{t|t-1}) \in [0.5, 2.0]$ for 95%+ of timesteps
- [ ] When $\rho_t \notin [0.5, 2.0]$: flag as "variance incoherence"
- [ ] On MSTR: fewer than 5% of timesteps flagged during non-crisis periods
- [ ] On SPY: fewer than 1% of timesteps flagged
- [ ] Fix: when incoherent, blend $R_t = 0.5 \cdot c \cdot h_t + 0.5 \cdot \hat{R}_{Kalman}$
- [ ] Validated on: MSTR, SPY, BTC-USD, NFLX, GOOGL, PLTR, IONQ

**Tasks**:
1. After Kalman update, compute $\hat{R}_{Kalman} = S_t - P_{t|t-1}$
2. Compute coherence ratio $\rho_t = h_t / \hat{R}_{Kalman}$
3. If $\rho_t \notin [0.5, 2.0]$: apply blended $R_t$
4. Track: fraction of timesteps blended per asset
5. Report: $\Delta$BIC, $\Delta$CRPS from coherence correction

### Story 10.2: GJR Leverage Effect Calibration

**As a** GARCH model capturing asymmetric volatility response,
**I need** leverage parameter $\gamma_{lev}$ calibrated to empirical news impact,
**So that** negative returns increase volatility more than positive returns of equal magnitude.

**Acceptance Criteria**:
- [ ] Equities (MSFT, GOOGL, NFLX): $\gamma_{lev} > 0$ (leverage effect present)
- [ ] Gold (GC=F): $\gamma_{lev} \approx 0$ (symmetric news impact)
- [ ] Crypto (BTC-USD): $\gamma_{lev}$ may be positive or negative (bubble dynamics)
- [ ] News impact curve: $h(z) = \omega + (\alpha + \gamma I(z<0))z^2$ plotted for each asset
- [ ] Empirical asymmetry matches fitted asymmetry to within 20%
- [ ] Validated on: MSFT, NFLX, TSLA, GC=F, BTC-USD, SI=F, META, DKNG

**Tasks**:
1. Compute empirical news impact: bin returns by magnitude, compute conditional variance
2. Fit GJR parameters: match empirical conditional variance to GJR formula
3. Compare fitted vs empirical news impact curves
4. Validate: $\gamma_{lev}$ positive for equities, near-zero for gold
5. Add news impact plot to diagnostics output

### Story 10.3: Rough Volatility (Hurst H) Integration

**As a** variance dynamics module capturing long-memory in volatility,
**I need** fractional Brownian motion Hurst exponent estimation,
**So that** the model captures the "rough" nature of realized volatility.

**Acceptance Criteria**:
- [ ] Hurst $H \in [0.05, 0.20]$ for most equity assets (Gatheral-Jaisson-Rosenbaum finding)
- [ ] Gold: $H \approx 0.10$-$0.15$ (moderate roughness)
- [ ] BTC-USD: $H$ potentially higher (less rough, more persistent)
- [ ] CRPS improvement > 2% for assets where $H < 0.15$ (rough regime)
- [ ] No degradation for assets where $H > 0.30$ (smooth regime -- disable rough vol)
- [ ] Validated on: MSFT, TSLA, GC=F, BTC-USD, RKLB, IWM, AMZN, SOFI

**Tasks**:
1. Estimate $H$ via variogram method on log-realized-volatility
2. Integrate into observation noise: $R_t = c \cdot \sigma_t^2 \cdot t^{2H-1}$ memory correction
3. Add $H$ to `UnifiedStudentTConfig`
4. Benchmark: rough vol on vs off, BIC/CRPS/PIT comparison
5. Auto-disable: if estimated $H > 0.30$, set $H = 0$ (revert to standard)

---

# PART V: HANSEN SKEW-T AND CONTAMINATED STUDENT-T

## Epic 11: Hansen Skew-t Observation Model Accuracy

**File**: `src/models/numba_kernels.py` (hansen kernels)
**Priority**: HIGH -- newest observation model, needs thorough validation

### Background

The Hansen (1994) skew-t distribution extends Student-t with an asymmetry parameter $\lambda$:

$$f(z | \nu, \lambda) = \begin{cases}
bc \left[1 + \frac{(bz+a)^2}{(1+\lambda)^2(\nu-2)}\right]^{-(\nu+1)/2} \frac{1}{1+\lambda} & z < -a/b \\
bc \left[1 + \frac{(bz+a)^2}{(1-\lambda)^2(\nu-2)}\right]^{-(\nu+1)/2} \frac{1}{1-\lambda} & z \geq -a/b
\end{cases}$$

where constants $(a, b, c)$ ensure zero mean and unit variance:
$$a = b \frac{m_1(\nu)}{\sqrt{1+3\lambda^2 - a^2}} \quad \text{(recursive)}$$

Critical accuracy concerns:
1. **Piecewise junction**: The PDF must be continuous at $z = -a/b$
2. **Normalization**: $\int f(z) dz = 1$ must hold to 1e-10
3. **Robust Kalman weights**: The weight function $w(z)$ must be monotone decreasing

### Story 11.1: Hansen Constants Numerical Precision

**As a** Hansen skew-t kernel computing constants $(a, b, c)$,
**I need** precision better than 1e-12 for all $(\nu, \lambda)$ in the valid domain,
**So that** the PDF integrates to exactly 1.0 and the Kalman weights are correct.

**Acceptance Criteria**:
- [ ] Constants match analytical formulas to within 1e-12
- [ ] For $\nu = 4, \lambda = 0.3$: $\int_{-\infty}^{\infty} f(z) dz = 1.0 \pm 10^{-10}$
- [ ] For $\lambda = 0$ (symmetric): Hansen reduces to standard Student-t to 1e-14
- [ ] No NaN for $\nu \in [2.1, 50]$, $\lambda \in [-0.99, 0.99]$
- [ ] Validated numerically on 100-point $(\nu, \lambda)$ grid

**Tasks**:
1. Implement analytical constant computation (not iterative)
2. Add numerical integration test: Simpson's rule on $[-20, 20]$ with 10,000 points
3. Verify symmetry: $f(z; \nu, 0) = t(z; \nu)$ for standard Student-t
4. Test boundary: $\lambda = \pm 0.99$, $\nu = 2.1$ (most extreme valid point)
5. Add `hansen_validate_constants()` diagnostic function

### Story 11.2: Hansen Skew-t Kalman Robust Weight Function

**As a** Kalman filter using Hansen observation noise,
**I need** robust weights $w(z)$ that downweight outliers asymmetrically,
**So that** crash outliers get more downweighting than rally outliers.

**Acceptance Criteria**:
- [ ] Weight function: $w(z) = \frac{(\nu+1)/\nu}{1 + z^2/(\nu \cdot s(\lambda)^2)}$ 
      where $s(\lambda)$ is the piecewise scale
- [ ] For $\lambda > 0$ (right-skew): right outliers get less downweighting
- [ ] For $\lambda < 0$ (left-skew): left outliers get less downweighting
- [ ] Monotonicity: $|z_1| > |z_2| \Rightarrow w(z_1) \leq w(z_2)$ (on each piece)
- [ ] On TSLA: fitted $\lambda < 0$ (heavier left tail), crash returns downweighted
- [ ] Validated on: TSLA, NFLX, MSTR, AAPL, RKLB, BTC-USD, SQ, AFRM

**Tasks**:
1. Verify `hansen_robust_weight_scalar()` produces monotone decreasing weights
2. Plot weight function for $\lambda \in \{-0.5, 0, 0.5\}$, $\nu = 5$
3. Compare Kalman trajectory: robust Hansen vs standard Student-t on TSLA
4. Measure: fraction of timesteps with $w < 0.5$ (heavy downweighting)
5. PIT comparison: Hansen skew-t vs symmetric Student-t on equity assets

### Story 11.3: Hansen Lambda Estimation via Profile Likelihood

**As a** model fitting the asymmetry parameter $\lambda$,
**I need** stable maximum likelihood estimation of $\lambda$ for each asset,
**So that** the skewness parameter reflects true distributional asymmetry.

**Acceptance Criteria**:
- [ ] $\hat{\lambda}$ estimated via profile likelihood: fix $(\nu, c, \phi, q)$, optimize $\lambda$
- [ ] Equities: $\hat{\lambda} < 0$ for 70%+ of assets (left-skew, crash risk)
- [ ] Gold: $\hat{\lambda} \approx 0 \pm 0.1$ (near-symmetric)
- [ ] MSTR: $|\hat{\lambda}| > 0.2$ (strongly asymmetric)
- [ ] Standard error of $\hat{\lambda}$ computed via Fisher information
- [ ] Validated on: MSFT, NFLX, TSLA, GC=F, MSTR, BTC-USD, RKLB, CRWD, UPST, ETH-USD

**Tasks**:
1. Profile likelihood: $\ell(\lambda) = \sum_t \log f(r_t | \hat{\theta}_{-\lambda}, \lambda)$
2. Optimize via bounded L-BFGS-B: $\lambda \in [-0.95, 0.95]$
3. Compute Fisher info: $I(\lambda) = -\ell''(\hat{\lambda})$, SE $= 1/\sqrt{I}$
4. Report: $\hat{\lambda} \pm 2\text{SE}$ per asset
5. BIC comparison: Hansen vs symmetric Student-t, select via $\Delta$BIC > 2

---

## Epic 12: Contaminated Student-t (CST) Model Accuracy

**File**: `src/models/numba_kernels.py` (cst kernels)
**Priority**: MEDIUM-HIGH -- CST handles crisis contamination; accuracy critical for tail risk

### Background

The Contaminated Student-t is a two-component mixture:
$$f(r) = (1 - \epsilon) \cdot t(r; \nu_{normal}) + \epsilon \cdot t(r; \nu_{crisis})$$

where $\epsilon$ is the contamination probability (typically 1-5%) and $\nu_{crisis} < \nu_{normal}$
gives heavier tails during crisis periods.

The CST posterior weight determines the Kalman update blending:
$$w_{CST}(r_t) = \frac{(1-\epsilon) \cdot t(r_t; \nu_n) \cdot w_n + \epsilon \cdot t(r_t; \nu_c) \cdot w_c}{(1-\epsilon) \cdot t(r_t; \nu_n) + \epsilon \cdot t(r_t; \nu_c)}$$

### Story 12.1: Contamination Probability Estimation

**As a** CST model fitting the contamination fraction $\epsilon$,
**I need** robust estimation that separates normal from crisis observations,
**So that** crisis periods are properly identified and not over-diluted.

**Acceptance Criteria**:
- [ ] $\hat{\epsilon}$ estimated via EM algorithm (E-step: posterior probabilities, M-step: MLE)
- [ ] SPY: $\hat{\epsilon} \approx 0.02$-$0.05$ (2-5% crisis ticks)
- [ ] MSTR: $\hat{\epsilon} \approx 0.05$-$0.10$ (frequent gap days)
- [ ] GC=F: $\hat{\epsilon} \approx 0.01$-$0.03$ (rare extreme moves)
- [ ] EM converges within 20 iterations for all test assets
- [ ] Validated on: SPY, MSTR, GC=F, BTC-USD, NFLX, RKLB, PLTR, IONQ

**Tasks**:
1. Implement EM: E-step computes posterior $p(\text{crisis}|r_t)$ for each observation
2. M-step: $\hat{\epsilon} = \frac{1}{n}\sum_t p(\text{crisis}|r_t)$
3. Initialize: $\epsilon_0 = 0.03$ (uninformative), $\nu_n = 8$, $\nu_c = 3$
4. Convergence criterion: $|\epsilon_{k+1} - \epsilon_k| < 10^{-4}$
5. Report: convergence trajectory and final $(\hat{\epsilon}, \hat{\nu}_n, \hat{\nu}_c)$ per asset

### Story 12.2: CST Robust Kalman Weight Accuracy

**As a** Kalman filter using CST posterior weights,
**I need** weights that correctly blend normal and crisis update magnitudes,
**So that** crisis observations don't corrupt the state estimate.

**Acceptance Criteria**:
- [ ] For $|z| < 2$ (normal): $w_{CST} \approx w_{normal}$ (dominated by normal component)
- [ ] For $|z| > 4$ (crisis): $w_{CST} \approx w_{crisis}$ (dominated by crisis component)
- [ ] Transition: smooth and monotone between normal and crisis regimes
- [ ] On MSTR during COVID crash: crisis weights activate for gap-down days
- [ ] Weight matches posterior probability: $|w_{CST} - p(\text{crisis}|r_t)| < 0.1$
- [ ] Validated on: MSTR, SPY, NFLX, BTC-USD, SI=F, DKNG, SOFI

**Tasks**:
1. Compute posterior probability: $p_c(r_t) = \frac{\epsilon \cdot t(r_t;\nu_c)}{(1-\epsilon)t(r_t;\nu_n) + \epsilon t(r_t;\nu_c)}$
2. Blend weights: $w_t = (1-p_c) w_n(r_t) + p_c w_c(r_t)$
3. Verify: plot $w_t$ vs $|z_t|$ -- should show smooth transition
4. Compare: CST weights vs standard Student-t weights on MSTR
5. PIT: CST model vs single Student-t, measure calibration improvement

### Story 12.3: CST vs Hansen Model Selection

**As a** BMA system choosing between Hansen skew-t and CST,
**I need** clear selection criteria based on data characteristics,
**So that** the appropriate tail model is selected for each asset.

**Acceptance Criteria**:
- [ ] CST preferred when: few extreme outliers, otherwise normal (contamination model)
- [ ] Hansen preferred when: systematic asymmetry (persistent skewness)
- [ ] Selection via BIC with model-specific parameter counts
- [ ] Agreement with heuristic: CST for > 60% of equities, Hansen for > 60% of metals
- [ ] Both models should not receive > 0.2 BMA weight simultaneously
- [ ] Validated on: all 25 test assets

**Tasks**:
1. Compute data diagnostics: excess kurtosis, skewness, outlier fraction (> 3 sigma)
2. Heuristic: if outlier fraction > 3% and skewness < |0.3| -> CST; else -> Hansen
3. BIC comparison: CST vs Hansen on all test assets
4. Report: selection table (asset x model x BIC x heuristic_agrees)
5. Investigate disagreements: why does heuristic differ from BIC?

---

# PART VI: NUMBA WRAPPER AND FILTER DISPATCH ACCURACY

## Epic 13: Filter Dispatch Correctness in Numba Wrappers

**File**: `src/models/numba_wrappers.py`
**Priority**: CRITICAL -- dispatch errors silently use wrong filter, corrupting all downstream

### Background

`numba_wrappers.py` is the gateway between Python and compiled Numba kernels. Every filter
call goes through this layer. A dispatch error (wrong kernel, wrong parameters, wrong array
shape) is catastrophic because:
1. Results look plausible (no NaN or crash)
2. But BIC/CRPS/PIT are subtly wrong
3. BMA selects wrong model based on corrupted scores

The current architecture has 10+ dispatch functions. Each must:
- Validate array shapes and dtypes
- Select correct kernel (Gaussian vs Student-t vs Hansen vs CST)
- Pass precomputed constants (gammaln values) correctly
- Handle fallback to pure Python when Numba is unavailable

### Story 13.1: Array Preparation Correctness Audit

**As a** filter dispatch layer handling 71,000+ array preparations per asset,
**I need** guaranteed float64 C-contiguous arrays without silent truncation,
**So that** Numba kernels receive correctly shaped and typed data.

**Acceptance Criteria**:
- [ ] `prepare_arrays()` preserves array values to full float64 precision
- [ ] No silent int32->float64 promotion (which truncates large integers)
- [ ] No silent float32->float64 promotion (which changes precision)
- [ ] All returned arrays are C-contiguous (Numba requirement)
- [ ] Performance: < 1 microsecond per call for 1-2 arrays
- [ ] Validated on: synthetic arrays with edge cases + all 25 test assets

**Tasks**:
1. Add type check: if input is int dtype, raise ValueError (not silent convert)
2. Add precision check: if input is float32, log warning before converting
3. Verify C-contiguity: `assert arr.flags['C_CONTIGUOUS']` in debug mode
4. Benchmark: `prepare_arrays()` call time for n=1,2,3 arrays
5. Edge case test: empty array, single element, NaN array, Inf array

### Story 13.2: Gamma Precomputation Consistency Across Filter Types

**As a** wrapper precomputing $\log\Gamma(\nu/2)$ and $\log\Gamma((\nu+1)/2)$,
**I need** identical gamma values passed to all Student-t kernels,
**So that** BIC scores from different filter variants are comparable.

**Acceptance Criteria**:
- [ ] `precompute_gamma_values(nu)` called once per $\nu$ value, result cached
- [ ] Same cached values used by: standard, momentum, LFO-CV, MS-q, unified filters
- [ ] If scipy unavailable: fall back to Lanczos (not Stirling) for consistency
- [ ] Cache hit rate > 99% (gamma recomputation is rare)
- [ ] Validated on: run full 25-asset tuning pipeline, verify no duplicate gamma calls

**Tasks**:
1. Add `functools.lru_cache` to `precompute_gamma_values()` (keyed on nu rounded to 6 dp)
2. Log cache statistics: hits, misses, unique nu values per asset
3. Verify: same gamma values in `run_phi_student_t_filter()` and `run_unified_phi_student_t_filter()`
4. Test: call with nu=3.000000 and nu=3.000001 -- should cache separately
5. Edge case: nu=2.1 (minimum valid), nu=50 (maximum in grid)

### Story 13.3: LFO-CV Fused Filter Equivalence Verification

**As a** leave-future-out cross-validation module fused with the Kalman filter,
**I need** mathematical equivalence between fused and separate implementations,
**So that** the 40% speed gain does not come at the cost of accuracy.

**Acceptance Criteria**:
- [ ] Fused LFO-CV score matches separate (filter + score) implementation to 1e-10
- [ ] For all test assets: $|\ell_{fused} - \ell_{separate}| < 10^{-8}$ per fold
- [ ] Overall LFO-CV weight difference: $|\Delta w_i| < 10^{-6}$ for all models
- [ ] Speed: fused is > 30% faster than separate (verify claimed 40%)
- [ ] Validated on: SPY, MSTR, BTC-USD, GC=F, RKLB, CRWD, SOFI

**Tasks**:
1. Run both implementations side-by-side on 5 test assets
2. Compare: per-fold log-predictive-density values
3. Compare: final model weights from fused vs separate
4. Benchmark: wall-clock time for fused vs separate on 2-year daily
5. Add regression test: assert equivalence to 1e-8 in test suite

---

## Epic 14: Momentum-Augmented Filter Integration Accuracy

**File**: `src/models/numba_wrappers.py`, `src/models/phi_gaussian.py`, `src/models/phi_student_t.py`
**Priority**: HIGH -- momentum augmentation is enabled by default (94.9% selection rate)

### Background

Momentum augmentation injects an exogenous input into the Kalman state equation:

$$\mu_t = \phi \mu_{t-1} + u_t + w_t$$

where $u_t = \alpha_t \cdot MOM_t - \beta_t \cdot MR_t$ combines momentum and mean reversion.
This is the "state-equation integration" approach validated by the expert panel.

Key accuracy concerns:
1. **Dynamic capping**: $|u_t| \leq k\sqrt{q}$ -- the cap must scale with process noise
2. **Phi shrinkage**: With momentum, $\phi$ must be shrunk toward 1.0 (not 0.0) to prevent
   $\phi$ and $\kappa$ (mean reversion) from being collinear
3. **Equilibrium estimation**: The mean-reversion target must be a state-space equilibrium,
   not a simple moving average

### Story 14.1: Dynamic Momentum Cap Calibration

**As a** momentum-augmented filter with vol-consistent capping,
**I need** the cap $u_{max} = k\sqrt{q}$ calibrated per asset class,
**So that** momentum injection is proportional to the state's natural variability.

**Acceptance Criteria**:
- [ ] $k = 3.0$ for standard assets (3-sigma cap relative to state noise)
- [ ] $k = 2.0$ for high-vol assets (MSTR, BTC-USD) -- tighter cap prevents instability
- [ ] $k = 4.0$ for slow assets (GC=F) -- wider cap allows meaningful injection
- [ ] Cap binding rate: 5-15% of timesteps (caps too rarely = ineffective, too often = signal loss)
- [ ] PIT improvement vs no-momentum baseline for 20+ of 25 test assets
- [ ] Validated on: MSTR, GC=F, MSFT, BTC-USD, RKLB, SPY, IWM, NFLX, PLTR, UPST

**Tasks**:
1. Implement asset-class dispatch for $k$: reuse `_detect_asset_class()`
2. Monitor cap binding: log fraction of timesteps where $|u_t| = u_{max}$
3. If binding rate > 20%: increase $k$ by 0.5 (auto-adapt)
4. If binding rate < 3%: decrease $k$ by 0.5 (momentum too weak)
5. Report: cap binding rate per asset with BIC/CRPS before/after

### Story 14.2: Phi Shrinkage Direction for Momentum Models

**As a** Bayesian prior on $\phi$ in momentum-augmented models,
**I need** shrinkage toward 1.0 (not 0.0) when momentum is active,
**So that** $\phi$ captures persistence while momentum handles trend changes.

**Acceptance Criteria**:
- [ ] Momentum active: $\phi_{prior} = 1.0$, $\tau = 0.10$ (tight shrinkage toward unit root)
- [ ] Momentum inactive: $\phi_{prior} = 0.0$, $\tau = 0.20$ (shrinkage toward mean reversion)
- [ ] Collinearity check: $|\text{corr}(\phi, \kappa)| < 0.5$ after shrinkage
- [ ] No $\phi > 1.05$ (explosive) or $\phi < -0.5$ (oscillatory) in final estimates
- [ ] Validated on: SPY, MSTR, GC=F, RKLB, BTC-USD, NFLX, NVDA, DKNG, AFRM

**Tasks**:
1. Modify `_phi_shrinkage_log_prior()`: check `momentum_enabled` flag
2. If momentum enabled: center = 1.0, tau = 0.10
3. If momentum disabled: center = 0.0, tau = 0.20
4. Compute correlation matrix of $(\phi, \kappa, \alpha_{mom})$ via bootstrap
5. Report: $\phi$ distribution across test universe with and without momentum

### Story 14.3: State-Space Equilibrium Estimation Accuracy

**As a** mean-reversion module estimating the equilibrium level,
**I need** a state-space equilibrium (not moving average) that is lag-free,
**So that** mean-reversion signals are timely and unbiased.

**Acceptance Criteria**:
- [ ] Equilibrium estimated via Kalman smoother (backward pass on filtered states)
- [ ] Lag: equilibrium responds within 5 days of a structural shift
- [ ] On SPY: equilibrium tracks 50-day moving average to within $\pm 0.5\%$
- [ ] On MSTR: equilibrium adapts to regime changes (not anchored to old levels)
- [ ] Mean-reversion signal: $MR_t = \kappa(\mu_t - \mu^*_t)$ is mean-zero over long horizons
- [ ] Validated on: SPY, MSTR, GC=F, IWM, BTC-USD, AAPL, META, IONQ

**Tasks**:
1. Implement Rauch-Tung-Striebel smoother for equilibrium estimation
2. Compare: smoother equilibrium vs 50-day MA, 100-day MA
3. Measure lag: cross-correlation between equilibrium changes and price changes
4. Test: inject structural break (level shift), verify equilibrium adapts within 5 days
5. Validate: $MR_t$ is approximately mean-zero: $|\bar{MR}| < 0.001$

---

# PART VII: PIT CALIBRATION AND PROPER SCORING

## Epic 15: Probability Integral Transform (PIT) Calibration Accuracy

**Files**: `src/models/gaussian.py`, `src/models/phi_gaussian.py`, `src/models/phi_student_t.py`
**Priority**: CRITICAL -- PIT calibration is the gold standard for probabilistic forecast quality

### Background

A well-calibrated model produces PIT values $u_t = F(r_t | \theta, \mathcal{F}_{t-1})$ that are
uniformly distributed on $[0, 1]$. Deviations from uniformity reveal specific misspecifications:

| PIT Shape | Interpretation | Root Cause |
|-----------|---------------|------------|
| U-shaped | Variance too narrow | $R_t$ underestimated; model overconfident |
| Inverse-U | Variance too wide | $R_t$ overestimated; model underconfident |
| Left-skewed | Mean too high | Positive bias in $\mu_{pred}$ |
| Right-skewed | Mean too low | Negative bias in $\mu_{pred}$ |
| Peaked at 0.5 | Tail mismatch | $\nu$ too low for the data |
| Flat with spikes at 0,1 | Tail underweight | $\nu$ too high (tails too thin) |

The critical requirement: PIT must use the **predictive** distribution, not the **filtered**
distribution. Using filtered values introduces look-ahead bias (the filter has already
processed $r_t$ when computing $\mu_{filt,t}$).

### Story 15.1: Predictive PIT Implementation Audit

**As a** calibration system ensuring proper PIT computation,
**I need** verification that ALL models use predictive (not filtered) distributions,
**So that** PIT-based diagnostics are unbiased.

**Acceptance Criteria**:
- [ ] `gaussian.py`: uses $(\mu_{pred}, S_{pred})$ in `pit_ks_predictive()` -- VERIFIED
- [ ] `phi_gaussian.py`: uses $(\mu_{pred}, S_{pred})$ in `pit_ks_predictive()` -- VERIFIED
- [ ] `phi_student_t.py`: uses $(\mu_{pred}, S_{pred})$ in `pit_ks_predictive()` -- VERIFIED
- [ ] `phi_student_t_unified.py`: uses predictive distribution -- VERIFIED
- [ ] Under no code path does `mu_filtered` or `P_filtered` enter PIT computation
- [ ] Unit test: inject known distribution, verify PIT is exactly uniform
- [ ] Validated on: synthetic data (known DGP) + SPY, MSFT

**Tasks**:
1. Audit each `pit_ks_predictive()` implementation: trace $\mu$ and $S$ source
2. Add assertion: `pit_uses_predictive = True` flag in each filter's return struct
3. Synthetic test: generate 10,000 samples from $N(0,1)$, fit model, verify PIT KS p > 0.50
4. Synthetic test: generate from $t(5)$, fit Student-t, verify PIT KS p > 0.50
5. Anti-test: deliberately use filtered values, verify PIT KS p < 0.01 (detecting the bug)

### Story 15.2: PIT-Driven Variance Inflation with Dead-Zone

**As a** calibration module correcting variance via PIT feedback,
**I need** a dead-zone that prevents over-correction in well-calibrated regions,
**So that** the variance inflation does not oscillate around the target.

**Acceptance Criteria**:
- [ ] Dead-zone: no correction when PIT MAD $\in [0.30, 0.55]$ (well-calibrated range)
- [ ] Below 0.30 (too peaked): decrease $\beta$ by 5% per iteration
- [ ] Above 0.55 (too uniform): increase $\beta$ by 5% per iteration
- [ ] Convergence: $\beta$ stabilizes within 10 iterations for all test assets
- [ ] Asset-class dead-zones: metals wider $[0.25, 0.60]$, crypto narrower $[0.28, 0.52]$
- [ ] Validated on: GC=F, SI=F, MSTR, BTC-USD, ETH-USD, SPY, MSFT, SQ

**Tasks**:
1. Implement iterative PIT-var correction with asset-class dead-zones
2. Log convergence trajectory: $\beta$ vs iteration number
3. Verify stability: $|\beta_{k+1} - \beta_k| < 0.01$ at convergence
4. Compare: PIT MAD before and after correction for all test assets
5. Edge case: asset with perfect PIT (MAD = 0.42) -- verify no correction applied

### Story 15.3: PIT Entropy as Calibration Diagnostic

**As a** model diagnostics system,
**I need** PIT entropy as a complementary calibration metric to KS,
**So that** I can detect subtle miscalibration patterns that KS misses.

**Acceptance Criteria**:
- [ ] PIT entropy $H(u) = -\sum_b p_b \log p_b$ where $p_b$ are bin probabilities
- [ ] Well-calibrated: $H(u) \approx \log(B)$ where $B$ = number of bins
- [ ] Entropy ratio: $H(u) / \log(B) \in [0.95, 1.05]$ for well-calibrated models
- [ ] Entropy detects peaked PIT (ratio < 0.90) that KS may miss at small $n$
- [ ] Validated on: all 25 test assets, using $B = 20$ bins

**Tasks**:
1. Implement `pit_entropy(pit_values, n_bins=20)` function
2. Compute reference: $H_{uniform} = \log(20) = 2.996$
3. Report entropy ratio per asset alongside KS p-value
4. Identify assets where KS passes but entropy fails (subtle miscalibration)
5. Add to tuning output: "PIT_entropy_ratio" column

---

## Epic 16: CRPS (Continuous Ranked Probability Score) Optimization

**Files**: `src/models/numba_kernels.py` (crps_student_t_kernel), `src/models/phi_student_t_unified.py`
**Priority**: HIGH -- CRPS is the primary scoring rule for forecast sharpness + calibration

### Background

The CRPS for a Student-t predictive distribution is:

$$\text{CRPS}(F_t, r_t) = \mathbb{E}[|X - r_t|] - 0.5 \mathbb{E}[|X - X'|]$$

where $X, X' \sim t(\nu, \mu_{pred}, \sigma_{pred})$. This decomposes into:
- **Reliability** (calibration): how well the distribution matches realized outcomes
- **Resolution** (sharpness): how concentrated the distribution is

The current implementation uses Monte Carlo estimation with $n = 1000$ samples per timestep,
which introduces MC noise of order $O(1/\sqrt{n})$. For a CRPS target of 0.018, the MC
standard error is ~0.001 -- 5% relative noise.

### Story 16.1: Closed-Form Student-t CRPS

**As a** scoring system evaluating distributional forecast quality,
**I need** a closed-form CRPS formula for Student-t distributions,
**So that** CRPS computation is exact (not MC-estimated) and deterministic.

**Acceptance Criteria**:
- [ ] Closed-form CRPS using the formula (Gneiting & Raftery, 2007):
      $\text{CRPS}(t_\nu, r) = z(2F_\nu(z) - 1) + 2f_\nu(z)\frac{\nu + z^2}{\nu - 1} - \frac{2\sqrt{\nu}}{\nu-1}\frac{B(0.5, \nu-0.5)}{B(0.5, \nu/2)^2}$
- [ ] Matches MC CRPS (n=100,000) to within 1e-6 relative error
- [ ] Deterministic: same inputs always produce same output
- [ ] Speed: 100x faster than MC with n=1000
- [ ] Validated on: SPY, MSTR, BTC-USD, GC=F, NFLX, CRWD, UPST

**Tasks**:
1. Implement closed-form CRPS in `numba_kernels.py` using Student-t PDF/CDF
2. Handle special case $\nu = 1$ (Cauchy) where integral diverges
3. Verify against MC reference: 25 assets x 5 $\nu$ values
4. Replace `crps_student_t_kernel()` MC implementation with closed-form
5. Benchmark: computation time per timestep, old vs new

### Story 16.2: CRPS-Optimal Sigma Shrinkage Calibration

**As a** forecast system seeking CRPS-optimal prediction intervals,
**I need** the optimal $\alpha_{crps}$ that minimizes average CRPS,
**So that** forecasts are as sharp as possible while remaining calibrated.

**Acceptance Criteria**:
- [ ] Analytical optimum: $\alpha^*_{crps} = \sqrt{\frac{\nu}{(\nu-2)(1+1/\nu)}}$ for $\nu > 2$
- [ ] For $\nu = 4$: $\alpha^* \approx 0.894$; for $\nu = 8$: $\alpha^* \approx 0.935$
- [ ] Empirical validation: grid search $\alpha$ on test fold matches analytical to within 0.02
- [ ] CRPS improves by > 3% on average when using optimal $\alpha$ vs $\alpha = 1$
- [ ] PIT MAD does not degrade by more than 0.03 (slight miscalibration acceptable for sharpness)
- [ ] Validated on: SPY, MSTR, GOOGL, GC=F, BTC-USD, RKLB, AMZN, AFRM

**Tasks**:
1. Implement analytical $\alpha^*_{crps}$ formula
2. Validate: grid search $\alpha \in [0.80, 0.82, ..., 1.00]$ on training data
3. Compare: analytical vs grid-selected $\alpha$ per asset
4. Compute: $\Delta$CRPS and $\Delta$PIT MAD for each test asset
5. Add `crps_optimal_alpha` to tuning output for transparency

### Story 16.3: CRPS Decomposition into Reliability and Resolution

**As a** model diagnostician,
**I need** CRPS decomposed into reliability (calibration) and resolution (sharpness),
**So that** I can distinguish between mis-calibrated and uninformative forecasts.

**Acceptance Criteria**:
- [ ] Decomposition: $\text{CRPS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$
- [ ] Reliability near 0 indicates good calibration (target < 0.002)
- [ ] Resolution should be large (target > 0.010 for informative forecasts)
- [ ] Assets with high reliability and low resolution: model is calibrated but useless
- [ ] Assets with low reliability and high resolution: model is sharp but mis-calibrated
- [ ] Validated on: all 25 test assets

**Tasks**:
1. Implement Hersbach (2000) CRPS decomposition via binned PIT
2. Report: (Reliability, Resolution, Uncertainty) per asset
3. Diagnostic: flag assets where reliability > 0.003 (mis-calibrated)
4. Diagnostic: flag assets where resolution < 0.008 (uninformative)
5. Plot: reliability vs resolution scatter for all 25 assets

---

# PART VIII: CROSS-ASSET VALIDATION SUITE

## Epic 17: Large Cap Equity Accuracy Benchmarks

**Assets**: MSFT, GOOGL, NFLX, AAPL, NVDA
**All Files**: All 7 model files
**Priority**: HIGH -- large caps are the "easy" case; failure here is a fundamental problem

### Background

Large cap equities have well-documented statistical properties:
- Returns approximately normal with slight negative skew
- Volatility clustering (GARCH effects) with leverage effect
- Intraday seasonality washed out in daily returns
- Earnings announcements create predictable volatility spikes
- Tails heavier than Gaussian but lighter than small caps

These assets should be the EASIEST to model. Any model that fails on large caps has a
fundamental structural problem.

### Story 17.1: MSFT as Gaussian Baseline Benchmark

**As a** model validation system establishing baseline performance,
**I need** MSFT modeled to near-theoretical-optimum accuracy,
**So that** deviations on other assets can be attributed to asset-specific factors.

**Acceptance Criteria**:
- [ ] Gaussian model on MSFT: PIT KS p > 0.20, CRPS < 0.015
- [ ] Student-t model on MSFT: BIC improvement > 5 nats over Gaussian
- [ ] $\hat{\nu}_{MSFT} \in [8, 20]$ (moderate tails, not extreme)
- [ ] Momentum augmentation: BIC improvement > 2 nats over non-momentum
- [ ] All proper scoring rules within "elite" targets (defined in Scoring Protocol)
- [ ] Stable over rolling 1-year windows (no performance cliff)

**Tasks**:
1. Tune MSFT with all 7 model types: Gaussian, phi-Gaussian, Student-t variants
2. Record: BIC, CRPS, PIT KS, CSS, FEC for each model
3. Select BMA winner: verify it's either phi-Gaussian or phi-Student-t
4. Rolling analysis: compute scores on 12 rolling 1-year windows
5. Identify: worst window and root-cause the degradation

### Story 17.2: GOOGL Earnings Seasonality Detection

**As a** model fitting GOOGL with known quarterly earnings volatility,
**I need** the model to detect and adapt to earnings-induced volatility spikes,
**So that** post-earnings PIT is well-calibrated (not U-shaped from vol spike).

**Acceptance Criteria**:
- [ ] GARCH model captures vol spike within 1 day of earnings
- [ ] MS-q model: $p_{stress}$ > 0.8 during earnings week
- [ ] PIT during earnings weeks (4 per year): KS p > 0.10
- [ ] PIT during non-earnings periods: KS p > 0.20
- [ ] Overall model not dominated by earnings days (< 2% of sample)
- [ ] Validated on: GOOGL, NFLX, AFRM, UPST (known for earnings vol)

**Tasks**:
1. Identify earnings dates from data (vol > 2x average = proxy)
2. Compute PIT separately for earnings weeks and non-earnings periods
3. Verify GARCH/MS-q response timing
4. If earnings PIT is poor: consider earnings-aware vol scaling
5. Report: earnings vs non-earnings PIT comparison table

### Story 17.3: NVDA/AAPL Cross-Correlation Stability

**As a** model system fitting correlated large caps,
**I need** model parameters for NVDA and AAPL that are stable to shared factor movements,
**So that** the models capture idiosyncratic behavior, not just market beta.

**Acceptance Criteria**:
- [ ] NVDA and AAPL $\phi$ estimates differ by > 0.1 (idiosyncratic persistence)
- [ ] Residual correlation after Kalman filtering: $|\rho(\epsilon_{NVDA}, \epsilon_{AAPL})| < 0.4$
- [ ] During market-wide moves (SPY > 2%): both models increase $q$ (MS-q response)
- [ ] During NVDA-specific moves (NVDA > 3%, SPY < 1%): only NVDA model responds
- [ ] PIT for both assets: KS p > 0.15
- [ ] Validated on: NVDA, AAPL, AMZN, PLTR, with SPY as market factor

**Tasks**:
1. Tune NVDA and AAPL independently; record all parameters
2. Compute residual correlation: $\rho(\hat{\epsilon}_{NVDA}, \hat{\epsilon}_{AAPL})$
3. Conditional analysis: split into market-wide vs idiosyncratic moves
4. Verify MS-q response: both activate on SPY-driven events
5. Report: parameter comparison table (NVDA vs AAPL)

---

## Epic 18: Small Cap and High Volatility Accuracy

**Assets**: MSTR (MicroStrategy), RKLB (Rocket Lab)
**All Files**: All 7 model files
**Priority**: CRITICAL -- these are the hardest assets; accuracy here demonstrates robustness

### Background

MSTR and RKLB present extreme modeling challenges:
- **MSTR**: BTC-correlated, frequent 10%+ daily moves, gap risk, low float periods
- **RKLB**: Growth-stage space company, binary event risk, short history post-IPO

For these assets:
- Gaussian models should clearly LOSE to Student-t (tails matter)
- $\nu$ should be low (3-6) reflecting heavy tails
- MS-q ratio should be high (100-500x) reflecting extreme regime differences
- Momentum augmentation may be unreliable (noise dominates signal)

### Story 18.1: MSTR Tail Distribution Validation

**As a** model fitting MicroStrategy's extreme return distribution,
**I need** the tail model to capture 10%+ daily moves without numerical failure,
**So that** risk estimates are reliable during crypto-correlated events.

**Acceptance Criteria**:
- [ ] Student-t $\hat{\nu} \leq 4$ for MSTR (very heavy tails)
- [ ] CST contamination: $\hat{\epsilon} > 0.05$ (frequent crisis observations)
- [ ] 99th percentile return captured: model's 1% VaR covers realized 1% worst
- [ ] No NaN or Inf in Kalman trajectory during any historical period
- [ ] CRPS < 0.030 (higher than large caps, but still calibrated)
- [ ] Validated on: MSTR full history (2020-2026)

**Tasks**:
1. Tune MSTR with full model suite: Gaussian through CST
2. Record: $\hat{\nu}$, $\hat{\epsilon}$, BIC, CRPS, PIT per model
3. VaR backtest: compute 1% VaR, count violations (target: 1% $\pm$ 0.5%)
4. Stress test: filter through March 2020, March 2023 without numerical issues
5. Compare: Student-t vs Hansen vs CST on MSTR (which tail model wins?)

### Story 18.2: RKLB Short History Robustness

**As a** model fitting Rocket Lab with limited historical data,
**I need** regularization that prevents overfitting on small samples,
**So that** parameter estimates are stable despite N < 500.

**Acceptance Criteria**:
- [ ] With N = 300 (approximately 1 year post-IPO): all models produce valid parameters
- [ ] $\phi$ shrinkage pulls estimate toward prior (stronger effect with small N)
- [ ] $\nu$ prior: $\hat{\nu} \in [4, 12]$ (not extreme values driven by small sample)
- [ ] Cross-validation folds: at least 3 folds even with small N
- [ ] PIT KS p > 0.05 (relaxed threshold for small sample)
- [ ] Validated on: RKLB, with artificial truncation tests at N = 200, 300, 500

**Tasks**:
1. Tune RKLB with full available history
2. Truncation test: re-tune with first 200, 300, 500 observations only
3. Parameter stability: $|\theta_{200} - \theta_{500}| / \theta_{500} < 0.30$ for all parameters
4. Compare: PIT KS at each sample size (expect degradation at N = 200)
5. Identify: minimum sample size for reliable tuning (where PIT KS p > 0.05)

### Story 18.3: MSTR-BTC Regime Correlation

**As a** model fitting MSTR (BTC-proxy equity),
**I need** the model to capture MSTR's unique BTC correlation structure,
**So that** regime transitions in BTC propagate correctly to MSTR.

**Acceptance Criteria**:
- [ ] MS-q stress probability: $\text{corr}(p_{stress}^{MSTR}, p_{stress}^{BTC}) > 0.6$
- [ ] BTC crash events: MSTR model enters stress regime within 1 day
- [ ] BTC rally events: MSTR model maintains elevated uncertainty (no premature calm)
- [ ] Residual after BTC factor: $|\epsilon_{MSTR|BTC}|$ has lower kurtosis than raw MSTR
- [ ] PIT during BTC-driven events: KS p > 0.10
- [ ] Validated on: MSTR + BTC-USD parallel tuning

**Tasks**:
1. Tune MSTR and BTC-USD independently; extract $p_{stress}(t)$ trajectories
2. Compute: $\text{corr}(p_{stress}^{MSTR}, p_{stress}^{BTC})$ on overlapping period
3. Identify BTC crash dates; verify MSTR stress activation timing
4. Compute factor residual: $\epsilon_{MSTR|BTC} = r_{MSTR} - \hat{\beta} r_{BTC}$
5. Report: comparative regime analysis table

---

## Epic 19: Precious Metals Accuracy

**Assets**: GC=F (Gold futures), SI=F (Silver futures)
**All Files**: All 7 model files
**Priority**: HIGH -- metals have unique dynamics; existing profiles need validation

### Background

Precious metals exhibit distinct statistical properties:
- **Gold**: Safe haven, macro-driven, slow regimes, jump processes, low daily vol
- **Silver**: Industrial + precious hybrid, explosive VoV, leveraged-gold dynamics
- Both: Mean-reverting at multi-month horizons, trending at multi-year horizons
- Both: Not equity-correlated (diversification value in portfolio context)

The existing asset-class profiles (`metals_gold`, `metals_silver`) in `phi_student_t.py`
and `phi_student_t_unified.py` were calibrated in February 2026 and need validation.

### Story 19.1: Gold (GC=F) Low-Volatility Regime Calibration

**As a** model fitting gold during extended low-volatility periods,
**I need** the filter to maintain meaningful state uncertainty without collapsing,
**So that** sudden regime transitions (e.g., geopolitical shocks) are detected promptly.

**Acceptance Criteria**:
- [ ] During low-vol periods (vol < 10th percentile): $P_t > 10^{-8}$ (no state collapse)
- [ ] MS-q transition: from low-vol to high-vol within 3 days of shock
- [ ] PIT during low-vol periods: KS p > 0.15 (not over-confident)
- [ ] Gold profile parameters validated: ms_sensitivity = 4.0, ewm_lambda = 0.97
- [ ] CRPS during transition periods (vol increasing): < 0.020
- [ ] Validated on: GC=F (2020-2026), with focus on 2020 COVID and 2022 rate hikes

**Tasks**:
1. Identify low-vol periods in GC=F: vol < 10th percentile for > 20 days
2. Monitor $P_t$ trajectory during these periods: verify no collapse
3. Measure transition speed: days from shock to $p_{stress} > 0.8$
4. Compare: gold profile vs generic profile on PIT/CRPS during transitions
5. Report: regime timeline with $p_{stress}$ overlay

### Story 19.2: Silver (SI=F) Explosive Volatility Handling

**As a** model fitting silver with extreme vol-of-vol dynamics,
**I need** robust handling of volatility explosions (>3x average vol in < 5 days),
**So that** the model does not produce degenerate predictions during silver squeezes.

**Acceptance Criteria**:
- [ ] During vol explosions: $R_t$ scales appropriately (no under/over-estimation by > 2x)
- [ ] Silver profile: ms_sensitivity = 4.5, ewm_lambda = 0.94 (faster than gold)
- [ ] VoV enhancement: $\gamma_{VoV}$ active and contributing during explosions
- [ ] No GARCH persistence > 0.999 (prevent infinite variance forecast)
- [ ] CRPS during explosion periods: < 0.035 (higher than calm, but bounded)
- [ ] Validated on: SI=F (2020-2026), with focus on 2021 silver squeeze

**Tasks**:
1. Identify vol explosion events in SI=F: daily vol > 3x trailing 20-day average
2. Track $R_t$ during these events: is it proportional to realized vol?
3. Monitor GARCH persistence: if approaching 1.0, apply stationarity correction
4. Compare: silver profile vs generic profile during explosions
5. Report: explosion event timeline with $R_t$, $p_{stress}$, CRPS overlay

### Story 19.3: Gold-Silver Cross-Calibration Consistency

**As a** system modeling two correlated precious metals,
**I need** parameter estimates for GC=F and SI=F that are internally consistent,
**So that** the gold/silver ratio implied by the models is stable and realistic.

**Acceptance Criteria**:
- [ ] $\phi_{gold}$ and $\phi_{silver}$ have same sign (both mean-reverting or both trending)
- [ ] $\nu_{silver} \leq \nu_{gold}$ (silver has heavier tails -- empirical fact)
- [ ] MS-q stress periods: > 70% overlap between gold and silver stress detections
- [ ] Model-implied gold/silver vol ratio: within 20% of historical ratio
- [ ] BMA model selection: similar model class selected for both (e.g., both Student-t)
- [ ] Validated on: GC=F + SI=F parallel tuning (2020-2026)

**Tasks**:
1. Tune GC=F and SI=F independently; compare all parameter estimates
2. Compute: $\text{corr}(p_{stress}^{gold}, p_{stress}^{silver})$
3. Verify: $\nu_{SI} \leq \nu_{GC}$ for all BMA-selected models
4. Compute: model-implied vol ratio vs historical vol ratio
5. Report: parameter comparison table with consistency flags

---

## Epic 20: Cryptocurrency Accuracy

**Assets**: BTC-USD (Bitcoin)
**All Files**: All 7 model files
**Priority**: HIGH -- BTC is the most non-Gaussian asset in the universe

### Background

Bitcoin presents unique challenges not found in any other asset class:
- **24/7 trading**: No overnight gaps, but weekend liquidity differs
- **Structural breaks**: Halving cycles, regulatory events, exchange failures
- **Extreme kurtosis**: Daily returns have kurtosis > 10 (vs ~5 for equities)
- **Regime-dependent skewness**: Positive skew in bull markets, negative in bear
- **Non-stationarity**: Volatility has 10x range from calm to crisis
- **Correlation shifts**: Increasing equity correlation since 2020 (institutional adoption)

### Story 20.1: BTC-USD Tail Heaviness Calibration

**As a** model fitting Bitcoin's extreme tail behavior,
**I need** $\nu$ selection that captures kurtosis > 10,
**So that** 5%+ daily moves are not treated as impossible events.

**Acceptance Criteria**:
- [ ] $\hat{\nu}_{BTC} \in [2.5, 4.0]$ (very heavy tails)
- [ ] 99.5th percentile return: model probability > $10^{-4}$ (not treated as impossible)
- [ ] Hansen $\lambda$: potentially regime-dependent (bull vs bear skew)
- [ ] CRPS < 0.035 (hardest asset -- relaxed target)
- [ ] PIT KS p > 0.05 (relaxed threshold for extreme non-normality)
- [ ] Validated on: BTC-USD (2020-2026)

**Tasks**:
1. Estimate excess kurtosis from BTC-USD daily returns
2. Map kurtosis to expected $\nu$: $\hat{\nu} = \frac{2(\kappa + 3)}{\kappa - 3} + 4$ (moments method)
3. Compare: moments-estimated $\nu$ vs MLE-estimated $\nu$
4. Compute: probability of 10%+ daily move under fitted model
5. Report: tail probability table at 1%, 0.5%, 0.1% levels

### Story 20.2: BTC-USD Structural Break Detection

**As a** model fitting Bitcoin across halving cycles and regulatory events,
**I need** structural break detection that resets filter state appropriately,
**So that** pre-break parameters do not contaminate post-break inference.

**Acceptance Criteria**:
- [ ] Break detection via CUSUM test on log-likelihood sequence
- [ ] Known breaks detected: COVID crash (Mar 2020), China ban (May 2021),
      FTX collapse (Nov 2022), ETF approval (Jan 2024)
- [ ] At detected break: state uncertainty $P_t$ reset to initial value
- [ ] Post-break convergence: filter stabilizes within 20 observations
- [ ] BIC improvement > 5 nats when breaks are properly handled
- [ ] Validated on: BTC-USD (2020-2026)

**Tasks**:
1. Implement CUSUM break detection on standardized innovations
2. Threshold: $|\text{CUSUM}_t| > 4\sqrt{n}$ signals break
3. At break: set $P_t = P_0$ (reset uncertainty), keep $\mu_t$ (maintain level)
4. Re-estimate parameters on post-break data (or use robust running MLE)
5. Report: break timeline with CUSUM trajectory

### Story 20.3: BTC-USD Weekend/Weekday Volatility Differential

**As a** model for a 24/7 asset with liquidity variation,
**I need** day-of-week volatility adjustment,
**So that** low-liquidity weekend returns are not over-weighted in the likelihood.

**Acceptance Criteria**:
- [ ] Compute empirical weekend vs weekday vol ratio: $\rho = \sigma_{weekend} / \sigma_{weekday}$
- [ ] If $\rho > 1.2$: apply weekend scaling $R_{weekend} = \rho^2 \cdot R_{weekday}$
- [ ] PIT during weekends: KS p > 0.10 (not systematically miscalibrated)
- [ ] Weekday PIT: KS p > 0.15 (should be better calibrated)
- [ ] Overall CRPS improvement > 1% from day-of-week adjustment
- [ ] Validated on: BTC-USD (2020-2026)

**Tasks**:
1. Compute: mean absolute return by day of week (Mon-Sun for BTC)
2. Estimate: weekend/weekday vol ratio $\rho$
3. If $\rho > 1.2$: modify observation noise $R_t = R_t \times \rho^2$ on weekends
4. Compute: PIT separately for weekends and weekdays
5. Report: day-of-week return distribution and PIT table

---

## Epic 21: Index Fund Accuracy

**Assets**: SPY, QQQ, IWM
**All Files**: All 7 model files
**Priority**: HIGH -- indices are mean-reversion anchors and portfolio benchmarks

### Background

Index funds have the most "well-behaved" returns in the universe:
- SPY (S&P 500): Diversified, low idiosyncratic risk, mean-reverting at monthly horizon
- QQQ (Nasdaq 100): Tech-heavy, growth tilt, higher vol than SPY
- IWM (Russell 2000): Small cap index, highest vol, most kurtosis of the three

These assets serve as calibration anchors: if the model works on SPY, the framework is sound.

### Story 21.1: SPY as Universal Calibration Anchor

**As a** model validation system,
**I need** SPY to achieve best-in-class scores across all metrics,
**So that** SPY can serve as the reference point for all other assets.

**Acceptance Criteria**:
- [ ] SPY BIC: lowest (best) among all test assets
- [ ] SPY CRPS: < 0.012 (tightest forecasts for most liquid asset)
- [ ] SPY PIT KS p: > 0.30 (excellent calibration)
- [ ] SPY CSS: > 0.80 (stable even during crisis)
- [ ] SPY FEC: > 0.85 (entropy tracks uncertainty well)
- [ ] SPY Hyvarinen: < 300 (no variance collapse)
- [ ] If SPY fails any metric: this is a FRAMEWORK BUG, not an asset-specific issue

**Tasks**:
1. Tune SPY with all model variants (14+ models per regime)
2. Record all 7 scoring metrics
3. Verify elite targets are met for each metric
4. If any metric fails: diagnose and fix framework before proceeding
5. SPY becomes the "golden test" that must pass before any release

### Story 21.2: QQQ Tech-Sector Volatility Premium

**As a** model fitting QQQ with its tech-heavy composition,
**I need** the model to capture the growth premium and higher volatility,
**So that** QQQ forecasts are wider than SPY (reflecting true uncertainty).

**Acceptance Criteria**:
- [ ] QQQ predictive $\sigma > 1.2 \times$ SPY predictive $\sigma$ (QQQ is riskier)
- [ ] QQQ $\hat{\nu} \leq$ SPY $\hat{\nu}$ (QQQ has heavier tails)
- [ ] QQQ momentum augmentation: stronger edge than SPY (tech trends)
- [ ] QQQ CRPS: < 0.015 (slightly wider than SPY target)
- [ ] QQQ PIT KS p: > 0.20
- [ ] Validated on: QQQ vs SPY parallel comparison (2020-2026)

**Tasks**:
1. Tune QQQ and SPY with identical model suite
2. Compare: predictive $\sigma$ ratio across all time periods
3. Compare: $\hat{\nu}$ estimates (QQQ should be lower)
4. Compare: momentum model BIC advantage (QQQ should be larger)
5. Report: side-by-side scoring table (QQQ vs SPY)

### Story 21.3: IWM Small-Cap Index Tail Risk

**As a** model fitting IWM (Russell 2000) with small-cap tail risk,
**I need** the model to capture the higher kurtosis and weaker mean-reversion,
**So that** IWM risk estimates are appropriately wider than large-cap indices.

**Acceptance Criteria**:
- [ ] IWM $\hat{\nu} <$ SPY $\hat{\nu}$ (heavier tails)
- [ ] IWM $\hat{\phi}$ closer to 0 than SPY (weaker drift persistence)
- [ ] IWM MS-q: higher stress frequency than SPY (more regime switching)
- [ ] IWM CRPS: < 0.020 (wider than SPY, tighter than individual small caps)
- [ ] IWM PIT during March 2020: KS p > 0.05 (hardest period for small caps)
- [ ] Validated on: IWM (2020-2026)

**Tasks**:
1. Tune IWM; compare parameters against SPY and QQQ
2. Focus: March 2020 crisis period -- verify model survived
3. Compare: stress regime frequency (IWM should have most)
4. Verify: tail parameter ordering $\nu_{IWM} \leq \nu_{QQQ} \leq \nu_{SPY}$
5. Report: index comparison table with all metrics

---

# PART IX: ADVANCED MATHEMATICAL IMPROVEMENTS

## Epic 22: Score-Driven (GAS) Extensions

**Files**: `src/models/phi_student_t_unified.py`, `src/models/numba_kernels.py`
**Priority**: MEDIUM-HIGH -- GAS is the next frontier for adaptive filtering

### Background

Generalized Autoregressive Score (GAS) models (Creal, Koopman, Lucas 2013) update parameters
via the score of the conditional distribution:

$$\theta_{t+1} = \omega + A \cdot s_t + B \cdot \theta_t$$

where $s_t = S_t^{-1} \nabla_t$ is the scaled score (gradient of log-density wrt $\theta$).

This framework can be applied to ANY parameter: $\nu$ (tail), $\lambda$ (skewness),
$\sigma$ (scale), $\phi$ (drift). The key advantage: parameter updates are
**information-theoretic optimal** under the model assumption.

Current implementation has GAS skewness (Tier 3) but not GAS-$\nu$ or GAS-$\phi$.

### Story 22.1: GAS-Nu (Score-Driven Tail Thickness)

**As a** tail model that must adapt to time-varying tail behavior,
**I need** score-driven $\nu_t$ updating at each timestep,
**So that** tails automatically thicken during crisis and thin during calm markets.

**Acceptance Criteria**:
- [ ] GAS-$\nu$ update: $\nu_{t+1} = \omega_\nu + \alpha_\nu s_{\nu,t} + \beta_\nu \nu_t$
- [ ] Score: $s_{\nu,t} = \frac{\partial \log t(r_t; \nu_t)}{\partial \nu_t} / I(\nu_t)$
- [ ] Stationarity: $|\beta_\nu| < 1$ and $\nu_t \in [2.1, 50]$ enforced
- [ ] On MSTR: $\nu_t$ drops below 4 during March 2020 and recovers within 30 days
- [ ] BIC improvement > 3 nats over static $\nu$ for heavy-tailed assets
- [ ] Validated on: MSTR, BTC-USD, NFLX, SI=F

**Tasks**:
1. Derive: $\frac{\partial \log t(r; \nu)}{\partial \nu}$ analytically (involves digamma function)
2. Implement: Fisher information $I(\nu) = \frac{1}{2}[\psi'(\nu/2) - \psi'((\nu+1)/2) - 2(\nu+3)/(\nu(\nu+1)^2)]$
3. Optimize: $(\omega_\nu, \alpha_\nu, \beta_\nu)$ via L-BFGS-B
4. Constrain: $\nu_t \in [2.1, 50]$ via sigmoid transformation
5. Compare: GAS-$\nu$ vs static $\nu$ on all 25 test assets

### Story 22.2: GAS-Phi (Score-Driven Drift Persistence)

**As a** drift model that encounters varying market regimes,
**I need** score-driven $\phi_t$ that adapts persistence to the current regime,
**So that** trending markets get high $\phi$ and ranging markets get low $\phi$.

**Acceptance Criteria**:
- [ ] GAS-$\phi$ update: $\phi_{t+1} = \omega_\phi + \alpha_\phi s_{\phi,t} + \beta_\phi \phi_t$
- [ ] Score: $s_{\phi,t} = \frac{\mu_{t-1}(r_t - \mu_{pred,t})}{S_t}$ (innovation weighted by state)
- [ ] Constrained: $\phi_t \in [-0.5, 1.05]$ via sigmoid transformation
- [ ] On SPY: $\phi_t$ higher during trending months, lower during ranging
- [ ] On GC=F: $\phi_t$ remains near 0 (gold is mean-reverting)
- [ ] Validated on: SPY, GC=F, MSTR, BTC-USD, MSFT

**Tasks**:
1. Derive: score of Gaussian/Student-t likelihood wrt $\phi$
2. Implement: GAS recursion with sigmoid constraint
3. Optimize: $(\omega_\phi, \alpha_\phi, \beta_\phi)$ with regularization
4. Visualize: $\phi_t$ trajectory overlaid with price for SPY and GC=F
5. Compare: GAS-$\phi$ vs static $\phi$ on BIC/CRPS/PIT

### Story 22.3: GAS Score Computation in Numba

**As a** computation-critical score engine,
**I need** GAS score computation compiled to Numba,
**So that** GAS models run at the same speed as static-parameter models.

**Acceptance Criteria**:
- [ ] `gas_score_student_t_kernel(r, nu, mu, sigma)` compiled with `@njit`
- [ ] Digamma function approximation accurate to 1e-8 for $x > 1$
- [ ] Trigamma function approximation accurate to 1e-8 for $x > 1$
- [ ] GAS-$\nu$ filter: < 2x slowdown vs static-$\nu$ filter
- [ ] No scipy dependency in the inner loop
- [ ] Validated: Numba kernel matches pure-Python implementation to 1e-12

**Tasks**:
1. Implement Numba digamma: asymptotic expansion for $x > 5$, recursion for $x \leq 5$
2. Implement Numba trigamma: asymptotic expansion for $x > 5$, recursion for $x \leq 5$
3. Compile GAS score kernel with `@njit(fastmath=False)` (precision matters)
4. Benchmark: ns/timestep for GAS kernel vs static kernel
5. Test: digamma/trigamma against scipy on $x \in [0.5, 1, 2, 5, 10, 50]$

---

## Epic 23: Jump-Diffusion Integration

**Files**: `src/models/phi_student_t_unified.py`, `src/models/numba_kernels.py`
**Priority**: MEDIUM -- deferred to after MS-q stabilizes, but architecturally important

### Background

The Merton jump-diffusion model extends continuous diffusion with discrete jumps:

$$dS_t / S_t = (\mu - \lambda_J \bar{k}) dt + \sigma dW_t + J_t dN_t$$

where:
- $N_t$ is a Poisson process with intensity $\lambda_J$
- $J_t \sim N(\mu_J, \sigma_J^2)$ is the jump size
- $\bar{k} = e^{\mu_J + \sigma_J^2/2} - 1$ is the compensator

In discrete time (daily returns):
$$r_t = \mu_t + \sigma_t \epsilon_t + \sum_{j=1}^{N_t} J_j$$

where $N_t \sim \text{Poisson}(\lambda_J)$ and $J_j \sim N(\mu_J, \sigma_J^2)$.

### Story 23.1: Poisson Jump Detection in Daily Returns

**As a** jump model identifying discontinuous price moves,
**I need** Bayesian posterior probability of jump occurrence at each timestep,
**So that** the filter can distinguish between diffusion outliers and genuine jumps.

**Acceptance Criteria**:
- [ ] Posterior jump probability: $p(N_t > 0 | r_t, \theta) > 0.5$ for genuine jumps
- [ ] On MSTR: identify > 80% of days with $|r_t| > 5\%$ as probable jumps
- [ ] On SPY: < 5% of days identified as jumps (mostly diffusion)
- [ ] On GC=F: jumps on geopolitical shock days (validate against known events)
- [ ] False positive rate: < 10% (non-jump days misidentified as jumps)
- [ ] Validated on: MSTR, SPY, GC=F, BTC-USD, SI=F

**Tasks**:
1. Compute: $p(N_t = 1|r_t) \propto \lambda_J \cdot N(r_t; \mu_t + \mu_J, \sigma_t^2 + \sigma_J^2)$
2. Normalize: $p(N_t = 0|r_t) + p(N_t = 1|r_t) \approx 1$ (ignore $N_t \geq 2$)
3. Estimate: $(\lambda_J, \mu_J, \sigma_J)$ via EM algorithm on returns
4. Validate: jump dates against known events (earnings, crashes, geopolitical)
5. Report: jump probability trajectory for each test asset

### Story 23.2: Jump-Diffusion Likelihood in Kalman Framework

**As a** Kalman filter incorporating jump dynamics,
**I need** a modified likelihood that accounts for jump possibility,
**So that** the model does not mis-attribute jumps to diffusion tail events.

**Acceptance Criteria**:
- [ ] Modified log-likelihood: $\ell_t = \log[(1-\lambda_J)f_{diff}(r_t) + \lambda_J f_{jump}(r_t)]$
- [ ] $f_{diff} = N(r_t; \mu_{pred}, S_t)$ (standard Kalman prediction)
- [ ] $f_{jump} = N(r_t; \mu_{pred} + \mu_J, S_t + \sigma_J^2)$ (jump-augmented)
- [ ] BIC accounts for 3 extra parameters $(\lambda_J, \mu_J, \sigma_J)$
- [ ] On MSTR: jump-diffusion BIC beats Student-t by > 5 nats
- [ ] On SPY: jump-diffusion does NOT beat Student-t (jumps rare in SPY)
- [ ] Validated on: MSTR, SPY, BTC-USD, NFLX

**Tasks**:
1. Implement mixture log-likelihood in Numba kernel
2. Modified Kalman update: weighted average of diffusion and jump updates
3. Estimate parameters: EM on complete data (E-step: jump posterior, M-step: MLE)
4. BIC comparison: jump-diffusion vs standard Student-t on all test assets
5. Report: when does jump-diffusion provide genuine improvement?

### Story 23.3: Jump Size Distribution for Different Asset Classes

**As a** jump model adapting to asset-specific jump characteristics,
**I need** calibrated jump parameters per asset class,
**So that** gold jumps (geopolitical) differ from equity jumps (earnings).

**Acceptance Criteria**:
- [ ] Equity jumps: $\mu_J < 0$ (downward bias), $\sigma_J \approx 0.03$-$0.05$
- [ ] Gold jumps: $\mu_J \approx 0$ (symmetric), $\sigma_J \approx 0.02$-$0.03$
- [ ] BTC jumps: $\mu_J$ varies by regime, $\sigma_J \approx 0.05$-$0.10$
- [ ] MSTR jumps: $\mu_J \approx 0$ (symmetric BTC exposure), $\sigma_J \approx 0.08$-$0.15$
- [ ] Jump intensity: $\lambda_J$ matches empirical frequency of extreme moves
- [ ] Validated on: MSTR, GC=F, BTC-USD, NFLX, AAPL

**Tasks**:
1. Estimate jump parameters via EM for each test asset
2. Group by asset class and compute class-average parameters
3. Validate: class-average as prior, asset-specific as posterior
4. Compare: using class-average vs asset-specific jump parameters
5. Report: jump parameter table per asset class

---

# PART X: OBSERVATION NOISE AND VARIANCE DYNAMICS

## Epic 24: Chi-Squared EWMA Online Scale Adaptation

**Files**: `src/models/phi_student_t.py`, `src/models/phi_student_t_unified.py`
**Priority**: HIGH -- online scale adaptation is the last line of defense against misspecification

### Background

After the Kalman filter produces innovations $e_t = r_t - \mu_{pred,t}$, the chi-squared
EWMA tracks the ratio $\chi^2_t = e_t^2 / S_t$ and uses it to adapt the observation noise
scale. In a correctly specified model, $\mathbb{E}[\chi^2_t] = 1$.

The EWMA tracker with dead-zone:
$$\bar{\chi}^2_t = \lambda \bar{\chi}^2_{t-1} + (1-\lambda) \chi^2_t$$

If $\bar{\chi}^2_t \notin [d_{lo}, d_{hi}]$: scale correction $c_{adj} = 1 / \bar{\chi}^2_t$.

Current implementation uses fixed dead-zone bounds. Asset-class customization of $\lambda$ and
dead-zone improves responsiveness.

### Story 24.1: Adaptive Dead-Zone Bounds Based on Sample Kurtosis

**As a** chi-squared EWMA tracker,
**I need** dead-zone bounds that reflect the tail heaviness of the innovation distribution,
**So that** heavy-tailed assets don't trigger constant correction (noise, not signal).

**Acceptance Criteria**:
- [ ] For Gaussian innovations ($\kappa = 3$): dead-zone $[0.30, 0.55]$ (current default)
- [ ] For Student-t($\nu=4$) innovations ($\kappa \approx 9$): dead-zone $[0.20, 0.65]$ (wider)
- [ ] For Student-t($\nu=20$) innovations ($\kappa \approx 3.3$): dead-zone $[0.28, 0.58]$
- [ ] Dead-zone width formula: $d_{width} = 0.25 + 0.05 \cdot (\hat{\kappa} - 3)$
- [ ] Correction frequency: 5-15% of timesteps (not too rare, not too frequent)
- [ ] Validated on: SPY (Gaussian-like), MSTR (heavy-tailed), GC=F (moderate)

**Tasks**:
1. Compute sample excess kurtosis from innovations
2. Set dead-zone: $d_{lo} = 0.425 - d_{width}/2$, $d_{hi} = 0.425 + d_{width}/2$
3. Monitor correction frequency per asset
4. Compare: fixed vs adaptive dead-zone on PIT/CRPS
5. Report: correction frequency and dead-zone bounds per asset

### Story 24.2: Chi-Squared EWMA Lambda Selection

**As a** scale tracker choosing between fast and slow adaptation,
**I need** $\lambda$ selected based on asset class volatility dynamics,
**So that** the tracker responds at the appropriate timescale.

**Acceptance Criteria**:
- [ ] Gold: $\lambda = 0.98$ (slow, ~50-day half-life)
- [ ] Silver: $\lambda = 0.95$ (moderate, ~20-day half-life)
- [ ] Large cap equity: $\lambda = 0.97$ (standard, ~33-day half-life)
- [ ] High vol equity: $\lambda = 0.94$ (fast, ~16-day half-life)
- [ ] Crypto: $\lambda = 0.93$ (fastest, ~14-day half-life)
- [ ] Half-life formula: $h = -\log(2) / \log(\lambda)$
- [ ] Validated on: GC=F, SI=F, MSFT, MSTR, BTC-USD

**Tasks**:
1. Compute empirical vol half-life for each asset: autocorrelation of $|r_t|$
2. Set $\lambda$ to match vol half-life: $\lambda = e^{-\log(2)/h}$
3. Compare: asset-class $\lambda$ vs optimal empirical $\lambda$
4. Run: PIT comparison with fixed $\lambda = 0.97$ vs adaptive $\lambda$
5. Report: empirical half-life, selected $\lambda$, PIT improvement per asset

### Story 24.3: VoV Enhancement Interaction with GARCH

**As a** model with both VoV and GARCH active,
**I need** damping that prevents double-counting of volatility dynamics,
**So that** the combined observation noise is calibrated (not inflated).

**Acceptance Criteria**:
- [ ] When GARCH active and VoV active: damping $\geq 0.3$ (reduce VoV contribution)
- [ ] When GARCH active and VoV inactive: no damping needed
- [ ] When GARCH inactive and VoV active: full VoV contribution (damping = 0)
- [ ] Total variance inflation $\leq 1.5$ during calm periods
- [ ] Total variance inflation $\leq 3.0$ during crisis periods
- [ ] Validated on: MSTR, NFLX, BTC-USD, GC=F, SPY

**Tasks**:
1. Implement damping logic: `if garch_active and vov_active: gamma_eff = gamma * (1 - damp)`
2. Monitor: total variance inflation ratio $R_t / R_{base}$ across time
3. Verify: inflation bounded by acceptance criteria values
4. Compare: with and without damping on PIT/CRPS
5. Report: variance inflation trajectory for crisis assets (MSTR, BTC-USD)

---

## Epic 25: Observation Noise Scale Parameter (c) Optimization

**Files**: `src/models/phi_gaussian.py`, `src/models/phi_student_t.py`, `src/models/gaussian.py`
**Priority**: HIGH -- c is the most fundamental parameter; small errors cascade everywhere

### Background

The scale parameter $c$ converts EWMA volatility to observation noise:
$$R_t = c \cdot \sigma^2_{EWMA}(t) \cdot \frac{\nu - 2}{\nu}$$

If $c > 1$: model assumes EWMA underestimates vol (adds noise)
If $c < 1$: model assumes EWMA overestimates vol (removes noise)
If $c = 1$: model trusts EWMA as-is

The optimization of $c$ is coupled with $q$ (process noise) because both affect the
innovation variance $S_t = \phi^2 P_{t-1} + q + R_t$.

### Story 25.1: Decoupled c and q Optimization

**As a** parameter optimizer fitting $(c, q)$ jointly,
**I need** a parameterization that decouples $c$ and $q$,
**So that** the optimizer does not trade off observation noise for process noise.

**Acceptance Criteria**:
- [ ] Signal-to-noise ratio: $\text{SNR} = q / (c \cdot \bar{\sigma}^2)$ as diagnostic
- [ ] $\text{SNR} \in [10^{-4}, 10^{-1}]$ for all test assets (if outside: flag)
- [ ] Optimization in $(\log_{10}(q), \log_{10}(c))$ space (decouples magnitudes)
- [ ] Correlation of $\hat{c}$ and $\hat{q}$ estimates: $|\rho| < 0.5$ (low coupling)
- [ ] BIC unchanged vs current joint optimization (same quality, better conditioning)
- [ ] Validated on: all 25 test assets

**Tasks**:
1. Reparameterize: optimize $(\log_{10}(q), \log_{10}(c))$ instead of $(q, c)$
2. Compute SNR per asset; verify within expected range
3. Measure: $\text{corr}(\hat{q}, \hat{c})$ via bootstrap on each asset
4. If $|\rho| > 0.5$: try orthogonal parameterization $(\text{SNR}, c)$
5. Report: SNR, coupling coefficient, BIC per asset

### Story 25.2: Scale Parameter Regularization

**As a** Bayesian optimizer with a prior on $c$,
**I need** a regularization prior that prevents extreme $c$ values,
**So that** the model does not over-correct the EWMA volatility estimate.

**Acceptance Criteria**:
- [ ] Prior: $\log_{10}(c) \sim N(\log_{10}(0.9), 0.3^2)$ (centered near 1, mildly regularized)
- [ ] Effect: $c \in [0.3, 3.0]$ with 95% probability under prior
- [ ] Extreme $c$ (< 0.1 or > 10) effectively impossible under prior
- [ ] For SPY: $\hat{c} \approx 1.0 \pm 0.2$ (EWMA is accurate for liquid assets)
- [ ] For MSTR: $\hat{c} > 1.0$ allowed (EWMA may underestimate extreme vol)
- [ ] Validated on: SPY, MSTR, GC=F, BTC-USD, MSFT

**Tasks**:
1. Add log-Gaussian prior on $c$ to negative log-likelihood
2. Tune prior strength: $\lambda_c = 0.1 / n_{obs}$ (decreasing with sample size)
3. Compare: regularized vs unregularized $\hat{c}$ on all test assets
4. Verify: no extreme $c$ values in any test asset after regularization
5. Report: $\hat{c}$ distribution across test universe, before and after

---

# PART XI: COMPREHENSIVE TESTING FRAMEWORK

## Epic 26: Automated Accuracy Regression Testing

**Files**: All 7 model files
**Priority**: CRITICAL -- without regression tests, improvements today become regressions tomorrow

### Background

The model suite currently has unit tests for individual functions but lacks a comprehensive
accuracy regression test that validates end-to-end scoring across the full test universe.
Every code change risks subtle accuracy degradation that is invisible without systematic testing.

### Story 26.1: Golden Score Registry

**As a** CI/CD pipeline preventing accuracy regressions,
**I need** a registry of expected scores for each (asset, model, metric) combination,
**So that** any code change that degrades accuracy by > 1% is automatically flagged.

**Acceptance Criteria**:
- [ ] Registry covers: 25 assets x 7 model types x 6 metrics = 1050 score entries
- [ ] Tolerance: BIC $\pm$ 5 nats, CRPS $\pm$ 5%, PIT KS $\pm$ 0.02, CSS $\pm$ 0.05
- [ ] Any score outside tolerance: test FAILS with specific diagnostic
- [ ] Registry updated after validated improvements (manual approval required)
- [ ] Test runs in < 10 minutes on standard hardware (use cached tuning results)
- [ ] Validated on: initial run on all 25 test assets, all 7 model types

**Tasks**:
1. Create `test_golden_scores.json` with baseline scores from current codebase
2. Implement `test_accuracy_regression.py` that loads registry and compares
3. Add tolerance per metric (different metrics have different sensitivity)
4. Run: full test suite on all 25 assets; verify all pass at current baseline
5. Integrate: add to `make tests` target

### Story 26.2: Cross-Asset Consistency Tests

**As a** model validation framework,
**I need** tests that verify expected cross-asset parameter relationships,
**So that** the models respect known financial relationships.

**Acceptance Criteria**:
- [ ] $\nu_{MSTR} \leq \nu_{SPY}$ (heavy-tailed assets have lower nu)
- [ ] $\sigma_{QQQ} \geq \sigma_{SPY}$ (QQQ is riskier than SPY)
- [ ] $\sigma_{IWM} \geq \sigma_{SPY}$ (small caps are riskier than large caps)
- [ ] $\nu_{silver} \leq \nu_{gold}$ (silver has heavier tails)
- [ ] $\phi_{gold} \leq \phi_{equity}$ (metals are more mean-reverting)
- [ ] Violations flagged as warnings (not failures -- relationships may change)

**Tasks**:
1. Define cross-asset relationship rules in `test_cross_asset.py`
2. Run tuning on all 25 assets; extract key parameters
3. Verify each rule; report violations
4. Investigate violations: are they genuine or model bugs?
5. Update rules annually as market structure evolves

### Story 26.3: Synthetic Data Validation Suite

**As a** model developer needing ground-truth verification,
**I need** tests on synthetic data where the true DGP (data generating process) is known,
**So that** model accuracy can be measured against absolute truth (not just relative metrics).

**Acceptance Criteria**:
- [ ] Gaussian DGP: model recovers $(q, c, \phi)$ to within 10% of true values
- [ ] Student-t DGP ($\nu = 5$): model recovers $\hat{\nu} \in [4, 6]$
- [ ] Regime-switching DGP: MS-q detects regime transitions with < 5-day lag
- [ ] Jump DGP: jump model identifies > 80% of injected jumps
- [ ] PIT is exactly uniform (KS p > 0.50) under correct model specification
- [ ] Sample sizes: N = 500, 1000, 2000

**Tasks**:
1. Implement DGP generators: Gaussian, Student-t, regime-switching, jump
2. For each DGP: generate 100 realizations, tune model, record parameter estimates
3. Compute: bias, RMSE, coverage of 95% CI for each parameter
4. Verify: PIT uniformity under correct specification
5. Report: parameter recovery table (DGP x model x parameter)

---

## Epic 27: Performance Profiling and Optimization

**Files**: `src/models/numba_kernels.py`, `src/models/numba_wrappers.py`
**Priority**: MEDIUM -- speed enables more extensive testing and faster iteration

### Background

The tuning pipeline processes 100+ assets, each requiring 14+ model fits across 5 regimes.
Total computation: ~1000 Kalman filter runs per asset. At current speeds (~50ms per 2-year
filter), this takes ~50 seconds per asset and ~90 minutes for the full universe.

Key bottlenecks:
1. **Numba compilation**: First call to each kernel incurs JIT overhead (~2-5 seconds)
2. **Array preparation**: `prepare_arrays()` called 71K+ times -- microsecond optimization matters
3. **Gamma precomputation**: Called O(1) per $\nu$, but cache misses expensive
4. **L-BFGS-B optimization**: 50-200 iterations, each requiring a full filter pass

### Story 27.1: AOT Compilation for Numba Kernels

**As a** production system needing predictable startup time,
**I need** Ahead-of-Time (AOT) compiled Numba kernels,
**So that** first-call latency is eliminated and tuning starts immediately.

**Acceptance Criteria**:
- [ ] AOT compilation of: `phi_student_t_filter_kernel`, `gaussian_filter_kernel`,
      `phi_gaussian_filter_kernel`, `garch_variance_kernel`, `crps_student_t_kernel`
- [ ] First-call latency: < 100ms (vs current ~3 seconds for JIT)
- [ ] No accuracy change (AOT produces identical results to JIT)
- [ ] Compilation step added to `make setup` (one-time cost)
- [ ] Fallback to JIT if AOT cache is stale
- [ ] Validated on: full 25-asset tuning pipeline

**Tasks**:
1. Create AOT compilation script: `compile_numba_kernels.py`
2. Define type signatures for each kernel (explicit, not inferred)
3. Add to `make setup`: `python compile_numba_kernels.py`
4. Benchmark: first-call latency with AOT vs JIT
5. Add cache invalidation: recompile when source files change

### Story 27.2: Batch Filter Optimization for BMA

**As a** BMA system running the same filter with 4 different $\nu$ values,
**I need** a batch filter that shares Kalman prediction across $\nu$ values,
**So that** redundant computation is eliminated.

**Acceptance Criteria**:
- [ ] Batch filter: single prediction step, 4 parallel update steps (one per $\nu$)
- [ ] Speedup: > 2x for 4-$\nu$ batch vs 4 sequential filter runs
- [ ] No accuracy change: batch results identical to sequential (verified to 1e-14)
- [ ] Memory: batch uses < 2x memory of single filter (shared prediction arrays)
- [ ] Validated on: SPY, MSTR (verify correctness on easy and hard assets)

**Tasks**:
1. Refactor: separate prediction step (shared) from update step ($\nu$-specific)
2. Implement `phi_student_t_filter_batch_kernel()` in `numba_kernels.py`
3. Wrapper: `run_phi_student_t_filter_batch()` in `numba_wrappers.py`
4. Verify: batch output matches sequential output to 1e-14 on all test assets
5. Benchmark: wall-clock time for batch vs sequential on 2-year daily data

### Story 27.3: Vectorized Multi-Asset Tuning

**As a** pipeline processing 100+ assets,
**I need** vectorized operations across assets (not just within assets),
**So that** CPU caches are utilized efficiently and parallel overhead is minimized.

**Acceptance Criteria**:
- [ ] Group assets by similar length (within 10%): process as batch
- [ ] Batch optimization: shared L-BFGS-B workspace reduces memory allocation
- [ ] Speedup: > 1.5x for batch vs sequential processing of 25 test assets
- [ ] Correctness: per-asset results identical whether processed in batch or individually
- [ ] Memory: batch peak memory < 2x sequential peak memory
- [ ] Validated on: full 25-asset test universe

**Tasks**:
1. Sort assets by series length; group into batches of similar length
2. Allocate shared workspace: pre-allocate arrays for largest batch
3. Run batch: iterate within batch using vectorized operations where possible
4. Compare: per-asset results from batch vs sequential
5. Benchmark: total wall-clock time for 25-asset batch vs sequential

---

# PART XII: END-TO-END ACCURACY VALIDATION

## Epic 28: Walk-Forward Out-of-Sample Validation

**Files**: All 7 model files
**Priority**: CRITICAL -- in-sample accuracy is meaningless without OOS validation

### Background

All previous epics measure in-sample or cross-validated accuracy. The ultimate test is
**true out-of-sample** (OOS) performance: fit the model on historical data, generate
forecasts for the next period, and measure accuracy on unseen data.

Walk-forward validation:
1. Train on $[1, T_{train}]$
2. Forecast $[T_{train}+1, T_{train}+H]$
3. Measure accuracy on forecast period
4. Slide window forward by $S$ days
5. Repeat

### Story 28.1: 1-Day Ahead OOS Forecast Accuracy

**As a** signal system generating daily trading decisions,
**I need** 1-day ahead forecast accuracy measured rigorously,
**So that** I can trust the signal quality for actual position sizing.

**Acceptance Criteria**:
- [ ] Walk-forward: train on 500+ days, forecast 1 day, slide by 1 day
- [ ] OOS CRPS within 10% of in-sample CRPS for 80%+ of test assets
- [ ] OOS PIT KS p > 0.05 for 80%+ of test assets
- [ ] Direction accuracy (sign of mean): > 52% for SPY, > 50% for all assets
- [ ] No asset with OOS CRPS > 2x in-sample CRPS (sign of overfitting)
- [ ] Validated on: all 25 test assets

**Tasks**:
1. Implement walk-forward loop: train on expanding window, forecast 1 day
2. Collect: OOS PIT values, OOS CRPS, OOS direction accuracy
3. Compare: OOS vs in-sample metrics per asset
4. Flag: assets with > 20% OOS degradation
5. Report: OOS performance table with degradation percentages

### Story 28.2: 7-Day and 30-Day Horizon OOS Accuracy

**As a** signal system generating multi-day position recommendations,
**I need** 7-day and 30-day forecast accuracy validated out-of-sample,
**So that** multi-horizon signals are trustworthy.

**Acceptance Criteria**:
- [ ] 7-day OOS CRPS: within 20% of 1-day CRPS (scaled by $\sqrt{7}$)
- [ ] 30-day OOS CRPS: within 30% of 1-day CRPS (scaled by $\sqrt{30}$)
- [ ] Prediction interval coverage: 95% interval covers 93-97% of outcomes
- [ ] Multi-horizon forecasts use $\phi^H$ decay (not flat extrapolation)
- [ ] Variance growth formula validated: actual variance growth matches predicted
- [ ] Validated on: SPY, GC=F, BTC-USD, MSTR, MSFT

**Tasks**:
1. Walk-forward with H=7 and H=30: forecast distribution at each horizon
2. Measure: CRPS at each horizon, prediction interval coverage
3. Compare: predicted vs realized variance growth from $t$ to $t+H$
4. Verify: $\text{Var}(r_{t:t+H}) \approx$ predicted $\sum_{j=0}^{H-1} \phi^{2j}q + R$
5. Report: horizon x asset accuracy table

### Story 28.3: Crisis-Period OOS Performance

**As a** system that must perform during extreme market conditions,
**I need** OOS accuracy validated specifically during crisis periods,
**So that** the models don't fail when they matter most.

**Acceptance Criteria**:
- [ ] COVID crash (March 2020): OOS CRPS < 2x normal-period CRPS
- [ ] SVB crisis (March 2023): OOS PIT KS p > 0.01 (relaxed but not degenerate)
- [ ] Rate shock (Oct 2023): MS-q activates within 3 days in OOS mode
- [ ] No model produces NaN or Inf during any crisis period
- [ ] CSS (Calibration Stability Under Stress) > 0.60 for all assets during crises
- [ ] Validated on: SPY, MSTR, BTC-USD, GC=F, NFLX, CRWD, UPST

**Tasks**:
1. Identify crisis windows: March 2020, March 2023, October 2023
2. Run walk-forward with forecast periods covering each crisis
3. Measure: OOS CRPS, PIT, CSS during each crisis window
4. Compare: crisis-period vs normal-period OOS performance ratio
5. Report: crisis performance dashboard per asset

---

## Epic 29: Model Comparison and Selection Accuracy

**Files**: All 7 model files, `src/models/vectorized_ops.py` (BMA weights)
**Priority**: HIGH -- BMA is only as good as the model selection mechanism

### Background

BMA weights are computed from BIC scores:
$$w_i = \frac{\exp(-0.5 \cdot \text{BIC}_i)}{\sum_j \exp(-0.5 \cdot \text{BIC}_j)}$$

The accuracy of model selection depends on:
1. BIC being a reliable proxy for out-of-sample performance
2. The model pool containing the correct (or close to correct) model
3. BMA weights being stable across time (not fluctuating wildly)

### Story 29.1: BIC vs OOS Log-Likelihood Correlation

**As a** model selection system using BIC as a proxy for OOS performance,
**I need** BIC to correlate strongly with true OOS log-likelihood,
**So that** BIC-selected models are genuinely the best out-of-sample.

**Acceptance Criteria**:
- [ ] Spearman rank correlation: $\rho(\text{BIC rank}, \text{OOS LL rank}) > 0.7$ for all assets
- [ ] BIC-best model matches OOS-best model in > 70% of time periods
- [ ] When BIC disagrees with OOS: BIC-selected model is within 2 nats of OOS-best
- [ ] Correlation stable across assets: std of $\rho$ across 25 assets < 0.15
- [ ] Validated on: all 25 test assets with rolling 1-year windows

**Tasks**:
1. For each rolling window: compute BIC and OOS LL for all models
2. Rank models by BIC and by OOS LL; compute Spearman $\rho$
3. Count: agreement rate (BIC-best = OOS-best)
4. When disagreement: measure BIC-selected model's OOS LL gap
5. Report: correlation table per asset, average across universe

### Story 29.2: BMA Weight Stability Over Time

**As a** signal system relying on BMA-weighted forecasts,
**I need** BMA weights that are stable (not fluctuating wildly between months),
**So that** signal characteristics are predictable and tradeable.

**Acceptance Criteria**:
- [ ] Monthly BMA weight change: $\sum_i |w_{i,t} - w_{i,t-1}| < 0.30$ (turnover < 30%)
- [ ] Dominant model (highest weight) switches < 3 times per year
- [ ] No model goes from $w > 0.3$ to $w < 0.05$ in a single month
- [ ] Weight stability metric: $\text{Var}(w_{i,t})$ averaged across models < 0.02
- [ ] Validated on: SPY, MSTR, GC=F, BTC-USD (diverse stability patterns)

**Tasks**:
1. Run monthly re-tuning on 2-year history for each test asset
2. Record BMA weights at each month
3. Compute: monthly turnover, switch frequency, variance of weights
4. Visualize: BMA weight evolution over 24 months for each test asset
5. Identify: assets with unstable weights and diagnose root cause

### Story 29.3: Model Pool Completeness Assessment

**As a** BMA system with 14+ candidate models,
**I need** evidence that the model pool covers the relevant distributional space,
**So that** no important model class is missing from the competition.

**Acceptance Criteria**:
- [ ] Model pool spans: Gaussian, Student-t ($\nu = 3,4,8,20$), Hansen skew-t, CST,
      each with and without momentum -- at minimum 14 models
- [ ] For each asset: BMA-best model achieves CRPS within 5% of individual-model-best
- [ ] Pool coverage: no single model dominates > 50% of assets (diverse selection)
- [ ] Missing model test: remove best model, verify 2nd-best is within 10% of best CRPS
- [ ] Validated on: all 25 test assets

**Tasks**:
1. Enumerate: full model pool with parameter counts
2. For each asset: find individual-model-best (oracle) and BMA-best
3. Compute: $\Delta$CRPS = BMA vs oracle, should be < 5%
4. Model dominance analysis: count how many assets each model wins
5. Robustness: remove each model in turn, measure BMA degradation

---

## Epic 30: Integrated Accuracy Dashboard

**Files**: All 7 model files + test infrastructure
**Priority**: MEDIUM-HIGH -- visibility enables continuous improvement

### Background

The final deliverable is an integrated dashboard that shows, at a glance, the accuracy
state of the entire model suite across the test universe. This dashboard should be
generated automatically and updated weekly.

### Story 30.1: Accuracy Heatmap (Asset x Metric)

**As a** quant team monitoring model quality,
**I need** a heatmap showing (asset, metric) accuracy with color-coded status,
**So that** degradations are immediately visible.

**Acceptance Criteria**:
- [ ] Heatmap: 15 rows (assets) x 6 columns (BIC, CRPS, PIT, CSS, FEC, Hyvarinen)
- [ ] Color: green (elite), yellow (acceptable), red (failing)
- [ ] Thresholds per metric defined in Scoring Protocol section
- [ ] Generated as Rich table (console) and HTML (web dashboard)
- [ ] Updated weekly via `make accuracy-report`
- [ ] Validated on: current codebase produces heatmap without errors

**Tasks**:
1. Implement `accuracy_dashboard.py` that tunes all 25 assets and scores
2. Generate Rich table with color-coded cells
3. Generate HTML version for web dashboard
4. Add `make accuracy-report` target to Makefile
5. Store historical snapshots for trend analysis

### Story 30.2: Accuracy Trend Over Time

**As a** team tracking improvement velocity,
**I need** historical accuracy trends (weekly snapshots),
**So that** I can verify that code changes improve (not degrade) overall accuracy.

**Acceptance Criteria**:
- [ ] Weekly snapshots stored in `src/data/accuracy_history/`
- [ ] Trend plot: average CRPS across test universe over time
- [ ] Trend plot: number of "red" cells in heatmap over time
- [ ] Alert: if average CRPS degrades by > 5% from previous week
- [ ] Historical data: at least 12 weeks before trend analysis is meaningful
- [ ] Generated via `make accuracy-trend`

**Tasks**:
1. Store weekly snapshot: `accuracy_{date}.json` with all (asset, metric) scores
2. Implement trend computation: rolling 4-week average CRPS
3. Implement alert: compare current vs previous week
4. Visualize: CRPS trend line with weekly data points
5. Add `make accuracy-trend` target

### Story 30.3: Per-File Accuracy Attribution

**As a** developer understanding which file changes affect which metrics,
**I need** attribution linking accuracy changes to specific file modifications,
**So that** I know which file to investigate when accuracy degrades.

**Acceptance Criteria**:
- [ ] Attribution: when CRPS degrades on MSTR, identify which file change caused it
- [ ] Methodology: bisect-style testing on recent git commits
- [ ] Granularity: per-file, per-function attribution when possible
- [ ] False positive rate: < 20% (attribution correctly identifies causal file)
- [ ] Useful for: debugging regressions, validating improvements
- [ ] Validated on: simulate file change, verify attribution catches it

**Tasks**:
1. Implement `accuracy_bisect.py`: binary search over git commits for accuracy change
2. For each candidate commit: re-tune selected assets, compare scores
3. Identify: commit (and file) that caused the change
4. Report: "CRPS on MSTR degraded by 8% due to change in phi_student_t.py at commit abc123"
5. Integrate with CI: auto-run on accuracy regression detection

---

# APPENDIX A: MATHEMATICAL REFERENCE

## Key Distribution Formulas

### Student-t Distribution

PDF: $f(x|\nu,\mu,\sigma) = \frac{\Gamma(\frac{\nu+1}{2})}{\sigma\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})} \left(1 + \frac{(x-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$

CDF: $F(x|\nu) = I_{x(t)}\left(\frac{\nu}{2}, \frac{1}{2}\right)$ where $x(t) = \frac{\nu}{\nu + t^2}$

Variance: $\text{Var}(X) = \frac{\nu}{\nu-2}\sigma^2$ for $\nu > 2$

Kurtosis: $\kappa = \frac{6}{\nu-4}$ for $\nu > 4$

### Hansen Skew-t Distribution

Constants: $a = 4\lambda c \frac{\nu-2}{\nu-1}$, $b^2 = 1 + 3\lambda^2 - a^2$, $c = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi(\nu-2)}\Gamma(\frac{\nu}{2})}$

### GJR-GARCH(1,1)

$h_t = \omega + (\alpha + \gamma I_{t-1}) \epsilon^2_{t-1} + \beta h_{t-1}$

Stationarity: $\alpha + \beta + 0.5\gamma < 1$

Unconditional variance: $\bar{h} = \frac{\omega}{1 - \alpha - \beta - 0.5\gamma}$

### Kalman Filter (Scalar)

Predict: $\mu_{t|t-1} = \phi\mu_{t-1|t-1}$, $P_{t|t-1} = \phi^2 P_{t-1|t-1} + q$

Innovation: $e_t = r_t - \mu_{t|t-1}$, $S_t = P_{t|t-1} + R_t$

Update: $K_t = P_{t|t-1}/S_t$, $\mu_{t|t} = \mu_{t|t-1} + K_t e_t$, $P_{t|t} = (1-K_t)P_{t|t-1}$

Log-likelihood: $\ell_t = -0.5[\log(2\pi) + \log(S_t) + e_t^2/S_t]$

## APPENDIX B: TEST UNIVERSE SUMMARY

| Asset | Type | Expected nu | Expected phi | Expected Model | Crisis Sensitivity |
|-------|------|-------------|--------------|----------------|-------------------|
| SPY | Index | 8-15 | 0.0-0.3 | phi-Gaussian or Student-t(20) | Moderate |
| QQQ | Index | 6-12 | 0.1-0.4 | phi-Student-t(8) | Moderate-High |
| IWM | Index | 5-10 | 0.0-0.2 | phi-Student-t(8) | High |
| MSFT | Large Cap | 8-20 | 0.1-0.4 | phi-Gaussian-momentum | Low |
| GOOGL | Large Cap | 8-15 | 0.1-0.3 | phi-Student-t(12) | Low-Moderate |
| NFLX | Large Cap | 4-8 | 0.1-0.5 | phi-Student-t(4)-momentum | High |
| AAPL | Large Cap | 8-15 | 0.1-0.3 | phi-Gaussian-momentum | Low |
| NVDA | Large Cap | 5-10 | 0.2-0.5 | phi-Student-t(8)-momentum | Moderate |
| AMZN | Large Cap | 6-12 | 0.1-0.4 | phi-Student-t(8)-momentum | Moderate |
| META | Large Cap | 5-10 | 0.1-0.4 | phi-Student-t(6)-momentum | Moderate-High |
| CRWD | Mid Cap | 4-8 | 0.1-0.4 | phi-Student-t(6)-momentum | High |
| DKNG | Mid Cap | 3-6 | 0.0-0.3 | phi-Student-t(4)-momentum | Very High |
| PLTR | Mid Cap | 4-8 | 0.1-0.4 | phi-Student-t(6)-momentum | High |
| SQ | Mid Cap | 3-6 | 0.0-0.3 | phi-Student-t(4)-CST | Very High |
| RKLB | Mid Cap | 3-6 | -0.1-0.3 | phi-Student-t(4)-CST | Very High |
| UPST | Small Cap | 2.5-4 | -0.1-0.3 | phi-Student-t(3)-CST | Extreme |
| AFRM | Small Cap | 2.5-5 | -0.1-0.3 | phi-Student-t(3)-Hansen | Extreme |
| IONQ | Small Cap | 2.5-4 | -0.2-0.2 | phi-Student-t(3)-CST | Extreme |
| SOFI | Small Cap | 3-6 | -0.1-0.3 | phi-Student-t(4)-momentum | Very High |
| MSTR | High Vol | 2.5-4 | -0.1-0.2 | phi-Student-t(3)-CST | Extreme |
| TSLA | High Vol | 4-8 | 0.0-0.3 | phi-Student-t(4)-momentum | Very High |
| GC=F | Gold | 6-12 | -0.1-0.1 | phi-Student-t(8)-MS-q | Low |
| SI=F | Silver | 4-8 | -0.1-0.2 | phi-Student-t(4)-VoV | Moderate-High |
| BTC-USD | Crypto | 2.5-4 | 0.0-0.3 | phi-Student-t(3)-Hansen | Extreme |
| ETH-USD | Crypto | 2.5-4 | 0.0-0.3 | phi-Student-t(3)-Hansen | Extreme |

## APPENDIX C: IMPLEMENTATION PRIORITY MATRIX

| Epic | Priority | Files Affected | Estimated Stories | Risk |
|------|----------|---------------|-------------------|------|
| 1. Gamma Precision | CRITICAL | numba_kernels.py | 3 | Low |
| 2. Vectorized Stability | HIGH | vectorized_ops.py | 3 | Low |
| 3. GARCH Improvements | HIGH | numba_kernels.py | 3 | Medium |
| 4. Phi-Gaussian Precision | HIGH | phi_gaussian.py | 4 | Low |
| 5. Gaussian Pipeline | HIGH | gaussian.py | 4 | Medium |
| 6. Student-t Nu Optimization | CRITICAL | phi_student_t.py | 3 | Medium |
| 7. MS-q Precision | CRITICAL | phi_student_t.py | 3 | Medium |
| 8. Two-Piece/Mixture | HIGH | phi_student_t.py, numba_kernels.py | 3 | Medium |
| 9. Unified Coherence | CRITICAL | phi_student_t_unified.py | 3 | High |
| 10. GARCH Integration | HIGH | phi_student_t_unified.py | 3 | Medium |
| 11. Hansen Skew-t | HIGH | numba_kernels.py | 3 | Medium |
| 12. CST Model | MEDIUM-HIGH | numba_kernels.py | 3 | Low |
| 13. Filter Dispatch | CRITICAL | numba_wrappers.py | 3 | Low |
| 14. Momentum Integration | HIGH | numba_wrappers.py, phi_gaussian.py, phi_student_t.py | 3 | Medium |
| 15. PIT Calibration | CRITICAL | gaussian.py, phi_gaussian.py, phi_student_t.py | 3 | Low |
| 16. CRPS Optimization | HIGH | numba_kernels.py, phi_student_t_unified.py | 3 | Medium |
| 17. Large Cap Benchmarks | HIGH | All | 3 | Low |
| 18. Small Cap/High Vol | CRITICAL | All | 3 | High |
| 19. Precious Metals | HIGH | All | 3 | Medium |
| 20. Cryptocurrency | HIGH | All | 3 | High |
| 21. Index Funds | HIGH | All | 3 | Low |
| 22. GAS Extensions | MEDIUM-HIGH | phi_student_t_unified.py, numba_kernels.py | 3 | High |
| 23. Jump-Diffusion | MEDIUM | phi_student_t_unified.py, numba_kernels.py | 3 | High |
| 24. Observation Noise | HIGH | phi_student_t.py, phi_student_t_unified.py | 3 | Medium |
| 25. Scale Parameter | HIGH | phi_gaussian.py, phi_student_t.py, gaussian.py | 2 | Low |
| 26. Regression Testing | CRITICAL | All + test infra | 3 | Low |
| 27. Performance | MEDIUM | numba_kernels.py, numba_wrappers.py | 3 | Low |
| 28. Walk-Forward OOS | CRITICAL | All | 3 | Medium |
| 29. Model Comparison | HIGH | All + vectorized_ops.py | 3 | Medium |
| 30. Dashboard | MEDIUM-HIGH | All + new file | 3 | Low |

**Total**: 30 Epics, 91 Stories, ~455 Tasks

---

## APPENDIX D: RISK MANAGEMENT AND MODEL SAFETY

### D.1 Model Failure Modes

Every mathematical model can fail. Understanding HOW it fails is more important than
preventing failure. The following failure taxonomy classifies known failure modes by
severity and detection difficulty.

| Failure Mode | Severity | Detection | Root Cause | Mitigation |
|--------------|----------|-----------|------------|------------|
| State collapse ($P_t \to 0$) | CRITICAL | Easy (NaN) | $q$ too small | Scale-aware $q_{min}$ |
| Variance explosion ($S_t \to \infty$) | CRITICAL | Easy (Inf) | GARCH persistence > 1 | Stationarity constraint |
| PIT U-shape | HIGH | Medium (visual) | $R_t$ underestimated | Variance inflation $\beta$ |
| PIT inverse-U | MEDIUM | Medium (visual) | $R_t$ overestimated | CRPS shrinkage $\alpha$ |
| BMA weight collapse (one model = 1.0) | HIGH | Easy (check) | BIC spread too large | Weight floor $w_{min} = 10^{-6}$ |
| Phi explosion ($\phi > 1.05$) | CRITICAL | Easy (check) | Insufficient regularization | Phi shrinkage prior |
| Nu collapse ($\nu \to 2$) | HIGH | Easy (check) | Too many outliers | Nu floor at 2.1 |
| Gammaln underflow | CRITICAL | Hard (subtle) | Stirling at low $\nu$ | Lanczos approximation |
| Innovation overflow ($|e_t| > 100\sigma$) | HIGH | Medium | Data error or flash crash | Robust Kalman weights |
| MS-q stuck in stress | MEDIUM | Medium | Sensitivity too high | Sensitivity calibration per asset |
| GARCH-VoV double counting | MEDIUM | Hard | Both active simultaneously | VoV damping when GARCH active |
| Momentum-phi collinearity | MEDIUM | Hard (subtle) | $\phi \to 1$ when momentum active | Phi shrinkage toward 1.0 |

### D.2 Circuit Breakers

Automated circuit breakers that halt model usage when safety conditions are violated:

**Level 1 -- Warning** (log and continue):
- PIT KS p-value < 0.05 for 3 consecutive tuning periods
- CRPS > 2x historical average for any single asset
- BMA weight turnover > 50% in a single month
- GAS parameter $|\theta_{t+1} - \theta_t| > 3\sigma_\theta$

**Level 2 -- Escalation** (flag for human review):
- PIT KS p-value < 0.01 for any asset
- CSS < 0.50 during stress period
- FEC < 0.60 for any asset
- Model produces NaN in any output field
- Nu estimated < 2.5 (tails may be too heavy for reliable inference)

**Level 3 -- Halt** (stop using model, revert to backup):
- Variance explosion: $S_t > 10^6 \times \text{median}(S)$
- State explosion: $|\mu_t| > 100 \times \text{std}(r)$
- Filter divergence: $|e_t| / \sqrt{S_t} > 50$ for 3 consecutive steps
- Log-likelihood = $-\infty$ or NaN for any observation

### D.3 Backup Models

When primary model fails circuit breakers, fall back to these baseline models:

| Primary | Backup | Trigger |
|---------|--------|---------|
| Unified phi-Student-t | Standard phi-Student-t($\nu=8$) | Any Level 3 failure |
| phi-Student-t | phi-Gaussian | Nu collapse or gammaln failure |
| phi-Gaussian | Random walk Gaussian | Phi explosion or state collapse |
| Random walk | Historical volatility | Any numerical failure |
| Historical volatility | Fixed 20% annualized | All models failed |

The backup chain ensures there is ALWAYS a valid forecast, even in catastrophic scenarios.
Each fallback level produces wider confidence intervals (less informative but safer).

### D.4 Model Monitoring Checklist

Weekly monitoring protocol for production model quality:

```
[ ] Run `make accuracy-report` -- all cells green/yellow (no red)
[ ] Check PIT histograms for all high-conviction signal assets
[ ] Verify BMA weight stability: turnover < 30%
[ ] Check GARCH persistence: all assets < 0.999
[ ] Review MS-q regime assignments: stress fraction < 20% in calm markets
[ ] Verify no NaN in tuning cache files
[ ] Compare 1-week forward CRPS to historical average
[ ] Review circuit breaker logs: no Level 2 or Level 3 triggers
```

### D.5 Data Quality Gates

Models are only as good as their input data. Enforce these gates before tuning:

| Gate | Condition | Action if Failed |
|------|-----------|-----------------|
| Minimum length | N >= 252 (1 year) | Skip asset (insufficient data) |
| Missing data | Gaps <= 5 consecutive days | Forward-fill small gaps |
| Missing data | Gaps > 5 consecutive days | Split into separate series |
| Extreme returns | $|r_t| > 50\%$ single day | Flag as potential data error |
| Zero variance | $\text{std}(r) < 10^{-8}$ | Skip (likely delisted/suspended) |
| Stale price | Same close > 10 consecutive days | Flag as potential data error |
| Volume collapse | Volume < 100 for > 5 days | Flag as illiquid |

---

## APPENDIX E: GLOSSARY OF MATHEMATICAL NOTATION

| Symbol | Definition | Typical Range |
|--------|-----------|---------------|
| $r_t$ | Log return at time $t$ | $[-0.20, 0.20]$ daily |
| $\mu_t$ | Filtered state (drift estimate) | $[-0.01, 0.01]$ |
| $P_t$ | State covariance (uncertainty) | $[10^{-10}, 10^{-3}]$ |
| $q$ | Process noise variance | $[10^{-8}, 10^{-3}]$ |
| $c$ | Observation noise scale | $[0.3, 3.0]$ |
| $\phi$ | AR(1) drift persistence | $[-0.5, 1.05]$ |
| $\nu$ | Student-t degrees of freedom | $[2.1, 50]$ |
| $R_t$ | Observation noise variance | $[10^{-6}, 10^{-2}]$ |
| $S_t$ | Innovation variance ($P_{pred} + R_t$) | $[10^{-6}, 10^{-2}]$ |
| $K_t$ | Kalman gain | $[0, 1]$ |
| $e_t$ | Innovation ($r_t - \mu_{pred}$) | $[-0.10, 0.10]$ |
| $\sigma_{EWMA}$ | EWMA volatility estimate | $[0.005, 0.10]$ |
| $h_t$ | GARCH conditional variance | $[10^{-6}, 10^{-2}]$ |
| $\omega$ | GARCH intercept | $[10^{-8}, 10^{-4}]$ |
| $\alpha$ | GARCH ARCH coefficient | $[0, 0.3]$ |
| $\beta$ | GARCH GARCH coefficient | $[0.5, 0.99]$ |
| $\gamma_{lev}$ | GJR leverage coefficient | $[0, 0.5]$ |
| $\lambda$ | Hansen skewness parameter | $[-0.99, 0.99]$ |
| $\epsilon$ | CST contamination probability | $[0.01, 0.10]$ |
| $p_{stress}$ | MS-q stress probability | $[0, 1]$ |
| $q_{calm}$ | MS-q calm process noise | $[10^{-7}, 10^{-5}]$ |
| $q_{stress}$ | MS-q stress process noise | $[10^{-5}, 10^{-3}]$ |
| $\alpha_{asym}$ | Smooth asymmetric nu parameter | $[-0.3, 0.3]$ |
| $k_{asym}$ | Smooth asymmetric nu sensitivity | $[0.5, 2.0]$ |
| $\gamma_{VoV}$ | Vol-of-vol coefficient | $[0, 1.0]$ |
| $u_t$ | Momentum/MR exogenous input | $[-k\sqrt{q}, k\sqrt{q}]$ |
| $\kappa$ | OU mean-reversion speed | $[0, 0.10]$ |
| BIC | Bayesian Information Criterion | $[-40000, -20000]$ |
| CRPS | Continuous Ranked Probability Score | $[0.008, 0.040]$ |
| PIT | Probability Integral Transform | $[0, 1]$ uniform if calibrated |
| CSS | Calibration Stability Under Stress | $[0, 1]$ higher = better |
| FEC | Forecast Entropy Consistency | $[0, 1]$ higher = better |
| $H$ | Hyvarinen score | $[-500, 2000]$ lower = better |

---

## APPENDIX F: IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-4)

**Focus**: Numerical precision and baseline model accuracy

| Week | Epics | Deliverables |
|------|-------|-------------|
| 1 | Epic 1 (Gamma), Epic 13 (Dispatch) | Lanczos gammaln, dispatch audit |
| 2 | Epic 2 (Vectorized), Epic 4 (Phi-Gaussian) | Stable variance, Joseph form |
| 3 | Epic 3 (GARCH), Epic 25 (Scale c) | Stationarity, c regularization |
| 4 | Epic 26 (Regression Tests) | Golden score registry, test suite |

**Exit Criteria**: All 25 test assets pass golden score tests. No NaN in any filter.

### Phase 2: Heavy Tail Accuracy (Weeks 5-8)

**Focus**: Student-t and advanced observation models

| Week | Epics | Deliverables |
|------|-------|-------------|
| 5 | Epic 6 (Nu Optimization) | Continuous nu, regime-dependent nu |
| 6 | Epic 7 (MS-q), Epic 8 (Two-Piece/Mixture) | Calibrated MS-q, model selection gate |
| 7 | Epic 11 (Hansen), Epic 12 (CST) | Validated Hansen/CST, selection criteria |
| 8 | Epic 15 (PIT), Epic 16 (CRPS) | Closed-form CRPS, PIT entropy |

**Exit Criteria**: CRPS < 0.020 for SPY. PIT KS p > 0.10 for 12+ assets.

### Phase 3: Unified Model and Integration (Weeks 9-12)

**Focus**: Unified pipeline coherence and momentum integration

| Week | Epics | Deliverables |
|------|-------|-------------|
| 9 | Epic 9 (Unified Coherence) | Tier interaction diagnostics |
| 10 | Epic 10 (GARCH Integration), Epic 14 (Momentum) | Coherence checks, cap calibration |
| 11 | Epic 5 (Gaussian Pipeline), Epic 24 (Observation Noise) | Multi-start, adaptive dead-zone |
| 12 | Integration testing | Full pipeline validation |

**Exit Criteria**: Unified model BIC within 5 nats of oracle. All tier interactions flagged.

### Phase 4: Cross-Asset Validation (Weeks 13-16)

**Focus**: Asset-specific accuracy and OOS validation

| Week | Epics | Deliverables |
|------|-------|-------------|
| 13 | Epic 17 (Large Cap), Epic 21 (Indices) | SPY golden test, cross-asset ordering |
| 14 | Epic 18 (Small Cap), Epic 20 (Crypto) | MSTR/BTC tail validation |
| 15 | Epic 19 (Metals), Asset-class profiles | Gold/Silver profile validation |
| 16 | Epic 28 (Walk-Forward OOS) | 1-day and 7-day OOS accuracy |

**Exit Criteria**: OOS CRPS within 15% of in-sample. Crisis-period CSS > 0.60.

### Phase 5: Advanced and Dashboard (Weeks 17-20)

**Focus**: GAS extensions, jump diffusion, monitoring

| Week | Epics | Deliverables |
|------|-------|-------------|
| 17 | Epic 22 (GAS Extensions) | GAS-nu, GAS-phi kernels |
| 18 | Epic 23 (Jump-Diffusion) | Jump detection, modified likelihood |
| 19 | Epic 29 (Model Comparison) | BIC-OOS correlation, weight stability |
| 20 | Epic 27 (Performance), Epic 30 (Dashboard) | AOT compilation, accuracy heatmap |

**Exit Criteria**: Full dashboard operational. Weekly accuracy monitoring active.

---

## APPENDIX G: KEY REFERENCES

1. Creal, D., Koopman, S.J., Lucas, A. (2013). "Generalized Autoregressive Score Models."
   *Journal of Applied Econometrics*, 28(5), 777-795. [GAS framework]

2. Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
   *International Economic Review*, 35(3), 705-730. [Hansen skew-t]

3. Gneiting, T., Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction,
   and Estimation." *JASA*, 102(477), 359-378. [CRPS theory]

4. Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). "Volatility is Rough."
   *Quantitative Finance*, 18(6), 933-949. [Rough volatility]

5. Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility."
   *Journal of Financial Econometrics*, 7(2), 174-196. [HAR model]

6. Glosten, L.R., Jagannathan, R., Runkle, D.E. (1993). "On the Relation between
   Expected Value and Volatility of Nominal Excess Return on Stocks."
   *Journal of Finance*, 48(5), 1779-1801. [GJR-GARCH]

7. Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous."
   *Journal of Financial Economics*, 3(1-2), 125-144. [Jump-diffusion]

8. Hersbach, H. (2000). "Decomposition of the Continuous Ranked Probability Score for
   Ensemble Prediction Systems." *Weather and Forecasting*, 15(5), 559-570. [CRPS decomposition]

9. Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*.
   Cambridge University Press. [Kalman filter foundations]

10. Rauch, H.E., Tung, F., Striebel, C.T. (1965). "Maximum Likelihood Estimates of
    Linear Dynamic Systems." *AIAA Journal*, 3(8), 1445-1450. [RTS smoother]

---

## APPENDIX H: SCORING PROTOCOL AND THRESHOLDS

### H.1 Metric Definitions and Elite Thresholds

| Metric | Formula | Elite | Acceptable | Failing |
|--------|---------|-------|------------|---------|
| BIC | $-2\ell + k\log(n)$ | < -33000 | < -29000 | >= -29000 |
| CRPS | $\mathbb{E}[|X-y|] - 0.5\mathbb{E}[|X-X'|]$ | < 0.015 | < 0.020 | >= 0.020 |
| PIT KS | $\sup|F_n(u) - u|$ | p > 0.20 | p > 0.05 | p <= 0.05 |
| CSS | Stress-period PIT stability | > 0.80 | > 0.65 | <= 0.65 |
| FEC | Entropy tracking coefficient | > 0.85 | > 0.75 | <= 0.75 |
| Hyvarinen | $0.5s^2 - 1/\sigma^2$ | < 500 | < 1000 | >= 1000 |
| DIG | Directional information gain | > 0.10 | > 0.05 | <= 0.05 |

### H.2 Asset-Class Specific Expectations

**Large Cap Equities** (MSFT, GOOGL, AAPL, NVDA, AMZN, META, NFLX):
- Expected model: phi-Gaussian or phi-Student-t($\nu \geq 8$)
- Expected CRPS: 0.010 - 0.015 (well-behaved, liquid)
- Expected PIT: p > 0.15 (Gaussian is often sufficient)
- Key risk: over-fitting to calm periods, poor crisis performance
- NFLX and META may require lower $\nu$ due to earnings-driven tails

**Mid Cap Equities** (CRWD, DKNG, PLTR, SQ, RKLB):
- Expected model: phi-Student-t($\nu \in [4, 8]$) with momentum
- Expected CRPS: 0.015 - 0.022 (more volatile, less liquid than large cap)
- Expected PIT: p > 0.08 (moderate calibration difficulty)
- Key risk: vol clustering around earnings, growth-stage regime shifts
- SQ and RKLB may need CST for contaminated observations

**Small Cap Equities** (UPST, AFRM, IONQ, SOFI):
- Expected model: phi-Student-t($\nu \leq 4$) with CST or Hansen skew-t
- Expected CRPS: 0.020 - 0.030 (thin liquidity, extreme kurtosis)
- Expected PIT: p > 0.03 (relaxed -- heavy tails make calibration hard)
- Key risk: nu collapse ($\nu \to 2$), gap risk on low volume, earnings shocks
- UPST and IONQ may require jump-diffusion augmentation

**High Volatility / Thematic** (MSTR, TSLA):
- Expected model: phi-Student-t($\nu \leq 5$) with momentum and CST
- Expected CRPS: 0.018 - 0.028 (extreme regime sensitivity)
- Expected PIT: p > 0.05 (acceptable with heavy tails)
- Key risk: MSTR BTC correlation, TSLA meme/momentum regime shifts

**Index Funds** (SPY, QQQ, IWM):
- Expected model: phi-Gaussian-momentum or phi-Student-t($\nu \geq 12$)
- Expected CRPS: 0.008 - 0.012 (best in universe)
- Expected PIT: p > 0.20 (diversification helps calibration)
- Key risk: crisis-period calibration breakdown

**Precious Metals** (GC=F, SI=F):
- Expected model: phi-Student-t with MS-q
- Expected CRPS: 0.012 - 0.018 (moderate difficulty)
- Expected PIT: p > 0.10 (regime-dependent)
- Key risk: gold/silver correlation regime switches

**Cryptocurrency** (BTC-USD, ETH-USD):
- Expected model: phi-Student-t($\nu \leq 4$) with Hansen skew-t
- Expected CRPS: 0.020 - 0.035 (hardest in universe)
- Expected PIT: p > 0.03 (relaxed due to extreme tails)
- Key risk: structural breaks, regulatory shocks, weekend gaps
- ETH-USD may exhibit higher correlation instability vs BTC-USD

### H.3 Composite Score Formula

The composite score aggregates all metrics into a single number:

$$\text{Score} = w_{BIC} \cdot \text{BIC}_{norm} + w_{CRPS} \cdot \text{CRPS}_{norm} + w_{PIT} \cdot \text{PIT}_{norm} + w_{CSS} \cdot \text{CSS} + w_{FEC} \cdot \text{FEC} + w_{Hyv} \cdot \text{Hyv}_{norm}$$

Default weights: $w_{BIC} = 0.25$, $w_{CRPS} = 0.25$, $w_{PIT} = 0.15$, $w_{CSS} = 0.15$, $w_{FEC} = 0.10$, $w_{Hyv} = 0.10$

Normalization: each metric mapped to $[0, 100]$ via asset-class-specific bounds.

### H.4 Improvement Tracking Protocol

For each epic implementation, record:

```
Before: {metric: value} for all 25 test assets (including mid cap: CRWD, DKNG, PLTR, SQ; small cap: UPST, AFRM, IONQ, SOFI)
Change: {description of code change}
After:  {metric: value} for all 25 test assets (including mid cap: CRWD, DKNG, PLTR, SQ; small cap: UPST, AFRM, IONQ, SOFI)
Delta:  {metric: after - before} per asset, average across universe
Status: IMPROVED / NEUTRAL / DEGRADED
```

An epic is considered successful if:
1. Average CRPS improved or unchanged across test universe
2. No single asset degraded by > 10% on any metric
3. All circuit breaker tests still pass
4. Walk-forward OOS accuracy not degraded

---

*"The purpose of computing is insight, not numbers." -- Richard Hamming*

*"All models are wrong, but some are useful." -- George Box*

*"The question is not whether the model is right, but whether the model is useful
and the uncertainty is honestly reported." -- This project's philosophy*

