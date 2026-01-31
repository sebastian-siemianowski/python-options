# 2-State Gaussian Mixture Model (GMM) Integration

## Overview

This document describes the implementation of the **2-State Gaussian Mixture Model** as recommended by the Expert Panel for enhanced return distribution modeling in the BMA framework.

## Expert Panel Recommendation

The implementation follows the highest-scoring solution (87.3/100 average) synthesizing:
- **Professor Zhang's Solution 2**: GMM as Monte Carlo Proposal Distribution
- **Professor Liu's Solution 2**: GMM on Volatility-Adjusted Returns

## Core Principle

> **"Bimodality is a hypothesis, not a certainty."**

The GMM captures bimodal return distributions that arise from alternating momentum and reversal regimes. If data doesn't support bimodality, the GMM degenerates to a single Gaussian and the system falls back to standard sampling.

## Mathematical Model

The 2-State GMM assumes:

$$p(z_t) = \pi_1 \mathcal{N}(z_t; \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(z_t; \mu_2, \sigma_2^2)$$

Where:
- $z_t = r_t / \sigma_t$ (volatility-standardized returns)
- $\pi_1, \pi_2$ = mixing proportions ($\pi_1 + \pi_2 = 1$)
- $\mu_1, \mu_2$ = component means
- $\sigma_1^2, \sigma_2^2$ = component variances

### Component Interpretation

| Component | Interpretation | Typical Characteristics |
|-----------|---------------|------------------------|
| Component 0 | "Momentum" | $\mu_0 > 0$, moderate $\sigma_0^2$ |
| Component 1 | "Reversal/Crisis" | $\mu_1 < 0$ or $\mu_1 \approx 0$, elevated $\sigma_1^2$ |

## Integration Architecture

### 1. Fitting Phase (`tune.py`)

GMM is fitted to volatility-adjusted returns after GARCH parameter estimation:

```
Returns → GARCH Volatility → Standardize → GMM Fit → Store Parameters
```

Key steps:
1. Standardize returns by EWMA volatility: $z_t = r_t / \sigma_t$
2. Winsorize extremes (1st-99th percentile)
3. Fit 2-component GMM via EM algorithm
4. Validate: check separation (> 0.5 Mahalanobis) and degeneracy (< 95% single component)
5. Store in `tuned_params["global"]["gmm"]`

### 2. Sampling Phase (`signals.py`)

For Gaussian-based models in BMA, samples are drawn from the GMM mixture:

$$r_{t+H}^{(m)} \sim \sum_{k=1}^{2} \pi_k \mathcal{N}(\mu_k \cdot H + \hat{\mu}_{kf}, \sigma_k^2 \cdot H + P_t)$$

Where:
- $\hat{\mu}_{kf}$ = Kalman filtered drift estimate
- $P_t$ = drift posterior variance

### Priority Order for Noise Distributions

The `run_regime_specific_mc` function selects distributions in this order:

1. **GMM** (if available and valid, for Gaussian-based models)
2. **NIG** (if α, β, δ specified)
3. **Student-t** (if ν specified)
4. **Gaussian** (fallback)

## Files Modified/Created

### Created:
- `src/models/gaussian_mixture.py` — Core GMM implementation
- `src/tests/test_gaussian_mixture.py` — Comprehensive tests
- `src/docs/GMM_INTEGRATION.md` — This documentation

### Modified:
- `src/models/__init__.py` — Export GMM components
- `src/tuning/tune.py` — GMM fitting in `tune_asset_q`
- `src/decision/signals.py` — GMM sampling in `run_regime_specific_mc` and `bayesian_model_average_mc`
- `src/tuning/tune_ux.py` — Display "+GMM" indicator

## GMM Parameters Stored

```json
{
  "global": {
    "gmm": {
      "weights": [0.6, 0.4],
      "means": [0.3, -0.2],
      "variances": [0.8, 1.5],
      "stds": [0.894, 1.225],
      "separation": 0.45,
      "is_degenerate": false,
      "is_well_separated": false
    },
    "gmm_diagnostics": {
      "fit_success": true,
      "converged": true,
      "n_iterations": 23,
      "n_obs": 1260,
      "separation": 0.45,
      "is_well_separated": false,
      "is_degenerate": false,
      "component_0_weight": 0.6,
      "component_1_weight": 0.4,
      "component_0_mean": 0.3,
      "component_1_mean": -0.2,
      "component_0_std": 0.894,
      "component_1_std": 1.225
    }
  }
}
```

## Fallback Behavior

| Condition | Behavior |
|-----------|----------|
| Insufficient data (< 100 obs) | Skip GMM, use single Gaussian |
| EM non-convergence | Skip GMM, use single Gaussian |
| Degenerate (π₁ > 95%) | Skip GMM, use single Gaussian |
| Student-t model selected | Ignore GMM, use Student-t |
| NIG model selected | Ignore GMM, use NIG |

## Expected Improvements

Based on Professor Zhang's assessment:
- **5-15%** improvement in Expected Utility estimation accuracy for fat-tailed assets
- **Reduced position size volatility** due to better tail modeling
- **Improved PIT calibration** for extreme return forecasts
- **Modest computational overhead** (GMM fitting is O(n) per EM iteration)

## UX Display

When GMM is fitted, the model indicator shows `+GMM`:

```
✓ AAPL: φ-Gaussian+GMM|q=1.2e-06|c=0.92|φ=+0.87|bic=1234|T=72%✓
```

## Diagnostic Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `separation` | Mahalanobis distance between means | > 0.5 |
| `is_well_separated` | Components sufficiently distinct | `true` |
| `is_degenerate` | One component dominates | `false` |
| `converged` | EM reached tolerance | `true` |
| `n_iterations` | EM iterations | < 100 |

## Risk Factors

1. **Regime identification instability**: During market transitions, GMM responsibilities become uncertain
2. **Overfitting**: Small samples may produce spurious bimodality
3. **Component swapping**: EM may swap component labels across tuning runs

The calibrated trust mechanism mitigates these risks by down-weighting drift when GMM responsibilities are ambiguous.

## References

1. McLachlan, G., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
2. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-38.
