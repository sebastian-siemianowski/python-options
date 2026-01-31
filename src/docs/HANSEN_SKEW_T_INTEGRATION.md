# Hansen Skew-t Distribution Integration

## Overview

This document describes the implementation of **Hansen's Skew-t Distribution** for regime-conditional asymmetric tail modeling in the BMA framework.

## Expert Panel Recommendation

The implementation follows the **Regime-Conditional Hansen Skew-t** solution (weighted score: 87.4/100), combining:
- **Hansen (1994) parameterization** for probability calculations
- **Regime-conditional λ parameters** extending existing `regime_params` structure
- **Fernández-Steel sampling transformation** for efficient Monte Carlo

## Mathematical Model

Hansen's skew-t extends symmetric Student-t with asymmetry parameter λ ∈ (-1, 1):

$$f(z|\nu, \lambda) = bc \cdot \left(1 + \frac{z^2}{(\nu-2)(1 \mp \lambda)^2}\right)^{-(\nu+1)/2}$$

Where:
- **a** = 4λc(ν-2)/(ν-1) — location adjustment
- **b** = √(1 + 3λ² - a²) — scale adjustment
- **c** = Γ((ν+1)/2) / [√(π(ν-2)) Γ(ν/2)] — normalizing constant

### Parameter Interpretation

| Parameter | Range | Interpretation |
|-----------|-------|----------------|
| ν | (2, ∞) | Degrees of freedom (tail heaviness) |
| λ | (-1, 1) | Skewness parameter |

| λ Value | Skew Direction | Financial Meaning |
|---------|----------------|-------------------|
| λ < 0 | Left-skewed | Crash risk (heavier left tail) |
| λ = 0 | Symmetric | Standard Student-t |
| λ > 0 | Right-skewed | Recovery potential (heavier right tail) |

### Regime-Conditional Skewness

Financial intuition for regime-specific λ:

| Regime | Expected λ | Rationale |
|--------|------------|-----------|
| Bull market | λ < 0 | Complacency leads to sudden crashes |
| Bear market | λ > 0 | Recovery spikes during capitulation |
| Neutral | λ ≈ 0 | Symmetric uncertainty |

## Integration Architecture

### 1. Fitting Phase (`tune.py`)

Hansen skew-t parameters are estimated globally via MLE:

```
Returns → Standardize → Hansen MLE (ν, λ) → Store Parameters
```

Key steps:
1. Standardize returns: z_t = (r_t - μ) / σ
2. Multi-start optimization for robustness
3. Compute comparison metrics vs symmetric Student-t
4. Store in `tuned_params["global"]["hansen_skew_t"]`

### 2. Sampling Phase (`signals.py`)

For Student-t models in BMA, samples use Hansen skew-t when λ is significant:

```python
if nu is not None and hansen_lambda is not None and abs(hansen_lambda) > 1e-6:
    samples = hansen_skew_t_rvs(n, nu, hansen_lambda)
else:
    samples = standard_t_samples(n, nu)  # Symmetric fallback
```

### Priority Order for Noise Distributions

`run_regime_specific_mc` selects distributions in this order:

1. **Hansen Skew-t** (if nu + hansen_lambda specified and |λ| > 0.001)
2. **NIG** (if α, β, δ specified)
3. **Symmetric Student-t** (if nu specified)
4. **Gaussian** (fallback)

## Files Created/Modified

### Created:
- `src/models/hansen_skew_t.py` — Core Hansen skew-t implementation
- `src/tests/test_hansen_skew_t.py` — Comprehensive test suite
- `src/docs/HANSEN_SKEW_T_INTEGRATION.md` — This documentation

### Modified:
- `src/models/__init__.py` — Export Hansen skew-t components
- `src/tuning/tune.py` — Hansen skew-t fitting in `tune_asset_q()`
- `src/decision/signals.py` — Hansen skew-t sampling in `run_regime_specific_mc()` and BMA loop
- `src/tuning/tune_ux.py` — Display "+Hλ←/→" indicator

## Hansen Skew-t Parameters Stored

```json
{
  "global": {
    "hansen_skew_t": {
      "nu": 8.5,
      "lambda": -0.15,
      "log_likelihood": -1234.5,
      "skew_direction": "left"
    },
    "hansen_skew_t_diagnostics": {
      "fit_success": true,
      "n_obs": 1260,
      "converged": true,
      "aic": 2473.0,
      "bic": 2483.2
    },
    "hansen_vs_symmetric_comparison": {
      "delta_aic": -5.2,
      "delta_bic": -3.1,
      "preference": "hansen_skew_t"
    }
  }
}
```

## Fallback Behavior

| Condition | Behavior |
|-----------|----------|
| Insufficient data (< 50 obs) | Skip Hansen, use symmetric t |
| MLE non-convergence | Skip Hansen, use symmetric t |
| |λ| < 0.01 | Treat as symmetric (no skew adjustment) |
| Gaussian model selected | Ignore Hansen (not applicable) |
| NIG model selected | Ignore Hansen, use NIG |

## Expected Improvements

Based on the expert panel assessment:
- **Improved probability calibration** for asymmetric return distributions
- **Better Expected Utility estimates** capturing directional tail risk
- **Reduced overconfidence** in regime transitions
- **Modest computational overhead** (Hansen MLE is O(n) per iteration)

## UX Display

When Hansen skew-t is fitted, the model indicator shows direction:

```
✓ AAPL: Student-t+Hλ←|q=1.2e-06|c=0.92|φ=+0.87|ν=8|λ=-0.15|bic=1234|T=72%✓
```

- `+Hλ←` — Left-skewed (λ < 0, crash risk)
- `+Hλ→` — Right-skewed (λ > 0, recovery potential)

## Diagnostic Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `lambda` | Skewness parameter | [-0.5, 0.5] |
| `delta_aic` | AIC difference vs symmetric | < -2 for Hansen preference |
| `delta_bic` | BIC difference vs symmetric | < -2 for Hansen preference |
| `converged` | MLE reached tolerance | `true` |

## Core Principle

> **"Asymmetry is a hypothesis, not a certainty."**

Hansen skew-t competes with symmetric alternatives via information criteria. If data doesn't support asymmetry, λ → 0 and the system falls back to symmetric Student-t.

## References

1. Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation." *International Economic Review*, 35(3), 705-730.
2. Fernández, C. & Steel, M.F.J. (1998). "On Bayesian Modeling of Fat Tails and Skewness." *Journal of the American Statistical Association*.
