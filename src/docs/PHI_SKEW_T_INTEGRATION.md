# φ-Skew-t Integration with Bayesian Model Averaging

## Overview

This document describes the implementation of **Proposal 5 — φ-Skew-t with Bayesian Model Averaging (BMA)**, which integrates the Fernández-Steel skew-t distribution as a candidate model in the existing BMA framework.

## Core Principle

> **"Skewness is a hypothesis, not a certainty."**

Rather than assuming asymmetric tails, the φ-Skew-t model **competes** with simpler distributions (Gaussian, symmetric Student-t) in the BMA ensemble. If data does not support skewness, the model weight collapses naturally toward symmetric alternatives.

## Model Set

The BMA ensemble now includes the following candidate distributions:

| Model | Parameters | Description |
|-------|------------|-------------|
| `kalman_gaussian` | q, c | Standard Kalman with Gaussian noise |
| `kalman_phi_gaussian` | q, c, φ | AR(1) drift with Gaussian noise |
| `phi_student_t_nu_{4,6,8,12,20}` | q, c, φ | AR(1) drift with Student-t tails (fixed ν) |
| `phi_skew_t_nu_{ν}_gamma_{γ}` | q, c, φ | AR(1) drift with skew-t tails (fixed ν, γ) |

## Fernández-Steel Parameterization

The skew-t distribution uses the Fernández-Steel parameterization:

```
f(z|ν,γ) = (2/(γ + 1/γ)) * [f_t(z/γ|ν) if z≥0 else f_t(γ*z|ν)]
```

Where:
- `f_t` is the standard Student-t density with ν degrees of freedom
- `γ = 1.0`: Symmetric (reduces to Student-t)
- `γ < 1.0`: Left-skewed (heavier left tail) — crash risk during stress
- `γ > 1.0`: Right-skewed (heavier right tail) — euphoria/melt-up risk

## Discrete Grid Approach

Both ν (degrees of freedom) and γ (skewness) use discrete grids:

```python
SKEW_T_NU_GRID = [4, 6, 8, 12, 20]
SKEW_T_GAMMA_GRID = [0.7, 0.85, 1.0, 1.15, 1.3]
```

Each (ν, γ) combination is treated as a separate model in the BMA ensemble. Symmetric models (γ=1.0) are excluded from the skew-t grid since they're already covered by `phi_student_t`.

## Integration Points

### 1. `src/models/phi_skew_t.py` (NEW)

New module implementing the `PhiSkewTDriftModel` class:
- `filter_phi()`: Kalman filter with skew-t observation noise
- `pit_ks()`: PIT calibration test using skew-t CDF
- `optimize_params_fixed_nu_gamma()`: Parameter optimization for fixed ν, γ
- `sample_skew_t()`: Random variate generation
- `cdf_skew_t()`: CDF for PIT computation
- `logpdf_skew_t()`: Log-density for likelihood

### 2. `src/tuning/tune.py`

Modified to include φ-Skew-t in the model fitting:

```python
# In fit_all_models_for_regime():
# Model 3: Phi-Skew-t with DISCRETE ν and γ GRID
for nu_fixed in SKEW_T_NU_GRID:
    for gamma_fixed in SKEW_GAMMA_GRID_ASYMMETRIC:
        model_name = get_skew_t_model_name(nu_fixed, gamma_fixed)
        # ... fit and store model
```

### 3. `src/decision/signals.py`

Modified to handle skew-t sampling in Monte Carlo:

```python
# In run_regime_specific_mc():
if use_skew_t:
    raw_t = rng.standard_t(df=nu, size=n_paths)
    skewed = np.where(raw_t >= 0, raw_t / gamma, raw_t * gamma)
    # ... apply to drift and observation noise
```

### 4. UX Display (`tune_ux.py`, `signals_ux.py`)

Updated to recognize and display skew-t models:
- Model type shown as "Skew-t(L)" or "Skew-t(R)" based on γ
- γ parameter displayed alongside ν when relevant

## BMA Weight Computation

Model weights are computed using BIC approximation to marginal likelihood:

```python
log_mls = np.array([m.log_marginal_likelihood for m in models])
log_mls -= np.max(log_mls)  # Numerical stability
weights = np.exp(log_mls)
weights /= np.sum(weights)
```

The BIC includes a complexity penalty, naturally penalizing the additional γ parameter if it doesn't improve fit.

## Predictive Sampling

BMA sampling follows a two-stage procedure:

1. Sample model indicator `M_k ~ Categorical(weights)`
2. Sample return from the selected model's predictive distribution

For skew-t models, sampling uses the Fernández-Steel transformation:
```python
t_samples = rng.standard_t(df=nu, size=n)
skew_samples = np.where(t_samples >= 0, t_samples / gamma, t_samples * gamma)
```

## Advantages

1. **Explicit model uncertainty** — Skewness is a hypothesis that can be tested
2. **Graceful degradation** — Falls back to symmetric models when evidence is weak
3. **No forced skewness** — Symmetric γ=1.0 already covered by Student-t
4. **BMA-compatible** — Uses same BIC framework as existing models
5. **Production-safe** — No MCMC or VI in the hot path

## Known Limitations

1. BIC is an approximation to true marginal likelihood
2. Requires fitting multiple candidate models (increased computation)
3. φ-Skew-t numerics require bounded optimization
4. Hyvärinen score uses Student-t approximation (full skew-t derivation pending)

## Usage

After running `make tune --force` to regenerate cache with skew-t models:

```bash
# Signal generation automatically uses BMA over all models including skew-t
python src/decision/signals.py --asset AAPL

# The output will show skew-t selection when appropriate
# Model: Skew-t(L)|q=1.2e-06|c=0.92|φ=+0.87|ν=6|γ=0.85|bic=1234
```

## Files Modified

- `src/models/__init__.py` — Export new skew-t components
- `src/models/phi_skew_t.py` — NEW: Core skew-t model implementation
- `src/tuning/tune.py` — Include skew-t in BMA model fitting
- `src/decision/signals.py` — Handle skew-t sampling in MC
- `src/tuning/tune_ux.py` — Display skew-t in progress output
- `src/tests/test_phi_skew_t.py` — NEW: Comprehensive tests

## References

1. Fernández, C., & Steel, M. F. (1998). On Bayesian modeling of fat tails and skewness. *Journal of the American Statistical Association*, 93(441), 359-371.

2. Azzalini, A., & Capitanio, A. (2003). Distributions generated by perturbation of symmetry with emphasis on a multivariate skew t-distribution. *Journal of the Royal Statistical Society: Series B*, 65(2), 367-389.
