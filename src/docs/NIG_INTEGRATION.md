# NIG (Normal-Inverse Gaussian) Integration with Bayesian Model Averaging

## Overview

This document describes the implementation of **Solution 2 — NIG as Parallel Model Class in BMA Framework**, which integrates the Normal-Inverse Gaussian distribution as a candidate model in the existing Bayesian Model Averaging framework.

## Core Principle

> **"Heavy tails and asymmetry are hypotheses, not certainties."**

Rather than assuming any particular tail behavior, the NIG model **competes** with simpler distributions (Gaussian, Student-t, Skew-t) in the BMA ensemble. If data does not support NIG's extra complexity, the BIC penalty causes model weight to collapse naturally toward simpler alternatives.

## What is NIG?

The Normal-Inverse Gaussian (NIG) distribution is a four-parameter continuous probability distribution with the following properties:

### Parameters

| Parameter | Range | Interpretation |
|-----------|-------|----------------|
| α (alpha) | α > 0 | Tail heaviness (smaller α = heavier tails) |
| β (beta) | \|β\| < α | Asymmetry (β < 0 = left-skewed, β > 0 = right-skewed) |
| δ (delta) | δ > 0 | Scale parameter |
| μ (mu) | μ ∈ ℝ | Location parameter |

### Key Properties

1. **Semi-Heavy Tails**: NIG tails decay faster than Cauchy but slower than Gaussian
2. **Native Asymmetry**: β parameter directly controls skewness
3. **Infinitely Divisible**: Compatible with Lévy processes
4. **Closed Form**: Density, CDF, and moments have closed-form expressions

### Comparison with Other Distributions

| Distribution | Parameters | Tails | Asymmetry |
|-------------|------------|-------|-----------|
| Gaussian | μ, σ | Light | No |
| Student-t | μ, σ, ν | Heavy (power law) | No |
| Skew-t | μ, σ, ν, γ | Heavy (power law) | Yes |
| NIG | μ, σ, α, β | Semi-heavy (exponential) | Yes |

NIG occupies a unique position: it has asymmetry like Skew-t but with different tail behavior (exponential vs power-law decay).

## Model Set

The BMA ensemble now includes the following candidate distributions:

| Model | Parameters | Description |
|-------|------------|-------------|
| `kalman_gaussian` | q, c | Standard Kalman with Gaussian noise |
| `kalman_phi_gaussian` | q, c, φ | AR(1) drift with Gaussian noise |
| `phi_student_t_nu_{4,6,8,12,20}` | q, c, φ | AR(1) drift with Student-t tails (fixed ν) |
| `phi_skew_t_nu_{ν}_gamma_{γ}` | q, c, φ | AR(1) drift with skew-t tails (fixed ν, γ) |
| `phi_nig_alpha_{α}_beta_{β}` | q, c, φ | AR(1) drift with NIG tails (fixed α, β) |

## Discrete Grid Approach

Both α (tail heaviness) and β (asymmetry) use discrete grids:

```python
NIG_ALPHA_GRID = [1.5, 2.5, 4.0, 8.0]
NIG_BETA_RATIO_GRID = [-0.3, 0.0, 0.3]  # Actual β = ratio * α
```

This gives 4 × 3 = 12 NIG sub-models in the BMA ensemble.

Each (α, β) combination is treated as a separate model. This avoids continuous optimization instability and allows proper BIC-based model selection.

## Integration Points

### 1. `src/models/phi_nig.py` (NEW)

New module implementing the `PhiNIGDriftModel` class:

- `filter_phi()`: Kalman filter with NIG observation noise
- `pit_ks()`: PIT calibration test using NIG CDF
- `optimize_params_fixed_nig()`: Parameter optimization for fixed α, β, δ
- `sample()`: Random variate generation
- `cdf()`: CDF for PIT computation
- `logpdf()`: Log-density for likelihood
- `fit_mle()`: MLE estimation of NIG parameters

### 2. `src/tuning/tune.py`

Modified to include NIG in the model fitting:

```python
# In fit_all_models_for_regime():
# Model 3: Phi-NIG with DISCRETE α and β GRID
for alpha_fixed in NIG_ALPHA_GRID:
    for beta_ratio in NIG_BETA_RATIO_GRID:
        beta_fixed = beta_ratio * alpha_fixed
        model_name = get_nig_model_name(alpha_fixed, beta_fixed)
        # ... fit and store model
```

### 3. `src/decision/signals.py`

Modified to handle NIG sampling in Monte Carlo:

```python
# In run_regime_specific_mc():
if use_nig:
    nig_samples = norminvgauss.rvs(nig_a, nig_b, loc=0, scale=1, size=n_paths)
    # ... apply to drift and observation noise
```

### 4. UX Display (`tune_ux.py`)

Updated to recognize and display NIG models:
- Model type shown as "NIG(L)" or "NIG(R)" based on β sign
- α and β parameters displayed in details when relevant

## BMA Weight Computation

Model weights are computed using BIC approximation to marginal likelihood:

```python
log_mls = np.array([m.log_marginal_likelihood for m in models])
log_mls -= np.max(log_mls)  # Numerical stability
weights = np.exp(log_mls)
weights /= np.sum(weights)
```

The BIC includes a complexity penalty, naturally penalizing the additional α and β parameters if they don't improve fit.

## Predictive Sampling

BMA sampling follows a two-stage procedure:

1. Sample model indicator `M_k ~ Categorical(weights)`
2. Sample return from the selected model's predictive distribution

For NIG models, sampling uses scipy's `norminvgauss` implementation:
```python
nig_samples = norminvgauss.rvs(a, b, loc=0, scale=1, size=n, random_state=rng)
```

Where `a = α·δ` and `b = β·δ` in scipy's parameterization.

## Advantages of NIG Integration

1. **Model Uncertainty**: NIG is a hypothesis that can be tested via BIC
2. **Graceful Degradation**: Falls back to simpler models when evidence is weak
3. **Different Tail Behavior**: Semi-heavy tails complement Student-t's power-law tails
4. **Native Asymmetry**: β parameter captures skewness directly
5. **BMA-Compatible**: Uses same BIC framework as existing models
6. **Production-Safe**: No MCMC or VI in the hot path

## Known Limitations

1. **Four Parameters**: More parameters than Student-t, requiring more data
2. **Estimation Instability**: β can be hard to estimate reliably
3. **Computational Cost**: Bessel function evaluation in density
4. **Approximations**: Hyvärinen score uses Gaussian approximation

## When NIG is Preferred

NIG may be selected over other distributions when:

1. **Semi-Heavy Tails Needed**: Tails heavier than Gaussian but lighter than Cauchy
2. **Asymmetric Returns**: Clear left or right skewness in the data
3. **Sufficient Data**: Enough observations to justify extra parameters
4. **Good Calibration**: NIG PIT p-value > 0.05

## Files Modified/Created

**Created:**
- `src/models/phi_nig.py` — Core NIG model implementation
- `src/tests/test_phi_nig.py` — Comprehensive tests
- `src/docs/NIG_INTEGRATION.md` — This documentation

**Modified:**
- `src/models/__init__.py` — Export NIG components
- `src/tuning/tune.py` — Include NIG in BMA model fitting
- `src/decision/signals.py` — Handle NIG sampling in MC
- `src/tuning/tune_ux.py` — Display NIG in progress output

## Usage

After running `make tune --force` to regenerate cache with NIG models:

```bash
# Signal generation automatically uses BMA over all models including NIG
python src/decision/signals.py --asset AAPL

# The output will show NIG selection when appropriate
# Model: NIG(L)|q=1.2e-06|c=0.92|φ=+0.87|α=2.5|β=-0.75|bic=1234
```

## References

1. Barndorff-Nielsen, O. E. (1997). Normal inverse Gaussian distributions and stochastic volatility modelling. *Scandinavian Journal of Statistics*, 24(1), 1-13.

2. Rydberg, T. H. (1997). The normal inverse Gaussian Lévy process: simulation and approximation. *Communications in Statistics: Stochastic Models*, 13(4), 887-910.

3. Prause, K. (1999). *The Generalized Hyperbolic Model: Estimation, Financial Derivatives, and Risk Measures*. PhD thesis, University of Freiburg.
