# EVT (Extreme Value Theory) Integration for Position Sizing

## Overview

This document describes the implementation of **Extreme Value Theory (EVT)** using the **Peaks Over Threshold (POT)** method with **Generalized Pareto Distribution (GPD)** fitting for robust expected loss estimation in position sizing.

## Expert Panel Recommendation

This implementation follows **Solution 2: EVT-Corrected Expected Loss for Position Sizing** (weighted score: 90.7/100), which was unanimously recommended by all three expert panel members:

- Prof. Chen Wei-Lin (Score: 88/100) — "Surgically elegant"
- Prof. Liu Jian-Ming (Score: 91/100) — "This is the correct surgical point"
- Prof. Wang Xiao-Rui (Score: 93/100) — "This is the money shot"

## Theoretical Foundation

### Pickands–Balkema–de Haan Theorem

For a broad class of distributions F, the conditional excess distribution converges to GPD:

$$\lim_{u \to x_F} P(X - u \leq y | X > u) = G_{\xi,\sigma}(y)$$

Where $G_{\xi,\sigma}$ is the Generalized Pareto Distribution with:
- **ξ (xi/shape)**: Tail index controlling tail heaviness
- **σ (sigma/scale)**: Scale parameter

### GPD Shape Parameter Interpretation

| ξ Value | Tail Type | Financial Meaning |
|---------|-----------|-------------------|
| ξ < 0 | Bounded (short) tails | Light-tailed, finite upper bound |
| ξ = 0 | Exponential tails | Gaussian-like tail decay |
| 0 < ξ < 0.5 | Moderate heavy tails | Power-law decay |
| ξ ≥ 0.5 | Very heavy tails | May have infinite variance |

### Relationship to Student-t

For Student-t distribution with ν degrees of freedom:

$$\xi = 1/\nu$$

| ν | ξ | Tail Type |
|---|---|-----------|
| 4 | 0.25 | Heavy |
| 6 | 0.17 | Moderate |
| 10 | 0.10 | Mild |
| ∞ | 0 | Gaussian |

## Conditional Tail Expectation (CTE)

The core metric used for position sizing is the CTE:

$$E[X | X > u] = u + \frac{\sigma}{1-\xi} \quad \text{for } \xi < 1$$

This provides the expected loss given that the loss exceeds the threshold u.

## Integration Architecture

### 1. Tuning Phase (`tune.py`)

EVT parameters are pre-computed during tuning:

```
Returns → Extract Losses → Select Threshold (90th percentile) 
       → Fit GPD (MLE/Hill/PWM) → Store Parameters
```

Parameters stored in `tuned_params["global"]["evt"]`:
- `xi`: GPD shape parameter
- `sigma`: GPD scale parameter  
- `threshold`: POT threshold
- `n_exceedances`: Number of tail observations
- `cte`: Conditional Tail Expectation
- `fit_success`: Whether fitting succeeded

### 2. Signal Phase (`signals.py`)

EVT-corrected expected loss replaces naive empirical mean:

```python
# OLD (naive empirical):
E_loss = np.mean(losses)

# NEW (EVT-corrected):
evt_loss, emp_loss, gpd_result = compute_evt_expected_loss(r_samples)
E_loss = evt_loss  # Uses GPD-based CTE
```

### Position Sizing Impact

```
eu_position_size = EU / max(E_loss, ε)
```

- **Heavy tails (ξ > 0.2)**: EVT E[loss] >> empirical → smaller positions
- **Light tails (ξ ≈ 0)**: EVT E[loss] ≈ empirical → minimal change
- **Fallback**: If GPD fails, use 1.5× empirical (conservative)

## Files Created/Modified

### Created:
- `src/calibration/evt_tail.py` — Core EVT/GPD implementation
- `src/tests/test_evt_tail.py` — Comprehensive test suite
- `src/docs/EVT_INTEGRATION.md` — This documentation

### Modified:
- `src/tuning/tune.py` — EVT fitting in `tune_asset_q()`
- `src/decision/signals.py` — EVT-corrected expected loss calculation
- `src/tuning/tune_ux.py` — Display "+EVTH/M/L" indicator

## EVT Parameters Stored

```json
{
  "global": {
    "evt": {
      "xi": 0.18,
      "sigma": 0.0034,
      "threshold": 0.0125,
      "n_exceedances": 98,
      "cte": 0.0167,
      "fit_success": true,
      "method": "mle",
      "is_heavy_tailed": false,
      "has_finite_mean": true,
      "implied_student_t_nu": 5.56
    },
    "evt_diagnostics": {
      "fit_success": true,
      "n_losses": 520,
      "n_total_obs": 1260
    },
    "evt_student_t_consistency": {
      "nu_student_t": 6.0,
      "xi_gpd": 0.18,
      "consistent": true,
      "relative_difference": 0.08
    }
  }
}
```

## Signal Dataclass Fields

New fields added to `Signal`:

| Field | Type | Description |
|-------|------|-------------|
| `expected_loss_empirical` | float | Naive empirical E[loss] for comparison |
| `evt_enabled` | bool | Whether EVT was used |
| `evt_xi` | float | GPD shape parameter |
| `evt_sigma` | float | GPD scale parameter |
| `evt_threshold` | float | POT threshold |
| `evt_n_exceedances` | int | Number of tail exceedances |
| `evt_fit_method` | str | 'mle', 'hill', 'pwm', or fallback |

## Fallback Behavior

| Condition | Behavior |
|-----------|----------|
| Insufficient losses (< 30) | 1.5× empirical E[loss] |
| GPD fitting fails | 1.5× empirical E[loss] |
| ξ ≥ 1 (infinite mean) | Conservative large estimate |
| EVT not available | 1.5× empirical E[loss] |

## Expected Impact on Position Sizing

Based on the expert panel assessment:

| Asset Tail Type | ξ Range | Position Size Change |
|-----------------|---------|---------------------|
| Heavy (Student-t ν<5) | ξ > 0.2 | -20% to -35% |
| Moderate (ν=5-10) | 0.1 < ξ < 0.2 | -10% to -20% |
| Light (ν>10) | ξ < 0.1 | -5% to -10% |
| Gaussian-like | ξ ≈ 0 | ~0% (minimal) |

## UX Display

When EVT is fitted, the model indicator shows tail severity:

```
✓ AAPL: Student-t+EVTH|q=1.2e-06|c=0.92|φ=+0.87|ν=8|ξ=0.25|bic=1234|T=72%✓
```

- `+EVTH` — Heavy tails (ξ > 0.2)
- `+EVTM` — Moderate tails (0.05 < ξ ≤ 0.2)
- `+EVTL` — Light tails (ξ ≤ 0.05)

## Fitting Methods

Three GPD fitting methods are available, with automatic fallback:

1. **MLE (Maximum Likelihood)** — Default, most efficient
2. **Hill Estimator** — More robust for small samples
3. **PWM (Probability Weighted Moments)** — Alternative robust method

Priority order: MLE → Hill → PWM

## Core Principle

> **"Finite MC samples cannot capture rare tail events. EVT provides principled extrapolation."**

The naive empirical mean of losses from 1000 MC samples cannot reliably estimate 1-in-10,000 events. GPD's parametric form extrapolates beyond observed data based on the mathematical structure of extreme values.

## References

1. Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics*.
2. Balkema, A.A. & de Haan, L. (1974). "Residual Life Time at Great Age." *Annals of Probability*.
3. McNeil, A.J. & Frey, R. (2000). "Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series." *Journal of Empirical Finance*.
4. Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.
