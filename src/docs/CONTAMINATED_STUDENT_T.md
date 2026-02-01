# Contaminated Student-t Mixture Model Integration

## Overview

This document describes the implementation of the **Regime-Indexed Contaminated Student-t** model, a hybrid approach recommended by the Expert Panel for capturing distinct fat-tail behavior in normal versus stressed market regimes.

## Expert Panel Recommendation

This implementation follows the **hybrid of Professor Zhang's Solution 1 with Professor Li's Solution 3**, combining:

1. **Regime-Indexed Parameter Cache** (Prof. Zhang, Score: 9.0/10): Pre-computed parameters, deterministic behavior
2. **Contaminated Student-t Model** (Prof. Li, Score: 8.5/10): Parsimonious, interpretable, robust

### Consensus Scores Summary

| Criterion | Prof. Chen | Prof. Li | Prof. Zhang | Average |
|-----------|------------|----------|-------------|---------|
| Regime-Conditional Student-t | 8.0 | 5.8 | 9.0 | **7.6** |
| Contaminated Student-t | 7.3 | 8.5 | 6.8 | **7.5** |
| **Hybrid (Selected)** | — | — | — | **8.3** |

## Mathematical Model

The Contaminated Student-t distribution has density:

$$p(r) = (1 - \varepsilon) \cdot t(r; \nu_{\text{normal}}) + \varepsilon \cdot t(r; \nu_{\text{crisis}})$$

Where:
- **ν_normal**: Degrees of freedom for typical market conditions (lighter tails, e.g., 12)
- **ν_crisis**: Degrees of freedom for extreme events (heavier tails, e.g., 4)
- **ε**: Contamination probability (linked to vol_regime, typically 5-15%)

### Core Principle

> *"5% of the time we're in crisis mode with ν=4, 95% of time we're normal with ν=12"*

This provides intuitive risk management interpretation while maintaining statistical rigor.

## Key Properties

### Tail Behavior

| Component | ν | Tail Description |
|-----------|---|------------------|
| Normal | 12 | Moderate heavy tails, finite 4th moment |
| Crisis | 4 | Very heavy tails, borderline finite variance |

### Effective ν (Harmonic Mean)

$$\nu_{\text{eff}} = \frac{1}{(1-\varepsilon)/\nu_{\text{normal}} + \varepsilon/\nu_{\text{crisis}}}$$

For ε=0.10, ν_normal=12, ν_crisis=4: ν_eff ≈ 9.6

### Graceful Degradation

- If ε → 0: Model collapses to single t(ν_normal)
- If ν_crisis → ν_normal: Model collapses to single Student-t
- Both provide natural fallback behavior

## Integration Architecture

### Phase 1: Tuning (`tune.py`)

```
Returns → Standardize → Profile Likelihood Search over (ν_normal, ν_crisis, ε)
       → Compute BIC comparison with single Student-t
       → Store parameters in global_data["contaminated_student_t"]
```

Stored parameters:
```json
{
  "contaminated_student_t": {
    "nu_normal": 12,
    "nu_crisis": 4,
    "epsilon": 0.08,
    "epsilon_source": "mle",
    "effective_nu": 9.2,
    "is_degenerate": false,
    "tail_heaviness_ratio": 3.0
  },
  "contaminated_student_t_diagnostics": {
    "fit_success": true,
    "delta_bic": -5.2,
    "mixture_preferred": true
  }
}
```

### Phase 2: Signal Generation (`signals.py`)

```
Extract cst_params from tuned_params
    ↓
For each Student-t model in BMA:
    ↓
run_regime_specific_mc() with contaminated sampling:
    - With probability ε: sample from t(ν_crisis)
    - With probability (1-ε): sample from t(ν_normal)
    ↓
Aggregate samples for Expected Utility calculation
```

### Sampling Implementation

```python
# In run_regime_specific_mc():
if use_contaminated_t:
    samples = contaminated_student_t_rvs(
        size=n_paths,
        nu_normal=cst_nu_normal,
        nu_crisis=cst_nu_crisis,
        epsilon=cst_epsilon,
        random_state=rng
    )
```

## Files Created/Modified

### Created:
- `src/models/contaminated_student_t.py` — Core distribution implementation
- `src/tests/test_contaminated_student_t.py` — Comprehensive test suite
- `src/docs/CONTAMINATED_STUDENT_T.md` — This documentation

### Modified:
- `src/models/__init__.py` — Added exports for contaminated_student_t
- `src/tuning/tune.py` — Added profile likelihood fitting in `tune_asset_q()`
- `src/decision/signals.py` — Updated `run_regime_specific_mc()` for contaminated sampling
- `src/tuning/tune_ux.py` — Added "+CST{ε}%" display indicator

## Signal Dataclass Fields

New fields added to `Signal`:

| Field | Type | Description |
|-------|------|-------------|
| `cst_enabled` | bool | Whether contaminated mixture was used |
| `cst_nu_normal` | float | Normal regime ν |
| `cst_nu_crisis` | float | Crisis regime ν |
| `cst_epsilon` | float | Crisis contamination probability |

## UX Display

When Contaminated Student-t is fitted, the model indicator shows:

```
✓ AAPL: Student-t+GMM+CST8%|q=1.2e-06|c=0.92|φ=+0.87|ν=12|ν_c=4|bic=1234|T=72%✓
```

- `+CST8%` — Contaminated Student-t with 8% crisis probability
- `ν_c=4` — Crisis component degrees of freedom

## Epsilon Determination

The contamination probability ε can be determined by:

1. **MLE (Default)**: Optimized via profile likelihood
2. **Vol-Regime Based**: Linked to historical high-volatility fraction
3. **Blended**: 50% MLE + 50% vol-regime

```python
if vol_regime_labels is not None:
    epsilon_from_regime = mean(vol_regime_labels == 1)
    epsilon_final = 0.5 * epsilon_mle + 0.5 * epsilon_from_regime
```

## Position Sizing Impact

The contaminated mixture produces heavier tails than single Student-t, leading to:

- **Larger E[loss]** estimates (more conservative)
- **Smaller position sizes** during crisis-prone regimes
- **Adaptive risk management** based on market conditions

### Example Impact

| Scenario | Single t(12) E[loss] | Contaminated E[loss] | Position Change |
|----------|---------------------|---------------------|-----------------|
| Calm (ε=5%) | 0.015 | 0.017 | -12% |
| Elevated (ε=10%) | 0.015 | 0.020 | -25% |
| Crisis (ε=20%) | 0.015 | 0.028 | -45% |

## Comparison with Single Student-t

The profile likelihood fitting compares mixture vs single Student-t using BIC:

```
BIC_mixture = -2 × LL_mixture + 3 × log(n)  # 3 params
BIC_single = -2 × LL_single + 1 × log(n)    # 1 param

delta_BIC = BIC_mixture - BIC_single
```

- **delta_BIC < -2**: Strong evidence for mixture
- **-2 ≤ delta_BIC ≤ 2**: Inconclusive
- **delta_BIC > 2**: Single Student-t preferred

## Parameter Constraints

| Parameter | Range | Default |
|-----------|-------|---------|
| ν_normal | [6, 50] | 12 |
| ν_crisis | [3, ν_normal] | 4 |
| ε | [0.01, 0.30] | 0.05 |

**Constraint**: ν_crisis ≤ ν_normal (crisis has heavier tails)

## Testing Strategy

1. **Unit Tests**: PDF integrates to 1, serialization round-trip
2. **Sampling Tests**: Mixing fraction approximately ε
3. **Fitting Tests**: Recovers parameters from known mixture
4. **Integration Tests**: End-to-end signal generation with mixture

## References

1. Tukey, J.W. (1960). "A Survey of Sampling from Contaminated Distributions." *Contributions to Probability and Statistics*.
2. Lange, K.L., Little, R.J.A., Taylor, J.M.G. (1989). "Robust Statistical Modeling Using the t Distribution." *Journal of the American Statistical Association*.
3. Andrews, D.F. & Mallows, C.L. (1974). "Scale Mixtures of Normal Distributions." *Journal of the Royal Statistical Society*.
