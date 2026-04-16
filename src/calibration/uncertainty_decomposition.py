"""
Story 11.2: Confidence Decomposition -- Epistemic vs Aleatoric Uncertainty
==========================================================================

Decomposes total predictive uncertainty into:
  - Epistemic (reducible): inter-model disagreement sigma_e^2
  - Aleatoric (irreducible): within-model noise sigma_a^2

Key insight: High epistemic -> models disagree -> reduce position size.
High aleatoric only -> direction known but inherent noise -> normal sizing.

The law of total variance:
    Var(Y) = E[Var(Y|M)] + Var(E[Y|M])
           = aleatoric     + epistemic

References:
    Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
    Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty Estimation"
"""
import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ===================================================================
# Constants
# ===================================================================

EPISTEMIC_HIGH_THRESHOLD = 0.5     # Epistemic fraction > 50% -> high epistemic
EPISTEMIC_SIZING_DISCOUNT = 0.5    # Reduce position size by 50% when epistemic dominates
MIN_TOTAL_VARIANCE = 1e-20         # Floor for total variance to avoid division by zero
SIZING_EPISTEMIC_POWER = 1.0       # Power for epistemic sizing discount curve


# ===================================================================
# Result Dataclass
# ===================================================================

@dataclass
class UncertaintyDecomposition:
    """Result of epistemic/aleatoric uncertainty decomposition."""
    epistemic_var: float       # Var(E[Y|M]) - inter-model disagreement
    aleatoric_var: float       # E[Var(Y|M)] - within-model noise
    total_var: float           # epistemic + aleatoric
    epistemic_fraction: float  # epistemic / total (0 to 1)
    n_models: int              # Number of models in ensemble
    ensemble_mean: float       # BMA weighted mean prediction
    ensemble_std: float        # sqrt(total_var)
    model_means: np.ndarray    # Individual model means
    model_stds: np.ndarray     # Individual model stds


# ===================================================================
# Core Functions
# ===================================================================

def decompose_uncertainty(
    model_means: np.ndarray,
    model_stds: np.ndarray,
    weights: np.ndarray,
) -> UncertaintyDecomposition:
    """Decompose predictive uncertainty into epistemic and aleatoric components.

    Uses the law of total variance:
        sigma_total^2 = sigma_aleatoric^2 + sigma_epistemic^2

    Where:
        sigma_aleatoric^2 = sum(w_m * sigma_m^2)     (within-model variance)
        sigma_epistemic^2 = sum(w_m * (mu_m - mu_bar)^2)  (inter-model disagreement)
        mu_bar = sum(w_m * mu_m)

    Args:
        model_means: Mean predictions from each model, shape (n_models,)
        model_stds: Standard deviations from each model, shape (n_models,)
        weights: BMA weights, shape (n_models,), must sum to ~1

    Returns:
        UncertaintyDecomposition
    """
    model_means = np.asarray(model_means, dtype=float)
    model_stds = np.asarray(model_stds, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_models = len(model_means)

    # Normalize weights to sum to 1
    w_sum = weights.sum()
    if w_sum > 0:
        w = weights / w_sum
    else:
        w = np.ones(n_models) / n_models

    # Ensemble mean
    mu_bar = np.dot(w, model_means)

    # Aleatoric: expected within-model variance
    sigma_a2 = np.dot(w, model_stds ** 2)

    # Epistemic: variance of model means
    sigma_e2 = np.dot(w, (model_means - mu_bar) ** 2)

    # Total
    total_var = sigma_a2 + sigma_e2
    total_var = max(total_var, MIN_TOTAL_VARIANCE)

    # Epistemic fraction
    epistemic_frac = sigma_e2 / total_var

    return UncertaintyDecomposition(
        epistemic_var=float(sigma_e2),
        aleatoric_var=float(sigma_a2),
        total_var=float(total_var),
        epistemic_fraction=float(epistemic_frac),
        n_models=n_models,
        ensemble_mean=float(mu_bar),
        ensemble_std=float(math.sqrt(total_var)),
        model_means=model_means,
        model_stds=model_stds,
    )


def compute_position_sizing_factor(
    decomposition: UncertaintyDecomposition,
    epistemic_threshold: float = EPISTEMIC_HIGH_THRESHOLD,
    discount: float = EPISTEMIC_SIZING_DISCOUNT,
) -> float:
    """Compute position sizing factor based on epistemic uncertainty.

    When epistemic uncertainty is high (models disagree), reduce position size.
    When only aleatoric (inherent noise), maintain normal sizing.

    Args:
        decomposition: UncertaintyDecomposition result
        epistemic_threshold: Fraction above which epistemic is "high"
        discount: Maximum discount factor when epistemic is high

    Returns:
        Sizing factor in (0, 1]. 1.0 = full size, lower = reduced.
    """
    ef = decomposition.epistemic_fraction

    if ef <= epistemic_threshold:
        # Below threshold: full sizing
        return 1.0

    # Above threshold: linearly discount from 1.0 to (1 - discount)
    # at epistemic_fraction = 1.0
    excess = (ef - epistemic_threshold) / (1.0 - epistemic_threshold)
    factor = 1.0 - discount * excess

    return max(0.1, float(factor))  # Floor at 10% position


def decompose_timeseries(
    model_means_ts: np.ndarray,
    model_stds_ts: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose uncertainty for a time series of predictions.

    Args:
        model_means_ts: shape (T, n_models) - means at each time step
        model_stds_ts: shape (T, n_models) - stds at each time step
        weights: shape (n_models,) - BMA weights (constant or per-step)

    Returns:
        (epistemic_var_ts, aleatoric_var_ts, epistemic_frac_ts) each shape (T,)
    """
    T = model_means_ts.shape[0]
    epistemic_ts = np.zeros(T)
    aleatoric_ts = np.zeros(T)
    frac_ts = np.zeros(T)

    for t in range(T):
        d = decompose_uncertainty(
            model_means_ts[t], model_stds_ts[t], weights
        )
        epistemic_ts[t] = d.epistemic_var
        aleatoric_ts[t] = d.aleatoric_var
        frac_ts[t] = d.epistemic_fraction

    return epistemic_ts, aleatoric_ts, frac_ts


def compute_model_agreement(
    model_means: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute model agreement score (0 = total disagreement, 1 = full agreement).

    Agreement is 1 - epistemic_fraction, measuring how much models agree on direction.

    Args:
        model_means: Mean predictions from each model, shape (n_models,)
        weights: BMA weights, shape (n_models,)

    Returns:
        Agreement score in [0, 1]
    """
    model_stds = np.zeros_like(model_means)  # Only care about means for agreement
    d = decompose_uncertainty(model_means, model_stds, weights)
    # When all means are the same, epistemic = 0, total = aleatoric = 0 -> fraction = 0
    # But with zero stds, total_var could be very small
    if d.total_var < MIN_TOTAL_VARIANCE * 10:
        return 1.0  # All models agree (and no variance)
    return 1.0 - d.epistemic_fraction


def sign_agreement_fraction(
    model_means: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Fraction of BMA weight that agrees on the sign of the prediction.

    Args:
        model_means: Mean predictions from each model, shape (n_models,)
        weights: BMA weights, shape (n_models,)

    Returns:
        Fraction of weight on the majority sign direction, in [0.5, 1.0]
    """
    weights = np.asarray(weights, dtype=float)
    model_means = np.asarray(model_means, dtype=float)

    w_sum = weights.sum()
    if w_sum <= 0:
        return 0.5

    w = weights / w_sum

    positive_weight = np.dot(w, (model_means > 0).astype(float))
    negative_weight = np.dot(w, (model_means < 0).astype(float))

    return max(positive_weight, negative_weight)
