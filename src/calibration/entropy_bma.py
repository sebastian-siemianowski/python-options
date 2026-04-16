"""
Story 6.1: Entropy-Regularized BMA Weights
===========================================

Prevents model collapse in BMA by adding an entropy penalty to the weight
optimization. Standard BIC-based BMA often produces sparse posteriors where
one model gets 95%+ weight, which is dangerous (fragility, no hedging,
overconfidence).

The entropy-regularized objective:
    w* = argmax_w [ sum_m w_m * ell_m  -  (1/tau) * sum_m w_m * log(w_m) ]

Temperature tau controls regularization:
    tau -> 0  gives uniform weights (maximum entropy)
    tau -> inf gives standard BIC weights (no regularization)

Story 6.2: Minimum Description Length Model Averaging
=====================================================

MDL-based weights as a BIC alternative. MDL penalizes complexity more
accurately for finite samples using Fisher information.

Story 6.3: Hierarchical BMA with Asset-Class Grouping
=====================================================

Hierarchical BMA shares model preference information within asset classes,
so assets with short histories borrow strength from similar assets.
"""
import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_WEIGHT = 0.80           # No single model can dominate > 80%
MIN_WEIGHT_FACTOR = 0.20    # Floor = 1 / (5 * M), i.e., 1/(5M)
DEFAULT_TAU = 1.0           # Default temperature
M_EFF_TARGET = 3.0          # Target effective number of models
TAU_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
WEIGHT_EPS = 1e-15          # Epsilon for log(w) stability


@dataclass
class EntropyBMAResult:
    """Result of entropy-regularized BMA weight computation."""
    weights: np.ndarray          # Regularized weights (M,)
    tau: float                   # Temperature used
    m_eff: float                 # Effective number of models
    max_weight: float            # Maximum weight
    entropy: float               # Shannon entropy of weights
    bic_weights: np.ndarray      # Original BIC weights for comparison
    tau_auto_tuned: bool         # Whether tau was auto-tuned


def _compute_bic_weights(log_likelihoods: np.ndarray,
                          n_params: np.ndarray,
                          n_obs: int) -> np.ndarray:
    """
    Compute standard BIC-based BMA weights.

    BIC_m = -2 * ell_m + k_m * log(n)
    w_m = exp(-BIC_m / 2) / sum exp(-BIC_j / 2)
    """
    bic = -2.0 * log_likelihoods + n_params * np.log(n_obs)
    # Shift for numerical stability
    bic_shifted = bic - np.min(bic)
    raw_weights = np.exp(-0.5 * bic_shifted)
    total = np.sum(raw_weights)
    if total < 1e-300:
        return np.ones(len(log_likelihoods)) / len(log_likelihoods)
    return raw_weights / total


def _entropy(w: np.ndarray) -> float:
    """Shannon entropy of weight vector."""
    w_safe = np.clip(w, WEIGHT_EPS, None)
    return -float(np.sum(w_safe * np.log(w_safe)))


def _m_eff(w: np.ndarray) -> float:
    """Effective number of models: exp(entropy)."""
    return float(np.exp(_entropy(w)))


def _entropy_regularized_weights(
    log_likelihoods: np.ndarray,
    n_params: np.ndarray,
    n_obs: int,
    tau: float,
) -> np.ndarray:
    """
    Compute entropy-regularized BMA weights at given temperature tau.

    The solution to the entropy-regularized optimization has closed form:
        w_m = exp(tau * ell_m - k_m * log(n) / 2) / Z

    This is equivalent to BIC with a temperature scaling on the likelihood.
    """
    M = len(log_likelihoods)
    min_weight = 1.0 / (5.0 * M)

    # Compute log-unnormalized weights
    bic_term = log_likelihoods - 0.5 * n_params * np.log(n_obs)

    # Temperature scaling: higher tau -> more peaked, lower tau -> more uniform
    log_w = tau * bic_term

    # Shift for stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w = w / np.sum(w)

    # Enforce floor + cap via iterative redistribution
    for _ in range(50):
        # Apply floor: lift any weight below min_weight
        below = w < min_weight
        if np.any(below):
            deficit = np.sum(min_weight - w[below])
            w[below] = min_weight
            # Take deficit from unconstrained weights proportionally
            free = ~below & (w > min_weight)
            if np.any(free):
                w[free] -= deficit * (w[free] / np.sum(w[free]))

        # Apply cap: reduce any weight above MAX_WEIGHT
        above = w > MAX_WEIGHT
        if np.any(above):
            excess = np.sum(w[above] - MAX_WEIGHT)
            w[above] = MAX_WEIGHT
            # Distribute excess to uncapped weights
            free = ~above & (w < MAX_WEIGHT)
            if np.any(free):
                w[free] += excess * (w[free] / np.sum(w[free]))

        # Check convergence
        if np.all(w >= min_weight - 1e-12) and np.all(w <= MAX_WEIGHT + 1e-12):
            break

    # Final normalization to handle any floating-point drift
    w = np.clip(w, min_weight, MAX_WEIGHT)
    w = w / np.sum(w)

    return w


def entropy_regularized_bma(
    log_likelihoods: np.ndarray,
    n_params: np.ndarray,
    n_obs: int,
    tau: Optional[float] = None,
    crps_matrix: Optional[np.ndarray] = None,
) -> EntropyBMAResult:
    """
    Compute entropy-regularized BMA weights.

    Parameters
    ----------
    log_likelihoods : ndarray, shape (M,)
        Log-likelihood for each of M models.
    n_params : ndarray, shape (M,)
        Number of parameters for each model.
    n_obs : int
        Number of observations.
    tau : float, optional
        Temperature parameter. If None, auto-tuned via LOO-CRPS if
        crps_matrix is provided, otherwise uses DEFAULT_TAU.
    crps_matrix : ndarray, shape (T, M), optional
        Per-observation CRPS for each model (for tau auto-tuning).

    Returns
    -------
    EntropyBMAResult
    """
    log_likelihoods = np.asarray(log_likelihoods, dtype=float)
    n_params = np.asarray(n_params, dtype=float)
    M = len(log_likelihoods)

    if M < 2:
        w = np.ones(M)
        return EntropyBMAResult(
            weights=w,
            tau=0.0,
            m_eff=1.0,
            max_weight=1.0,
            entropy=0.0,
            bic_weights=w.copy(),
            tau_auto_tuned=False,
        )

    # Compute baseline BIC weights
    bic_weights = _compute_bic_weights(log_likelihoods, n_params, n_obs)

    auto_tuned = False

    if tau is not None:
        # Use provided tau directly (no adaptive reduction)
        w = _entropy_regularized_weights(log_likelihoods, n_params, n_obs, tau)
    elif crps_matrix is not None:
        # Auto-tune tau via LOO-CRPS
        tau, w = _auto_tune_tau(log_likelihoods, n_params, n_obs, crps_matrix)
        auto_tuned = True
        # Ensure M_eff target via adaptive tau reduction
        for _ in range(15):
            if _m_eff(w) >= M_EFF_TARGET or tau < 0.01:
                break
            tau *= 0.5
            w = _entropy_regularized_weights(log_likelihoods, n_params, n_obs, tau)
    else:
        # Use default tau with adaptive reduction for M_eff target
        tau = DEFAULT_TAU
        w = _entropy_regularized_weights(log_likelihoods, n_params, n_obs, tau)
        for _ in range(15):
            if _m_eff(w) >= M_EFF_TARGET or tau < 0.01:
                break
            tau *= 0.5
            w = _entropy_regularized_weights(log_likelihoods, n_params, n_obs, tau)

    return EntropyBMAResult(
        weights=w,
        tau=tau,
        m_eff=_m_eff(w),
        max_weight=float(np.max(w)),
        entropy=_entropy(w),
        bic_weights=bic_weights,
        tau_auto_tuned=auto_tuned,
    )


def _auto_tune_tau(
    log_likelihoods: np.ndarray,
    n_params: np.ndarray,
    n_obs: int,
    crps_matrix: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Auto-tune tau by minimizing combined CRPS over the tau grid.

    For each candidate tau, compute weights, then evaluate
    combined_crps = sum_t sum_m w_m * crps_matrix[t, m].
    Select tau that minimizes combined CRPS.
    """
    best_tau = DEFAULT_TAU
    best_crps = float('inf')
    best_w = None

    for tau_candidate in TAU_GRID:
        w = _entropy_regularized_weights(log_likelihoods, n_params, n_obs, tau_candidate)
        # Combined CRPS: mean over time of weighted CRPS
        combined = float(np.mean(crps_matrix @ w))
        if combined < best_crps:
            best_crps = combined
            best_tau = tau_candidate
            best_w = w

    if best_w is None:
        best_w = _entropy_regularized_weights(log_likelihoods, n_params, n_obs, DEFAULT_TAU)

    return best_tau, best_w


# ---------------------------------------------------------------------------
# Story 6.2: Minimum Description Length Model Averaging
# ---------------------------------------------------------------------------

@dataclass
class MDLResult:
    """Result of MDL-based model weight computation."""
    weights: np.ndarray          # MDL weights (M,)
    mdl_scores: np.ndarray       # MDL scores per model (lower = better)
    bic_scores: np.ndarray       # BIC scores (half-scale) for comparison
    fisher_penalties: np.ndarray  # Fisher information penalty per model
    bic_weights: np.ndarray      # Standard BIC weights for comparison


def _estimate_fisher_logdet(
    log_likelihoods: np.ndarray,
    n_params: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """
    Estimate log|I_1(theta*)| from log-likelihoods.

    For Gaussian models: sigma^2 = exp(-2*ell/n - 1 - log(2pi))
    Per-parameter Fisher info: -log(sigma^2) = 2*ell/n + 1 + log(2pi)
    log|I_1| = k * per_param_fisher

    This gives positive Fisher info for well-fit models (typical daily returns),
    and larger penalties for complex models with high Fisher curvature.
    """
    ll = np.asarray(log_likelihoods, dtype=float)
    k = np.asarray(n_params, dtype=float)
    n = float(n_obs)

    per_param_fisher = 2.0 * ll / n + 1.0 + np.log(2.0 * np.pi)
    return k * per_param_fisher


def mdl_weights(
    log_likelihoods: np.ndarray,
    n_params: np.ndarray,
    n_obs: int,
    fisher_info_logdet: Optional[np.ndarray] = None,
) -> MDLResult:
    """
    Compute MDL-based model weights.

    MDL formula (Rissanen's two-part code):
        MDL_i = -ell_i + (k_i/2) * log(n/(2pi)) + (1/2) * log|I_1(theta_i)|

    Where I_1 is the per-observation Fisher information matrix.

    Properties:
        - For n > 500: MDL ~ BIC (constant Fisher term negligible)
        - For n < 200: MDL penalizes complexity more than BIC
          (Fisher penalty amplifies k for well-fit models)
        - Selects simpler models for short-history assets

    Parameters
    ----------
    log_likelihoods : ndarray, shape (M,)
        Log-likelihood for each model at MLE.
    n_params : ndarray, shape (M,)
        Number of parameters for each model.
    n_obs : int
        Number of observations.
    fisher_info_logdet : ndarray, shape (M,), optional
        log|I_1(theta*)| for each model. If None, estimated from
        log-likelihoods via Gaussian approximation.

    Returns
    -------
    MDLResult
    """
    ll = np.asarray(log_likelihoods, dtype=float)
    k = np.asarray(n_params, dtype=float)
    n = float(n_obs)
    M = len(ll)

    # BIC scores (half-scale): -ell + (k/2)*log(n)
    bic_scores = -ll + (k / 2.0) * np.log(n)

    # BIC weights
    bic_weights = _compute_bic_weights(ll, k, n_obs)

    if M < 2:
        return MDLResult(
            weights=np.ones(M),
            mdl_scores=-ll,
            bic_scores=bic_scores,
            fisher_penalties=np.zeros(M),
            bic_weights=bic_weights,
        )

    # Fisher information
    if fisher_info_logdet is not None:
        log_det_I1 = np.asarray(fisher_info_logdet, dtype=float)
    else:
        log_det_I1 = _estimate_fisher_logdet(ll, k, n_obs)

    fisher_penalties = 0.5 * log_det_I1

    # MDL scores: -ell + (k/2)*log(n/(2pi)) + (1/2)*log|I_1|
    mdl_scores = -ll + (k / 2.0) * np.log(n / (2.0 * np.pi)) + fisher_penalties

    # Convert to weights via softmin (lower MDL = better)
    delta = mdl_scores - np.min(mdl_scores)
    log_w = -delta
    # Numerical stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total < 1e-300:
        w = np.ones(M) / M
    else:
        w = w / total

    return MDLResult(
        weights=w,
        mdl_scores=mdl_scores,
        bic_scores=bic_scores,
        fisher_penalties=fisher_penalties,
        bic_weights=bic_weights,
    )


# ---------------------------------------------------------------------------
# Story 6.3: Hierarchical BMA with Asset-Class Grouping
# ---------------------------------------------------------------------------

SHRINKAGE_N_REF = 500.0  # Reference sample size for shrinkage calibration
ASSET_CLASSES = frozenset([
    "Large Cap", "Mid Cap", "Small Cap", "Index", "Metals", "Crypto",
])


@dataclass
class AssetBMAInput:
    """Input for one asset in hierarchical BMA."""
    symbol: str
    weights: np.ndarray           # BMA weights (M,)
    log_likelihoods: np.ndarray   # Per-model log-likelihoods (M,)
    n_params: np.ndarray          # Per-model parameter counts (M,)
    n_obs: int
    asset_class: str


@dataclass
class HierarchicalBMAResult:
    """Result of hierarchical BMA pooling."""
    asset_weights: dict       # symbol -> hierarchical weights ndarray
    group_priors: dict        # asset_class -> prior weights ndarray
    shrinkage_factors: dict   # symbol -> alpha float
    original_weights: dict    # symbol -> original weights ndarray


def _compute_shrinkage(n_obs: int, n_ref: float = SHRINKAGE_N_REF) -> float:
    """Shrinkage factor: more borrowing for fewer observations.

    alpha = n_ref / (n_obs + n_ref)
    - n_obs = 100: alpha = 0.83 (strong borrowing)
    - n_obs = 500: alpha = 0.50 (moderate)
    - n_obs = 2000: alpha = 0.20 (weak)
    """
    return n_ref / (float(n_obs) + n_ref)


def hierarchical_bma(
    assets: list,
) -> HierarchicalBMAResult:
    """
    Hierarchical BMA: pool information within asset-class groups.

    For each asset, shrink its BMA weights toward the group prior
    (mean weights of all assets in the same class). Shrinkage is
    proportional to 1/n_obs: assets with less data borrow more
    from the group.

    Parameters
    ----------
    assets : list of AssetBMAInput
        Each asset must have: symbol, weights, log_likelihoods,
        n_params, n_obs, asset_class.

    Returns
    -------
    HierarchicalBMAResult
    """
    if not assets:
        return HierarchicalBMAResult(
            asset_weights={},
            group_priors={},
            shrinkage_factors={},
            original_weights={},
        )

    # --- Step 1: Group assets by class ---
    groups: dict = {}
    for a in assets:
        groups.setdefault(a.asset_class, []).append(a)

    # --- Step 2: Compute group prior (mean weights per class) ---
    group_priors: dict = {}
    for cls, members in groups.items():
        # All members must have same number of models
        M = len(members[0].weights)
        stacked = np.zeros((len(members), M))
        for i, m in enumerate(members):
            w = np.asarray(m.weights, dtype=float)
            if len(w) != M:
                # Pad or truncate to match first member
                padded = np.zeros(M)
                padded[:min(len(w), M)] = w[:M]
                total = np.sum(padded)
                if total > 0:
                    padded /= total
                else:
                    padded = np.ones(M) / M
                stacked[i] = padded
            else:
                stacked[i] = w
        prior = np.mean(stacked, axis=0)
        # Normalize
        total = np.sum(prior)
        if total > 0:
            prior = prior / total
        else:
            prior = np.ones(M) / M
        group_priors[cls] = prior

    # --- Step 3: Shrink each asset toward its group prior ---
    asset_weights: dict = {}
    shrinkage_factors: dict = {}
    original_weights: dict = {}

    for a in assets:
        alpha = _compute_shrinkage(a.n_obs)
        w_orig = np.asarray(a.weights, dtype=float)
        w_prior = group_priors[a.asset_class]

        # Handle dimension mismatch
        M = len(w_orig)
        if len(w_prior) != M:
            prior = np.ones(M) / M
        else:
            prior = w_prior

        # Shrink
        w_hier = (1.0 - alpha) * w_orig + alpha * prior

        # Normalize
        total = np.sum(w_hier)
        if total > 0:
            w_hier = w_hier / total
        else:
            w_hier = np.ones(M) / M

        asset_weights[a.symbol] = w_hier
        shrinkage_factors[a.symbol] = alpha
        original_weights[a.symbol] = w_orig

    return HierarchicalBMAResult(
        asset_weights=asset_weights,
        group_priors=group_priors,
        shrinkage_factors=shrinkage_factors,
        original_weights=original_weights,
    )
