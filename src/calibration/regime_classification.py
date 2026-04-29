"""
Epic 19: Probabilistic Regime Classification
==============================================

Replaces hard regime boundaries with soft sigmoid transitions and
temporal dynamics via HMM, plus per-regime forecast quality tracking.

Story 19.1: soft_regime_membership     -- Sigmoid-based soft membership
Story 19.2: hmm_regime_fit            -- Hidden Markov Model for regime dynamics
Story 19.3: regime_forecast_quality   -- Per-regime forecast quality tracking
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit  # sigmoid

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ===================================================================
# Constants
# ===================================================================

# Regime names
LOW_VOL_TREND = "LOW_VOL_TREND"
HIGH_VOL_TREND = "HIGH_VOL_TREND"
LOW_VOL_RANGE = "LOW_VOL_RANGE"
HIGH_VOL_RANGE = "HIGH_VOL_RANGE"
CRISIS_JUMP = "CRISIS_JUMP"

ALL_REGIMES = [LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE,
               HIGH_VOL_RANGE, CRISIS_JUMP]
N_REGIMES = len(ALL_REGIMES)

# Sigmoid transition parameters
# vol_relative boundaries (vol / median_vol)
VOL_BOUNDARY_LOW_HIGH = 1.3       # Low <-> High vol boundary
VOL_BOUNDARY_HIGH_CRISIS = 2.0    # High vol <-> Crisis boundary
VOL_BOUNDARY_LOW_CALM = 0.85      # Below -> low vol

# Sigmoid steepness: higher = sharper transitions
SIGMOID_STEEPNESS_VOL = 8.0       # For vol transitions
SIGMOID_STEEPNESS_DRIFT = 6.0     # For trend/range distinction

# Drift threshold (absolute drift / vol)
DRIFT_THRESHOLD = 0.05

# HMM defaults
HMM_N_ITER = 100
HMM_TOL = 1e-4
HMM_MIN_OBS = 50
HMM_PERSISTENCE_PRIOR = 0.95     # Prior for A_kk (regime stickiness)

# Forecast quality tracking
FQ_DEFAULT_WINDOW = 126           # ~6 months
FQ_MIN_HIT_RATE = 0.48           # Below this, suppress signals
FQ_MIN_OBS_REGIME = 20           # Min obs in regime for quality estimate

# CRPS helper epsilon
CRPS_EPS = 1e-12


# ===================================================================
# Story 19.1: Soft Regime Membership via Sigmoid Transitions
# ===================================================================

@dataclass(frozen=True)
class SoftRegimeMembership:
    """Result of soft regime classification.

    Attributes
    ----------
    probabilities : np.ndarray
        Shape (5,) array of regime membership probabilities summing to 1.
        Order: [LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE,
                HIGH_VOL_RANGE, CRISIS_JUMP]
    dominant_regime : str
        Most probable regime.
    dominant_prob : float
        Probability of the dominant regime.
    vol_relative : float
        Volatility relative to median.
    drift_relative : float
        Drift signal relative to volatility.
    """
    probabilities: np.ndarray
    dominant_regime: str
    dominant_prob: float
    vol_relative: float
    drift_relative: float


def soft_regime_membership(
    vol: float,
    drift: float,
    median_vol: float,
    steepness_vol: float = SIGMOID_STEEPNESS_VOL,
    steepness_drift: float = SIGMOID_STEEPNESS_DRIFT,
) -> SoftRegimeMembership:
    """Compute soft regime membership probabilities.

    Uses sigmoid transitions instead of hard thresholds. At boundaries,
    probabilities are smoothly shared between adjacent regimes.

    Parameters
    ----------
    vol : float
        Current EWMA volatility.
    drift : float
        Current drift estimate (absolute value used for trend/range).
    median_vol : float
        Median of historical volatility.
    steepness_vol : float
        Sigmoid steepness for volatility transitions.
    steepness_drift : float
        Sigmoid steepness for trend/range distinction.

    Returns
    -------
    SoftRegimeMembership
        Membership probabilities over 5 regimes.
    """
    median_vol_safe = max(abs(median_vol), 1e-12)
    vol_safe = max(abs(vol), 1e-12)
    vol_relative = vol_safe / median_vol_safe

    # Drift relative to volatility (trend strength)
    drift_relative = abs(drift) / vol_safe if vol_safe > 1e-12 else 0.0

    # --- Stage 1: Vol-level soft memberships ---
    # P(crisis) via sigmoid around crisis boundary
    p_crisis = float(expit(steepness_vol * (vol_relative - VOL_BOUNDARY_HIGH_CRISIS)))

    # P(high_vol) = not-crisis AND above low-high boundary
    p_high_raw = float(expit(steepness_vol * (vol_relative - VOL_BOUNDARY_LOW_HIGH)))
    p_high = p_high_raw * (1.0 - p_crisis)

    # P(low_vol) = remainder
    p_low = 1.0 - p_crisis - p_high

    # --- Stage 2: Trend/Range within vol level ---
    p_trend = float(expit(steepness_drift * (drift_relative - DRIFT_THRESHOLD)))
    p_range = 1.0 - p_trend

    # --- Compose into 5 regimes ---
    probs = np.zeros(N_REGIMES)
    probs[0] = p_low * p_trend        # LOW_VOL_TREND
    probs[1] = p_high * p_trend       # HIGH_VOL_TREND
    probs[2] = p_low * p_range        # LOW_VOL_RANGE
    probs[3] = p_high * p_range       # HIGH_VOL_RANGE
    probs[4] = p_crisis               # CRISIS_JUMP

    # Normalize (should already sum to ~1, but ensure)
    total = probs.sum()
    if total > 1e-12:
        probs /= total
    else:
        probs = np.ones(N_REGIMES) / N_REGIMES

    dominant_idx = int(np.argmax(probs))
    dominant_regime = ALL_REGIMES[dominant_idx]
    dominant_prob = float(probs[dominant_idx])

    return SoftRegimeMembership(
        probabilities=probs,
        dominant_regime=dominant_regime,
        dominant_prob=dominant_prob,
        vol_relative=vol_relative,
        drift_relative=drift_relative,
    )


def soft_regime_membership_array(
    vol: np.ndarray,
    drift: np.ndarray,
    median_vol: float,
) -> np.ndarray:
    """Compute soft regime membership for arrays.

    Returns
    -------
    np.ndarray
        Shape (n, 5) array of regime membership probabilities.
    """
    vol = np.asarray(vol, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    n = len(vol)
    result = np.zeros((n, N_REGIMES))

    for i in range(n):
        membership = soft_regime_membership(
            float(vol[i]), float(drift[i]), median_vol,
        )
        result[i] = membership.probabilities

    return result


def soft_bma_weights(
    regime_weights: Dict[str, np.ndarray],
    membership: np.ndarray,
) -> np.ndarray:
    """Compute BMA weights mixed across regimes using soft membership.

    Parameters
    ----------
    regime_weights : dict
        Mapping regime_name -> array of model BMA weights for that regime.
        All weight arrays must have the same length (n_models).
    membership : np.ndarray
        Shape (5,) soft regime membership probabilities.

    Returns
    -------
    np.ndarray
        Mixed BMA weights: w_mixed = sum_k pi_k * w_k
    """
    if not regime_weights:
        return np.array([])

    # Get n_models from first entry
    first_key = next(iter(regime_weights))
    n_models = len(regime_weights[first_key])
    mixed = np.zeros(n_models)

    for i, regime in enumerate(ALL_REGIMES):
        if regime in regime_weights:
            w_k = np.asarray(regime_weights[regime], dtype=np.float64)
            mixed += membership[i] * w_k

    # Normalize
    total = mixed.sum()
    if total > 1e-12:
        mixed /= total

    return mixed


# ===================================================================
# Story 19.2: Hidden Markov Model for Regime Dynamics
# ===================================================================

@dataclass
class HMMRegimeResult:
    """Result of HMM regime fitting.

    Attributes
    ----------
    transition_matrix : np.ndarray
        Shape (K, K) transition matrix A where A[i,j] = P(regime_j | regime_i).
    stationary_dist : np.ndarray
        Shape (K,) stationary distribution.
    filtered_probs : np.ndarray
        Shape (T, K) filtered regime probabilities at each timestep.
    emission_means : np.ndarray
        Shape (K,) emission means for each regime.
    emission_vars : np.ndarray
        Shape (K,) emission variances for each regime.
    n_iter : int
        Number of Baum-Welch iterations.
    log_likelihood : float
        Final log-likelihood.
    converged : bool
        Whether Baum-Welch converged.
    """
    transition_matrix: np.ndarray
    stationary_dist: np.ndarray
    filtered_probs: np.ndarray
    emission_means: np.ndarray
    emission_vars: np.ndarray
    n_iter: int
    log_likelihood: float
    converged: bool


def _gaussian_emission(x: float, mu: float, var: float) -> float:
    """Gaussian emission probability p(x | mu, var)."""
    var_safe = max(var, 1e-12)
    return math.exp(-0.5 * (x - mu) ** 2 / var_safe) / math.sqrt(
        2.0 * math.pi * var_safe
    )


def _forward_pass(obs: np.ndarray, A: np.ndarray, pi0: np.ndarray,
                   means: np.ndarray, variances: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Forward algorithm: compute alpha_t(k) = P(obs_{1:t}, regime_t=k).

    Returns alpha (T, K) and scale factors (T,).
    """
    T = len(obs)
    K = len(pi0)
    alpha = np.zeros((T, K))
    scales = np.zeros(T)

    # t=0
    for k in range(K):
        alpha[0, k] = pi0[k] * _gaussian_emission(obs[0], means[k], variances[k])
    scales[0] = alpha[0].sum()
    if scales[0] > 1e-300:
        alpha[0] /= scales[0]
    else:
        alpha[0] = np.ones(K) / K
        scales[0] = 1e-300

    # t=1..T-1
    for t in range(1, T):
        for k in range(K):
            alpha[t, k] = sum(
                alpha[t - 1, j] * A[j, k] for j in range(K)
            ) * _gaussian_emission(obs[t], means[k], variances[k])
        scales[t] = alpha[t].sum()
        if scales[t] > 1e-300:
            alpha[t] /= scales[t]
        else:
            alpha[t] = np.ones(K) / K
            scales[t] = 1e-300

    return alpha, scales


def _backward_pass(obs: np.ndarray, A: np.ndarray,
                    means: np.ndarray, variances: np.ndarray,
                    scales: np.ndarray) -> np.ndarray:
    """Backward algorithm: compute beta_t(k)."""
    T = len(obs)
    K = A.shape[0]
    beta = np.zeros((T, K))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        for k in range(K):
            for j in range(K):
                beta[t, k] += (
                    A[k, j]
                    * _gaussian_emission(obs[t + 1], means[j], variances[j])
                    * beta[t + 1, j]
                )
        if scales[t + 1] > 1e-300:
            beta[t] /= scales[t + 1]

    return beta


def _emission_matrix(obs: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Vectorized Gaussian emission probabilities, shape (T, K)."""
    var_safe = np.maximum(np.asarray(variances, dtype=np.float64), 1e-12)
    diff = obs[:, None] - means[None, :]
    denom = np.sqrt(2.0 * math.pi * var_safe)
    emissions = np.exp(-0.5 * diff * diff / var_safe[None, :]) / denom[None, :]
    return np.maximum(emissions, 1e-300)


def _forward_backward_fast(
    obs: np.ndarray,
    A: np.ndarray,
    pi0: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scaled forward/backward pass with vectorized emissions."""
    T = len(obs)
    K = len(pi0)
    emissions = _emission_matrix(obs, means, variances)
    alpha = np.empty((T, K), dtype=np.float64)
    beta = np.empty((T, K), dtype=np.float64)
    scales = np.empty(T, dtype=np.float64)

    alpha[0] = pi0 * emissions[0]
    scales[0] = float(alpha[0].sum())
    if scales[0] > 1e-300:
        alpha[0] /= scales[0]
    else:
        alpha[0] = 1.0 / K
        scales[0] = 1e-300

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * emissions[t]
        scales[t] = float(alpha[t].sum())
        if scales[t] > 1e-300:
            alpha[t] /= scales[t]
        else:
            alpha[t] = 1.0 / K
            scales[t] = 1e-300

    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (emissions[t + 1] * beta[t + 1])
        if scales[t + 1] > 1e-300:
            beta[t] /= scales[t + 1]

    return alpha, beta, scales, emissions


def hmm_regime_fit(
    vol: np.ndarray,
    drift: np.ndarray,
    n_regimes: int = N_REGIMES,
    n_iter: int = HMM_N_ITER,
    tol: float = HMM_TOL,
) -> HMMRegimeResult:
    """Fit a Gaussian HMM to vol/drift features via Baum-Welch.

    Uses the vol_relative as the primary observation sequence.

    Parameters
    ----------
    vol : array
        Volatility series.
    drift : array
        Drift series.
    n_regimes : int
        Number of hidden states (default 5).
    n_iter : int
        Maximum Baum-Welch iterations.
    tol : float
        Convergence tolerance for log-likelihood.

    Returns
    -------
    HMMRegimeResult
        Fitted HMM parameters and filtered probabilities.
    """
    vol = np.asarray(vol, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    T = len(vol)
    K = n_regimes

    if T < HMM_MIN_OBS:
        return HMMRegimeResult(
            transition_matrix=np.eye(K) * HMM_PERSISTENCE_PRIOR
            + np.ones((K, K)) * (1.0 - HMM_PERSISTENCE_PRIOR) / K,
            stationary_dist=np.ones(K) / K,
            filtered_probs=np.ones((T, K)) / K,
            emission_means=np.linspace(0.5, 2.5, K),
            emission_vars=np.ones(K) * 0.5,
            n_iter=0,
            log_likelihood=-1e12,
            converged=False,
        )

    # Use vol as observation (could also use vol + drift features)
    # Normalize vol to have mean ~1.0
    median_vol = np.median(vol[vol > 1e-12]) if np.any(vol > 1e-12) else 1.0
    obs = vol / max(median_vol, 1e-12)
    obs = np.clip(obs, 0.01, 10.0)

    # Initialize parameters
    # Spread emission means across vol range
    percentiles = np.linspace(10, 90, K)
    means = np.percentile(obs, percentiles)
    variances = np.ones(K) * np.var(obs) / K
    variances = np.maximum(variances, 1e-6)

    # Initialize transition matrix (sticky)
    A = np.eye(K) * HMM_PERSISTENCE_PRIOR
    off_diag = (1.0 - HMM_PERSISTENCE_PRIOR) / max(K - 1, 1)
    for i in range(K):
        for j in range(K):
            if i != j:
                A[i, j] = off_diag

    pi0 = np.ones(K) / K

    prev_ll = -np.inf
    converged = False
    final_iter = 0

    for iteration in range(n_iter):
        # E-step: Forward-Backward
        alpha, beta, scales, emissions = _forward_backward_fast(
            obs, A, pi0, means, variances)

        # Compute gamma_t(k) = P(state_t = k | obs)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma /= gamma_sum

        # Compute xi_t(i,j) = P(state_t=i, state_{t+1}=j | obs)
        next_likelihood = emissions[1:] * beta[1:]
        xi = alpha[:-1, :, None] * A[None, :, :] * next_likelihood[:, None, :]
        xi_sum = xi.sum(axis=(1, 2), keepdims=True)
        xi = np.divide(
            xi,
            np.maximum(xi_sum, 1e-300),
            out=np.zeros_like(xi),
            where=xi_sum > 1e-300,
        )

        # M-step
        # Update pi0
        pi0 = gamma[0].copy()
        pi0 = np.maximum(pi0, 1e-10)
        pi0 /= pi0.sum()

        # Update A
        A_num = xi.sum(axis=0)
        A_den = gamma[:T - 1].sum(axis=0)
        for i in range(K):
            if A_den[i] > 1e-10:
                A[i] = A_num[i] / A_den[i]
            else:
                A[i] = np.ones(K) / K
        # Normalize rows
        for i in range(K):
            row_sum = A[i].sum()
            if row_sum > 1e-10:
                A[i] /= row_sum

        # Update emission means and variances
        g_sums = gamma.sum(axis=0)
        valid_g = g_sums > 1e-10
        if np.any(valid_g):
            means_new = means.copy()
            means_new[valid_g] = (gamma[:, valid_g].T @ obs) / g_sums[valid_g]
            means = means_new
            diff = obs[:, None] - means[None, :]
            variances_new = variances.copy()
            variances_new[valid_g] = (
                (gamma[:, valid_g] * diff[:, valid_g] * diff[:, valid_g]).sum(axis=0)
                / g_sums[valid_g]
            )
            variances = np.maximum(variances_new, 1e-6)

        # Log-likelihood
        ll = float(np.sum(np.log(np.maximum(scales, 1e-300))))
        final_iter = iteration + 1

        if abs(ll - prev_ll) < tol:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    # Compute stationary distribution from A
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.maximum(stationary, 0.0)
        if stationary.sum() > 1e-12:
            stationary /= stationary.sum()
        else:
            stationary = np.ones(K) / K
    except np.linalg.LinAlgError:
        stationary = np.ones(K) / K

    return HMMRegimeResult(
        transition_matrix=A,
        stationary_dist=stationary,
        filtered_probs=gamma,
        emission_means=means,
        emission_vars=variances,
        n_iter=final_iter,
        log_likelihood=float(prev_ll),
        converged=converged,
    )


# ===================================================================
# Story 19.3: Regime-Specific Forecast Quality Tracking
# ===================================================================

@dataclass(frozen=True)
class RegimeForecastQuality:
    """Per-regime forecast quality metrics.

    Attributes
    ----------
    hit_rate : Dict[str, float]
        Hit rate per regime (fraction of correct directional calls).
    crps : Dict[str, float]
        CRPS per regime (lower is better).
    n_obs : Dict[str, int]
        Number of observations per regime.
    confidence_scaling : Dict[str, float]
        Scaling factor for confidence: hit_rate_regime / hit_rate_avg.
    suppressed_regimes : List[str]
        Regimes with hit_rate < FQ_MIN_HIT_RATE.
    average_hit_rate : float
        Average hit rate across all regimes.
    """
    hit_rate: Dict[str, float]
    crps: Dict[str, float]
    n_obs: Dict[str, int]
    confidence_scaling: Dict[str, float]
    suppressed_regimes: List[str]
    average_hit_rate: float


def _compute_crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """CRPS for a Gaussian forecast N(mu, sigma^2)."""
    from scipy.stats import norm
    sigma_safe = max(abs(sigma), 1e-12)
    z = (y - mu) / sigma_safe
    crps = sigma_safe * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )
    return float(crps)


def regime_forecast_quality(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    regime_labels: List[str],
    prediction_sigma: Optional[np.ndarray] = None,
    window: int = FQ_DEFAULT_WINDOW,
) -> RegimeForecastQuality:
    """Compute per-regime forecast quality metrics.

    Parameters
    ----------
    predictions : array
        Predicted values (e.g., predicted returns or mu).
    outcomes : array
        Actual observed outcomes.
    regime_labels : list of str
        Regime label for each observation.
    prediction_sigma : array, optional
        Predicted standard deviations (for CRPS). If None, uses
        MAD-based estimate per regime.
    window : int
        Rolling window size for metrics.

    Returns
    -------
    RegimeForecastQuality
        Per-regime metrics with confidence scaling.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    n = len(predictions)

    if n != len(outcomes) or n != len(regime_labels):
        raise ValueError(
            f"Length mismatch: predictions={n}, outcomes={len(outcomes)}, "
            f"labels={len(regime_labels)}"
        )

    # Use most recent `window` observations
    if n > window:
        predictions = predictions[-window:]
        outcomes = outcomes[-window:]
        regime_labels = regime_labels[-window:]
        if prediction_sigma is not None:
            prediction_sigma = np.asarray(prediction_sigma, dtype=np.float64)[-window:]
        n = window

    hit_rate = {}
    crps = {}
    n_obs = {}

    for regime in ALL_REGIMES:
        mask = np.array([label == regime for label in regime_labels])
        count = int(mask.sum())
        n_obs[regime] = count

        if count < FQ_MIN_OBS_REGIME:
            hit_rate[regime] = 0.5  # Default to neutral
            crps[regime] = 0.05  # Default
            continue

        pred_r = predictions[mask]
        out_r = outcomes[mask]

        # Hit rate: fraction of correct directional calls
        correct = np.sign(pred_r) == np.sign(out_r)
        # Handle zeros: sign(0) = 0, count as wrong
        hit_rate[regime] = float(np.mean(correct))

        # CRPS (if sigma provided)
        if prediction_sigma is not None:
            sig_r = prediction_sigma[mask]
            crps_vals = np.array([
                _compute_crps_gaussian(float(pred_r[i]), float(sig_r[i]), float(out_r[i]))
                for i in range(count)
            ])
            crps[regime] = float(np.mean(crps_vals))
        else:
            # Use MAE as CRPS proxy
            crps[regime] = float(np.mean(np.abs(pred_r - out_r)))

    # Average hit rate
    valid_rates = [hit_rate[r] for r in ALL_REGIMES if n_obs[r] >= FQ_MIN_OBS_REGIME]
    avg_hit_rate = float(np.mean(valid_rates)) if valid_rates else 0.5

    # Confidence scaling: hit_rate / avg_hit_rate
    confidence_scaling = {}
    for regime in ALL_REGIMES:
        if avg_hit_rate > 1e-12:
            confidence_scaling[regime] = hit_rate[regime] / avg_hit_rate
        else:
            confidence_scaling[regime] = 1.0

    # Suppress regimes with hit rate below threshold
    suppressed = [
        regime for regime in ALL_REGIMES
        if n_obs[regime] >= FQ_MIN_OBS_REGIME and hit_rate[regime] < FQ_MIN_HIT_RATE
    ]

    return RegimeForecastQuality(
        hit_rate=hit_rate,
        crps=crps,
        n_obs=n_obs,
        confidence_scaling=confidence_scaling,
        suppressed_regimes=suppressed,
        average_hit_rate=avg_hit_rate,
    )


def adjusted_confidence(
    base_confidence: float,
    regime: str,
    quality: RegimeForecastQuality,
) -> float:
    """Apply regime-specific confidence scaling.

    Parameters
    ----------
    base_confidence : float
        Raw model confidence.
    regime : str
        Current regime.
    quality : RegimeForecastQuality
        Quality tracking result.

    Returns
    -------
    float
        Adjusted confidence (0 if regime is suppressed).
    """
    if regime in quality.suppressed_regimes:
        return 0.0

    scaling = quality.confidence_scaling.get(regime, 1.0)
    return float(np.clip(base_confidence * scaling, 0.0, 1.0))
