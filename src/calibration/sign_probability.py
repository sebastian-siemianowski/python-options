"""
Story 5.1: Parameter Uncertainty Propagation into Sign Probability
===================================================================

Computes P(r_{t+1} > 0 | data) by integrating over Kalman state uncertainty P_t,
rather than assuming mu_t is perfectly known.

For Gaussian models:
    P(r > 0) = Phi(mu_t / sqrt(P_t + c * sigma_t^2))

    This is the closed-form marginal: if mu ~ N(mu_t, P_t) and r|mu ~ N(mu, c*sigma_t^2),
    then r ~ N(mu_t, P_t + c*sigma_t^2) marginally, so P(r>0) = Phi(mu_t / sqrt(total_var)).

For Student-t models:
    P(r > 0) = (1/N) sum_{i=1}^{N} F_nu(r > 0 | mu_i, sigma_t)
    where mu_i ~ N(mu_t, P_t)  (Monte Carlo integration over parameter uncertainty)

This produces wider but better-calibrated sign probabilities.
"""
import os
import sys
import logging
import numpy as np
from typing import Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MC_SAMPLES_DEFAULT = 10000
P_T_FLOOR = 1e-14          # Minimum P_t to prevent division by zero
SIGMA_FLOOR = 1e-10         # Minimum sigma_t


def sign_prob_with_uncertainty(
    mu_t: float,
    P_t: float,
    sigma_t: float,
    c: float = 1.0,
    model: str = 'gaussian',
    nu: Optional[float] = None,
    n_mc: int = MC_SAMPLES_DEFAULT,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Compute P(r_{t+1} > 0 | data) integrating over Kalman state uncertainty.

    Instead of the plug-in estimate P(r>0|mu_t, sigma_t), this integrates:
        P(r > 0 | data) = E_{mu ~ N(mu_t, P_t)} [P(r > 0 | mu, sigma_t)]

    Parameters
    ----------
    mu_t : float
        Kalman posterior drift estimate (filtered state).
    P_t : float
        Kalman posterior state variance (estimation uncertainty).
        Larger P_t -> more uncertainty -> sign probability closer to 0.5.
    sigma_t : float
        Observation noise scale (from EWMA/GK volatility).
    c : float
        Observation noise multiplier (c * sigma_t^2 is observation variance).
    model : str
        'gaussian' for closed-form, 'student_t' for Monte Carlo.
    nu : float, optional
        Degrees of freedom (required when model='student_t').
    n_mc : int
        Number of MC samples for Student-t integration (default 10000).
    rng_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        P(r_{t+1} > 0 | data) in [0.01, 0.99] (clipped for safety).
    """
    # Floor P_t and sigma_t
    P_t = max(P_t, P_T_FLOOR)
    sigma_t = max(abs(sigma_t), SIGMA_FLOOR)
    obs_var = c * sigma_t ** 2

    if model == 'gaussian':
        return _sign_prob_gaussian(mu_t, P_t, obs_var)
    elif model == 'student_t':
        if nu is None:
            raise ValueError("nu (degrees of freedom) required for student_t model")
        return _sign_prob_student_t(mu_t, P_t, obs_var, nu, n_mc, rng_seed)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'gaussian' or 'student_t'.")


def _sign_prob_gaussian(mu_t: float, P_t: float, obs_var: float) -> float:
    """
    Closed-form Gaussian sign probability with parameter uncertainty.

    If mu ~ N(mu_t, P_t) and r|mu ~ N(mu, obs_var), then marginalizing:
        r ~ N(mu_t, P_t + obs_var)
        P(r > 0) = Phi(mu_t / sqrt(P_t + obs_var))
    """
    from scipy.stats import norm
    total_var = P_t + obs_var
    z = mu_t / np.sqrt(total_var)
    p = float(norm.cdf(z))
    return float(np.clip(p, 0.01, 0.99))


def _sign_prob_student_t(
    mu_t: float,
    P_t: float,
    obs_var: float,
    nu: float,
    n_mc: int,
    rng_seed: Optional[int],
) -> float:
    """
    Monte Carlo Student-t sign probability with parameter uncertainty.

    Sample mu_i ~ N(mu_t, P_t) for i=1..n_mc, then:
        P(r > 0) = (1/N) sum_i F_nu((0 - mu_i) / sigma_obs; nu)
                  = (1/N) sum_i [1 - F_nu(-mu_i / sigma_obs; nu)]

    where F_nu is the Student-t CDF with nu degrees of freedom.
    """
    from scipy.stats import t as student_t_dist

    rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random.RandomState()
    sigma_obs = np.sqrt(obs_var)

    # Sample mu from posterior: mu_i ~ N(mu_t, P_t)
    mu_samples = rng.normal(mu_t, np.sqrt(P_t), size=n_mc)

    # For each sampled mu, compute P(r > 0 | mu, sigma_obs, nu)
    # r|mu ~ mu + sigma_obs * t_nu, so P(r > 0) = P(t_nu > -mu/sigma_obs)
    z_scores = mu_samples / sigma_obs
    # P(r > 0 | mu_i) = F_nu(z_i)  (since t_nu CDF gives P(X < z))
    p_positive = student_t_dist.cdf(z_scores, df=nu)

    # Average over MC samples
    p = float(np.mean(p_positive))
    return float(np.clip(p, 0.01, 0.99))


def sign_prob_no_uncertainty(
    mu_t: float,
    sigma_t: float,
    c: float = 1.0,
    model: str = 'gaussian',
    nu: Optional[float] = None,
) -> float:
    """
    Standard plug-in sign probability (no parameter uncertainty).

    P(r > 0) = Phi(mu_t / (sqrt(c) * sigma_t))       for Gaussian
    P(r > 0) = F_nu(mu_t / (sqrt(c) * sigma_t))      for Student-t

    This is the baseline that Story 5.1 improves upon.
    """
    sigma_t = max(abs(sigma_t), SIGMA_FLOOR)
    sigma_obs = np.sqrt(c) * sigma_t
    z = mu_t / sigma_obs

    if model == 'gaussian':
        from scipy.stats import norm
        p = float(norm.cdf(z))
    elif model == 'student_t':
        if nu is None:
            raise ValueError("nu required for student_t")
        from scipy.stats import t as student_t_dist
        p = float(student_t_dist.cdf(z, df=nu))
    else:
        raise ValueError(f"Unknown model: {model}")

    return float(np.clip(p, 0.01, 0.99))


def compute_sign_prob_ece(
    predicted_probs: np.ndarray,
    actual_signs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for sign probabilities.

    ECE = sum_b (n_b / N) * |avg_confidence_b - avg_accuracy_b|

    Parameters
    ----------
    predicted_probs : ndarray, shape (T,)
        Predicted P(r > 0) for each observation.
    actual_signs : ndarray, shape (T,)
        Binary: 1 if r > 0, 0 otherwise.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float : ECE in [0, 1]. Lower = better calibrated. Target: < 0.05.
    """
    N = len(predicted_probs)
    if N == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        mask = (predicted_probs >= bin_edges[b]) & (predicted_probs < bin_edges[b + 1])
        if b == n_bins - 1:  # Include right edge for last bin
            mask = mask | (predicted_probs == bin_edges[b + 1])
        n_b = np.sum(mask)
        if n_b == 0:
            continue
        avg_confidence = np.mean(predicted_probs[mask])
        avg_accuracy = np.mean(actual_signs[mask])
        ece += (n_b / N) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def compute_hit_rate_at_threshold(
    predicted_probs: np.ndarray,
    actual_signs: np.ndarray,
    threshold: float = 0.60,
) -> Tuple[float, int]:
    """
    Compute hit rate for predictions above a confidence threshold.

    Only considers predictions where P(r>0) >= threshold (for longs)
    or P(r>0) <= 1-threshold (for shorts).

    Parameters
    ----------
    predicted_probs : ndarray
        Predicted P(r > 0).
    actual_signs : ndarray
        Binary actual signs.
    threshold : float
        Minimum confidence for inclusion.

    Returns
    -------
    (hit_rate, n_predictions) : (float, int)
    """
    # Long predictions: P(r>0) >= threshold
    long_mask = predicted_probs >= threshold
    # Short predictions: P(r>0) <= 1-threshold
    short_mask = predicted_probs <= (1.0 - threshold)

    # For long calls: correct if actual sign = 1
    long_correct = np.sum(actual_signs[long_mask]) if np.any(long_mask) else 0
    # For short calls: correct if actual sign = 0
    short_correct = np.sum(1 - actual_signs[short_mask]) if np.any(short_mask) else 0

    total_predictions = int(np.sum(long_mask) + np.sum(short_mask))
    total_correct = int(long_correct + short_correct)

    if total_predictions == 0:
        return 0.0, 0

    return float(total_correct / total_predictions), total_predictions


# ---------------------------------------------------------------------------
# Story 5.2: Asymmetric Sign Probability for Skewed Distributions
# ---------------------------------------------------------------------------

def _two_piece_student_t_cdf_at_zero(mu: float, sigma: float,
                                       nu_L: float, nu_R: float) -> float:
    """
    P(X > 0) for two-piece Student-t with location mu, scale sigma.

    The two-piece Student-t density (different nu for each half):
        f(x) = t_{nu_L}((x-mu)/sigma) / sigma,   x < mu
        f(x) = t_{nu_R}((x-mu)/sigma) / sigma,   x >= mu

    This integrates to 1 because each half-t contributes 0.5.

    Result:
        mu >= 0:  P(X > 0) = T_{nu_L}(mu/sigma)    [left tail controls crash risk]
        mu < 0:   P(X > 0) = T_{nu_R}(mu/sigma)     [right tail controls recovery]
    """
    from scipy.stats import t as t_dist

    z = mu / sigma

    if mu >= 0:
        p = t_dist.cdf(z, df=nu_L)
    else:
        p = t_dist.cdf(z, df=nu_R)

    return float(np.clip(p, 0.01, 0.99))


def sign_prob_skewed(
    mu_t: float,
    P_t: float,
    sigma_t: float,
    nu_L: float,
    nu_R: float,
    c: float = 1.0,
    n_mc: int = MC_SAMPLES_DEFAULT,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Asymmetric sign probability using two-piece Student-t with parameter uncertainty.

    Integrates over Kalman state uncertainty and uses different tail parameters
    for left (nu_L) and right (nu_R) tails.

    When nu_L < nu_R (heavy left tail, lighter right tail):
        -> P(r < 0) increases relative to symmetric model
        -> This captures crash risk asymmetry

    Parameters
    ----------
    mu_t : float
        Kalman posterior drift.
    P_t : float
        Kalman posterior state variance.
    sigma_t : float
        Observation noise scale.
    nu_L : float
        Left tail degrees of freedom (lower = heavier crash tail).
    nu_R : float
        Right tail degrees of freedom (higher = lighter recovery tail).
    c : float
        Observation noise multiplier.
    n_mc : int
        Monte Carlo samples.
    rng_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float : P(r > 0 | data) in [0.01, 0.99].
    """
    from scipy.stats import t as t_dist

    P_t = max(P_t, P_T_FLOOR)
    sigma_t = max(abs(sigma_t), SIGMA_FLOOR)
    sigma_obs = np.sqrt(c) * sigma_t

    rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random.RandomState()

    # Sample mu from posterior
    mu_samples = rng.normal(mu_t, np.sqrt(P_t), size=n_mc)

    # For each mu_i, compute P(r > 0) under two-piece Student-t
    p_positive = np.zeros(n_mc)
    for i in range(n_mc):
        p_positive[i] = _two_piece_student_t_cdf_at_zero(
            mu_samples[i], sigma_obs, nu_L, nu_R
        )

    p = float(np.mean(p_positive))
    return float(np.clip(p, 0.01, 0.99))


def sign_prob_skewed_no_uncertainty(
    mu_t: float,
    sigma_t: float,
    nu_L: float,
    nu_R: float,
    c: float = 1.0,
) -> float:
    """
    Asymmetric sign probability without parameter uncertainty (plug-in).

    Uses the two-piece Student-t directly at mu_t.
    """
    sigma_t = max(abs(sigma_t), SIGMA_FLOOR)
    sigma_obs = np.sqrt(c) * sigma_t
    return _two_piece_student_t_cdf_at_zero(mu_t, sigma_obs, nu_L, nu_R)


# ---------------------------------------------------------------------------
# Story 5.3: Multi-Horizon Sign Probability with Drift Accumulation
# ---------------------------------------------------------------------------

# Standard forecast horizons (days)
STANDARD_HORIZONS = (1, 3, 7, 30, 90)


def multi_horizon_sign_prob(
    mu_t: float,
    P_t: float,
    phi: float,
    sigma_t: float,
    c: float,
    H: int,
    model: str = 'gaussian',
    nu: Optional[float] = None,
    n_mc: int = MC_SAMPLES_DEFAULT,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Compute H-step-ahead sign probability P(r_{t+H} > 0 | data).

    Propagates the AR(1) Kalman state forward H steps:
        Drift:    mu_{t+H} = phi^H * mu_t
        Variance: Var_{t+H} = P_t * sum_{j=0}^{H-1} phi^{2j}  +  H * c * sigma_t^2

    The first term captures state uncertainty growing through AR(1) dynamics.
    The second term captures accumulated observation noise over H steps.

    Parameters
    ----------
    mu_t : float
        Current Kalman posterior drift estimate.
    P_t : float
        Current Kalman posterior state variance.
    phi : float
        AR(1) persistence parameter (typically 0.95-0.999).
    sigma_t : float
        Current observation noise scale.
    c : float
        Observation noise multiplier.
    H : int
        Forecast horizon in days (must be >= 1).
    model : str
        'gaussian' or 'student_t'.
    nu : float, optional
        Degrees of freedom for Student-t model.
    n_mc : int
        MC samples for Student-t integration.
    rng_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float : P(r_{t+H} > 0 | data) in [0.01, 0.99].
    """
    if H < 1:
        raise ValueError(f"Horizon H must be >= 1, got {H}")

    P_t = max(P_t, P_T_FLOOR)
    sigma_t = max(abs(sigma_t), SIGMA_FLOOR)

    # H-step drift: phi^H * mu_t
    mu_H = (phi ** H) * mu_t

    # H-step state variance: P_t * sum_{j=0}^{H-1} phi^{2j}
    phi_sq = phi ** 2
    if abs(phi_sq - 1.0) < 1e-12:
        # phi ~ 1.0: geometric sum = H
        state_var_mult = float(H)
    else:
        # Geometric series: (1 - phi^{2H}) / (1 - phi^2)
        state_var_mult = (1.0 - phi_sq ** H) / (1.0 - phi_sq)

    P_H = P_t * state_var_mult

    # Accumulated observation variance: H * c * sigma_t^2
    obs_var_H = H * c * sigma_t ** 2

    # Total predictive variance
    total_var_H = P_H + obs_var_H

    if model == 'gaussian':
        from scipy.stats import norm
        z = mu_H / np.sqrt(total_var_H)
        p = float(norm.cdf(z))
        return float(np.clip(p, 0.01, 0.99))
    elif model == 'student_t':
        if nu is None:
            raise ValueError("nu required for student_t model")
        from scipy.stats import t as t_dist
        rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random.RandomState()
        # Sample from H-step predictive posterior for mu
        mu_samples = rng.normal(mu_H, np.sqrt(P_H), size=n_mc)
        sigma_obs_H = np.sqrt(obs_var_H)
        z_scores = mu_samples / sigma_obs_H
        p_positive = t_dist.cdf(z_scores, df=nu)
        p = float(np.mean(p_positive))
        return float(np.clip(p, 0.01, 0.99))
    else:
        raise ValueError(f"Unknown model: {model}")


def multi_horizon_sign_prob_all(
    mu_t: float,
    P_t: float,
    phi: float,
    sigma_t: float,
    c: float,
    horizons: tuple = STANDARD_HORIZONS,
    model: str = 'gaussian',
    nu: Optional[float] = None,
    n_mc: int = MC_SAMPLES_DEFAULT,
    rng_seed: Optional[int] = None,
) -> dict:
    """
    Compute sign probabilities for all standard horizons.

    Returns
    -------
    dict : {H: P(r_{t+H} > 0)} for each H in horizons.
    """
    result = {}
    for H in horizons:
        result[H] = multi_horizon_sign_prob(
            mu_t, P_t, phi, sigma_t, c, H,
            model=model, nu=nu, n_mc=n_mc, rng_seed=rng_seed,
        )
    return result


def multi_horizon_prediction_interval(
    mu_t: float,
    P_t: float,
    phi: float,
    sigma_t: float,
    c: float,
    H: int,
    coverage: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute the H-step-ahead prediction interval at given coverage.

    For Gaussian predictive: [mu_H - z*sqrt(var_H), mu_H + z*sqrt(var_H)]
    where z is the normal quantile for the desired coverage.

    Parameters
    ----------
    mu_t, P_t, phi, sigma_t, c, H : as in multi_horizon_sign_prob
    coverage : float
        Coverage probability (e.g. 0.90 for 90% PI).

    Returns
    -------
    (lower, upper) : Tuple[float, float]
    """
    from scipy.stats import norm

    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")

    P_t = max(P_t, P_T_FLOOR)
    sigma_t = max(abs(sigma_t), SIGMA_FLOOR)

    mu_H = (phi ** H) * mu_t

    phi_sq = phi ** 2
    if abs(phi_sq - 1.0) < 1e-12:
        state_var_mult = float(H)
    else:
        state_var_mult = (1.0 - phi_sq ** H) / (1.0 - phi_sq)

    P_H = P_t * state_var_mult
    obs_var_H = H * c * sigma_t ** 2
    total_var_H = P_H + obs_var_H

    alpha = (1.0 - coverage) / 2.0
    z = norm.ppf(1.0 - alpha)
    std_H = np.sqrt(total_var_H)

    return (mu_H - z * std_H, mu_H + z * std_H)
