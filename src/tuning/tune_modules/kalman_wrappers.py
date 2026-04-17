"""
Kalman filter wrappers: compatibility stubs and predictive PIT/score computation.

Extracted from tune.py (Story 3.1).
"""
from typing import Dict, Tuple

import numpy as np

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403


__all__ = [
    "kalman_filter_drift",
    "kalman_filter_drift_phi",
    "compute_pit_ks_pvalue",
    "optimize_q_mle",
    "kalman_filter_drift_phi_student_t",
    "compute_pit_ks_pvalue_student_t",
    "compute_predictive_pit_student_t",
    "compute_predictive_scores_student_t",
    "compute_predictive_pit_gaussian",
    "compute_predictive_scores_gaussian",
]


def kalman_filter_drift(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for GaussianDriftModel.filter()"""
    return GaussianDriftModel.filter(returns, vol, q, c)


def kalman_filter_drift_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for GaussianDriftModel.filter_phi()"""
    return GaussianDriftModel.filter_phi(returns, vol, q, c, phi)


def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
    """Compatibility wrapper for GaussianDriftModel.pit_ks()"""
    return GaussianDriftModel.pit_ks(returns, mu_filtered, vol, P_filtered, c)


def optimize_q_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-1,
    c_min: float = 0.3,
    c_max: float = 3.0,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, Dict]:
    """Delegate Gaussian q/c optimization to GaussianDriftModel for modularity."""
    return GaussianDriftModel.optimize_params(
        returns=returns,
        vol=vol,
        train_frac=train_frac,
        q_min=q_min,
        q_max=q_max,
        c_min=c_min,
        c_max=c_max,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )


def kalman_filter_drift_phi_student_t(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for PhiStudentTDriftModel.filter_phi()"""
    return PhiStudentTDriftModel.filter_phi(returns, vol, q, c, phi, nu)


def compute_pit_ks_pvalue_student_t(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
    """Compatibility wrapper for PhiStudentTDriftModel.pit_ks()"""
    return PhiStudentTDriftModel.pit_ks(returns, mu_filtered, vol, P_filtered, c, nu)


def compute_predictive_pit_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    ELITE FIX: Compute proper PIT using PREDICTIVE distribution.
    
    This function runs the filter to get predictive values (before seeing y_t)
    and computes PIT correctly. The key insight: PIT transforms y_t through
    the CDF of its PRIOR predictive distribution, not the posterior.
    
    Args:
        returns: Return observations
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence
        nu: Degrees of freedom
        
    Returns:
        Tuple of (KS statistic, KS p-value, mu_pred, S_pred)
    """
    # Run filter with predictive output
    mu_filt, P_filt, mu_pred, S_pred, ll = PhiStudentTDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi, nu
    )
    
    # Compute PIT using predictive distribution
    ks_stat, pit_p = PhiStudentTDriftModel.pit_ks_predictive(
        returns, mu_pred, S_pred, nu
    )
    
    return ks_stat, pit_p, mu_pred, S_pred


def compute_predictive_scores_student_t(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
    nu: float,
) -> Tuple[float, float]:
    """
    ELITE FIX: Compute Hyvärinen and CRPS using PREDICTIVE distribution.
    
    Uses predictive mean and variance (before seeing y_t) for proper
    out-of-sample scoring.
    
    Args:
        returns: Return observations
        mu_pred: Predictive means from filter_phi_with_predictive
        S_pred: Predictive variances from filter_phi_with_predictive
        nu: Degrees of freedom
        
    Returns:
        Tuple of (Hyvärinen score, CRPS)
    """
    # Compute Student-t scale from predictive variance
    if nu > 2:
        forecast_scale = np.sqrt(S_pred * (nu - 2) / nu)
    else:
        forecast_scale = np.sqrt(S_pred)
    
    # Compute scores using predictive values
    hyvarinen = compute_hyvarinen_score_student_t(returns, mu_pred, forecast_scale, nu)
    crps = compute_crps_student_t_inline(returns, mu_pred, forecast_scale, nu)
    
    return hyvarinen, crps


def compute_predictive_pit_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    ELITE FIX: Compute proper PIT using PREDICTIVE distribution for Gaussian.
    
    Args:
        returns: Return observations
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence (default 1.0 for random walk)
        
    Returns:
        Tuple of (KS statistic, KS p-value, mu_pred, S_pred)
    """
    # Run filter with predictive output
    mu_filt, P_filt, mu_pred, S_pred, ll = GaussianDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi
    )
    
    # Compute PIT using predictive distribution
    ks_stat, pit_p = GaussianDriftModel.pit_ks_predictive(
        returns, mu_pred, S_pred
    )
    
    return ks_stat, pit_p, mu_pred, S_pred


def compute_predictive_scores_gaussian(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    ELITE FIX: Compute Hyvärinen and CRPS using PREDICTIVE distribution for Gaussian.
    """
    forecast_std = np.sqrt(S_pred)
    
    hyvarinen = compute_hyvarinen_score_gaussian(returns, mu_pred, forecast_std)
    crps = compute_crps_gaussian_inline(returns, mu_pred, forecast_std)
    
    return hyvarinen, crps


