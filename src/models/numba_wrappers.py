"""
Wrapper functions that handle scipy calls before invoking Numba kernels.

Separation of concerns:
    - Numba kernels: Pure numeric loops, no Python objects
    - Wrappers: Array preparation, gamma precomputation, fallback handling

Architectural Invariant:
    There is NO bare Student-t wrapper. All Student-t filtering requires φ.

Model variant mapping:
    - Gaussian base: gaussian_filter_kernel
    - φ-Gaussian: phi_gaussian_filter_kernel
    - φ-Student-t: phi_student_t_filter_kernel (the ONLY Student-t variant)
    - φ-Gaussian+Mom (CRSP/CELH/DPRO): momentum_phi_gaussian_filter_kernel
    - φ-Student-t+Mom (GLDW/MAGD/BKSY/ASTS): momentum_phi_student_t_filter_kernel

Author: Quantitative Systems Team
Date: 2026-02-04
"""

from typing import Tuple, Dict, List, Optional
import numpy as np

# Try to import scipy for gamma precomputation
try:
    from scipy.special import gammaln
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Try to import Numba kernels
try:
    from .numba_kernels import (
        gaussian_filter_kernel,
        phi_gaussian_filter_kernel,
        phi_student_t_filter_kernel,
        momentum_phi_gaussian_filter_kernel,
        momentum_phi_student_t_filter_kernel,
        student_t_filter_with_lfo_cv_kernel,
        gaussian_filter_with_lfo_cv_kernel,
        ms_q_student_t_filter_kernel,
    )
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# =============================================================================
# AVAILABILITY CHECKS
# =============================================================================

def is_numba_available() -> bool:
    """Check if Numba kernels compiled successfully."""
    return _NUMBA_AVAILABLE


def is_scipy_available() -> bool:
    """Check if scipy is available for gamma precomputation."""
    return _SCIPY_AVAILABLE


# =============================================================================
# ARRAY PREPARATION
# =============================================================================

def prepare_arrays(*arrays) -> Tuple[np.ndarray, ...]:
    """
    Ensure arrays are contiguous float64 for Numba.
    
    This is CRITICAL for Numba performance - non-contiguous arrays
    cause massive slowdowns due to cache misses.
    """
    return tuple(
        np.ascontiguousarray(arr.flatten(), dtype=np.float64) 
        for arr in arrays
    )


# =============================================================================
# GAMMA PRECOMPUTATION (for φ-Student-t)
# =============================================================================

def precompute_gamma_values(nu: float) -> Tuple[float, float]:
    """
    Precompute gamma function values for Student-t.
    
    Using scipy.special.gammaln ensures correctness at low ν
    where Stirling's approximation has significant error.
    
    This is why we precompute in Python rather than approximating in Numba:
    at ν=4, Stirling error can flip BMA model rankings.
    
    Parameters
    ----------
    nu : float
        Degrees of freedom
        
    Returns
    -------
    log_gamma_half_nu : float
        gammaln(ν/2)
    log_gamma_half_nu_plus_half : float
        gammaln((ν+1)/2)
    """
    if not _SCIPY_AVAILABLE:
        # Fallback to Stirling approximation (less accurate at low ν)
        def _stirling_gammaln(x: float) -> float:
            return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi)
        
        log_gamma_half_nu = _stirling_gammaln(nu / 2.0)
        log_gamma_half_nu_plus_half = _stirling_gammaln((nu + 1.0) / 2.0)
    else:
        log_gamma_half_nu = float(gammaln(nu / 2.0))
        log_gamma_half_nu_plus_half = float(gammaln((nu + 1.0) / 2.0))
    
    return log_gamma_half_nu, log_gamma_half_nu_plus_half


# =============================================================================
# BASE MODEL WRAPPERS
# =============================================================================

def run_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Gaussian Kalman filter (random walk drift).
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns
    vol : np.ndarray
        EWMA volatility estimates
    q : float
        Process noise variance
    c : float
        Observation noise scale
    P0 : float
        Initial state covariance
        
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    return gaussian_filter_kernel(returns, vol, float(q), float(c), float(P0))


def run_phi_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Gaussian Kalman filter (AR(1) drift).
    
    Parameters
    ----------
    phi : float
        AR(1) persistence coefficient
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    return phi_gaussian_filter_kernel(
        returns, vol, float(q), float(c), float(phi), float(P0)
    )


def run_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Student-t Kalman filter (AR(1) drift, heavy-tailed observations).
    
    This is the ONLY Student-t variant. There is no bare Student-t.
    
    Parameters
    ----------
    phi : float
        AR(1) persistence coefficient
    nu : float
        Degrees of freedom (typically from grid: 4, 6, 8, 12, 20)
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return phi_student_t_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        float(P0)
    )


# =============================================================================
# MOMENTUM-AUGMENTED WRAPPERS
# =============================================================================

def run_momentum_phi_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Gaussian filter with momentum augmentation.
    
    Used by: CRSP (φ-Gaussian+Mom+EVTM+CST20%)
             CELH (φ-Gaussian+Mom+EVTH+CST17%)
             DPRO (φ-Gaussian+Mom+EVTH+CST19%)
    
    Parameters
    ----------
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal to add to drift prediction
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol, momentum_adjustment = prepare_arrays(
        returns, vol, momentum_adjustment
    )
    return momentum_phi_gaussian_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi),
        momentum_adjustment,
        float(P0)
    )


def run_momentum_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Student-t filter with momentum augmentation.
    
    Used by: GLDW (φ-Student-t+Mom+EVTH+CST17%)
             MAGD (φ-Student-t+Mom+Hλ←+EVTM+CST17%)
             BKSY (φ-Student-t+Mom+Hλ→+EVTH+CST17%)
             ASTS (φ-Student-t+Mom+Hλ→+EVTH+CST14%)
    
    Notes on augmentation layers:
    - Hλ← : Hierarchical λ with backward-looking momentum
    - Hλ→ : Hierarchical λ with forward-looking momentum
    - EVTH/EVTM: EVT tail handling affects vol estimation UPSTREAM
    - CST##%: CVaR constraint affects position sizing DOWNSTREAM
    
    None of these alter the Kalman filter mathematics.
    
    Parameters
    ----------
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal (may include Hλ scaling)
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol, momentum_adjustment = prepare_arrays(
        returns, vol, momentum_adjustment
    )
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return momentum_phi_student_t_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        momentum_adjustment,
        float(P0)
    )


# =============================================================================
# BATCH PROCESSING FOR BMA (multiple ν values)
# =============================================================================

def run_phi_student_t_filter_batch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid: List[float],
    P0: float = 1e-4
) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run φ-Student-t filter for multiple ν values (discrete grid BMA).
    
    Returns dict mapping ν -> (mu_filtered, P_filtered, log_likelihood)
    
    Used for Bayesian Model Averaging over ν ∈ {4, 6, 8, 12, 20}
    
    This batch function amortizes the cost of:
    - Array preparation (done once)
    - Gamma precomputation (done per ν, but efficiently)
    
    Parameters
    ----------
    nu_grid : List[float]
        List of ν values to evaluate (e.g., [4, 6, 8, 12, 20])
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    # Prepare arrays once
    returns, vol = prepare_arrays(returns, vol)
    results = {}
    
    for nu in nu_grid:
        log_g1, log_g2 = precompute_gamma_values(nu)
        mu, P, ll = phi_student_t_filter_kernel(
            returns, vol,
            float(q), float(c), float(phi), float(nu),
            log_g1, log_g2,
            float(P0)
        )
        results[nu] = (mu, P, ll)
    
    return results


def run_momentum_phi_student_t_filter_batch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid: List[float],
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run momentum-augmented φ-Student-t filter for multiple ν values.
    
    Parameters
    ----------
    nu_grid : List[float]
        List of ν values to evaluate
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol, momentum_adjustment = prepare_arrays(
        returns, vol, momentum_adjustment
    )
    results = {}
    
    for nu in nu_grid:
        log_g1, log_g2 = precompute_gamma_values(nu)
        mu, P, ll = momentum_phi_student_t_filter_kernel(
            returns, vol,
            float(q), float(c), float(phi), float(nu),
            log_g1, log_g2,
            momentum_adjustment,
            float(P0)
        )
        results[nu] = (mu, P, ll)
    
    return results


# =============================================================================
# MS-q AND FUSED LFO-CV WRAPPERS (February 2026)
# =============================================================================
# These wrappers provide:
# 1. Numba-accelerated MS-q filtering (10× speedup)
# 2. Fused LFO-CV computation (40% overall speedup by avoiding second pass)
# =============================================================================

def run_student_t_filter_with_lfo_cv(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run φ-Student-t filter with FUSED LFO-CV computation.
    
    This is 40% faster than running filter + separate LFO-CV computation
    because it computes the predictive log-density during the single pass.
    
    Parameters
    ----------
    lfo_start_frac : float
        Fraction of data before starting LFO-CV accumulation (default 0.5)
        
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
        Mean predictive log-density from t=lfo_start to T
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return student_t_filter_with_lfo_cv_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        float(P0), float(lfo_start_frac)
    )


def run_gaussian_filter_with_lfo_cv(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run Gaussian filter with FUSED LFO-CV computation.
    
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    
    return gaussian_filter_with_lfo_cv_kernel(
        returns, vol,
        float(q), float(c), float(phi),
        float(P0), float(lfo_start_frac)
    )


def run_ms_q_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu: float,
    q_calm: float,
    q_stress: float,
    sensitivity: float = 2.0,
    threshold: float = 1.3,
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Run Numba-accelerated MS-q Student-t filter with fused LFO-CV.
    
    This provides ~10× speedup over pure Python implementation.
    
    Parameters
    ----------
    q_calm : float
        Process noise in calm regime
    q_stress : float
        Process noise in stress regime (typically 100× q_calm)
    sensitivity : float
        Sigmoid sensitivity to vol_relative (default 2.0)
    threshold : float
        Vol_relative threshold for regime transition (default 1.3)
    lfo_start_frac : float
        Fraction of data before starting LFO-CV accumulation
        
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    q_t : np.ndarray
        Time-varying process noise
    p_stress : np.ndarray
        Probability of stress regime at each timestep
    log_likelihood : float
    lfo_cv_score : float
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return ms_q_student_t_filter_kernel(
        returns, vol,
        float(c), float(phi), float(nu),
        float(q_calm), float(q_stress),
        float(sensitivity), float(threshold),
        log_g1, log_g2,
        float(P0), float(lfo_start_frac)
    )


def run_student_t_filter_with_lfo_cv_batch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid: List[float],
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Run fused filter+LFO-CV for multiple ν values (BMA optimization).
    
    Returns dict mapping ν -> (mu, P, log_likelihood, lfo_cv_score)
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    results = {}
    
    for nu in nu_grid:
        log_g1, log_g2 = precompute_gamma_values(nu)
        mu, P, ll, lfo = student_t_filter_with_lfo_cv_kernel(
            returns, vol,
            float(q), float(c), float(phi), float(nu),
            log_g1, log_g2,
            float(P0), float(lfo_start_frac)
        )
        results[nu] = (mu, P, ll, lfo)
    
    return results