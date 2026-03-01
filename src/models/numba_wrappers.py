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
        unified_phi_student_t_filter_kernel,
        unified_phi_student_t_filter_extended_kernel,
        gaussian_cv_test_fold_kernel,
        phi_gaussian_cv_test_fold_kernel,
        gas_q_filter_gaussian_kernel,
        build_garch_kernel,
        chi2_ewm_correction_kernel,
        pit_var_stretching_kernel,
        phi_student_t_cv_test_fold_kernel,
        compute_ms_process_noise_ewm_kernel,
        stage6_ewm_fold_kernel,
        ewm_mu_correction_kernel,
        gaussian_score_fold_kernel,
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
    
    Used for Bayesian Model Averaging over ν ∈ {4, 8, 20}
    
    This batch function amortizes the cost of:
    - Array preparation (done once)
    - Gamma precomputation (done per ν, but efficiently)
    
    Parameters
    ----------
    nu_grid : List[float]
        List of ν values to evaluate (e.g., [4, 8, 20])
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


# =============================================================================
# UNIFIED φ-STUDENT-T WRAPPER (VoV + MS-q + Smooth Asymmetric ν + Momentum)
# =============================================================================

def run_unified_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu_base: float,
    # MS-q arrays (precomputed)
    q_t: np.ndarray,
    p_stress: np.ndarray,
    # VoV arrays (precomputed)
    vov_rolling: np.ndarray,
    gamma_vov: float,
    vov_damping: float,
    # Asymmetry parameters
    alpha_asym: float,
    k_asym: float,
    # Momentum
    momentum: np.ndarray,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run Numba-accelerated unified φ-Student-t filter.
    
    This is the wrapper for the elite unified model combining:
      1. Smooth Asymmetric ν (Lanczos gammaln for dynamic ν)
      2. Probabilistic MS-q regime switching
      3. VoV scaling with redundancy damping
      4. Momentum drift input
      5. Robust Student-t weighting
    
    All time-varying arrays must be precomputed before calling:
      - q_t: from compute_ms_process_noise_smooth()
      - p_stress: from compute_ms_process_noise_smooth()
      - vov_rolling: rolling std of log(vol)
      - momentum: exogenous signal or zeros
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns
    vol : np.ndarray
        EWMA volatility
    c : float
        Observation noise scale
    phi : float
        AR(1) persistence
    nu_base : float
        Base degrees of freedom
    q_t : np.ndarray
        Time-varying process noise
    p_stress : np.ndarray
        Stress probability per timestep
    vov_rolling : np.ndarray
        Rolling vol-of-vol
    gamma_vov : float
        VoV sensitivity
    vov_damping : float
        Redundancy damping factor
    alpha_asym : float
        Asymmetry parameter (negative = heavier left tail)
    k_asym : float
        Asymmetry transition sharpness
    momentum : np.ndarray
        Momentum signal per timestep
    P0 : float
        Initial state covariance
        
    Returns
    -------
    mu_filtered : np.ndarray
        Posterior state mean
    P_filtered : np.ndarray
        Posterior state variance
    mu_pred : np.ndarray
        Prior predictive mean (for PIT)
    S_pred : np.ndarray
        Prior predictive variance (for PIT)
    log_likelihood : float
        Total log-likelihood
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available for unified filter")
    
    # Prepare all arrays as contiguous float64
    returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
    vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
    q_t = np.ascontiguousarray(q_t.flatten(), dtype=np.float64)
    p_stress = np.ascontiguousarray(p_stress.flatten(), dtype=np.float64)
    vov_rolling = np.ascontiguousarray(vov_rolling.flatten(), dtype=np.float64)
    momentum = np.ascontiguousarray(momentum.flatten(), dtype=np.float64)
    
    return unified_phi_student_t_filter_kernel(
        returns, vol,
        float(c), float(phi), float(nu_base),
        q_t, p_stress,
        vov_rolling, float(gamma_vov), float(vov_damping),
        float(alpha_asym), float(k_asym),
        momentum, float(P0)
    )


def is_unified_filter_available() -> bool:
    """Check if Numba unified filter kernel is available."""
    return _NUMBA_AVAILABLE


def run_unified_phi_student_t_filter_extended(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu_base: float,
    q_t: np.ndarray,
    p_stress: np.ndarray,
    vov_rolling: np.ndarray,
    gamma_vov: float,
    vov_damping: float,
    alpha_asym: float,
    k_asym: float,
    momentum: np.ndarray,
    P0: float,
    # Extended parameters
    risk_prem: float = 0.0,
    mu_drift: float = 0.0,
    skew_kappa: float = 0.0,
    skew_rho: float = 0.0,
    jump_var: float = 0.0,
    jump_intensity: float = 0.0,
    jump_sensitivity: float = 0.0,
    jump_mean: float = 0.0,
    ewm_lambda: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run extended unified phi-Student-t filter kernel.

    Handles all features: risk premium, mu drift, GAS skew,
    Merton jump-diffusion, and causal EWM correction.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available for extended unified filter")

    returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
    vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
    q_t = np.ascontiguousarray(q_t.flatten(), dtype=np.float64)
    p_stress = np.ascontiguousarray(p_stress.flatten(), dtype=np.float64)
    vov_rolling = np.ascontiguousarray(vov_rolling.flatten(), dtype=np.float64)
    momentum = np.ascontiguousarray(momentum.flatten(), dtype=np.float64)

    return unified_phi_student_t_filter_extended_kernel(
        returns, vol,
        float(c), float(phi), float(nu_base),
        q_t, p_stress,
        vov_rolling, float(gamma_vov), float(vov_damping),
        float(alpha_asym), float(k_asym),
        momentum, float(P0),
        float(risk_prem), float(mu_drift),
        float(skew_kappa), float(skew_rho),
        float(jump_var), float(jump_intensity),
        float(jump_sensitivity), float(jump_mean),
        float(ewm_lambda),
    )


# =============================================================================
# CV TEST-FOLD FORWARD-PASS WRAPPERS
# =============================================================================

def run_gaussian_cv_test_fold(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    std_buf: np.ndarray,
    std_offset: int,
    std_max: int,
) -> Tuple[float, int, int]:
    """Run Numba-accelerated Gaussian CV test-fold forward pass."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return gaussian_cv_test_fold_kernel(
        returns, vol_sq,
        float(q), float(c),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
        std_buf, int(std_offset), int(std_max),
    )


def run_phi_gaussian_cv_test_fold(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    std_buf: np.ndarray,
    std_offset: int,
    std_max: int,
) -> Tuple[float, int, int]:
    """Run Numba-accelerated φ-Gaussian CV test-fold forward pass."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return phi_gaussian_cv_test_fold_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
        std_buf, int(std_offset), int(std_max),
    )


def is_cv_kernel_available() -> bool:
    """Check if Numba CV test-fold kernels are compiled and available."""
    return _NUMBA_AVAILABLE

# =============================================================================
# GAS-Q GAUSSIAN FILTER WRAPPER
# =============================================================================

def run_gas_q_filter_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    omega: float,
    alpha: float,
    beta: float,
    q_init: float,
    q_min: float,
    q_max: float,
    score_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run Numba-accelerated GAS-Q Gaussian filter.

    Returns
    -------
    mu_filtered, P_filtered, q_path, score_path, log_likelihood
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    returns, vol = prepare_arrays(returns, vol)
    vol_sq = vol * vol
    n = len(returns)
    mu_filtered = np.zeros(n, dtype=np.float64)
    P_filtered = np.zeros(n, dtype=np.float64)
    q_path = np.zeros(n, dtype=np.float64)
    score_path = np.zeros(n, dtype=np.float64)

    log_ll = gas_q_filter_gaussian_kernel(
        returns, vol_sq, float(c), float(phi),
        float(omega), float(alpha), float(beta),
        float(q_init), float(q_min), float(q_max), float(score_scale),
        mu_filtered, P_filtered, q_path, score_path,
    )
    return mu_filtered, P_filtered, q_path, score_path, float(log_ll)


# =============================================================================
# BUILD-GARCH WRAPPER
# =============================================================================

def run_build_garch(
    n_train: int,
    innovations: np.ndarray,
    sq_inn: np.ndarray,
    neg_ind: np.ndarray,
    garch_omega: float,
    garch_alpha: float,
    garch_leverage: float,
    garch_beta: float,
    unconditional_var: float,
    q_stress_ratio: float,
    rho_c: float,
    kap_c: float,
    eta_c: float = 0.0,
    reg_c: float = 0.0,
) -> np.ndarray:
    """
    Run Numba-accelerated GJR-GARCH variance construction.

    Returns h array of length n_train.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    h_out = np.zeros(n_train, dtype=np.float64)
    build_garch_kernel(
        int(n_train),
        np.ascontiguousarray(innovations, dtype=np.float64),
        np.ascontiguousarray(sq_inn, dtype=np.float64),
        np.ascontiguousarray(neg_ind, dtype=np.float64),
        float(garch_omega), float(garch_alpha),
        float(garch_leverage), float(garch_beta),
        float(unconditional_var), float(q_stress_ratio),
        float(rho_c), float(kap_c), float(eta_c), float(reg_c),
        h_out,
    )
    return h_out


# =============================================================================
# CHI² EWM CORRECTION WRAPPER
# =============================================================================

def run_chi2_ewm_correction(
    z_raw: np.ndarray,
    chi2_target: float,
    chi2_lambda: float = 0.98,
) -> np.ndarray:
    """
    Run Numba-accelerated chi² EWM scale correction.

    Returns scale adjustment array (same length as z_raw).
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    z_raw = np.ascontiguousarray(z_raw, dtype=np.float64)
    scale_adj = np.ones(len(z_raw), dtype=np.float64)
    chi2_ewm_correction_kernel(z_raw, float(chi2_target), float(chi2_lambda), scale_adj)
    return scale_adj


# =============================================================================
# PIT-VARIANCE STRETCHING WRAPPER
# =============================================================================

def run_pit_var_stretching(
    pit_values: np.ndarray,
) -> np.ndarray:
    """
    Run Numba-accelerated PIT-variance stretching in-place.

    Returns the modified pit_values array.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    pit_values = np.ascontiguousarray(pit_values, dtype=np.float64)
    pit_var_stretching_kernel(pit_values)
    return pit_values


def is_gas_q_kernel_available() -> bool:
    """Check if GAS-Q Numba kernel is available."""
    return _NUMBA_AVAILABLE


# =============================================================================
# phi-STUDENT-T CV TEST-FOLD WRAPPER
# =============================================================================

def run_phi_student_t_cv_test_fold(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_scale: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    nu_adjust: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
) -> float:
    """
    Run Numba-accelerated phi-Student-t CV test-fold forward pass.

    Returns log-likelihood of the validation fold.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return phi_student_t_cv_test_fold_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi),
        float(nu_scale), float(log_norm_const), float(neg_exp),
        float(inv_nu), float(nu_adjust),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
    )


# =============================================================================
# MS PROCESS NOISE EWM WRAPPER
# =============================================================================

def run_compute_ms_process_noise_ewm(
    vol: np.ndarray,
    lam: float,
    warmup_mean: float,
    warmup_var: float,
) -> np.ndarray:
    """Run Numba-accelerated EWM z-score computation for MS process noise."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return compute_ms_process_noise_ewm_kernel(
        np.ascontiguousarray(vol.flatten(), dtype=np.float64),
        float(lam), float(warmup_mean), float(warmup_var),
    )


# =============================================================================
# STAGE 6 EWM FOLD WRAPPER
# =============================================================================

def run_stage6_ewm_fold(
    it_arr: np.ndarray,
    Sb_arr: np.ndarray,
    ee: int,
    ve: int,
    lam: float,
    init_em: float,
    init_en: float,
    init_ed: float,
) -> tuple:
    """Run Numba-accelerated Stage 6 EWM fold computation."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return stage6_ewm_fold_kernel(
        np.ascontiguousarray(it_arr, dtype=np.float64),
        np.ascontiguousarray(Sb_arr, dtype=np.float64),
        int(ee), int(ve), float(lam),
        float(init_em), float(init_en), float(init_ed),
    )


# =============================================================================
# STAGE 5f EWM CORRECTION WRAPPER
# =============================================================================

def run_ewm_mu_correction(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    lam: float,
    n_train: int,
) -> np.ndarray:
    """Run Numba-accelerated Stage 5f EWM bias correction."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return ewm_mu_correction_kernel(
        np.ascontiguousarray(returns, dtype=np.float64),
        np.ascontiguousarray(mu_pred, dtype=np.float64),
        float(lam), int(n_train),
    )


# =============================================================================
# GAUSSIAN SCORE FOLD WRAPPER
# =============================================================================

def run_gaussian_score_fold(
    it_arr: np.ndarray,
    Sb_arr: np.ndarray,
    ee: int,
    ve: int,
    lam: float,
    init_em: float,
    init_en: float,
    init_ed: float,
) -> tuple:
    """Run Numba-accelerated Gaussian Stage 5 _score_fold."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return gaussian_score_fold_kernel(
        np.ascontiguousarray(it_arr, dtype=np.float64),
        np.ascontiguousarray(Sb_arr, dtype=np.float64),
        int(ee), int(ve), float(lam),
        float(init_em), float(init_en), float(init_ed),
    )