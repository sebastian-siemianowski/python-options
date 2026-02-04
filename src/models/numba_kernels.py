"""
Numba JIT-compiled Kalman filter kernels for performance-critical loops.

Architectural Invariant:
    There is NO bare Student-t model. All Student-t behavior is defined
    ONLY in conjunction with φ-dynamics (AR(1) drift persistence).

Supported model families:
    - Gaussian (random walk drift)
    - φ-Gaussian (AR(1) drift)
    - φ-Student-t (AR(1) drift with heavy-tailed observation noise)
    - Momentum-augmented variants of the above

Design principles:
    - fastmath=True ONLY for Gaussian and φ-Gaussian kernels
    - fastmath=False for all φ-Student-t kernels (tail-sensitive BIC ranking)
    - P0 passed explicitly for future extensibility
    - Log-likelihood contributions clamped to prevent crisis-tick domination
    - φ-Student-t likelihoods use precomputed gamma values (passed from Python)
    - Kernels contain NO Python objects, NO scipy, NO dynamic allocation

Author: Quantitative Systems Team
Date: 2026-02-04
"""

from numba import njit
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum log-likelihood contribution per observation (prevents outlier domination)
_MAX_LL_CONTRIB = 50.0

# Minimum variance floor to prevent numerical instability
_MIN_VARIANCE = 1e-12

# Log(2π) precomputed for Gaussian likelihood
_LOG_2PI = np.log(2.0 * np.pi)


# =============================================================================
# GAUSSIAN KERNELS (fastmath=True safe)
# =============================================================================

@njit(cache=True, fastmath=True)
def gaussian_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    P0: float = 1e-4
) -> tuple:
    """
    Gaussian Kalman filter kernel - random walk drift (φ=1 implicit).
    
    State equation: μ_t = μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:    r_t = μ_t + ε_t,      ε_t ~ N(0, c·vol_t²)
    
    Parameters
    ----------
    returns : np.ndarray
        Contiguous float64 array of log returns
    vol : np.ndarray
        Contiguous float64 array of EWMA volatility estimates
    q : float
        Process noise variance (drift evolution uncertainty)
    c : float
        Observation noise scale multiplier
    P0 : float
        Initial state covariance
    
    Returns
    -------
    mu_filtered : np.ndarray
        Filtered drift estimates
    P_filtered : np.ndarray
        Filtered state covariances
    log_likelihood : float
        Total log-likelihood
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        # Predict step (random walk: μ_pred = μ)
        mu_pred = mu
        P_pred = P + q
        
        # Observation variance
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        forecast_var = P_pred + R
        
        if forecast_var > _MIN_VARIANCE:
            K = P_pred / forecast_var
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            # Clamp innovation contribution to prevent outlier domination
            innov_sq = innovation * innovation
            innov_sq_scaled = innov_sq / forecast_var
            if innov_sq_scaled > 100.0:
                innov_sq_scaled = 100.0
            
            ll_contrib = -0.5 * (_LOG_2PI + np.log(forecast_var) + innov_sq_scaled)
            if ll_contrib < -_MAX_LL_CONTRIB:
                ll_contrib = -_MAX_LL_CONTRIB
            log_likelihood += ll_contrib
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    return mu_filtered, P_filtered, log_likelihood


@njit(cache=True, fastmath=True)
def phi_gaussian_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    P0: float = 1e-4
) -> tuple:
    """
    φ-Gaussian Kalman filter kernel - AR(1) drift dynamics.
    
    State equation: μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:    r_t = μ_t + ε_t,        ε_t ~ N(0, c·vol_t²)
    
    Parameters
    ----------
    phi : float
        AR(1) persistence coefficient (|φ| < 1 for stationarity)
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi
    
    for t in range(n):
        # Predict step with AR(1) dynamics
        mu_pred = phi * mu
        P_pred = phi_sq * P + q
        
        # Observation variance
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            innov_sq = innovation * innovation
            innov_sq_scaled = innov_sq / S
            if innov_sq_scaled > 100.0:
                innov_sq_scaled = 100.0
            
            ll_contrib = -0.5 * (_LOG_2PI + np.log(S) + innov_sq_scaled)
            if ll_contrib < -_MAX_LL_CONTRIB:
                ll_contrib = -_MAX_LL_CONTRIB
            log_likelihood += ll_contrib
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    return mu_filtered, P_filtered, log_likelihood


# =============================================================================
# φ-STUDENT-T KERNELS (fastmath=False for tail correctness)
# =============================================================================

@njit(cache=True, fastmath=False)
def student_t_logpdf_kernel(
    x: float,
    nu: float,
    mu: float,
    scale: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float
) -> float:
    """
    Student-t log-density with precomputed gamma values.
    
    fastmath=False ensures IEEE compliance for likelihood ranking in BMA/BIC.
    
    Parameters
    ----------
    x : float
        Observation value
    nu : float
        Degrees of freedom (from discrete grid: 4, 6, 8, 12, 20)
    mu : float
        Location parameter
    scale : float
        Scale parameter
    log_gamma_half_nu : float
        Precomputed gammaln(ν/2) from scipy
    log_gamma_half_nu_plus_half : float
        Precomputed gammaln((ν+1)/2) from scipy
    
    Returns
    -------
    logpdf : float
    """
    if scale <= _MIN_VARIANCE or nu <= 0.0:
        return -1e12
    
    z = (x - mu) / scale
    z_sq = z * z
    
    # Use precomputed gamma values (avoids Stirling error at low ν)
    log_norm = (log_gamma_half_nu_plus_half 
                - log_gamma_half_nu 
                - 0.5 * np.log(nu * np.pi * scale * scale))
    log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + z_sq / nu)
    
    return log_norm + log_kernel


@njit(cache=True, fastmath=False)
def phi_student_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    P0: float = 1e-4
) -> tuple:
    """
    φ-Student-t Kalman filter kernel - AR(1) drift with heavy-tailed observations.
    
    This is the ONLY Student-t variant. There is no bare Student-t model.
    
    State equation: μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:    r_t = μ_t + ε_t,        ε_t ~ t_ν(0, c·vol_t²)
    
    Parameters
    ----------
    phi : float
        AR(1) persistence (φ=0 mean-reverting, φ→1 random walk)
    nu : float
        Degrees of freedom (discrete grid: 4, 6, 8, 12, 20)
    log_gamma_half_nu : float
        Precomputed gammaln(ν/2)
    log_gamma_half_nu_plus_half : float
        Precomputed gammaln((ν+1)/2)
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi
    
    # Robustified Kalman gain adjustment for heavy tails
    nu_adjust = nu / (nu + 3.0)
    if nu_adjust > 1.0:
        nu_adjust = 1.0
    
    for t in range(n):
        # Predict step with AR(1) dynamics
        mu_pred = phi * mu
        P_pred = phi_sq * P + q
        
        # Observation variance
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            forecast_scale = np.sqrt(S)
            
            # Student-t log-likelihood
            ll_t = student_t_logpdf_kernel(
                returns[t], nu, mu_pred, forecast_scale,
                log_gamma_half_nu, log_gamma_half_nu_plus_half
            )
            
            # Clamp contribution
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            # Robustified Kalman gain for heavy tails
            K = nu_adjust * P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    return mu_filtered, P_filtered, log_likelihood


# =============================================================================
# MOMENTUM-AUGMENTED KERNELS
# =============================================================================

@njit(cache=True, fastmath=True)
def momentum_phi_gaussian_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> tuple:
    """
    φ-Gaussian filter with momentum-adjusted drift prediction.
    
    Used by: CRSP, CELH, DPRO augmented models
    
    The momentum_adjustment is added to the predicted drift at each step,
    allowing the filter to incorporate trend/momentum signals WITHOUT
    modifying the Kalman update equations.
    
    Parameters
    ----------
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal to add to drift prediction
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi
    
    for t in range(n):
        # Momentum-augmented prediction (momentum enters ONLY here)
        mu_pred = phi * mu + momentum_adjustment[t]
        P_pred = phi_sq * P + q
        
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            innov_sq = innovation * innovation
            innov_sq_scaled = innov_sq / S
            if innov_sq_scaled > 100.0:
                innov_sq_scaled = 100.0
            
            ll_contrib = -0.5 * (_LOG_2PI + np.log(S) + innov_sq_scaled)
            if ll_contrib < -_MAX_LL_CONTRIB:
                ll_contrib = -_MAX_LL_CONTRIB
            log_likelihood += ll_contrib
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    return mu_filtered, P_filtered, log_likelihood


@njit(cache=True, fastmath=False)
def momentum_phi_student_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> tuple:
    """
    φ-Student-t filter with momentum-adjusted drift prediction.
    
    Used by: GLDW, MAGD, BKSY, ASTS augmented models
    
    Supports:
    - Hierarchical λ (Hλ← or Hλ→): affects momentum_adjustment computation upstream
    - EVT tail handling (EVTH/EVTM): affects vol estimation upstream
    - CVaR constraints (CST14-20%): affects position sizing downstream
    
    Momentum, EVT, and λ do NOT alter Kalman filter mathematics.
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi
    
    nu_adjust = nu / (nu + 3.0)
    if nu_adjust > 1.0:
        nu_adjust = 1.0
    
    for t in range(n):
        # Momentum-augmented prediction (momentum enters ONLY here)
        mu_pred = phi * mu + momentum_adjustment[t]
        P_pred = phi_sq * P + q
        
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            forecast_scale = np.sqrt(S)
            
            ll_t = student_t_logpdf_kernel(
                returns[t], nu, mu_pred, forecast_scale,
                log_gamma_half_nu, log_gamma_half_nu_plus_half
            )
            
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            K = nu_adjust * P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    return mu_filtered, P_filtered, log_likelihood
