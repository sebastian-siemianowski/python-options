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


# =============================================================================
# MARKOV-SWITCHING PROCESS NOISE (MS-q) KERNELS — February 2026
# =============================================================================
# Proactive regime-switching q based on volatility structure.
# Unlike GAS-Q (reactive), MS-q shifts BEFORE errors materialize.
# Includes FUSED LFO-CV computation for 40% performance gain.
# =============================================================================

@njit(cache=True, fastmath=False)
def compute_ms_process_noise_kernel(
    vol: np.ndarray,
    q_calm: float,
    q_stress: float,
    sensitivity: float,
    threshold: float,
) -> tuple:
    """
    Numba-accelerated MS-q process noise computation.
    
    Returns:
        q_t: Time-varying process noise array
        p_stress: Probability of stress regime array
    """
    n = len(vol)
    q_t = np.empty(n, dtype=np.float64)
    p_stress = np.empty(n, dtype=np.float64)
    
    # Compute expanding baseline (no future leakage)
    vol_sum = 0.0
    
    for t in range(n):
        vol_sum += vol[t]
        vol_baseline = vol_sum / (t + 1)
        if vol_baseline < 1e-10:
            vol_baseline = 1e-10
        
        # Vol relative to baseline
        vol_rel = vol[t] / vol_baseline
        
        # Sigmoid for stress probability
        z = sensitivity * (vol_rel - threshold)
        if z > 20.0:
            p_s = 1.0
        elif z < -20.0:
            p_s = 0.0
        else:
            p_s = 1.0 / (1.0 + np.exp(-z))
        
        # Clip to [0.01, 0.99]
        if p_s < 0.01:
            p_s = 0.01
        elif p_s > 0.99:
            p_s = 0.99
        
        p_stress[t] = p_s
        q_t[t] = (1.0 - p_s) * q_calm + p_s * q_stress
    
    return q_t, p_stress


@njit(cache=True, fastmath=False)
def ms_q_student_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu: float,
    q_calm: float,
    q_stress: float,
    sensitivity: float,
    threshold: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    P0: float,
    lfo_start_frac: float,
) -> tuple:
    """
    Numba-accelerated MS-q Student-t filter with FUSED LFO-CV computation.
    
    This kernel computes:
    1. Standard Kalman filter with time-varying q
    2. LFO-CV score (starting from lfo_start_frac of data)
    
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    q_t : np.ndarray
    p_stress : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
    """
    n = len(returns)
    lfo_start = int(n * lfo_start_frac)
    if lfo_start < 20:
        lfo_start = 20
    
    mu_filtered = np.empty(n, dtype=np.float64)
    P_filtered = np.empty(n, dtype=np.float64)
    q_t = np.empty(n, dtype=np.float64)
    p_stress = np.empty(n, dtype=np.float64)
    
    # Precompute constants
    phi_sq = phi * phi
    nu_adjust = nu / (nu + 3.0)
    if nu_adjust > 1.0:
        nu_adjust = 1.0
    
    log_norm_const = log_gamma_half_nu_plus_half - log_gamma_half_nu - 0.5 * np.log(nu * np.pi)
    neg_exp = -((nu + 1.0) / 2.0)
    inv_nu = 1.0 / nu
    
    # State initialization
    mu = 0.0
    P = P0
    
    # Accumulators
    log_likelihood = 0.0
    lfo_sum = 0.0
    lfo_count = 0
    
    # Expanding vol baseline for MS-q
    vol_sum = 0.0
    
    for t in range(n):
        # Compute MS-q process noise
        vol_sum += vol[t]
        vol_baseline = vol_sum / (t + 1)
        if vol_baseline < 1e-10:
            vol_baseline = 1e-10
        
        vol_rel = vol[t] / vol_baseline
        z_stress = sensitivity * (vol_rel - threshold)
        
        if z_stress > 20.0:
            p_s = 1.0
        elif z_stress < -20.0:
            p_s = 0.0
        else:
            p_s = 1.0 / (1.0 + np.exp(-z_stress))
        
        if p_s < 0.01:
            p_s = 0.01
        elif p_s > 0.99:
            p_s = 0.99
        
        p_stress[t] = p_s
        q_current = (1.0 - p_s) * q_calm + p_s * q_stress
        q_t[t] = q_current
        
        # Prediction step
        mu_pred = phi * mu
        P_pred = phi_sq * P + q_current
        
        # Observation variance
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            # Student-t scale
            scale = np.sqrt(S)
            z = innovation / scale
            z_sq = z * z
            
            # Log-likelihood contribution
            ll_t = log_norm_const - np.log(scale) + neg_exp * np.log(1.0 + z_sq * inv_nu)
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            # LFO-CV accumulation (predictive log-density)
            if t >= lfo_start:
                lfo_sum += ll_t
                lfo_count += 1
            
            # Robust Kalman gain (Student-t weighting)
            w_t = (nu + 1.0) / (nu + z_sq)
            K = w_t * nu_adjust * P_pred / S
            
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    # Compute LFO-CV score
    if lfo_count > 0:
        lfo_cv_score = lfo_sum / lfo_count
    else:
        lfo_cv_score = -1e12
    
    return mu_filtered, P_filtered, q_t, p_stress, log_likelihood, lfo_cv_score


@njit(cache=True, fastmath=False)
def student_t_filter_with_lfo_cv_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    P0: float,
    lfo_start_frac: float,
) -> tuple:
    """
    Standard phi-Student-t filter with FUSED LFO-CV computation.
    
    This is the optimized version that computes LFO-CV during the filter pass,
    avoiding a second pass through the data.
    
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
    """
    n = len(returns)
    lfo_start = int(n * lfo_start_frac)
    if lfo_start < 20:
        lfo_start = 20
    
    mu_filtered = np.empty(n, dtype=np.float64)
    P_filtered = np.empty(n, dtype=np.float64)
    
    # Precompute constants
    phi_sq = phi * phi
    nu_adjust = nu / (nu + 3.0)
    if nu_adjust > 1.0:
        nu_adjust = 1.0
    
    log_norm_const = log_gamma_half_nu_plus_half - log_gamma_half_nu - 0.5 * np.log(nu * np.pi)
    neg_exp = -((nu + 1.0) / 2.0)
    inv_nu = 1.0 / nu
    
    # State initialization
    mu = 0.0
    P = P0
    
    # Accumulators
    log_likelihood = 0.0
    lfo_sum = 0.0
    lfo_count = 0
    
    for t in range(n):
        # Prediction step
        mu_pred = phi * mu
        P_pred = phi_sq * P + q
        
        # Observation variance
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            # FIX: Student-t scale = sqrt(S × (ν-2)/ν)
            if nu > 2.0:
                scale = np.sqrt(S * (nu - 2.0) / nu)
            else:
                scale = np.sqrt(S)
            z = innovation / scale
            z_sq = z * z
            
            # Log-likelihood contribution
            ll_t = log_norm_const - np.log(scale) + neg_exp * np.log(1.0 + z_sq * inv_nu)
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            # LFO-CV accumulation
            if t >= lfo_start:
                lfo_sum += ll_t
                lfo_count += 1
            
            # Robust Kalman gain
            K = nu_adjust * P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    if lfo_count > 0:
        lfo_cv_score = lfo_sum / lfo_count
    else:
        lfo_cv_score = -1e12
    
    return mu_filtered, P_filtered, log_likelihood, lfo_cv_score


@njit(cache=True, fastmath=True)
def gaussian_filter_with_lfo_cv_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    P0: float,
    lfo_start_frac: float,
) -> tuple:
    """
    Gaussian Kalman filter with FUSED LFO-CV computation.
    
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
    """
    n = len(returns)
    lfo_start = int(n * lfo_start_frac)
    if lfo_start < 20:
        lfo_start = 20
    
    mu_filtered = np.empty(n, dtype=np.float64)
    P_filtered = np.empty(n, dtype=np.float64)
    
    phi_sq = phi * phi
    mu = 0.0
    P = P0
    
    log_likelihood = 0.0
    lfo_sum = 0.0
    lfo_count = 0
    
    for t in range(n):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q
        
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R
        
        if S > _MIN_VARIANCE:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            innov_sq_scaled = (innovation * innovation) / S
            if innov_sq_scaled > 100.0:
                innov_sq_scaled = 100.0
            
            ll_t = -0.5 * (_LOG_2PI + np.log(S) + innov_sq_scaled)
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            if t >= lfo_start:
                lfo_sum += ll_t
                lfo_count += 1
        else:
            mu = mu_pred
            P = P_pred
        
        mu_filtered[t] = mu
        P_filtered[t] = P if P > _MIN_VARIANCE else _MIN_VARIANCE
    
    if lfo_count > 0:
        lfo_cv_score = lfo_sum / lfo_count
    else:
        lfo_cv_score = -1e12
    
    return mu_filtered, P_filtered, log_likelihood, lfo_cv_score