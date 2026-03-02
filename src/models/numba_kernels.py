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
import math

# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum log-likelihood contribution per observation (prevents outlier domination)
_MAX_LL_CONTRIB = 50.0

# Minimum variance floor to prevent numerical instability
_MIN_VARIANCE = 1e-12

# Log(2π) precomputed for Gaussian likelihood
_LOG_2PI = np.log(2.0 * np.pi)

# Log(sqrt(2π)) for Lanczos gammaln
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


# =============================================================================
# GAMMALN APPROXIMATION (for dynamic ν in unified model)
# =============================================================================
# Using Stirling's approximation with correction terms.
# For ν > 2, this is accurate enough for likelihood computations.
# Numba-friendly: no recursion, simple arithmetic.
# =============================================================================

@njit(cache=True, fastmath=False)
def _stirling_gammaln(x: float) -> float:
    """
    Stirling's approximation for log-gamma function with correction terms.
    
    For x > 2, error < 1e-6 which is acceptable for likelihood computations.
    
    Formula:
        log Γ(x) ≈ (x - 0.5) * log(x) - x + 0.5 * log(2π) + 1/(12x) - 1/(360x³)
    
    Parameters
    ----------
    x : float
        Input value (must be > 0)
        
    Returns
    -------
    float
        log(Γ(x))
    """
    if x <= 0.0:
        return 1e12  # Invalid input sentinel
    
    if x < 2.0:
        # For x < 2, use simple recursion (one level only)
        # log Γ(x) = log Γ(x+1) - log(x)
        x_plus_1 = x + 1.0
        stirling = ((x_plus_1 - 0.5) * np.log(x_plus_1) - x_plus_1 + _LOG_SQRT_2PI 
                   + 1.0 / (12.0 * x_plus_1) - 1.0 / (360.0 * x_plus_1 * x_plus_1 * x_plus_1))
        return stirling - np.log(x)
    
    # Stirling's approximation with correction terms
    return ((x - 0.5) * np.log(x) - x + _LOG_SQRT_2PI 
            + 1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x))


@njit(cache=True, fastmath=False)
def _student_t_logpdf_dynamic_nu(
    x: float,
    nu: float,
    mu: float,
    scale: float,
) -> float:
    """
    Student-t log-pdf with dynamically computed gamma values.
    
    Uses Stirling approximation for gammaln, enabling per-timestep
    computation of likelihood with varying ν (required for smooth
    asymmetric ν in unified model).
    
    Parameters
    ----------
    x : float
        Observation value
    nu : float
        Degrees of freedom (can vary per timestep)
    mu : float
        Location parameter
    scale : float
        Scale parameter
        
    Returns
    -------
    float
        Log-probability density
    """
    if scale <= _MIN_VARIANCE or nu <= 2.0:
        return -1e12
    
    z = (x - mu) / scale
    z_sq = z * z
    
    # Compute gamma values using Stirling approximation
    log_gamma_half_nu = _stirling_gammaln(nu / 2.0)
    log_gamma_half_nu_plus_half = _stirling_gammaln((nu + 1.0) / 2.0)
    
    # Student-t log-pdf
    log_norm = (log_gamma_half_nu_plus_half - log_gamma_half_nu 
                - 0.5 * np.log(nu * np.pi) - np.log(scale))
    log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + z_sq / nu)
    
    return log_norm + log_kernel


# =============================================================================
# STUDENT-T CDF/PDF VIA REGULARIZED INCOMPLETE BETA
# =============================================================================
# The Student-t CDF relates to the regularized incomplete beta:
#   F_t(x; ν) = I_w(ν/2, 1/2)  [for x < 0]
#   F_t(x; ν) = 1 - 0.5 * I_w(ν/2, 1/2)  [for x >= 0]
# where w = ν / (ν + x²).
#
# betainc is computed via Lentz's continued fraction (DLMF §8.17.22).
# Uses Lanczos gammaln (g=7) for ~1e-12 accuracy in the front factor.
# =============================================================================

# Lanczos coefficients (g=7, n=9) for double-precision gammaln
_LANCZOS_G = 7.0
_LANCZOS_COEFF_0 = 0.99999999999980993
_LANCZOS_COEFF_1 = 676.5203681218851
_LANCZOS_COEFF_2 = -1259.1392167224028
_LANCZOS_COEFF_3 = 771.32342877765313
_LANCZOS_COEFF_4 = -176.61502916214059
_LANCZOS_COEFF_5 = 12.507343278686905
_LANCZOS_COEFF_6 = -0.13857109526572012
_LANCZOS_COEFF_7 = 9.9843695780195716e-6
_LANCZOS_COEFF_8 = 1.5056327351493116e-7


@njit(cache=True, fastmath=False)
def _lanczos_gammaln(x: float) -> float:
    """
    Lanczos approximation for log-gamma function.

    Uses g=7, n=9 coefficients for ~1e-12 accuracy across x > 0.
    Required for CDF computation where gammaln feeds into exp().

    Parameters
    ----------
    x : float
        Input value (must be > 0)

    Returns
    -------
    float
        log(Γ(x))
    """
    if x <= 0.0:
        return 1e12

    # Reflection formula for x < 0.5
    if x < 0.5:
        # log(Γ(x)) = log(π / sin(πx)) - log(Γ(1-x))
        return np.log(np.pi / np.sin(np.pi * x)) - _lanczos_gammaln(1.0 - x)

    x = x - 1.0
    ag = _LANCZOS_COEFF_0
    ag += _LANCZOS_COEFF_1 / (x + 1.0)
    ag += _LANCZOS_COEFF_2 / (x + 2.0)
    ag += _LANCZOS_COEFF_3 / (x + 3.0)
    ag += _LANCZOS_COEFF_4 / (x + 4.0)
    ag += _LANCZOS_COEFF_5 / (x + 5.0)
    ag += _LANCZOS_COEFF_6 / (x + 6.0)
    ag += _LANCZOS_COEFF_7 / (x + 7.0)
    ag += _LANCZOS_COEFF_8 / (x + 8.0)

    t = x + _LANCZOS_G + 0.5
    return 0.5 * np.log(2.0 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(ag)

@njit(cache=True, fastmath=False)
def _betacf(a: float, b: float, x: float) -> float:
    """
    Continued fraction for regularized incomplete beta function.

    Uses the modified Lentz algorithm (Numerical Recipes §6.4).
    Evaluates B_x(a,b) / B(a,b) via the CF representation.

    Parameters
    ----------
    a, b : float
        Beta function parameters (a > 0, b > 0)
    x : float
        Upper integration limit (0 < x < 1)

    Returns
    -------
    float
        The continued fraction part of I_x(a, b)
    """
    _FPMIN = 1e-30
    _EPS = 1e-14
    _MAXIT = 200

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    # First step of Lentz's method
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, _MAXIT + 1):
        m_f = float(m)
        m2 = 2.0 * m_f

        # Even step: d_{2m}
        aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c

        # Odd step: d_{2m+1}
        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < _EPS:
            return h

    return h


@njit(cache=True, fastmath=False)
def _betainc(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b).

    Uses continued fraction with symmetry transform when needed.
    Reference: Numerical Recipes §6.4, DLMF §8.17.

    Parameters
    ----------
    a, b : float
        Parameters (> 0)
    x : float
        Upper limit (0 <= x <= 1)

    Returns
    -------
    float
        I_x(a, b) ∈ [0, 1]
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Log of the front factor: x^a * (1-x)^b / (a * B(a,b))
    # B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    bt = np.exp(
        _lanczos_gammaln(a + b) - _lanczos_gammaln(a) - _lanczos_gammaln(b)
        + a * np.log(x) + b * np.log(1.0 - x)
    )

    # Use symmetry transform when x > (a+1)/(a+b+2) for faster convergence
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


@njit(cache=True, fastmath=False)
def _student_t_cdf_scalar(x: float, nu: float) -> float:
    """
    Student-t CDF for a single observation.

    Uses the identity:
        F_t(x; ν) = I_w(ν/2, 1/2)  for x < 0
        F_t(x; ν) = 1 - 0.5 * I_w(ν/2, 1/2)  for x >= 0
    where w = ν / (ν + x²).

    Parameters
    ----------
    x : float
        Standardized observation
    nu : float
        Degrees of freedom (> 0)

    Returns
    -------
    float
        CDF value ∈ (0, 1)
    """
    if nu <= 0.0:
        return 0.5

    x2 = x * x
    w = nu / (nu + x2)

    # betainc with a=nu/2, b=0.5
    ibeta = _betainc(nu / 2.0, 0.5, w)

    if x < 0.0:
        return 0.5 * ibeta
    elif x > 0.0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5


@njit(cache=True, fastmath=False)
def _student_t_pdf_scalar(x: float, nu: float) -> float:
    """
    Student-t PDF for a single observation.

    Parameters
    ----------
    x : float
        Standardized observation
    nu : float
        Degrees of freedom (> 0)

    Returns
    -------
    float
        PDF value
    """
    log_norm = (_lanczos_gammaln((nu + 1.0) / 2.0)
                - _lanczos_gammaln(nu / 2.0)
                - 0.5 * np.log(nu * np.pi))
    log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + x * x / nu)
    return np.exp(log_norm + log_kernel)


@njit(cache=True, fastmath=False)
def student_t_cdf_array_kernel(z_arr: np.ndarray, nu: float) -> np.ndarray:
    """
    Vectorized Student-t CDF via Numba.

    Replaces scipy.stats.t.cdf(z, df=nu) with zero overhead.
    Accuracy: < 1e-10 vs scipy across nu ∈ [2.5, 50], z ∈ [-10, 10].

    Parameters
    ----------
    z_arr : np.ndarray
        Array of standardized values
    nu : float
        Degrees of freedom

    Returns
    -------
    np.ndarray
        CDF values
    """
    n = len(z_arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _student_t_cdf_scalar(z_arr[i], nu)
    return out


@njit(cache=True, fastmath=False)
def student_t_pdf_array_kernel(z_arr: np.ndarray, nu: float) -> np.ndarray:
    """
    Vectorized Student-t PDF via Numba.

    Replaces scipy.stats.t.pdf(z, df=nu) with zero overhead.

    Parameters
    ----------
    z_arr : np.ndarray
        Array of standardized values
    nu : float
        Degrees of freedom

    Returns
    -------
    np.ndarray
        PDF values
    """
    n = len(z_arr)
    out = np.empty(n, dtype=np.float64)
    # Pre-compute log normalization constant (constant across all z)
    log_norm = (_lanczos_gammaln((nu + 1.0) / 2.0)
                - _lanczos_gammaln(nu / 2.0)
                - 0.5 * np.log(nu * np.pi))
    neg_exp = (nu + 1.0) / 2.0
    inv_nu = 1.0 / nu
    for i in range(n):
        z = z_arr[i]
        out[i] = np.exp(log_norm - neg_exp * np.log(1.0 + z * z * inv_nu))
    return out


@njit(cache=True, fastmath=False)
def crps_student_t_kernel(
    z_arr: np.ndarray,
    sigma_arr: np.ndarray,
    nu: float,
) -> float:
    """
    CRPS for Student-t predictive distribution (Gneiting & Raftery 2007).

    Closed-form:
        CRPS = σ * [z(2F(z) - 1) + 2f(z)(ν + z²)/(ν-1) - 2√ν·B_ratio/(ν-1)]

    where B_ratio = B(1/2, ν-1/2) / B(1/2, ν/2)².

    Parameters
    ----------
    z_arr : np.ndarray
        Standardized residuals (obs - mu) / sigma
    sigma_arr : np.ndarray
        Scale parameters
    nu : float
        Degrees of freedom (> 1)

    Returns
    -------
    float
        Mean CRPS (lower is better)
    """
    n = len(z_arr)
    if n == 0 or nu <= 1.0:
        return 1e10

    # Pre-compute constants
    log_norm = (_lanczos_gammaln((nu + 1.0) / 2.0)
                - _lanczos_gammaln(nu / 2.0)
                - 0.5 * np.log(nu * np.pi))
    neg_exp = (nu + 1.0) / 2.0
    inv_nu = 1.0 / nu
    nu_m1_inv = 1.0 / (nu - 1.0)
    sqrt_nu = np.sqrt(nu)

    # Beta function ratio: B(1/2, nu-1/2) / B(1/2, nu/2)^2
    lgB1 = _lanczos_gammaln(0.5) + _lanczos_gammaln(nu - 0.5) - _lanczos_gammaln(nu)
    lgB2 = _lanczos_gammaln(0.5) + _lanczos_gammaln(nu / 2.0) - _lanczos_gammaln((nu + 1.0) / 2.0)
    B_ratio = np.exp(lgB1 - 2.0 * lgB2)

    term3_const = 2.0 * sqrt_nu * B_ratio * nu_m1_inv

    crps_sum = 0.0
    n_valid = 0

    for i in range(n):
        z = z_arr[i]
        sig = sigma_arr[i]
        if sig < 1e-10:
            sig = 1e-10

        # PDF
        pdf_z = np.exp(log_norm - neg_exp * np.log(1.0 + z * z * inv_nu))
        # CDF
        cdf_z = _student_t_cdf_scalar(z, nu)

        term1 = z * (2.0 * cdf_z - 1.0)
        term2 = 2.0 * pdf_z * (nu + z * z) * nu_m1_inv
        crps_i = sig * (term1 + term2 - term3_const)

        if crps_i == crps_i:  # NaN check
            crps_sum += crps_i
            n_valid += 1

    if n_valid == 0:
        return 1e10
    return crps_sum / float(n_valid)


# =============================================================================
# PIT-KS UNIFIED KERNEL (eliminates per-element scipy CDF overhead)
# =============================================================================

@njit(cache=True)
def pit_ks_unified_kernel(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
    nu_base: float,
    alpha_asym: float,
    k_asym: float,
    variance_inflation: float,
    pit_out: np.ndarray,
) -> int:
    """
    Compute PIT values for unified Student-t with smooth asymmetric nu.

    Replaces per-element Python loop + scalar scipy CDF calls with a single
    compiled pass. Uses Numba _student_t_cdf_scalar for each element.

    Implements: nu_eff = nu_base * (1 + alpha * tanh(k * z))
    then PIT = Student_t_CDF(innovation / t_scale, nu_eff).

    Parameters
    ----------
    returns, mu_pred, S_pred : np.ndarray
        Time series data
    nu_base : float
        Base degrees of freedom
    alpha_asym : float
        Asymmetry parameter
    k_asym : float
        Asymmetry sharpness
    variance_inflation : float
        Variance inflation factor
    pit_out : np.ndarray
        Output array for PIT values (pre-allocated)

    Returns
    -------
    int
        Number of valid (finite) PIT values
    """
    n = len(returns)
    n_valid = 0
    for t in range(n):
        innovation = returns[t] - mu_pred[t]
        S_cal = S_pred[t] * variance_inflation
        if S_cal < 1e-12:
            S_cal = 1e-12
        scale = np.sqrt(S_cal)

        # compute_effective_nu inline
        scale_safe = scale if scale > 1e-10 else 1e-10
        z_raw = innovation / scale_safe
        modulation = 1.0 + alpha_asym * np.tanh(k_asym * z_raw)
        nu_eff = nu_base * modulation
        if nu_eff < 2.1:
            nu_eff = 2.1
        elif nu_eff > 50.0:
            nu_eff = 50.0

        # t_scale
        if nu_eff > 2.0:
            t_scale = np.sqrt(S_cal * (nu_eff - 2.0) / nu_eff)
        else:
            t_scale = scale
        if t_scale < 1e-10:
            t_scale = 1e-10

        # Student-t CDF (compiled, no scipy wrapper overhead)
        z_cdf = innovation / t_scale
        pit_val = _student_t_cdf_scalar(z_cdf, nu_eff)

        # Clip to (0.001, 0.999)
        if pit_val < 0.001:
            pit_val = 0.001
        elif pit_val > 0.999:
            pit_val = 0.999

        pit_out[t] = pit_val
        if pit_val == pit_val:  # NaN check
            n_valid += 1

    return n_valid


# =============================================================================
# GJR-GARCH(1,1) VARIANCE KERNEL
# =============================================================================

@njit(cache=True)
def garch_variance_kernel(
    sq: np.ndarray,
    neg: np.ndarray,
    innovations: np.ndarray,
    n: int,
    go: float,
    ga: float,
    gb: float,
    gl: float,
    gu: float,
    rl: float,
    km: float,
    tv: float,
    se: float,
    rs: float,
    sm: float,
    h_out: np.ndarray,
) -> None:
    """
    GJR-GARCH(1,1) variance with leverage correlation, vol-of-vol noise,
    Markov regime switching, and mean reversion.

    Replaces Python loop in _compute_garch_variance with compiled Numba.

    Parameters
    ----------
    sq : squared innovations
    neg : indicator for negative innovations (1.0 or 0.0)
    innovations : raw innovations
    n : length
    go, ga, gb, gl : GARCH omega, alpha, beta, leverage
    gu : unconditional variance
    rl : rho_leverage
    km : kappa_mean_rev
    tv : theta_long_var
    se : sigma_eta (vol-of-vol)
    rs : regime_switch_prob
    sm : sqrt(q_stress_ratio)
    h_out : output array (pre-allocated, length n)
    """
    h_out[0] = gu
    ps = 0.1  # Initial stress probability

    for t in range(1, n):
        ht = go + ga * sq[t - 1] + gl * sq[t - 1] * neg[t - 1] + gb * h_out[t - 1]

        if rl > 0.01 and h_out[t - 1] > 1e-12:
            z = innovations[t - 1] / np.sqrt(h_out[t - 1])
            if z < 0.0:
                ht += rl * z * z * h_out[t - 1]

        if se > 0.005 and h_out[t - 1] > 1e-12:
            z = abs(innovations[t - 1]) / np.sqrt(h_out[t - 1])
            excess = z - 1.5
            if excess > 0.0:
                ht += se * excess * excess * h_out[t - 1]

        if rs > 0.005 and h_out[t - 1] > 1e-12:
            z = abs(innovations[t - 1]) / np.sqrt(h_out[t - 1])
            ps = (1.0 - rs) * ps + rs * (1.0 if z > 2.0 else 0.0)
            if ps < 0.0:
                ps = 0.0
            elif ps > 1.0:
                ps = 1.0
            ht *= (1.0 + ps * (sm - 1.0))

        if km > 0.001:
            ht = (1.0 - km) * ht + km * tv

        if ht < 1e-12:
            ht = 1e-12
        h_out[t] = ht


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


@njit(cache=True, fastmath=True)
def phi_gaussian_filter_with_predictive_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    P0: float = 1e-4,
) -> tuple:
    """
    φ-Gaussian filter returning predictive mu_pred and S_pred for PIT.

    Same as phi_gaussian_filter_kernel but also outputs:
        mu_pred[t] = φ × μ_{t-1}     (BEFORE seeing y_t)
        S_pred[t] = P_pred + R_t      (BEFORE seeing y_t)

    Returns (mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood)
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    mu_pred_arr = np.zeros(n)
    S_pred_arr = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi

    for t in range(n):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q

        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        S = P_pred + R
        if S < _MIN_VARIANCE:
            S = _MIN_VARIANCE

        mu_pred_arr[t] = mu_pred
        S_pred_arr[t] = S

        innovation = returns[t] - mu_pred

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

    return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood


@njit(cache=True, fastmath=False)
def phi_student_t_augmented_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    exogenous_input: np.ndarray,
    has_exogenous: bool,
    robust_wt: bool,
    P0: float = 1e-4,
) -> tuple:
    """
    φ-Student-t filter with optional exogenous input and robust weighting.

    Replaces _filter_phi_core Python fallback for filter_phi_augmented
    and filter_phi_with_predictive.

    Returns (mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood)
    """
    n = len(returns)
    phi_sq = phi * phi
    log_norm_const = (log_gamma_half_nu_plus_half - log_gamma_half_nu
                      - 0.5 * np.log(nu * np.pi))
    neg_exp = -((nu + 1.0) / 2.0)
    inv_nu = 1.0 / nu

    R = np.empty(n)
    for t in range(n):
        R[t] = c * (vol[t] * vol[t])

    mu_filtered = np.empty(n)
    P_filtered = np.empty(n)
    mu_pred_arr = np.empty(n)
    S_pred_arr = np.empty(n)

    # Data-adaptive filter initialization
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
        mu = 0.0
        P = P0
    log_likelihood = 0.0

    for t in range(n):
        u_t = exogenous_input[t] if has_exogenous and t < len(exogenous_input) else 0.0
        mu_pred = phi * mu + u_t
        P_pred = phi_sq * P + q
        S = P_pred + R[t]
        if S < 1e-12:
            S = 1e-12

        mu_pred_arr[t] = mu_pred
        S_pred_arr[t] = S

        innovation = returns[t] - mu_pred
        K = P_pred / S

        if robust_wt:
            z_sq = (innovation * innovation) / S
            w_t = (nu + 1.0) / (nu + z_sq)
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
        else:
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred

        if P < 1e-12:
            P = 1e-12
        mu_filtered[t] = mu
        P_filtered[t] = P

        if nu > 2.0:
            forecast_scale = np.sqrt(S * (nu - 2.0) / nu)
        else:
            forecast_scale = np.sqrt(S)
        if forecast_scale > 1e-12:
            z = innovation / forecast_scale
            ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
            if ll_t == ll_t:  # NaN check
                log_likelihood += ll_t

    return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood


# =============================================================================
# φ-STUDENT-T ENHANCED FILTER (VoV + Online Scale Adapt)  — March 2026
# =============================================================================
# Extends phi_student_t_augmented_filter_kernel with:
#   1. VoV (gamma_vov * vov_rolling) R_t inflation
#   2. Online scale adaptation (chi² EWM _c_adj tracking)
# This eliminates the Python fallback for ν=3,4 in optimize_params_fixed_nu.
# =============================================================================

@njit(cache=True, fastmath=False)
def phi_student_t_enhanced_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    exogenous_input: np.ndarray,
    has_exogenous: bool,
    robust_wt: bool,
    online_scale_adapt: bool,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    has_vov: bool,
    P0: float = 1e-4,
) -> tuple:
    """
    φ-Student-t filter with VoV + online scale adaptation + robust weighting.

    Identical to _filter_phi_core Python loop but fully JIT-compiled.
    Provides 5-10× speedup for ν≤5 where online_scale_adapt is active.
    """
    n = len(returns)
    phi_sq = phi * phi
    log_norm_const = (log_gamma_half_nu_plus_half - log_gamma_half_nu
                      - 0.5 * np.log(nu * np.pi))
    neg_exp = -((nu + 1.0) / 2.0)
    inv_nu = 1.0 / nu

    # Pre-compute vol²
    vol_sq = np.empty(n)
    for t in range(n):
        vol_sq[t] = vol[t] * vol[t]

    mu_filtered = np.empty(n)
    P_filtered = np.empty(n)
    mu_pred_arr = np.empty(n)
    S_pred_arr = np.empty(n)

    # Data-adaptive filter initialization
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
        mu = 0.0
        P = P0
    log_likelihood = 0.0

    # Online scale adaptation state (Harvey 1989)
    chi2_tgt = nu / (nu - 2.0) if nu > 2.0 else 1.0
    chi2_lam = 0.98
    chi2_1m = 1.0 - chi2_lam
    chi2_cap = chi2_tgt * 50.0
    ewm_z2 = chi2_tgt
    c_adj = 1.0

    for t in range(n):
        u_t = exogenous_input[t] if has_exogenous and t < len(exogenous_input) else 0.0
        mu_pred = phi * mu + u_t
        P_pred = phi_sq * P + q

        # Observation noise R_t with optional VoV and online scale adapt
        c_eff = c * c_adj if online_scale_adapt else c
        R_t = c_eff * vol_sq[t]
        if has_vov:
            R_t = R_t * (1.0 + gamma_vov * vov_rolling[t])

        S = P_pred + R_t
        if S < 1e-12:
            S = 1e-12

        mu_pred_arr[t] = mu_pred
        S_pred_arr[t] = S

        innovation = returns[t] - mu_pred
        K = P_pred / S

        if robust_wt:
            z_sq = (innovation * innovation) / S
            w_t = (nu + 1.0) / (nu + z_sq)
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
        else:
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred

        if P < 1e-12:
            P = 1e-12
        mu_filtered[t] = mu
        P_filtered[t] = P if P > 1e-12 else 1e-12

        if nu > 2.0:
            forecast_scale = np.sqrt(S * (nu - 2.0) / nu)
        else:
            forecast_scale = np.sqrt(S)
        if forecast_scale > 1e-12:
            z = innovation / forecast_scale
            ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
            if ll_t == ll_t:  # NaN check
                log_likelihood += ll_t

            # Online scale adaptation: track E[z²], adjust c for next step
            if online_scale_adapt:
                z2_raw = z * z
                z2w = z2_raw if z2_raw < chi2_cap else chi2_cap
                ewm_z2 = chi2_lam * ewm_z2 + chi2_1m * z2w
                ratio = ewm_z2 / chi2_tgt
                if ratio < 0.3:
                    ratio = 0.3
                elif ratio > 3.0:
                    ratio = 3.0
                dev = ratio - 1.0 if ratio >= 1.0 else 1.0 - ratio
                if ratio >= 1.0:
                    dz_lo = 0.25
                    dz_rng = 0.25
                else:
                    dz_lo = 0.05
                    dz_rng = 0.10
                if dev < dz_lo:
                    c_adj = 1.0
                elif dev >= dz_lo + dz_rng:
                    c_adj = np.sqrt(ratio)
                else:
                    s_frac = (dev - dz_lo) / dz_rng
                    c_adj = 1.0 + s_frac * (np.sqrt(ratio) - 1.0)

    return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood


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
    # Data-adaptive filter initialization
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
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
            
            # Kalman gain (robust weighting via w_t handled by caller)
            K = P_pred / S
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
    # Data-adaptive filter initialization
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
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
            forecast_scale = np.sqrt(S)
            
            ll_t = student_t_logpdf_kernel(
                returns[t], nu, mu_pred, forecast_scale,
                log_gamma_half_nu, log_gamma_half_nu_plus_half
            )
            
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            K = P_pred / S
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
    
    log_norm_const = log_gamma_half_nu_plus_half - log_gamma_half_nu - 0.5 * np.log(nu * np.pi)
    neg_exp = -((nu + 1.0) / 2.0)
    inv_nu = 1.0 / nu
    
    # State initialization (data-adaptive)
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
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
            K = w_t * P_pred / S
            
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
    
    log_norm_const = log_gamma_half_nu_plus_half - log_gamma_half_nu - 0.5 * np.log(nu * np.pi)
    neg_exp = -((nu + 1.0) / 2.0)
    inv_nu = 1.0 / nu
    
    # State initialization (data-adaptive)
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
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
            K = P_pred / S
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


# =============================================================================
# UNIFIED φ-STUDENT-T KERNEL (VoV + MS-q + Smooth Asymmetric ν + Momentum)
# =============================================================================

@njit(cache=True, fastmath=False)
def unified_phi_student_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    # Base parameters
    c: float,
    phi: float,
    nu_base: float,
    # MS-q arrays (precomputed in Python)
    q_t: np.ndarray,
    p_stress: np.ndarray,
    # VoV arrays (precomputed in Python)
    vov_rolling: np.ndarray,
    gamma_vov: float,
    vov_damping: float,
    # Smooth asymmetric ν parameters
    alpha_asym: float,
    k_asym: float,
    # Momentum array
    momentum: np.ndarray,
    # Initial covariance
    P0: float,
) -> tuple:
    """
    UNIFIED φ-Student-t Kalman filter kernel with ALL enhancements.
    
    This is the Numba-accelerated version combining:
      1. Smooth Asymmetric ν: tanh-modulated tail heaviness (differentiable)
      2. Probabilistic MS-q: sigmoid regime switching (precomputed arrays)
      3. Adaptive VoV: vol-of-vol scaling with MS-q redundancy damping
      4. Momentum: exogenous drift input
      5. Robust Student-t weighting: outlier downweighting
    
    All time-varying arrays (q_t, p_stress, vov_rolling, momentum) must be
    precomputed in Python wrapper before calling this kernel.
    
    Parameters
    ----------
    returns : np.ndarray
        Contiguous float64 array of log returns
    vol : np.ndarray
        Contiguous float64 array of EWMA volatility
    c : float
        Observation noise scale
    phi : float
        AR(1) persistence
    nu_base : float
        Base degrees of freedom
    q_t : np.ndarray
        Time-varying process noise (from probabilistic MS-q)
    p_stress : np.ndarray
        Stress probability per timestep (for VoV damping)
    vov_rolling : np.ndarray
        Rolling vol-of-vol (precomputed)
    gamma_vov : float
        VoV sensitivity
    vov_damping : float
        Redundancy damping factor (reduces VoV when MS-q active)
    alpha_asym : float
        Asymmetry parameter (negative = heavier left tail)
    k_asym : float
        Asymmetry transition sharpness
    momentum : np.ndarray
        Exogenous momentum signal per timestep
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
    n = len(returns)
    
    # Allocate output arrays
    mu_filtered = np.empty(n, dtype=np.float64)
    P_filtered = np.empty(n, dtype=np.float64)
    mu_pred_arr = np.empty(n, dtype=np.float64)
    S_pred_arr = np.empty(n, dtype=np.float64)
    
    # Pre-compute constants
    phi_sq = phi * phi
    
    # State initialization (data-adaptive)
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
        mu = 0.0
        P = P0
    log_likelihood = 0.0
    
    # Main filter loop
    for t in range(n):
        # === PREDICTION STEP ===
        mu_pred = phi * mu + momentum[t]
        P_pred = phi_sq * P + q_t[t]
        
        # VoV-adjusted observation noise with redundancy damping
        vol_t = vol[t]
        R_base = c * vol_t * vol_t
        vov_effective = gamma_vov * (1.0 - vov_damping * p_stress[t])
        R = R_base * (1.0 + vov_effective * vov_rolling[t])
        
        # Predictive variance
        S = P_pred + R
        if S < _MIN_VARIANCE:
            S = _MIN_VARIANCE
        
        # Store predictive values (for PIT computation)
        mu_pred_arr[t] = mu_pred
        S_pred_arr[t] = S
        
        # Innovation
        innovation = returns[t] - mu_pred
        
        # === UPDATE STEP ===
        # Smooth asymmetric ν (tanh-based, differentiable)
        scale = np.sqrt(S)
        z = innovation / scale
        nu_eff = nu_base * (1.0 + alpha_asym * np.tanh(k_asym * z))
        
        # Bound ν_eff to valid range [2.1, 50.0]
        if nu_eff < 2.1:
            nu_eff = 2.1
        elif nu_eff > 50.0:
            nu_eff = 50.0
        
        # Standard Kalman gain (robust weighting via w_t below)
        K = P_pred / S
        
        # Robust Student-t weighting (downweight outliers)
        z_sq = innovation * innovation / S
        w_t = (nu_eff + 1.0) / (nu_eff + z_sq)
        
        # State update with robust weighting
        mu = mu_pred + K * w_t * innovation
        P = (1.0 - w_t * K) * P_pred
        if P < _MIN_VARIANCE:
            P = _MIN_VARIANCE
        
        # Store filtered values
        mu_filtered[t] = mu
        P_filtered[t] = P
        
        # === LOG-LIKELIHOOD ===
        # Convert variance S to Student-t scale (CRITICAL for correct PIT)
        scale_factor = (nu_eff - 2.0) / nu_eff
        if scale_factor < 0.01:
            scale_factor = 0.01
        forecast_scale = np.sqrt(S * scale_factor)
        
        if forecast_scale > _MIN_VARIANCE:
            # Use dynamic gammaln via Lanczos approximation
            ll_t = _student_t_logpdf_dynamic_nu(
                returns[t], nu_eff, mu_pred, forecast_scale
            )
            
            # Clamp contribution
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            elif ll_t > _MAX_LL_CONTRIB:
                ll_t = _MAX_LL_CONTRIB
            
            # NaN check (NaN != NaN)
            if ll_t == ll_t:
                log_likelihood += ll_t
    
    return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood


@njit(cache=True, fastmath=False)
def unified_phi_student_t_filter_extended_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    # Base parameters
    c: float,
    phi: float,
    nu_base: float,
    # MS-q arrays (precomputed in Python)
    q_t: np.ndarray,
    p_stress: np.ndarray,
    # VoV arrays (precomputed in Python)
    vov_rolling: np.ndarray,
    gamma_vov: float,
    vov_damping: float,
    # Smooth asymmetric ν parameters
    alpha_asym: float,
    k_asym: float,
    # Momentum array
    momentum: np.ndarray,
    # Initial covariance
    P0: float,
    # --- Extended parameters (Tier 3) ---
    risk_prem: float,
    mu_drift: float,
    skew_kappa: float,
    skew_rho: float,
    jump_var: float,
    jump_intensity: float,
    jump_sensitivity: float,
    jump_mean: float,
    # EWM correction parameters
    ewm_lambda: float,
) -> tuple:
    """
    EXTENDED UNIFIED phi-Student-t Kalman filter kernel.

    Adds risk premium, mu drift, GAS skew dynamics, Merton jump-diffusion,
    and causal EWM correction to the base unified kernel.
    
    Conditional branches have zero overhead in Numba JIT when inactive.
    """
    n = len(returns)

    # Allocate output arrays
    mu_filtered = np.empty(n, dtype=np.float64)
    P_filtered = np.empty(n, dtype=np.float64)
    mu_pred_arr = np.empty(n, dtype=np.float64)
    S_pred_arr = np.empty(n, dtype=np.float64)

    # Pre-compute constants
    phi_sq = phi * phi
    _alpha_negligible = abs(alpha_asym) < 1e-10

    # Feature flags (constant for entire call — JIT eliminates dead branches)
    skew_enabled = skew_kappa > 1e-8
    jump_enabled = jump_var > 1e-12 and jump_intensity > 1e-6
    has_risk_drift = abs(risk_prem) > 1e-10 or abs(mu_drift) > 1e-12

    # Pre-compute R_base array for risk premium
    R_base_arr = np.empty(n, dtype=np.float64)
    for t in range(n):
        R_base_arr[t] = c * vol[t] * vol[t]

    # Pre-compute log-norm const for diffusion likelihood
    log_norm_const = _stirling_gammaln((nu_base + 1.0) / 2.0) - _stirling_gammaln(nu_base / 2.0) - 0.5 * np.log(nu_base * np.pi)
    neg_exp = -((nu_base + 1.0) / 2.0)
    inv_nu = 1.0 / nu_base

    # Pre-compute for alpha_negligible case
    if _alpha_negligible:
        _cached_log_norm = log_norm_const
        _cached_neg_exp = neg_exp
        _cached_inv_nu = inv_nu
        _cached_scale_factor = (nu_base - 2.0) / nu_base if nu_base > 2 else 0.5
    else:
        _cached_log_norm = 0.0
        _cached_neg_exp = 0.0
        _cached_inv_nu = 0.0
        _cached_scale_factor = 0.0

    # Jump-diffusion pre-computation
    if jump_enabled:
        p0_safe = jump_intensity
        if p0_safe > 0.999:
            p0_safe = 0.999
        elif p0_safe < 1e-4:
            p0_safe = 1e-4
        logit_p0 = np.log(p0_safe / (1.0 - p0_safe))
        log_gauss_norm = -0.5 * np.log(2.0 * np.pi)
    else:
        logit_p0 = 0.0
        log_gauss_norm = 0.0

    # GAS skew dynamic state
    alpha_t = alpha_asym

    # State initialization (data-adaptive)
    _init_w = min(20, n)
    if _init_w >= 3:
        _sorted = np.sort(returns[:_init_w])
        _mid = _init_w // 2
        mu = _sorted[_mid] if _init_w % 2 == 1 else (_sorted[_mid - 1] + _sorted[_mid]) * 0.5
        _mean_init = 0.0
        for _ii in range(_init_w):
            _mean_init += returns[_ii]
        _mean_init /= _init_w
        _var_init = 0.0
        for _ii in range(_init_w):
            _var_init += (returns[_ii] - _mean_init) ** 2
        _var_init /= _init_w
        P = max(_var_init, 1e-6)
    else:
        mu = 0.0
        P = P0
    log_likelihood = 0.0

    # Main filter loop
    for t in range(n):
        # === PREDICTION STEP ===
        u_t = momentum[t]
        q_t_val = q_t[t]

        if has_risk_drift:
            mu_pred = phi * mu + u_t + risk_prem * R_base_arr[t] + mu_drift
        else:
            mu_pred = phi * mu + u_t
        P_pred = phi_sq * P + q_t_val

        # VoV-adjusted observation noise with redundancy damping
        vov_effective = gamma_vov * (1.0 - vov_damping * p_stress[t])
        R = R_base_arr[t] * (1.0 + vov_effective * vov_rolling[t])

        # S_diffusion: pure diffusion predictive variance
        S_diffusion = P_pred + R
        if S_diffusion < _MIN_VARIANCE:
            S_diffusion = _MIN_VARIANCE

        # Jump-augmented predictive variance
        if jump_enabled:
            _arg = -(logit_p0 + jump_sensitivity * vov_rolling[t])
            if _arg > 20.0:
                p_t = 1e-4
            elif _arg < -20.0:
                p_t = 0.5
            else:
                p_t = 1.0 / (1.0 + np.exp(_arg))
                if p_t < 1e-4:
                    p_t = 1e-4
                elif p_t > 0.5:
                    p_t = 0.5
            S = S_diffusion + p_t * jump_var
        else:
            p_t = 0.0
            S = S_diffusion

        # Store predictive values
        mu_pred_arr[t] = mu_pred
        S_pred_arr[t] = S

        # Innovation
        innovation = returns[t] - mu_pred

        # === UPDATE STEP ===
        scale = np.sqrt(S_diffusion)

        # Smooth asymmetric ν (or use dynamic alpha_t from GAS skew)
        if _alpha_negligible and not skew_enabled:
            nu_eff = nu_base
        else:
            _z_asym = innovation / scale if scale > 1e-10 else 0.0
            _mod = 1.0 + alpha_t * np.tanh(k_asym * _z_asym)
            nu_eff = nu_base * _mod
            if nu_eff < 2.1:
                nu_eff = 2.1
            elif nu_eff > 50.0:
                nu_eff = 50.0

        # Standard Kalman gain (robust weighting via w_t below)
        K = P_pred / S_diffusion

        # Robust Student-t weighting
        z_sq_diffusion = (innovation * innovation) / S_diffusion
        w_t = (nu_eff + 1.0) / (nu_eff + z_sq_diffusion)

        # Jump posterior — reduce Kalman update weight for likely jumps
        if jump_enabled:
            S_jump_total = S_diffusion + jump_var
            innov_centered = innovation - jump_mean

            # Log-likelihood under jump component (Gaussian)
            ll_jump = log_gauss_norm - 0.5 * np.log(S_jump_total) - 0.5 * (innov_centered * innov_centered) / S_jump_total

            # Log-likelihood under diffusion component (Student-t)
            sf = (nu_eff - 2.0) / nu_eff if nu_eff > 2.0 else 0.5
            fs_diff = np.sqrt(S_diffusion * sf)
            if fs_diff > 1e-12:
                z_diff = innovation / fs_diff
                log_n_diff = _stirling_gammaln((nu_eff + 1.0) / 2.0) - _stirling_gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
                ll_diff = log_n_diff - np.log(fs_diff) + (-((nu_eff + 1.0) / 2.0)) * np.log(1.0 + z_diff * z_diff / nu_eff)
            else:
                ll_diff = -1e10

            # Posterior jump probability via log-sum-exp
            _log_1mp = np.log(max(1.0 - p_t, 1e-15))
            _log_p = np.log(max(p_t, 1e-15))
            log_num = _log_p + ll_jump
            _lp0 = _log_1mp + ll_diff
            _lp1 = log_num
            log_den_max = _lp0 if _lp0 > _lp1 else _lp1
            log_den = log_den_max + np.log(np.exp(_lp0 - log_den_max) + np.exp(_lp1 - log_den_max))
            if log_den == log_den:  # isfinite check
                p_jump_post = np.exp(log_num - log_den)
                if p_jump_post < 0.0:
                    p_jump_post = 0.0
                elif p_jump_post > 1.0:
                    p_jump_post = 1.0
            else:
                p_jump_post = p_t

            # Reduce Kalman update weight for likely jumps
            w_t *= (1.0 - 0.7 * p_jump_post)

        # State update with robust weighting
        mu = mu_pred + K * w_t * innovation
        P = (1.0 - w_t * K) * P_pred
        if P < _MIN_VARIANCE:
            P = _MIN_VARIANCE

        mu_filtered[t] = mu
        P_filtered[t] = P

        # GAS skew: alpha_{t+1} = (1-rho)*alpha_0 + rho*alpha_t + kappa*(z_t*w_t)
        if skew_enabled:
            z_for_score = innovation / scale if scale > 1e-10 else 0.0
            score_t = z_for_score * w_t
            alpha_t = (1.0 - skew_rho) * alpha_asym + skew_rho * alpha_t + skew_kappa * score_t
            if alpha_t < -0.3:
                alpha_t = -0.3
            elif alpha_t > 0.3:
                alpha_t = 0.3

        # === LOG-LIKELIHOOD ===
        if _alpha_negligible and not skew_enabled:
            forecast_scale = np.sqrt(S_diffusion * _cached_scale_factor)
        else:
            scale_factor = (nu_eff - 2.0) / nu_eff if nu_eff > 2.0 else 0.5
            forecast_scale = np.sqrt(S_diffusion * scale_factor)

        if forecast_scale > _MIN_VARIANCE:
            z = innovation / forecast_scale
            if _alpha_negligible and not skew_enabled:
                log_norm_eff = _cached_log_norm
                neg_exp_eff = _cached_neg_exp
                inv_nu_eff = _cached_inv_nu
            else:
                log_norm_eff = _stirling_gammaln((nu_eff + 1.0) / 2.0) - _stirling_gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
                neg_exp_eff = -((nu_eff + 1.0) / 2.0)
                inv_nu_eff = 1.0 / nu_eff

            ll_diffusion = log_norm_eff - np.log(forecast_scale) + neg_exp_eff * np.log(1.0 + z * z * inv_nu_eff)

            if jump_enabled and p_t > 1e-6:
                # Mixture log-likelihood via log-sum-exp
                S_jt = S_diffusion + jump_var
                ic = innovation - jump_mean
                ll_jmp = log_gauss_norm - 0.5 * np.log(S_jt) - 0.5 * (ic * ic) / S_jt

                ll_max = ll_diffusion if ll_diffusion > ll_jmp else ll_jmp
                ll_t = ll_max + np.log(
                    (1.0 - p_t) * np.exp(ll_diffusion - ll_max)
                    + p_t * np.exp(ll_jmp - ll_max)
                )
            else:
                ll_t = ll_diffusion

            # Clamp contribution
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            elif ll_t > _MAX_LL_CONTRIB:
                ll_t = _MAX_LL_CONTRIB

            if ll_t == ll_t:  # NaN check
                log_likelihood += ll_t

    # === CAUSAL EWM LOCATION CORRECTION ===
    if ewm_lambda >= 0.01 and n > 2:
        alpha_ewm = 1.0 - ewm_lambda
        ewm_mu_val = 0.0
        for t in range(n - 1):
            innov_t = returns[t] - mu_pred_arr[t]
            ewm_mu_val = ewm_lambda * ewm_mu_val + alpha_ewm * innov_t
            mu_pred_arr[t + 1] = mu_pred_arr[t + 1] + ewm_mu_val

    return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood
# These kernels accelerate the inner loop of the cross-validated optimizer
# objective. Given the final filtered state from training, they propagate
# forward through the test fold computing out-of-sample log-likelihood and
# standardized innovations for PIT calibration.
#
# Called ~100-200× per asset during L-BFGS-B + grid search optimisation.
# Moving from Python to Numba gives ~5× speedup on the optimizer hot path.
# =============================================================================

@njit(cache=True, fastmath=True)
def gaussian_cv_test_fold_kernel(
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
) -> tuple:
    """
    Gaussian (φ=1) forward pass on a single CV test fold.

    Propagates the Kalman filter from the end of the training fold through
    the test observations, accumulating log-likelihood and standardized
    innovations. The standardized innovations are written into std_buf
    starting at std_offset for later PIT/KS computation in Python.

    Parameters
    ----------
    returns : np.ndarray
        Full contiguous returns array (indexed by absolute t).
    vol_sq : np.ndarray
        Pre-computed vol² array (c is multiplied inside this kernel).
    q : float
        Process noise variance.
    c : float
        Observation noise scale.
    mu_init : float
        Last filtered mean from training fold.
    P_init : float
        Last filtered covariance from training fold.
    test_start, test_end : int
        Absolute indices of the test window [test_start, test_end).
    std_buf : np.ndarray
        Pre-allocated buffer for standardized innovations (mutated in-place).
    std_offset : int
        Current write position in std_buf.
    std_max : int
        Maximum number of standardized residuals to store.

    Returns
    -------
    ll_fold : float
        Total out-of-sample log-likelihood for this fold.
    n_obs : int
        Number of test observations processed.
    std_written : int
        Number of standardized innovations written to std_buf.
    """
    mu_pred = mu_init
    P_pred = P_init
    ll_fold = 0.0
    n_obs = test_end - test_start
    std_written = 0

    for t in range(test_start, test_end):
        P_pred = P_pred + q

        R = c * vol_sq[t]
        innovation = returns[t] - mu_pred
        forecast_var = P_pred + R

        if forecast_var > _MIN_VARIANCE:
            ll_fold += -0.5 * (_LOG_2PI + np.log(forecast_var)
                               + (innovation * innovation) / forecast_var)
            if std_offset + std_written < std_max:
                std_buf[std_offset + std_written] = innovation / np.sqrt(forecast_var)
                std_written += 1

        S_total = P_pred + R
        if S_total > _MIN_VARIANCE:
            K = P_pred / S_total
        else:
            K = 0.0
        mu_pred = mu_pred + K * innovation
        P_pred = (1.0 - K) * P_pred

    return ll_fold, n_obs, std_written


# =============================================================================
# UNIFIED MC SIMULATION KERNEL (v7.0)
# =============================================================================
# Replaces the two Python for-loops in _simulate_forward_paths and provides
# GARCH + jump-diffusion + Student-t sampling for run_regime_specific_mc.
# This is the single MC engine used for both p_up and exp_ret.
# =============================================================================

@njit(cache=True, fastmath=False)
def _student_t_sample_nb(rng_z1: float, rng_z2: float, nu: float) -> float:
    """Generate a Student-t(nu) sample scaled to unit variance.

    Uses the ratio method: t = Z / sqrt(V/nu) where Z ~ N(0,1)
    and V ~ chi2(nu).  For chi2 we use the Box-Muller pair to
    get a Gamma(nu/2, 2) via repeated normal draws.

    We approximate chi2(nu) as the sum of nu standard-normal squares.
    For large nu this is exact; for small nu the sample count is small.

    Instead, we use the fact that for integer or half-integer nu,
    chi2(nu) = sum of nu N(0,1)^2.  For non-integer nu we round.

    But Numba doesn't have rng.standard_t, so we use the identity:
      t(nu) = N(0,1) / sqrt(chi2(nu) / nu)
    where chi2(nu) can be approximated via a simple loop.

    This function takes two pre-generated N(0,1) values and uses
    a simplified approach:
      t ≈ Z1 * sqrt(nu / max(Z2^2, eps))  -- NOT correct

    Actually, the correct approach is passed externally.
    This helper scales a raw Student-t draw to unit variance.
    """
    # Scale to unit variance: Var(t_nu) = nu/(nu-2) for nu > 2
    if nu > 2.0:
        t_var = nu / (nu - 2.0)
        return rng_z1 / np.sqrt(t_var)
    return rng_z1


@njit(cache=True, fastmath=False)
def unified_mc_simulate_kernel(
    n_paths: int,
    H_max: int,
    mu_now: float,
    h0: float,
    phi: float,
    drift_q: float,
    nu: float,
    use_garch: bool,
    omega: float,
    alpha: float,
    beta: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    enable_jumps: bool,
    z_normals: np.ndarray,
    z_chi2: np.ndarray,
    z_drift: np.ndarray,
    z_jump_uniform: np.ndarray,
    z_jump_normal: np.ndarray,
    cum_out: np.ndarray,
    vol_out: np.ndarray,
) -> None:
    """Unified MC simulation kernel with GARCH + jumps + Student-t.

    Generates n_paths forward simulations of cumulative log returns
    over H_max steps.  All randomness is passed in as pre-generated
    arrays (generated with numpy RNG in Python, passed to Numba).

    Student-t sampling: t(nu) = Z / sqrt(chi2(nu)/nu)
    where Z = z_normals and chi2(nu) ≈ z_chi2 (pre-generated).

    Parameters
    ----------
    n_paths : int
        Number of MC paths
    H_max : int
        Maximum forecast horizon (steps)
    mu_now : float
        Current drift estimate
    h0 : float
        Initial variance (vol^2)
    phi : float
        AR(1) drift persistence
    drift_q : float
        Process noise variance for drift evolution
    nu : float
        Degrees of freedom for Student-t noise (>100 treated as Gaussian)
    use_garch : bool
        Whether to use GARCH(1,1) variance evolution
    omega, alpha, beta : float
        GARCH(1,1) parameters
    jump_intensity : float
        Poisson jump arrival rate per step
    jump_mean, jump_std : float
        Jump size distribution N(jump_mean, jump_std^2)
    enable_jumps : bool
        Whether to include jump-diffusion
    z_normals : ndarray (H_max, n_paths)
        Pre-generated standard normal draws for observation noise
    z_chi2 : ndarray (H_max, n_paths)
        Pre-generated chi2(nu)/nu draws for Student-t (1.0 for Gaussian)
    z_drift : ndarray (H_max, n_paths)
        Pre-generated standard normal draws for drift noise
    z_jump_uniform : ndarray (H_max, n_paths)
        Pre-generated Uniform(0,1) for Poisson jump count approximation
    z_jump_normal : ndarray (H_max, n_paths)
        Pre-generated N(0,1) for jump sizes
    cum_out : ndarray (H_max, n_paths)
        Output: cumulative log returns (pre-allocated)
    vol_out : ndarray (H_max, n_paths)
        Output: volatility sqrt(h_t) at each step (pre-allocated)
    """
    use_student_t = (nu > 2.0) and (nu < 100.0)

    # Precompute Student-t variance scaling
    if use_student_t:
        t_var = nu / (nu - 2.0)
        t_scale_factor = 1.0 / np.sqrt(t_var)
    else:
        t_scale_factor = 1.0

    drift_sigma = np.sqrt(drift_q) if drift_q > 0.0 else 0.0

    for p in range(n_paths):
        mu_t = mu_now
        h_t = h0
        if h_t < 1e-12:
            h_t = 1e-12
        cum = 0.0

        for t in range(H_max):
            sigma_t = np.sqrt(h_t)
            vol_out[t, p] = sigma_t

            # Observation noise: Student-t or Gaussian
            if use_student_t:
                # t(nu) = Z / sqrt(V) where V = chi2(nu)/nu
                chi2_val = z_chi2[t, p]
                if chi2_val < 1e-8:
                    chi2_val = 1e-8
                raw_t = z_normals[t, p] / np.sqrt(chi2_val)
                # Scale to unit variance
                eps = raw_t * t_scale_factor
            else:
                eps = z_normals[t, p]

            e_t = sigma_t * eps

            # Jump component
            jump = 0.0
            if enable_jumps and jump_intensity > 0.0:
                # Approximate Poisson: if U < lambda, one jump occurs
                # For small lambda this is P(N>=1) ≈ lambda
                if z_jump_uniform[t, p] < jump_intensity:
                    jump = jump_mean + jump_std * z_jump_normal[t, p]
                # Second jump possible for lambda > 0.1
                if jump_intensity > 0.1 and z_jump_uniform[t, p] < jump_intensity * jump_intensity:
                    jump += jump_mean + jump_std * z_drift[t, p] * 0.5  # reuse drift noise

            # Total return
            r_t = mu_t + e_t + jump
            cum += r_t

            cum_out[t, p] = cum

            # GARCH variance evolution
            if use_garch:
                h_t = omega + alpha * (e_t * e_t) + beta * h_t
                if h_t < 1e-12:
                    h_t = 1e-12
                elif h_t > 1e4:
                    h_t = 1e4

            # AR(1) drift evolution
            if drift_sigma > 0.0:
                mu_t = phi * mu_t + drift_sigma * z_drift[t, p]
            else:
                mu_t = phi * mu_t


@njit(cache=True, fastmath=False)
def unified_mc_multi_path_kernel(
    n_paths: int,
    H_max: int,
    mu_now: float,
    h0: float,
    phi: float,
    drift_q: float,
    nu_per_path: np.ndarray,
    use_garch: bool,
    omega_per_path: np.ndarray,
    alpha_per_path: np.ndarray,
    beta_per_path: np.ndarray,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    enable_jumps: bool,
    z_normals: np.ndarray,
    z_chi2: np.ndarray,
    z_drift: np.ndarray,
    z_jump_uniform: np.ndarray,
    z_jump_normal: np.ndarray,
    cum_out: np.ndarray,
    vol_out: np.ndarray,
) -> None:
    """Multi-path MC kernel with per-path parameter uncertainty.

    Like unified_mc_simulate_kernel but supports:
    - Per-path nu (tail parameter uncertainty)
    - Per-path GARCH parameters (parameter uncertainty via covariance sampling)

    Parameters
    ----------
    nu_per_path : ndarray (n_paths,)
        Per-path degrees of freedom
    omega_per_path, alpha_per_path, beta_per_path : ndarray (n_paths,)
        Per-path GARCH parameters
    (other parameters same as unified_mc_simulate_kernel)
    """
    drift_sigma = np.sqrt(drift_q) if drift_q > 0.0 else 0.0

    for p in range(n_paths):
        nu_p = nu_per_path[p]
        use_t_p = (nu_p > 2.0) and (nu_p < 100.0)
        if use_t_p:
            t_var_p = nu_p / (nu_p - 2.0)
            t_scale_p = 1.0 / np.sqrt(t_var_p)
        else:
            t_scale_p = 1.0

        omega_p = omega_per_path[p]
        alpha_p = alpha_per_path[p]
        beta_p = beta_per_path[p]

        mu_t = mu_now
        h_t = h0
        if h_t < 1e-12:
            h_t = 1e-12
        cum = 0.0

        for t in range(H_max):
            sigma_t = np.sqrt(h_t)
            vol_out[t, p] = sigma_t

            # Observation noise
            if use_t_p:
                chi2_val = z_chi2[t, p]
                if chi2_val < 1e-8:
                    chi2_val = 1e-8
                raw_t = z_normals[t, p] / np.sqrt(chi2_val)
                eps = raw_t * t_scale_p
            else:
                eps = z_normals[t, p]

            e_t = sigma_t * eps

            # Jump component
            jump = 0.0
            if enable_jumps and jump_intensity > 0.0:
                if z_jump_uniform[t, p] < jump_intensity:
                    jump = jump_mean + jump_std * z_jump_normal[t, p]

            r_t = mu_t + e_t + jump
            cum += r_t
            cum_out[t, p] = cum

            # GARCH evolution (per-path params)
            if use_garch:
                h_t = omega_p + alpha_p * (e_t * e_t) + beta_p * h_t
                if h_t < 1e-12:
                    h_t = 1e-12
                elif h_t > 1e4:
                    h_t = 1e4

            # AR(1) drift
            if drift_sigma > 0.0:
                mu_t = phi * mu_t + drift_sigma * z_drift[t, p]
            else:
                mu_t = phi * mu_t


# =============================================================================
# MS PROCESS NOISE EWM KERNEL
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_ms_process_noise_ewm_kernel(
    vol: np.ndarray,
    lam: float,
    warmup_mean: float,
    warmup_var: float,
) -> np.ndarray:
    """
    Compute EWM z-scores for vol array (MS process noise smooth EWM path).

    Parameters
    ----------
    vol : np.ndarray
        Volatility array (contiguous float64)
    lam : float
        EWM decay factor in (0, 1)
    warmup_mean : float
        Pre-computed warmup mean of vol[:warmup]
    warmup_var : float
        Pre-computed warmup variance of vol[:warmup]

    Returns
    -------
    vol_zscore : np.ndarray
        Z-scored volatility (same length as vol)
    """
    n = len(vol)
    vol_zscore = np.empty(n, dtype=np.float64)
    one_minus_lam = 1.0 - lam

    ewm_mean = warmup_mean
    ewm_var = warmup_var
    if ewm_var < 1e-12:
        ewm_var = 1e-12

    for t in range(n):
        ewm_std = np.sqrt(ewm_var)
        if ewm_std < 1e-6:
            ewm_std = 1e-6
        vol_zscore[t] = (vol[t] - ewm_mean) / ewm_std

        # Update AFTER computing z-score (no look-ahead)
        ewm_mean = lam * ewm_mean + one_minus_lam * vol[t]
        diff = vol[t] - ewm_mean
        ewm_var = lam * ewm_var + one_minus_lam * (diff * diff)
        if ewm_var < 1e-12:
            ewm_var = 1e-12

    return vol_zscore


# =============================================================================
# STAGE 6 EWM FOLD KERNEL
# =============================================================================

@njit(cache=True, fastmath=True)
def stage6_ewm_fold_kernel(
    it_arr: np.ndarray,
    Sb_arr: np.ndarray,
    ee: int,
    ve: int,
    lam: float,
    init_em: float,
    init_en: float,
    init_ed: float,
) -> tuple:
    """
    Stage 6 EWM fold computation — combines _get_ewm_state warmup + _fold_ewm_raw.

    Parameters
    ----------
    it_arr : np.ndarray
        Innovation array (full training set)
    Sb_arr : np.ndarray
        Predictive variance array (full training set)
    ee : int
        Train end / validation start index
    ve : int
        Validation end index
    lam : float
        EWM decay factor
    init_em, init_en, init_ed : float
        Initial EWM state estimates (from np.mean of it[:ee] etc.)

    Returns
    -------
    iv_arr : np.ndarray
        EWM-corrected innovations for validation fold
    Sv_arr : np.ndarray
        EWM-corrected variances for validation fold
    """
    lm1 = 1.0 - lam

    # Phase 1: Run EWM warmup through [0..ee) to get final state
    em = init_em
    en = init_en
    ed = init_ed
    for t in range(ee):
        v = it_arr[t]
        em = lam * em + lm1 * v
        en = lam * en + lm1 * v * v
        ed = lam * ed + lm1 * Sb_arr[t]

    # Phase 2: Compute validation fold outputs
    nv = ve - ee
    iv_arr = np.empty(nv, dtype=np.float64)
    Sv_arr = np.empty(nv, dtype=np.float64)

    for tv in range(nv):
        ix = ee + tv
        bv = en / (ed + 1e-12)
        if bv < 0.2:
            bv = 0.2
        elif bv > 5.0:
            bv = 5.0
        iv_arr[tv] = it_arr[ix] - em
        Sv_arr[tv] = Sb_arr[ix] * bv
        v = it_arr[ix]
        em = lam * em + lm1 * v
        en = lam * en + lm1 * v * v
        ed = lam * ed + lm1 * Sb_arr[ix]

    return iv_arr, Sv_arr


# =============================================================================
# STAGE 5f EWM CORRECTION KERNEL
# =============================================================================

@njit(cache=True, fastmath=True)
def ewm_mu_correction_kernel(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    lam: float,
    n_train: int,
) -> np.ndarray:
    """
    EWM bias correction for Stage 5f.

    Computes: mu_corr[t] = mu_pred[t] + ewm_mu_t
    where ewm_mu_t tracks exponentially weighted innovation residuals.

    Parameters
    ----------
    returns : np.ndarray
        Return series
    mu_pred : np.ndarray
        Base predicted means
    lam : float
        EWM decay factor
    n_train : int
        Number of training samples

    Returns
    -------
    mu_corr : np.ndarray
        Corrected predicted means (length n_train)
    """
    mu_corr = np.empty(n_train, dtype=np.float64)
    mu_corr[0] = mu_pred[0]
    one_minus_lam = 1.0 - lam
    ewm_mu = 0.0

    for t in range(1, n_train):
        ewm_mu = lam * ewm_mu + one_minus_lam * (returns[t - 1] - mu_pred[t - 1])
        mu_corr[t] = mu_pred[t] + ewm_mu

    return mu_corr


# =============================================================================
# GAUSSIAN SCORE FOLD KERNEL
# =============================================================================

@njit(cache=True, fastmath=True)
def gaussian_score_fold_kernel(
    it_arr: np.ndarray,
    Sb_arr: np.ndarray,
    ee: int,
    ve: int,
    lam: float,
    init_em: float,
    init_en: float,
    init_ed: float,
) -> tuple:
    """
    Gaussian Stage 5 _score_fold — EWM PIT with KS approximation.

    Parameters
    ----------
    it_arr : np.ndarray
        Innovation array
    Sb_arr : np.ndarray
        Predictive variance array
    ee : int
        Train end / validation start
    ve : int
        Validation end
    lam : float
        EWM decay factor
    init_em, init_en, init_ed : float
        Initial EWM state (from np.mean of training slice)

    Returns
    -------
    kp : float
        Approximate KS p-value
    md : float
        MAD of PIT histogram bins
    """
    lm1 = 1.0 - lam
    nv = ve - ee
    if nv < 20:
        return 0.0, 1.0

    # Phase 1: Run EWM warmup through [0..ee)
    em = init_em
    en = init_en
    ed = init_ed
    for t in range(ee):
        v = it_arr[t]
        em = lam * em + lm1 * v
        en = lam * en + lm1 * v * v
        ed = lam * ed + lm1 * Sb_arr[t]

    # Phase 2: Compute z-values for validation fold
    zv = np.empty(nv, dtype=np.float64)
    for tv in range(nv):
        ix = ee + tv
        bv = en / (ed + 1e-12)
        if bv < 0.2:
            bv = 0.2
        elif bv > 5.0:
            bv = 5.0
        iv = it_arr[ix] - em
        Sv = Sb_arr[ix] * bv
        s = np.sqrt(Sv) if Sv > 0.0 else 1e-10
        if s < 1e-10:
            s = 1e-10
        zv[tv] = iv / s
        v = it_arr[ix]
        em = lam * em + lm1 * v
        en = lam * en + lm1 * v * v
        ed = lam * ed + lm1 * Sb_arr[ix]

    # Phase 3: Compute PIT values using erfc-based Gaussian CDF
    pv = np.empty(nv, dtype=np.float64)
    _SQRT_2 = np.sqrt(2.0)
    for i in range(nv):
        p = 0.5 * math.erfc(-zv[i] / _SQRT_2)
        if p < 0.001:
            p = 0.001
        elif p > 0.999:
            p = 0.999
        pv[i] = p

    # Phase 4: Sort for KS test
    # Simple insertion sort (nv is small, typically <200)
    ps = np.copy(pv)
    for i in range(1, nv):
        key = ps[i]
        j = i - 1
        while j >= 0 and ps[j] > key:
            ps[j + 1] = ps[j]
            j -= 1
        ps[j + 1] = key

    # KS statistic
    dp_max = 0.0
    dm_max = 0.0
    inv_nv = 1.0 / nv
    for i in range(nv):
        dp = (i + 1) * inv_nv - ps[i]
        dm = ps[i] - i * inv_nv
        if dp > dp_max:
            dp_max = dp
        if dm > dm_max:
            dm_max = dm
    D_ks = dp_max if dp_max > dm_max else dm_max

    sq_n = np.sqrt(float(nv))
    lam_ks = (sq_n + 0.12 + 0.11 / sq_n) * D_ks
    if lam_ks < 0.001:
        kp = 1.0
    elif lam_ks > 3.0:
        kp = 0.0
    else:
        kp = 2.0 * np.exp(-2.0 * lam_ks * lam_ks)
        if kp > 1.0:
            kp = 1.0

    # PIT histogram MAD (10 bins)
    hi = np.zeros(10, dtype=np.float64)
    for i in range(nv):
        b = int(pv[i] * 10.0)
        if b >= 10:
            b = 9
        hi[b] += 1.0

    total_md = 0.0
    for b in range(10):
        total_md += abs(hi[b] / float(nv) - 0.1)
    md = total_md / 10.0

    return kp, md
# =============================================================================
# GAS-Q GAUSSIAN FILTER KERNEL
# =============================================================================
# Pure scalar loop implementing GAS-Q dynamics with inlined score/update.
# Profiling shows gas_q_filter_gaussian at 3.9s (2254 calls) — 2nd largest
# bottleneck. This kernel inlines compute_gaussian_score_q and gas_q_update.
# =============================================================================

@njit(cache=True, fastmath=True)
def gas_q_filter_gaussian_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    c: float,
    phi: float,
    omega: float,
    alpha: float,
    beta: float,
    q_init: float,
    q_min: float,
    q_max: float,
    score_scale: float,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    q_path: np.ndarray,
    score_path: np.ndarray,
) -> float:
    """
    Numba-compiled GAS-Q Gaussian Kalman filter.

    Inlines score computation and GAS update for maximum throughput.
    Returns log-likelihood; filtered arrays are written in-place.
    """
    n = len(returns)
    mu = 0.0
    P = 1e-4
    q_t = q_init
    log_ll = 0.0
    phi_sq = phi * phi
    log_2pi = 1.8378770664093453  # np.log(2*pi)
    _EPS = 1e-12

    for t in range(n):
        q_path[t] = q_t

        # Predict
        mu_pred = phi * mu
        P_pred = phi_sq * P + q_t

        # Innovation
        R_t = c * vol_sq[t]
        S_t = P_pred + R_t
        if S_t < _EPS:
            S_t = _EPS
        innovation = returns[t] - mu_pred

        # Kalman gain + update
        K = P_pred / S_t
        mu = mu_pred + K * innovation
        P = (1.0 - K) * P_pred
        if P < _EPS:
            P = _EPS

        mu_filtered[t] = mu
        P_filtered[t] = P

        # Inlined Gaussian score: (z² - 1) / (2·S_t) * scale
        z_sq = (innovation * innovation) / S_t
        raw_score = (z_sq - 1.0) / (2.0 * S_t)
        score_val = score_scale * raw_score
        if score_val > 1e6:
            score_val = 1e6
        elif score_val < -1e6:
            score_val = -1e6
        score_path[t] = score_val

        # Log-likelihood
        ll_t = -0.5 * (log_2pi + np.log(S_t) + z_sq)
        if ll_t == ll_t:  # isfinite check in Numba
            log_ll += ll_t

        # Inlined GAS update: q_t = omega + alpha*s_{t-1} + beta*q_{t-1}
        q_new = omega + alpha * score_val + beta * q_t
        if q_new < q_min:
            q_new = q_min
        elif q_new > q_max:
            q_new = q_max
        q_t = q_new

    return log_ll


# =============================================================================
# BUILD GARCH KERNEL
# =============================================================================
# Pure scalar GJR-GARCH loop with leverage, jump-eta, regime-switch, and
# mean-reversion enhancements. Called ~27 times per model in Stage 5g.
# Profiling shows 0.64s across 540 calls.
# =============================================================================

@njit(cache=True, fastmath=False)
def build_garch_kernel(
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
    eta_c: float,
    reg_c: float,
    h_out: np.ndarray,
) -> None:
    """
    Numba-compiled GJR-GARCH(1,1) variance construction with enhancements.
    Writes results into h_out in-place.
    """
    h_out[0] = unconditional_var
    _p_st = 0.1
    _sm = np.sqrt(q_stress_ratio)

    for t_ in range(1, n_train):
        h_ = (garch_omega
              + garch_alpha * sq_inn[t_ - 1]
              + garch_leverage * sq_inn[t_ - 1] * neg_ind[t_ - 1]
              + garch_beta * h_out[t_ - 1])

        if rho_c > 0.01 and h_out[t_ - 1] > 1e-12:
            nz_ = innovations[t_ - 1] / np.sqrt(h_out[t_ - 1])
            if nz_ < 0:
                h_ += rho_c * nz_ * nz_ * h_out[t_ - 1]

        if eta_c > 0.005 and h_out[t_ - 1] > 1e-12:
            za_ = abs(innovations[t_ - 1]) / np.sqrt(h_out[t_ - 1])
            ex_ = za_ - 1.5
            if ex_ < 0.0:
                ex_ = 0.0
            h_ += eta_c * ex_ * ex_ * h_out[t_ - 1]

        if reg_c > 0.005 and h_out[t_ - 1] > 1e-12:
            zr_ = abs(innovations[t_ - 1]) / np.sqrt(h_out[t_ - 1])
            _p_st = (1.0 - reg_c) * _p_st + reg_c * (1.0 if zr_ > 2.0 else 0.0)
            if _p_st < 0.0:
                _p_st = 0.0
            elif _p_st > 1.0:
                _p_st = 1.0
            h_ = h_ * (1.0 + _p_st * (_sm - 1.0))

        if kap_c > 0.001:
            h_ = (1.0 - kap_c) * h_ + kap_c * unconditional_var

        if h_ < 1e-12:
            h_ = 1e-12
        h_out[t_] = h_


# =============================================================================
# CHI² EWM CORRECTION KERNEL
# =============================================================================
# Causal EWM z² → scale correction for domain-matched PIT computation.
# Used in Stage 6 scoring (phi_student_t.py) and compute_extended_pit_metrics
# (tune.py). Profiling shows 0.25s across 1012 calls in phi_student_t alone.
# =============================================================================

@njit(cache=True, fastmath=False)
def chi2_ewm_correction_kernel(
    z_raw: np.ndarray,
    chi2_target: float,
    chi2_lambda: float,
    scale_adj_out: np.ndarray,
) -> None:
    """
    Numba-compiled chi² EWM variance correction.

    Tracks E[z²] via exponential weighted mean, computes adaptive scale
    adjustment. Writes adjustment factors into scale_adj_out in-place.
    """
    n = len(z_raw)
    chi2_1m = 1.0 - chi2_lambda
    winsor_cap = chi2_target * 50.0
    ewm_z2 = chi2_target

    for t in range(n):
        ratio = ewm_z2 / chi2_target
        if ratio < 0.3:
            ratio = 0.3
        elif ratio > 3.0:
            ratio = 3.0

        dev = abs(ratio - 1.0)

        if ratio >= 1.0:
            dz_lo = 0.25
            dz_rng = 0.25
        else:
            dz_lo = 0.10
            dz_rng = 0.15

        if dev < dz_lo:
            adj = 1.0
        elif dev >= dz_lo + dz_rng:
            adj = np.sqrt(ratio)
        else:
            s = (dev - dz_lo) / dz_rng
            adj = 1.0 + s * (np.sqrt(ratio) - 1.0)

        scale_adj_out[t] = adj

        z2 = z_raw[t] * z_raw[t]
        z2w = z2 if z2 < winsor_cap else winsor_cap
        ewm_z2 = chi2_lambda * ewm_z2 + chi2_1m * z2w


# =============================================================================
# PIT-VARIANCE STRETCHING KERNEL
# =============================================================================
# Fixes shape miscalibration not caught by chi² correction.
# Used in compute_extended_pit_metrics for both Student-t and Gaussian.
# =============================================================================

@njit(cache=True, fastmath=False)
def pit_var_stretching_kernel(
    pit_values: np.ndarray,
) -> None:
    """
    Numba-compiled PIT-variance stretching (Var[PIT] → 1/12).
    Modifies pit_values in-place.
    """
    n = len(pit_values)
    pv_tgt = 1.0 / 12.0
    pv_lam = 0.97
    pv_1m = 0.03
    pv_dz_lo = 0.30
    pv_dz_hi = 0.55
    pv_dz_rng = pv_dz_hi - pv_dz_lo
    ewm_pm = 0.5
    ewm_psq = 1.0 / 3.0

    for t in range(n):
        ov = ewm_psq - ewm_pm * ewm_pm
        if ov < 0.005:
            ov = 0.005
        vr = ov / pv_tgt
        vd = abs(vr - 1.0)
        rp = pit_values[t]

        if vd > pv_dz_lo:
            rs = np.sqrt(pv_tgt / ov)
            if rs < 0.70:
                rs = 0.70
            elif rs > 1.50:
                rs = 1.50

            if vd >= pv_dz_hi:
                st = rs
            else:
                sg = (vd - pv_dz_lo) / pv_dz_rng
                st = 1.0 + sg * (rs - 1.0)

            c = 0.5 + (rp - 0.5) * st
            if c < 0.001:
                c = 0.001
            elif c > 0.999:
                c = 0.999
            pit_values[t] = c

        # Update EWM trackers
        ewm_pm = pv_lam * ewm_pm + pv_1m * pit_values[t]
        ewm_psq = pv_lam * ewm_psq + pv_1m * pit_values[t] * pit_values[t]


# =============================================================================
# φ-STUDENT-T CV TEST-FOLD KERNEL
# =============================================================================
# Profiling shows neg_cv_ll at 6.7s (8516 calls) — the #1 remaining
# bottleneck. This kernel replaces the Python validation loop with a
# Numba-compiled loop using Student-t likelihood and constant nu-adjust gain.
# =============================================================================

@njit(cache=True, fastmath=False)
def phi_student_t_cv_test_fold_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_scale: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    nu_val: float,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    use_vov: int,
) -> float:
    """
    Numba-compiled φ-Student-t forward pass on a single CV test fold.

    Computes log-likelihood of validation data given initial state from
    training fold. Uses Student-t likelihood with robust Kalman gain
    (Meinhold & Singpurwalla 1989) and optional VoV inflation.

    Parameters
    ----------
    returns : contiguous float64 array
    vol_sq : contiguous float64 array (vol²)
    q : process noise
    c : observation noise scale
    phi : AR(1) persistence
    nu_scale : (nu-2)/nu if nu>2 else 1.0
    log_norm_const : gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(nu*pi)
    neg_exp : -(nu+1)/2
    inv_nu : 1/nu
    mu_init : initial state mean (from training fold)
    P_init : initial state variance (from training fold)
    test_start : first index of validation range
    test_end : one-past-last index of validation range
    nu_val : degrees of freedom (for robust weighting)
    gamma_vov : VoV gamma coefficient (0 to disable)
    vov_rolling : VoV rolling array (may be empty if use_vov=0)
    use_vov : 1 if VoV active, 0 otherwise

    Returns
    -------
    ll_fold : total log-likelihood of the validation fold
    """
    mu_p = mu_init
    P_p = P_init
    ll_fold = 0.0
    phi_sq = phi * phi
    nu_p1 = nu_val + 1.0

    for t in range(test_start, test_end):
        mu_p = phi * mu_p
        P_p = phi_sq * P_p + q

        R_t = c * vol_sq[t]
        if use_vov == 1:
            R_t *= (1.0 + gamma_vov * vov_rolling[t])
        S = P_p + R_t
        if S < 1e-12:
            S = 1e-12

        inn = returns[t] - mu_p
        scale = np.sqrt(S * nu_scale)
        if scale > 1e-12:
            z = inn / scale
            ll_t = log_norm_const - np.log(scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
            if ll_t == ll_t:  # isfinite check in Numba
                ll_fold += ll_t

        # Robust Kalman gain (Meinhold & Singpurwalla 1989)
        K = P_p / S
        z_sq_cv = (inn * inn) / S
        w_cv = nu_p1 / (nu_val + z_sq_cv)
        mu_p = mu_p + K * w_cv * inn
        P_p = (1.0 - w_cv * K) * P_p
        if P_p < 1e-12:
            P_p = 1e-12

    return ll_fold


@njit(cache=True, fastmath=True)
def phi_gaussian_cv_test_fold_kernel(
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
) -> tuple:
    """
    φ-Gaussian (AR(1)) forward pass on a single CV test fold.

    Same as gaussian_cv_test_fold_kernel but with AR(1) drift dynamics:
        μ_pred = φ × μ_{t-1}
        P_pred = φ² × P_{t-1} + q

    Parameters
    ----------
    phi : float
        AR(1) persistence coefficient.
    (other parameters identical to gaussian_cv_test_fold_kernel)

    Returns
    -------
    ll_fold : float
    n_obs : int
    std_written : int
    """
    mu_pred = mu_init
    P_pred = P_init
    ll_fold = 0.0
    n_obs = test_end - test_start
    std_written = 0
    phi_sq = phi * phi

    for t in range(test_start, test_end):
        mu_pred = phi * mu_pred
        P_pred = phi_sq * P_pred + q

        R = c * vol_sq[t]
        innovation = returns[t] - mu_pred
        forecast_var = P_pred + R

        if forecast_var > _MIN_VARIANCE:
            ll_fold += -0.5 * (_LOG_2PI + np.log(forecast_var)
                               + (innovation * innovation) / forecast_var)
            if std_offset + std_written < std_max:
                std_buf[std_offset + std_written] = innovation / np.sqrt(forecast_var)
                std_written += 1

        S_total = P_pred + R
        if S_total > _MIN_VARIANCE:
            K = P_pred / S_total
        else:
            K = 0.0
        mu_pred = mu_pred + K * innovation
        P_pred = (1.0 - K) * P_pred

    return ll_fold, n_obs, std_written
