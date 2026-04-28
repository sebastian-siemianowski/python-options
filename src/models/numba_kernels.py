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
    
    # Compute gamma values using Lanczos approximation (1e-12 precision)
    log_gamma_half_nu = _lanczos_gammaln(nu / 2.0)
    log_gamma_half_nu_plus_half = _lanczos_gammaln((nu + 1.0) / 2.0)
    
    # Student-t log-pdf
    log_norm = (log_gamma_half_nu_plus_half - log_gamma_half_nu 
                - 0.5 * np.log(nu * np.pi) - np.log(scale))
    log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + z_sq / nu)
    
    return log_norm + log_kernel


# =============================================================================
# HANSEN SKEW-T KERNELS — Scalar (March 2026)
# =============================================================================
# Hansen (1994) piecewise asymmetric Student-t distribution.
# Integrated into Kalman filter observation likelihood for skewness-aware
# state estimation. All constants (a, b, c_const) are precomputed outside
# the filter loop for efficiency.
#
# f(z|ν,λ) = bc × [1 + (bz+a)²/((1±λ)²(ν-2))]^{-(ν+1)/2} / (1±λ)
# where ± depends on z < -a/b (left) vs z ≥ -a/b (right).
# =============================================================================

@njit(cache=True, fastmath=False)
def hansen_constants_kernel(nu: float, lambda_: float) -> tuple:
    """
    Compute Hansen's skew-t constants (a, b, c_const).
    
    Parameters
    ----------
    nu : float
        Degrees of freedom (> 2)
    lambda_ : float
        Skewness parameter ∈ (-1, 1)
        
    Returns
    -------
    a, b, c_const : float
    """
    if nu <= 2.0:
        nu = 2.01
    if lambda_ > 0.999:
        lambda_ = 0.999
    elif lambda_ < -0.999:
        lambda_ = -0.999
    
    # c = Γ((ν+1)/2) / [√(π(ν-2)) Γ(ν/2)]
    log_c = (_lanczos_gammaln((nu + 1.0) / 2.0)
             - _lanczos_gammaln(nu / 2.0)
             - 0.5 * np.log(np.pi * (nu - 2.0)))
    c_const = np.exp(log_c)
    
    # a = 4λc(ν-2)/(ν-1)
    a = 4.0 * lambda_ * c_const * ((nu - 2.0) / (nu - 1.0))
    
    # b = √(1 + 3λ² - a²)
    b_sq = 1.0 + 3.0 * lambda_ * lambda_ - a * a
    if b_sq <= 0.0:
        b_sq = 1e-10
    b = np.sqrt(b_sq)
    
    return a, b, c_const


@njit(cache=True, fastmath=False)
def hansen_skew_t_logpdf_scalar(
    x: float,
    nu: float,
    lambda_: float,
    a: float,
    b: float,
    c_const: float,
    mu: float,
    scale: float,
) -> float:
    """
    Hansen's skew-t scalar log-PDF for use inside Kalman filter loops.
    
    Parameters
    ----------
    x : float
        Observation value
    nu : float
        Degrees of freedom (> 2)
    lambda_ : float
        Skewness parameter ∈ (-1, 1)
    a, b, c_const : float
        Precomputed Hansen constants from hansen_constants_kernel()
    mu : float
        Location parameter
    scale : float
        Scale parameter (forecast_scale from Kalman filter)
        
    Returns
    -------
    float
        Log-probability density
    """
    if scale <= _MIN_VARIANCE or nu <= 2.0:
        return -1e12
    
    # Standardize
    z = (x - mu) / scale
    
    # Cutpoint
    cutpoint = -a / b
    neg_half_nu_plus_1 = -(nu + 1.0) / 2.0
    inv_nu_minus_2 = 1.0 / (nu - 2.0)
    
    log_b = np.log(b)
    log_c = np.log(c_const)
    
    if z < cutpoint:
        # Left region: scale (1-lambda). When lambda<0 (left-skew), (1-lambda)>1 → heavier left tail
        z_eff = (b * z + a) / (1.0 - lambda_)
        log_kernel = neg_half_nu_plus_1 * np.log(1.0 + z_eff * z_eff * inv_nu_minus_2)
        ll = log_b + log_c + log_kernel - np.log(scale)
    else:
        # Right region: scale (1+lambda). When lambda>0 (right-skew), (1+lambda)>1 → heavier right tail
        z_eff = (b * z + a) / (1.0 + lambda_)
        log_kernel = neg_half_nu_plus_1 * np.log(1.0 + z_eff * z_eff * inv_nu_minus_2)
        ll = log_b + log_c + log_kernel - np.log(scale)
    
    return ll


@njit(cache=True, fastmath=False)
def hansen_robust_weight_scalar(
    innovation: float,
    S: float,
    nu: float,
    lambda_: float,
    a: float,
    b: float,
) -> float:
    """
    Robust weight w_t for Hansen Skew-t filter update.
    
    For the piecewise Hansen density, the effective z uses the
    side-specific scaling: z_eff = (b*z + a) / (1 ± λ).
    Then w_t = (ν+1) / (ν + z_eff²).
    
    Parameters
    ----------
    innovation : float
        r_t - mu_pred
    S : float
        Forecast variance
    nu, lambda_, a, b : float
        Hansen parameters
    
    Returns
    -------
    w_t : float
        Robust weight for Kalman update
    """
    if S <= _MIN_VARIANCE:
        return 1.0
    
    forecast_scale = np.sqrt(S * (nu - 2.0) / nu) if nu > 2.0 else np.sqrt(S)
    z = innovation / max(forecast_scale, 1e-12)
    cutpoint = -a / b
    
    if z < cutpoint:
        z_eff = (b * z + a) / (1.0 - lambda_)
    else:
        z_eff = (b * z + a) / (1.0 + lambda_)
    
    w_t = (nu + 1.0) / (nu + z_eff * z_eff)
    return w_t


# =============================================================================
# HANSEN CONSTANTS VALIDATION — Story 11.1 (Numerical Precision)
# =============================================================================
# Diagnostic function to verify Hansen constant precision and PDF
# normalization across the (nu, lambda) domain.
# =============================================================================


def hansen_validate_constants(
    nu: float,
    lambda_: float,
    n_quad: int = 10000,
    z_range: float = 20.0,
) -> dict:
    """
    Validate Hansen skew-t constants for numerical precision.

    Checks:
    1. Constants (a, b, c) match analytical formulas to 1e-12
    2. PDF integrates to 1.0 via Simpson's rule on [-z_range, z_range]
    3. For lambda=0, reduces to standard Student-t to 1e-14
    4. PDF is continuous at the cutpoint z = -a/b

    Parameters
    ----------
    nu : float
        Degrees of freedom (> 2)
    lambda_ : float
        Skewness parameter in (-1, 1)
    n_quad : int
        Number of quadrature points for Simpson's rule (must be even)
    z_range : float
        Integration domain [-z_range, z_range]

    Returns
    -------
    dict with keys:
        a, b, c_const : computed constants
        integral : numerical integral of PDF
        integral_error : |integral - 1.0|
        cutpoint_continuous : bool, PDF continuous at -a/b
        cutpoint_gap : absolute gap at cutpoint
        symmetric_match : bool (only if lambda_ == 0)
        symmetric_max_error : float (only if lambda_ == 0)
        grid_valid : bool (no NaN in constants)
    """
    import math

    a, b, c_const = hansen_constants_kernel(nu, lambda_)

    # --- Analytical verification ---
    nu_c = max(nu, 2.01)
    lam_c = max(-0.999, min(0.999, lambda_))

    log_c_ref = (math.lgamma((nu_c + 1.0) / 2.0)
                 - math.lgamma(nu_c / 2.0)
                 - 0.5 * math.log(math.pi * (nu_c - 2.0)))
    c_ref = math.exp(log_c_ref)
    a_ref = 4.0 * lam_c * c_ref * ((nu_c - 2.0) / (nu_c - 1.0))
    b_sq_ref = 1.0 + 3.0 * lam_c * lam_c - a_ref * a_ref
    b_ref = math.sqrt(max(b_sq_ref, 1e-10))

    const_errors = {
        "a_error": abs(a - a_ref),
        "b_error": abs(b - b_ref),
        "c_error": abs(c_const - c_ref),
    }

    # --- Simpson's rule integration ---
    if n_quad % 2 != 0:
        n_quad += 1
    h = 2.0 * z_range / n_quad
    integral = 0.0
    for i in range(n_quad + 1):
        z_i = -z_range + i * h
        lp = hansen_skew_t_logpdf_scalar(z_i, nu, lambda_, a, b, c_const, 0.0, 1.0)
        f_i = math.exp(lp)
        if i == 0 or i == n_quad:
            integral += f_i
        elif i % 2 == 1:
            integral += 4.0 * f_i
        else:
            integral += 2.0 * f_i
    integral *= h / 3.0

    # --- Cutpoint continuity ---
    cutpoint = -a / b
    eps_cut = 1e-10
    lp_left = hansen_skew_t_logpdf_scalar(
        cutpoint - eps_cut, nu, lambda_, a, b, c_const, 0.0, 1.0
    )
    lp_right = hansen_skew_t_logpdf_scalar(
        cutpoint + eps_cut, nu, lambda_, a, b, c_const, 0.0, 1.0
    )
    cutpoint_gap = abs(math.exp(lp_left) - math.exp(lp_right))

    result = {
        "a": a,
        "b": b,
        "c_const": c_const,
        "integral": integral,
        "integral_error": abs(integral - 1.0),
        "cutpoint_continuous": cutpoint_gap < 1e-8,
        "cutpoint_gap": cutpoint_gap,
        "grid_valid": not (math.isnan(a) or math.isnan(b) or math.isnan(c_const)),
        **const_errors,
    }

    # --- Symmetric check ---
    if abs(lambda_) < 1e-15:
        max_sym_err = 0.0
        # Hansen with lambda=0 uses (nu-2) parameterization for unit variance.
        # This equals Student-t with scale = sqrt((nu-2)/nu).
        sym_scale = math.sqrt((nu - 2.0) / nu) if nu > 2.0 else 1.0
        test_points = [0.0, 0.5, 1.0, 2.0, 5.0, -0.5, -1.0, -2.0, -5.0]
        for z_pt in test_points:
            lp_hansen = hansen_skew_t_logpdf_scalar(
                z_pt, nu, 0.0, a, b, c_const, 0.0, 1.0
            )
            lp_student = _student_t_logpdf_dynamic_nu(z_pt, nu, 0.0, sym_scale)
            max_sym_err = max(max_sym_err, abs(lp_hansen - lp_student))
        result["symmetric_match"] = max_sym_err < 1e-14
        result["symmetric_max_error"] = max_sym_err

    return result


# =============================================================================
# HANSEN LAMBDA ESTIMATION — Story 11.3 (Profile Likelihood)
# =============================================================================


def hansen_estimate_lambda(
    innovations: np.ndarray,
    nu: float,
    scale: float = 1.0,
    lambda_bounds: tuple = (-0.95, 0.95),
) -> dict:
    """
    Estimate Hansen skewness parameter lambda via profile likelihood.

    Fixes (nu, scale) and optimizes lambda using L-BFGS-B.
    Computes Fisher information for standard error.

    Parameters
    ----------
    innovations : np.ndarray
        Standardized residuals (z_t = (r_t - mu_t) / scale_t)
    nu : float
        Degrees of freedom (fixed)
    scale : float
        Observation scale (fixed, default 1.0 for pre-standardized data)
    lambda_bounds : tuple
        Bounds for lambda optimization

    Returns
    -------
    dict with keys:
        lambda_hat : float, estimated skewness parameter
        se_lambda : float, standard error from Fisher information
        log_likelihood : float, maximized log-likelihood
        bic_hansen : float, BIC for Hansen model (1 extra param)
        bic_symmetric : float, BIC for symmetric Student-t
        delta_bic : float, bic_symmetric - bic_hansen (>0 favors Hansen)
        converged : bool
    """
    from scipy.optimize import minimize

    n = len(innovations)
    if n < 10:
        return {
            "lambda_hat": 0.0,
            "se_lambda": float("inf"),
            "log_likelihood": float("-inf"),
            "bic_hansen": float("inf"),
            "bic_symmetric": float("inf"),
            "delta_bic": 0.0,
            "converged": False,
        }

    def neg_profile_ll(lam_arr):
        lam = float(lam_arr[0])
        a, b, c_const = hansen_constants_kernel(nu, lam)
        ll = 0.0
        for i in range(n):
            lp = hansen_skew_t_logpdf_scalar(
                innovations[i], nu, lam, a, b, c_const, 0.0, scale
            )
            ll += lp
        return -ll

    result = minimize(
        neg_profile_ll,
        x0=[0.0],
        method="L-BFGS-B",
        bounds=[lambda_bounds],
    )
    lambda_hat = float(result.x[0])
    ll_hansen = -result.fun
    converged = result.success

    # Symmetric log-likelihood (lambda=0)
    a0, b0, c0 = hansen_constants_kernel(nu, 0.0)
    ll_sym = 0.0
    for i in range(n):
        ll_sym += hansen_skew_t_logpdf_scalar(
            innovations[i], nu, 0.0, a0, b0, c0, 0.0, scale
        )

    # BIC: -2*LL + k*log(n)
    import math
    log_n = math.log(n)
    # Hansen has 1 extra parameter (lambda) vs symmetric
    bic_hansen = -2.0 * ll_hansen + 1.0 * log_n
    bic_symmetric = -2.0 * ll_sym

    # Fisher information via numerical second derivative
    eps_fi = 1e-5
    ll_plus = -neg_profile_ll([lambda_hat + eps_fi])
    ll_minus = -neg_profile_ll([lambda_hat - eps_fi])
    ll_center = ll_hansen
    fisher_info = -(ll_plus - 2.0 * ll_center + ll_minus) / (eps_fi * eps_fi)
    se_lambda = 1.0 / math.sqrt(max(fisher_info, 1e-12))

    return {
        "lambda_hat": lambda_hat,
        "se_lambda": se_lambda,
        "log_likelihood": ll_hansen,
        "bic_hansen": bic_hansen,
        "bic_symmetric": bic_symmetric,
        "delta_bic": bic_symmetric - bic_hansen,
        "converged": converged,
    }


# =============================================================================
# CONTAMINATED STUDENT-T KERNEL — Scalar (March 2026)
# =============================================================================
# CST: p(x) = (1-ε) × t(x; ν_normal) + ε × t(x; ν_crisis)
# Uses log-sum-exp for numerical stability. Integrated as a pipeline
# =============================================================================
# CST CONTAMINATION ESTIMATION — Story 12.1 (EM Algorithm)
# =============================================================================


def estimate_cst_contamination(
    innovations: np.ndarray,
    nu_normal_init: float = 8.0,
    nu_crisis_init: float = 3.0,
    epsilon_init: float = 0.03,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> dict:
    """
    Estimate CST contamination probability via EM algorithm.

    E-step: posterior p(crisis | r_t) for each observation.
    M-step: epsilon = mean(posteriors), nu updates via moment matching.

    Parameters
    ----------
    innovations : np.ndarray
        Standardized residuals
    nu_normal_init, nu_crisis_init : float
        Initial degrees of freedom
    epsilon_init : float
        Initial contamination probability
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence threshold for |epsilon_k+1 - epsilon_k|

    Returns
    -------
    dict with keys:
        epsilon_hat : float, estimated contamination probability
        nu_normal : float, normal component DoF
        nu_crisis : float, crisis component DoF
        posteriors : np.ndarray, posterior crisis probabilities per obs
        n_iter : int, iterations used
        converged : bool
        trajectory : list of epsilon values per iteration
    """
    import math

    n = len(innovations)
    if n < 10:
        return {
            "epsilon_hat": epsilon_init,
            "nu_normal": nu_normal_init,
            "nu_crisis": nu_crisis_init,
            "posteriors": np.zeros(n),
            "n_iter": 0,
            "converged": False,
            "trajectory": [],
        }

    eps = epsilon_init
    nu_n = nu_normal_init
    nu_c = nu_crisis_init
    posteriors = np.zeros(n)
    trajectory = [eps]

    for iteration in range(max_iter):
        # E-step: compute posterior p(crisis | r_t)
        for i in range(n):
            ll_n = _student_t_logpdf_dynamic_nu(innovations[i], nu_n, 0.0, 1.0)
            ll_c = _student_t_logpdf_dynamic_nu(innovations[i], nu_c, 0.0, 1.0)

            log_post_n = math.log(max(1.0 - eps, 1e-15)) + ll_n
            log_post_c = math.log(max(eps, 1e-15)) + ll_c
            max_log = max(log_post_n, log_post_c)
            denom = math.exp(log_post_n - max_log) + math.exp(log_post_c - max_log)
            posteriors[i] = math.exp(log_post_c - max_log) / denom

        # M-step: update epsilon
        eps_new = float(np.mean(posteriors))
        eps_new = max(1e-4, min(eps_new, 0.5))

        trajectory.append(eps_new)

        if abs(eps_new - eps) < tol:
            eps = eps_new
            break
        eps = eps_new

    return {
        "epsilon_hat": eps,
        "nu_normal": nu_n,
        "nu_crisis": nu_c,
        "posteriors": posteriors,
        "n_iter": len(trajectory) - 1,
        "converged": len(trajectory) - 1 < max_iter,
        "trajectory": trajectory,
    }


# =============================================================================
# CONTAMINATED STUDENT-T KERNEL — Scalar (March 2026)
# =============================================================================
# CST: p(x) = (1-ε) × t(x; ν_normal) + ε × t(x; ν_crisis)
# Uses log-sum-exp for numerical stability. Integrated as a pipeline
# stage within each model's Kalman filter.
# =============================================================================

@njit(cache=True, fastmath=False)
def cst_logpdf_scalar(
    x: float,
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
    mu: float,
    scale: float,
) -> float:
    """
    Contaminated Student-t scalar log-PDF.
    
    p(x) = (1-ε) × t(x; ν_normal, μ, σ) + ε × t(x; ν_crisis, μ, σ)
    
    Uses log-sum-exp trick for numerical stability.
    
    Parameters
    ----------
    x : float
        Observation
    nu_normal : float
        DoF for normal component
    nu_crisis : float
        DoF for crisis component (typically < nu_normal → heavier tails)
    epsilon : float
        Contamination probability (0 < ε < 1)
    mu : float
        Location (shared)
    scale : float
        Scale (shared, = forecast_scale from Kalman)
        
    Returns
    -------
    float
        Log-probability density
    """
    if scale <= _MIN_VARIANCE or nu_normal <= 2.0 or nu_crisis <= 2.0:
        return -1e12
    
    # Normal component log-pdf
    ll_normal = _student_t_logpdf_dynamic_nu(x, nu_normal, mu, scale)
    # Crisis component log-pdf
    ll_crisis = _student_t_logpdf_dynamic_nu(x, nu_crisis, mu, scale)
    
    # Log-sum-exp
    log_w_normal = np.log(1.0 - epsilon) + ll_normal
    log_w_crisis = np.log(epsilon) + ll_crisis
    
    max_log = max(log_w_normal, log_w_crisis)
    ll = max_log + np.log(np.exp(log_w_normal - max_log) + np.exp(log_w_crisis - max_log))
    
    return ll


@njit(cache=True, fastmath=False)
def cst_robust_weight_scalar(
    innovation: float,
    S: float,
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
) -> float:
    """
    Robust weight w_t for CST filter update.
    
    Uses posterior-weighted ν from the mixture:
    posterior_crisis = ε × t(z; ν_crisis) / [sum]
    w_normal = (ν_normal+1)/(ν_normal+z²)
    w_crisis = (ν_crisis+1)/(ν_crisis+z²)
    w_t = (1-posterior_crisis) × w_normal + posterior_crisis × w_crisis
    
    This provides soft adaptation: in calm regimes ν_normal dominates,
    in crisis the heavier ν_crisis tail is used for robustification.
    """
    if S <= _MIN_VARIANCE:
        return 1.0
    
    z_sq = (innovation * innovation) / S
    
    # Normal component weight
    w_normal = (nu_normal + 1.0) / (nu_normal + z_sq)
    # Crisis component weight
    w_crisis = (nu_crisis + 1.0) / (nu_crisis + z_sq)
    
    # Posterior probability of crisis component (via Bayes' rule on kernels)
    # For Student-t: kernel ∝ (1 + z²/ν)^{-(ν+1)/2}
    log_k_normal = -(nu_normal + 1.0) / 2.0 * np.log(1.0 + z_sq / nu_normal)
    log_k_crisis = -(nu_crisis + 1.0) / 2.0 * np.log(1.0 + z_sq / nu_crisis)
    
    log_post_normal = np.log(1.0 - epsilon) + log_k_normal
    log_post_crisis = np.log(epsilon) + log_k_crisis
    max_log = max(log_post_normal, log_post_crisis)
    
    post_crisis = np.exp(log_post_crisis - max_log) / (
        np.exp(log_post_normal - max_log) + np.exp(log_post_crisis - max_log)
    )
    
    w_t = (1.0 - post_crisis) * w_normal + post_crisis * w_crisis
    return w_t


# =============================================================================
# CST vs HANSEN MODEL SELECTION — Story 12.3
# =============================================================================


def select_tail_model(
    innovations: np.ndarray,
    nu: float = 5.0,
    outlier_threshold: float = 3.0,
    outlier_frac_cutoff: float = 0.03,
    skew_cutoff: float = 0.3,
) -> dict:
    """
    Select between CST and Hansen tail models based on data diagnostics.

    Heuristic rule:
    - If outlier fraction > outlier_frac_cutoff AND |skewness| < skew_cutoff -> CST
    - Otherwise -> Hansen

    Also computes BIC for both models when possible.

    Parameters
    ----------
    innovations : np.ndarray
        Standardized residuals
    nu : float
        Degrees of freedom for both models
    outlier_threshold : float
        Sigma threshold for outlier detection
    outlier_frac_cutoff : float
        Fraction threshold for CST preference
    skew_cutoff : float
        Skewness threshold for Hansen preference

    Returns
    -------
    dict with keys:
        selected_model : str, 'cst' or 'hansen'
        heuristic_model : str, model chosen by heuristic
        skewness : float
        excess_kurtosis : float
        outlier_fraction : float
        bic_cst : float (if computed)
        bic_hansen : float (if computed)
        bic_model : str (model preferred by BIC)
        heuristic_agrees : bool
    """
    import math
    from scipy.stats import skew, kurtosis

    n = len(innovations)
    if n < 20:
        return {
            "selected_model": "hansen",
            "heuristic_model": "hansen",
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "outlier_fraction": 0.0,
            "bic_cst": float("inf"),
            "bic_hansen": float("inf"),
            "bic_model": "hansen",
            "heuristic_agrees": True,
        }

    skw = float(skew(innovations))
    kurt = float(kurtosis(innovations))
    outlier_frac = float(np.mean(np.abs(innovations) > outlier_threshold))

    # Heuristic
    if outlier_frac > outlier_frac_cutoff and abs(skw) < skew_cutoff:
        heuristic = "cst"
    else:
        heuristic = "hansen"

    # BIC for Hansen
    hansen_result = hansen_estimate_lambda(innovations, nu)
    bic_hansen = hansen_result["bic_hansen"]

    # BIC for CST (estimate epsilon, then compute log-likelihood)
    cst_result = estimate_cst_contamination(innovations, nu_normal_init=nu)
    eps_hat = cst_result["epsilon_hat"]
    nu_crisis = 3.0  # fixed crisis DoF

    ll_cst = 0.0
    for i in range(n):
        ll_cst += cst_logpdf_scalar(
            innovations[i], nu, nu_crisis, eps_hat, 0.0, 1.0
        )
    # CST has 1 parameter (epsilon)
    bic_cst = -2.0 * ll_cst + 1.0 * math.log(n)

    bic_model = "cst" if bic_cst < bic_hansen else "hansen"

    # Final selection: use BIC if clear, otherwise heuristic
    selected = bic_model

    return {
        "selected_model": selected,
        "heuristic_model": heuristic,
        "skewness": skw,
        "excess_kurtosis": kurt,
        "outlier_fraction": outlier_frac,
        "bic_cst": bic_cst,
        "bic_hansen": bic_hansen,
        "bic_model": bic_model,
        "heuristic_agrees": heuristic == bic_model,
    }


# =============================================================================
# HANSEN SKEW-T FILTER KERNEL — Full Kalman (March 2026)
# =============================================================================
# φ-Student-t filter with Hansen observation noise instead of symmetric
# Student-t. Enables skewness-aware state estimation as a pipeline stage.
# =============================================================================

@njit(cache=True, fastmath=False)
def phi_hansen_skew_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    hansen_lambda: float,
    exogenous_input: np.ndarray,
    has_exogenous: bool,
    online_scale_adapt: bool,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    has_vov: bool,
    P0: float = 1e-4,
) -> tuple:
    """
    φ-Student-t filter with Hansen Skew-t observation noise.
    
    Identical interface to phi_student_t_enhanced_filter_kernel but
    uses asymmetric Hansen logpdf and robust weighting.
    
    Parameters
    ----------
    hansen_lambda : float
        Skewness parameter ∈ (-1, 1). λ=0 reduces to symmetric.
    """
    n = len(returns)
    phi_sq = phi * phi
    
    # Precompute Hansen constants
    a, b_h, c_const = hansen_constants_kernel(nu, hansen_lambda)
    
    # Pre-compute vol²
    vol_sq = np.empty(n)
    for t in range(n):
        vol_sq[t] = vol[t] * vol[t]
    
    mu_filtered = np.empty(n)
    P_filtered = np.empty(n)
    mu_pred_arr = np.empty(n)
    S_pred_arr = np.empty(n)
    
    # Data-adaptive initialization
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
    
    # Online scale adaptation state
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
        
        # Observation noise
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
        
        # Hansen-aware robust weighting
        w_t = hansen_robust_weight_scalar(innovation, S, nu, hansen_lambda, a, b_h)
        mu = mu_pred + K * w_t * innovation
        P = (1.0 - w_t * K) * P_pred
        
        if P < 1e-12:
            P = 1e-12
        mu_filtered[t] = mu
        P_filtered[t] = P if P > 1e-12 else 1e-12
        
        # Compute forecast scale and log-likelihood
        if nu > 2.0:
            forecast_scale = np.sqrt(S * (nu - 2.0) / nu)
        else:
            forecast_scale = np.sqrt(S)
        
        if forecast_scale > 1e-12:
            ll_t = hansen_skew_t_logpdf_scalar(
                returns[t], nu, hansen_lambda, a, b_h, c_const,
                mu_pred, forecast_scale)
            if ll_t == ll_t:  # NaN check
                log_likelihood += ll_t
            
            # Online scale adaptation
            if online_scale_adapt:
                z = innovation / forecast_scale
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
# CST FILTER KERNEL — Full Kalman (March 2026)
# =============================================================================

@njit(cache=True, fastmath=False)
def phi_cst_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
    exogenous_input: np.ndarray,
    has_exogenous: bool,
    online_scale_adapt: bool,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    has_vov: bool,
    P0: float = 1e-4,
) -> tuple:
    """
    φ-Student-t filter with Contaminated Student-t observation noise.
    
    Uses (1-ε)t(ν_normal) + εt(ν_crisis) mixture for observation likelihood,
    with posterior-weighted robust updates.
    """
    n = len(returns)
    phi_sq = phi * phi
    
    vol_sq = np.empty(n)
    for t in range(n):
        vol_sq[t] = vol[t] * vol[t]
    
    mu_filtered = np.empty(n)
    P_filtered = np.empty(n)
    mu_pred_arr = np.empty(n)
    S_pred_arr = np.empty(n)
    
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
    
    # OSA state
    # Use effective ν for chi² target: weighted average of components
    nu_eff = (1.0 - epsilon) * nu_normal + epsilon * nu_crisis
    chi2_tgt = nu_eff / (nu_eff - 2.0) if nu_eff > 2.0 else 1.0
    chi2_lam = 0.98
    chi2_1m = 0.02
    chi2_cap = chi2_tgt * 50.0
    ewm_z2 = chi2_tgt
    c_adj = 1.0
    
    for t in range(n):
        u_t = exogenous_input[t] if has_exogenous and t < len(exogenous_input) else 0.0
        mu_pred = phi * mu + u_t
        P_pred = phi_sq * P + q
        
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
        
        # CST-aware robust weighting
        w_t = cst_robust_weight_scalar(innovation, S, nu_normal, nu_crisis, epsilon)
        mu = mu_pred + K * w_t * innovation
        P = (1.0 - w_t * K) * P_pred
        
        if P < 1e-12:
            P = 1e-12
        mu_filtered[t] = mu
        P_filtered[t] = P if P > 1e-12 else 1e-12
        
        # Log-likelihood via CST mixture
        if nu_normal > 2.0:
            forecast_scale = np.sqrt(S * (nu_normal - 2.0) / nu_normal)
        else:
            forecast_scale = np.sqrt(S)
        
        if forecast_scale > 1e-12:
            ll_t = cst_logpdf_scalar(
                returns[t], nu_normal, nu_crisis, epsilon,
                mu_pred, forecast_scale)
            if ll_t == ll_t:
                log_likelihood += ll_t
            
            if online_scale_adapt:
                z = innovation / forecast_scale
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
    _MAXIT = 300

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


# =============================================================================
# STUDENT-T QUANTILE FUNCTION (PPF) -- Story 1.3
# =============================================================================
# Newton-Raphson inversion of Student-t CDF.
# Initial guess via Abramowitz & Stegun 26.7.5 rational approximation.
# Convergence: |F(x_n) - p| < 1e-10 within 15 iterations.
# =============================================================================

@njit(cache=True, fastmath=False)
def student_t_ppf_scalar(p: float, nu: float) -> float:
    """
    Student-t quantile function (inverse CDF) for a single probability.

    Uses Newton-Raphson with a rational approximation initial guess.
    Numba-compatible -- can be used inside @njit loops.

    Parameters
    ----------
    p : float
        Probability value in (0, 1)
    nu : float
        Degrees of freedom (> 2.0)

    Returns
    -------
    float
        x such that P(T <= x) = p
    """
    if p <= 0.0:
        return -1e12
    if p >= 1.0:
        return 1e12
    if nu <= 0.0:
        return 0.0

    # Use symmetry: if p < 0.5, compute ppf(1-p) and negate
    if p < 0.5:
        return -student_t_ppf_scalar(1.0 - p, nu)

    # For p = 0.5, median is 0
    if abs(p - 0.5) < 1e-15:
        return 0.0

    # Initial guess via Abramowitz & Stegun 26.7.5
    # Normal approximation: z_0 = Phi^{-1}(p)
    # Rational approximation for Phi^{-1}(p) when p > 0.5
    t_val = np.sqrt(-2.0 * np.log(1.0 - p))
    # Coefficients for Abramowitz & Stegun 26.2.23
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    z_normal = t_val - (c0 + c1 * t_val + c2 * t_val * t_val) / (
        1.0 + d1 * t_val + d2 * t_val * t_val + d3 * t_val * t_val * t_val
    )

    # Cornish-Fisher correction for Student-t (Abramowitz & Stegun 26.7.5)
    g1 = (z_normal * z_normal * z_normal + z_normal) / (4.0 * nu)
    g2 = ((5.0 * z_normal ** 5 + 16.0 * z_normal ** 3 + 3.0 * z_normal)
          / (96.0 * nu * nu))
    x = z_normal + g1 + g2

    # Newton-Raphson refinement: x_{n+1} = x_n - (F(x_n) - p) / f(x_n)
    for _iter in range(25):
        cdf_val = _student_t_cdf_scalar(x, nu)
        pdf_val = _student_t_pdf_scalar(x, nu)

        if pdf_val < 1e-30:
            break

        err = cdf_val - p
        if abs(err) < 1e-13:
            break

        x = x - err / pdf_val

    return x


@njit(cache=True, fastmath=False)
def student_t_ppf_array_kernel(p_arr: np.ndarray, nu: float) -> np.ndarray:
    """
    Vectorized Student-t PPF via Numba.

    Parameters
    ----------
    p_arr : np.ndarray
        Array of probability values in (0, 1)
    nu : float
        Degrees of freedom

    Returns
    -------
    np.ndarray
        Quantile values
    """
    n = len(p_arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = student_t_ppf_scalar(p_arr[i], nu)
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


@njit(cache=True, fastmath=False)
def crps_student_t_numerical_kernel(
    z_arr: np.ndarray,
    sigma_arr: np.ndarray,
    nu: float,
) -> float:
    """Correct CRPS for Student-t using numerical Gini half-mean-difference.

    v7.6: Replaces the analytic B_ratio formula (crps_student_t_kernel) which
    computes the C(ν) constant incorrectly. The analytic formula gives
    C(100) ≈ 0.808 vs correct numerical value ≈ 0.569.

    Uses the same approach as signals_calibration_numba.py:crps_student_t_nb
    but with the standardized-residual interface (z, sigma, nu) for backward
    compatibility with diagnostics.py/numba_wrappers.py.

    g(ν) = 2∫ x · F_ν(x) · f_ν(x) dx  (200-point trapezoidal quadrature)

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

    if nu < 2.01:
        nu = 2.01

    # Compute g(ν) via numerical quadrature (200 points, ~10K ops)
    N_QUAD = 200
    L = min(30.0, max(10.0, 4.0 * np.sqrt(nu / max(nu - 2.0, 0.1))))
    h = 2.0 * L / N_QUAD
    g_nu = 0.0
    for i in range(N_QUAD + 1):
        x = -L + i * h
        fx = _student_t_pdf_scalar(x, nu)
        Fx = _student_t_cdf_scalar(x, nu)
        val = x * Fx * fx
        if i == 0 or i == N_QUAD:
            g_nu += 0.5 * val
        else:
            g_nu += val
    g_nu = 2.0 * g_nu * h

    # Pre-compute PDF constants
    log_norm = (_lanczos_gammaln((nu + 1.0) / 2.0)
                - _lanczos_gammaln(nu / 2.0)
                - 0.5 * np.log(nu * np.pi))
    neg_exp = (nu + 1.0) / 2.0
    inv_nu = 1.0 / nu
    nu_m1_inv = 1.0 / (nu - 1.0)

    crps_sum = 0.0
    n_valid = 0

    for i in range(n):
        z = z_arr[i]
        sig = sigma_arr[i]
        if sig < 1e-10:
            sig = 1e-10

        pdf_z = np.exp(log_norm - neg_exp * np.log(1.0 + z * z * inv_nu))
        cdf_z = _student_t_cdf_scalar(z, nu)

        term1 = z * (2.0 * cdf_z - 1.0)
        term2 = 2.0 * pdf_z * (nu + z * z) * nu_m1_inv
        crps_i = sig * (term1 + term2 - g_nu)

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
def garch_h0_from_trailing(sq: np.ndarray, window: int = 20) -> float:
    """Story 3.2: Compute robust h0 from trailing squared innovations.

    Uses median of first `window` squared innovations as h0.
    Falls back to -1.0 (use unconditional) if insufficient data.
    """
    n = len(sq)
    if n < window:
        return -1.0
    # Collect first `window` values and compute median via sort
    buf = np.empty(window, dtype=np.float64)
    for i in range(window):
        buf[i] = sq[i]
    buf.sort()
    if window % 2 == 0:
        med = 0.5 * (buf[window // 2 - 1] + buf[window // 2])
    else:
        med = buf[window // 2]
    # Fallback: if median is tiny, return -1 to use unconditional
    if med < 1e-10:
        return -1.0
    return med


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
    # v7.8: Tier 4 params
    liq_stress_coeff: float = 0.0,
    leverage_dynamic_decay: float = 0.0,
    # Story 3.2: regime-aware initialization
    h0_override: float = -1.0,
) -> None:
    """
    GJR-GARCH(1,1) variance with leverage correlation, vol-of-vol noise,
    Markov regime switching, mean reversion, dynamic leverage, and
    liquidity-volatility feedback.

    v7.8: Added liq_stress_coeff (Brunnermeier-Pedersen liquidity spiral)
    and leverage_dynamic_decay (crash-clustering GJR amplification).

    Story 3.2: h0_override > 0 uses trailing realized variance instead of
    unconditional. Typical usage: h0_override = median(eps^2_{1:20}).

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
    liq_stress_coeff : float
        Liquidity-volatility feedback lambda_liq. 0 = disabled.
    leverage_dynamic_decay : float
        EWM decay for dynamic GJR leverage. 0 = disabled.
    h0_override : float
        If > 0, use this as h_0 instead of unconditional variance (gu).
        Set to median(eps^2_{1:20}) for regime-aware init.
    """
    # Story 3.2: use h0_override if valid, else fall back to unconditional
    if h0_override > 1e-12:
        h_out[0] = h0_override
    else:
        h_out[0] = gu
    ps = 0.1  # Initial stress probability
    use_dynamic_lev = leverage_dynamic_decay > 0.01 and gl > 1e-8
    use_liq = liq_stress_coeff > 0.005 and tv > 1e-12
    neg_frac_ema = 0.5  # Neutral initialization

    for t in range(1, n):
        ht = go + ga * sq[t - 1] + gb * h_out[t - 1]

        # GJR leverage (v7.8: with dynamic amplification)
        if use_dynamic_lev:
            if neg[t - 1] > 0.5:
                gamma_dyn = gl * max(0.5, min(2.0,
                    1.0 + 2.0 * (neg_frac_ema - 0.5)))
                ht += gamma_dyn * sq[t - 1]
            # Update EWM
            neg_frac_ema = (1.0 - leverage_dynamic_decay) * neg_frac_ema + leverage_dynamic_decay * neg[t - 1]
        else:
            ht += gl * sq[t - 1] * neg[t - 1]

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

        # v7.8 Tier 4: Liquidity-volatility feedback
        if use_liq:
            vol_ratio = ht / tv
            if vol_ratio > 1.0:
                excess = min(vol_ratio - 1.0, 3.0)  # Cap to prevent divergence
                amp = min(1.0 + liq_stress_coeff * excess * excess, 2.0)
                ht *= amp
            # Hard cap: vol can't exceed 50x unconditional
            if ht > 50.0 * tv:
                ht = 50.0 * tv

        if ht < 1e-12:
            ht = 1e-12
        # Story 3.1: hard cap at 100x unconditional to prevent explosion
        if gu > 1e-12 and ht > 100.0 * gu:
            ht = 100.0 * gu
        h_out[t] = ht


# =============================================================================
# PIT KL DIVERGENCE KERNEL (v7.8 entropy stabilizer)
# =============================================================================

@njit(cache=True)
def pit_kl_uniform_kernel(pit_values: np.ndarray, n_bins: int = 20) -> float:
    """Compute KL divergence of PIT values from Uniform(0,1) distribution.

    Uses histogram-based density estimation for Numba compatibility.
    D_KL(PIT || Uniform) = sum_k f_k * ln(f_k) where f_k is the
    histogram density in bin k (normalized so integral = 1).

    Parameters
    ----------
    pit_values : ndarray
        PIT values in [0, 1]
    n_bins : int
        Number of histogram bins (default 20)

    Returns
    -------
    kl_div : float
        KL divergence >= 0. Zero means perfectly uniform.
    """
    n = len(pit_values)
    if n < 10:
        return 0.0

    # Build histogram
    counts = np.zeros(n_bins, dtype=np.float64)
    for i in range(n):
        p = pit_values[i]
        if p < 0.0:
            p = 0.0
        elif p >= 1.0:
            p = 1.0 - 1e-10
        b = int(p * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        counts[b] += 1.0

    # Convert to density (integral = 1 means each bin has density = count/(n * bin_width))
    # For uniform, density = 1/n_bins * n_bins = 1.0 in each bin
    # Simpler: use count fractions vs expected uniform fraction
    kl = 0.0
    expected = float(n) / float(n_bins)
    for k in range(n_bins):
        if counts[k] > 0.5:  # At least one observation
            ratio = counts[k] / expected
            kl += (counts[k] / float(n)) * np.log(ratio)

    if kl < 0.0:
        kl = 0.0  # Numerical safety
    return kl


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
def phi_gaussian_train_state_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    train_start: int,
    train_end: int,
    P0: float = 1e-4,
) -> tuple:
    """
    Terminal-state φ-Gaussian filter for CV training prefixes.

    Mirrors phi_gaussian_filter_kernel's state update but returns only the
    final filtered state, avoiding full mu/P allocations in optimizer loops.
    """
    mu = 0.0
    P = P0
    phi_sq = phi * phi
    log_likelihood = 0.0

    for t in range(train_start, train_end):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q
        R = c * vol_sq[t]
        innovation = returns[t] - mu_pred
        S = P_pred + R

        if S > _MIN_VARIANCE:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred

            innov_sq_scaled = (innovation * innovation) / S
            if innov_sq_scaled > 100.0:
                innov_sq_scaled = 100.0
            ll_contrib = -0.5 * (_LOG_2PI + np.log(S) + innov_sq_scaled)
            if ll_contrib < -_MAX_LL_CONTRIB:
                ll_contrib = -_MAX_LL_CONTRIB
            log_likelihood += ll_contrib
        else:
            mu = mu_pred
            P = P_pred

    return mu, P if P > _MIN_VARIANCE else _MIN_VARIANCE, log_likelihood


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
            # Student-t scale = sqrt(S * (nu-2)/nu) for predictive density
            if nu > 2.0:
                scale = np.sqrt(S * (nu - 2.0) / nu)
            else:
                scale = np.sqrt(S)
            z = innovation / scale
            z_sq = z * z
            
            # Log-likelihood contribution (Student-t predictive density)
            ll_t = log_norm_const - np.log(scale) + neg_exp * np.log(1.0 + z_sq * inv_nu)
            if ll_t < -_MAX_LL_CONTRIB:
                ll_t = -_MAX_LL_CONTRIB
            log_likelihood += ll_t
            
            # LFO-CV accumulation
            if t >= lfo_start:
                lfo_sum += ll_t
                lfo_count += 1
            
            # Student-t robust weighting: downweight outliers
            # w_t = (nu + 1) / (nu + z_sq_S) where z_sq_S = innovation^2 / S
            z_sq_S = (innovation * innovation) / S
            w_t = (nu + 1.0) / (nu + z_sq_S)
            
            # Robust Kalman gain with Student-t weighting
            K = P_pred / S
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
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
    log_norm_const = _lanczos_gammaln((nu_base + 1.0) / 2.0) - _lanczos_gammaln(nu_base / 2.0) - 0.5 * np.log(nu_base * np.pi)
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
                log_n_diff = _lanczos_gammaln((nu_eff + 1.0) / 2.0) - _lanczos_gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
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
                log_norm_eff = _lanczos_gammaln((nu_eff + 1.0) / 2.0) - _lanczos_gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
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
# AD TAIL-CORRECTION KERNELS (March 2026)
# =============================================================================
# Numba-accelerated kernels for the Anderson-Darling correction pipeline.
# Three stages: TWSC (Tail-Weighted Scale Correction),
#               SPTG (Semi-Parametric EVT Tail Grafting CDF evaluation),
#               Gaussian normal CDF (for SPTG Gaussian path).
# =============================================================================

@njit(cache=True, fastmath=False)
def ad_twsc_kernel(
    z_arr: np.ndarray,
    ewma_lambda: float,
    alpha_quantile: float,
    kappa: float,
    max_inflate: float,
    deadzone: float,
) -> np.ndarray:
    """
    Tail-Weighted Scale Correction (TWSC) — Numba inner loop.

    Tracks separate left-tail and right-tail exceedance frequencies via EWMA.
    When observed tail frequency exceeds the theoretical rate (alpha_quantile)
    beyond a deadzone, inflates the local scale to bring tails back in line.

    The key insight: if the model predicts 5% in each tail but we observe 8%,
    the scale is too tight. Inflating scale widens the CDF and pushes tail
    PIT values away from 0/1, directly improving the AD statistic.

    Parameters
    ----------
    z_arr : np.ndarray
        Standardized residuals z_t = (r_t - mu_pred_t) / scale_t
    ewma_lambda : float
        EWMA decay for tail frequency tracking (0.97 = ~33 obs halflife)
    alpha_quantile : float
        Theoretical tail probability threshold (0.05 = 5% in each tail)
    kappa : float
        Correction strength: inflate = 1 + kappa * (f_tail/f_expected - 1)
    max_inflate : float
        Maximum scale inflation factor (prevents over-correction)
    deadzone : float
        Minimum excess (f_tail/f_expected - 1) before correction activates

    Returns
    -------
    np.ndarray
        Per-observation scale inflation factors (multiply original scale by these)
    """
    n = len(z_arr)
    scale_adj = np.ones(n, dtype=np.float64)

    # Compute threshold from alpha. For simplicity use a fixed z_alpha.
    # For alpha=0.05: ~1.645 for Gaussian, but we use a Student-t friendly
    # heuristic: just count how often |z| exceeds the expected tail rate.
    # The threshold is set so that under the model, P(|z| > threshold) ≈ 2*alpha.
    # We use z_alpha = 1.645 as a reasonable default for alpha=0.05
    # (this is the normal quantile — slightly conservative for Student-t,
    #  which has even more tail mass, making the correction even more appropriate).
    if alpha_quantile <= 0.0 or alpha_quantile >= 0.5:
        return scale_adj

    # Normal quantile approximation (Beasley-Springer-Moro, sufficient for Numba)
    # For alpha ∈ [0.01, 0.20], this gives z_alpha with <0.001 error.
    p = alpha_quantile
    # Rational approximation for probit
    a0 = 2.515517
    a1 = 0.802853
    a2 = 0.010328
    b1 = 1.432788
    b2 = 0.189269
    b3 = 0.001308
    t_val = math.sqrt(-2.0 * math.log(p))
    z_alpha = t_val - (a0 + a1 * t_val + a2 * t_val * t_val) / (
        1.0 + b1 * t_val + b2 * t_val * t_val + b3 * t_val * t_val * t_val
    )

    f_expected = alpha_quantile  # Expected frequency in each tail

    # Initialize EWMA trackers at the expected rate
    f_left = f_expected
    f_right = f_expected
    lam = ewma_lambda
    one_m_lam = 1.0 - lam

    for t in range(n):
        z = z_arr[t]
        # Update tail frequency trackers
        is_left = 1.0 if z < -z_alpha else 0.0
        is_right = 1.0 if z > z_alpha else 0.0
        f_left = lam * f_left + one_m_lam * is_left
        f_right = lam * f_right + one_m_lam * is_right

        # Compute inflation from the worse tail
        excess_left = f_left / f_expected - 1.0
        excess_right = f_right / f_expected - 1.0
        excess = max(excess_left, excess_right)

        if excess > deadzone:
            active_excess = excess - deadzone
            inflate = 1.0 + kappa * active_excess
            inflate = min(inflate, max_inflate)
            scale_adj[t] = inflate

    return scale_adj


@njit(cache=True, fastmath=False)
def ad_sptg_cdf_student_t_scalar(
    z: float,
    nu: float,
    xi_left: float,
    sigma_left: float,
    u_left: float,
    xi_right: float,
    sigma_right: float,
    u_right: float,
    p_left: float,
    p_right: float,
) -> float:
    """
    Semi-Parametric Tail-Grafted CDF for Student-t (single observation).

    Replaces the model CDF in the tails with a GPD-based CDF while keeping
    the bulk (middle region) from the original Student-t. This corrects
    tail shape misspecification that the AD test is specifically sensitive to.

    Piecewise definition:
        z < -u_left:  F(z) = p_left * GPD_left_cdf(|z| - u_left)
        z > +u_right: F(z) = 1 - p_right * GPD_right_survival(z - u_right)
        otherwise:    F(z) = Student-t CDF(z, nu)  [original model]

    Parameters
    ----------
    z : float
        Standardized residual
    nu : float
        Degrees of freedom for bulk Student-t CDF
    xi_left, sigma_left : float
        GPD shape and scale for left tail
    u_left : float
        Threshold (positive) for left tail (|z| > u_left triggers GPD)
    xi_right, sigma_right : float
        GPD shape and scale for right tail
    u_right : float
        Threshold (positive) for right tail (z > u_right triggers GPD)
    p_left : float
        Probability mass below -u_left under the model (= F(-u_left))
    p_right : float
        Probability mass above +u_right under the model (= 1 - F(u_right))

    Returns
    -------
    float
        PIT value ∈ (0, 1)
    """
    if z < -u_left:
        # Left tail: GPD CDF for exceedances above threshold
        y = (-z) - u_left  # exceedance (positive)
        if y < 0.0:
            y = 0.0
        if abs(xi_left) < 1e-10:
            # Exponential tail (xi ≈ 0)
            gpd_cdf = 1.0 - math.exp(-y / sigma_left)
        else:
            arg = 1.0 + xi_left * y / sigma_left
            if arg <= 0.0:
                gpd_cdf = 1.0
            else:
                gpd_cdf = 1.0 - arg ** (-1.0 / xi_left)
        # P(Z < z) = p_left * (1 - GPD_survival) where GPD is fitted on |Z| > u_left
        # Actually: PIT = p_left * (1 - gpd_cdf) because further into left tail = smaller PIT
        pit = p_left * (1.0 - gpd_cdf)
        return max(1e-10, min(1.0 - 1e-10, pit))

    elif z > u_right:
        # Right tail: GPD CDF for exceedances above threshold
        y = z - u_right  # exceedance (positive)
        if y < 0.0:
            y = 0.0
        if abs(xi_right) < 1e-10:
            gpd_cdf = 1.0 - math.exp(-y / sigma_right)
        else:
            arg = 1.0 + xi_right * y / sigma_right
            if arg <= 0.0:
                gpd_cdf = 1.0
            else:
                gpd_cdf = 1.0 - arg ** (-1.0 / xi_right)
        # P(Z < z) = 1 - p_right * (1 - gpd_cdf)
        pit = 1.0 - p_right * (1.0 - gpd_cdf)
        return max(1e-10, min(1.0 - 1e-10, pit))

    else:
        # Bulk: use original Student-t CDF
        pit = _student_t_cdf_scalar(z, nu)
        return max(1e-10, min(1.0 - 1e-10, pit))


@njit(cache=True, fastmath=False)
def ad_sptg_cdf_student_t_array(
    z_arr: np.ndarray,
    nu: float,
    xi_left: float,
    sigma_left: float,
    u_left: float,
    xi_right: float,
    sigma_right: float,
    u_right: float,
    p_left: float,
    p_right: float,
) -> np.ndarray:
    """Vectorized SPTG CDF for Student-t (calls scalar kernel in loop)."""
    n = len(z_arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = ad_sptg_cdf_student_t_scalar(
            z_arr[i], nu,
            xi_left, sigma_left, u_left,
            xi_right, sigma_right, u_right,
            p_left, p_right,
        )
    return out


@njit(cache=True, fastmath=False)
def _ndtr_scalar(x: float) -> float:
    """
    Standard normal CDF Φ(x) — Numba-compatible.

    Uses the Abramowitz & Stegun rational approximation (7.1.26),
    max error < 1.5e-7 across the entire real line.
    """
    # Handle extreme values
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    # Symmetry: Φ(-x) = 1 - Φ(x)
    neg = x < 0.0
    ax = abs(x)

    # Constants for rational approximation
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * ax)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pdf = math.exp(-0.5 * ax * ax) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - pdf * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)

    if neg:
        return 1.0 - cdf
    return cdf


@njit(cache=True, fastmath=False)
def ad_sptg_cdf_gaussian_scalar(
    z: float,
    xi_left: float,
    sigma_left: float,
    u_left: float,
    xi_right: float,
    sigma_right: float,
    u_right: float,
    p_left: float,
    p_right: float,
) -> float:
    """
    Semi-Parametric Tail-Grafted CDF for Gaussian (single observation).

    Same piecewise structure as Student-t variant but uses Φ(z) in the bulk.
    """
    if z < -u_left:
        y = (-z) - u_left
        if y < 0.0:
            y = 0.0
        if abs(xi_left) < 1e-10:
            gpd_cdf = 1.0 - math.exp(-y / sigma_left)
        else:
            arg = 1.0 + xi_left * y / sigma_left
            if arg <= 0.0:
                gpd_cdf = 1.0
            else:
                gpd_cdf = 1.0 - arg ** (-1.0 / xi_left)
        pit = p_left * (1.0 - gpd_cdf)
        return max(1e-10, min(1.0 - 1e-10, pit))

    elif z > u_right:
        y = z - u_right
        if y < 0.0:
            y = 0.0
        if abs(xi_right) < 1e-10:
            gpd_cdf = 1.0 - math.exp(-y / sigma_right)
        else:
            arg = 1.0 + xi_right * y / sigma_right
            if arg <= 0.0:
                gpd_cdf = 1.0
            else:
                gpd_cdf = 1.0 - arg ** (-1.0 / xi_right)
        pit = 1.0 - p_right * (1.0 - gpd_cdf)
        return max(1e-10, min(1.0 - 1e-10, pit))

    else:
        pit = _ndtr_scalar(z)
        return max(1e-10, min(1.0 - 1e-10, pit))


@njit(cache=True, fastmath=False)
def ad_sptg_cdf_gaussian_array(
    z_arr: np.ndarray,
    xi_left: float,
    sigma_left: float,
    u_left: float,
    xi_right: float,
    sigma_right: float,
    u_right: float,
    p_left: float,
    p_right: float,
) -> np.ndarray:
    """Vectorized SPTG CDF for Gaussian (calls scalar kernel in loop)."""
    n = len(z_arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = ad_sptg_cdf_gaussian_scalar(
            z_arr[i],
            xi_left, sigma_left, u_left,
            xi_right, sigma_right, u_right,
            p_left, p_right,
        )
    return out


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
    # v7.6: Enriched MC params from tuned Student-t / unified models
    garch_leverage: float = 0.0,
    variance_inflation: float = 1.0,
    mu_drift: float = 0.0,
    alpha_asym: float = 0.0,
    k_asym: float = 2.0,
    risk_premium_sensitivity: float = 0.0,
    # v7.7: Tier 2 — vol mean-reversion, CRPS shrinkage, MS process noise, rough vol
    kappa_mean_rev: float = 0.0,
    theta_long_var: float = 0.0,
    crps_sigma_shrinkage: float = 1.0,
    ms_sensitivity: float = 0.0,
    q_stress_ratio: float = 1.0,
    rough_hurst: float = 0.0,
    frac_weights: np.ndarray = np.empty(0, dtype=np.float64),
    # v7.7: Tier 3 — vol-of-vol, asymmetric ν, regime switching, GAS skew, loc bias
    sigma_eta: float = 0.0,
    t_df_asym: float = 0.0,
    regime_switch_prob: float = 0.0,
    gamma_vov: float = 0.0,
    vov_damping: float = 0.0,
    skew_score_sensitivity: float = 0.0,
    skew_persistence: float = 0.97,
    loc_bias_var_coeff: float = 0.0,
    loc_bias_drift_coeff: float = 0.0,
    q_vol_coupling: float = 0.0,
    # v7.8: Tier 4 — dynamic leverage, liquidity stress
    leverage_dynamic_decay: float = 0.0,
    liq_stress_coeff: float = 0.0,
    # Story 1.4: Dual-frequency drift propagation
    phi_slow: float = 0.0,
    mu_slow_0: float = 0.0,
    # Story 3.4: Asset-class-aware per-step return cap
    return_cap: float = 0.30,
) -> None:
    """Unified MC simulation kernel with GJR-GARCH + jumps + Student-t.

    v7.7: Full Tier 2 + Tier 3 MC integration.
    v7.8: Tier 4 — dynamic leverage + liquidity-volatility feedback.
      - garch_leverage (GJR-γ): asymmetric variance
      - variance_inflation (β): calibrated predictive variance scaling
      - mu_drift: systematic drift bias correction
      - alpha_asym + k_asym: asymmetric tail thickness
      - risk_premium_sensitivity: variance-conditional drift (ICAPM)

    Tier 2 (v7.7):
      - kappa_mean_rev + theta_long_var: vol mean-reversion (Heston 1993)
      - crps_sigma_shrinkage: CRPS-optimal sigma tightening
      - ms_sensitivity + q_stress_ratio: MS process noise for drift
      - rough_hurst + frac_weights: fractional vol memory (Gatheral 2018)

    Tier 3 (v7.7):
      - sigma_eta: vol-of-vol noise (Heston discrete analog)
      - t_df_asym: static two-piece ν offset
      - regime_switch_prob: Markov switching on observation variance
      - gamma_vov + vov_damping: VoV observation noise
      - skew_score_sensitivity + skew_persistence: GAS dynamic skew
      - loc_bias_var_coeff + loc_bias_drift_coeff: location bias correction
      - q_vol_coupling: process noise volatility coupling (dead param)

    Tier 4 (v7.8):
      - leverage_dynamic_decay: EWM crash-clustering → amplifies GJR γ
      - liq_stress_coeff: Brunnermeier-Pedersen liquidity-vol feedback

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
    garch_leverage : float
        GJR asymmetric GARCH leverage γ. h_t += γ·ε²·I(ε<0)
    variance_inflation : float
        Calibrated predictive variance multiplier β (scales h0)
    mu_drift : float
        Additive drift bias correction from tuned model
    alpha_asym : float
        Asymmetric tail parameter: ν_eff = ν·(1 + α·tanh(k·z))
        α<0 means heavier left tail (typical for equities)
    k_asym : float
        Transition sharpness for asymmetric ν (default 2.0)
    risk_premium_sensitivity : float
        ICAPM variance-conditional drift: E[r] += λ·h_t
    kappa_mean_rev : float
        Variance mean-reversion speed κ ∈ [0, 0.3]. h_t = (1-κ)·h_garch + κ·θ_long
    theta_long_var : float
        Long-term variance target for mean reversion
    crps_sigma_shrinkage : float
        CRPS sigma multiplier ∈ [0.5, 1.0]. Applied to h0.
    ms_sensitivity : float
        MS process noise sigmoid sensitivity. 0 = disabled.
    q_stress_ratio : float
        q_stress = drift_q × q_stress_ratio. 1.0 = no effect.
    rough_hurst : float
        Hurst exponent H ∈ [0, 0.5]. 0 = disabled.
    frac_weights : ndarray
        Pre-computed fractional differencing weights for rough vol. Empty = disabled.
    sigma_eta : float
        Vol-of-vol noise ∈ [0, 0.5]. 0 = disabled.
    t_df_asym : float
        Static two-piece ν offset. 0 = symmetric.
    regime_switch_prob : float
        Observation-layer regime switching ∈ [0, 0.15]. 0 = disabled.
    gamma_vov : float
        VoV observation noise sensitivity ∈ [0, 1.0]. 0 = disabled.
    vov_damping : float
        Reduce VoV when MS-q stress is active ∈ [0, 0.5].
    skew_score_sensitivity : float
        GAS skew κ_λ ≥ 0. 0 = static alpha_asym.
    skew_persistence : float
        GAS skew persistence ρ_λ ∈ [0.90, 0.99].
    loc_bias_var_coeff : float
        Location bias variance coefficient a ∈ [-0.5, 0.5].
    loc_bias_drift_coeff : float
        Location bias drift coefficient b ∈ [-0.5, 0.5].
    q_vol_coupling : float
        Process noise volatility coupling ζ ∈ [0, 1.0]. Dead param (always 0).
    leverage_dynamic_decay : float
        EWM decay for crash-clustering neg-return fraction ∈ [0, 0.8].
        0 = disabled (static GJR). When active, GJR γ is amplified during
        sustained negative return clustering (Bouchaud & Potters 2003).
    liq_stress_coeff : float
        Liquidity-volatility feedback λ_liq ∈ [0, 0.5].
        0 = disabled. Amplifies h_t when vol exceeds θ_long via
        quadratic excess: h_t *= (1 + λ·max(0, h/θ - 1)²).
        Models Brunnermeier-Pedersen liquidity spiral.
    """
    use_student_t = (nu > 2.0) and (nu < 100.0)
    use_gjr = (garch_leverage > 1e-8) and use_garch
    use_asym = use_student_t and (abs(alpha_asym) > 1e-8)
    use_risk_premium = abs(risk_premium_sensitivity) > 1e-10

    # v7.7: Feature flags for Tier 2+3
    use_kappa = kappa_mean_rev > 0.001 and theta_long_var > 1e-12
    use_ms_q = ms_sensitivity > 0.01 and q_stress_ratio > 1.01
    use_rough = rough_hurst > 0.001 and len(frac_weights) > 0
    use_sigma_eta = sigma_eta > 0.005
    use_t_df_asym = use_student_t and abs(t_df_asym) > 0.05
    use_regime_sw = regime_switch_prob > 0.005
    use_gamma_vov = gamma_vov > 0.005
    use_gas_skew = use_student_t and skew_score_sensitivity > 1e-6
    use_loc_bias = abs(loc_bias_var_coeff) > 1e-6 or abs(loc_bias_drift_coeff) > 1e-6
    use_q_vol_coupling = q_vol_coupling > 0.001 and theta_long_var > 1e-12

    # v7.8: Feature flags for Tier 4
    use_dynamic_lev = leverage_dynamic_decay > 0.01 and garch_leverage > 1e-8 and use_garch
    use_liq_stress = liq_stress_coeff > 0.005 and theta_long_var > 1e-12 and use_garch

    # Story 1.4: Dual-frequency drift flag
    use_dual_freq = phi_slow > 0.0 and abs(mu_slow_0) > 1e-15

    # Rough vol: max lag from frac_weights length (capped at 50)
    rough_max_lag = min(len(frac_weights), 50) if use_rough else 0

    # Precompute Student-t variance scaling
    if use_student_t:
        t_var = nu / (nu - 2.0)
        t_scale_factor = 1.0 / np.sqrt(t_var)
    else:
        t_scale_factor = 1.0

    drift_sigma = np.sqrt(drift_q) if drift_q > 0.0 else 0.0

    # v7.6+7.7: Apply variance_inflation AND crps_sigma_shrinkage to initial variance
    h0_cal = h0 * variance_inflation * crps_sigma_shrinkage

    # ================================================================
    # GARCH VARIANCE CAP (v7.9): Prevent GARCH explosion
    # ================================================================
    # With Student-t(nu<=4) innovations, extreme draws compound with
    # GARCH feedback creating unrealistic variance growth (e.g., h_t>1e4).
    # Cap at 25× initial calibrated variance, floored at 0.005 (~7% daily).
    # This bounds sigma_daily to at most ~sqrt(25)*sigma_init, preserving
    # crisis dynamics while preventing pathological MC paths.
    # ================================================================
    h_dyn_cap = 25.0 * h0_cal
    if h_dyn_cap < 0.005:
        h_dyn_cap = 0.005

    for p in range(n_paths):
        mu_t = mu_now + mu_drift
        h_t = h0_cal
        if h_t < 1e-12:
            h_t = 1e-12
        cum = 0.0
        # Story 1.4: Slow drift component (deterministic decay)
        mu_slow_t = mu_slow_0 if use_dual_freq else 0.0

        # v7.7 Tier 3: per-path state variables
        p_stress_obs = 0.1  # regime_switch_prob state
        alpha_t = alpha_asym  # GAS skew state
        # v7.7: VoV causal EWM state for gamma_vov
        log_h_ema = np.log(max(h_t, 1e-12))
        log_h_var_ema = 0.0
        # v7.7: MS-q causal EWM state
        vol_ema = h_t
        # v7.7: Rough vol circular buffer
        sq_buf = np.zeros(50)
        # v7.8 Tier 4: dynamic leverage EWM state
        neg_frac_ema = 0.5  # Start neutral (50% neg returns)

        for t in range(H_max):
            sigma_t = np.sqrt(h_t)
            vol_out[t, p] = sigma_t

            # ================================================================
            # OBSERVATION NOISE: Student-t or Gaussian
            # ================================================================
            if use_student_t:
                chi2_val = z_chi2[t, p]
                if chi2_val < 1e-8:
                    chi2_val = 1e-8
                raw_t = z_normals[t, p] / np.sqrt(chi2_val)

                # v7.7: Static two-piece ν (t_df_asym)
                # Applied BEFORE dynamic alpha_asym
                if use_t_df_asym:
                    if raw_t < 0.0:
                        nu_piece = max(2.5, nu - t_df_asym)
                    else:
                        nu_piece = max(2.5, nu + t_df_asym)
                    # Re-scale with piece-specific variance
                    t_var_piece = nu_piece / (nu_piece - 2.0)
                    eps = raw_t / np.sqrt(t_var_piece)
                elif use_asym or use_gas_skew:
                    # v7.6: Dynamic asymmetric tail thickness
                    # v7.7: Use alpha_t (GAS-evolving) instead of static alpha_asym
                    a_eff = alpha_t if use_gas_skew else alpha_asym
                    nu_eff = nu * (1.0 + a_eff * np.tanh(k_asym * raw_t))
                    if nu_eff < 2.5:
                        nu_eff = 2.5
                    elif nu_eff > 200.0:
                        nu_eff = 200.0
                    t_var_eff = nu_eff / (nu_eff - 2.0)
                    eps = raw_t / np.sqrt(t_var_eff)
                else:
                    eps = raw_t * t_scale_factor
            else:
                eps = z_normals[t, p]

            e_t = sigma_t * eps

            # Jump component
            jump = 0.0
            if enable_jumps and jump_intensity > 0.0:
                if z_jump_uniform[t, p] < jump_intensity:
                    jump = jump_mean + jump_std * z_jump_normal[t, p]
                if jump_intensity > 0.1 and z_jump_uniform[t, p] < jump_intensity * jump_intensity:
                    jump += jump_mean + jump_std * z_drift[t, p] * 0.5

            # v7.6: Variance-conditional risk premium (ICAPM)
            rp = 0.0
            if use_risk_premium:
                rp = risk_premium_sensitivity * h_t

            # v7.7 Tier 3: Location bias correction
            loc_bias = 0.0
            if use_loc_bias:
                if abs(loc_bias_var_coeff) > 1e-6 and theta_long_var > 1e-12:
                    loc_bias += loc_bias_var_coeff * (h_t - theta_long_var)
                if abs(loc_bias_drift_coeff) > 1e-6:
                    sign_mu = 1.0 if mu_t >= 0.0 else -1.0
                    loc_bias += loc_bias_drift_coeff * sign_mu * np.sqrt(abs(mu_t))

            # Total return
            r_t = mu_t + rp + loc_bias + e_t + jump
            # Story 1.4: Add slow drift component
            if use_dual_freq:
                r_t += mu_slow_t
            # Story 3.4: Asset-class-aware per-step return cap
            if r_t > return_cap:
                r_t = return_cap
            elif r_t < -return_cap:
                r_t = -return_cap
            cum += r_t
            cum_out[t, p] = cum

            # ================================================================
            # GARCH VARIANCE EVOLUTION (with Tier 2+3 enhancements)
            # ================================================================
            if use_garch:
                e2 = e_t * e_t
                h_t = omega + alpha * e2 + beta * h_t
                # v7.6: GJR asymmetric leverage (v7.8: with dynamic amplification)
                if use_gjr and e_t < 0.0:
                    if use_dynamic_lev:
                        # Amplify GJR γ when crash-clustering is detected
                        # neg_frac_ema > 0.5 → more negatives → amplify
                        gamma_dyn = garch_leverage * max(0.5, min(2.0,
                            1.0 + 2.0 * (neg_frac_ema - 0.5)))
                        h_t += gamma_dyn * e2
                    else:
                        h_t += garch_leverage * e2

                # v7.7 Tier 2: Variance mean-reversion (Heston)
                if use_kappa:
                    h_t = (1.0 - kappa_mean_rev) * h_t + kappa_mean_rev * theta_long_var

                # v7.7 Tier 3: Vol-of-vol noise (sigma_eta)
                if use_sigma_eta:
                    z_std = abs(e_t) / sigma_t if sigma_t > 1e-8 else 0.0
                    excess = z_std - 1.5
                    if excess > 0.0:
                        h_t += sigma_eta * excess * excess * h_t

                # v7.7 Tier 3: Regime switching on observation variance
                if use_regime_sw:
                    z_rs = abs(e_t) / sigma_t if sigma_t > 1e-8 else 0.0
                    ind_stress = 1.0 if z_rs > 2.0 else 0.0
                    p_stress_obs = (1.0 - regime_switch_prob) * p_stress_obs + regime_switch_prob * ind_stress
                    # Amplify variance by stress probability
                    h_t *= (1.0 + p_stress_obs * (np.sqrt(q_stress_ratio) - 1.0))

                # v7.7 Tier 3: VoV (gamma_vov) observation noise
                if use_gamma_vov:
                    log_h = np.log(max(h_t, 1e-12))
                    log_h_ema = 0.9 * log_h_ema + 0.1 * log_h
                    diff_lh = log_h - log_h_ema
                    log_h_var_ema = 0.9 * log_h_var_ema + 0.1 * (diff_lh * diff_lh)
                    vov_t = np.sqrt(log_h_var_ema)
                    # Damping: reduce VoV when MS-q stress is active
                    gamma_eff = gamma_vov
                    if vov_damping > 0.0 and use_ms_q:
                        # Use vol_ema-based stress proxy for damping
                        vol_z_dam = (h_t - vol_ema) / max(np.sqrt(vol_ema), 1e-8) if vol_ema > 1e-12 else 0.0
                        p_stress_dam = 1.0 / (1.0 + np.exp(-ms_sensitivity * vol_z_dam))
                        gamma_eff = gamma_vov * (1.0 - vov_damping * p_stress_dam)
                    h_t *= (1.0 + gamma_eff * vov_t)

                # v7.7 Tier 2: Rough volatility memory
                if use_rough:
                    sq_buf[t % 50] = e2
                    if t >= 1:
                        h_rough = 0.0
                        n_lags = min(t, rough_max_lag)
                        for j in range(n_lags):
                            h_rough += frac_weights[j] * sq_buf[(t - 1 - j) % 50]
                        # Blend weight: rougher H → more fractional influence
                        rw = max(0.0, 0.3 * (1.0 - 2.0 * rough_hurst))
                        h_t = (1.0 - rw) * h_t + rw * h_rough

                # v7.8 Tier 4: Liquidity-volatility feedback (Brunnermeier-Pedersen)
                if use_liq_stress:
                    vol_ratio = h_t / theta_long_var
                    if vol_ratio > 1.0:
                        excess = min(vol_ratio - 1.0, 3.0)
                        amp = min(1.0 + liq_stress_coeff * excess * excess, 2.0)
                        h_t *= amp
                    if h_t > 50.0 * theta_long_var:
                        h_t = 50.0 * theta_long_var

                # v7.8 Tier 4: Update dynamic leverage EWM
                if use_dynamic_lev:
                    neg_ind = 1.0 if e_t < 0.0 else 0.0
                    neg_frac_ema = (1.0 - leverage_dynamic_decay) * neg_frac_ema + leverage_dynamic_decay * neg_ind

                if h_t < 1e-12:
                    h_t = 1e-12
                elif h_t > h_dyn_cap:
                    h_t = h_dyn_cap

            # ================================================================
            # AR(1) DRIFT EVOLUTION (with Tier 2+3 enhancements)
            # ================================================================
            # v7.7 Tier 2: MS process noise for drift
            drift_sigma_t = drift_sigma
            if use_ms_q:
                ewm_alpha_ms = 0.05  # ~20-day half-life
                vol_ema = (1.0 - ewm_alpha_ms) * vol_ema + ewm_alpha_ms * h_t
                vol_z = (h_t - vol_ema) / max(np.sqrt(vol_ema), 1e-8) if vol_ema > 1e-12 else 0.0
                p_stress_ms = 1.0 / (1.0 + np.exp(-ms_sensitivity * vol_z))
                q_t = (1.0 - p_stress_ms) * drift_q + p_stress_ms * drift_q * q_stress_ratio
                # v7.7 Tier 3: q_vol_coupling
                if use_q_vol_coupling and theta_long_var > 1e-12:
                    q_t *= (1.0 + q_vol_coupling * max(0.0, h_t / theta_long_var - 1.0))
                drift_sigma_t = np.sqrt(q_t) if q_t > 0.0 else 0.0
            elif use_q_vol_coupling and theta_long_var > 1e-12:
                # q_vol_coupling without MS-q
                q_t = drift_q * (1.0 + q_vol_coupling * max(0.0, h_t / theta_long_var - 1.0))
                drift_sigma_t = np.sqrt(q_t) if q_t > 0.0 else 0.0

            if drift_sigma_t > 0.0:
                mu_t = phi * mu_t + drift_sigma_t * z_drift[t, p]
            else:
                mu_t = phi * mu_t

            # Story 1.4: Slow drift deterministic decay
            if use_dual_freq:
                mu_slow_t = phi_slow * mu_slow_t

            # v7.7 Tier 3: GAS dynamic skew update
            if use_gas_skew:
                z_score_gas = e_t / sigma_t if sigma_t > 1e-8 else 0.0
                # Student-t score weight
                w_gas = (nu + 1.0) / (nu + z_score_gas * z_score_gas)
                score_gas = z_score_gas * w_gas
                alpha_t = (1.0 - skew_persistence) * alpha_asym + skew_persistence * alpha_t + skew_score_sensitivity * score_gas
                # Clamp
                if alpha_t < -0.3:
                    alpha_t = -0.3
                elif alpha_t > 0.3:
                    alpha_t = 0.3


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
    # v7.6: GJR leverage per path
    gamma_per_path: np.ndarray = np.empty(0, dtype=np.float64),
    variance_inflation: float = 1.0,
    mu_drift: float = 0.0,
    # v7.7: Tier 2 + Tier 3 params (scalar — shared across paths)
    alpha_asym: float = 0.0,
    k_asym: float = 2.0,
    risk_premium_sensitivity: float = 0.0,
    kappa_mean_rev: float = 0.0,
    theta_long_var: float = 0.0,
    crps_sigma_shrinkage: float = 1.0,
    ms_sensitivity: float = 0.0,
    q_stress_ratio: float = 1.0,
    rough_hurst: float = 0.0,
    frac_weights: np.ndarray = np.empty(0, dtype=np.float64),
    sigma_eta: float = 0.0,
    t_df_asym: float = 0.0,
    regime_switch_prob: float = 0.0,
    gamma_vov: float = 0.0,
    vov_damping: float = 0.0,
    skew_score_sensitivity: float = 0.0,
    skew_persistence: float = 0.97,
    loc_bias_var_coeff: float = 0.0,
    loc_bias_drift_coeff: float = 0.0,
    q_vol_coupling: float = 0.0,
    # v7.8: Tier 4 — dynamic leverage, liquidity stress
    leverage_dynamic_decay: float = 0.0,
    liq_stress_coeff: float = 0.0,
    # Story 1.4: Dual-frequency drift propagation
    phi_slow: float = 0.0,
    mu_slow_0: float = 0.0,
    # Story 3.4: Asset-class-aware per-step return cap
    return_cap: float = 0.30,
) -> None:
    """Multi-path MC kernel with per-path parameter uncertainty.

    Like unified_mc_simulate_kernel but supports:
    - Per-path nu (tail parameter uncertainty)
    - Per-path GARCH parameters (parameter uncertainty via covariance sampling)
    - Per-path GJR leverage (v7.6)
    - All Tier 2+3 params as scalars shared across paths

    Parameters
    ----------
    nu_per_path : ndarray (n_paths,)
        Per-path degrees of freedom
    omega_per_path, alpha_per_path, beta_per_path : ndarray (n_paths,)
        Per-path GARCH parameters
    gamma_per_path : ndarray (n_paths,)
        Per-path GJR leverage (v7.6). Empty array = no leverage.
    variance_inflation : float
        Calibrated predictive variance multiplier β (scales h0)
    mu_drift : float
        Additive drift bias correction
    (other parameters same as unified_mc_simulate_kernel)
    """
    drift_sigma = np.sqrt(drift_q) if drift_q > 0.0 else 0.0
    has_gamma = len(gamma_per_path) >= n_paths
    h0_cal = h0 * variance_inflation * crps_sigma_shrinkage

    # v7.9: Dynamic GARCH variance cap (same as unified_mc_simulate_kernel)
    h_dyn_cap = 25.0 * h0_cal
    if h_dyn_cap < 0.005:
        h_dyn_cap = 0.005

    # v7.7 feature flags
    use_kappa = kappa_mean_rev > 0.001 and theta_long_var > 1e-12
    use_ms_q = ms_sensitivity > 0.01 and q_stress_ratio > 1.01
    use_rough = rough_hurst > 0.001 and len(frac_weights) > 0
    use_sigma_eta = sigma_eta > 0.005
    use_regime_sw = regime_switch_prob > 0.005
    use_gamma_vov = gamma_vov > 0.005
    use_loc_bias = abs(loc_bias_var_coeff) > 1e-6 or abs(loc_bias_drift_coeff) > 1e-6
    use_q_vol_coupling = q_vol_coupling > 0.001 and theta_long_var > 1e-12
    use_risk_premium = abs(risk_premium_sensitivity) > 1e-10
    rough_max_lag = min(len(frac_weights), 50) if use_rough else 0

    # v7.8: Tier 4 feature flags
    use_dynamic_lev = leverage_dynamic_decay > 0.01 and use_garch
    use_liq_stress = liq_stress_coeff > 0.005 and theta_long_var > 1e-12 and use_garch

    # Story 1.4: Dual-frequency drift flag
    use_dual_freq = phi_slow > 0.0 and abs(mu_slow_0) > 1e-15

    for p in range(n_paths):
        nu_p = nu_per_path[p]
        use_t_p = (nu_p > 2.0) and (nu_p < 100.0)
        use_t_df_asym_p = use_t_p and abs(t_df_asym) > 0.05
        use_asym_p = use_t_p and (abs(alpha_asym) > 1e-8)
        use_gas_skew_p = use_t_p and skew_score_sensitivity > 1e-6

        if use_t_p:
            t_var_p = nu_p / (nu_p - 2.0)
            t_scale_p = 1.0 / np.sqrt(t_var_p)
        else:
            t_scale_p = 1.0

        omega_p = omega_per_path[p]
        alpha_p = alpha_per_path[p]
        beta_p = beta_per_path[p]
        gamma_p = gamma_per_path[p] if has_gamma else 0.0

        mu_t = mu_now + mu_drift
        h_t = h0_cal
        if h_t < 1e-12:
            h_t = 1e-12
        cum = 0.0
        # Story 1.4: Slow drift component (deterministic decay)
        mu_slow_t = mu_slow_0 if use_dual_freq else 0.0

        # Per-path state variables for Tier 3
        p_stress_obs = 0.1
        alpha_t = alpha_asym
        log_h_ema = np.log(max(h_t, 1e-12))
        log_h_var_ema = 0.0
        vol_ema = h_t
        sq_buf = np.zeros(50)
        # v7.8 Tier 4: dynamic leverage EWM state
        neg_frac_ema = 0.5

        for t in range(H_max):
            sigma_t = np.sqrt(h_t)
            vol_out[t, p] = sigma_t

            # Observation noise
            if use_t_p:
                chi2_val = z_chi2[t, p]
                if chi2_val < 1e-8:
                    chi2_val = 1e-8
                raw_t = z_normals[t, p] / np.sqrt(chi2_val)

                if use_t_df_asym_p:
                    if raw_t < 0.0:
                        nu_piece = max(2.5, nu_p - t_df_asym)
                    else:
                        nu_piece = max(2.5, nu_p + t_df_asym)
                    t_var_piece = nu_piece / (nu_piece - 2.0)
                    eps = raw_t / np.sqrt(t_var_piece)
                elif use_asym_p or use_gas_skew_p:
                    a_eff = alpha_t if use_gas_skew_p else alpha_asym
                    nu_eff = nu_p * (1.0 + a_eff * np.tanh(k_asym * raw_t))
                    if nu_eff < 2.5:
                        nu_eff = 2.5
                    elif nu_eff > 200.0:
                        nu_eff = 200.0
                    t_var_eff = nu_eff / (nu_eff - 2.0)
                    eps = raw_t / np.sqrt(t_var_eff)
                else:
                    eps = raw_t * t_scale_p
            else:
                eps = z_normals[t, p]

            e_t = sigma_t * eps

            # Jump component
            jump = 0.0
            if enable_jumps and jump_intensity > 0.0:
                if z_jump_uniform[t, p] < jump_intensity:
                    jump = jump_mean + jump_std * z_jump_normal[t, p]

            # Risk premium
            rp = 0.0
            if use_risk_premium:
                rp = risk_premium_sensitivity * h_t

            # Location bias
            loc_bias = 0.0
            if use_loc_bias:
                if abs(loc_bias_var_coeff) > 1e-6 and theta_long_var > 1e-12:
                    loc_bias += loc_bias_var_coeff * (h_t - theta_long_var)
                if abs(loc_bias_drift_coeff) > 1e-6:
                    sign_mu = 1.0 if mu_t >= 0.0 else -1.0
                    loc_bias += loc_bias_drift_coeff * sign_mu * np.sqrt(abs(mu_t))

            r_t = mu_t + rp + loc_bias + e_t + jump
            # Story 1.4: Add slow drift component
            if use_dual_freq:
                r_t += mu_slow_t
            # Story 3.4: Asset-class-aware per-step return cap
            if r_t > return_cap:
                r_t = return_cap
            elif r_t < -return_cap:
                r_t = -return_cap
            cum += r_t
            cum_out[t, p] = cum

            # GJR-GARCH evolution (per-path params)
            if use_garch:
                e2 = e_t * e_t
                h_t = omega_p + alpha_p * e2 + beta_p * h_t
                if gamma_p > 1e-8 and e_t < 0.0:
                    if use_dynamic_lev:
                        gamma_dyn = gamma_p * max(0.5, min(2.0,
                            1.0 + 2.0 * (neg_frac_ema - 0.5)))
                        h_t += gamma_dyn * e2
                    else:
                        h_t += gamma_p * e2

                # Variance mean-reversion
                if use_kappa:
                    h_t = (1.0 - kappa_mean_rev) * h_t + kappa_mean_rev * theta_long_var

                # Vol-of-vol noise
                if use_sigma_eta:
                    z_std = abs(e_t) / sigma_t if sigma_t > 1e-8 else 0.0
                    excess = z_std - 1.5
                    if excess > 0.0:
                        h_t += sigma_eta * excess * excess * h_t

                # Regime switching on obs variance
                if use_regime_sw:
                    z_rs = abs(e_t) / sigma_t if sigma_t > 1e-8 else 0.0
                    ind_stress = 1.0 if z_rs > 2.0 else 0.0
                    p_stress_obs = (1.0 - regime_switch_prob) * p_stress_obs + regime_switch_prob * ind_stress
                    h_t *= (1.0 + p_stress_obs * (np.sqrt(q_stress_ratio) - 1.0))

                # VoV observation noise
                if use_gamma_vov:
                    log_h = np.log(max(h_t, 1e-12))
                    log_h_ema = 0.9 * log_h_ema + 0.1 * log_h
                    diff_lh = log_h - log_h_ema
                    log_h_var_ema = 0.9 * log_h_var_ema + 0.1 * (diff_lh * diff_lh)
                    vov_t = np.sqrt(log_h_var_ema)
                    gamma_eff = gamma_vov
                    if vov_damping > 0.0 and use_ms_q:
                        vol_z_dam = (h_t - vol_ema) / max(np.sqrt(vol_ema), 1e-8) if vol_ema > 1e-12 else 0.0
                        p_stress_dam = 1.0 / (1.0 + np.exp(-ms_sensitivity * vol_z_dam))
                        gamma_eff = gamma_vov * (1.0 - vov_damping * p_stress_dam)
                    h_t *= (1.0 + gamma_eff * vov_t)

                # Rough volatility memory
                if use_rough:
                    sq_buf[t % 50] = e2
                    if t >= 1:
                        h_rough = 0.0
                        n_lags = min(t, rough_max_lag)
                        for j in range(n_lags):
                            h_rough += frac_weights[j] * sq_buf[(t - 1 - j) % 50]
                        rw = max(0.0, 0.3 * (1.0 - 2.0 * rough_hurst))
                        h_t = (1.0 - rw) * h_t + rw * h_rough

                # v7.8 Tier 4: Liquidity-volatility feedback
                if use_liq_stress:
                    vol_ratio = h_t / theta_long_var
                    if vol_ratio > 1.0:
                        excess = min(vol_ratio - 1.0, 3.0)
                        amp = min(1.0 + liq_stress_coeff * excess * excess, 2.0)
                        h_t *= amp
                    if h_t > 50.0 * theta_long_var:
                        h_t = 50.0 * theta_long_var

                # v7.8 Tier 4: Update dynamic leverage EWM
                if use_dynamic_lev:
                    neg_ind = 1.0 if e_t < 0.0 else 0.0
                    neg_frac_ema = (1.0 - leverage_dynamic_decay) * neg_frac_ema + leverage_dynamic_decay * neg_ind

                if h_t < 1e-12:
                    h_t = 1e-12
                elif h_t > h_dyn_cap:
                    h_t = h_dyn_cap

            # AR(1) drift with MS process noise
            drift_sigma_t = drift_sigma
            if use_ms_q:
                ewm_alpha_ms = 0.05
                vol_ema = (1.0 - ewm_alpha_ms) * vol_ema + ewm_alpha_ms * h_t
                vol_z = (h_t - vol_ema) / max(np.sqrt(vol_ema), 1e-8) if vol_ema > 1e-12 else 0.0
                p_stress_ms = 1.0 / (1.0 + np.exp(-ms_sensitivity * vol_z))
                q_t = (1.0 - p_stress_ms) * drift_q + p_stress_ms * drift_q * q_stress_ratio
                if use_q_vol_coupling and theta_long_var > 1e-12:
                    q_t *= (1.0 + q_vol_coupling * max(0.0, h_t / theta_long_var - 1.0))
                drift_sigma_t = np.sqrt(q_t) if q_t > 0.0 else 0.0
            elif use_q_vol_coupling and theta_long_var > 1e-12:
                q_t = drift_q * (1.0 + q_vol_coupling * max(0.0, h_t / theta_long_var - 1.0))
                drift_sigma_t = np.sqrt(q_t) if q_t > 0.0 else 0.0

            if drift_sigma_t > 0.0:
                mu_t = phi * mu_t + drift_sigma_t * z_drift[t, p]
            else:
                mu_t = phi * mu_t

            # Story 1.4: Slow drift deterministic decay
            if use_dual_freq:
                mu_slow_t = phi_slow * mu_slow_t

            # GAS dynamic skew update
            if use_gas_skew_p:
                z_score_gas = e_t / sigma_t if sigma_t > 1e-8 else 0.0
                w_gas = (nu_p + 1.0) / (nu_p + z_score_gas * z_score_gas)
                score_gas = z_score_gas * w_gas
                alpha_t = (1.0 - skew_persistence) * alpha_asym + skew_persistence * alpha_t + skew_score_sensitivity * score_gas
                if alpha_t < -0.3:
                    alpha_t = -0.3
                elif alpha_t > 0.3:
                    alpha_t = 0.3


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
# REGIME-CONDITIONAL c KERNELS (Tune.md Story 2.1)
# =============================================================================
# Time-varying observation noise: R_t = c_t * vol_t^2  where c_t = c[regime_t]
# In trending regimes, drift explains more variance -> smaller c.
# In crisis regimes, everything is noise -> larger c.
# =============================================================================


@njit(cache=True, fastmath=True)
def build_c_array_from_regimes(
    regime_labels: np.ndarray,
    c_per_regime: np.ndarray,
) -> np.ndarray:
    """
    Expand per-regime c values into a time-varying c_t array.

    Parameters
    ----------
    regime_labels : array of int, shape (T,)
        Regime label for each time step (0-4).
    c_per_regime : array of float, shape (5,)
        c value for each regime (indexed by regime id).

    Returns
    -------
    c_array : array of float, shape (T,)
        Time-varying c_t = c_per_regime[regime_labels[t]].
    """
    n = len(regime_labels)
    c_array = np.empty(n, dtype=np.float64)
    for t in range(n):
        r = regime_labels[t]
        if r < 0 or r >= len(c_per_regime):
            c_array[t] = 1.0  # fallback
        else:
            c_array[t] = c_per_regime[r]
    return c_array


@njit(cache=True, fastmath=True)
def regime_c_gaussian_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c_array: np.ndarray,
    phi: float,
    P0: float = 1e-4,
) -> tuple:
    """
    phi-Gaussian Kalman filter with time-varying c_t (regime-conditional).

    Same as phi_gaussian_filter_kernel but R_t = c_array[t] * vol_t^2.

    Parameters
    ----------
    c_array : array of float, shape (T,)
        Time-varying observation noise scale.

    Returns
    -------
    (mu_filtered, P_filtered, log_likelihood)
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi

    for t in range(n):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q

        vol_t = vol[t]
        R = c_array[t] * (vol_t * vol_t)
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
def regime_c_student_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c_array: np.ndarray,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    P0: float = 1e-4,
) -> tuple:
    """
    phi-Student-t Kalman filter with time-varying c_t (regime-conditional).

    Same as phi_student_t_filter_kernel but R_t = c_array[t] * vol_t^2.

    Returns
    -------
    (mu_filtered, P_filtered, log_likelihood)
    """
    n = len(returns)
    mu = 0.0
    P = P0
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    phi_sq = phi * phi

    log_nu = np.log(nu)
    log_pi = np.log(np.pi)
    log_norm = (log_gamma_half_nu_plus_half
                - log_gamma_half_nu
                - 0.5 * (log_nu + log_pi))

    for t in range(n):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q

        vol_t = vol[t]
        R = c_array[t] * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R

        if S > _MIN_VARIANCE:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred

            m_dist_sq = (innovation * innovation) / S
            ll_contrib = (log_norm
                          - 0.5 * np.log(S)
                          - 0.5 * (nu + 1.0) * np.log(1.0 + m_dist_sq / nu))
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
# RV-ADAPTIVE PROCESS NOISE KERNELS (Tune.md Story 1.1)
# =============================================================================
# Proactive process noise q_t that scales with realized volatility acceleration:
#   q_t = q_base * exp(gamma * delta_log_vol_sq)
# Unlike GAS-Q (reactive to errors), RV-Q shifts BEFORE errors materialize
# because realized vol expansion is observable in real time via OHLC data.
# =============================================================================

@njit(cache=True, fastmath=False)
def rv_adaptive_q_kernel(
    vol: np.ndarray,
    q_base: float,
    gamma: float,
    q_min: float = 1e-8,
    q_max: float = 1e-2,
) -> np.ndarray:
    """
    Compute time-varying process noise q_t from realized volatility changes.

    q_t = q_base * exp(gamma * delta_log(vol_t^2))

    Parameters
    ----------
    vol : np.ndarray
        Realized volatility array (e.g. from Garman-Klass), length T.
    q_base : float
        Baseline process noise (unconditional level).
    gamma : float
        Sensitivity to vol acceleration. gamma=0 recovers static q.
    q_min : float
        Floor to prevent filter freeze (default 1e-8).
    q_max : float
        Ceiling to prevent filter divergence (default 1e-2).

    Returns
    -------
    q_path : np.ndarray
        Time-varying process noise, length T.
    """
    n = len(vol)
    q_path = np.empty(n, dtype=np.float64)

    # First timestep: no delta available, use q_base
    q_path[0] = q_base
    if q_path[0] < q_min:
        q_path[0] = q_min
    elif q_path[0] > q_max:
        q_path[0] = q_max

    for t in range(1, n):
        vol_curr = vol[t]
        vol_prev = vol[t - 1]

        # Compute delta log(vol^2) = 2 * delta log(vol)
        # Guard against zero or negative vol
        if vol_curr > 1e-15 and vol_prev > 1e-15:
            delta_log_vol_sq = 2.0 * (np.log(vol_curr) - np.log(vol_prev))
        else:
            delta_log_vol_sq = 0.0

        # q_t = q_base * exp(gamma * delta)
        exponent = gamma * delta_log_vol_sq
        # Clamp exponent to prevent overflow
        if exponent > 20.0:
            exponent = 20.0
        elif exponent < -20.0:
            exponent = -20.0

        q_t = q_base * np.exp(exponent)

        # Enforce bounds
        if q_t < q_min:
            q_t = q_min
        elif q_t > q_max:
            q_t = q_max

        q_path[t] = q_t

    return q_path


@njit(cache=True, fastmath=True)
def rv_adaptive_q_gaussian_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q_base: float,
    gamma: float,
    c: float,
    phi: float,
    q_min: float = 1e-8,
    q_max: float = 1e-2,
    P0: float = 1e-4,
    mu_filtered: np.ndarray = np.empty(0),
    P_filtered: np.ndarray = np.empty(0),
    q_path: np.ndarray = np.empty(0),
) -> float:
    """
    phi-Gaussian Kalman filter with RV-adaptive process noise.

    State:       mu_t = phi * mu_{t-1} + w_t,  w_t ~ N(0, q_t)
    Observation: r_t  = mu_t + eps_t,           eps_t ~ N(0, c * vol_t^2)

    q_t = q_base * exp(gamma * delta_log(vol_t^2))

    Parameters
    ----------
    returns, vol : np.ndarray
        Observations and realized volatility.
    q_base : float
        Baseline process noise.
    gamma : float
        RV-feedback sensitivity. gamma=0 recovers static q.
    c, phi : float
        Observation noise scaling and AR(1) persistence.
    mu_filtered, P_filtered, q_path : np.ndarray
        Pre-allocated output arrays (length T). Written in-place.

    Returns
    -------
    log_likelihood : float
    """
    n = len(returns)
    mu = 0.0
    P = P0
    log_ll = 0.0
    phi_sq = phi * phi
    log_2pi = 1.8378770664093453
    _EPS = 1e-12

    # Compute first q value
    q_t = q_base
    if q_t < q_min:
        q_t = q_min
    elif q_t > q_max:
        q_t = q_max

    for t in range(n):
        # Compute adaptive q_t from vol changes
        if t > 0:
            vol_curr = vol[t]
            vol_prev = vol[t - 1]
            if vol_curr > 1e-15 and vol_prev > 1e-15:
                delta_log_vol_sq = 2.0 * (np.log(vol_curr) - np.log(vol_prev))
            else:
                delta_log_vol_sq = 0.0
            exponent = gamma * delta_log_vol_sq
            if exponent > 20.0:
                exponent = 20.0
            elif exponent < -20.0:
                exponent = -20.0
            q_t = q_base * np.exp(exponent)
            if q_t < q_min:
                q_t = q_min
            elif q_t > q_max:
                q_t = q_max

        if len(q_path) > 0:
            q_path[t] = q_t

        # Predict step with AR(1) and adaptive q
        mu_pred = phi * mu
        P_pred = phi_sq * P + q_t

        # Innovation
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R

        if S > _EPS:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < _EPS:
                P = _EPS

            z_sq = (innovation * innovation) / S
            if z_sq > 100.0:
                z_sq = 100.0
            ll_t = -0.5 * (log_2pi + np.log(S) + z_sq)
            if ll_t == ll_t:
                log_ll += ll_t
        else:
            mu = mu_pred
            P = P_pred

        if len(mu_filtered) > 0:
            mu_filtered[t] = mu
        if len(P_filtered) > 0:
            P_filtered[t] = P if P > _EPS else _EPS

    return log_ll


@njit(cache=True, fastmath=True)
def rv_adaptive_q_gaussian_filter_precomputed_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    delta_log_vol_sq: np.ndarray,
    q_base: float,
    gamma: float,
    c: float,
    phi: float,
    q_min: float = 1e-8,
    q_max: float = 1e-2,
    P0: float = 1e-4,
    mu_filtered: np.ndarray = np.empty(0),
    P_filtered: np.ndarray = np.empty(0),
    q_path: np.ndarray = np.empty(0),
) -> float:
    """RV-Q Gaussian filter using precomputed vol_sq and delta_log(vol^2)."""
    n = len(returns)
    mu = 0.0
    P = P0
    log_ll = 0.0
    phi_sq = phi * phi
    log_2pi = 1.8378770664093453
    _EPS = 1e-12

    q_t = q_base
    if q_t < q_min:
        q_t = q_min
    elif q_t > q_max:
        q_t = q_max

    for t in range(n):
        if t > 0:
            exponent = gamma * delta_log_vol_sq[t]
            if exponent > 20.0:
                exponent = 20.0
            elif exponent < -20.0:
                exponent = -20.0
            q_t = q_base * np.exp(exponent)
            if q_t < q_min:
                q_t = q_min
            elif q_t > q_max:
                q_t = q_max

        if len(q_path) > 0:
            q_path[t] = q_t

        mu_pred = phi * mu
        P_pred = phi_sq * P + q_t
        R = c * vol_sq[t]
        innovation = returns[t] - mu_pred
        S = P_pred + R

        if S > _EPS:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < _EPS:
                P = _EPS

            z_sq = (innovation * innovation) / S
            if z_sq > 100.0:
                z_sq = 100.0
            ll_t = -0.5 * (log_2pi + np.log(S) + z_sq)
            if ll_t == ll_t:
                log_ll += ll_t
        else:
            mu = mu_pred
            P = P_pred

        if len(mu_filtered) > 0:
            mu_filtered[t] = mu
        if len(P_filtered) > 0:
            P_filtered[t] = P if P > _EPS else _EPS

    return log_ll


@njit(cache=True, fastmath=True)
def rv_adaptive_q_student_t_filter_kernel(
    returns: np.ndarray,
    vol: np.ndarray,
    q_base: float,
    gamma: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    q_min: float = 1e-8,
    q_max: float = 1e-2,
    P0: float = 1e-4,
    mu_filtered: np.ndarray = np.empty(0),
    P_filtered: np.ndarray = np.empty(0),
    q_path: np.ndarray = np.empty(0),
) -> float:
    """
    phi-Student-t Kalman filter with RV-adaptive process noise.

    State:       mu_t = phi * mu_{t-1} + w_t,  w_t ~ N(0, q_t)
    Observation: r_t  = mu_t + eps_t,           eps_t ~ t_nu(0, c * vol_t^2)

    q_t = q_base * exp(gamma * delta_log(vol_t^2))

    Returns
    -------
    log_likelihood : float
    """
    n = len(returns)

    # Data-adaptive initialization (same as phi_student_t_filter_kernel)
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

    log_ll = 0.0
    phi_sq = phi * phi
    _EPS = 1e-12

    # First q value
    q_t = q_base
    if q_t < q_min:
        q_t = q_min
    elif q_t > q_max:
        q_t = q_max

    for t in range(n):
        # Compute adaptive q_t from vol changes
        if t > 0:
            vol_curr = vol[t]
            vol_prev = vol[t - 1]
            if vol_curr > 1e-15 and vol_prev > 1e-15:
                delta_log_vol_sq = 2.0 * (np.log(vol_curr) - np.log(vol_prev))
            else:
                delta_log_vol_sq = 0.0
            exponent = gamma * delta_log_vol_sq
            if exponent > 20.0:
                exponent = 20.0
            elif exponent < -20.0:
                exponent = -20.0
            q_t = q_base * np.exp(exponent)
            if q_t < q_min:
                q_t = q_min
            elif q_t > q_max:
                q_t = q_max

        if len(q_path) > 0:
            q_path[t] = q_t

        # Predict step with AR(1) and adaptive q
        mu_pred = phi * mu
        P_pred = phi_sq * P + q_t

        # Observation variance
        vol_t = vol[t]
        R = c * (vol_t * vol_t)
        innovation = returns[t] - mu_pred
        S = P_pred + R

        if S > _EPS:
            forecast_scale = np.sqrt(S)

            # Student-t log-likelihood
            ll_t = student_t_logpdf_kernel(
                returns[t], nu, mu_pred, forecast_scale,
                log_gamma_half_nu, log_gamma_half_nu_plus_half
            )
            if ll_t < -100.0:
                ll_t = -100.0
            log_ll += ll_t

            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            mu = mu_pred
            P = P_pred

        if len(mu_filtered) > 0:
            mu_filtered[t] = mu
        if len(P_filtered) > 0:
            P_filtered[t] = P if P > _EPS else _EPS

    return log_ll


@njit(cache=True, fastmath=True)
def rv_adaptive_q_student_t_filter_precomputed_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    delta_log_vol_sq: np.ndarray,
    q_base: float,
    gamma: float,
    c: float,
    phi: float,
    nu: float,
    log_gamma_half_nu: float,
    log_gamma_half_nu_plus_half: float,
    q_min: float = 1e-8,
    q_max: float = 1e-2,
    P0: float = 1e-4,
    mu_filtered: np.ndarray = np.empty(0),
    P_filtered: np.ndarray = np.empty(0),
    q_path: np.ndarray = np.empty(0),
) -> float:
    """RV-Q Student-t filter using precomputed vol_sq and delta_log(vol^2)."""
    n = len(returns)

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

    log_ll = 0.0
    phi_sq = phi * phi
    _EPS = 1e-12

    q_t = q_base
    if q_t < q_min:
        q_t = q_min
    elif q_t > q_max:
        q_t = q_max

    for t in range(n):
        if t > 0:
            exponent = gamma * delta_log_vol_sq[t]
            if exponent > 20.0:
                exponent = 20.0
            elif exponent < -20.0:
                exponent = -20.0
            q_t = q_base * np.exp(exponent)
            if q_t < q_min:
                q_t = q_min
            elif q_t > q_max:
                q_t = q_max

        if len(q_path) > 0:
            q_path[t] = q_t

        mu_pred = phi * mu
        P_pred = phi_sq * P + q_t
        R = c * vol_sq[t]
        innovation = returns[t] - mu_pred
        S = P_pred + R

        if S > _EPS:
            forecast_scale = np.sqrt(S)
            ll_t = student_t_logpdf_kernel(
                returns[t], nu, mu_pred, forecast_scale,
                log_gamma_half_nu, log_gamma_half_nu_plus_half
            )
            if ll_t < -100.0:
                ll_t = -100.0
            log_ll += ll_t

            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            mu = mu_pred
            P = P_pred

        if len(mu_filtered) > 0:
            mu_filtered[t] = mu
        if len(P_filtered) > 0:
            P_filtered[t] = P if P > _EPS else _EPS

    return log_ll


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
        z_sq_cv = (inn * inn) / S
        scale = np.sqrt(S * nu_scale)
        if scale > 1e-12:
            z = inn / scale
            ll_t = log_norm_const - np.log(scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
            if ll_t == ll_t:  # isfinite check in Numba
                ll_fold += ll_t

        # Robust Kalman gain (Meinhold & Singpurwalla 1989)
        K = P_p / S
        w_cv = nu_p1 / (nu_val + z_sq_cv)
        mu_p = mu_p + K * w_cv * inn
        P_p = (1.0 - w_cv * K) * P_p
        if P_p < 1e-12:
            P_p = 1e-12

    return ll_fold


@njit(cache=True, fastmath=False)
def phi_student_t_cv_test_fold_stats_kernel(
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
) -> tuple:
    """CV fold likelihood plus capped innovation² / variance diagnostics."""
    mu_p = mu_init
    P_p = P_init
    ll_fold = 0.0
    obs_count = 0
    z2_sum = 0.0
    z2_cap = 50.0
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
        z_sq_cv = (inn * inn) / S
        scale = np.sqrt(S * nu_scale)
        if scale > 1e-12:
            z = inn / scale
            ll_t = log_norm_const - np.log(scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
            if ll_t == ll_t:
                ll_fold += ll_t
                obs_count += 1
                if z_sq_cv == z_sq_cv:
                    z2_sum += z_sq_cv if z_sq_cv < z2_cap else z2_cap

        K = P_p / S
        w_cv = nu_p1 / (nu_val + z_sq_cv)
        mu_p = mu_p + K * w_cv * inn
        P_p = (1.0 - w_cv * K) * P_p
        if P_p < 1e-12:
            P_p = 1e-12

    return ll_fold, obs_count, z2_sum


@njit(cache=True, fastmath=False)
def phi_student_t_train_state_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    train_start: int,
    train_end: int,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    use_vov: int,
    robust_wt: int,
    online_scale_adapt: int,
) -> tuple:
    """
    Terminal-state φ-Student-t training-fold filter.

    Same state/update math as phi_student_t_enhanced_filter_kernel, but avoids
    allocating mu/P/predictive arrays when CV only needs the last state.
    """
    n = train_end - train_start
    if n <= 0:
        return 0.0, 1e-4, 0.0

    phi_sq = phi * phi
    nu_scale = (nu - 2.0) / nu if nu > 2.0 else 1.0

    init_w = 20 if n > 20 else n
    if init_w >= 3:
        init_vals = np.empty(init_w)
        mean_init = 0.0
        for i in range(init_w):
            val = returns[train_start + i]
            init_vals[i] = val
            mean_init += val
        init_vals.sort()
        mid = init_w // 2
        if init_w % 2 == 1:
            mu = init_vals[mid]
        else:
            mu = (init_vals[mid - 1] + init_vals[mid]) * 0.5
        mean_init /= init_w
        var_init = 0.0
        for i in range(init_w):
            diff = returns[train_start + i] - mean_init
            var_init += diff * diff
        var_init /= init_w
        P = var_init if var_init > 1e-6 else 1e-6
    else:
        mu = 0.0
        P = 1e-4

    chi2_tgt = nu / (nu - 2.0) if nu > 2.0 else 1.0
    chi2_lam = 0.98
    chi2_1m = 1.0 - chi2_lam
    chi2_cap = chi2_tgt * 50.0
    ewm_z2 = chi2_tgt
    c_adj = 1.0
    log_likelihood = 0.0
    nu_p1 = nu + 1.0

    for t in range(train_start, train_end):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q

        c_eff = c * c_adj if online_scale_adapt == 1 else c
        R_t = c_eff * vol_sq[t]
        if use_vov == 1:
            R_t *= (1.0 + gamma_vov * vov_rolling[t])

        S = P_pred + R_t
        if S < 1e-12:
            S = 1e-12

        innovation = returns[t] - mu_pred
        K = P_pred / S

        if robust_wt == 1:
            z_sq_update = (innovation * innovation) / S
            w_t = nu_p1 / (nu + z_sq_update)
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
        else:
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred

        if P < 1e-12:
            P = 1e-12

        forecast_scale = np.sqrt(S * nu_scale)
        if forecast_scale > 1e-12:
            z = innovation / forecast_scale
            ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
            if ll_t == ll_t:
                log_likelihood += ll_t

            if online_scale_adapt == 1:
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

    return mu, P, log_likelihood


@njit(cache=True, fastmath=False)
def phi_student_t_improved_train_state_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    scale_factor: float,
    train_start: int,
    train_end: int,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    use_vov: int,
    online_scale_adapt: int,
    p_min: float,
    p_max_default: float,
) -> tuple:
    """
    Terminal-state filter for phi_student_t_improved training folds.

    Mirrors ImprovedPhiStudentTDriftModel._filter_phi_core for the optimizer
    case: no exogenous input, robust Student-t precision update, optional OSA,
    optional VoV, and Joseph-positive covariance update.
    """
    n = train_end - train_start
    if n <= 0:
        return 0.0, 1e-4, 0.0

    vol_sum = 0.0
    for t in range(train_start, train_end):
        vol_sum += vol_sq[t]
    vol_mean = vol_sum / n
    vol_var_med = vol_mean
    # For small training folds, a mean floor is close enough to seed p_cap;
    # the state path still uses exact per-timestep vol_sq.
    if n > 2:
        tmp = np.empty(n)
        for i in range(n):
            tmp[i] = vol_sq[train_start + i]
        tmp.sort()
        mid = n // 2
        if n % 2 == 1:
            vol_var_med = tmp[mid]
        else:
            vol_var_med = 0.5 * (tmp[mid - 1] + tmp[mid])
    if vol_var_med < 1e-12:
        vol_var_med = 1e-12

    p_floor = p_min if p_min > 1e-12 else 1e-12
    p_cap = p_max_default
    if 100.0 * vol_var_med > p_cap:
        p_cap = 100.0 * vol_var_med
    if 1000.0 * q > p_cap:
        p_cap = 1000.0 * q
    if p_floor * 10.0 > p_cap:
        p_cap = p_floor * 10.0

    phi_sq = phi * phi
    chi2_tgt = nu / (nu - 2.0) if nu > 2.0 else 1.0
    chi2_lam = 0.985
    chi2_1m = 1.0 - chi2_lam
    chi2_cap = chi2_tgt * 50.0
    ewm_z2 = chi2_tgt
    c_adj = 1.0
    osa_strength = (chi2_tgt - 1.0) / 0.5 if nu > 2.0 else 1.0
    if osa_strength > 1.0:
        osa_strength = 1.0
    elif osa_strength < 0.0:
        osa_strength = 0.0

    mu = 0.0
    P = vol_var_med
    if 10.0 * q > P:
        P = 10.0 * q
    if P < p_floor:
        P = p_floor
    if P > p_cap:
        P = p_cap
    log_likelihood = 0.0
    nu_p1 = nu + 1.0

    for t in range(train_start, train_end):
        mu_pred = phi * mu
        P_pred = phi_sq * P + q
        if P_pred < p_floor:
            P_pred = p_floor

        c_eff = c * c_adj if online_scale_adapt == 1 else c
        R_t = c_eff * vol_sq[t]
        if R_t < 1e-20:
            R_t = 1e-20
        if use_vov == 1:
            vov_mult = 1.0 + gamma_vov * vov_rolling[t]
            if vov_mult < 0.05:
                vov_mult = 0.05
            elif vov_mult > 20.0:
                vov_mult = 20.0
            R_t *= vov_mult
            if R_t < 1e-20:
                R_t = 1e-20

        S = P_pred + R_t
        if S < 1e-20:
            S = 1e-20
        forecast_scale = np.sqrt(S * scale_factor)
        if forecast_scale < 1e-10:
            forecast_scale = 1e-10

        innovation = returns[t] - mu_pred
        z = innovation / forecast_scale
        ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log1p((z * z) * inv_nu)
        if ll_t == ll_t:
            log_likelihood += ll_t

        z_sq_s = (innovation * innovation) / S
        w_t = nu_p1 / (nu + z_sq_s)
        if w_t < 0.05:
            w_t = 0.05
        elif w_t > 20.0:
            w_t = 20.0
        R_eff = R_t / w_t
        S_eff = P_pred + R_eff
        if S_eff < 1e-20:
            S_eff = 1e-20
        K = P_pred / S_eff
        one_minus_K = 1.0 - K
        mu = mu_pred + K * innovation
        P = one_minus_K * one_minus_K * P_pred + K * K * R_eff
        if P < p_floor:
            P = p_floor
        elif P > p_cap:
            P = p_cap

        if online_scale_adapt == 1:
            z2w = z * z
            if z2w > chi2_cap:
                z2w = chi2_cap
            ewm_z2 = chi2_lam * ewm_z2 + chi2_1m * z2w
            ratio = ewm_z2 / chi2_tgt if chi2_tgt > 1e-12 else 1.0
            if ratio < 0.35:
                ratio = 0.35
            elif ratio > 2.85:
                ratio = 2.85
            dev = ratio - 1.0 if ratio >= 1.0 else 1.0 - ratio
            if dev < 0.04:
                c_adj = 1.0
            else:
                c_adj = 1.0 + osa_strength * (np.sqrt(ratio) - 1.0)
                if c_adj < 0.4:
                    c_adj = 0.4
                elif c_adj > 2.5:
                    c_adj = 2.5

    return mu, P, log_likelihood


@njit(cache=True, fastmath=False)
def phi_student_t_improved_cv_test_fold_kernel(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    scale_factor: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    gamma_vov: float,
    vov_rolling: np.ndarray,
    use_vov: int,
    online_scale_adapt: int,
    p_floor: float,
    p_cap: float,
    z2_cap: float,
) -> tuple:
    """
    Validation-fold scorer for phi_student_t_improved.

    Mirrors the Python loop inside PhiStudentTDriftModel.optimize_params_fixed_nu:
    Student-t predictive log score, precision-weighted Joseph covariance update,
    optional VoV inflation, and graduated online scale adaptation.  The kernel
    returns the sufficient statistics used by the optimizer objective.
    """
    mu_p = mu_init
    P_p = P_init
    total_ll = 0.0
    obs_count = 0
    z2_count = 0
    z2_sum = 0.0

    phi_sq = phi * phi
    nu_p1 = nu + 1.0
    chi2_tgt = nu / (nu - 2.0) if nu > 2.0 else 1.0
    chi2_lam = 0.985
    chi2_1m = 1.0 - chi2_lam
    chi2_cap = chi2_tgt * 50.0
    ewm_z2 = chi2_tgt
    c_adj = 1.0
    osa_strength = (chi2_tgt - 1.0) / 0.5 if nu > 2.0 else 1.0
    if osa_strength > 1.0:
        osa_strength = 1.0
    elif osa_strength < 0.0:
        osa_strength = 0.0

    if p_floor < 1e-12:
        p_floor = 1e-12
    if p_cap < p_floor * 10.0:
        p_cap = p_floor * 10.0
    if z2_cap <= 0.0:
        z2_cap = 50.0

    for t in range(test_start, test_end):
        mu_pred = phi * mu_p
        P_pred = phi_sq * P_p + q
        if P_pred < p_floor:
            P_pred = p_floor

        c_eff = c * c_adj
        R_t = c_eff * vol_sq[t]
        if R_t < 1e-20:
            R_t = 1e-20
        if use_vov == 1:
            vov_mult = 1.0 + gamma_vov * vov_rolling[t]
            if vov_mult < 0.05:
                vov_mult = 0.05
            R_t *= vov_mult

        S = P_pred + R_t
        if S < 1e-20:
            S = 1e-20
        scale = np.sqrt(S * scale_factor)
        if scale < 1e-10:
            scale = 1e-10

        inn = returns[t] - mu_pred
        z = inn / scale
        ll_t = log_norm_const - np.log(scale) + neg_exp * np.log1p((z * z) * inv_nu)
        if np.isfinite(ll_t):
            total_ll += ll_t
            obs_count += 1

        z_sq_s = (inn * inn) / S
        if np.isfinite(z_sq_s):
            if z_sq_s < z2_cap:
                z2_sum += z_sq_s
            else:
                z2_sum += z2_cap
            z2_count += 1

        w_t = nu_p1 / (nu + z_sq_s)
        if w_t < 0.05:
            w_t = 0.05
        elif w_t > 20.0:
            w_t = 20.0
        R_eff = R_t / w_t
        if R_eff < 1e-20:
            R_eff = 1e-20
        S_eff = P_pred + R_eff
        if S_eff < 1e-20:
            S_eff = 1e-20
        K = P_pred / S_eff
        one_minus_K = 1.0 - K
        mu_p = mu_pred + K * inn
        P_p = one_minus_K * one_minus_K * P_pred + K * K * R_eff
        if P_p < p_floor:
            P_p = p_floor
        elif P_p > p_cap:
            P_p = p_cap

        if online_scale_adapt == 1:
            z2w = z * z
            if z2w > chi2_cap:
                z2w = chi2_cap
            ewm_z2 = chi2_lam * ewm_z2 + chi2_1m * z2w
            ratio = ewm_z2 / chi2_tgt if chi2_tgt > 1e-12 else 1.0
            if ratio < 0.35:
                ratio = 0.35
            elif ratio > 2.85:
                ratio = 2.85
            dev = ratio - 1.0 if ratio >= 1.0 else 1.0 - ratio
            if dev < 0.04:
                c_adj = 1.0
            else:
                c_adj = 1.0 + osa_strength * (np.sqrt(ratio) - 1.0)
                if c_adj < 0.4:
                    c_adj = 0.4
                elif c_adj > 2.5:
                    c_adj = 2.5

    return total_ll, obs_count, z2_count, z2_sum


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


# =============================================================================
# ONLINE c UPDATE KERNEL (Tune.md Story 2.3)
# =============================================================================
# Innovation-variance monitoring for adaptive observation noise scalar c.
#
# Update rule:
#   ratio_t = v_t^2 / R_t              (innovation variance ratio)
#   c_{t+1} = c_t + eta_t * (ratio_t - 1)
#   eta_t   = max(eta_min, eta_init * decay^t)
#
# When calibrated, E[v_t^2 / R_t] = 1 (innovations are consistent with R_t).
# If ratio > 1 persistently, c is too low -> increase.
# If ratio < 1 persistently, c is too high -> decrease.
# =============================================================================

@njit(cache=True, fastmath=True)
def online_c_update_kernel(
    innovations,       # float64[N] : v_t = r_t - mu_pred_t
    vol_sq,            # float64[N] : sigma_t^2 (EWMA variance at each step)
    c_init,            # float64    : initial c value from MLE
    eta_init,          # float64    : initial learning rate
    eta_min,           # float64    : minimum learning rate floor
    eta_decay,         # float64    : multiplicative decay per step (e.g. 0.999)
    c_min,             # float64    : lower bound for c
    c_max,             # float64    : upper bound for c
):
    """
    Online c update via exponential smoothing of innovation variance ratio.

    Returns:
        c_path : float64[N] - the c value at each time step
        eta_path : float64[N] - the learning rate at each time step
        ratio_ema : float64 - final EMA of innovation variance ratio
    """
    N = len(innovations)
    c_path = np.empty(N, dtype=np.float64)
    eta_path = np.empty(N, dtype=np.float64)

    c_t = c_init
    eta_t = eta_init
    # EMA of ratio for diagnostics (smooth estimate of E[v^2/R])
    ratio_ema = 1.0
    ema_alpha = 0.05  # EMA smoothing for ratio tracking

    for t in range(N):
        v = innovations[t]
        vsq = vol_sq[t]

        # Current observation variance using current c
        R_t = c_t * vsq
        if R_t < 1e-20:
            # Degenerate: skip update
            c_path[t] = c_t
            eta_path[t] = eta_t
            eta_t = max(eta_min, eta_t * eta_decay)
            continue

        # Innovation variance ratio
        ratio_t = (v * v) / R_t

        # Clip extreme ratios to prevent jumps (winsorize at [0.01, 100])
        if ratio_t > 100.0:
            ratio_t = 100.0
        if ratio_t < 0.01:
            ratio_t = 0.01

        # Update EMA of ratio
        ratio_ema = (1.0 - ema_alpha) * ratio_ema + ema_alpha * ratio_t

        # c update: move toward making ratio = 1
        c_new = c_t + eta_t * (ratio_t - 1.0)

        # Enforce bounds
        if c_new < c_min:
            c_new = c_min
        if c_new > c_max:
            c_new = c_max

        c_path[t] = c_new
        eta_path[t] = eta_t

        c_t = c_new
        eta_t = max(eta_min, eta_t * eta_decay)

    return c_path, eta_path, ratio_ema
