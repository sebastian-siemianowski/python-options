"""
Story 4.1: Leave-One-Out CRPS Computation per Model
====================================================

Numba-accelerated LOO-CRPS computation for Gaussian and Student-t predictive
distributions. Produces per-observation CRPS vectors used for CRPS stacking
in Story 4.2.

LOO here means: at each time step t, the model's one-step-ahead predictive
distribution F_t is evaluated against the realized return r_t. This gives
the matrix CRPS_{m,t} needed for stacking weights.

Formulas:
  Gaussian: CRPS(N(mu,sigma^2), y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
  Student-t: CRPS(t_nu(mu,sigma), y) = sigma * [z*(2*T_nu(z)-1) + 2*t_nu(z)*(nu+z^2)/(nu-1) - g(nu)]

Both compiled with @njit(cache=True) for < 10ms on 1000-step series.
"""
import os
import sys
import math
import numpy as np
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT_PI = math.sqrt(math.pi)
_INV_SQRT_PI = 1.0 / _SQRT_PI
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Gaussian helpers (pure Numba)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _norm_cdf(x):
    """Standard normal CDF via erfc."""
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


@njit(cache=True, fastmath=True)
def _norm_pdf(x):
    """Standard normal PDF."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


# ---------------------------------------------------------------------------
# Student-t helpers (pure Numba)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _log_gamma(x):
    """Lanczos approximation to log-gamma (|error| < 2e-10)."""
    g = 7.0
    c = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x)
    x -= 1.0
    a = c[0]
    t = x + g + 0.5
    for i in range(1, 9):
        a += c[i] / (x + float(i))
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


@njit(cache=True, fastmath=True)
def _t_pdf(x, nu):
    """Student-t PDF."""
    log_norm = _log_gamma((nu + 1.0) / 2.0) - _log_gamma(nu / 2.0) \
               - 0.5 * math.log(nu * math.pi)
    log_kernel = -((nu + 1.0) / 2.0) * math.log(1.0 + x * x / nu)
    return math.exp(log_norm + log_kernel)


@njit(cache=True, fastmath=True)
def _regularized_incomplete_beta(a, b, x):
    """Regularized incomplete beta function I_x(a,b) via continued fraction."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Use symmetry relation if needed
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(b, a, 1.0 - x)

    log_prefix = _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b) \
                 + a * math.log(x) + b * math.log(1.0 - x)
    prefix = math.exp(log_prefix)

    # Lentz continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 201):
        mf = float(m)
        # Even step
        numerator = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= c * d

        # Odd step
        numerator = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return prefix * f / a


@njit(cache=True, fastmath=True)
def _t_cdf(x, nu):
    """Student-t CDF via incomplete beta function."""
    t2 = x * x
    p = _regularized_incomplete_beta(nu / 2.0, 0.5, nu / (nu + t2))
    if x >= 0.0:
        return 0.5 * (1.0 + (1.0 - p))
    else:
        return 0.5 * p


@njit(cache=True, fastmath=True)
def _compute_g_nu(nu, n_quad=200):
    """
    Gini half-mean-difference g(nu) = 0.5 * E|X-X'| for standard t_nu.

    Computed via trapezoidal quadrature:
      g(nu) = 2 * integral_{-L}^{L} x * F_nu(x) * f_nu(x) dx

    The closed-form Beta function approach is numerically INCORRECT
    (see signals_calibration_numba.py comments). Quadrature is the gold standard.
    """
    if nu <= 1.01:
        nu = 1.01
    # Wider limits for heavier tails
    L = min(30.0, max(10.0, 4.0 * math.sqrt(nu / max(nu - 2.0, 0.1))))
    h = 2.0 * L / n_quad
    total = 0.0
    for i in range(n_quad + 1):
        x = -L + i * h
        fx = _t_pdf(x, nu)
        Fx = _t_cdf(x, nu)
        val = x * Fx * fx
        if i == 0 or i == n_quad:
            total += 0.5 * val
        else:
            total += val
    return 2.0 * total * h


# ---------------------------------------------------------------------------
# LOO-CRPS: Gaussian
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def loo_crps_gaussian(mu, sigma, returns):
    """
    Compute per-observation LOO-CRPS for Gaussian predictive distribution.

    CRPS(N(mu_t, sigma_t^2), r_t) = sigma_t * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]

    Parameters
    ----------
    mu : ndarray, shape (T,)
        Predictive means from Kalman filter at each time step.
    sigma : ndarray, shape (T,)
        Predictive standard deviations at each time step.
    returns : ndarray, shape (T,)
        Realized returns.

    Returns
    -------
    ndarray, shape (T,)
        Per-observation CRPS. Lower = better.
    """
    n = len(mu)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sig = sigma[i]
        if sig < 1e-12:
            sig = 1e-12
        z = (returns[i] - mu[i]) / sig
        cdf_z = _norm_cdf(z)
        pdf_z = _norm_pdf(z)
        out[i] = sig * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - _INV_SQRT_PI)
    return out


# ---------------------------------------------------------------------------
# LOO-CRPS: Student-t
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def loo_crps_student_t(mu, sigma, nu, returns):
    """
    Compute per-observation LOO-CRPS for Student-t predictive distribution.

    CRPS(t_nu(mu_t, sigma_t), r_t) = sigma_t * [z*(2*T(z)-1) + 2*t(z)*(nu+z^2)/(nu-1) - g(nu)]

    where g(nu) is the Student-t Gini coefficient.

    Parameters
    ----------
    mu : ndarray, shape (T,)
        Predictive locations.
    sigma : ndarray, shape (T,)
        Predictive scales.
    nu : float or ndarray
        Degrees of freedom. If scalar-like (all same), g(nu) computed once.
    returns : ndarray, shape (T,)
        Realized returns.

    Returns
    -------
    ndarray, shape (T,)
        Per-observation CRPS. Lower = better.
    """
    n = len(mu)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    # Detect scalar nu (common case: same nu for all observations)
    nu0 = nu[0]
    all_same = True
    for k in range(1, n):
        if nu[k] != nu0:
            all_same = False
            break

    if all_same:
        nu_val = nu0
        if nu_val < 2.01:
            nu_val = 2.01
        g_nu = _compute_g_nu(nu_val)
        for i in range(n):
            sig = sigma[i]
            if sig < 1e-12:
                sig = 1e-12
            z = (returns[i] - mu[i]) / sig
            cdf_z = _t_cdf(z, nu_val)
            pdf_z = _t_pdf(z, nu_val)
            out[i] = sig * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z * (nu_val + z * z) / (nu_val - 1.0) - g_nu)
    else:
        for i in range(n):
            sig = sigma[i]
            if sig < 1e-12:
                sig = 1e-12
            nu_i = nu[i]
            if nu_i < 2.01:
                nu_i = 2.01
            z = (returns[i] - mu[i]) / sig
            cdf_z = _t_cdf(z, nu_i)
            pdf_z = _t_pdf(z, nu_i)
            g_nu = _compute_g_nu(nu_i)
            out[i] = sig * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z * (nu_i + z * z) / (nu_i - 1.0) - g_nu)

    return out


# ---------------------------------------------------------------------------
# Mean LOO-CRPS (convenience)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def loo_crps_gaussian_mean(mu, sigma, returns):
    """Mean LOO-CRPS for Gaussian -- single scalar for optimizer objective."""
    scores = loo_crps_gaussian(mu, sigma, returns)
    total = 0.0
    n = len(scores)
    for i in range(n):
        total += scores[i]
    return total / max(n, 1)


@njit(cache=True, fastmath=True)
def loo_crps_student_t_mean(mu, sigma, nu, returns):
    """Mean LOO-CRPS for Student-t -- single scalar for optimizer objective."""
    scores = loo_crps_student_t(mu, sigma, nu, returns)
    total = 0.0
    n = len(scores)
    for i in range(n):
        total += scores[i]
    return total / max(n, 1)
