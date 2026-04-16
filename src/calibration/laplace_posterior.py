"""
Story 10.1: Laplace Posterior Approximation for Parameter MC
=============================================================

Implements Laplace approximation of the parameter posterior:

    p(theta | data, model) ~ N(theta_hat, H^{-1})

where H is the negative Hessian of the log-likelihood at the MLE theta_hat.
This allows proper propagation of parameter uncertainty into predictive
distributions, inflating variance beyond the plug-in estimate.

Architecture:
  1. Evaluate log-likelihood at MLE via Kalman filter (Gaussian or Student-t)
  2. Compute Hessian via central finite differences (general, works for any model)
  3. Regularize if ill-conditioned (ridge when kappa > MAX_CONDITION)
  4. Invert to get posterior covariance Sigma = (-H)^{-1}
  5. Compute variance inflation factor for predictive distributions

Key equations:
  - Hessian: H_ij = d^2 ell / d theta_i d theta_j
  - Regularized: H_reg = H - lambda * I  (ensure negative definite)
  - Sigma = (-H_reg)^{-1}
  - Predictive: Var_pred = Var_model + J^T Sigma J  (delta method)
  - Variance inflation: ratio = Var_pred / Var_model in [1.05, 1.20] expected
"""
import os
import sys
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Condition number threshold for ridge regularization
MAX_CONDITION_NUMBER = 1e4

# Minimum ridge for numerical stability
MIN_RIDGE = 1e-10

# Finite difference step sizes (relative to parameter scale)
FD_STEP_RELATIVE = 1e-4     # Relative step for finite differences
FD_STEP_MIN = 1e-7          # Absolute minimum step

# Variance inflation bounds
MIN_VARIANCE_INFLATION = 1.0    # Cannot decrease variance
MAX_VARIANCE_INFLATION = 2.0    # Cap at 2x (prevents explosion)
EXPECTED_INFLATION_LOW = 1.05   # Typical lower bound
EXPECTED_INFLATION_HIGH = 1.20  # Typical upper bound

# Minimum observations for reliable Hessian
MIN_OBS_FOR_LAPLACE = 50


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class LaplaceResult:
    """Result of Laplace posterior approximation.

    Attributes
    ----------
    theta_hat : np.ndarray
        MLE parameter vector.
    covariance : np.ndarray
        Posterior covariance matrix Sigma = (-H)^{-1}.
    hessian : np.ndarray
        Negative Hessian at MLE (should be positive semi-definite).
    is_regularized : bool
        Whether ridge regularization was applied.
    condition_number : float
        Condition number of the negative Hessian.
    ridge_lambda : float
        Ridge penalty applied (0.0 if none needed).
    log_likelihood : float
        Log-likelihood at MLE.
    n_obs : int
        Number of observations used.
    param_names : Tuple[str, ...]
        Names of parameters in theta_hat order.
    variance_inflation : float
        Ratio of parameter-uncertainty-inflated variance to plug-in variance.
    param_std : np.ndarray
        Marginal standard deviations sqrt(diag(Sigma)).
    """
    theta_hat: np.ndarray
    covariance: np.ndarray
    hessian: np.ndarray
    is_regularized: bool
    condition_number: float
    ridge_lambda: float
    log_likelihood: float
    n_obs: int
    param_names: Tuple[str, ...]
    variance_inflation: float
    param_std: np.ndarray


# ---------------------------------------------------------------------------
# Log-Likelihood Evaluators
# ---------------------------------------------------------------------------

def _ll_gaussian(returns: np.ndarray, vol: np.ndarray,
                 c: float, phi: float, q: float) -> float:
    """Evaluate Gaussian Kalman filter log-likelihood at (c, phi, q)."""
    if c <= 0 or q <= 0:
        return -1e18
    try:
        from models.gaussian import GaussianDriftModel
        _, _, ll = GaussianDriftModel.filter_phi(returns, vol, q, c, phi)
        if np.isfinite(ll):
            return float(ll)
    except Exception:
        pass
    return -1e18


def _ll_student_t(returns: np.ndarray, vol: np.ndarray,
                  c: float, phi: float, q: float, nu: float) -> float:
    """Evaluate Student-t Kalman filter log-likelihood at (c, phi, q, nu)."""
    if c <= 0 or q <= 0 or nu <= 2.0:
        return -1e18
    try:
        from models.phi_student_t_unified import UnifiedPhiStudentTModel
        _, _, ll = UnifiedPhiStudentTModel.filter_phi(returns, vol, q, c, phi, nu)
        if np.isfinite(ll):
            return float(ll)
    except Exception:
        pass
    return -1e18


# ---------------------------------------------------------------------------
# Hessian Computation (Central Finite Differences)
# ---------------------------------------------------------------------------

def _compute_fd_step(theta_val: float, relative: float = FD_STEP_RELATIVE,
                     minimum: float = FD_STEP_MIN) -> float:
    """Compute finite difference step size, scaled to parameter magnitude."""
    return max(abs(theta_val) * relative, minimum)


def _compute_hessian_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute 3x3 Hessian of Gaussian log-likelihood w.r.t. theta = (c, phi, q).

    Uses central finite differences:
      d^2f/dx_i dx_j = [f(x+h_i+h_j) - f(x+h_i-h_j)
                        - f(x-h_i+h_j) + f(x-h_i-h_j)] / (4 h_i h_j)

    For diagonal:
      d^2f/dx_i^2 = [f(x+h_i) - 2f(x) + f(x-h_i)] / h_i^2
    """
    n_params = len(theta)
    H = np.zeros((n_params, n_params))

    c, phi, q = theta[0], theta[1], theta[2]
    ll_center = _ll_gaussian(returns, vol, c, phi, q)

    steps = np.array([_compute_fd_step(theta[i]) for i in range(n_params)])

    def ll_at(params):
        return _ll_gaussian(returns, vol, params[0], params[1], params[2])

    # Diagonal elements
    for i in range(n_params):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += steps[i]
        theta_m[i] -= steps[i]
        H[i, i] = (ll_at(theta_p) - 2.0 * ll_center + ll_at(theta_m)) / (steps[i] ** 2)

    # Off-diagonal elements
    for i in range(n_params):
        for j in range(i + 1, n_params):
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()
            theta_pp[i] += steps[i]; theta_pp[j] += steps[j]
            theta_pm[i] += steps[i]; theta_pm[j] -= steps[j]
            theta_mp[i] -= steps[i]; theta_mp[j] += steps[j]
            theta_mm[i] -= steps[i]; theta_mm[j] -= steps[j]
            cross = (ll_at(theta_pp) - ll_at(theta_pm) - ll_at(theta_mp) + ll_at(theta_mm))
            H[i, j] = cross / (4.0 * steps[i] * steps[j])
            H[j, i] = H[i, j]

    return H


def _compute_hessian_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    theta: np.ndarray,
    nu: float,
) -> np.ndarray:
    """
    Compute 3x3 Hessian of Student-t log-likelihood w.r.t. theta = (c, phi, q).

    nu is treated as fixed (not optimized in the Hessian) because it is
    typically selected from a discrete grid and held fixed during BMA.
    """
    n_params = len(theta)
    H = np.zeros((n_params, n_params))

    c, phi, q = theta[0], theta[1], theta[2]
    ll_center = _ll_student_t(returns, vol, c, phi, q, nu)

    steps = np.array([_compute_fd_step(theta[i]) for i in range(n_params)])

    def ll_at(params):
        return _ll_student_t(returns, vol, params[0], params[1], params[2], nu)

    # Diagonal elements
    for i in range(n_params):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += steps[i]
        theta_m[i] -= steps[i]
        H[i, i] = (ll_at(theta_p) - 2.0 * ll_center + ll_at(theta_m)) / (steps[i] ** 2)

    # Off-diagonal elements
    for i in range(n_params):
        for j in range(i + 1, n_params):
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()
            theta_pp[i] += steps[i]; theta_pp[j] += steps[j]
            theta_pm[i] += steps[i]; theta_pm[j] -= steps[j]
            theta_mp[i] -= steps[i]; theta_mp[j] += steps[j]
            theta_mm[i] -= steps[i]; theta_mm[j] -= steps[j]
            cross = (ll_at(theta_pp) - ll_at(theta_pm) - ll_at(theta_mp) + ll_at(theta_mm))
            H[i, j] = cross / (4.0 * steps[i] * steps[j])
            H[j, i] = H[i, j]

    return H


def compute_hessian(
    returns: np.ndarray,
    vol: np.ndarray,
    theta: np.ndarray,
    family: str = "gaussian",
    nu: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Hessian of log-likelihood for the given model family.

    Parameters
    ----------
    returns : array
        Return series.
    vol : array
        Volatility series (same length as returns).
    theta : array
        Parameter vector [c, phi, q].
    family : str
        "gaussian" or "student_t".
    nu : float, optional
        Degrees of freedom (required for student_t).

    Returns
    -------
    np.ndarray
        Hessian matrix (n_params x n_params).
    """
    theta = np.asarray(theta, dtype=np.float64)

    if family == "gaussian":
        return _compute_hessian_gaussian(returns, vol, theta)
    elif family == "student_t":
        if nu is None:
            raise ValueError("nu required for student_t family")
        return _compute_hessian_student_t(returns, vol, theta, nu)
    else:
        raise ValueError(f"Unknown family: {family}. Use 'gaussian' or 'student_t'.")


# ---------------------------------------------------------------------------
# Hessian Regularization
# ---------------------------------------------------------------------------

def _compute_condition_number(neg_hessian: np.ndarray) -> float:
    """Compute condition number of the negative Hessian."""
    try:
        eigenvalues = np.linalg.eigvalsh(neg_hessian)
        abs_eig = np.abs(eigenvalues)
        min_eig = abs_eig.min()
        if min_eig < 1e-15:
            return 1e12
        return float(abs_eig.max() / min_eig)
    except np.linalg.LinAlgError:
        return 1e12


def regularize_hessian(
    neg_hessian: np.ndarray,
    max_condition: float = MAX_CONDITION_NUMBER,
) -> Tuple[np.ndarray, float, bool]:
    """
    Ensure neg_hessian is positive definite with bounded condition number.

    The negative Hessian (-H) should be positive definite at the MLE.
    If not, or if ill-conditioned, add ridge: -H_reg = -H + lambda * I.

    Parameters
    ----------
    neg_hessian : np.ndarray
        The negative Hessian matrix (-H).
    max_condition : float
        Maximum allowed condition number.

    Returns
    -------
    reg_neg_hessian : np.ndarray
        Regularized negative Hessian.
    ridge_lambda : float
        Ridge penalty applied (0.0 if none needed).
    was_regularized : bool
        True if regularization was applied.
    """
    n = neg_hessian.shape[0]

    try:
        eigenvalues = np.linalg.eigvalsh(neg_hessian)
    except np.linalg.LinAlgError:
        # Fallback: diagonal regularization
        ridge = np.abs(neg_hessian).max() * 0.1 + MIN_RIDGE
        return neg_hessian + ridge * np.eye(n), ridge, True

    min_eig = eigenvalues.min()
    max_eig = eigenvalues.max()

    # Case 1: Not positive definite -- some eigenvalues <= 0
    if min_eig <= 0:
        ridge = abs(min_eig) + MIN_RIDGE + max(abs(max_eig), 1.0) * 1e-4
        reg = neg_hessian + ridge * np.eye(n)
        return reg, ridge, True

    # Case 2: Check condition number
    kappa = max_eig / min_eig
    if kappa > max_condition:
        # Choose ridge so that condition number = max_condition
        # (max_eig) / (min_eig + lambda) = max_condition
        # => lambda = max_eig / max_condition - min_eig
        ridge = max(max_eig / max_condition - min_eig, MIN_RIDGE)
        reg = neg_hessian + ridge * np.eye(n)
        return reg, ridge, True

    # No regularization needed
    return neg_hessian, 0.0, False


# ---------------------------------------------------------------------------
# Core Laplace Posterior
# ---------------------------------------------------------------------------

def laplace_posterior(
    returns: np.ndarray,
    vol: np.ndarray,
    theta_hat: np.ndarray,
    family: str = "gaussian",
    nu: Optional[float] = None,
    param_names: Optional[Tuple[str, ...]] = None,
    max_condition: float = MAX_CONDITION_NUMBER,
) -> LaplaceResult:
    """
    Compute Laplace approximation to the parameter posterior.

    p(theta | data, model) ~ N(theta_hat, Sigma)
    where Sigma = (-H)^{-1} and H is the Hessian of log-likelihood at theta_hat.

    Parameters
    ----------
    returns : np.ndarray
        Return series (length T).
    vol : np.ndarray
        Volatility series (length T).
    theta_hat : np.ndarray
        MLE parameter vector [c, phi, q].
    family : str
        "gaussian" or "student_t".
    nu : float, optional
        Degrees of freedom (required if family="student_t").
    param_names : tuple of str, optional
        Names for each parameter. Defaults to ("c", "phi", "q").
    max_condition : float
        Maximum condition number before ridge regularization.

    Returns
    -------
    LaplaceResult
        Full posterior approximation result.
    """
    returns = np.asarray(returns, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    n_obs = len(returns)

    if param_names is None:
        param_names = ("c", "phi", "q")

    # Evaluate log-likelihood at MLE
    if family == "gaussian":
        ll_mle = _ll_gaussian(returns, vol, theta_hat[0], theta_hat[1], theta_hat[2])
    else:
        ll_mle = _ll_student_t(returns, vol, theta_hat[0], theta_hat[1], theta_hat[2], nu)

    # Compute Hessian
    H = compute_hessian(returns, vol, theta_hat, family=family, nu=nu)

    # Negate to get negative Hessian (should be PSD at MLE)
    neg_H = -H

    # Compute condition number before regularization
    cond_raw = _compute_condition_number(neg_H)

    # Regularize if needed
    neg_H_reg, ridge_lambda, was_regularized = regularize_hessian(
        neg_H, max_condition=max_condition
    )

    cond_final = _compute_condition_number(neg_H_reg)

    # Invert to get posterior covariance
    try:
        covariance = np.linalg.inv(neg_H_reg)
    except np.linalg.LinAlgError:
        # Last resort: pseudoinverse
        covariance = np.linalg.pinv(neg_H_reg)
        was_regularized = True

    # Ensure covariance diagonal is non-negative
    for i in range(covariance.shape[0]):
        if covariance[i, i] < 0:
            covariance[i, i] = MIN_RIDGE

    # Parameter standard deviations
    param_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    # Compute variance inflation factor
    variance_inflation = compute_variance_inflation(
        theta_hat, covariance, family=family, nu=nu
    )

    return LaplaceResult(
        theta_hat=theta_hat.copy(),
        covariance=covariance,
        hessian=H,
        is_regularized=was_regularized,
        condition_number=cond_final,
        ridge_lambda=ridge_lambda,
        log_likelihood=ll_mle,
        n_obs=n_obs,
        param_names=param_names,
        variance_inflation=variance_inflation,
        param_std=param_std,
    )


# ---------------------------------------------------------------------------
# Variance Inflation via Delta Method
# ---------------------------------------------------------------------------

def compute_variance_inflation(
    theta_hat: np.ndarray,
    covariance: np.ndarray,
    family: str = "gaussian",
    nu: Optional[float] = None,
) -> float:
    """
    Compute the variance inflation factor from parameter uncertainty.

    Uses the delta method: if sigma^2 = g(theta), then
      Var(sigma^2) ~ J^T Sigma J
    where J = dg/dtheta is the Jacobian of the predictive variance w.r.t. parameters.

    For the Kalman filter, the one-step-ahead predictive variance is:
      S = phi^2 * P + q + c * vol^2

    At steady state: P_ss = q / (1 - phi^2)  (for |phi| < 1)
    So: S = phi^2 * q / (1 - phi^2) + q + c * vol^2
          = q / (1 - phi^2) + c * vol^2

    The variance inflation is:
      ratio = (S + J^T Sigma J) / S

    For Student-t, the predictive variance is additionally scaled by (nu-2)/nu.

    Parameters
    ----------
    theta_hat : array [c, phi, q]
    covariance : 2D array, posterior covariance
    family : str
    nu : float, optional

    Returns
    -------
    float
        Variance inflation ratio (>= 1.0).
    """
    c, phi, q = theta_hat[0], theta_hat[1], theta_hat[2]

    # Steady-state predictive variance (using vol=1 as reference)
    phi2 = phi ** 2
    denom = max(1.0 - phi2, 1e-6)  # Prevent division by zero
    P_ss = q / denom
    S_ss = P_ss + q + c  # At vol=1

    if S_ss < 1e-15:
        return 1.0

    # Jacobian of S w.r.t. (c, phi, q)
    # dS/dc = 1 (from c * vol^2 at vol=1)
    # dS/dphi = 2*phi*q / (1-phi^2)^2  (chain rule on P_ss)
    # dS/dq = 1/(1-phi^2) + 1  (from P_ss + q)
    J = np.zeros(3)
    J[0] = 1.0  # dS/dc
    J[1] = 2.0 * phi * q / (denom ** 2)  # dS/dphi
    J[2] = 1.0 / denom + 1.0  # dS/dq

    # Delta method: Var(S) ~ J^T Sigma J
    var_S = float(J @ covariance @ J)

    # Student-t scaling: predictive variance is S * (nu-2)/nu
    if family == "student_t" and nu is not None and nu > 2:
        scale = (nu - 2.0) / nu
        S_ss *= scale
        var_S *= scale ** 2

    # Inflation ratio
    ratio = (S_ss + abs(var_S)) / S_ss
    return float(np.clip(ratio, MIN_VARIANCE_INFLATION, MAX_VARIANCE_INFLATION))


# ---------------------------------------------------------------------------
# Predictive Variance with Parameter Uncertainty
# ---------------------------------------------------------------------------

def predictive_variance_with_uncertainty(
    sigma_model: float,
    laplace_result: LaplaceResult,
) -> float:
    """
    Inflate model predictive standard deviation by parameter uncertainty.

    sigma_pred = sigma_model * sqrt(variance_inflation)

    Parameters
    ----------
    sigma_model : float
        Plug-in predictive standard deviation (from Kalman filter).
    laplace_result : LaplaceResult
        Laplace posterior approximation.

    Returns
    -------
    float
        Inflated predictive standard deviation.
    """
    inflation = max(laplace_result.variance_inflation, MIN_VARIANCE_INFLATION)
    return sigma_model * math.sqrt(inflation)


# ---------------------------------------------------------------------------
# Sample from Laplace Posterior
# ---------------------------------------------------------------------------

def sample_from_laplace(
    laplace_result: LaplaceResult,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw parameter samples from the Laplace posterior.

    theta^(i) ~ N(theta_hat, Sigma)

    Samples are constrained to valid parameter ranges:
      c > 0, q > 0, |phi| <= 0.999

    Parameters
    ----------
    laplace_result : LaplaceResult
        Laplace approximation result.
    n_samples : int
        Number of samples to draw.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_params).
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_hat = laplace_result.theta_hat
    cov = laplace_result.covariance

    # Draw from multivariate normal
    try:
        samples = rng.multivariate_normal(theta_hat, cov, size=n_samples)
    except np.linalg.LinAlgError:
        # Fallback: independent draws from marginals
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        samples = rng.normal(theta_hat, std, size=(n_samples, len(theta_hat)))

    # Enforce parameter constraints
    # theta = [c, phi, q] by convention
    samples[:, 0] = np.maximum(samples[:, 0], 1e-8)   # c > 0
    samples[:, 1] = np.clip(samples[:, 1], -0.999, 0.999)  # |phi| < 1
    samples[:, 2] = np.maximum(samples[:, 2], 1e-10)  # q > 0

    return samples


# ---------------------------------------------------------------------------
# Prediction Interval Coverage
# ---------------------------------------------------------------------------

def compute_prediction_interval(
    mu: float,
    sigma_model: float,
    laplace_result: LaplaceResult,
    alpha: float = 0.10,
    family: str = "gaussian",
    nu: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute (1-alpha) prediction interval accounting for parameter uncertainty.

    Parameters
    ----------
    mu : float
        Predictive mean.
    sigma_model : float
        Plug-in predictive std.
    laplace_result : LaplaceResult
        Laplace posterior result.
    alpha : float
        Significance level (0.10 for 90% PI).
    family : str
        "gaussian" or "student_t".
    nu : float, optional
        Degrees of freedom for Student-t.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    sigma_inflated = predictive_variance_with_uncertainty(sigma_model, laplace_result)

    if family == "student_t" and nu is not None and nu > 2:
        from scipy.stats import t as t_dist
        z = t_dist.ppf(1.0 - alpha / 2.0, df=nu)
    else:
        from scipy.stats import norm
        z = norm.ppf(1.0 - alpha / 2.0)

    lower = mu - z * sigma_inflated
    upper = mu + z * sigma_inflated
    return (lower, upper)


def compute_pi_coverage(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    alpha: float = 0.10,
    family: str = "gaussian",
    nu: Optional[float] = None,
) -> float:
    """
    Compute empirical coverage of (1-alpha) prediction intervals.

    Parameters
    ----------
    returns : array
        Realized returns.
    mu_pred : array
        Predicted means (same length).
    sigma_pred : array
        Predicted standard deviations (same length).
    alpha : float
        Significance level.
    family : str
        Distribution family.
    nu : float, optional
        Degrees of freedom.

    Returns
    -------
    float
        Empirical coverage in [0, 1].
    """
    n = len(returns)
    if n == 0:
        return 0.0

    if family == "student_t" and nu is not None and nu > 2:
        from scipy.stats import t as t_dist
        z = t_dist.ppf(1.0 - alpha / 2.0, df=nu)
    else:
        from scipy.stats import norm
        z = norm.ppf(1.0 - alpha / 2.0)

    covered = 0
    for i in range(n):
        lower = mu_pred[i] - z * sigma_pred[i]
        upper = mu_pred[i] + z * sigma_pred[i]
        if lower <= returns[i] <= upper:
            covered += 1

    return covered / n
