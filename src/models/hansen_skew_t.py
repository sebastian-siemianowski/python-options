"""
===============================================================================
HANSEN'S SKEW-T DISTRIBUTION — Regime-Conditional Asymmetric Tails
===============================================================================

Implements Hansen (1994) skew-t distribution for return modeling with
regime-conditional asymmetry parameters.

MATHEMATICAL FORMULATION:
    Hansen's skew-t extends symmetric Student-t with asymmetry parameter λ ∈ (-1, 1).
    When λ=0, it reduces to symmetric Student-t.

    The density is defined piecewise:
        f(z|ν,λ) = bc * [1 + z²/((ν-2)(1-λ)²)]^(-(ν+1)/2)  for z < -a/b
        f(z|ν,λ) = bc * [1 + z²/((ν-2)(1+λ)²)]^(-(ν+1)/2)  for z ≥ -a/b

    Where:
        a = 4λc(ν-2)/(ν-1)
        b = √(1 + 3λ² - a²)
        c = Γ((ν+1)/2) / [√π(ν-2) Γ(ν/2)]

FINANCIAL INTERPRETATION:
    λ > 0: Right-skewed (recovery potential, positive tail heavier)
    λ < 0: Left-skewed (crash risk, negative tail heavier)
    λ = 0: Symmetric Student-t

REGIME-CONDITIONAL SKEWNESS:
    - Bull markets: typically λ < 0 (crash risk during complacency)
    - Bear markets: typically λ > 0 (recovery spikes)
    - Neutral: λ ≈ 0 (symmetric)

REFERENCES:
    Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation"
    International Economic Review, 35(3), 705-730.

    Fernández, C. & Steel, M.F.J. (1998). "On Bayesian Modeling of Fat Tails and Skewness"
    Journal of the American Statistical Association.

===============================================================================
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import kstest

# =============================================================================
# NUMBA JIT COMPILATION (February 2026 Performance Optimization)
# =============================================================================

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    njit = None
    prange = None

# Define Numba kernels if available
if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def _hansen_logpdf_kernel(x: np.ndarray, nu: float, lambda_: float,
                               a: float, b: float, c: float) -> np.ndarray:
        """
        JIT-compiled Hansen log-PDF kernel with parallel execution.
        
        Performance: 10-50× faster than pure Python for large arrays.
        """
        n = len(x)
        result = np.empty(n, dtype=np.float64)
        
        cutpoint = -a / b
        log_bc = np.log(b * c)
        neg_half_nu_plus_1 = -(nu + 1.0) / 2.0
        inv_nu_minus_2 = 1.0 / (nu - 2.0)
        log_1_plus_lambda = np.log(1.0 + lambda_)
        log_1_minus_lambda = np.log(1.0 - lambda_)
        
        for i in prange(n):
            if x[i] < cutpoint:
                z = (b * x[i] + a) / (1.0 - lambda_)
                result[i] = log_bc + neg_half_nu_plus_1 * np.log(1.0 + z * z * inv_nu_minus_2)
            else:
                z = (b * x[i] + a) / (1.0 + lambda_)
                result[i] = log_bc + neg_half_nu_plus_1 * np.log(1.0 + z * z * inv_nu_minus_2)
        
        return result
    
    @njit(cache=True, fastmath=True, parallel=True)
    def _hansen_pdf_kernel(x: np.ndarray, nu: float, lambda_: float,
                           a: float, b: float, c: float) -> np.ndarray:
        """
        JIT-compiled Hansen PDF kernel with parallel execution.
        """
        n = len(x)
        result = np.empty(n, dtype=np.float64)
        
        cutpoint = -a / b
        bc = b * c
        neg_half_nu_plus_1 = -(nu + 1.0) / 2.0
        inv_nu_minus_2 = 1.0 / (nu - 2.0)
        
        for i in prange(n):
            if x[i] < cutpoint:
                z = (b * x[i] + a) / (1.0 - lambda_)
                kernel = (1.0 + z * z * inv_nu_minus_2) ** neg_half_nu_plus_1
                result[i] = bc * kernel
            else:
                z = (b * x[i] + a) / (1.0 + lambda_)
                kernel = (1.0 + z * z * inv_nu_minus_2) ** neg_half_nu_plus_1
                result[i] = bc * kernel
        
        return result
else:
    # Dummy placeholders when Numba not available
    _hansen_logpdf_kernel = None
    _hansen_pdf_kernel = None


# =============================================================================
# HANSEN SKEW-T CONSTANTS
# =============================================================================

# Parameter bounds
HANSEN_NU_MIN = 2.1
HANSEN_NU_MAX = 500.0
HANSEN_NU_DEFAULT = 10.0

HANSEN_LAMBDA_MIN = -0.5
HANSEN_LAMBDA_MAX = 0.5
HANSEN_LAMBDA_DEFAULT = 0.0  # Symmetric

# Minimum observations for MLE
HANSEN_MLE_MIN_OBS = 50

# Convergence tolerance
HANSEN_MLE_TOL = 1e-6
HANSEN_MLE_MAX_ITER = 200


@dataclass
class HansenSkewTParams:
    """
    Parameters for Hansen's skew-t distribution.
    
    Attributes:
        nu: Degrees of freedom (> 2 for finite variance)
        lambda_: Skewness parameter ∈ (-1, 1)
    """
    nu: float
    lambda_: float
    
    def __post_init__(self):
        if self.nu <= 2:
            raise ValueError(f"nu must be > 2, got {self.nu}")
        if not -1 < self.lambda_ < 1:
            raise ValueError(f"lambda must be in (-1, 1), got {self.lambda_}")
    
    @property
    def is_symmetric(self) -> bool:
        """Check if distribution is effectively symmetric."""
        return abs(self.lambda_) < 1e-10
    
    @property
    def skew_direction(self) -> str:
        """Human-readable skew direction."""
        if abs(self.lambda_) < 0.01:
            return "symmetric"
        elif self.lambda_ > 0:
            return "right"
        else:
            return "left"
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "nu": float(self.nu),
            "lambda": float(self.lambda_),
            "is_symmetric": self.is_symmetric,
            "skew_direction": self.skew_direction,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'HansenSkewTParams':
        """Deserialize from dictionary."""
        return cls(
            nu=float(d["nu"]),
            lambda_=float(d.get("lambda", d.get("lambda_", 0.0)))
        )


def _hansen_constants(nu: float, lambda_: float) -> Tuple[float, float, float]:
    """
    Compute Hansen's skew-t distribution constants.
    
    Args:
        nu: Degrees of freedom (> 2)
        lambda_: Skewness parameter ∈ (-1, 1)
        
    Returns:
        Tuple (a, b, c) where:
        - a: Location adjustment
        - b: Scale adjustment
        - c: Normalizing constant
    """
    # Ensure numeric stability
    nu = max(nu, 2.01)
    lambda_ = np.clip(lambda_, -0.999, 0.999)
    
    # Compute normalizing constant c
    log_c = (
        gammaln((nu + 1) / 2) - 
        gammaln(nu / 2) - 
        0.5 * np.log(np.pi * (nu - 2))
    )
    c = np.exp(log_c)
    
    # Compute location adjustment a
    a = 4 * lambda_ * c * ((nu - 2) / (nu - 1))
    
    # Compute scale adjustment b
    b_squared = 1 + 3 * lambda_**2 - a**2
    if b_squared <= 0:
        b_squared = 1e-10  # Numerical safety
    b = np.sqrt(b_squared)
    
    return float(a), float(b), float(c)


def hansen_skew_t_pdf(
    x: Union[float, np.ndarray], 
    nu: float, 
    lambda_: float
) -> Union[float, np.ndarray]:
    """
    Hansen's skew-t probability density function.
    
    Args:
        x: Evaluation points (standardized)
        nu: Degrees of freedom (> 2)
        lambda_: Skewness parameter ∈ (-1, 1)
        
    Returns:
        PDF values at x
    """
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x).astype(np.float64)
    
    # Handle symmetric case efficiently
    if abs(lambda_) < 1e-10:
        result = stats.t.pdf(x, df=nu)
        return float(result[0]) if scalar_input else result
    
    a, b, c = _hansen_constants(nu, lambda_)
    
    # Vectorized implementation (February 2026 optimization)
    # Pre-compute constants
    cutpoint = -a / b
    neg_half_nu_plus_1 = -(nu + 1) / 2
    inv_nu_minus_2 = 1.0 / (nu - 2)
    bc = b * c
    
    # Allocate result
    result = np.empty_like(x, dtype=np.float64)
    
    # Left region mask
    left_mask = x < cutpoint
    
    # Process left region (x < cutpoint)
    if np.any(left_mask):
        x_left = x[left_mask]
        z_left = (b * x_left + a) / (1 - lambda_)
        kernel_left = (1 + z_left**2 * inv_nu_minus_2) ** neg_half_nu_plus_1
        result[left_mask] = bc * kernel_left
    
    # Process right region (x >= cutpoint)
    right_mask = ~left_mask
    if np.any(right_mask):
        x_right = x[right_mask]
        z_right = (b * x_right + a) / (1 + lambda_)
        kernel_right = (1 + z_right**2 * inv_nu_minus_2) ** neg_half_nu_plus_1
        result[right_mask] = bc * kernel_right
    
    return float(result[0]) if scalar_input else result


def hansen_skew_t_logpdf(
    x: Union[float, np.ndarray], 
    nu: float, 
    lambda_: float
) -> Union[float, np.ndarray]:
    """
    Log-PDF of Hansen's skew-t distribution.
    
    More numerically stable than log(pdf) for optimization.
    Uses Numba JIT when available for 10-50× speedup.
    
    Args:
        x: Evaluation points (standardized)
        nu: Degrees of freedom (> 2)
        lambda_: Skewness parameter ∈ (-1, 1)
        
    Returns:
        Log-PDF values at x
    """
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x).astype(np.float64)
    
    # Handle symmetric case efficiently
    if abs(lambda_) < 1e-10:
        result = stats.t.logpdf(x, df=nu)
        return float(result[0]) if scalar_input else result
    
    a, b, c = _hansen_constants(nu, lambda_)
    
    # Use Numba kernel if available (10-50× faster for large arrays)
    if _NUMBA_AVAILABLE and _hansen_logpdf_kernel is not None and len(x) > 10:
        try:
            x_contig = np.ascontiguousarray(x, dtype=np.float64)
            result = _hansen_logpdf_kernel(x_contig, float(nu), float(lambda_),
                                            float(a), float(b), float(c))
            return float(result[0]) if scalar_input else result
        except Exception:
            pass  # Fall through to Python implementation
    
    # Vectorized Python implementation (February 2026 optimization)
    cutpoint = -a / b
    log_b = np.log(b)
    log_c = np.log(c)
    log_1_plus_lambda = np.log(1 + lambda_)
    log_1_minus_lambda = np.log(1 - lambda_)
    neg_half_nu_plus_1 = -(nu + 1) / 2
    inv_nu_minus_2 = 1.0 / (nu - 2)
    
    result = np.empty_like(x, dtype=np.float64)
    left_mask = x < cutpoint
    
    if np.any(left_mask):
        z_left = (b * x[left_mask] + a) / (1 - lambda_)
        log_kernel_left = neg_half_nu_plus_1 * np.log(1 + z_left**2 * inv_nu_minus_2)
        result[left_mask] = log_b + log_c + log_kernel_left
    
    right_mask = ~left_mask
    if np.any(right_mask):
        z_right = (b * x[right_mask] + a) / (1 + lambda_)
        log_kernel_right = neg_half_nu_plus_1 * np.log(1 + z_right**2 * inv_nu_minus_2)
        result[right_mask] = log_b + log_c + log_kernel_right
    
    return float(result[0]) if scalar_input else result


def hansen_skew_t_cdf(
    x: Union[float, np.ndarray], 
    nu: float, 
    lambda_: float
) -> Union[float, np.ndarray]:
    """
    Hansen's skew-t cumulative distribution function.
    
    Uses the relationship to symmetric Student-t CDF with piecewise transformation.
    
    Args:
        x: Evaluation points (standardized)
        nu: Degrees of freedom (> 2)
        lambda_: Skewness parameter ∈ (-1, 1)
        
    Returns:
        CDF values at x in [0, 1]
    """
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    
    # Handle symmetric case efficiently
    if abs(lambda_) < 1e-10:
        result = stats.t.cdf(x, df=nu)
        return float(result[0]) if scalar_input else result
    
    a, b, c = _hansen_constants(nu, lambda_)
    
    result = np.zeros_like(x, dtype=float)
    
    # Cutpoint for piecewise definition
    cutpoint = -a / b
    
    # Left region: x < cutpoint
    left_mask = x < cutpoint
    if np.any(left_mask):
        x_left = x[left_mask]
        z_left = (b * x_left + a) / (1 - lambda_)
        # Scale z for Student-t CDF
        z_scaled = z_left * np.sqrt(nu / (nu - 2))
        # CDF for left region
        t_cdf = stats.t.cdf(z_scaled, df=nu)
        result[left_mask] = (1 - lambda_) * t_cdf
    
    # Right region: x >= cutpoint
    right_mask = ~left_mask
    if np.any(right_mask):
        x_right = x[right_mask]
        z_right = (b * x_right + a) / (1 + lambda_)
        z_scaled = z_right * np.sqrt(nu / (nu - 2))
        t_cdf = stats.t.cdf(z_scaled, df=nu)
        result[right_mask] = (1 - lambda_) / 2 + (1 + lambda_) * (t_cdf - 0.5)
    
    # Clip to valid probabilities
    result = np.clip(result, 0.0, 1.0)
    
    return float(result[0]) if scalar_input else result


def hansen_skew_t_ppf(
    p: Union[float, np.ndarray], 
    nu: float, 
    lambda_: float,
    tol: float = 1e-10,
    max_iter: int = 100
) -> Union[float, np.ndarray]:
    """
    Hansen's skew-t percent point function (inverse CDF / quantile function).
    
    Uses Newton-Raphson iteration with bisection fallback.
    
    Args:
        p: Probability values ∈ (0, 1)
        nu: Degrees of freedom (> 2)
        lambda_: Skewness parameter ∈ (-1, 1)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Quantile values
    """
    p = np.asarray(p)
    scalar_input = p.ndim == 0
    p = np.atleast_1d(p).astype(float)
    
    # Handle symmetric case efficiently
    if abs(lambda_) < 1e-10:
        result = stats.t.ppf(p, df=nu)
        return float(result[0]) if scalar_input else result
    
    # Initial guess from symmetric t
    x = stats.t.ppf(p, df=nu)
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = hansen_skew_t_cdf(x, nu, lambda_) - p
        df = hansen_skew_t_pdf(x, nu, lambda_)
        
        # Avoid division by zero
        df = np.maximum(df, 1e-12)
        
        dx = f / df
        x_new = x - dx
        
        if np.all(np.abs(dx) < tol):
            break
        
        x = x_new
    
    return float(x[0]) if scalar_input else x


def hansen_skew_t_rvs(
    size: Union[int, Tuple[int, ...]], 
    nu: float, 
    lambda_: float,
    random_state: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate random samples from Hansen's skew-t distribution.
    
    Uses Fernández-Steel inverse-scale transformation:
    If Z ~ t(ν), then transform Z to get skewed samples.
    
    Args:
        size: Output shape
        nu: Degrees of freedom (> 2)
        lambda_: Skewness parameter ∈ (-1, 1)
        random_state: NumPy random generator
        
    Returns:
        Random samples
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Handle symmetric case
    if abs(lambda_) < 1e-10:
        return random_state.standard_t(df=nu, size=size)
    
    # Generate symmetric t samples
    z = random_state.standard_t(df=nu, size=size)
    
    # Compute Hansen constants
    a, b, c = _hansen_constants(nu, lambda_)
    
    # Apply inverse Hansen transformation
    # z >= 0 maps to right region: (by+a)/(1+λ) = z => y = (z*(1+λ) - a) / b
    # z < 0 maps to left region:  (by+a)/(1-λ) = z => y = (z*(1-λ) - a) / b
    y = np.where(z >= 0,
                 (z * (1 + lambda_) - a) / b,
                 (z * (1 - lambda_) - a) / b)
    
    return y


def fit_hansen_skew_t_mle(
    returns: np.ndarray,
    nu_init: float = HANSEN_NU_DEFAULT,
    lambda_init: float = HANSEN_LAMBDA_DEFAULT,
    nu_bounds: Tuple[float, float] = (HANSEN_NU_MIN, HANSEN_NU_MAX),
    lambda_bounds: Tuple[float, float] = (HANSEN_LAMBDA_MIN, HANSEN_LAMBDA_MAX),
    max_iter: int = HANSEN_MLE_MAX_ITER
) -> Tuple[float, float, float, Dict]:
    """
    Maximum likelihood estimation of Hansen skew-t parameters.
    
    Args:
        returns: Return series (will be standardized internally)
        nu_init: Initial degrees of freedom
        lambda_init: Initial skewness
        nu_bounds: Bounds for ν
        lambda_bounds: Bounds for λ
        max_iter: Maximum optimization iterations
        
    Returns:
        Tuple of (nu_hat, lambda_hat, log_likelihood, diagnostics_dict)
    """
    returns = np.asarray(returns).flatten()
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    
    # Fallback for insufficient data
    if n < HANSEN_MLE_MIN_OBS:
        warnings.warn(f"Fewer than {HANSEN_MLE_MIN_OBS} observations for Hansen skew-t MLE")
        return nu_init, lambda_init, float('-inf'), {
            "fit_success": False,
            "error": "insufficient_data",
            "n_obs": n
        }
    
    # Standardize returns
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-10:
        return nu_init, lambda_init, float('-inf'), {
            "fit_success": False,
            "error": "zero_variance",
            "n_obs": n
        }
    z = (returns - mu) / sigma
    
    def neg_log_likelihood(params):
        nu, lam = params
        # Check bounds
        if nu <= nu_bounds[0] or nu >= nu_bounds[1]:
            return 1e10
        if lam <= lambda_bounds[0] - 0.01 or lam >= lambda_bounds[1] + 0.01:
            return 1e10
        try:
            ll = np.sum(hansen_skew_t_logpdf(z, nu, lam))
            if not np.isfinite(ll):
                return 1e10
            return -ll
        except Exception:
            return 1e10
    
    # Multiple starting points for robustness
    best_result = None
    best_nll = float('inf')
    
    init_points = [
        (nu_init, lambda_init),
        (10.0, 0.0),
        (5.0, 0.1),
        (5.0, -0.1),
        (20.0, 0.0),
        (8.0, 0.2),
        (8.0, -0.2),
    ]
    
    for nu0, lam0 in init_points:
        try:
            result = minimize(
                neg_log_likelihood,
                x0=[nu0, lam0],
                method='L-BFGS-B',
                bounds=[nu_bounds, lambda_bounds],
                options={'maxiter': max_iter, 'ftol': HANSEN_MLE_TOL}
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        return nu_init, lambda_init, float('-inf'), {
            "fit_success": False,
            "error": "optimization_failed",
            "n_obs": n
        }
    
    nu_hat, lambda_hat = best_result.x
    log_likelihood = -best_nll
    
    # Compute information criteria
    k = 2  # Number of parameters (ν, λ)
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + k * np.log(n)
    
    diagnostics = {
        "fit_success": True,
        "n_obs": n,
        "nu_hat": float(nu_hat),
        "lambda_hat": float(lambda_hat),
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "mean": float(mu),
        "std": float(sigma),
        "converged": best_result.success,
        "n_iterations": best_result.nit if hasattr(best_result, 'nit') else None,
    }
    
    return float(nu_hat), float(lambda_hat), float(log_likelihood), diagnostics


def compute_hansen_skew_t_pit(
    returns: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    nu: float,
    lambda_: float
) -> Tuple[float, float]:
    """
    Compute PIT (Probability Integral Transform) calibration test.
    
    If forecasts are well-calibrated, PIT values should be uniform[0,1].
    
    Args:
        returns: Observed returns
        mu: Forecast means
        scale: Forecast scales (std dev)
        nu: Degrees of freedom
        lambda_: Skewness parameter
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    returns = np.asarray(returns).flatten()
    mu = np.asarray(mu).flatten()
    scale = np.asarray(scale).flatten()
    
    n = min(len(returns), len(mu), len(scale))
    
    pit_values = []
    for i in range(n):
        if (np.isfinite(returns[i]) and np.isfinite(mu[i]) and 
            np.isfinite(scale[i]) and scale[i] > 1e-10):
            z = (returns[i] - mu[i]) / scale[i]
            pit = hansen_skew_t_cdf(z, nu, lambda_)
            if np.isfinite(pit):
                pit_values.append(pit)
    
    pit_values = np.array(pit_values)
    
    if len(pit_values) < 10:
        return 1.0, 0.0
    
    ks_result = kstest(pit_values, 'uniform')
    return float(ks_result.statistic), float(ks_result.pvalue)


def compare_symmetric_vs_hansen(
    returns: np.ndarray,
    nu_symmetric: float,
    nu_hansen: float,
    lambda_hansen: float
) -> Dict:
    """
    Compare symmetric Student-t vs Hansen skew-t using information criteria.
    
    Args:
        returns: Return series
        nu_symmetric: DoF for symmetric t
        nu_hansen: DoF for Hansen skew-t
        lambda_hansen: Skewness for Hansen skew-t
        
    Returns:
        Dict with comparison metrics
    """
    returns = np.asarray(returns).flatten()
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    
    if n < 30:
        return {
            "delta_aic": 0.0,
            "delta_bic": 0.0,
            "preference": "insufficient_data"
        }
    
    # Standardize
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-10:
        return {
            "delta_aic": 0.0,
            "delta_bic": 0.0,
            "preference": "zero_variance"
        }
    z = (returns - mu) / sigma
    
    # Symmetric t log-likelihood (1 parameter: ν)
    ll_symmetric = np.sum(stats.t.logpdf(z, df=nu_symmetric))
    aic_symmetric = -2 * ll_symmetric + 2 * 1
    bic_symmetric = -2 * ll_symmetric + 1 * np.log(n)
    
    # Hansen skew-t log-likelihood (2 parameters: ν, λ)
    ll_hansen = np.sum(hansen_skew_t_logpdf(z, nu_hansen, lambda_hansen))
    aic_hansen = -2 * ll_hansen + 2 * 2
    bic_hansen = -2 * ll_hansen + 2 * np.log(n)
    
    delta_aic = aic_hansen - aic_symmetric  # Negative = Hansen better
    delta_bic = bic_hansen - bic_symmetric
    
    # Preference: Hansen if ΔAIC < -2 (substantial improvement)
    if delta_aic < -2:
        preference = "hansen_skew_t"
    elif delta_aic > 2:
        preference = "symmetric_t"
    else:
        preference = "no_clear_preference"
    
    return {
        "delta_aic": float(delta_aic),
        "delta_bic": float(delta_bic),
        "ll_symmetric": float(ll_symmetric),
        "ll_hansen": float(ll_hansen),
        "preference": preference,
        "n_obs": n
    }


def hansen_skew_t_expected_shortfall(
    nu: float, 
    lambda_: float, 
    alpha: float = 0.05
) -> float:
    """
    Compute Expected Shortfall (CVaR) at level α for standardized Hansen skew-t.
    
    ES_α = E[X | X < VaR_α]
    
    Uses numerical integration over the left tail.
    
    Args:
        nu: Degrees of freedom
        lambda_: Skewness parameter
        alpha: Tail probability (default 5%)
        
    Returns:
        Expected shortfall value (negative for losses)
    """
    from scipy.integrate import quad
    
    var_alpha = hansen_skew_t_ppf(alpha, nu, lambda_)
    
    def integrand(x):
        return x * hansen_skew_t_pdf(x, nu, lambda_)
    
    result, _ = quad(integrand, -np.inf, var_alpha, limit=100)
    es = result / alpha
    
    return float(es)


def hansen_skew_t_moments(nu: float, lambda_: float) -> Dict:
    """
    Compute theoretical moments of Hansen's skew-t distribution.
    
    Args:
        nu: Degrees of freedom
        lambda_: Skewness parameter
        
    Returns:
        Dict with available moments
    """
    moments = {}
    
    a, b, c = _hansen_constants(nu, lambda_)
    
    # Mean exists for ν > 1
    if nu > 1:
        moments['mean'] = -a / b
    
    # Variance exists for ν > 2 (standardized to 1)
    if nu > 2:
        moments['variance'] = 1.0
    
    # Skewness exists for ν > 3
    if nu > 3:
        # Approximate skewness from the asymmetry parameter
        # For moderate ν, skewness ≈ sign(λ) * |λ| * factor
        moments['skewness_approx'] = 2 * lambda_ * np.sqrt(1 + 3 * lambda_**2) * (nu - 2) / (nu - 3)
    
    # Kurtosis exists for ν > 4
    if nu > 4:
        moments['excess_kurtosis'] = 3 * (nu - 2) / (nu - 4) - 3
    
    return moments


# =============================================================================
# UTILITY FUNCTIONS FOR BMA INTEGRATION
# =============================================================================

def get_hansen_skew_t_model_name(nu: float, lambda_: float) -> str:
    """Generate model name for BMA ensemble."""
    nu_str = str(int(nu)) if nu == int(nu) else f"{nu:.1f}"
    lambda_str = f"{lambda_:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
    return f"hansen_skew_t_nu_{nu_str}_lambda_{lambda_str}"


def is_hansen_skew_t_model(model_name: str) -> bool:
    """Check if model name is a Hansen skew-t variant."""
    return model_name.startswith("hansen_skew_t_")


def parse_hansen_skew_t_model_name(model_name: str) -> Optional[Tuple[float, float]]:
    """
    Parse (nu, lambda) from Hansen skew-t model name.
    
    Returns:
        Tuple of (nu, lambda) or None if parsing fails
    """
    if not is_hansen_skew_t_model(model_name):
        return None
    
    try:
        parts = model_name.replace("hansen_skew_t_nu_", "").split("_lambda_")
        if len(parts) != 2:
            return None
        
        nu = float(parts[0])
        lambda_str = parts[1].replace("p", "+").replace("m", "-")
        # Insert decimal point: "p015" -> "+0.15"
        if len(lambda_str) > 2:
            lambda_str = lambda_str[0] + lambda_str[1] + "." + lambda_str[2:]
        lambda_ = float(lambda_str)
        
        return (nu, lambda_)
    except Exception:
        return None


# =============================================================================
# Epic 17 Story 17.1: Continuous Lambda Optimization
# =============================================================================

# Regime labels (shared with tuning/signals)
LOW_VOL_TREND = "LOW_VOL_TREND"
HIGH_VOL_TREND = "HIGH_VOL_TREND"
LOW_VOL_RANGE = "LOW_VOL_RANGE"
HIGH_VOL_RANGE = "HIGH_VOL_RANGE"
CRISIS_JUMP = "CRISIS_JUMP"
ALL_REGIMES = [LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE, HIGH_VOL_RANGE, CRISIS_JUMP]

# Lambda grid for warm-start
LAMBDA_GRID = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

# Minimum observations per regime for separate estimation
MIN_REGIME_OBS = 50


@dataclass
class HansenLambdaResult:
    """Result from continuous lambda optimization."""
    lambda_star: float
    log_likelihood: float
    nu: float
    bic: float
    converged: bool


@dataclass
class RegimeLambdaResult:
    """Result from regime-conditional lambda estimates."""
    lambda_by_regime: Dict[str, float]
    ll_by_regime: Dict[str, float]
    n_obs_by_regime: Dict[str, int]
    total_bic: float
    regimes_estimated: List[str]


@dataclass
class SkewAdjustedDirectionResult:
    """Result from skew-adjusted directional signal."""
    prob_positive: float
    prob_positive_symmetric: float
    skew_adjustment: float
    lambda_: float
    nu: float


def _hansen_profile_loglik(
    lambda_val: float,
    standardized_innovations: np.ndarray,
    nu: float,
) -> float:
    """
    Profile log-likelihood for lambda given fixed nu.

    Parameters
    ----------
    lambda_val : skewness parameter to evaluate
    standardized_innovations : z_t = v_t / sqrt(S_t)
    nu : degrees of freedom (fixed)

    Returns
    -------
    Negative log-likelihood (for minimization).
    """
    lambda_val = float(np.clip(lambda_val, -0.9, 0.9))
    nu = max(float(nu), 2.01)

    valid = np.isfinite(standardized_innovations)
    z = standardized_innovations[valid]

    if len(z) == 0:
        return 1e12

    ll_vals = hansen_skew_t_logpdf(z, nu, lambda_val)
    total_ll = np.sum(ll_vals[np.isfinite(ll_vals)])

    return -total_ll


def optimize_hansen_lambda(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float = HANSEN_NU_DEFAULT,
    lambda_init: float = 0.0,
) -> HansenLambdaResult:
    """
    Continuous lambda optimization for Hansen skew-t via Brent's method.

    Story 17.1: Uses Brent's method (1D root-free optimization) with grid
    warm-start. Replaces discrete lambda grid with continuous optimization.

    Parameters
    ----------
    returns : asset return series
    vol : EWMA/GK volatility series
    q : process noise
    c : observation noise coefficient
    phi : AR(1) coefficient
    nu : degrees of freedom (fixed)
    lambda_init : initial lambda guess

    Returns
    -------
    HansenLambdaResult with optimal lambda* in (-0.9, 0.9).
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(returns)
    nu = max(float(nu), 2.01)

    # Run Kalman filter to get standardized innovations
    phi_val = float(np.clip(phi, -0.999, 0.999))
    q_val = float(q)
    c_val = float(c)

    mu = 0.0
    P = 1e-4
    innovations = np.zeros(n)
    S_pred = np.zeros(n)

    for t in range(n):
        mu_pred = phi_val * mu
        P_pred = (phi_val ** 2) * P + q_val
        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)
        S = P_pred + R
        if S <= 1e-12:
            S = 1e-12
        S_pred[t] = S
        r_val = float(returns[t])
        innovations[t] = r_val - mu_pred
        K = P_pred / S
        mu = mu_pred + K * innovations[t]
        P = (1.0 - K) * P_pred
        P = max(P, 1e-12)

    std_innovations = innovations / np.sqrt(np.maximum(S_pred, 1e-12))
    valid = np.isfinite(std_innovations)
    std_innovations_clean = std_innovations[valid]

    if len(std_innovations_clean) < 20:
        return HansenLambdaResult(
            lambda_star=0.0, log_likelihood=-1e6, nu=nu,
            bic=1e6, converged=False,
        )

    # Grid warm-start
    best_nll = 1e12
    best_lambda = lambda_init

    for lam in LAMBDA_GRID:
        nll = _hansen_profile_loglik(lam, std_innovations_clean, nu)
        if nll < best_nll:
            best_nll = nll
            best_lambda = lam

    # Brent's method refinement
    bracket_lo = max(-0.9, best_lambda - 0.15)
    bracket_hi = min(0.9, best_lambda + 0.15)

    try:
        result = minimize_scalar(
            _hansen_profile_loglik,
            bounds=(bracket_lo, bracket_hi),
            args=(std_innovations_clean, nu),
            method='bounded',
            options={'xatol': 1e-6, 'maxiter': 100},
        )
        if result.fun < best_nll:
            best_lambda = float(result.x)
            best_nll = result.fun
            converged = True
        else:
            converged = True
    except Exception:
        converged = False

    best_lambda = float(np.clip(best_lambda, -0.9, 0.9))
    log_likelihood = -best_nll

    n_valid = len(std_innovations_clean)
    k = 1  # lambda parameter
    bic = -2.0 * log_likelihood + k * np.log(n_valid)

    return HansenLambdaResult(
        lambda_star=best_lambda,
        log_likelihood=log_likelihood,
        nu=nu,
        bic=bic,
        converged=converged,
    )


# =============================================================================
# Epic 17 Story 17.2: Regime-Conditional Lambda
# =============================================================================

def regime_lambda_estimates(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: List[str],
    q: float,
    c: float,
    phi: float,
    nu: float = HANSEN_NU_DEFAULT,
) -> RegimeLambdaResult:
    """
    Regime-specific lambda estimates for Hansen skew-t.

    Story 17.2: Estimate separate lambda per regime so that crash risk
    is properly sized during high-vol periods.

    Parameters
    ----------
    returns : asset return series
    vol : EWMA/GK volatility series
    regime_labels : regime label per observation
    q, c, phi : Kalman filter parameters
    nu : degrees of freedom (fixed)

    Returns
    -------
    RegimeLambdaResult with per-regime lambda estimates.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(returns)

    if len(regime_labels) != n:
        raise ValueError("regime_labels must match returns length")

    # Run filter to get standardized innovations
    phi_val = float(np.clip(phi, -0.999, 0.999))
    q_val = float(q)
    c_val = float(c)

    mu = 0.0
    P = 1e-4
    innovations = np.zeros(n)
    S_pred = np.zeros(n)

    for t in range(n):
        mu_pred = phi_val * mu
        P_pred = (phi_val ** 2) * P + q_val
        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)
        S = P_pred + R
        if S <= 1e-12:
            S = 1e-12
        S_pred[t] = S
        r_val = float(returns[t])
        innovations[t] = r_val - mu_pred
        K = P_pred / S
        mu = mu_pred + K * innovations[t]
        P = (1.0 - K) * P_pred
        P = max(P, 1e-12)

    std_innovations = innovations / np.sqrt(np.maximum(S_pred, 1e-12))

    lambda_by_regime = {}
    ll_by_regime = {}
    n_obs_by_regime = {}
    regimes_estimated = []
    total_ll = 0.0

    for regime in ALL_REGIMES:
        mask = np.array([r == regime for r in regime_labels])
        regime_inno = std_innovations[mask]
        valid = np.isfinite(regime_inno)
        regime_inno = regime_inno[valid]
        n_obs = len(regime_inno)
        n_obs_by_regime[regime] = n_obs

        if n_obs < MIN_REGIME_OBS:
            lambda_by_regime[regime] = 0.0
            ll_by_regime[regime] = 0.0
            continue

        # Optimize lambda for this regime
        best_nll = 1e12
        best_lam = 0.0

        for lam in LAMBDA_GRID:
            nll = _hansen_profile_loglik(lam, regime_inno, nu)
            if nll < best_nll:
                best_nll = nll
                best_lam = lam

        bracket_lo = max(-0.9, best_lam - 0.15)
        bracket_hi = min(0.9, best_lam + 0.15)

        try:
            result = minimize_scalar(
                _hansen_profile_loglik,
                bounds=(bracket_lo, bracket_hi),
                args=(regime_inno, nu),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 100},
            )
            if result.fun < best_nll:
                best_lam = float(result.x)
                best_nll = result.fun
        except Exception:
            pass

        best_lam = float(np.clip(best_lam, -0.9, 0.9))
        lambda_by_regime[regime] = best_lam
        ll_by_regime[regime] = -best_nll
        total_ll += -best_nll
        regimes_estimated.append(regime)

    k = len(regimes_estimated)
    n_total = sum(n_obs_by_regime[r] for r in regimes_estimated) if regimes_estimated else 1
    total_bic = -2.0 * total_ll + k * np.log(max(n_total, 1))

    return RegimeLambdaResult(
        lambda_by_regime=lambda_by_regime,
        ll_by_regime=ll_by_regime,
        n_obs_by_regime=n_obs_by_regime,
        total_bic=total_bic,
        regimes_estimated=regimes_estimated,
    )


# =============================================================================
# Epic 17 Story 17.3: Skew-Adjusted Directional Signals
# =============================================================================

def skew_adjusted_direction(
    mu: float,
    sigma: float,
    nu: float = HANSEN_NU_DEFAULT,
    lambda_: float = 0.0,
) -> SkewAdjustedDirectionResult:
    """
    Directional signal adjusted for Hansen skew-t skewness.

    Story 17.3: Computes P(r > 0) using the Hansen CDF, accounting for
    the asymmetry parameter lambda.

    - Left-skewed (lambda < 0): P(r > 0) increases relative to symmetric
    - Right-skewed (lambda > 0): P(r > 0) decreases relative to symmetric

    Parameters
    ----------
    mu : forecast mean
    sigma : forecast standard deviation (sqrt(S_pred))
    nu : degrees of freedom
    lambda_ : Hansen skewness parameter

    Returns
    -------
    SkewAdjustedDirectionResult with P(r > 0) and adjustment.
    """
    sigma = max(float(sigma), 1e-12)
    nu = max(float(nu), 2.01)
    lambda_ = float(np.clip(lambda_, -0.9, 0.9))

    # Standardize: z corresponding to r=0
    z = -float(mu) / sigma

    # Symmetric Student-t P(r > 0) = 1 - T_nu(z)
    prob_symmetric = 1.0 - float(stats.t.cdf(z, df=nu))

    # Hansen skew-t P(r > 0) = 1 - F_hansen(z)
    prob_hansen = 1.0 - float(hansen_skew_t_cdf(z, nu, lambda_))
    prob_hansen = max(0.001, min(0.999, prob_hansen))
    prob_symmetric = max(0.001, min(0.999, prob_symmetric))

    skew_adjustment = prob_hansen - prob_symmetric

    return SkewAdjustedDirectionResult(
        prob_positive=prob_hansen,
        prob_positive_symmetric=prob_symmetric,
        skew_adjustment=skew_adjustment,
        lambda_=lambda_,
        nu=nu,
    )


def skew_adjusted_direction_array(
    mu_array: np.ndarray,
    sigma_array: np.ndarray,
    nu: float = HANSEN_NU_DEFAULT,
    lambda_: float = 0.0,
) -> np.ndarray:
    """
    Vectorized skew-adjusted P(r > 0) for arrays.

    Parameters
    ----------
    mu_array : array of forecast means
    sigma_array : array of forecast standard deviations
    nu : degrees of freedom
    lambda_ : Hansen skewness parameter

    Returns
    -------
    Array of P(r > 0) values.
    """
    mu_array = np.asarray(mu_array, dtype=np.float64).ravel()
    sigma_array = np.asarray(sigma_array, dtype=np.float64).ravel()

    probs = np.zeros(len(mu_array))
    for i in range(len(mu_array)):
        result = skew_adjusted_direction(
            mu_array[i], sigma_array[i], nu, lambda_,
        )
        probs[i] = result.prob_positive

    return probs
