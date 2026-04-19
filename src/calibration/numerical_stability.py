"""
Epic 28: Numerical Stability at Extreme Values

Provides numerically stable primitives for extreme conditions:
1. Safe Student-t log-pdf at low nu and extreme z
2. Covariance clamping for Kalman filter stability
3. Kahan compensated summation for log-likelihood accumulation

References:
- Kahan (1965): Pracniques for reducing floating-point errors
- Shaw (2006): Accurate Student-t evaluation in the tails
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from scipy.special import gammaln

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 28.1: Safe Student-t
NU_MIN = 2.01            # Minimum nu (variance diverges at nu=2)
NU_GAUSSIAN_APPROX = 100 # Above this, use Gaussian approximation
LOG_2PI = np.log(2.0 * np.pi)

# Story 28.2: Covariance bounds
P_MIN_DEFAULT = 1e-10    # Floor: filter always updates
P_MAX_DEFAULT = 1.0      # Ceiling: filter never loses all info

# Story 28.3: Kahan summation
KAHAN_RELATIVE_ERROR_TARGET = 1e-14


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CovarianceClampResult:
    """Result of covariance clamping."""
    P_clamped: float
    was_floored: bool
    was_ceilinged: bool
    P_original: float
    P_min: float
    P_max: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "P_clamped": float(self.P_clamped),
            "was_floored": self.was_floored,
            "was_ceilinged": self.was_ceilinged,
            "P_original": float(self.P_original),
        }


@dataclass
class KahanSumResult:
    """Result of Kahan compensated summation."""
    total: float
    compensation: float  # Running error compensation
    n_elements: int
    naive_total: float   # For comparison

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": float(self.total),
            "compensation": float(self.compensation),
            "n_elements": self.n_elements,
            "naive_total": float(self.naive_total),
        }


# ---------------------------------------------------------------------------
# Story 28.1: Safe Student-t Evaluation at Low nu
# ---------------------------------------------------------------------------

def safe_student_t_logpdf(
    x: np.ndarray,
    nu: float,
    mu: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Numerically stable Student-t log-pdf.

    Uses log-space computation throughout to avoid overflow at extreme z
    and low nu. Matches scipy.stats.t.logpdf to high precision.

    log p(x | nu, mu, scale) =
        gammaln((nu+1)/2) - gammaln(nu/2)
        - 0.5 * log(nu * pi) - log(scale)
        - (nu+1)/2 * log(1 + z^2/nu)

    where z = (x - mu) / scale.

    Parameters
    ----------
    x : array-like
        Observed values.
    nu : float
        Degrees of freedom (> 2.0).
    mu : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).

    Returns
    -------
    np.ndarray
        Log-pdf values (same shape as x).
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    nu = float(max(nu, NU_MIN))

    # Protect against zero scale
    scale_safe = np.maximum(scale, 1e-15)

    # Standardized values
    z = (x - mu) / scale_safe

    # Log-space computation (no overflow possible)
    # gammaln is always stable
    log_norm = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale_safe)
    )

    # Key: log(1 + z^2/nu) -- use np.log1p for small z^2/nu
    z2_over_nu = z ** 2 / nu
    log_kernel = -((nu + 1.0) / 2.0) * np.log1p(z2_over_nu)

    result = log_norm + log_kernel

    # Replace any remaining NaN/Inf with -1e30 (effectively zero probability)
    result = np.where(np.isfinite(result), result, -1e30)

    return result


def safe_student_t_logpdf_scalar(
    x: float,
    nu: float,
    mu: float,
    scale: float,
) -> float:
    """Scalar version of safe_student_t_logpdf."""
    nu = max(nu, NU_MIN)
    scale = max(scale, 1e-15)

    z = (x - mu) / scale
    z2_over_nu = z * z / nu

    log_norm = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
    )
    log_kernel = -((nu + 1.0) / 2.0) * np.log1p(z2_over_nu)

    result = log_norm + log_kernel
    if not np.isfinite(result):
        return -1e30
    return float(result)


# ---------------------------------------------------------------------------
# Story 28.2: Filter Covariance Floor and Ceiling
# ---------------------------------------------------------------------------

def clamp_covariance(
    P: float,
    P_min: float = P_MIN_DEFAULT,
    P_max: float = P_MAX_DEFAULT,
) -> CovarianceClampResult:
    """
    Clamp Kalman filter covariance to valid range.

    Enforces P_min <= P <= P_max at every timestep to prevent:
    - P -> 0: filter stops learning (Kalman gain K -> 0)
    - P -> inf: filter ignores prior (Kalman gain K -> 1)

    Parameters
    ----------
    P : float
        Current state covariance.
    P_min : float
        Floor value (default 1e-10).
    P_max : float
        Ceiling value (default 1.0).

    Returns
    -------
    CovarianceClampResult
        Clamped covariance and diagnostics.
    """
    P = float(P)
    P_min = float(P_min)
    P_max = float(P_max)

    if P_min > P_max:
        raise ValueError(f"P_min ({P_min}) must be <= P_max ({P_max})")

    # Handle NaN/Inf
    if not np.isfinite(P):
        return CovarianceClampResult(
            P_clamped=P_min,
            was_floored=True,
            was_ceilinged=False,
            P_original=P,
            P_min=P_min,
            P_max=P_max,
        )

    was_floored = P < P_min
    was_ceilinged = P > P_max

    P_clamped = max(P_min, min(P, P_max))

    return CovarianceClampResult(
        P_clamped=P_clamped,
        was_floored=was_floored,
        was_ceilinged=was_ceilinged,
        P_original=P,
        P_min=P_min,
        P_max=P_max,
    )


def clamp_covariance_array(
    P_array: np.ndarray,
    P_min: float = P_MIN_DEFAULT,
    P_max: float = P_MAX_DEFAULT,
) -> np.ndarray:
    """
    Vectorized covariance clamping for arrays.

    Parameters
    ----------
    P_array : array-like
        Array of covariance values.
    P_min, P_max : float
        Bounds.

    Returns
    -------
    np.ndarray
        Clamped covariance array.
    """
    P = np.asarray(P_array, dtype=np.float64)
    # Replace NaN/Inf with P_min
    P = np.where(np.isfinite(P), P, P_min)
    return np.clip(P, P_min, P_max)


# ---------------------------------------------------------------------------
# Story 28.3: Kahan Compensated Summation
# ---------------------------------------------------------------------------

def kahan_sum(values: np.ndarray) -> KahanSumResult:
    """
    Kahan compensated summation for log-likelihood accumulation.

    Reduces floating-point rounding error from O(n * eps) to O(eps),
    critical for BIC ranking where 0.5+ nats matter.

    Parameters
    ----------
    values : array-like
        Values to sum (typically log-likelihood contributions).

    Returns
    -------
    KahanSumResult
        Compensated sum and diagnostics.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()

    if len(vals) == 0:
        return KahanSumResult(
            total=0.0,
            compensation=0.0,
            n_elements=0,
            naive_total=0.0,
        )

    # Filter NaN/Inf
    valid = np.isfinite(vals)
    vals_clean = vals[valid]

    if len(vals_clean) == 0:
        return KahanSumResult(
            total=0.0,
            compensation=0.0,
            n_elements=0,
            naive_total=0.0,
        )

    # Kahan summation
    total = 0.0
    compensation = 0.0

    for v in vals_clean:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t

    # Naive sum for comparison
    naive = float(np.sum(vals_clean))

    return KahanSumResult(
        total=float(total),
        compensation=float(compensation),
        n_elements=len(vals_clean),
        naive_total=naive,
    )


def kahan_sum_value(values: np.ndarray) -> float:
    """
    Convenience wrapper returning just the compensated sum value.

    Parameters
    ----------
    values : array-like
        Values to sum.

    Returns
    -------
    float
        Compensated sum.
    """
    return kahan_sum(values).total
