"""
Regime-Conditional Observation Noise (c) Estimation
====================================================
Tune.md Epic 2, Story 2.1

Estimates separate c values per regime:
  c_trend < c_range < c_crisis

The observation model is:
  r_t = mu_t + sqrt(c_t * sigma_t^2) * epsilon_t

where c_t = c[regime_t] is regime-dependent.

In trending regimes, drift explains more variance -> c should be smaller.
In crisis regimes, everything is noise -> c should be larger.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize

try:
    from models.numba_kernels import (
        build_c_array_from_regimes,
        regime_c_gaussian_filter_kernel,
        regime_c_student_t_filter_kernel,
        phi_gaussian_filter_kernel,
    )
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

try:
    from models.regime import MarketRegime, assign_regime_labels
except ImportError:
    pass


# Number of regimes
N_REGIMES = 5

# Default c per regime (initial guess)
DEFAULT_C_PER_REGIME = np.array([
    0.8,   # LOW_VOL_TREND  (drift matters -> smaller c)
    1.0,   # HIGH_VOL_TREND (drift + vol -> moderate c)
    1.2,   # LOW_VOL_RANGE  (noise-dominated -> larger c)
    1.3,   # HIGH_VOL_RANGE (choppy -> larger c)
    1.8,   # CRISIS_JUMP    (all noise -> maximal c)
], dtype=np.float64)

# Bounds for c per regime
C_MIN = 0.2
C_MAX = 5.0


@dataclass
class RegimeCConfig:
    """Configuration for regime-conditional c estimation."""
    c_per_regime: np.ndarray = field(default_factory=lambda: DEFAULT_C_PER_REGIME.copy())
    enabled: bool = True

    @property
    def c_trend(self) -> float:
        """c for LOW_VOL_TREND (regime 0)."""
        return float(self.c_per_regime[0])

    @property
    def c_crisis(self) -> float:
        """c for CRISIS_JUMP (regime 4)."""
        return float(self.c_per_regime[4])

    def get_c_array(self, regime_labels: np.ndarray) -> np.ndarray:
        """Build time-varying c_t array from regime labels."""
        if KERNELS_AVAILABLE:
            return build_c_array_from_regimes(regime_labels, self.c_per_regime)
        # Pure Python fallback
        n = len(regime_labels)
        c_array = np.empty(n, dtype=np.float64)
        for t in range(n):
            r = int(regime_labels[t])
            if 0 <= r < len(self.c_per_regime):
                c_array[t] = self.c_per_regime[r]
            else:
                c_array[t] = 1.0
        return c_array


@dataclass
class RegimeCResult:
    """Result of regime-conditional c estimation."""
    c_per_regime: np.ndarray
    log_likelihood: float
    ll_static: float  # log-likelihood with scalar c (for comparison)
    delta_ll: float
    delta_bic: float
    n_params_static: int  # 1 (scalar c)
    n_params_regime: int  # 5 (one per regime)
    regime_counts: Dict[int, int] = field(default_factory=dict)
    fit_success: bool = True


def _count_regimes(regime_labels: np.ndarray) -> Dict[int, int]:
    """Count observations per regime."""
    counts = {}
    for r in range(N_REGIMES):
        counts[r] = int(np.sum(regime_labels == r))
    return counts


def fit_regime_c(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    q: float,
    phi: float,
    c_scalar: float = 1.0,
    nu: Optional[float] = None,
    train_frac: float = 0.7,
    log_gamma_half_nu: float = 0.0,
    log_gamma_half_nu_plus_half: float = 0.0,
) -> RegimeCResult:
    """
    Estimate per-regime c values via profile MLE.

    Parameters
    ----------
    returns : array, shape (T,)
    vol : array, shape (T,)
    regime_labels : array of int, shape (T,)
        Regime label per time step (0-4).
    q : float
        Process noise variance (held fixed).
    phi : float
        AR(1) persistence (held fixed).
    c_scalar : float
        Scalar c from global MLE (used as baseline).
    nu : float or None
        If None, use Gaussian filter. Else, Student-t with given nu.
    train_frac : float
        Fraction of data used for training.

    Returns
    -------
    RegimeCResult with per-regime c estimates and diagnostics.
    """
    returns = np.asarray(returns, dtype=np.float64).flatten()
    vol = np.asarray(vol, dtype=np.float64).flatten()
    regime_labels = np.asarray(regime_labels, dtype=np.int64).flatten()
    n = len(returns)
    n_train = int(n * train_frac)

    regime_counts = _count_regimes(regime_labels)

    # Baseline: scalar c log-likelihood
    if nu is None:
        _, _, ll_static = phi_gaussian_filter_kernel(
            returns[:n_train], vol[:n_train], q, c_scalar, phi)
    else:
        from models.numba_kernels import phi_student_t_filter_kernel
        _, _, ll_static = phi_student_t_filter_kernel(
            returns[:n_train], vol[:n_train], q, c_scalar, phi,
            nu, log_gamma_half_nu, log_gamma_half_nu_plus_half)

    # Initialize c per regime from scalar c (slight regime-aware offsets)
    c0 = np.array([
        c_scalar * 0.8,   # LOW_VOL_TREND
        c_scalar * 1.0,   # HIGH_VOL_TREND
        c_scalar * 1.1,   # LOW_VOL_RANGE
        c_scalar * 1.2,   # HIGH_VOL_RANGE
        c_scalar * 1.5,   # CRISIS_JUMP
    ], dtype=np.float64)

    train_ret = returns[:n_train]
    train_vol = vol[:n_train]
    train_reg = regime_labels[:n_train]

    def neg_ll(log_c_vec):
        """Negative log-likelihood for per-regime c (parameterized in log-space)."""
        c_vec = np.exp(log_c_vec)
        c_vec = np.clip(c_vec, C_MIN, C_MAX)
        c_arr = build_c_array_from_regimes(train_reg, c_vec)
        if nu is None:
            _, _, ll = regime_c_gaussian_filter_kernel(
                train_ret, train_vol, q, c_arr, phi)
        else:
            _, _, ll = regime_c_student_t_filter_kernel(
                train_ret, train_vol, q, c_arr, phi, nu,
                log_gamma_half_nu, log_gamma_half_nu_plus_half)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    # L-BFGS-B in log(c) space
    log_c0 = np.log(np.clip(c0, C_MIN, C_MAX))
    bounds = [(np.log(C_MIN), np.log(C_MAX))] * N_REGIMES

    result = minimize(neg_ll, log_c0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-8})

    c_opt = np.exp(result.x)
    c_opt = np.clip(c_opt, C_MIN, C_MAX)

    # Compute full-sample log-likelihood with optimized c
    c_arr_full = build_c_array_from_regimes(regime_labels, c_opt)
    if nu is None:
        _, _, ll_regime = regime_c_gaussian_filter_kernel(
            returns, vol, q, c_arr_full, phi)
    else:
        _, _, ll_regime = regime_c_student_t_filter_kernel(
            returns, vol, q, c_arr_full, phi, nu,
            log_gamma_half_nu, log_gamma_half_nu_plus_half)

    # BIC comparison: regime-c has 5 params, scalar c has 1
    n_params_static = 1
    n_params_regime = N_REGIMES
    bic_static = -2.0 * ll_static + n_params_static * np.log(n)
    bic_regime = -2.0 * ll_regime + n_params_regime * np.log(n)
    delta_bic = bic_regime - bic_static  # negative = regime is better

    return RegimeCResult(
        c_per_regime=c_opt,
        log_likelihood=float(ll_regime),
        ll_static=float(ll_static),
        delta_ll=float(ll_regime - ll_static),
        delta_bic=float(delta_bic),
        n_params_static=n_params_static,
        n_params_regime=n_params_regime,
        regime_counts=regime_counts,
        fit_success=bool(result.success or np.isfinite(ll_regime)),
    )


def filter_regime_c_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    phi: float,
    config: RegimeCConfig,
    regime_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Gaussian Kalman filter with regime-conditional c.

    Returns (mu_filtered, P_filtered, log_likelihood).
    """
    c_array = config.get_c_array(regime_labels)
    return regime_c_gaussian_filter_kernel(
        np.asarray(returns, dtype=np.float64).flatten(),
        np.asarray(vol, dtype=np.float64).flatten(),
        q, c_array, phi)


def filter_regime_c_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    phi: float,
    nu: float,
    config: RegimeCConfig,
    regime_labels: np.ndarray,
    log_gamma_half_nu: float = 0.0,
    log_gamma_half_nu_plus_half: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Student-t Kalman filter with regime-conditional c.

    Returns (mu_filtered, P_filtered, log_likelihood).
    """
    c_array = config.get_c_array(regime_labels)
    return regime_c_student_t_filter_kernel(
        np.asarray(returns, dtype=np.float64).flatten(),
        np.asarray(vol, dtype=np.float64).flatten(),
        q, c_array, phi, nu,
        log_gamma_half_nu, log_gamma_half_nu_plus_half)
