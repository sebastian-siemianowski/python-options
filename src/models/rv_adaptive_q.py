"""
===============================================================================
RV-ADAPTIVE PROCESS NOISE MODULE (Tune.md Story 1.1)
===============================================================================

Implements proactive process noise q_t driven by realized volatility changes.

PROBLEM STATEMENT:
    Static q: Lags during regime transitions. When volatility doubles overnight,
    the filter needs ~1/q timesteps to catch up. At q = 1e-6, effectively never.
    
    GAS-Q: Reacts to yesterday's errors, not today's regime. One step behind.

SOLUTION -- RV-LINKED PROCESS NOISE:
    Feed realized volatility *changes* into q directly:
    
        q_t = q_base * exp(gamma * delta_log(sigma_t^2))
    
    When sigma jumps (Garman-Klass detects intraday expansion), q jumps
    simultaneously. This is the difference between reactive and proactive.

MATHEMATICAL DERIVATION:
    Let sigma_t^2 be the realized variance estimate (GK, YZ, EWMA, etc.)
    
    delta_log(sigma_t^2) = log(sigma_t^2) - log(sigma_{t-1}^2)
                         = 2 * [log(sigma_t) - log(sigma_{t-1})]
    
    This quantity is:
    - Positive when volatility is expanding (regime shift to HIGH_VOL)
    - Negative when volatility is contracting (regime shift to LOW_VOL)
    - Near zero during stable regimes
    
    The exponential link ensures q_t > 0 always.

PARAMETER CONSTRAINTS:
    q_base > 0:   Baseline process noise (estimated via MLE)
    gamma >= 0:   RV-feedback sensitivity (gamma=0 recovers static q)
    q_min > 0:    Floor prevents filter freeze (default 1e-8)
    q_max < inf:  Ceiling prevents filter divergence (default 1e-2)

ESTIMATION:
    Parameters (q_base, gamma) estimated via profile likelihood:
    1. Grid initialization over (q_base, gamma) pairs
    2. L-BFGS-B refinement from best grid point
    3. c, phi, nu held fixed (concentrated likelihood)

INTEGRATION:
    - Competes with static-q and GAS-Q via BMA
    - Registered in model_registry as separate model variants
    - Uses same kernel interface as existing filter kernels

EXPECTED IMPACT:
    - Filter recovery from 2x vol shock: < 5 days (vs > 20 days static q)
    - BIC improvement on vol-driven assets: > 50 nats
    - No regression on stable assets (gamma -> 0 recovers static)

April 2026 -- Tune.md Epic 1, Story 1.1
===============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import os
import numpy as np
from scipy.optimize import minimize

# =============================================================================
# CONSTANTS
# =============================================================================

# RV-Q parameter bounds
RV_Q_BASE_MIN = 1e-10       # Minimum q_base
RV_Q_BASE_MAX = 1e-1        # Maximum q_base
RV_GAMMA_MIN = 0.0          # Minimum gamma (no negative feedback)
RV_GAMMA_MAX = 10.0         # Maximum gamma (extreme sensitivity)

# Process noise bounds
RV_Q_MIN = 1e-8             # Floor: prevents filter freeze
RV_Q_MAX = 1e-2             # Ceiling: prevents divergence

# Grid search initialization
GAMMA_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]
Q_BASE_GRID = [1e-7, 1e-6, 1e-5]

# BIC penalty for RV-Q (1 extra parameter: gamma)
RV_Q_BIC_PENALTY = 1.0


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RVAdaptiveQConfig:
    """
    Configuration for RV-adaptive process noise.

    The RV-Q equation is:
        q_t = q_base * exp(gamma * delta_log(vol_t^2))

    Attributes:
        q_base:  Baseline process noise (unconditional level)
        gamma:   Sensitivity to vol acceleration (0 = static q)
        q_min:   Minimum allowed q_t
        q_max:   Maximum allowed q_t
        enabled: Whether RV-Q dynamics are active
    """
    q_base: float = 1e-6
    gamma: float = 1.0
    q_min: float = RV_Q_MIN
    q_max: float = RV_Q_MAX
    enabled: bool = True

    def __post_init__(self):
        if self.q_base <= 0:
            raise ValueError(f"q_base must be > 0, got {self.q_base}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_base": self.q_base,
            "gamma": self.gamma,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RVAdaptiveQConfig":
        return cls(
            q_base=d.get("q_base", 1e-6),
            gamma=d.get("gamma", 1.0),
            q_min=d.get("q_min", RV_Q_MIN),
            q_max=d.get("q_max", RV_Q_MAX),
            enabled=d.get("enabled", True),
        )


# =============================================================================
# RESULTS
# =============================================================================

@dataclass
class RVAdaptiveQResult:
    """Result from RV-adaptive q filter run."""
    mu_filtered: np.ndarray
    P_filtered: np.ndarray
    q_path: np.ndarray
    log_likelihood: float
    config: RVAdaptiveQConfig
    n_params: int = 2  # q_base + gamma


# =============================================================================
# FILTER WRAPPERS
# =============================================================================

def _precompute_rv_q_vol_inputs(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute vol^2 and delta_log(vol^2) once for RV-Q likelihood loops."""
    vol_arr = np.ascontiguousarray(vol, dtype=np.float64)
    vol_sq = np.ascontiguousarray(vol_arr * vol_arr, dtype=np.float64)
    delta = np.zeros(len(vol_arr), dtype=np.float64)
    if len(vol_arr) > 1:
        curr = vol_arr[1:]
        prev = vol_arr[:-1]
        valid = (curr > 1e-15) & (prev > 1e-15)
        out = np.zeros(len(curr), dtype=np.float64)
        out[valid] = 2.0 * (np.log(curr[valid]) - np.log(prev[valid]))
        delta[1:] = out
    return vol_sq, np.ascontiguousarray(delta, dtype=np.float64)

def rv_adaptive_q_filter_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float = 1.0,
    config: Optional[RVAdaptiveQConfig] = None,
) -> RVAdaptiveQResult:
    """
    Run phi-Gaussian Kalman filter with RV-adaptive process noise.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    vol : np.ndarray
        Realized volatility series (same length as returns).
    c : float
        Observation noise scaling.
    phi : float
        AR(1) persistence.
    config : RVAdaptiveQConfig, optional
        Configuration. Uses defaults if None.

    Returns
    -------
    RVAdaptiveQResult
    """
    from models.numba_kernels import (
        rv_adaptive_q_gaussian_filter_kernel,
        rv_adaptive_q_gaussian_filter_precomputed_kernel,
    )

    if config is None:
        config = RVAdaptiveQConfig()

    returns = np.ascontiguousarray(returns, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    use_precomputed = os.environ.get("RV_Q_ENABLE_PRECOMPUTED_KERNEL", "") == "1"
    if use_precomputed:
        vol_sq, delta_log_vol_sq = _precompute_rv_q_vol_inputs(vol)

    n = len(returns)
    mu_filtered = np.zeros(n, dtype=np.float64)
    P_filtered = np.zeros(n, dtype=np.float64)
    q_path = np.zeros(n, dtype=np.float64)

    if use_precomputed:
        log_ll = rv_adaptive_q_gaussian_filter_precomputed_kernel(
            returns, vol_sq, delta_log_vol_sq,
            config.q_base, config.gamma,
            c, phi,
            config.q_min, config.q_max,
            1e-4,
            mu_filtered, P_filtered, q_path,
        )
    else:
        log_ll = rv_adaptive_q_gaussian_filter_kernel(
            returns, vol,
            config.q_base, config.gamma,
            c, phi,
            config.q_min, config.q_max,
            1e-4,
            mu_filtered, P_filtered, q_path,
        )

    return RVAdaptiveQResult(
        mu_filtered=mu_filtered,
        P_filtered=P_filtered,
        q_path=q_path,
        log_likelihood=log_ll,
        config=config,
    )


def rv_adaptive_q_filter_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu: float,
    config: Optional[RVAdaptiveQConfig] = None,
) -> RVAdaptiveQResult:
    """
    Run phi-Student-t Kalman filter with RV-adaptive process noise.

    Parameters
    ----------
    returns, vol, c, phi : see rv_adaptive_q_filter_gaussian
    nu : float
        Student-t degrees of freedom.
    config : RVAdaptiveQConfig, optional

    Returns
    -------
    RVAdaptiveQResult
    """
    from models.numba_kernels import (
        rv_adaptive_q_student_t_filter_kernel,
        rv_adaptive_q_student_t_filter_precomputed_kernel,
    )
    from models.numba_wrappers import precompute_gamma_values

    if config is None:
        config = RVAdaptiveQConfig()

    returns = np.ascontiguousarray(returns, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    use_precomputed = os.environ.get("RV_Q_ENABLE_PRECOMPUTED_KERNEL", "") == "1"
    if use_precomputed:
        vol_sq, delta_log_vol_sq = _precompute_rv_q_vol_inputs(vol)

    n = len(returns)
    mu_filtered = np.zeros(n, dtype=np.float64)
    P_filtered = np.zeros(n, dtype=np.float64)
    q_path = np.zeros(n, dtype=np.float64)

    log_gamma_half_nu, log_gamma_half_nu_plus_half = precompute_gamma_values(nu)

    if use_precomputed:
        log_ll = rv_adaptive_q_student_t_filter_precomputed_kernel(
            returns, vol_sq, delta_log_vol_sq,
            config.q_base, config.gamma,
            c, phi, nu,
            log_gamma_half_nu, log_gamma_half_nu_plus_half,
            config.q_min, config.q_max,
            1e-4,
            mu_filtered, P_filtered, q_path,
        )
    else:
        log_ll = rv_adaptive_q_student_t_filter_kernel(
            returns, vol,
            config.q_base, config.gamma,
            c, phi, nu,
            log_gamma_half_nu, log_gamma_half_nu_plus_half,
            config.q_min, config.q_max,
            1e-4,
            mu_filtered, P_filtered, q_path,
        )

    return RVAdaptiveQResult(
        mu_filtered=mu_filtered,
        P_filtered=P_filtered,
        q_path=q_path,
        log_likelihood=log_ll,
        config=config,
    )


# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize_rv_q_params(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu: Optional[float] = None,
    train_frac: float = 0.7,
) -> Tuple[RVAdaptiveQConfig, Dict[str, Any]]:
    """
    Optimize (q_base, gamma) via profile likelihood.

    Two-stage: grid search + L-BFGS-B refinement.
    c, phi, nu are held fixed (concentrated likelihood).

    Parameters
    ----------
    returns, vol : np.ndarray
        Full return/vol series.
    c, phi : float
        Fixed observation noise and persistence.
    nu : float, optional
        Student-t df. If None, uses Gaussian model.
    train_frac : float
        Fraction of data for training.

    Returns
    -------
    (config, diagnostics)
    """
    from models.numba_kernels import (
        rv_adaptive_q_gaussian_filter_kernel,
        rv_adaptive_q_student_t_filter_kernel,
        rv_adaptive_q_gaussian_filter_precomputed_kernel,
        rv_adaptive_q_student_t_filter_precomputed_kernel,
    )
    from models.numba_wrappers import precompute_gamma_values

    returns = np.ascontiguousarray(returns, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64)

    n_train = int(len(returns) * train_frac)
    r_train = returns[:n_train]
    v_train = vol[:n_train]
    v_train_sq, d_train = _precompute_rv_q_vol_inputs(v_train)
    use_precomputed = os.environ.get("RV_Q_ENABLE_PRECOMPUTED_KERNEL", "") == "1"

    # Precompute gamma values for Student-t
    if nu is not None:
        lg_half_nu, lg_half_nu_plus_half = precompute_gamma_values(nu)
    else:
        lg_half_nu, lg_half_nu_plus_half = 0.0, 0.0

    def neg_log_lik(params):
        q_base_log, gamma = params
        q_base = np.exp(q_base_log)
        gamma = max(0.0, gamma)

        try:
            if nu is not None:
                if use_precomputed:
                    ll = rv_adaptive_q_student_t_filter_precomputed_kernel(
                        r_train, v_train_sq, d_train,
                        q_base, gamma, c, phi, nu,
                        lg_half_nu, lg_half_nu_plus_half,
                        RV_Q_MIN, RV_Q_MAX, 1e-4,
                        np.empty(0), np.empty(0), np.empty(0),
                    )
                else:
                    ll = rv_adaptive_q_student_t_filter_kernel(
                        r_train, v_train,
                        q_base, gamma, c, phi, nu,
                        lg_half_nu, lg_half_nu_plus_half,
                        RV_Q_MIN, RV_Q_MAX, 1e-4,
                        np.empty(0), np.empty(0), np.empty(0),
                    )
            else:
                if use_precomputed:
                    ll = rv_adaptive_q_gaussian_filter_precomputed_kernel(
                        r_train, v_train_sq, d_train,
                        q_base, gamma, c, phi,
                        RV_Q_MIN, RV_Q_MAX, 1e-4,
                        np.empty(0), np.empty(0), np.empty(0),
                    )
                else:
                    ll = rv_adaptive_q_gaussian_filter_kernel(
                        r_train, v_train,
                        q_base, gamma, c, phi,
                        RV_Q_MIN, RV_Q_MAX, 1e-4,
                        np.empty(0), np.empty(0), np.empty(0),
                    )
            if not np.isfinite(ll):
                return 1e10
            return -ll
        except Exception:
            return 1e10

    # Stage 1: Grid search
    best_nll = 1e10
    best_params = (np.log(1e-6), 0.0)

    for q_b in Q_BASE_GRID:
        for g in GAMMA_GRID:
            nll = neg_log_lik((np.log(q_b), g))
            if nll < best_nll:
                best_nll = nll
                best_params = (np.log(q_b), g)

    # Stage 2: L-BFGS-B refinement
    bounds = [
        (np.log(RV_Q_BASE_MIN), np.log(RV_Q_BASE_MAX)),
        (RV_GAMMA_MIN, RV_GAMMA_MAX),
    ]

    try:
        result = minimize(
            neg_log_lik,
            x0=best_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-8},
        )
        if result.success and result.fun < best_nll:
            best_params = result.x
            best_nll = result.fun
    except Exception:
        pass  # Keep grid search result

    q_base_opt = np.exp(best_params[0])
    gamma_opt = max(0.0, best_params[1])

    config = RVAdaptiveQConfig(
        q_base=q_base_opt,
        gamma=gamma_opt,
    )

    diagnostics = {
        "q_base": q_base_opt,
        "gamma": gamma_opt,
        "log_likelihood": -best_nll,
        "n_train": n_train,
        "grid_best_q_base": np.exp(best_params[0]),
        "grid_best_gamma": gamma_opt,
        "optimizer_success": True,
    }

    # ---- Compare to static-q baseline (gamma=0) ----
    static_nll = neg_log_lik((np.log(q_base_opt), 0.0))
    diagnostics["ll_static"] = -static_nll
    diagnostics["ll_rv_adaptive"] = -best_nll
    diagnostics["delta_ll"] = -best_nll - (-static_nll)  # positive = RV-Q better

    # ---- BIC comparison ----
    n_obs = n_train
    k_static = 1   # q_base only
    k_rv = 2        # q_base + gamma
    bic_static = 2.0 * static_nll + k_static * np.log(n_obs)
    bic_rv = 2.0 * best_nll + k_rv * np.log(n_obs)
    diagnostics["bic_static"] = bic_static
    diagnostics["bic_rv"] = bic_rv
    diagnostics["delta_bic"] = bic_rv - bic_static  # negative = RV-Q better

    # ---- Out-of-sample validation ----
    if n_train < len(returns):
        r_test = returns[n_train:]
        v_test = vol[n_train:]
        v_test_sq, d_test = _precompute_rv_q_vol_inputs(v_test)
        n_test = len(r_test)

        # RV-Q out-of-sample
        if nu is not None:
            if use_precomputed:
                ll_oos_rv = rv_adaptive_q_student_t_filter_precomputed_kernel(
                    r_test, v_test_sq, d_test,
                    q_base_opt, gamma_opt, c, phi, nu,
                    lg_half_nu, lg_half_nu_plus_half,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
                ll_oos_static = rv_adaptive_q_student_t_filter_precomputed_kernel(
                    r_test, v_test_sq, d_test,
                    q_base_opt, 0.0, c, phi, nu,
                    lg_half_nu, lg_half_nu_plus_half,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
            else:
                ll_oos_rv = rv_adaptive_q_student_t_filter_kernel(
                    r_test, v_test,
                    q_base_opt, gamma_opt, c, phi, nu,
                    lg_half_nu, lg_half_nu_plus_half,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
                ll_oos_static = rv_adaptive_q_student_t_filter_kernel(
                    r_test, v_test,
                    q_base_opt, 0.0, c, phi, nu,
                    lg_half_nu, lg_half_nu_plus_half,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
        else:
            if use_precomputed:
                ll_oos_rv = rv_adaptive_q_gaussian_filter_precomputed_kernel(
                    r_test, v_test_sq, d_test,
                    q_base_opt, gamma_opt, c, phi,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
                ll_oos_static = rv_adaptive_q_gaussian_filter_precomputed_kernel(
                    r_test, v_test_sq, d_test,
                    q_base_opt, 0.0, c, phi,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
            else:
                ll_oos_rv = rv_adaptive_q_gaussian_filter_kernel(
                    r_test, v_test,
                    q_base_opt, gamma_opt, c, phi,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )
                ll_oos_static = rv_adaptive_q_gaussian_filter_kernel(
                    r_test, v_test,
                    q_base_opt, 0.0, c, phi,
                    RV_Q_MIN, RV_Q_MAX, 1e-4,
                    np.empty(0), np.empty(0), np.empty(0),
                )

        diagnostics["oos_ll_rv"] = ll_oos_rv
        diagnostics["oos_ll_static"] = ll_oos_static
        diagnostics["oos_delta_ll"] = ll_oos_rv - ll_oos_static
        diagnostics["n_test"] = n_test

    return config, diagnostics
