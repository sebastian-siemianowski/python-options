"""
===============================================================================
PHI-GAUSSIAN DRIFT MODEL — Kalman Filter with AR(1) Drift and Gaussian Noise
===============================================================================

Implements a state-space model with AR(1) drift dynamics:

    State equation:    μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,         ε_t ~ N(0, c·σ²_t)

Parameters:
    q:   Process noise variance (drift evolution uncertainty)
    c:   Observation noise scale (multiplier on EWMA variance)
    φ:   AR(1) persistence coefficient (φ=1 is random walk, φ=0 is mean reversion)

The model includes an explicit Gaussian shrinkage prior on φ:
    φ_r ~ N(φ_global, τ²)

This prevents unit-root instability and small-sample hallucinations.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest, norm

# Filter cache for deterministic result reuse
try:
    from .filter_cache import (
        cached_phi_gaussian_filter,
        get_filter_cache,
        FilterCacheKey,
        FILTER_CACHE_ENABLED,
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    FILTER_CACHE_ENABLED = False

# Numba wrappers for JIT-compiled filters (optional performance enhancement)
try:
    from .numba_wrappers import (
        is_numba_available,
        run_phi_gaussian_filter,
        run_gaussian_filter,
    )
    _USE_NUMBA = is_numba_available()
except ImportError:
    _USE_NUMBA = False
    run_phi_gaussian_filter = None
    run_gaussian_filter = None


# =============================================================================
# φ SHRINKAGE PRIOR CONSTANTS (self-contained, no external dependencies)
# =============================================================================

PHI_SHRINKAGE_TAU_MIN = 1e-3
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05


def _phi_shrinkage_log_prior(
    phi_r: float,
    phi_global: float,
    tau: float,
    tau_min: float = PHI_SHRINKAGE_TAU_MIN
) -> float:
    """Compute Gaussian shrinkage log-prior for φ."""
    tau_safe = max(tau, tau_min)
    if not np.isfinite(phi_global):
        return float('-inf')
    deviation = phi_r - phi_global
    return -0.5 * (deviation ** 2) / (tau_safe ** 2)


def _lambda_to_tau(lam: float, lam_min: float = 1e-12) -> float:
    """Convert legacy penalty weight λ to Gaussian prior std τ."""
    lam_safe = max(lam, lam_min)
    return 1.0 / math.sqrt(2.0 * lam_safe)


def _compute_phi_prior_diagnostics(
    phi_r: float,
    phi_global: float,
    tau: float,
    log_likelihood: float
) -> Dict[str, Optional[float]]:
    """Compute diagnostic information for φ shrinkage prior."""
    log_prior = _phi_shrinkage_log_prior(phi_r, phi_global, tau)
    ratio = None
    if np.isfinite(log_prior) and np.isfinite(log_likelihood):
        abs_prior = abs(log_prior)
        abs_ll = abs(log_likelihood)
        if abs_ll > 1e-12:
            ratio = abs_prior / abs_ll
    return {
        'phi_prior_logp': float(log_prior) if np.isfinite(log_prior) else None,
        'phi_likelihood_logp': float(log_likelihood) if np.isfinite(log_likelihood) else None,
        'phi_prior_likelihood_ratio': float(ratio) if ratio is not None else None,
        'phi_deviation_from_global': float(phi_r - phi_global),
        'phi_tau_used': float(tau),
    }


def _kalman_filter_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimized Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t.
    
    Performance optimizations (February 2026):
    - Pre-compute phi_sq and R array once
    - Pre-compute log_2pi constant
    - Use np.empty instead of np.zeros
    - Ensure contiguous array access
    """
    n = len(returns)
    
    # Convert to contiguous float64 arrays once
    returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
    vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
    
    # Extract scalar values once
    q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
    c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))
    
    # Pre-compute constants
    phi_sq = phi_val * phi_val
    log_2pi = np.log(2 * np.pi)
    
    # Pre-compute R array (vectorized)
    R = c_val * (vol * vol)

    mu = 0.0
    P = 1e-4
    mu_filtered = np.empty(n, dtype=np.float64)
    P_filtered = np.empty(n, dtype=np.float64)
    log_likelihood = 0.0

    for t in range(n):
        mu_pred = phi_val * mu
        P_pred = phi_sq * P + q_val

        S = P_pred + R[t]
        if S <= 1e-12:
            S = 1e-12
        
        innovation = returns[t] - mu_pred
        K = P_pred / S

        mu = mu_pred + K * innovation
        P = (1.0 - K) * P_pred
        if P < 1e-12:
            P = 1e-12

        mu_filtered[t] = mu
        P_filtered[t] = P

        # Inlined log-likelihood
        log_likelihood += -0.5 * (log_2pi + np.log(S) + (innovation * innovation) / S)

    return mu_filtered, P_filtered, float(log_likelihood)


def _kalman_filter_phi_with_trajectory(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Kalman filter with per-timestep likelihood trajectory for fold slicing.
    
    Returns (mu_filtered, P_filtered, total_log_likelihood, loglik_trajectory).
    """
    n = len(returns)
    q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
    c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))

    mu = 0.0
    P = 1e-4
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    loglik_trajectory = np.zeros(n)
    log_likelihood = 0.0

    for t in range(n):
        mu_pred = phi_val * mu
        P_pred = (phi_val ** 2) * P + q_val

        vol_t = vol[t]
        vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
        R = c_val * (vol_scalar ** 2)

        ret_t = returns[t]
        r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
        innovation = r_val - mu_pred

        S = P_pred + R
        if S <= 1e-12:
            S = 1e-12
        K = P_pred / S

        mu = mu_pred + K * innovation
        P = (1.0 - K) * P_pred
        P = float(max(P, 1e-12))

        mu_filtered[t] = mu
        P_filtered[t] = P

        ll_t = -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)
        loglik_trajectory[t] = ll_t
        log_likelihood += ll_t

    return mu_filtered, P_filtered, float(log_likelihood), loglik_trajectory


class PhiGaussianDriftModel:
    """Encapsulates Gaussian Kalman drift with persistence φ for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t.
        
        Uses Numba JIT-compiled kernel when available for 10-50x speedup.
        Falls back to pure Python implementation if Numba is unavailable.
        """
        if _USE_NUMBA:
            try:
                return run_phi_gaussian_filter(returns, vol, q, c, phi)
            except Exception:
                # Graceful fallback on any Numba execution error
                pass
        return _kalman_filter_phi(returns, vol, q, c, phi)

    @staticmethod
    def filter_python(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pure Python implementation (for testing and fallback)."""
        return _kalman_filter_phi(returns, vol, q, c, phi)

    @staticmethod
    def filter_with_trajectory(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        regime_id: str = "global",
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Kalman filter with per-timestep likelihood trajectory.
        
        Enables fold-aware CV likelihood slicing without re-execution.
        Uses deterministic result cache when available.
        
        Returns (mu_filtered, P_filtered, total_log_likelihood, loglik_trajectory).
        """
        if use_cache and _CACHE_AVAILABLE and FILTER_CACHE_ENABLED:
            return cached_phi_gaussian_filter(
                returns, vol, q, c, phi,
                filter_fn=_kalman_filter_phi_with_trajectory,
                regime_id=regime_id
            )
        return _kalman_filter_phi_with_trajectory(returns, vol, q, c, phi)

    @classmethod
    def optimize_params(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        phi_min: float = -0.999,
        phi_max: float = 0.999,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, float, Dict]:
        """Jointly optimize (q, c, φ) via maximum likelihood with Bayesian regularization."""
        n = len(returns)

        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))

        if vol_mean > 0:
            vol_cv = vol_std / vol_mean
        else:
            vol_cv = 0.0
        if ret_std > 0:
            rv_ratio = abs(ret_mean) / ret_std
        else:
            rv_ratio = 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))

        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window

        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        def negative_penalized_ll_cv_phi(params: np.ndarray) -> float:
            log_q, log_c, phi = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))

            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
                return 1e12

            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []

            for train_start, train_end, test_start, test_end in fold_splits:
                try:
                    ret_train = returns_robust[train_start:train_end]
                    vol_train = vol[train_start:train_end]

                    if len(ret_train) < 3:
                        continue

                    mu_filt_train, P_filt_train, _ = cls.filter(ret_train, vol_train, q, c, phi_clip)

                    mu_final = float(mu_filt_train[-1])
                    P_final = float(P_filt_train[-1])

                    ll_fold = 0.0
                    mu_pred = mu_final
                    P_pred = P_final

                    for t in range(test_start, test_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q

                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())

                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
                            standardized_innov = innovation / np.sqrt(forecast_var)
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(standardized_innov))
                            ll_fold += ll_contrib

                        K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (test_end - test_start)

                except Exception:
                    continue

            if total_obs == 0:
                return 1e12

            avg_ll_oos = total_ll_oos / max(total_obs, 1)

            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = norm.cdf(all_standardized)

                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)

                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)

                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)

                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass

            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_c_target = np.log10(0.9)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2
            
            # Explicit φ shrinkage prior (Gaussian)
            phi_lambda_effective = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale
            phi_tau = _lambda_to_tau(phi_lambda_effective)
            log_prior_phi = _phi_shrinkage_log_prior(
                phi_r=phi_clip,
                phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
                tau=phi_tau
            )

            penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + log_prior_phi + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)
        phi_grid = np.linspace(phi_min, phi_max, 5)
        log_q_grid = np.linspace(log_q_min, log_q_max, 4)
        log_c_grid = np.linspace(log_c_min, log_c_max, 3)

        best_neg_ll = float('inf')
        best_log_q_grid = adaptive_prior_mean
        best_log_c_grid = np.log10(0.9)
        best_phi_grid = 0.0

        for lq in log_q_grid:
            for lc in log_c_grid:
                for ph in phi_grid:
                    try:
                        neg_ll = negative_penalized_ll_cv_phi(np.array([lq, lc, ph]))
                        if neg_ll < best_neg_ll:
                            best_neg_ll = neg_ll
                            best_log_q_grid = lq
                            best_log_c_grid = lc
                            best_phi_grid = ph
                    except Exception:
                        continue

        grid_best_q = 10 ** best_log_q_grid
        grid_best_c = 10 ** best_log_c_grid
        grid_best_phi = float(np.clip(best_phi_grid, phi_min, phi_max))

        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max)]
        start_points = [
            np.array([best_log_q_grid, best_log_c_grid, best_phi_grid]),
            np.array([adaptive_prior_mean, np.log10(0.9), 0.0]),
            np.array([adaptive_prior_mean, np.log10(0.7), 0.3]),
            np.array([adaptive_prior_mean, np.log10(1.2), -0.3]),
            np.array([best_log_q_grid + 0.5, best_log_c_grid, best_phi_grid]),
            np.array([best_log_q_grid - 0.5, best_log_c_grid, best_phi_grid]),
            np.array([best_log_q_grid, best_log_c_grid + 0.2, best_phi_grid]),
            np.array([best_log_q_grid, best_log_c_grid - 0.2, best_phi_grid]),
        ]

        best_result = None
        best_fun = float('inf')

        for x0 in start_points:
            try:
                result = minimize(
                    negative_penalized_ll_cv_phi,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )

                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            log_q_opt, log_c_opt, phi_opt = best_result.x
            q_optimal = 10 ** log_q_opt
            c_optimal = 10 ** log_c_opt
            phi_optimal = float(np.clip(phi_opt, phi_min, phi_max))
            ll_optimal = -best_result.fun
        else:
            q_optimal = grid_best_q
            c_optimal = grid_best_c
            phi_optimal = grid_best_phi
            ll_optimal = -best_neg_ll

        # Compute φ shrinkage prior diagnostics for auditability
        n_obs_approx = len(returns)
        prior_scale_diag = 1.0 / max(n_obs_approx, 100)
        phi_lambda_eff_diag = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale_diag
        phi_tau_diag = _lambda_to_tau(phi_lambda_eff_diag)
        phi_prior_diag = _compute_phi_prior_diagnostics(
            phi_r=phi_optimal,
            phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
            tau=phi_tau_diag,
            log_likelihood=ll_optimal
        )

        diagnostics = {
            'grid_best_q': float(grid_best_q),
            'grid_best_c': float(grid_best_c),
            'refined_best_q': float(q_optimal),
            'refined_best_c': float(c_optimal),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_result is not None and (best_result.success if best_result else False),
            # φ shrinkage prior diagnostics (auditability)
            **phi_prior_diag,
        }

        return q_optimal, c_optimal, phi_optimal, ll_optimal, diagnostics
