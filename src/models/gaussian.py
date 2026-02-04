"""
===============================================================================
GAUSSIAN DRIFT MODEL — Kalman Filter with Gaussian Observation Noise
===============================================================================

Implements a local-level state-space model for drift estimation:

    State equation:    μ_t = μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,      ε_t ~ N(0, c·σ²_t)

Parameters:
    q: Process noise variance (drift evolution uncertainty)
    c: Observation noise scale (multiplier on EWMA variance)

The model is estimated via cross-validated MLE with Bayesian regularization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest, norm

# Numba wrappers for JIT-compiled filters (optional performance enhancement)
try:
    from .numba_wrappers import (
        is_numba_available,
        run_gaussian_filter,
        run_phi_gaussian_filter,
    )
    _USE_NUMBA = is_numba_available()
except ImportError:
    _USE_NUMBA = False
    run_gaussian_filter = None
    run_phi_gaussian_filter = None


class GaussianDriftModel:
    """Encapsulates Gaussian Kalman drift model logic for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Optimized Kalman filter for drift estimation.
        
        Performance optimizations (February 2026):
        - Pre-compute R array once
        - Pre-compute log_2pi constant
        - Use np.empty instead of np.zeros
        - Ensure contiguous array access
        """
        n = len(returns)
        
        # Convert to contiguous float64 arrays once
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)

        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
        
        # Pre-compute constants
        log_2pi = np.log(2 * np.pi)
        
        # Pre-compute R array (vectorized)
        R = c_val * (vol * vol)

        mu = 0.0
        P = 1e-4

        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = mu
            P_pred = P + q_val

            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12

            K = P_pred / S
            innovation = returns[t] - mu_pred

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            # Inlined log-likelihood
            log_likelihood += -0.5 * (log_2pi + np.log(S) + (innovation * innovation) / S)

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def filter_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t.
        
        Uses Numba JIT-compiled kernel when available (10-50× speedup).
        """
        # Try Numba-accelerated version first
        if _USE_NUMBA:
            try:
                return run_phi_gaussian_filter(returns, vol, q, c, phi)
            except Exception:
                pass  # Fall through to Python implementation
        
        return GaussianDriftModel._filter_phi_python(returns, vol, q, c, phi)
    
    @staticmethod
    def _filter_phi_python(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pure Python implementation of φ-Gaussian filter (for fallback and testing)."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
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

            log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def pit_ks(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
        """PIT/KS for Gaussian forecasts including parameter uncertainty.
        
        Computes the Probability Integral Transform (PIT) and performs a 
        Kolmogorov-Smirnov test for uniformity. If the model is well-calibrated,
        the PIT values should be uniformly distributed on [0, 1].
        
        Numerical stability: We enforce a minimum floor on forecast_std to prevent
        division by zero. When forecast_std is effectively zero, the model has
        collapsed to a degenerate distribution, which indicates calibration failure.
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Compute forecast standard deviation with numerical floor
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_std = np.sqrt(np.maximum(forecast_var, 1e-20))
        
        # Additional safety: ensure no zero values slip through
        forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
        
        standardized = (returns_flat - mu_flat) / forecast_std
        
        # Handle any remaining NaN/Inf values that could arise from edge cases
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            # All values invalid - return worst-case KS statistic
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        pit_values = norm.cdf(standardized_clean)
        
        # Need at least 2 points for KS test
        if len(pit_values) < 2:
            return 1.0, 0.0
            
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

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
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, Dict]:
        """Jointly optimize (q, c) via maximum likelihood with enhanced Bayesian regularization."""
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

        def negative_penalized_ll_cv(params: np.ndarray) -> float:
            log_q, log_c = params
            q = 10 ** log_q
            c = 10 ** log_c

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

                    mu_filt_train, P_filt_train, _ = cls.filter(returns_robust[train_start:train_end], vol_train, q, c)

                    mu_final = float(mu_filt_train[-1])
                    P_final = float(P_filt_train[-1])

                    ll_fold = 0.0
                    mu_pred = mu_final
                    P_pred = P_final

                    for t in range(test_start, test_end):
                        P_pred = P_pred + q

                        if np.ndim(returns_robust[t]) == 0:
                            ret_t = float(returns_robust[t])
                        else:
                            ret_t = float(returns_robust[t].item())
                        if np.ndim(vol[t]) == 0:
                            vol_t = float(vol[t])
                        else:
                            vol_t = float(vol[t].item())

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

            c_target = 0.9
            log_c_target = np.log10(c_target)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2

            penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + calibration_penalty

            if not np.isfinite(penalized_ll):
                return 1e12

            return -penalized_ll

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)

        log_q_grid = np.concatenate([
            np.linspace(log_q_min, adaptive_prior_mean - 1.0, 5),
            np.linspace(adaptive_prior_mean - 1.0, adaptive_prior_mean + 1.0, 7),
            np.linspace(adaptive_prior_mean + 1.0, log_q_max, 3)
        ])

        log_c_grid = np.concatenate([
            np.linspace(log_c_min, np.log10(0.7), 3),
            np.linspace(np.log10(0.7), np.log10(1.0), 7),
            np.linspace(np.log10(1.0), log_c_max, 2)
        ])

        best_neg_ll = float('inf')
        best_log_q_grid = adaptive_prior_mean
        best_log_c_grid = np.log10(0.9)

        for lq in log_q_grid:
            for lc in log_c_grid:
                try:
                    neg_ll = negative_penalized_ll_cv(np.array([lq, lc]))
                    if neg_ll < best_neg_ll:
                        best_neg_ll = neg_ll
                        best_log_q_grid = lq
                        best_log_c_grid = lc
                except Exception:
                    continue

        grid_best_q = 10 ** best_log_q_grid
        grid_best_c = 10 ** best_log_c_grid

        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max)]

        best_result = None
        best_fun = float('inf')

        start_points = [
            np.array([best_log_q_grid, best_log_c_grid]),
            np.array([adaptive_prior_mean, np.log10(0.9)]),
            np.array([adaptive_prior_mean, np.log10(0.7)]),
            np.array([adaptive_prior_mean, np.log10(1.2)]),
            np.array([best_log_q_grid - 0.5, best_log_c_grid]),
            np.array([best_log_q_grid + 0.5, best_log_c_grid]),
            np.array([best_log_q_grid, best_log_c_grid - 0.2]),
            np.array([best_log_q_grid, best_log_c_grid + 0.2]),
            np.array([-7.0, 0.0]),
            np.array([-5.0, 0.0]),
        ]

        for x0 in start_points:
            try:
                result = minimize(
                    negative_penalized_ll_cv,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-7}
                )

                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            log_q_opt, log_c_opt = best_result.x
            q_optimal = 10 ** log_q_opt
            c_optimal = 10 ** log_c_opt
            ll_optimal = -best_result.fun
        else:
            q_optimal = grid_best_q
            c_optimal = grid_best_c
            ll_optimal = -best_neg_ll

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
            'ret_mean': float(ret_mean),
            'ret_std': float(ret_std),
            'n_folds': int(len(fold_splits)),
            'adaptive_regularization': True,
            'robust_optimization': True,
            'winsorized': True,
            'optimization_successful': best_result is not None and (best_result.success if best_result else False)
        }

        return q_optimal, c_optimal, ll_optimal, diagnostics
