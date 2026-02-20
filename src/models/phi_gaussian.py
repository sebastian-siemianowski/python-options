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
from typing import Dict, Optional, Tuple, List, Callable

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
# ELITE TUNING CONFIGURATION (February 2026)
# =============================================================================
# Import from phi_student_t to maintain consistency
# =============================================================================
try:
    from .phi_student_t import (
        ELITE_TUNING_ENABLED,
        CURVATURE_PENALTY_WEIGHT,
        COHERENCE_PENALTY_WEIGHT,
        HESSIAN_EPSILON,
        MAX_CONDITION_NUMBER,
        _compute_curvature_penalty,
        _compute_coherence_penalty,
        _compute_fragility_index,
    )
    _ELITE_TUNING_AVAILABLE = True
except ImportError:
    _ELITE_TUNING_AVAILABLE = False
    ELITE_TUNING_ENABLED = False


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
        
        # ================================================================
        # ELITE FIX 2: Scale-aware q_min
        # ================================================================
        # Prevent deterministic state collapse by setting q_min relative
        # to observation variance. This ensures state evolution maintains
        # meaningful uncertainty.
        # ================================================================
        vol_var_median = float(np.median(vol ** 2))
        q_min_scaled = max(1e-10, 0.001 * vol_var_median)
        q_min = max(q_min, q_min_scaled, 1e-8)  # Hard floor at 1e-8

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
            
            # ================================================================
            # ELITE FIX 3: Strong φ-q regularization for deterministic collapse
            # ================================================================
            # When |φ| > 0.95: state becomes near-deterministic
            # When log_q < -7: no state uncertainty propagation
            # Both together cause overconfident forecasts → PIT failure
            #
            # Use STRONG penalty (500x) because φ→1 collapse is severe
            # ================================================================
            phi_near_one_penalty = max(0.0, abs(phi_clip) - 0.95) ** 2
            q_very_small_penalty = max(0.0, -7.0 - log_q) ** 2
            state_regularization = -500.0 * (phi_near_one_penalty + q_very_small_penalty)

            penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + log_prior_phi + calibration_penalty + state_regularization
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

        # =====================================================================
        # ELITE TUNING: Curvature and Fragility Analysis (February 2026)
        # =====================================================================
        elite_diagnostics = {}
        if _ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED:
            optimal_params = np.array([np.log10(q_optimal), np.log10(c_optimal), phi_optimal])
            
            try:
                curvature_penalty, condition_number, curv_diag = _compute_curvature_penalty(
                    negative_penalized_ll_cv_phi, optimal_params, bounds, 
                    HESSIAN_EPSILON, MAX_CONDITION_NUMBER
                )
                elite_diagnostics['curvature'] = curv_diag
                elite_diagnostics['condition_number'] = float(condition_number)
            except Exception:
                curvature_penalty = 0.0
                condition_number = 1.0
                elite_diagnostics['curvature'] = {'error': 'computation_failed'}
            
            try:
                fragility_index, frag_components = _compute_fragility_index(
                    condition_number, np.array([]), 0.0
                )
                elite_diagnostics['fragility_index'] = float(fragility_index)
                elite_diagnostics['fragility_components'] = frag_components
                elite_diagnostics['fragility_warning'] = fragility_index > 0.5
            except Exception:
                elite_diagnostics['fragility_index'] = 0.5
                elite_diagnostics['fragility_warning'] = False

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
            'elite_tuning_enabled': _ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED,
            'elite_diagnostics': elite_diagnostics if (_ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED) else None,
            # φ shrinkage prior diagnostics (auditability)
            **phi_prior_diag,
        }

        return q_optimal, c_optimal, phi_optimal, ll_optimal, diagnostics

    @classmethod
    def pit_ks_unified(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        variance_inflation: float = 1.0,
    ) -> Tuple[float, float, Dict]:
        """
        PIT/KS calibration for φ-Gaussian using PREDICTIVE distribution.
        
        ELITE IMPROVEMENTS (February 2026):
        - Variance inflation for calibration
        - Isotonic recalibration
        - Berkowitz LR test
        - PIT autocorrelation diagnostics
        
        Args:
            returns: Return series
            mu_pred: Predictive means from filter
            S_pred: Predictive variances from filter
            variance_inflation: Multiplier for predictive variance (default 1.0)
            
        Returns:
            Tuple of (ks_statistic, ks_pvalue, metrics_dict)
        """
        returns = np.asarray(returns).flatten()
        mu_pred = np.asarray(mu_pred).flatten()
        S_pred = np.asarray(S_pred).flatten()
        n = len(returns)
        
        # Compute raw PIT values using Gaussian CDF
        S_calibrated = S_pred * variance_inflation
        innovations = returns - mu_pred
        z = innovations / np.sqrt(np.maximum(S_calibrated, 1e-12))
        pit_raw = norm.cdf(z)
        pit_raw = np.clip(pit_raw, 0.001, 0.999)
        
        # Raw KS test
        ks_result_raw = kstest(pit_raw, 'uniform')
        ks_stat_raw = float(ks_result_raw.statistic)
        ks_pvalue_raw = float(ks_result_raw.pvalue)
        
        # =====================================================================
        # ELITE: Isotonic recalibration (from Student-t pipeline)
        # =====================================================================
        pit_clean = pit_raw
        ks_stat = ks_stat_raw
        ks_pvalue = ks_pvalue_raw
        isotonic_applied = False
        
        if ks_pvalue_raw < 0.10 and n >= 100:
            try:
                from calibration.isotonic_recalibration import IsotonicRecalibrator
                recalibrator = IsotonicRecalibrator()
                result = recalibrator.fit(pit_raw)
                if not result.is_identity:
                    pit_clean = recalibrator.transform(pit_raw)
                    ks_result_cal = kstest(pit_clean, 'uniform')
                    # Only use if it improves calibration
                    if ks_result_cal.pvalue > ks_pvalue_raw:
                        ks_stat = float(ks_result_cal.statistic)
                        ks_pvalue = float(ks_result_cal.pvalue)
                        isotonic_applied = True
            except ImportError:
                pass
        
        # Histogram MAD
        hist, _ = np.histogram(pit_clean, bins=10, range=(0, 1))
        hist_freq = hist / len(pit_clean)
        hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
        
        # Calibration grade
        if hist_mad < 0.02:
            grade = "A"
        elif hist_mad < 0.05:
            grade = "B"
        elif hist_mad < 0.10:
            grade = "C"
        else:
            grade = "F"
        
        # =====================================================================
        # ELITE: Berkowitz LR test + PIT autocorrelation
        # =====================================================================
        berkowitz_lr, berkowitz_p = float('nan'), float('nan')
        pit_acf = {}
        try:
            from .elite_pit_diagnostics import compute_berkowitz_lr_test, compute_pit_autocorrelation
            berkowitz_lr, berkowitz_p, _ = compute_berkowitz_lr_test(pit_clean)
            pit_acf = compute_pit_autocorrelation(pit_clean)
        except ImportError:
            pass
        
        metrics = {
            "n_samples": len(pit_clean),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "ks_pvalue_raw": ks_pvalue_raw,
            "ks_improvement": float(ks_pvalue - ks_pvalue_raw),
            "histogram_mad": hist_mad,
            "calibration_grade": grade,
            "calibrated": hist_mad < 0.05,
            "variance_inflation": variance_inflation,
            "isotonic_applied": isotonic_applied,
            # ELITE diagnostics
            "berkowitz_lr": float(berkowitz_lr) if np.isfinite(berkowitz_lr) else None,
            "berkowitz_pvalue": float(berkowitz_p) if np.isfinite(berkowitz_p) else None,
            "pit_autocorr_lag1": pit_acf.get('autocorrelations', {}).get('lag_1'),
            "ljung_box_pvalue": pit_acf.get('ljung_box_pvalue'),
            "has_dynamic_misspec": pit_acf.get('has_autocorrelation', False),
        }
        
        return ks_stat, ks_pvalue, metrics

    @classmethod
    def optimize_variance_inflation(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        train_frac: float = 0.7,
    ) -> float:
        """
        Grid search for optimal variance inflation to minimize PIT MAD.
        
        ELITE IMPROVEMENT (February 2026): Auto-calibrate observation noise
        to achieve uniform PIT distribution.
        
        Args:
            returns, vol: Data arrays
            q, c, phi: Base model parameters
            train_frac: Fraction for training
            
        Returns:
            Optimal variance inflation factor
        """
        n = len(returns)
        n_train = int(n * train_frac)
        returns_train = returns[:n_train]
        vol_train = vol[:n_train]
        
        def compute_pit_mad(vi):
            try:
                mu_filt, P_filt, _ = cls.filter(returns_train, vol_train, q, c * vi, phi)
            except Exception:
                return 1.0
            
            pit_values = []
            for t in range(len(returns_train)):
                if t == 0:
                    mu_pred = 0.0
                    P_pred = 1e-4 + q
                else:
                    mu_pred = phi * mu_filt[t-1]
                    P_pred = (phi ** 2) * P_filt[t-1] + q
                
                R = c * vi * (vol_train[t] ** 2)
                S = P_pred + R
                z = (returns_train[t] - mu_pred) / np.sqrt(max(S, 1e-12))
                pit_values.append(norm.cdf(z))
            
            pit_values = np.clip(pit_values, 0.001, 0.999)
            hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
            return float(np.mean(np.abs(hist / len(pit_values) - 0.1)))
        
        # Grid search
        best_vi, best_mad = 1.0, compute_pit_mad(1.0)
        for vi in [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
            mad = compute_pit_mad(vi)
            if mad < best_mad:
                best_mad, best_vi = mad, vi
        
        return best_vi
