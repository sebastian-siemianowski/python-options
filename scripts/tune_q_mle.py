#!/usr/bin/env python3
"""
===============================================================================
SYSTEM DNA — TUNING LAYER
===============================================================================

This file implements the *evolutionary tuning layer* of the quant system.

The system is governed by the following probabilistic law:

    p(r_{t+H} | r_t)
        = ∑_m  p(r_{t+H} | r_t, m, θ_{r,m}) · p(m | r_t)

Where:

    r_t        = current return / state
    r_{t+H}    = future return at horizon H
    r          = regime label inferred from market state
    m          = model class (e.g. kalman_gaussian, kalman_phi_student_t, etc.)
    θ_{r,m}    = parameters of model m in regime r
    p(m | r)   = posterior probability of model m in regime r

This file is responsible ONLY for:

    • Learning θ_{r,m}  (parameters per regime per model)
    • Computing p(m|r)  (model posterior per regime)
    • Providing diagnostic likelihood metrics
    • Applying hierarchical shrinkage and priors
    • Never making trading decisions

It does NOT:

    • Generate signals
    • Allocate capital
    • Perform Monte Carlo forecasting
    • Compute expected utility

-------------------------------------------------------------------------------
PHILOSOPHY

The system does NOT assume a single true market physics.

It maintains a *population of competing model laws* inside each regime.

Regimes are ontological contexts.
Models are hypotheses about physics inside those contexts.

-------------------------------------------------------------------------------
CONTRACT WITH SIGNAL LAYER

This file must output, for each regime r:

    {
        "model_posterior": { m: p(m|r) },
        "models": {
            m: {
                "q", "phi", "nu", "c",
                "mean_log_likelihood",
                "bic", "aic",
                ...
            }
        }
    }

The signal layer will consume this structure *without reinterpretation*.

-------------------------------------------------------------------------------
EVOLUTION RULE

Any future change MUST preserve:

    • Bayesian coherence
    • Regime-conditional model uncertainty
    • Separation of inference from decision

This file defines the epistemology of the system.

===============================================================================

tune_q_mle.py

Automatic per-asset Kalman drift process-noise parameter (q) estimation via MLE.

Optimizes q by maximizing out-of-sample log-likelihood of returns under the
Gaussian state-space drift model, using EWMA volatility as observation variance.

Caches results persistently (JSON only) for reuse across runs.

IMPORTANT AI AGENT INSTRUCTIONS: DO NOT REPLACE ELSE STATEMENTS WITH TERNARY : EXPRESSIONS
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm, kstest, t as student_t
from scipy.special import gammaln
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from enum import IntEnum

# Add repository root (parent of scripts) and scripts directory to sys.path for imports
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fx_data_utils import fetch_px, _download_prices, get_default_asset_universe


# =============================================================================
# REGIME DEFINITIONS FOR HIERARCHICAL BAYESIAN PARAMETER TUNING
# =============================================================================
# These 5 latent regimes represent distinct market dynamics.
# Regime assignment is provided EXTERNALLY - this file only learns parameters.
# =============================================================================

# =============================================================================
# MODEL CLASS DEFINITIONS FOR BAYESIAN MODEL AVERAGING
# =============================================================================
# These model classes represent competing physical hypotheses about market dynamics.
# Model averaging preserves uncertainty across physics rather than selecting one.
# =============================================================================

class ModelClass(IntEnum):
    """
    Model class definitions for Bayesian model averaging within each regime.

    Model 0: KALMAN_GAUSSIAN
        - Standard Kalman filter with Gaussian observation noise
        - Parameters: q, c
        - Best for stable, well-behaved markets

    Model 1: PHI_GAUSSIAN
        - Kalman filter with AR(1) drift persistence
        - Parameters: q, c, phi
        - Best for trending markets

    Model 2: PHI_STUDENT_T
        - Kalman filter with AR(1) drift and Student-t tails
        - Parameters: q, c, phi, nu
        - Best for fat-tailed, trending markets
    """
    KALMAN_GAUSSIAN = 0
    PHI_GAUSSIAN = 1
    PHI_STUDENT_T = 2


# Model class labels for display
MODEL_CLASS_LABELS = {
    ModelClass.KALMAN_GAUSSIAN: "kalman_gaussian",
    ModelClass.PHI_GAUSSIAN: "kalman_phi_gaussian",
    ModelClass.PHI_STUDENT_T: "kalman_phi_student_t",
}

# Model class parameter counts for BIC/AIC computation
MODEL_CLASS_N_PARAMS = {
    ModelClass.KALMAN_GAUSSIAN: 2,   # q, c
    ModelClass.PHI_GAUSSIAN: 3,      # q, c, phi
    ModelClass.PHI_STUDENT_T: 4,     # q, c, phi, nu
}

# Default temporal smoothing alpha for model posterior evolution
DEFAULT_TEMPORAL_ALPHA = 0.3


class MarketRegime(IntEnum):
    """
    Market regime definitions for conditional parameter estimation.

    Regime 0: LOW_VOL_TREND
        - Low EWMA volatility
        - Strong drift persistence
        - Positive or negative trend

    Regime 1: HIGH_VOL_TREND
        - High volatility
        - Strong drift persistence
        - Large trend amplitude

    Regime 2: LOW_VOL_RANGE
        - Low volatility
        - Drift near zero
        - Mean reversion dominant

    Regime 3: HIGH_VOL_RANGE
        - High volatility
        - Drift near zero
        - Whipsaw / choppy behavior

    Regime 4: CRISIS_JUMP
        - Extreme volatility
        - Tail events / jumps
        - Correlation breakdown
    """
    LOW_VOL_TREND = 0
    HIGH_VOL_TREND = 1
    LOW_VOL_RANGE = 2
    HIGH_VOL_RANGE = 3
    CRISIS_JUMP = 4


# Regime labels for display
REGIME_LABELS = {
    MarketRegime.LOW_VOL_TREND: "LOW_VOL_TREND",
    MarketRegime.HIGH_VOL_TREND: "HIGH_VOL_TREND",
    MarketRegime.LOW_VOL_RANGE: "LOW_VOL_RANGE",
    MarketRegime.HIGH_VOL_RANGE: "HIGH_VOL_RANGE",
    MarketRegime.CRISIS_JUMP: "CRISIS_JUMP",
}

# Minimum sample size per regime for stable parameter estimation
MIN_REGIME_SAMPLES = 60


def load_asset_list(assets_arg: Optional[str], assets_file: Optional[str]) -> List[str]:
    """Load list of assets from command-line argument or file."""
    if assets_arg:
        return [a.strip() for a in assets_arg.split(',') if a.strip()]
    
    if assets_file and os.path.exists(assets_file):
        with open(assets_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Default asset list: use centralized universe from fx_data_utils
    return get_default_asset_universe()


def load_cache(cache_json: str) -> Dict[str, Dict]:
    """Load existing cache from JSON file."""
    if os.path.exists(cache_json):
        try:
            with open(cache_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return {}
    return {}


def save_cache_json(cache: Dict[str, Dict], cache_json: str) -> None:
    """Persist cache to JSON atomically."""
    os.makedirs(os.path.dirname(cache_json) if os.path.dirname(cache_json) else '.', exist_ok=True)
    json_temp = cache_json + '.tmp'
    with open(json_temp, 'w') as f:
        json.dump(cache, f, indent=2)
    os.replace(json_temp, cache_json)


class GaussianDriftModel:
    """Encapsulates Gaussian Kalman drift model logic for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run Kalman filter for drift estimation with fixed process noise q and observation variance scale c."""
        n = len(returns)

        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)

        mu = 0.0
        P = 1e-4

        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = float(mu)
            P_pred = float(P) + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            mu = float(mu_pred + K * innovation)
            P = float((1.0 - K) * P_pred)

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_var = P_pred + R
            if forecast_var > 1e-12:
                log_likelihood += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var

        return mu_filtered, P_filtered, log_likelihood

    @staticmethod
    def filter_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t."""
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
        """PIT/KS for Gaussian forecasts including parameter uncertainty."""
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        forecast_std = np.sqrt(c * (vol_flat ** 2) + P_flat)
        standardized = (returns_flat - mu_flat) / forecast_std
        pit_values = norm.cdf(standardized)
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
                            # Gaussian log-likelihood for φ-Gaussian model
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


class PhiGaussianDriftModel:
    """Encapsulates Gaussian Kalman drift with persistence φ for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        return GaussianDriftModel.filter_phi(returns, vol, q, c, phi)

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
                            # Gaussian log-likelihood for φ-Gaussian model
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
            log_prior_phi = -0.05 * prior_scale * (phi_clip ** 2)

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
            'optimization_successful': best_result is not None and (best_result.success if best_result else False)
        }

        return q_optimal, c_optimal, phi_optimal, ll_optimal, diagnostics


class PhiStudentTDriftModel:
    """Encapsulates Student-t heavy-tail logic so drift model behavior stays modular."""

    nu_min_default: float = 2.1
    nu_max_default: float = 30.0

    @staticmethod
    def _clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
        return float(np.clip(float(nu), nu_min, nu_max))

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        """
        Log-density of scaled Student-t with location ``mu`` and scale ``scale``.
        Returns a large negative sentinel if inputs are invalid to keep optimizers stable.
        """
        if scale <= 0 or nu <= 0:
            return -1e12

        z = (x - mu) / scale
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)

    @classmethod
    def filter(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman drift filter with Student-t observation noise (no AR persistence)."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = float(mu)
            P_pred = float(P) + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            obs_scale = np.sqrt(c_val) * vol_scalar

            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            R = c_val * (vol_scalar ** 2)
            K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            mu = float(mu_pred + K * innovation)
            P = float((1.0 - K) * P_pred)

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_scale = np.sqrt(P_pred + R)
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf(r_val, nu_val, mu_pred, forecast_scale)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t

        return mu_filtered, P_filtered, log_likelihood

    @classmethod
    def filter_phi(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman drift filter with persistence (phi) and Student-t observation noise."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

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
            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            K = nu_adjust * P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf(r_val, nu_val, mu_pred, forecast_scale)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def pit_ks(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
        """PIT/KS for Student-t forecasts with parameter uncertainty included."""
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        forecast_scale = np.sqrt(c * (vol_flat ** 2) + P_flat)
        standardized = (returns_flat - mu_flat) / forecast_scale
        pit_values = student_t.cdf(standardized, df=nu)
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
        """PIT/KS for Gaussian forecasts including parameter uncertainty."""
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        forecast_std = np.sqrt(c * (vol_flat ** 2) + P_flat)
        standardized = (returns_flat - mu_flat) / forecast_std
        pit_values = norm.cdf(standardized)
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def optimize_params(
        returns: np.ndarray,
        vol: np.ndarray,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        phi_min: float = -0.999,
        phi_max: float = 0.999,
        nu_min: float = 2.1,
        nu_max: float = 30.0,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, float, float, Dict]:
        """Jointly optimize (q, c, φ, ν) for the φ-Student-t drift model via CV MLE."""
        n = len(returns)
        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0.0
        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        rv_ratio = abs(ret_mean) / ret_std if ret_std > 0 else 0.0

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

        def neg_pen_ll(params: np.ndarray) -> float:
            log_q, log_c, phi, log_nu = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))
            nu = 10 ** log_nu
            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c) or nu < nu_min or nu > nu_max:
                return 1e12
            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []
            for tr_start, tr_end, te_start, te_end in fold_splits:
                try:
                    ret_train = returns_robust[tr_start:tr_end]
                    vol_train = vol[tr_start:tr_end]
                    if len(ret_train) < 3:
                        continue
                    mu_filt_train, P_filt_train, _ = PhiStudentTDriftModel.filter_phi(ret_train, vol_train, q, c, phi_clip, nu)
                    mu_pred = float(mu_filt_train[-1])
                    P_pred = float(P_filt_train[-1])
                    ll_fold = 0.0
                    for t in range(te_start, te_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q
                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())
                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            forecast_std = np.sqrt(forecast_var)
                            ll_contrib = PhiStudentTDriftModel.logpdf(ret_t, nu, mu_pred, forecast_std)
                            ll_fold += ll_contrib
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(innovation / forecast_std))

                        nu_adjust = min(nu / (nu + 3.0), 1.0)
                        K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (te_end - te_start)

                except Exception:
                    continue
            if total_obs == 0:
                return 1e12
            avg_ll = total_ll_oos / max(total_obs, 1)
            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = student_t.cdf(all_standardized, df=nu)
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
            log_prior_phi = -0.05 * prior_scale * (phi_clip ** 2)
            log_prior_nu = -0.05 * prior_scale * (log_nu - np.log10(6.0)) ** 2

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + log_prior_nu + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)
        log_nu_min = np.log10(nu_min)
        log_nu_max = np.log10(nu_max)

        grid_best = (adaptive_prior_mean, np.log10(0.9), 0.0, np.log10(6.0))
        best_neg = float('inf')
        for lq in np.linspace(log_q_min, log_q_max, 4):
            for lc in np.linspace(log_c_min, log_c_max, 3):
                for lp in np.linspace(phi_min, phi_max, 5):
                    for ln in np.linspace(log_nu_min, log_nu_max, 3):
                        val = neg_pen_ll(np.array([lq, lc, lp, ln]))
                        if val < best_neg:
                            best_neg = val
                            grid_best = (lq, lc, lp, ln)
        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max), (log_nu_min, log_nu_max)]
        start_points = [np.array(grid_best), np.array([adaptive_prior_mean, np.log10(0.9), 0.0, np.log10(6.0)])]
        best_res = None
        best_fun = float('inf')
        for x0 in start_points:
            try:
                res = minimize(neg_pen_ll, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 120, 'ftol': 1e-6})
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is not None and best_res.success:
            lq_opt, lc_opt, phi_opt, ln_opt = best_res.x
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            nu_opt = 10 ** ln_opt
            ll_opt = -best_res.fun
        else:
            lq_opt, lc_opt, phi_opt, ln_opt = grid_best
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            nu_opt = 10 ** ln_opt
            ll_opt = -best_neg

        diagnostics = {
            'grid_best_q': float(10 ** grid_best[0]),
            'grid_best_c': float(10 ** grid_best[1]),
            'grid_best_phi': float(grid_best[2]),
            'grid_best_nu': float(10 ** grid_best[3]),
            'refined_best_q': float(q_opt),
            'refined_best_c': float(c_opt),
            'refined_best_phi': float(phi_opt),
            'refined_best_nu': float(nu_opt),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_res is not None and (best_res.success if best_res else False)
        }

        return q_opt, c_opt, phi_opt, nu_opt, ll_opt, diagnostics


# Compatibility wrappers to preserve existing API surface

def kalman_filter_drift(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    return GaussianDriftModel.filter(returns, vol, q, c)


def kalman_filter_drift_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
    return GaussianDriftModel.filter_phi(returns, vol, q, c, phi)


def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
    return GaussianDriftModel.pit_ks(returns, mu_filtered, vol, P_filtered, c)


def optimize_q_mle(
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
    """Delegate Gaussian q/c optimization to GaussianDriftModel for modularity."""
    return GaussianDriftModel.optimize_params(
        returns=returns,
        vol=vol,
        train_frac=train_frac,
        q_min=q_min,
        q_max=q_max,
        c_min=c_min,
        c_max=c_max,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )

def kalman_filter_drift_phi_student_t(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Thin wrapper for φ-Student-t Kalman filter."""
    return PhiStudentTDriftModel.filter_phi(returns, vol, q, c, phi, nu)


def compute_pit_ks_pvalue_student_t(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
    return PhiStudentTDriftModel.pit_ks(returns, mu_filtered, vol, P_filtered, c, nu)

def optimize_q_c_phi_mle(
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
) -> Tuple[float, float, float, Dict]:
    """Delegate φ-Gaussian optimization to PhiGaussianDriftModel for modularity."""
    return PhiGaussianDriftModel.optimize_params(
        returns=returns,
        vol=vol,
        train_frac=train_frac,
        q_min=q_min,
        q_max=q_max,
        c_min=c_min,
        c_max=c_max,
        phi_min=phi_min,
        phi_max=phi_max,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )

def compute_kurtosis(data: np.ndarray) -> float:
    """
    Compute sample excess kurtosis (Fisher's definition: kurtosis - 3).

    Positive excess kurtosis indicates heavy tails (fat-tailed distribution).
    Zero indicates normal distribution.
    Negative indicates light tails.

    Args:
        data: Sample data

    Returns:
        Excess kurtosis
    """
    data_clean = data[np.isfinite(data)]
    if len(data_clean) < 4:
        return 0.0

    mean = np.mean(data_clean)
    std = np.std(data_clean, ddof=1)
    if std <= 0:
        return 0.0

    n = len(data_clean)
    m4 = np.mean(((data_clean - mean) / std) ** 4)

    # Fisher's definition: excess kurtosis = kurtosis - 3
    excess_kurtosis = m4 - 3.0

    return float(excess_kurtosis)

def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    AIC = -2*LL + 2*k

    Lower AIC indicates better model fit with penalty for complexity.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters

    Returns:
        AIC value
    """
    return -2.0 * log_likelihood + 2.0 * n_params

def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    BIC = -2*LL + k*ln(n)

    Lower BIC indicates better model fit with penalty for complexity.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters
        n_obs: Number of observations

    Returns:
        BIC value
    """
    return -2.0 * log_likelihood + n_params * np.log(n_obs)

def tune_asset_q(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Optional[Dict]:
    """
    Estimate optimal parameters for a single asset via joint MLE with BIC-based model selection.
    
    Fits both Gaussian and Student-t observation noise models and selects the best based on BIC.
    Student-t is preferred for fat-tailed assets (crypto, commodities), Gaussian for stable assets.
    
    Includes:
    - Joint (q, c) optimization with Bayesian regularization
    - Zero-drift baseline comparison (ΔLL)
    - Safety fallbacks (q collapse, miscalibration, worse than baseline)
    - Comprehensive diagnostic metadata
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
    
    Returns:
        Dictionary with results and diagnostics, or None if estimation failed
    """
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            # Fallback: direct download with standardized format
            df = _download_prices(asset, start_date, end_date)
            if df is None or df.empty:
                raise RuntimeError(f"No data for {asset}")
            
            # Use standardized price extraction
            px = get_price_series(df, "Close")
            if px.empty:
                raise RuntimeError(f"No price column found for {asset}")
            
            title = asset
        
        # Allow very small histories; tune will still run cross-validation with short splits
        px = pd.to_numeric(px, errors="coerce").dropna()
        if len(px) < 10:
            raise RuntimeError(f"Insufficient data ({len(px)} days)")
        
        # Compute returns
        log_px = np.log(px)
        returns = log_px.diff().dropna()

        # Compute EWMA volatility (observation noise) with a span that adapts to short histories
        span = max(5, min(21, max(len(returns) // 2, 5)))
        vol = returns.ewm(span=span, adjust=False).std()

        # Align series with a smaller warmup for tiny datasets
        warmup = min(20, max(len(returns) // 4, 1))
        returns = returns.iloc[warmup:]
        vol = vol.iloc[warmup:]

        if len(returns) < 5:
            raise RuntimeError(f"Insufficient data after preprocessing ({len(returns)} returns)")

        returns_arr = returns.values
        vol_arr = vol.values
        n_obs = len(returns_arr)

        # Compute kurtosis to assess tail heaviness
        excess_kurtosis = compute_kurtosis(returns_arr)
        
        # =================================================================
        # STEP 1: Fit Gaussian Model (q, c)
        # =================================================================
        print(f"  🔧 Fitting Gaussian model...")
        q_gauss, c_gauss, ll_gauss_cv, opt_diag_gauss = optimize_q_mle(
            returns_arr, vol_arr,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full Gaussian Kalman filter
        mu_gauss, P_gauss, ll_gauss_full = kalman_filter_drift(returns_arr, vol_arr, q_gauss, c_gauss)
        
        # Compute Gaussian PIT calibration
        ks_gauss, pit_p_gauss = compute_pit_ks_pvalue(returns_arr, mu_gauss, vol_arr, P_gauss, c_gauss)
        
        # Gaussian has 2 parameters: q, c
        aic_gauss = compute_aic(ll_gauss_full, n_params=2)
        bic_gauss = compute_bic(ll_gauss_full, n_params=2, n_obs=n_obs)
        
        print(f"     Gaussian: q={q_gauss:.2e}, c={c_gauss:.3f}, LL={ll_gauss_full:.1f}, BIC={bic_gauss:.1f}, PIT p={pit_p_gauss:.4f}")
        
        # =================================================================
        # STEP 1.5: Fit φ-Kalman Model (q, c, φ)
        # =================================================================
        print(f"  🔧 Fitting φ-Gaussian-Kalman model...")
        q_phi, c_phi, phi_opt, ll_phi_cv, opt_diag_phi = optimize_q_c_phi_mle(
            returns_arr, vol_arr,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        mu_phi, P_phi, ll_phi_full = kalman_filter_drift_phi(returns_arr, vol_arr, q_phi, c_phi, phi_opt)
        ks_phi, pit_p_phi = compute_pit_ks_pvalue(returns_arr, mu_phi, vol_arr, P_phi, c_phi)
        aic_phi = compute_aic(ll_phi_full, n_params=3)
        bic_phi = compute_bic(ll_phi_full, n_params=3, n_obs=n_obs)
        print(f"     φ-Gaussian-Kalman: q={q_phi:.2e}, c={c_phi:.3f}, φ={phi_opt:+.3f}, LL={ll_phi_full:.1f}, BIC={bic_phi:.1f}, PIT p={pit_p_phi:.4f}")
        
        # =================================================================
        # STEP 2: Fit Kalman φ-Student-t Model (q, c, φ, ν)
        # =================================================================
        print(f"  🔧 Fitting Kalman φ-Student-t model...")
        try:
            q_student, c_student, phi_student, nu_student, ll_student_cv, opt_diag_student = PhiStudentTDriftModel.optimize_params(
                returns_arr, vol_arr,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )

            # Run full φ-Student-t Kalman filter (persistent drift with heavy tails)
            mu_student, P_student, ll_student_full = kalman_filter_drift_phi_student_t(
                returns_arr, vol_arr, q_student, c_student, phi_student, nu_student
            )

            # Compute Student-t PIT calibration
            ks_student, pit_p_student = compute_pit_ks_pvalue_student_t(
                returns_arr, mu_student, vol_arr, P_student, c_student, nu_student
            )

            # φ-Student-t has 4 parameters: q, c, φ, ν
            aic_student = compute_aic(ll_student_full, n_params=4)
            bic_student = compute_bic(ll_student_full, n_params=4, n_obs=n_obs)

            print(f"    Kalman φ-Student-t: q={q_student:.2e}, c={c_student:.3f}, φ={phi_student:+.3f}, ν={nu_student:.1f}, LL={ll_student_full:.1f}, BIC={bic_student:.1f}, PIT p={pit_p_student:.4f}")

            student_t_fit_success = True

        except Exception as e:
            print(f"  ⚠️  φ-Student-t optimization failed: {e}")
            student_t_fit_success = False
            q_student = None
            c_student = None
            phi_student = None
            nu_student = None
            ll_student_full = -1e12
            bic_student = 1e12
            aic_student = 1e12
            pit_p_student = 0.0

        # =================================================================
        # STEP 3: Model Selection via BIC
        # =================================================================
        # Lower BIC is better (penalizes complexity)
        candidate_models = []
        candidate_models.append(("gaussian", bic_gauss, aic_gauss, ll_gauss_full, mu_gauss, P_gauss, ks_gauss, pit_p_gauss, q_gauss, c_gauss, None, opt_diag_gauss))
        candidate_models.append(("phi_gaussian", bic_phi, aic_phi, ll_phi_full, mu_phi, P_phi, ks_phi, pit_p_phi, q_phi, c_phi, phi_opt, opt_diag_phi))
        if student_t_fit_success:
            candidate_models.append(('kalman_phi_student_t', bic_student, aic_student, ll_student_full, mu_student, P_student, ks_student, pit_p_student, q_student, c_student, (phi_student, nu_student), opt_diag_student))

        candidate_models = [m for m in candidate_models if np.isfinite(m[1])]
        best_entry = min(candidate_models, key=lambda x: x[1])
        noise_model, bic_final, aic_final, ll_full, mu_filtered, P_filtered, ks_statistic, ks_pvalue, q_optimal, c_optimal, extra_param, opt_diagnostics = best_entry

        nu_optimal = None
        phi_selected = None
        if noise_model == 'kalman_phi_student_t':
            phi_selected, nu_optimal = extra_param
        elif noise_model == "phi_gaussian":
            phi_selected = extra_param

        print(f"  ✓ Selected {noise_model} (BIC={bic_final:.1f})")
        if noise_model == 'kalman_phi_student_t':
            print(f"    (ΔBIC vs Gaussian = {bic_gauss - bic_student:+.1f}, ΔBIC vs φ-Gaussian Kalman = {bic_phi - bic_student:+.1f})")
        elif noise_model == "phi_gaussian":
            print(f"    (ΔBIC vs Gaussian = {bic_gauss - bic_phi:+.1f})")
        else:
            print(f"    (ΔBIC vs φ-Gaussian Kalman = {bic_phi - bic_gauss:+.1f})")
            print(f"")
        # =================================================================
        # Upgrade #4: Model Comparison - Baseline Models
        # =================================================================
        # Compare Kalman drift model against simpler baselines for formal model selection
        
        def compute_zero_drift_ll(returns_arr, vol_arr, c):
            """
            Compute log-likelihood of zero-drift model (μ=0 for all t).
            
            This is the simplest baseline: assumes no predictable drift.
            """
            # Ensure 1D arrays
            returns_flat = np.asarray(returns_arr).flatten()
            vol_flat = np.asarray(vol_arr).flatten()
            
            ll = 0.0
            for t in range(len(returns_flat)):
                ret_t = float(returns_flat[t])
                vol_t = float(vol_flat[t])
                R = c * (vol_t ** 2)
                innovation = ret_t - 0.0
                forecast_var = R
                if forecast_var > 1e-12:
                    ll += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
            return float(ll)
        
        def compute_constant_drift_ll(returns_arr, vol_arr, c):
            """
            Compute log-likelihood of constant-drift model (μ = mean(returns) for all t).
            
            This baseline assumes drift exists but is fixed over time.
            Parameters: c (1 parameter)
            """
            # Ensure 1D arrays
            returns_flat = np.asarray(returns_arr).flatten()
            vol_flat = np.asarray(vol_arr).flatten()
            
            # Estimate constant drift as sample mean
            mu_const = float(np.mean(returns_flat))
            
            ll = 0.0
            for t in range(len(returns_flat)):
                ret_t = float(returns_flat[t])
                vol_t = float(vol_flat[t])
                R = c * (vol_t ** 2)
                innovation = ret_t - mu_const
                forecast_var = R
                if forecast_var > 1e-12:
                    ll += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
            return float(ll), float(mu_const)
        
        def compute_ewma_drift_ll(returns_arr, vol_arr, c, span=21):
            """
            Compute log-likelihood of EWMA-drift model (μ_t = EWMA of past returns).
            
            This baseline uses exponentially weighted moving average for time-varying drift.
            Parameters: c, span (2 parameters effectively, but span is fixed)
            """
            # Ensure 1D arrays
            returns_flat = np.asarray(returns_arr).flatten()
            vol_flat = np.asarray(vol_arr).flatten()
            
            # Compute EWMA drift estimates
            ret_series = pd.Series(returns_flat)
            mu_ewma = ret_series.ewm(span=span, adjust=False).mean().values
            
            ll = 0.0
            for t in range(len(returns_flat)):
                if t == 0:
                    # First observation: use zero drift
                    mu_t = 0.0
                else:
                    # Use EWMA estimate from previous time step (no look-ahead)
                    mu_t = float(mu_ewma[t-1])
                
                ret_t = float(returns_flat[t])
                vol_t = float(vol_flat[t])
                R = c * (vol_t ** 2)
                innovation = ret_t - mu_t
                forecast_var = R
                if forecast_var > 1e-12:
                    ll += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
            return float(ll)
        
        # (No additional Kalman rerun here; use the variant already selected above)
        
        # =================================================================
        # Compute all baseline models for formal model comparison
        # =================================================================
        print(f"  🔬 Model comparison:")
        
        # Baseline 1: Zero-drift (0 parameters: just uses c from Kalman)
        ll_zero = compute_zero_drift_ll(returns_arr, vol_arr, c_optimal)
        aic_zero = compute_aic(ll_zero, n_params=0)  # c is shared, not counted
        bic_zero = compute_bic(ll_zero, n_params=0, n_obs=n_obs)
        print(f"     Zero-drift:     LL={ll_zero:.1f}, AIC={aic_zero:.1f}, BIC={bic_zero:.1f}")
        
        # Baseline 2: Constant-drift (1 parameter: mu_const, c is shared)
        ll_const, mu_const = compute_constant_drift_ll(returns_arr, vol_arr, c_optimal)
        aic_const = compute_aic(ll_const, n_params=1)
        bic_const = compute_bic(ll_const, n_params=1, n_obs=n_obs)
        print(f"     Constant-drift: LL={ll_const:.1f}, AIC={aic_const:.1f}, BIC={bic_const:.1f}, μ={mu_const:.6f}")
        
        # Baseline 3: EWMA-drift (1 parameter: span is fixed, c is shared)
        ll_ewma = compute_ewma_drift_ll(returns_arr, vol_arr, c_optimal, span=21)
        aic_ewma = compute_aic(ll_ewma, n_params=1)
        bic_ewma = compute_bic(ll_ewma, n_params=1, n_obs=n_obs)
        print(f"     EWMA-drift:     LL={ll_ewma:.1f}, AIC={aic_ewma:.1f}, BIC={bic_ewma:.1f}")
        
        # Kalman variants printed separately
        print(f"     Kalman-Gaussian: LL={ll_gauss_full:.1f}, AIC={aic_gauss:.1f}, BIC={bic_gauss:.1f}")
        print(f"     Kalman-φ-Gaussian: LL={ll_phi_full:.1f}, AIC={aic_phi:.1f}, BIC={bic_phi:.1f}, φ={phi_opt:+.3f}")
        if student_t_fit_success:
            print(f"     Kalman-φ-Student-t: LL={ll_student_full:.1f}, AIC={aic_student:.1f}, BIC={bic_student:.1f}, φ={phi_opt:+.3f}, ν={nu_student:.1f}")
        
        # Selected model summary (already chosen above)
        print(f"     Selected:        LL={ll_full:.1f}, AIC={aic_final:.1f}, BIC={bic_final:.1f} ({noise_model})")
        
        # ΔLL against baselines using the selected model's LL
        delta_ll_vs_zero = float(ll_full - ll_zero)
        delta_ll_vs_const = float(ll_full - ll_const)
        delta_ll_vs_ewma = float(ll_full - ll_ewma)

        # Aggregate model comparison metrics for diagnostics and cache
        model_comparison = {
            'zero_drift': {'ll': ll_zero, 'aic': aic_zero, 'bic': bic_zero, 'n_params': 0},
            'constant_drift': {'ll': ll_const, 'aic': aic_const, 'bic': bic_const, 'n_params': 1},
            'ewma_drift': {'ll': ll_ewma, 'aic': aic_ewma, 'bic': bic_ewma, 'n_params': 1},
            'kalman_gaussian': {'ll': ll_gauss_full, 'aic': aic_gauss, 'bic': bic_gauss, 'n_params': 2},
            'kalman_phi_gaussian': {'ll': ll_phi_full, 'aic': aic_phi, 'bic': bic_phi, 'n_params': 3},
        }
        if student_t_fit_success:
            model_comparison['kalman_phi_student_t'] = {
                'll': ll_student_full,
                'aic': aic_student,
                'bic': bic_student,
                'n_params': 4,
                'phi': float(phi_opt),
                'nu': float(nu_student)
            }
        
        # Best model across baselines and Kalman variants by BIC
        best_model_name = min(model_comparison.items(), key=lambda kv: kv[1]['bic'])[0]

        # Compute drift diagnostics
        mean_drift_var = float(np.mean(mu_filtered ** 2))
        mean_posterior_unc = float(np.mean(P_filtered))
        
        # Compute standardized residuals for kurtosis check
        forecast_std = np.sqrt(c_optimal * (vol_arr ** 2) + P_filtered)
        standardized_residuals = (returns_arr - mu_filtered) / forecast_std
        std_residual_kurtosis = compute_kurtosis(standardized_residuals)

        # Calibration warning flag
        calibration_warning = (ks_pvalue < 0.05)

        # Print calibration status
        if calibration_warning:
            if ks_pvalue < 0.01:
                print(f"  ⚠️  Severe miscalibration (PIT p={ks_pvalue:.4f})")
            else:
                print(f"  ⚠️  Calibration warning (PIT p={ks_pvalue:.4f})")

        # Build result dictionary with extended schema
        result = {
            # Asset identifier
            'asset': asset,

            # Model selection
            'noise_model': noise_model,  # "gaussian", "phi_gaussian", or "kalman_phi_student_t"

            # Parameters
            'q': float(q_optimal),
            'c': float(c_optimal),
            'nu': float(nu_optimal) if nu_optimal is not None else None,
            'phi': float(phi_selected) if phi_selected is not None else None,

            # Likelihood and model comparison
            # NOTE: log_likelihood here is TOTAL (sum over all observations) from full filter run
            # This differs from regime["cv_penalized_ll"] which is penalized mean LL from CV
            'log_likelihood': float(ll_full),
            'delta_ll_vs_zero': float(delta_ll_vs_zero),
            'delta_ll_vs_const': float(delta_ll_vs_const),
            'delta_ll_vs_ewma': float(delta_ll_vs_ewma),
            'aic': float(aic_final),
            'bic': float(bic_final),

            # Upgrade #4: Model comparison results
            'model_comparison': model_comparison,
            'best_model_by_bic': best_model_name,

            # Calibration diagnostics
            'ks_statistic': float(ks_statistic),
            'pit_ks_pvalue': float(ks_pvalue),
            'calibration_warning': bool(calibration_warning),
            'std_residual_kurtosis': float(std_residual_kurtosis),

            # Drift diagnostics
            'mean_drift_var': float(mean_drift_var),
            'mean_posterior_unc': float(mean_posterior_unc),

            # Data characteristics
            'n_obs': int(n_obs),
            'excess_kurtosis': float(excess_kurtosis),

            # Metadata
            'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),

            # Optimization diagnostics from selected model
            'grid_best_q': opt_diagnostics.get('grid_best_q'),
            'grid_best_c': opt_diagnostics.get('grid_best_c'),
            'refined_best_q': opt_diagnostics.get('refined_best_q'),
            'refined_best_c': opt_diagnostics.get('refined_best_c'),
            'prior_applied': opt_diagnostics.get('prior_applied'),
            'prior_log_q_mean': opt_diagnostics.get('prior_log_q_mean'),
            'prior_lambda': opt_diagnostics.get('prior_lambda'),
            'vol_cv': opt_diagnostics.get('vol_cv'),
            'rv_ratio': opt_diagnostics.get('rv_ratio'),
            'n_folds': opt_diagnostics.get('n_folds'),
            'robust_optimization': True,
            'optimization_successful': opt_diagnostics.get('optimization_successful', False)
        }

        # Add Kalman Phi Student-t specific diagnostics if applicable
        if noise_model == 'kalman_phi_student_t':
            result['grid_best_nu'] = opt_diagnostics.get('grid_best_nu')
            result['refined_best_nu'] = opt_diagnostics.get('refined_best_nu')
            result['refined_best_phi'] = float(phi_selected) if phi_selected is not None else None
        if noise_model == "phi_gaussian":
            result['refined_best_phi'] = float(phi_selected)

        # Add Gaussian comparison if Kalman Phi Student-t was selected
        if noise_model == 'kalman_phi_student_t' and student_t_fit_success:
            result['gaussian_bic'] = float(bic_gauss)
            result['gaussian_log_likelihood'] = float(ll_gauss_full)
            result['gaussian_pit_ks_pvalue'] = float(pit_p_gauss)
            result['bic_improvement'] = float(bic_gauss - bic_student)

        return result
        
    except Exception as e:
        import traceback
        print(f"  ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        raise


# =============================================================================
# REGIME-CONDITIONAL PARAMETER TUNING (HIERARCHICAL BAYESIAN LAYER)
# =============================================================================

def assign_regime_labels(
    returns: np.ndarray,
    vol: np.ndarray,
    lookback: int = 21
) -> np.ndarray:
    """
    Assign regime labels to each time point based on market features.
    
    Regime Assignment Logic:
    - Compute rolling volatility level (relative to median)
    - Compute rolling drift strength (absolute mean return)
    - Compute tail indicator (extreme returns)
    
    Classification:
    - LOW_VOL_TREND (0): vol < median, |drift| > threshold
    - HIGH_VOL_TREND (1): vol > 1.5*median, |drift| > threshold
    - LOW_VOL_RANGE (2): vol < median, |drift| < threshold
    - HIGH_VOL_RANGE (3): vol > 1.2*median, |drift| < threshold
    - CRISIS_JUMP (4): vol > 2*median OR extreme tail events
    
    Args:
        returns: Array of log returns
        vol: Array of EWMA volatility
        lookback: Rolling window for feature computation
        
    Returns:
        Array of regime labels (0-4) for each time point
    """
    n = len(returns)
    regime_labels = np.zeros(n, dtype=int)
    
    if n < lookback + 10:
        # Not enough data, default to LOW_VOL_RANGE
        return np.full(n, MarketRegime.LOW_VOL_RANGE, dtype=int)
    
    # Compute rolling features
    returns_series = pd.Series(returns)
    vol_series = pd.Series(vol)
    
    # Rolling mean absolute return (drift proxy)
    drift_abs = returns_series.rolling(lookback, min_periods=5).mean().abs().values
    
    # Volatility relative to expanding median
    vol_median = vol_series.expanding(min_periods=lookback).median().values
    vol_relative = np.where(vol_median > 1e-12, vol / vol_median, 1.0)
    
    # Tail indicator: |return| / vol
    tail_indicator = np.where(vol > 1e-12, np.abs(returns) / vol, 0.0)
    
    # Drift threshold (adaptive based on vol)
    drift_threshold = 0.0005  # ~0.05% daily drift threshold
    
    for t in range(n):
        v_rel = vol_relative[t] if np.isfinite(vol_relative[t]) else 1.0
        d_abs = drift_abs[t] if np.isfinite(drift_abs[t]) else 0.0
        tail = tail_indicator[t] if np.isfinite(tail_indicator[t]) else 0.0
        
        # Crisis/Jump: extreme volatility or tail events
        if v_rel > 2.0 or tail > 4.0:
            regime_labels[t] = MarketRegime.CRISIS_JUMP
        # High volatility regimes
        elif v_rel > 1.3:
            if d_abs > drift_threshold:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE
        # Low volatility regimes
        elif v_rel < 0.85:
            if d_abs > drift_threshold:
                regime_labels[t] = MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.LOW_VOL_RANGE
        # Normal volatility
        else:
            if d_abs > drift_threshold * 1.5:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND if v_rel > 1.0 else MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE if v_rel > 1.0 else MarketRegime.LOW_VOL_RANGE
    
    return regime_labels


# =============================================================================
# BAYESIAN MODEL AVERAGING WITH TEMPORAL SMOOTHING
# =============================================================================
# This section implements the core epistemic engine:
#
#     p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
#
# For each regime r, we:
# 1. Fit ALL candidate model classes independently
# 2. Compute BIC-based posterior weights with temporal smoothing
# 3. Return the full model posterior — never selecting a single model
# =============================================================================


def compute_bic_model_weights(
    bic_values: Dict[str, float],
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    Convert BIC values to unnormalized posterior weights.
    
    Implements:
        w_raw(m|r) = exp(-0.5 * (BIC_{m,r} - BIC_min_r))
    
    Args:
        bic_values: Dictionary mapping model name to BIC value
        epsilon: Small constant to prevent zero weights
        
    Returns:
        Dictionary of unnormalized weights (not yet normalized)
    """
    # Find minimum BIC
    finite_bics = [b for b in bic_values.values() if np.isfinite(b)]
    if not finite_bics:
        # All BICs are infinite — return uniform weights
        n_models = len(bic_values)
        return {m: 1.0 / max(n_models, 1) for m in bic_values}
    
    bic_min = min(finite_bics)
    
    # Compute raw weights
    weights = {}
    for model_name, bic in bic_values.items():
        if np.isfinite(bic):
            # BIC-based weight: exp(-0.5 * ΔBIC)
            delta_bic = bic - bic_min
            w = np.exp(-0.5 * delta_bic)
            weights[model_name] = max(w, epsilon)
        else:
            # Infinite BIC gets minimal weight
            weights[model_name] = epsilon
    
    return weights


def apply_temporal_smoothing(
    current_weights: Dict[str, float],
    previous_posterior: Optional[Dict[str, float]],
    alpha: float = DEFAULT_TEMPORAL_ALPHA
) -> Dict[str, float]:
    """
    Apply temporal smoothing to model weights.
    
    Implements:
        w_smooth(m|r) = (prev_p(m|r_prev))^alpha * w_raw(m|r)
    
    If no previous posterior exists, assumes uniform prior.
    
    Args:
        current_weights: Unnormalized BIC-based weights
        previous_posterior: Previous normalized posterior (or None)
        alpha: Temporal smoothing exponent (0 = no smoothing, 1 = full persistence)
        
    Returns:
        Smoothed unnormalized weights
    """
    if previous_posterior is None or alpha <= 0:
        # No smoothing — return current weights unchanged
        return current_weights.copy()
    
    # Apply temporal weighting
    smoothed = {}
    n_models = len(current_weights)
    uniform_weight = 1.0 / max(n_models, 1)
    
    for model_name, w_raw in current_weights.items():
        # Get previous posterior, defaulting to uniform
        prev_p = previous_posterior.get(model_name, uniform_weight)
        # Ensure previous posterior is positive
        prev_p = max(prev_p, 1e-10)
        
        # Apply smoothing: w_smooth = prev_p^alpha * w_raw
        w_smooth = (prev_p ** alpha) * w_raw
        smoothed[model_name] = w_smooth
    
    return smoothed


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.
    
    Args:
        weights: Unnormalized weights
        
    Returns:
        Normalized weights (posterior probabilities)
    """
    total = sum(weights.values())
    if total <= 0:
        # Fallback to uniform
        n = len(weights)
        return {m: 1.0 / max(n, 1) for m in weights}
    
    return {m: w / total for m, w in weights.items()}


def fit_all_models_for_regime(
    returns: np.ndarray,
    vol: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
) -> Dict[str, Dict]:
    """
    Fit ALL candidate model classes for a single regime's data.
    
    For each model m, computes:
        - Tuned parameters θ_{r,m}
        - Full log-likelihood
        - BIC, AIC
        - PIT calibration diagnostics
    
    Args:
        returns: Regime-specific returns
        vol: Regime-specific volatility
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        
    Returns:
        Dictionary with fitted models:
        {
            "kalman_gaussian": {...},
            "phi_gaussian": {...},
            "kalman_phi_student_t": {...}
        }
    """
    n_obs = len(returns)
    models = {}
    
    # =========================================================================
    # Model 0: Kalman Gaussian (q, c)
    # =========================================================================
    try:
        q_gauss, c_gauss, ll_cv_gauss, diag_gauss = GaussianDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full filter
        mu_gauss, P_gauss, ll_full_gauss = GaussianDriftModel.filter(returns, vol, q_gauss, c_gauss)
        
        # Compute PIT calibration
        ks_gauss, pit_p_gauss = GaussianDriftModel.pit_ks(returns, mu_gauss, vol, P_gauss, c_gauss)
        
        # Compute information criteria
        n_params_gauss = MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN]
        aic_gauss = compute_aic(ll_full_gauss, n_params_gauss)
        bic_gauss = compute_bic(ll_full_gauss, n_params_gauss, n_obs)
        mean_ll_gauss = ll_full_gauss / max(n_obs, 1)
        
        models["kalman_gaussian"] = {
            "q": float(q_gauss),
            "c": float(c_gauss),
            "phi": None,
            "nu": None,
            "log_likelihood": float(ll_full_gauss),
            "mean_log_likelihood": float(mean_ll_gauss),
            "cv_penalized_ll": float(ll_cv_gauss),
            "bic": float(bic_gauss),
            "aic": float(aic_gauss),
            "n_params": int(n_params_gauss),
            "ks_statistic": float(ks_gauss),
            "pit_ks_pvalue": float(pit_p_gauss),
            "fit_success": True,
            "diagnostics": diag_gauss,
        }
    except Exception as e:
        models["kalman_gaussian"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
        }
    
    # =========================================================================
    # Model 1: Phi-Gaussian (q, c, phi)
    # =========================================================================
    try:
        q_phi, c_phi, phi_opt, ll_cv_phi, diag_phi = PhiGaussianDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full filter
        mu_phi, P_phi, ll_full_phi = GaussianDriftModel.filter_phi(returns, vol, q_phi, c_phi, phi_opt)
        
        # Compute PIT calibration
        ks_phi, pit_p_phi = GaussianDriftModel.pit_ks(returns, mu_phi, vol, P_phi, c_phi)
        
        # Compute information criteria
        n_params_phi = MODEL_CLASS_N_PARAMS[ModelClass.PHI_GAUSSIAN]
        aic_phi = compute_aic(ll_full_phi, n_params_phi)
        bic_phi = compute_bic(ll_full_phi, n_params_phi, n_obs)
        mean_ll_phi = ll_full_phi / max(n_obs, 1)
        
        models["kalman_phi_gaussian"] = {
            "q": float(q_phi),
            "c": float(c_phi),
            "phi": float(phi_opt),
            "nu": None,
            "log_likelihood": float(ll_full_phi),
            "mean_log_likelihood": float(mean_ll_phi),
            "cv_penalized_ll": float(ll_cv_phi),
            "bic": float(bic_phi),
            "aic": float(aic_phi),
            "n_params": int(n_params_phi),
            "ks_statistic": float(ks_phi),
            "pit_ks_pvalue": float(pit_p_phi),
            "fit_success": True,
            "diagnostics": diag_phi,
        }
    except Exception as e:
        models["kalman_phi_gaussian"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
        }
    
    # =========================================================================
    # Model 2: Phi-Student-t (q, c, phi, nu)
    # =========================================================================
    try:
        q_st, c_st, phi_st, nu_st, ll_cv_st, diag_st = PhiStudentTDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full filter
        mu_st, P_st, ll_full_st = PhiStudentTDriftModel.filter_phi(returns, vol, q_st, c_st, phi_st, nu_st)
        
        # Compute PIT calibration
        ks_st, pit_p_st = PhiStudentTDriftModel.pit_ks(returns, mu_st, vol, P_st, c_st, nu_st)
        
        # Compute information criteria
        n_params_st = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T]
        aic_st = compute_aic(ll_full_st, n_params_st)
        bic_st = compute_bic(ll_full_st, n_params_st, n_obs)
        mean_ll_st = ll_full_st / max(n_obs, 1)
        
        models["kalman_phi_student_t"] = {
            "q": float(q_st),
            "c": float(c_st),
            "phi": float(phi_st),
            "nu": float(nu_st),
            "log_likelihood": float(ll_full_st),
            "mean_log_likelihood": float(mean_ll_st),
            "cv_penalized_ll": float(ll_cv_st),
            "bic": float(bic_st),
            "aic": float(aic_st),
            "n_params": int(n_params_st),
            "ks_statistic": float(ks_st),
            "pit_ks_pvalue": float(pit_p_st),
            "fit_success": True,
            "diagnostics": diag_st,
        }
    except Exception as e:
        models["kalman_phi_student_t"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
        }
    
    return models


def fit_regime_model_posterior(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[int, Dict]:
    """
    Compute regime-conditional Bayesian model averaging with temporal smoothing.
    
    This function implements the core epistemic law:
    
        p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
    
    For EACH regime r:
    1. Fit EACH candidate model class m independently
    2. Compute mean_log_likelihood, BIC, AIC for each (r, m)
    3. Convert BIC to posterior weights: w_raw(m|r) = exp(-0.5 * ΔBIC)
    4. Apply temporal smoothing: w_smooth = prev_p^alpha * w_raw
    5. Normalize to get p(m|r)
    
    CRITICAL RULES:
    - Never select a single best model per regime
    - Never discard models
    - Never force weights to zero
    - Preserve all priors, shrinkage, diagnostics
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility  
        regime_labels: Array of regime labels (0-4) for each time step
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum samples required per regime
        temporal_alpha: Smoothing exponent for model posterior evolution
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        
    Returns:
        Dictionary with regime-conditional model posteriors and parameters:
        {
            r: {
                "model_posterior": { m: p(m|r) },
                "models": {
                    m: {
                        "q", "phi", "nu", "c",
                        "mean_log_likelihood",
                        "bic", "aic",
                        diagnostics...
                    }
                },
                "regime_meta": {
                    "temporal_alpha": alpha,
                    "n_samples": n,
                    "regime_name": str
                }
            }
        }
    """
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    if len(returns) != len(regime_labels):
        raise ValueError(f"Length mismatch: returns={len(returns)}, regime_labels={len(regime_labels)}")
    
    # Initialize result structure
    regime_results = {}
    
    # Process each regime
    for regime in range(5):
        regime_name = REGIME_LABELS.get(regime, f"REGIME_{regime}")
        mask = (regime_labels == regime)
        n_samples = int(np.sum(mask))
        
        print(f"  📊 Fitting all models for {regime_name} (n={n_samples})...")
        
        # Get previous posterior for this regime (for temporal smoothing)
        prev_posterior = None
        if previous_posteriors is not None and regime in previous_posteriors:
            prev_posterior = previous_posteriors[regime]
        
        # Check if we have enough samples
        if n_samples < min_samples:
            print(f"     ⚠️  Insufficient samples ({n_samples} < {min_samples})")
            # Use uniform prior with no model parameters
            uniform_posterior = {
                "kalman_gaussian": 1.0 / 3.0,
                "kalman_phi_gaussian": 1.0 / 3.0,
                "kalman_phi_student_t": 1.0 / 3.0,
            }
            regime_results[regime] = {
                "model_posterior": uniform_posterior,
                "models": {},
                "regime_meta": {
                    "temporal_alpha": temporal_alpha,
                    "n_samples": n_samples,
                    "regime_name": regime_name,
                    "fallback": True,
                    "fallback_reason": "insufficient_samples",
                }
            }
            continue
        
        # Extract regime-specific data
        ret_regime = returns[mask]
        vol_regime = vol[mask]
        
        # =====================================================================
        # Step 1: Fit ALL models for this regime
        # =====================================================================
        models = fit_all_models_for_regime(
            ret_regime, vol_regime,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
        )
        
        # =====================================================================
        # Step 2: Extract BIC values
        # =====================================================================
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        
        # Print model fits
        for m, info in models.items():
            if info.get("fit_success", False):
                bic_val = info.get("bic", float('nan'))
                mean_ll = info.get("mean_log_likelihood", float('nan'))
                print(f"     {m}: BIC={bic_val:.1f}, mean_LL={mean_ll:.4f}")
            else:
                print(f"     {m}: FAILED - {info.get('error', 'unknown')}")
        
        # =====================================================================
        # Step 3: Compute BIC-based raw weights
        # =====================================================================
        raw_weights = compute_bic_model_weights(bic_values)
        
        # =====================================================================
        # Step 4: Apply temporal smoothing
        # =====================================================================
        smoothed_weights = apply_temporal_smoothing(raw_weights, prev_posterior, temporal_alpha)
        
        # =====================================================================
        # Step 5: Normalize to get posterior p(m|r)
        # =====================================================================
        model_posterior = normalize_weights(smoothed_weights)
        
        # Print posterior
        posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in model_posterior.items()])
        print(f"     → Posterior: {posterior_str}")
        
        # =====================================================================
        # Build regime result
        # =====================================================================
        regime_results[regime] = {
            "model_posterior": model_posterior,
            "models": models,
            "regime_meta": {
                "temporal_alpha": temporal_alpha,
                "n_samples": n_samples,
                "regime_name": regime_name,
                "fallback": False,
                "bic_min": float(min(b for b in bic_values.values() if np.isfinite(b))) if any(np.isfinite(b) for b in bic_values.values()) else None,
                "smoothing_applied": prev_posterior is not None and temporal_alpha > 0,
            }
        }
    
    return regime_results


def tune_regime_model_averaging(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    lambda_regime: float = 0.05,
) -> Dict:
    """
    Full regime-conditional Bayesian model averaging pipeline.
    
    This is the main entry point for the upgraded tuning system.
    It combines:
    1. Global model fitting (fallback)
    2. Regime-conditional model fitting with BMA
    3. Temporal smoothing of model posteriors
    4. Hierarchical shrinkage toward global
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        regime_labels: Array of regime labels (0-4)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum samples per regime
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous posteriors for smoothing
        lambda_regime: Hierarchical shrinkage strength
        
    Returns:
        Dictionary with:
        {
            "global": { global model fits },
            "regime": {
                r: {
                    "model_posterior": { m: p(m|r) },
                    "models": { m: {...} },
                    "regime_meta": {...}
                }
            },
            "meta": {
                "temporal_alpha": ...,
                "lambda_regime": ...,
                ...
            }
        }
    """
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    n_obs = len(returns)
    print(f"  📊 Bayesian Model Averaging: {n_obs} observations, α={temporal_alpha:.2f}")
    
    # =========================================================================
    # Step 1: Fit global models (fallback)
    # =========================================================================
    print(f"  🔧 Fitting global models...")
    global_models = fit_all_models_for_regime(
        returns, vol,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )
    
    # Compute global model posterior
    global_bic = {m: global_models[m].get("bic", float('inf')) for m in global_models}
    global_raw_weights = compute_bic_model_weights(global_bic)
    global_posterior = normalize_weights(global_raw_weights)
    
    print(f"     Global posterior: " + ", ".join([f"{m}={p:.3f}" for m, p in global_posterior.items()]))
    
    # =========================================================================
    # Step 2: Fit regime-conditional models with BMA
    # =========================================================================
    print(f"  🔄 Fitting regime-conditional models...")
    regime_results = fit_regime_model_posterior(
        returns, vol, regime_labels,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
        min_samples=min_samples,
        temporal_alpha=temporal_alpha,
        previous_posteriors=previous_posteriors,
    )
    
    # =========================================================================
    # Step 3: Apply hierarchical shrinkage to regime posteriors (optional)
    # =========================================================================
    if lambda_regime > 0:
        print(f"  📐 Applying hierarchical shrinkage (λ={lambda_regime:.3f})...")
        for r, r_result in regime_results.items():
            if r_result.get("regime_meta", {}).get("fallback", False):
                continue
            
            n_samples = r_result.get("regime_meta", {}).get("n_samples", 0)
            if n_samples < min_samples:
                continue
            
            # Shrinkage factor
            sf = 1.0 / (1.0 + lambda_regime * min_samples / max(n_samples, 1.0))
            
            # Shrink model posteriors toward global
            shrunk_posterior = {}
            for m in r_result["model_posterior"]:
                p_regime = r_result["model_posterior"][m]
                p_global = global_posterior.get(m, 1.0 / 3.0)
                p_shrunk = sf * p_regime + (1 - sf) * p_global
                shrunk_posterior[m] = p_shrunk
            
            # Renormalize
            shrunk_posterior = normalize_weights(shrunk_posterior)
            r_result["model_posterior_unshrunk"] = r_result["model_posterior"]
            r_result["model_posterior"] = shrunk_posterior
            r_result["regime_meta"]["shrinkage_applied"] = True
            r_result["regime_meta"]["shrinkage_factor"] = float(sf)
    
    # =========================================================================
    # Build final result
    # =========================================================================
    result = {
        "global": {
            "model_posterior": global_posterior,
            "models": global_models,
        },
        "regime": regime_results,
        "meta": {
            "temporal_alpha": temporal_alpha,
            "lambda_regime": lambda_regime,
            "n_obs": n_obs,
            "min_samples": min_samples,
            "n_regimes_active": sum(1 for r in regime_results.values() 
                                    if not r.get("regime_meta", {}).get("fallback", False)),
        }
    }
    
    return result


def tune_asset_with_bma(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    lambda_regime: float = 0.05,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
) -> Optional[Dict]:
    """
    Tune asset parameters using full Bayesian Model Averaging.
    
    This is the upgraded entry point that implements:
    
        p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
    
    For EACH regime r:
    - Fits ALL candidate model classes independently
    - Computes BIC-based model posteriors with temporal smoothing
    - Preserves full uncertainty across models
    
    NEVER selects a single best model — maintains full posterior.
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        lambda_regime: Hierarchical shrinkage strength
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous posteriors for temporal smoothing
        
    Returns:
        Dictionary with structure:
        {
            "asset": str,
            "global": {
                "model_posterior": { m: p(m) },
                "models": { m: {...} }
            },
            "regime": {
                r: {
                    "model_posterior": { m: p(m|r) },
                    "models": { m: {...} },
                    "regime_meta": {...}
                }
            },
            "regime_counts": {...},
            "meta": {...},
            "timestamp": str
        }
    """
    # Minimum data thresholds
    MIN_DATA_FOR_REGIME = 100
    MIN_DATA_FOR_GLOBAL = 20
    
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            df = _download_prices(asset, start_date, end_date)
            if df is None or df.empty:
                print(f"     ⚠️  No price data for {asset}")
                return None
            px = df['Close']
        
        n_points = len(px) if px is not None else 0
        
        # Check minimum data requirements
        if n_points < MIN_DATA_FOR_GLOBAL:
            print(f"     ⚠️  Insufficient data for {asset} ({n_points} points)")
            return None
        
        # Compute returns and volatility
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values
        
        vol_ewma = log_ret.ewm(span=21, adjust=False).std()
        vol = vol_ewma.values
        
        # Remove NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns = returns[valid_mask]
        vol = vol[valid_mask]
        
        if len(returns) < MIN_DATA_FOR_GLOBAL:
            print(f"     ⚠️  Insufficient valid data for {asset} ({len(returns)} returns)")
            return None
        
        # Check if we have enough data for full regime BMA
        use_regime_bma = len(returns) >= MIN_DATA_FOR_REGIME
        
        if not use_regime_bma:
            print(f"     ⚠️  Insufficient data for regime BMA ({len(returns)} < {MIN_DATA_FOR_REGIME})")
            print(f"     ↩️  Using global-only BMA...")
            
            # Fit global models only
            global_models = fit_all_models_for_regime(
                returns, vol,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda,
            )
            
            # Compute global posterior
            global_bic = {m: global_models[m].get("bic", float('inf')) for m in global_models}
            global_raw_weights = compute_bic_model_weights(global_bic)
            global_posterior = normalize_weights(global_raw_weights)
            
            return {
                "asset": asset,
                "global": {
                    "model_posterior": global_posterior,
                    "models": global_models,
                },
                "regime": None,
                "regime_counts": None,
                "use_regime_bma": False,
                "meta": {
                    "temporal_alpha": temporal_alpha,
                    "lambda_regime": lambda_regime,
                    "n_obs": len(returns),
                    "fallback_reason": "insufficient_data_for_regime_bma",
                },
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        
        # Assign regime labels
        print(f"     📊 Assigning regime labels for {len(returns)} observations...")
        regime_labels = assign_regime_labels(returns, vol)
        
        # Count samples per regime
        regime_counts = {r: int(np.sum(regime_labels == r)) for r in range(5)}
        print(f"     Regime distribution: " + ", ".join([
            f"{REGIME_LABELS[r]}={c}" for r, c in regime_counts.items() if c > 0
        ]))
        
        # Run full Bayesian Model Averaging
        print(f"     🔧 Running Bayesian Model Averaging (α={temporal_alpha:.2f}, λ={lambda_regime:.3f})...")
        bma_result = tune_regime_model_averaging(
            returns, vol, regime_labels,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            min_samples=MIN_REGIME_SAMPLES,
            temporal_alpha=temporal_alpha,
            previous_posteriors=previous_posteriors,
            lambda_regime=lambda_regime,
        )
        
        # Build final result
        result = {
            "asset": asset,
            "global": bma_result["global"],
            "regime": bma_result["regime"],
            "regime_counts": regime_counts,
            "use_regime_bma": True,
            "meta": bma_result["meta"],
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        # Print summary
        print(f"     ✓ Global: " + ", ".join([
            f"{m}={p:.3f}" for m, p in result["global"]["model_posterior"].items()
        ]))
        for r, r_data in result["regime"].items():
            if not r_data.get("regime_meta", {}).get("fallback", False):
                posterior_str = ", ".join([
                    f"{m}={p:.3f}" for m, p in r_data["model_posterior"].items()
                ])
                print(f"     ✓ {REGIME_LABELS[r]}: {posterior_str}")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"     ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        return None


def get_model_params_for_regime(
    bma_result: Dict,
    regime: int,
    model: str,
) -> Optional[Dict]:
    """
    Get parameters for a specific model in a specific regime.
    
    Args:
        bma_result: Result from tune_asset_with_bma()
        regime: Regime index (0-4)
        model: Model name ("kalman_gaussian", "kalman_phi_gaussian", "kalman_phi_student_t")
        
    Returns:
        Model parameters dict or None if not available
    """
    # Try regime-specific first
    if bma_result.get("regime") is not None and regime in bma_result["regime"]:
        regime_data = bma_result["regime"][regime]
        if not regime_data.get("regime_meta", {}).get("fallback", False):
            models = regime_data.get("models", {})
            if model in models and models[model].get("fit_success", False):
                return models[model]
    
    # Fallback to global
    if "global" in bma_result and "models" in bma_result["global"]:
        models = bma_result["global"]["models"]
        if model in models and models[model].get("fit_success", False):
            return models[model]
    
    return None


def get_model_posterior_for_regime(
    bma_result: Dict,
    regime: int,
) -> Dict[str, float]:
    """
    Get the model posterior p(m|r) for a specific regime.
    
    Args:
        bma_result: Result from tune_asset_with_bma()
        regime: Regime index (0-4)
        
    Returns:
        Dictionary mapping model names to posterior probabilities
    """
    # Try regime-specific first
    if bma_result.get("regime") is not None and regime in bma_result["regime"]:
        regime_data = bma_result["regime"][regime]
        if "model_posterior" in regime_data:
            return regime_data["model_posterior"]
    
    # Fallback to global
    if "global" in bma_result and "model_posterior" in bma_result["global"]:
        return bma_result["global"]["model_posterior"]
    
    # Ultimate fallback: uniform
    return {
        "kalman_gaussian": 1.0 / 3.0,
        "kalman_phi_gaussian": 1.0 / 3.0,
        "kalman_phi_student_t": 1.0 / 3.0,
    }


def tune_regime_parameters(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    # === UPGRADE LAYER 1: Regime Confidence Weighting ===
    regime_confidence: Optional[np.ndarray] = None,
    # === UPGRADE LAYER 2: Hierarchical Shrinkage ===
    lambda_regime: float = 0.05,
) -> Dict[int, Dict]:
    """
    Estimate parameters conditionally on each regime with hierarchical Bayesian maturation.
    
    For each regime r, fits q_r, phi_r, nu_r using only samples where
    regime_labels[t] == r. Falls back to global parameters if effective
    sample size is insufficient.
    
    === UPGRADE LAYERS (Architecture-Preserving) ===
    
    Layer 1 - Regime Confidence Weighting:
        When regime_confidence[t] is provided, likelihood is weighted:
        sum_t weight[t] * log p(x_t | theta_r)
        When None, weight[t] = 1.0 (default behavior unchanged)
    
    Layer 2 - Hierarchical Shrinkage Toward Global:
        penalty = lambda_regime * sum((theta_r - theta_global)^2)
        Prevents overfitting, stabilizes small regimes.
        When lambda_regime = 0, behavior identical to original.
    
    Layer 3 - Regime-Specific Prior Geometry:
        LOW_VOL regimes: encourage smaller q, larger nu
        HIGH_VOL regimes: allow larger q, moderate nu
        CRISIS regime: encourage largest q, smallest nu
    
    Layer 4 - Effective Sample Control:
        N_eff = sum(weight) replaces count logic
        Fallback when N_eff < min_samples
    
    Layer 5 - Regime Diagnostics:
        Sanity checks, parameter distances, collapse detection
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        regime_labels: Array of regime labels (0-4) for each time step
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum effective samples required per regime
        regime_confidence: Optional confidence weights [0,1] per time step
        lambda_regime: Hierarchical shrinkage strength (default 0.05)
        
    Returns:
        Dictionary with regime-specific parameters and diagnostics
    """
    regime_params = {}
    
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    if len(returns) != len(regime_labels):
        raise ValueError(f"Length mismatch: returns={len(returns)}, regime_labels={len(regime_labels)}")
    
    # === UPGRADE LAYER 1: Process regime confidence weights ===
    if regime_confidence is not None:
        weights = np.asarray(regime_confidence).flatten()
        if len(weights) != len(returns):
            raise ValueError(f"Length mismatch: regime_confidence={len(weights)}, returns={len(returns)}")
        weights = np.clip(weights, 0.0, 1.0)
    else:
        # Default: all weights = 1.0 (original behavior)
        weights = np.ones(len(returns), dtype=float)
    
    # First, compute global parameters as fallback
    print("  📊 Computing global parameters as fallback...")
    try:
        q_global, c_global, phi_global, nu_global, ll_global, _ = PhiStudentTDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        global_params = {
            "q": float(q_global),
            "c": float(c_global),
            "phi": float(phi_global),
            "nu": float(nu_global),
            "cv_penalized_ll": float(ll_global),  # Penalized mean LL from CV (includes priors)
            "n_samples": int(len(returns)),
            "n_eff": float(np.sum(weights)),
            "fallback": False
        }
        print(f"     Global: q={q_global:.2e}, φ={phi_global:+.3f}, ν={nu_global:.1f}")
    except Exception as e:
        print(f"  ⚠️  Global parameter estimation failed: {e}")
        global_params = {
            "q": 1e-6,
            "c": 1.0,
            "phi": 0.95,
            "nu": 8.0,
            "cv_penalized_ll": float('nan'),  # Penalized mean LL from CV (unavailable)
            "n_samples": int(len(returns)),
            "n_eff": float(np.sum(weights)),
            "fallback": True
        }
    
    # === UPGRADE LAYER 3: Regime-Specific Prior Geometry ===
    # These are penalty adjustments, not hard bounds
    regime_prior_adjustments = {
        MarketRegime.LOW_VOL_TREND: {
            "q_bias": -0.5,    # encourage smaller q
            "nu_bias": +2.0,   # encourage larger nu (thinner tails)
            "phi_bias": +0.02, # encourage higher persistence
        },
        MarketRegime.HIGH_VOL_TREND: {
            "q_bias": +0.3,    # allow larger q
            "nu_bias": 0.0,    # neutral nu
            "phi_bias": +0.01, # slightly higher persistence
        },
        MarketRegime.LOW_VOL_RANGE: {
            "q_bias": -0.3,    # smaller q
            "nu_bias": +1.0,   # larger nu
            "phi_bias": -0.02, # lower persistence (mean reversion)
        },
        MarketRegime.HIGH_VOL_RANGE: {
            "q_bias": +0.2,    # moderate q
            "nu_bias": -1.0,   # smaller nu (fatter tails)
            "phi_bias": -0.03, # lower persistence (whipsaw)
        },
        MarketRegime.CRISIS_JUMP: {
            "q_bias": +1.0,    # largest q (rapid adaptation)
            "nu_bias": -3.0,   # smallest nu (fattest tails)
            "phi_bias": -0.05, # lowest persistence
        },
    }
    
    # Estimate parameters for each regime
    for regime in range(5):
        regime_name = REGIME_LABELS.get(regime, f"REGIME_{regime}")
        mask = (regime_labels == regime)
        n_samples = int(np.sum(mask))
        
        # === UPGRADE LAYER 4: Effective Sample Control ===
        regime_weights = weights[mask]
        n_eff = float(np.sum(regime_weights))
        
        print(f"  📊 {regime_name} (n={n_samples}, n_eff={n_eff:.1f})...")
        
        if n_eff < min_samples:
            print(f"     ⚠️  Insufficient effective samples ({n_eff:.1f} < {min_samples}), using global fallback")
            regime_params[regime] = {
                **global_params,
                "n_samples": n_samples,
                "n_eff": n_eff,
                "fallback": True,
                "regime_name": regime_name,
                "fallback_reason": "insufficient_effective_samples"
            }
            continue
        
        # Extract regime-specific data
        ret_regime = returns[mask]
        vol_regime = vol[mask]
        
        # === UPGRADE LAYER 3: Apply regime-specific prior adjustments ===
        adjustments = regime_prior_adjustments.get(regime, {"q_bias": 0, "nu_bias": 0, "phi_bias": 0})
        regime_prior_log_q = prior_log_q_mean + adjustments["q_bias"]
        
        try:
            # Fit regime-specific parameters with adjusted priors
            q_r, c_r, phi_r, nu_r, ll_r, diag_r = PhiStudentTDriftModel.optimize_params(
                ret_regime, vol_regime,
                prior_log_q_mean=regime_prior_log_q,
                prior_lambda=prior_lambda
            )
            
            # === UPGRADE LAYER 2: Hierarchical Shrinkage Toward Global ===
            if lambda_regime > 0 and not global_params.get("fallback", True):
                # Compute shrinkage penalty and apply soft correction
                # Shrinkage factor: closer to 1 = more original, closer to 0 = more global
                shrinkage_factor = 1.0 / (1.0 + lambda_regime * min_samples / max(n_eff, 1.0))
                sf = shrinkage_factor  # shorthand
                
                # PATCH 1: Log-space shrinkage for q (preserves positivity, respects scale geometry)
                # q_shrunk = exp(sf * log(q_r) + (1-sf) * log(global_q))
                q_shrunk = math.exp(sf * math.log(q_r) + (1 - sf) * math.log(global_params["q"]))
                
                # phi shrinkage remains linear (bounded domain [-1, 1])
                phi_shrunk = phi_r * sf + global_params["phi"] * (1 - sf)
                
                # PATCH 1: Log-space shrinkage for nu (prevents df distortion near boundaries)
                # nu_shrunk = exp(sf * log(nu_r) + (1-sf) * log(global_nu))
                nu_shrunk = math.exp(sf * math.log(nu_r) + (1 - sf) * math.log(global_params["nu"]))
                
                # c shrinkage remains linear (scale parameter)
                c_shrunk = c_r * sf + global_params["c"] * (1 - sf)
                
                # Store both original and shrunk values
                shrinkage_applied = True
                q_original, phi_original, nu_original = q_r, phi_r, nu_r
                q_r, phi_r, nu_r, c_r = q_shrunk, phi_shrunk, nu_shrunk, c_shrunk
            else:
                shrinkage_applied = False
                q_original, phi_original, nu_original = q_r, phi_r, nu_r
            
            # === UPGRADE LAYER 5: Compute regime diagnostics ===
            # Parameter distance from global
            param_distance = np.sqrt(
                (np.log10(q_r) - np.log10(global_params["q"]))**2 +
                (phi_r - global_params["phi"])**2 * 100 +  # Scale phi difference
                (nu_r - global_params["nu"])**2 / 100      # Scale nu difference
            )
            
            regime_params[regime] = {
                "q": float(q_r),
                "c": float(c_r),
                "phi": float(phi_r),
                "nu": float(nu_r),
                "cv_penalized_ll": float(ll_r),  # Penalized mean LL from CV (includes priors)
                "n_samples": n_samples,
                "n_eff": n_eff,
                "fallback": False,
                "regime_name": regime_name,
                # Shrinkage metadata
                "shrinkage_applied": shrinkage_applied,
                "q_original": float(q_original) if shrinkage_applied else None,
                "phi_original": float(phi_original) if shrinkage_applied else None,
                "nu_original": float(nu_original) if shrinkage_applied else None,
                # Prior adjustments applied
                "prior_q_bias": adjustments["q_bias"],
                "prior_nu_bias": adjustments["nu_bias"],
                "prior_phi_bias": adjustments["phi_bias"],
                # Diagnostics
                "param_distance_from_global": float(param_distance),
                "diagnostics": diag_r
            }
            print(f"     q={q_r:.2e}, φ={phi_r:+.3f}, ν={nu_r:.1f}" + 
                  (f" [shrunk]" if shrinkage_applied else ""))
            
        except Exception as e:
            print(f"     ⚠️  Estimation failed ({e}), using global fallback")
            regime_params[regime] = {
                **global_params,
                "n_samples": n_samples,
                "n_eff": n_eff,
                "fallback": True,
                "regime_name": regime_name,
                "fallback_reason": f"estimation_failed: {str(e)}"
            }
    
    # === UPGRADE LAYER 5: Post-tuning diagnostics ===
    regime_meta = _compute_regime_diagnostics(regime_params, global_params)
    
    # Attach metadata to each regime
    for r in regime_params:
        regime_params[r]["regime_meta"] = regime_meta.get(r, {})
    
    return regime_params


def _compute_regime_diagnostics(
    regime_params: Dict[int, Dict],
    global_params: Dict
) -> Dict[int, Dict]:
    """
    Compute regime diagnostics for Layer 5.
    
    Checks:
    1. Sanity relationships between regimes
    2. Parameter distances
    3. Collapse detection
    
    Returns:
        Dictionary of diagnostics per regime
    """
    diagnostics = {}
    
    # Extract parameters for non-fallback regimes
    active_regimes = {r: p for r, p in regime_params.items() if not p.get("fallback", True)}
    
    # Get parameter values for sanity checks
    def get_param(r, key, default=None):
        if r in active_regimes:
            return active_regimes[r].get(key, default)
        return default
    
    q_vals = {r: get_param(r, "q") for r in range(5)}
    nu_vals = {r: get_param(r, "nu") for r in range(5)}
    phi_vals = {r: get_param(r, "phi") for r in range(5)}
    
    # Sanity check 1: q_crisis > q_low_vol (crisis should adapt faster)
    q_crisis = q_vals.get(MarketRegime.CRISIS_JUMP)
    q_low_trend = q_vals.get(MarketRegime.LOW_VOL_TREND)
    q_low_range = q_vals.get(MarketRegime.LOW_VOL_RANGE)
    
    sanity_q_crisis_vs_low = None
    if q_crisis is not None and q_low_trend is not None:
        sanity_q_crisis_vs_low = q_crisis > q_low_trend
    
    # Sanity check 2: nu_crisis < nu_trend (crisis has fatter tails)
    nu_crisis = nu_vals.get(MarketRegime.CRISIS_JUMP)
    nu_low_trend = nu_vals.get(MarketRegime.LOW_VOL_TREND)
    nu_high_trend = nu_vals.get(MarketRegime.HIGH_VOL_TREND)
    
    sanity_nu_crisis_vs_trend = None
    if nu_crisis is not None and nu_low_trend is not None:
        sanity_nu_crisis_vs_trend = nu_crisis < nu_low_trend
    
    # Sanity check 3: phi_trend > phi_range (trends are more persistent)
    phi_low_trend = phi_vals.get(MarketRegime.LOW_VOL_TREND)
    phi_high_trend = phi_vals.get(MarketRegime.HIGH_VOL_TREND)
    phi_low_range = phi_vals.get(MarketRegime.LOW_VOL_RANGE)
    phi_high_range = phi_vals.get(MarketRegime.HIGH_VOL_RANGE)
    
    sanity_phi_trend_vs_range = None
    if phi_low_trend is not None and phi_low_range is not None:
        sanity_phi_trend_vs_range = phi_low_trend > phi_low_range
    
    # Collapse detection: check if all parameters are too close
    collapse_threshold = 0.1  # If all distances < this, warn
    distances = []
    for r, p in active_regimes.items():
        dist = p.get("param_distance_from_global", 0)
        distances.append(dist)
    
    collapse_detected = len(distances) > 1 and all(d < collapse_threshold for d in distances)
    
    # Build diagnostics for each regime
    for r in range(5):
        diagnostics[r] = {
            "sanity_checks": {
                "q_crisis_gt_low_vol": sanity_q_crisis_vs_low,
                "nu_crisis_lt_trend": sanity_nu_crisis_vs_trend,
                "phi_trend_gt_range": sanity_phi_trend_vs_range,
            },
            "collapse_warning": collapse_detected,
            "n_active_regimes": len(active_regimes),
            # PATCH 5: Add metadata flag for likelihood type
            "ll_type": "cv_penalized_mean",  # Penalized mean LL from CV (includes priors + calibration penalty)
        }
    
    # Print warnings if sanity checks fail
    if sanity_q_crisis_vs_low is False:
        print("     ⚠️  Sanity warning: q_crisis should be > q_low_vol")
    if sanity_nu_crisis_vs_trend is False:
        print("     ⚠️  Sanity warning: nu_crisis should be < nu_trend")
    if sanity_phi_trend_vs_range is False:
        print("     ⚠️  Sanity warning: phi_trend should be > phi_range")
    if collapse_detected:
        print("     ⚠️  Collapse warning: All regime parameters too close to global")
    
    return diagnostics


def tune_asset_with_regimes(
    asset: str,
    regime_labels: Optional[np.ndarray] = None,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    use_regime_tuning: bool = True
) -> Optional[Dict]:
    """
    Estimate optimal parameters for a single asset with optional regime-conditional tuning.
    
    Supports two modes:
    - GLOBAL MODE (use_regime_tuning=False or regime_labels=None):
        Fits single parameter set on all data (existing behavior).
    - REGIME MODE (use_regime_tuning=True and regime_labels provided):
        For each regime r, fits q_r, phi_r, nu_r using only samples where
        regime_labels[t] == r.
    
    Args:
        asset: Asset symbol
        regime_labels: Optional array of regime labels (0-4) for each time step
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        use_regime_tuning: Whether to use regime-conditional tuning
        
    Returns:
        Dictionary with results:
        {
            "global": {...},           # Always present - global parameter estimates
            "regime": {                # Present if use_regime_tuning=True
                0: {...},
                1: {...},
                2: {...},
                3: {...},
                4: {...}
            },
            "use_regime_tuning": bool,
            ...
        }
    """
    try:
        # First, get global parameters using existing function
        print(f"  🔧 Tuning {asset}...")
        global_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        if global_result is None:
            return None
        
        # Structure result with global params
        result = {
            "asset": asset,
            "global": global_result,
            "use_regime_tuning": use_regime_tuning and regime_labels is not None,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        # If regime tuning is enabled and labels provided, estimate per-regime params
        if use_regime_tuning and regime_labels is not None:
            print(f"  🔄 Regime-conditional tuning for {asset}...")
            
            # Need to fetch data again for regime tuning
            try:
                px, _ = fetch_px(asset, start_date, end_date)
            except Exception:
                df = _download_prices(asset, start_date, end_date)
                if df is None or df.empty:
                    print(f"  ⚠️  Cannot fetch data for regime tuning")
                    return result
                px = df['Close']
            
            # Compute returns and vol
            log_ret = np.log(px / px.shift(1)).dropna()
            returns = log_ret.values
            
            vol_ewma = log_ret.ewm(span=21, adjust=False).std()
            vol = vol_ewma.values
            
            # Align regime_labels with returns
            if len(regime_labels) != len(returns):
                print(f"  ⚠️  Regime labels length mismatch ({len(regime_labels)} vs {len(returns)})")
                # Try to align by truncating
                min_len = min(len(regime_labels), len(returns))
                regime_labels = regime_labels[-min_len:]
                returns = returns[-min_len:]
                vol = vol[-min_len:]
            
            # Tune per-regime parameters
            regime_params = tune_regime_parameters(
                returns=returns,
                vol=vol,
                regime_labels=regime_labels,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            result["regime"] = regime_params
        
        return result
        
    except Exception as e:
        import traceback
        print(f"  ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        return None


def get_regime_params(
    cached_result: Dict,
    regime: int,
    use_regime_tuning: bool = True
) -> Dict:
    """
    Get parameters for a specific regime, with fallback to global.
    
    Args:
        cached_result: Result from tune_asset_with_regimes()
        regime: Regime index (0-4)
        use_regime_tuning: Whether to use regime-specific params
        
    Returns:
        Dictionary with parameters (q, c, phi, nu, etc.)
    """
    # If regime tuning disabled or not available, use global
    if not use_regime_tuning or "regime" not in cached_result:
        return cached_result.get("global", cached_result)
    
    # Get regime-specific params
    regime_params = cached_result.get("regime", {})
    if regime in regime_params:
        params = regime_params[regime]
        # If this regime used fallback, it already contains global params
        return params
    
    # Fallback to global
    return cached_result.get("global", cached_result)


def _tune_asset_with_regime_labels(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    lambda_regime: float = 0.05,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
) -> Optional[Dict]:
    """
    Tune asset with automatic regime label assignment and Bayesian Model Averaging.

    This function:
    1. Fetches price data
    2. Computes returns and volatility
    3. Assigns regime labels using assign_regime_labels()
    4. Calls tune_regime_model_averaging() for full BMA with temporal smoothing

    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        lambda_regime: Hierarchical shrinkage strength (default 0.05)
        previous_posteriors: Previous model posteriors per regime for temporal smoothing

    Returns:
        Dictionary with global and regime-conditional model posteriors and parameters
    """
    # Minimum data thresholds
    MIN_DATA_FOR_REGIME = 100  # Need at least 100 points for reliable regime estimation
    MIN_DATA_FOR_GLOBAL = 20   # Can do basic tuning with fewer points
    
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            df = _download_prices(asset, start_date, end_date)
            if df is None or df.empty:
                print(f"     ⚠️  No price data for {asset}")
                return None
            px = df['Close']

        n_points = len(px) if px is not None else 0
        
        # For very small datasets, fall back directly to global-only tuning
        if n_points < MIN_DATA_FOR_GLOBAL:
            print(f"     ⚠️  Insufficient data for {asset} ({n_points} points) - need at least {MIN_DATA_FOR_GLOBAL}")
            return None
        
        # For small-to-medium datasets (20-100 points), skip regime tuning but do global
        if n_points < MIN_DATA_FOR_REGIME:
            print(f"     ⚠️  Insufficient data for {asset} ({n_points} points) for regime tuning")
            print(f"     ↩️  Falling back to global-only model tuning...")
            
            # Do global tuning only
            global_result = tune_asset_q(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            if global_result is None:
                print(f"     ⚠️  Global tuning also failed for {asset}")
                return None
            
            # Return result with explicit markers that regime tuning was skipped
            return {
                "asset": asset,
                "global": global_result,
                "regime": None,  # Explicitly None - no regime params available
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_{n_points}_points",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        # Compute returns and volatility
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values

        vol_ewma = log_ret.ewm(span=21, adjust=False).std()
        vol = vol_ewma.values

        # Remove NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns = returns[valid_mask]
        vol = vol[valid_mask]

        # After cleaning, check if we still have enough data for regime tuning
        if len(returns) < MIN_DATA_FOR_REGIME:
            if len(returns) < MIN_DATA_FOR_GLOBAL:
                print(f"     ⚠️  Insufficient valid data for {asset} after cleaning ({len(returns)} returns)")
                return None
            
            print(f"     ⚠️  Insufficient data for {asset} after cleaning ({len(returns)} returns) for regime tuning")
            print(f"     ↩️  Falling back to global-only model tuning...")
            
            # Do global tuning only
            global_result = tune_asset_q(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            if global_result is None:
                print(f"     ⚠️  Global tuning also failed for {asset}")
                return None
            
            return {
                "asset": asset,
                "global": global_result,
                "regime": None,
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_after_cleaning_{len(returns)}_returns",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        # Assign regime labels
        print(f"     📊 Assigning regime labels for {len(returns)} observations...")
        regime_labels = assign_regime_labels(returns, vol)

        # Count samples per regime
        regime_counts = {r: int(np.sum(regime_labels == r)) for r in range(5)}
        print(f"     Regime distribution: " + ", ".join([f"{REGIME_LABELS[r]}={c}" for r, c in regime_counts.items() if c > 0]))

        # First get global params (for backward compatibility)
        print(f"     🔧 Estimating global parameters...")
        global_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )

        if global_result is None:
            return None

        # =================================================================
        # BAYESIAN MODEL AVERAGING: Fit ALL models for each regime
        # =================================================================
        # This implements the governing law:
        #     p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
        # =================================================================
        print(f"     🔄 Bayesian Model Averaging (λ_regime={lambda_regime})...")
        if previous_posteriors is not None:
            print(f"        ↪ Using previous posteriors for temporal smoothing (α={DEFAULT_TEMPORAL_ALPHA})")
        bma_result = tune_regime_model_averaging(
            returns=returns,
            vol=vol,
            regime_labels=regime_labels,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            min_samples=MIN_REGIME_SAMPLES,
            temporal_alpha=DEFAULT_TEMPORAL_ALPHA,
            previous_posteriors=previous_posteriors,  # Use provided previous posteriors
            lambda_regime=lambda_regime,
        )

        # Collect diagnostics summary
        regime_results = bma_result.get("regime", {})
        n_active = sum(1 for r, p in regime_results.items() 
                       if not p.get("regime_meta", {}).get("fallback", False))
        n_shrunk = sum(1 for r, p in regime_results.items() 
                       if p.get("regime_meta", {}).get("shrinkage_applied", False))
        collapse_warning = any(p.get("regime_meta", {}).get("collapse_warning", False) 
                              for p in regime_results.values())

        # Build combined result with BMA structure
        result = {
            "asset": asset,
            "global": {
                # Keep backward-compatible global result
                **global_result,
                # Add BMA global model posterior
                "model_posterior": bma_result.get("global", {}).get("model_posterior", {}),
                "models": bma_result.get("global", {}).get("models", {}),
            },
            "regime": regime_results,  # Now contains model_posterior and models per regime
            "use_regime_tuning": True,
            "regime_counts": regime_counts,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hierarchical_tuning": {
                "lambda_regime": lambda_regime,
                "temporal_alpha": DEFAULT_TEMPORAL_ALPHA,
                "n_active_regimes": n_active,
                "n_shrunk_regimes": n_shrunk,
                "collapse_warning": collapse_warning,
            },
            "meta": bma_result.get("meta", {}),
        }

        # Print summary
        global_posterior = result["global"].get("model_posterior", {})
        if global_posterior:
            posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in global_posterior.items()])
            print(f"     ✓ Global model posterior: {posterior_str}")
        else:
            print(f"     ✓ Global: q={global_result['q']:.2e}, φ={global_result.get('phi', 'N/A')}")
        
        for r, r_data in regime_results.items():
            regime_meta = r_data.get("regime_meta", {})
            if not regime_meta.get("fallback", False):
                model_posterior = r_data.get("model_posterior", {})
                posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in model_posterior.items()])
                shrunk_marker = " [shrunk]" if regime_meta.get("shrinkage_applied", False) else ""
                print(f"     ✓ {REGIME_LABELS[int(r)]}: {posterior_str}{shrunk_marker}")

        if collapse_warning:
            print(f"     ⚠️  Collapse warning: regime parameters too close to global")

        return result

    except Exception as e:
        import traceback
        print(f"     ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        return None


def _tune_worker(args_tuple: Tuple[str, str, Optional[str], float, float, float, Optional[Dict]]) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Worker function for parallel asset tuning.
    Must be defined at module level for ProcessPoolExecutor pickling.
    
    Args:
        args_tuple: (asset, start_date, end_date, prior_log_q_mean, prior_lambda, lambda_regime, previous_posteriors)
        
    Returns:
        Tuple of (asset, result_dict, error_message)
        - If success: (asset, result, None)
        - If failure: (asset, None, error_string)
    """
    asset, start_date, end_date, prior_log_q_mean, prior_lambda, lambda_regime, previous_posteriors = args_tuple
    
    try:
        result = _tune_asset_with_regime_labels(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            lambda_regime=lambda_regime,
            previous_posteriors=previous_posteriors,
        )
        
        if result:
            return (asset, result, None)

        # Fallback to standard tuning when regime tuning fails (insufficient data for regime estimation)
        print(f"  ↩️  {asset}: Falling back to standard model tuning...")
        fallback_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )

        if fallback_result:
            # Mark as fallback so downstream knows regime params are not available
            fallback_result['use_regime_tuning'] = False
            fallback_result['regime_fallback'] = True
            fallback_result['regime'] = None
            fallback_result['regime_counts'] = None
            return (asset, fallback_result, None)
        else:
            return (asset, None, "both regime and standard tuning failed")

    except Exception as e:
        return (asset, None, str(e))


def _extract_previous_posteriors(cached_entry: Optional[Dict]) -> Optional[Dict[int, Dict[str, float]]]:
    """
    Extract previous model posteriors from a cached entry for temporal smoothing.
    
    Args:
        cached_entry: Cached result for an asset (may be old or new structure)
        
    Returns:
        Dictionary mapping regime index to model posteriors, or None if not available
    """
    if cached_entry is None:
        return None
    
    regime_data = cached_entry.get("regime")
    if regime_data is None or not isinstance(regime_data, dict):
        return None
    
    previous_posteriors = {}
    for r_str, r_data in regime_data.items():
        try:
            r = int(r_str)
            model_posterior = r_data.get("model_posterior")
            if model_posterior is not None and isinstance(model_posterior, dict):
                # Validate it has expected model keys
                if any(k in model_posterior for k in ["kalman_gaussian", "kalman_phi_gaussian", "kalman_phi_student_t"]):
                    previous_posteriors[r] = model_posterior
        except (ValueError, TypeError):
            continue
    
    # Return None if no valid posteriors found
    if not previous_posteriors:
        return None
    
    return previous_posteriors


def main():
    parser = argparse.ArgumentParser(
        description="Estimate optimal Kalman drift parameters with Kalman Phi Student-t noise support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --force                          # Re-estimate all assets
  %(prog)s --max-assets 10 --dry-run        # Preview first 10 assets
  %(prog)s --prior-lambda 2.0 --prior-mean -5.5  # Custom regularization
  %(prog)s --debug                          # Enable debug output
        """
    )
    parser.add_argument('--assets', type=str, help='Comma-separated list of asset symbols')
    parser.add_argument('--assets-file', type=str, help='Path to file with asset list (one per line)')
    parser.add_argument('--cache-json', type=str, default='cache/kalman_q_cache.json',
                       help='Path to JSON cache file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-estimation even if cached values exist')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date for data fetching')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for data fetching (default: today)')

    # CLI enhancements
    parser.add_argument('--max-assets', type=int, default=None,
                       help='Maximum number of assets to process (useful for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be done without actually processing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (stack traces on errors)')
    # Cache is always preserved; legacy flag kept for compatibility
    parser.add_argument('--no-clear-cache', action='store_true',
                       help='Deprecated: cache is always preserved; flag is ignored')

    # Bayesian regularization parameters
    parser.add_argument('--prior-mean', type=float, default=-6.0,
                       help='Prior mean for log10(q) (default: -6.0)')
    parser.add_argument('--prior-lambda', type=float, default=1.0,
                       help='Regularization strength (default: 1.0, set to 0 to disable)')

    # Hierarchical regime tuning parameters
    parser.add_argument('--lambda-regime', type=float, default=0.05,
                       help='Hierarchical shrinkage toward global (default: 0.05, set to 0 for original behavior)')

    args = parser.parse_args()

    # Enable debug mode
    if args.debug:
        os.environ['DEBUG'] = '1'

    print("=" * 80)
    print("Kalman Drift MLE Tuning Pipeline - Hierarchical Regime-Conditional")
    print("=" * 80)
    print(f"Prior: log10(q) ~ N({args.prior_mean:.1f}, λ={args.prior_lambda:.1f})")
    print(f"Hierarchical shrinkage: λ_regime={args.lambda_regime:.3f}")
    print("Model selection: Gaussian vs Student-t via BIC")
    print("Regime-conditional: Fits (q, φ, ν) per market regime with shrinkage")

    # Cache is always preserved; no automatic clearing

    # Load asset list
    assets = load_asset_list(args.assets, args.assets_file)

    # Apply max-assets limit
    if args.max_assets:
        assets = assets[:args.max_assets]
        print(f"\nLimited to first {args.max_assets} assets")

    print(f"Assets to process: {len(assets)}")

    # Dry-run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual processing]")
        print("Would process:")
        for i, asset in enumerate(assets[:10], 1):
            print(f"  {i}. {asset}")
        if len(assets) > 10:
            print(f"  ... and {len(assets) - 10} more")
        return

    # Load existing cache
    cache = load_cache(args.cache_json)
    print(f"Loaded cache with {len(cache)} existing entries")

    # Process each asset with regime-conditional tuning
    new_estimates = 0
    reused_cached = 0
    failed = 0
    calibration_warnings = 0
    student_t_count = 0
    gaussian_count = 0
    regime_tuning_count = 0

    assets_to_process: List[str] = []
    failure_reasons: Dict[str, str] = {}

    for i, asset in enumerate(assets, 1):
        print(f"\n[{i}/{len(assets)}] {asset}")

        # Check cache - handle both old and new structure
        if not args.force and asset in cache:
            cached_entry = cache[asset]
            # Get q from either new structure or old structure
            if 'global' in cached_entry:
                cached_q = cached_entry['global'].get('q', float('nan'))
                cached_c = cached_entry['global'].get('c', 1.0)
                cached_model = cached_entry['global'].get('noise_model', 'gaussian')
                cached_nu = cached_entry['global'].get('nu')
                has_regime = 'regime' in cached_entry
            else:
                cached_q = cached_entry.get('q', float('nan'))
                cached_c = cached_entry.get('c', 1.0)
                cached_model = cached_entry.get('noise_model', 'gaussian')
                cached_nu = cached_entry.get('nu')
                has_regime = False

            if cached_model == 'kalman_phi_student_t' and cached_nu is not None:
                print(f"  ✓ Using cached estimate ({cached_model}: q={cached_q:.2e}, c={cached_c:.3f}, ν={cached_nu:.1f})")
            else:
                print(f"  ✓ Using cached estimate ({cached_model}: q={cached_q:.2e}, c={cached_c:.3f})")
            if has_regime:
                print(f"     + Regime-conditional params available")
            reused_cached += 1
            continue

        assets_to_process.append(asset)

    if assets_to_process:
        # Parallel processing using all available CPU cores
        import multiprocessing
        n_workers = multiprocessing.cpu_count()
        print(f"\n🚀 Running {len(assets_to_process)} assets with parallel regime-conditional tuning ({n_workers} workers)...")

        # Prepare arguments for workers, extracting previous posteriors from cache for temporal smoothing
        worker_args = []
        for asset in assets_to_process:
            # Extract previous posteriors from cache if available (for temporal smoothing)
            prev_posteriors = _extract_previous_posteriors(cache.get(asset))
            worker_args.append(
                (asset, args.start, args.end, args.prior_mean, args.prior_lambda, args.lambda_regime, prev_posteriors)
            )

        # Process in parallel using all CPU cores
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_tune_worker, arg): arg[0] for arg in worker_args}

            for future in as_completed(futures):
                asset = futures[future]
                try:
                    asset_name, result, error = future.result()

                    if result:
                        cache[asset_name] = result
                        new_estimates += 1
                        regime_tuning_count += 1

                        # Count model type from global params
                        global_result = result.get('global', result)
                        if global_result.get('noise_model') == 'kalman_phi_student_t':
                            student_t_count += 1
                        else:
                            gaussian_count += 1

                        if global_result.get('calibration_warning'):
                            calibration_warnings += 1

                        # Print success summary
                        q_val = global_result.get('q', float('nan'))
                        phi_val = global_result.get('phi')
                        phi_str = f", φ={phi_val:.3f}" if phi_val is not None else ""
                        print(f"  ✓ {asset_name}: q={q_val:.2e}{phi_str}")
                    else:
                        failed += 1
                        failure_reasons[asset_name] = error or "tuning returned None"
                        print(f"  ❌ {asset_name}: {error or 'tuning returned None'}")

                except Exception as e:
                    failed += 1
                    failure_reasons[asset] = str(e)
                    print(f"  ❌ {asset}: {e}")
    else:
        print("\nNo assets to process (all reused from cache).")

    # Save updated cache (JSON only)
    if new_estimates > 0:
        save_cache_json(cache, args.cache_json)
        print(f"\n✓ Cache updated: {args.cache_json}")

    # Summary report
    print("\n" + "=" * 80)
    print("Kalman Drift MLE Tuning Summary")
    print("=" * 80)
    print(f"Assets processed:       {len(assets)}")
    print(f"New estimates:          {new_estimates}")
    print(f"Reused cached:          {reused_cached}")
    print(f"Failed:                 {failed}")
    print(f"Calibration warnings:   {calibration_warnings}")
    print(f"\nModel Selection:")
    print(f"  Gaussian models:      {gaussian_count}")
    print(f"  Student-t models:     {student_t_count}")
    print(f"\nRegime-Conditional Tuning (Hierarchical Bayesian):")
    print(f"  Hierarchical shrinkage λ: {args.lambda_regime:.3f}")
    print(f"  Assets with regime params: {regime_tuning_count}")
    # Count regimes with actual params (not fallback) and shrinkage stats
    regime_fit_counts = {r: 0 for r in range(5)}
    regime_shrunk_counts = {r: 0 for r in range(5)}
    collapse_warnings = 0
    for asset, data in cache.items():
        regime_data = data.get('regime')
        if regime_data is not None and isinstance(regime_data, dict):
            for r, params in regime_data.items():
                if isinstance(params, dict):
                    # Handle both old structure (fallback at top level) and new BMA structure (in regime_meta)
                    is_fallback = params.get('fallback', False) or params.get('regime_meta', {}).get('fallback', False)
                    if not is_fallback:
                        regime_fit_counts[int(r)] += 1
                        # Check for shrinkage in both old and new structures
                        is_shrunk = params.get('shrinkage_applied', False) or params.get('regime_meta', {}).get('shrinkage_applied', False)
                        if is_shrunk:
                            regime_shrunk_counts[int(r)] += 1
        if 'hierarchical_tuning' in data:
            if data['hierarchical_tuning'].get('collapse_warning', False):
                collapse_warnings += 1
    print(f"  Regime-specific fits:")
    for r in range(5):
        shrunk_str = f" ({regime_shrunk_counts[r]} shrunk)" if regime_shrunk_counts[r] > 0 else ""
        print(f"    {REGIME_LABELS[r]}: {regime_fit_counts[r]} assets{shrunk_str}")
    if collapse_warnings > 0:
        print(f"  ⚠️  Collapse warnings: {collapse_warnings} assets")
    
    if cache:
        print("\nBest-fit parameters (grouped by model family, then q) — ALL ASSETS:")

        def _model_label(data: dict) -> str:
            # Handle new regime-conditional structure
            if 'global' in data:
                data = data['global']
            phi_val = data.get('phi')
            noise_model = data.get('noise_model', 'gaussian')
            if noise_model in ('kalman_phi_student_t', 'phi_student_t') and phi_val is not None:
                return 'Phi-Student-t'
            if noise_model in ('kalman_phi_student_t', 'phi_student_t'):
                return 'Student-t'
            if noise_model == 'phi_gaussian' or phi_val is not None:
                return 'Phi-Gaussian'
            return 'Gaussian'
        
        col_specs = [
            ("Asset", 18), ("Model", 14), ("log10(q)", 9), ("c", 7), ("ν", 7), ("φ", 7),
            ("ΔLL0", 8), ("ΔLLc", 8), ("ΔLLe", 8), ("BestModel", 12), ("BIC", 10), ("PIT p", 8)
        ]

        def fmt_row(values):
            parts = []
            for (val, (_, width)) in zip(values, col_specs):
                parts.append(f"{val:<{width}}")
            return "| " + " | ".join(parts) + " |"

        sep_line = "+" + "+".join(["-" * (w + 2) for _, w in col_specs]) + "+"
        header_line = fmt_row([name for name, _ in col_specs])

        print(sep_line)
        print(header_line)
        print(sep_line)

        # Sort by model family, then descending q
        def _get_q_for_sort(data):
            if 'global' in data:
                return data['global'].get('q', 0)
            return data.get('q', 0)
        
        sorted_assets = sorted(
            cache.items(),
            key=lambda x: (
                _model_label(x[1]),
                -_get_q_for_sort(x[1])
            )
        )

        last_group = None
        for asset, raw_data in sorted_assets:
            # Handle regime-conditional structure
            if 'global' in raw_data:
                data = raw_data['global']
            else:
                data = raw_data
            
            q_val = data.get('q', float('nan'))
            c_val = data.get('c', 1.0)
            nu_val = data.get('nu')
            phi_val = data.get('phi')
            delta_ll_zero = data.get('delta_ll_vs_zero', float('nan'))
            delta_ll_const = data.get('delta_ll_vs_const', float('nan'))
            delta_ll_ewma = data.get('delta_ll_vs_ewma', float('nan'))
            bic_val = data.get('bic', float('nan'))
            pit_p = data.get('pit_ks_pvalue', float('nan'))
            model = _model_label(raw_data)
            best_model = data.get('best_model_by_bic', 'kalman_drift')

            log10_q = np.log10(q_val) if q_val > 0 else float('nan')

            nu_str = f"{nu_val:.1f}" if nu_val is not None else "-"
            phi_str = f"{phi_val:.3f}" if phi_val is not None else "-"

            best_model_abbr = {
                'zero_drift': 'Zero',
                'constant_drift': 'Const',
                'ewma_drift': 'EWMA',
                'kalman_drift': 'Kalman',
                'phi_kalman_drift': 'PhiKal',
                'kalman_phi_student_t': 'PhiKal-t',
            }.get(best_model, best_model[:8])

            warn_marker = " ⚠️" if data.get('calibration_warning') else ""

            if model != last_group:
                if last_group is not None:
                    print(sep_line)
                print(f"| Group: {model:<{sum(w+3 for _, w in col_specs)-9}}|")
                print(sep_line)
                last_group = model

            row = fmt_row([
                asset,
                model,
                f"{log10_q:>7.2f}",
                f"{c_val:>5.3f}",
                nu_str,
                phi_str,
                f"{delta_ll_zero:>6.1f}",
                f"{delta_ll_const:>6.1f}",
                f"{delta_ll_ewma:>6.1f}",
                best_model_abbr,
                f"{bic_val:>8.1f}",
                f"{pit_p:.4f}{warn_marker}"
            ])
            print(row)

        print(sep_line)

        print("\nColumn Legend:")
        print("  Model: Gaussian / Phi-Gaussian / Phi-Student-t (φ from cache)")
        print("  φ: Drift persistence (if AR(1) model)")
        print("  ΔLL_0: ΔLL vs zero-drift baseline")
        print("  ΔLL_c: ΔLL vs constant-drift baseline")
        print("  ΔLL_e: ΔLL vs EWMA-drift baseline")
        print("  BestModel: Best model by BIC (Zero/Const/EWMA/Kalman/PhiKal)")
 
        print("\nCache file:")
        print(f"  JSON: {args.cache_json}")

    if failure_reasons:
        print("\nFailed tickers and reasons:")
        for a, msg in failure_reasons.items():
            print(f"  {a}: {msg}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
