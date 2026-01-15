#!/usr/bin/env python3
"""
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

# Add repository root (parent of scripts) and scripts directory to sys.path for imports
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fx_data_utils import fetch_px, _download_prices, get_default_asset_universe


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


def kalman_filter_drift(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Kalman filter for drift estimation with fixed process noise q and observation variance scale c.
    
    State-space model:
        μ_t = μ_{t-1} + w_t,  w_t ~ N(0, q)         (state evolution)
        r_t = μ_t + v_t,      v_t ~ N(0, c·σ_t²)    (observation with scaled variance)
    
    Args:
        returns: Observed returns
        vol: EWMA volatility estimates
        q: Process noise variance (drift evolution)
        c: Observation variance scale factor (corrects EWMA bias)
    
    Returns:
        mu_filtered: Posterior mean of drift at each time step
        P_filtered: Posterior variance of drift at each time step
        log_likelihood: Total log-likelihood of observations
    """
    n = len(returns)
    
    # Ensure q and c are scalars
    q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
    c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
    
    # Initialize state
    mu = 0.0  # Initial drift estimate
    P = 1e-4  # Initial uncertainty
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        # Predict
        mu_pred = float(mu)
        P_pred = float(P) + q_val
        
        # Observation variance with scale factor (extract scalar from array)
        vol_t = vol[t]
        vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
        R = c_val * (vol_scalar ** 2)
        
        # Update (Kalman gain)
        K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
        
        # Innovation (extract scalar from array)
        ret_t = returns[t]
        r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
        innovation = r_val - mu_pred
        
        # Update state (keep as Python float)
        mu = float(mu_pred + K * innovation)
        P = float((1.0 - K) * P_pred)
        
        # Store filtered estimates
        mu_filtered[t] = mu
        P_filtered[t] = P
        
        # Accumulate log-likelihood: log p(r_t | past) = log N(r_t; μ_pred, P_pred + R)
        forecast_var = P_pred + R
        if forecast_var > 1e-12:
            log_likelihood += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
    
    return mu_filtered, P_filtered, log_likelihood


def student_t_logpdf(x: float, nu: float, mu: float, scale: float) -> float:
    """
    Compute log-probability density of scaled Student-t distribution.
    
    Student-t with location mu, scale sigma, and degrees of freedom nu:
        p(x | nu, mu, sigma) = Γ((nu+1)/2) / (Γ(nu/2) * sqrt(nu*pi*sigma²)) 
                              * [1 + (x-mu)²/(nu*sigma²)]^(-(nu+1)/2)
    
    Args:
        x: Observation value
        nu: Degrees of freedom (nu > 0, typically nu >= 2 for finite variance)
        mu: Location parameter (mean for nu > 1)
        scale: Scale parameter (related to variance: Var = scale² * nu/(nu-2) for nu > 2)
    
    Returns:
        Log-probability density
    """
    if scale <= 0 or nu <= 0:
        return -1e12
    
    # Standardized residual
    z = (x - mu) / scale
    
    # Log-density using gammaln for numerical stability
    log_norm = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * (scale ** 2))
    log_kernel = -((nu + 1) / 2) * np.log(1 + (z ** 2) / nu)
    
    return float(log_norm + log_kernel)


def kalman_filter_drift_student_t(returns: np.ndarray, vol: np.ndarray, q: float, c: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Kalman filter for drift estimation with Student-t observation noise.
    
    State-space model:
        μ_t = μ_{t-1} + w_t,  w_t ~ N(0, q)         (state evolution - Gaussian)
        r_t = μ_t + v_t,      v_t ~ t(nu, 0, c·σ_t) (observation - Student-t with heavy tails)
    
    This handles fat-tailed returns (crypto, commodities) better than Gaussian noise.
    As nu → ∞, converges to Gaussian Kalman filter.
    
    Args:
        returns: Observed returns
        vol: EWMA volatility estimates
        q: Process noise variance (drift evolution)
        c: Observation variance scale factor
        nu: Student-t degrees of freedom (2 < nu < 30 typically)
    
    Returns:
        mu_filtered: Posterior mean of drift at each time step
        P_filtered: Posterior variance of drift at each time step
        log_likelihood: Total log-likelihood of observations under Student-t
    """
    n = len(returns)
    
    # Ensure scalar parameters
    q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
    c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
    nu_val = float(nu) if np.ndim(nu) == 0 else float(nu.item()) if hasattr(nu, 'item') else float(nu)
    
    # Validate nu
    if nu_val < 2.1 or nu_val > 30:
        nu_val = float(np.clip(nu_val, 2.1, 30))
    
    # Initialize state
    mu = 0.0
    P = 1e-4
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        # Predict
        mu_pred = float(mu)
        P_pred = float(P) + q_val
        
        # Observation scale
        vol_t = vol[t]
        vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
        obs_scale = np.sqrt(c_val) * vol_scalar
        
        # For Student-t, use robust Kalman gain approximation
        # Standard gain assumes Gaussian, but we adjust for heavy tails
        # For nu > 6, behavior is close to Gaussian; for nu ≈ 4, more conservative
        nu_adjust = min(nu_val / (nu_val + 3), 1.0)  # Shrinks gain for low nu
        
        # Observation variance (effective for gain computation)
        R = c_val * (vol_scalar ** 2)
        
        # Kalman gain (adjusted for tail heaviness)
        K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
        
        # Innovation
        ret_t = returns[t]
        r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
        innovation = r_val - mu_pred
        
        # Update state
        mu = float(mu_pred + K * innovation)
        P = float((1.0 - K) * P_pred)
        
        # Store filtered estimates
        mu_filtered[t] = mu
        P_filtered[t] = P
        
        # Accumulate log-likelihood: log p(r_t | past) under Student-t
        # Total forecast variance includes state uncertainty
        forecast_scale = np.sqrt(P_pred + R)
        if forecast_scale > 1e-12:
            ll_t = student_t_logpdf(r_val, nu_val, mu_pred, forecast_scale)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
    
    return mu_filtered, P_filtered, log_likelihood


def kalman_filter_drift_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t.

    Args:
        returns: Observed returns.
        vol: Observation noise std (EWMA volatility).
        q: Process noise variance.
        c: Observation variance scale.
        phi: Drift persistence (-1 < φ < 1.999). φ>0 trend, φ<0 mean-reversion.

    Returns:
        mu_filtered: Posterior mean of drift.
        P_filtered: Posterior variance of drift.
        log_likelihood: Total log-likelihood.
    """
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


def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
    """
    Compute PIT (Probability Integral Transform) and KS test statistic + p-value.
    
    Well-calibrated forecasts should have PIT ~ Uniform(0,1).
    
    Args:
        returns: Observed returns
        mu_filtered: Kalman filtered drift estimates
        vol: Volatility estimates
        P_filtered: Drift posterior variance
        c: Observation variance scale factor
    
    Returns:
        ks_statistic: KS test statistic
        ks_pvalue: KS test p-value
    """
    # Ensure all inputs are 1D arrays
    returns_flat = np.asarray(returns).flatten()
    mu_flat = np.asarray(mu_filtered).flatten()
    vol_flat = np.asarray(vol).flatten()
    P_flat = np.asarray(P_filtered).flatten()
    
    # Total forecast variance = scaled observation variance + parameter uncertainty
    forecast_std = np.sqrt(c * (vol_flat ** 2) + P_flat)
    
    # Standardize returns
    standardized = (returns_flat - mu_flat) / forecast_std
    
    # Compute PIT values
    pit_values = norm.cdf(standardized)
    
    # KS test against uniform distribution
    ks_result = kstest(pit_values, 'uniform')
    
    # Extract statistic and p-value
    return float(ks_result.statistic), float(ks_result.pvalue)


def compute_pit_ks_pvalue_student_t(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
    """
    Compute PIT (Probability Integral Transform) for Student-t forecasts.
    
    Well-calibrated Student-t forecasts should have PIT ~ Uniform(0,1).
    
    Args:
        returns: Observed returns
        mu_filtered: Kalman filtered drift estimates
        vol: Volatility estimates
        P_filtered: Drift posterior variance
        c: Observation variance scale factor
        nu: Student-t degrees of freedom
    
    Returns:
        ks_statistic: KS test statistic
        ks_pvalue: KS test p-value
    """
    # Ensure all inputs are 1D arrays
    returns_flat = np.asarray(returns).flatten()
    mu_flat = np.asarray(mu_filtered).flatten()
    vol_flat = np.asarray(vol).flatten()
    P_flat = np.asarray(P_filtered).flatten()
    
    # Total forecast scale = sqrt(scaled observation variance + parameter uncertainty)
    forecast_scale = np.sqrt(c * (vol_flat ** 2) + P_flat)
    
    # Standardize returns for Student-t
    standardized = (returns_flat - mu_flat) / forecast_scale
    
    # Compute PIT values using Student-t CDF
    pit_values = student_t.cdf(standardized, df=nu)
    
    # KS test against uniform distribution
    ks_result = kstest(pit_values, 'uniform')
    
    # Extract statistic and p-value
    return float(ks_result.statistic), float(ks_result.pvalue)


def optimize_q_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-1,  # EXPANDED: from 1e-2 to 1e-1 to allow higher process noise
    c_min: float = 0.3,   # EXPANDED: from 0.5 to 0.3 for more flexibility
    c_max: float = 3.0,   # EXPANDED: from 2.0 to 3.0 for more flexibility
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, Dict]:
    """
    Jointly optimize (q, c) via maximum likelihood with enhanced Bayesian regularization.
    
    Production-Grade Improvements:
    - Walk-forward cross-validation without look-ahead bias
    - Robust outlier handling in likelihood computation
    - Adaptive regularization based on market regime characteristics
    - Enhanced numerical stability with safe log computations
    - Asset-specific prior calibration based on return/volatility statistics
    
    Args:
        returns: Return series
        vol: Volatility series
        train_frac: Fraction of data for training (used for walk-forward validation)
        q_min, q_max: Bounds for process noise q
        c_min, c_max: Bounds for observation variance scale c
        prior_log_q_mean: Prior mean for log10(q) (default: -6)
        prior_lambda: Regularization strength (default: 1.0)
    
    Returns:
        q_optimal: Best-fit process noise
        c_optimal: Best-fit observation variance scale
        ll_optimal: Out-of-sample log-likelihood at optimum
        diagnostics: Dictionary with optimization diagnostics
    """
    n = len(returns)
    
    # Winsorize returns to handle extreme outliers (Level-7 robustness)
    # Clip at 99.5th percentile to prevent single extreme events from dominating
    ret_p005 = np.percentile(returns, 0.5)
    ret_p995 = np.percentile(returns, 99.5)
    returns_robust = np.clip(returns, ret_p005, ret_p995)
    
    # Compute data-driven statistics for smarter initialization
    ret_std = float(np.std(returns_robust))
    ret_mean = float(np.mean(returns_robust))
    vol_mean = float(np.mean(vol))
    vol_std = float(np.std(vol))
    
    if vol_mean > 0:
        vol_cv = vol_std / vol_mean  # Coefficient of variation
    else:
        vol_cv = 0.0
    
    # Compute return-to-volatility ratio (Sharpe-like metric for drift strength)
    # Higher ratio suggests stronger persistent drift → allow higher q
    if ret_std > 0:
        rv_ratio = abs(ret_mean) / ret_std
    else:
        rv_ratio = 0.0
    
    # Adaptive prior based on volatility regime and return characteristics
    # High vol_cv suggests more regime changes → higher q
    # High rv_ratio suggests persistent drift → higher q
    if vol_cv > 0.5 or rv_ratio > 0.15:
        # High volatility variability or strong drift: allow more drift evolution
        adaptive_prior_mean = prior_log_q_mean + 0.5
        adaptive_lambda = prior_lambda * 0.5  # Weaker regularization
    elif vol_cv < 0.2 and rv_ratio < 0.05:
        # Very stable regime with weak drift: enforce strong regularization
        adaptive_prior_mean = prior_log_q_mean - 0.3
        adaptive_lambda = prior_lambda * 1.5
    else:
        # Normal regime: default regularization
        adaptive_prior_mean = prior_log_q_mean
        adaptive_lambda = prior_lambda
    
    # Use expanding window walk-forward cross-validation (proper time-series CV)
    # Each fold trains on [0, train_end] and tests on [train_end+1, test_end]
    min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
    test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
    
    fold_splits = []
    train_end = min_train
    while train_end + test_window <= n:
        test_end = min(train_end + test_window, n)
        if test_end - train_end >= 20:  # Minimum test size
            fold_splits.append((0, train_end, train_end, test_end))
        train_end += test_window
    
    # If no folds created, use simple train/test split
    if not fold_splits:
        split_idx = int(n * train_frac)
        fold_splits = [(0, split_idx, split_idx, n)]
    
    def negative_penalized_ll_cv(params: np.ndarray) -> float:
        """Objective: negative penalized cross-validated log-likelihood with calibration term."""
        log_q, log_c = params
        q = 10 ** log_q
        c = 10 ** log_c
        
        # Validate parameters
        if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
            return 1e12
        
        # Compute average out-of-sample log-likelihood across folds
        total_ll_oos = 0.0
        total_obs = 0
        
        # Track standardized innovations for calibration check
        all_standardized = []
        
        for train_start, train_end, test_start, test_end in fold_splits:
            try:
                # Run Kalman filter ONLY on training data (no look-ahead)
                ret_train = returns_robust[train_start:train_end]
                vol_train = vol[train_start:train_end]
                
                if len(ret_train) < 3:
                    continue
                
                mu_filt_train, P_filt_train, _ = kalman_filter_drift(returns_robust[train_start:train_end], vol_train, q, c)
                
                # Get final state from training period
                mu_final = float(mu_filt_train[-1])
                P_final = float(P_filt_train[-1])
                
                # Compute one-step-ahead predictions on test set
                ll_fold = 0.0
                mu_pred = mu_final
                P_pred = P_final
                
                for t in range(test_start, test_end):
                    # One-step prediction
                    P_pred = P_pred + q
                    
                    # Extract scalar values safely
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
                    
                    # Safe log-likelihood computation with numerical stability
                    if forecast_var > 1e-12:
                        # Robust likelihood: downweight extreme innovations (Student-t-like behavior)
                        # Use Huber-like loss for extreme deviations
                        standardized_innov = innovation / np.sqrt(forecast_var)
                        standardized_innov_abs = abs(standardized_innov)
                        
                        # Store for calibration check (limit to reasonable range)
                        if len(all_standardized) < 1000:  # Limit memory
                            all_standardized.append(float(standardized_innov))
                        
                        if standardized_innov_abs > 5.0:  # Extreme outlier
                            # Use linear penalty beyond 5 sigma instead of quadratic
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 12.5 - 5.0 * (standardized_innov_abs - 5.0)
                        else:
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
                        
                        ll_fold += ll_contrib
                    
                    # Update state with observation (for next prediction)
                    K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                    mu_pred = mu_pred + K * innovation
                    P_pred = (1.0 - K) * P_pred
                
                total_ll_oos += ll_fold
                total_obs += (test_end - test_start)
                
            except Exception:
                # Skip problematic folds
                continue
        
        if total_obs == 0:
            return 1e12
        
        # Average likelihood per observation
        avg_ll_oos = total_ll_oos / max(total_obs, 1)
        
        # PRODUCTION FIX: Add explicit calibration penalty based on PIT distribution
        # Well-calibrated forecasts should have standardized innovations ~ N(0,1)
        # which means PIT ~ Uniform(0,1)
        calibration_penalty = 0.0
        if len(all_standardized) >= 30:  # Need sufficient samples
            try:
                # Compute PIT values from standardized innovations
                pit_values = norm.cdf(all_standardized)
                
                # KS test against uniform distribution
                ks_result = kstest(pit_values, 'uniform')
                ks_stat = float(ks_result.statistic)
                
                # STRENGTHENED: Much heavier penalty for poor calibration
                # Penalize deviations from uniform distribution aggressively
                # KS statistic ranges from 0 (perfect) to 1 (worst)
                # Target: KS stat < 0.05 for well-calibrated model
                if ks_stat > 0.05:
                    # Apply aggressive quadratic penalty
                    # Scale to make this comparable to likelihood (typically in range -1000 to 0)
                    calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)
                    
                    # Additional linear penalty for moderate miscalibration
                    if ks_stat > 0.10:
                        calibration_penalty -= 100.0 * (ks_stat - 0.10)
                    
                    # Severe penalty for extreme miscalibration
                    if ks_stat > 0.15:
                        calibration_penalty -= 200.0 * (ks_stat - 0.15)
            except Exception:
                pass  # Skip calibration penalty if computation fails
        
        # Add adaptive Bayesian prior regularization on q
        # Scale priors by 1/n to make them weak relative to data (standard Bayesian practice)
        prior_scale = 1.0 / max(total_obs, 100)  # Normalize by number of observations
        log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
        
        # Add weak regularization on c to prefer values near 0.9-1.0
        # EWMA typically underestimates volatility by ~5-15%, so c should be around 0.85-1.0
        c_target = 0.9
        log_c_target = np.log10(c_target)
        log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2  # Weak prior on c
        
        # Total penalized likelihood with calibration term
        penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + calibration_penalty
        
        if not np.isfinite(penalized_ll):
            return 1e12
        
        return -penalized_ll  # Minimize negative = maximize
    
    # Enhanced grid search in log-space
    log_q_min = np.log10(q_min)
    log_q_max = np.log10(q_max)
    log_c_min = np.log10(c_min)
    log_c_max = np.log10(c_max)
    
    # Finer 2D grid search (15×12 = 180 evaluations) with smarter spacing
    # More granularity near expected optimal regions
    log_q_grid = np.concatenate([
        np.linspace(log_q_min, adaptive_prior_mean - 1.0, 5),
        np.linspace(adaptive_prior_mean - 1.0, adaptive_prior_mean + 1.0, 7),
        np.linspace(adaptive_prior_mean + 1.0, log_q_max, 3)
    ])
    
    # c grid: focus around 0.8-1.0 (typical EWMA bias correction range)
    log_c_grid = np.concatenate([
        np.linspace(log_c_min, np.log10(0.7), 3),
        np.linspace(np.log10(0.7), np.log10(1.0), 7),
        np.linspace(np.log10(1.0), log_c_max, 2)
    ])
    
    best_neg_ll = float('inf')
    best_log_q_grid = adaptive_prior_mean  # Initialize at adaptive prior
    best_log_c_grid = np.log10(0.9)  # Initialize at c=0.9 (typical EWMA correction)
    
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
    
    # Store grid search result for diagnostics
    grid_best_q = 10 ** best_log_q_grid
    grid_best_c = 10 ** best_log_c_grid
    
    # Fine optimization via bounded minimize with multiple starts
    bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max)]
    
    # Try multiple initializations for robustness
    best_result = None
    best_fun = float('inf')
    
    # ENHANCED: More diverse starting points to explore parameter space
    start_points = [
        np.array([best_log_q_grid, best_log_c_grid]),  # Grid best
        np.array([adaptive_prior_mean, np.log10(0.9)]),  # Adaptive prior with c=0.9
        np.array([adaptive_prior_mean, np.log10(0.7)]),  # Adaptive prior with c=0.7
        np.array([adaptive_prior_mean, np.log10(1.2)]),  # Adaptive prior with c=1.2
        np.array([best_log_q_grid - 0.5, best_log_c_grid]),  # Lower q neighbor
        np.array([best_log_q_grid + 0.5, best_log_c_grid]),  # Higher q neighbor
        np.array([best_log_q_grid, best_log_c_grid - 0.2]),  # Lower c neighbor
        np.array([best_log_q_grid, best_log_c_grid + 0.2]),  # Higher c neighbor
        np.array([-7.0, 0.0]),  # Low q, mid c
        np.array([-5.0, 0.0]),  # High q, mid c
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
        # Fallback to grid search
        q_optimal = grid_best_q
        c_optimal = grid_best_c
        ll_optimal = -best_neg_ll
    
    # Comprehensive diagnostics for production monitoring
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


def optimize_q_c_nu_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-1,
    c_min: float = 0.3,
    c_max: float = 3.0,
    nu_min: float = 2.1,
    nu_max: float = 30.0,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, float, Dict]:
    """
    Jointly optimize (q, c, ν) for Student-t observation noise via maximum likelihood.
    
    Extends optimize_q_mle to include tail heaviness parameter ν.
    Uses walk-forward CV and robust likelihood computation.
    
    Args:
        returns: Return series
        vol: Volatility series
        train_frac: Fraction of data for training
        q_min, q_max: Bounds for process noise q
        c_min, c_max: Bounds for observation variance scale c
        nu_min, nu_max: Bounds for Student-t degrees of freedom
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
    
    Returns:
        q_optimal: Best-fit process noise
        c_optimal: Best-fit observation variance scale
        nu_optimal: Best-fit degrees of freedom
        ll_optimal: Out-of-sample log-likelihood at optimum
        diagnostics: Dictionary with optimization diagnostics
    """
    n = len(returns)
    
    # Winsorize returns for robustness
    ret_p005 = np.percentile(returns, 0.5)
    ret_p995 = np.percentile(returns, 99.5)
    returns_robust = np.clip(returns, ret_p005, ret_p995)
    
    # Compute statistics for adaptive prior
    ret_std = float(np.std(returns_robust))
    ret_mean = float(np.mean(returns_robust))
    vol_mean = float(np.mean(vol))
    vol_std = float(np.std(vol))
    
    if vol_mean > 0:
        vol_cv = vol_std / vol_mean  # Coefficient of variation
    else:
        vol_cv = 0.0
    if ret_std > 0:
        rv_ratio = abs(ret_mean) / ret_std
    else:
        rv_ratio = 0.0
    
    # Adaptive prior
    if vol_cv > 0.5 or rv_ratio > 0.15:
        adaptive_prior_mean = prior_log_q_mean + 0.5
        adaptive_lambda = prior_lambda * 0.5
    elif vol_cv < 0.2 and rv_ratio < 0.05:
        adaptive_prior_mean = prior_log_q_mean - 0.3
        adaptive_lambda = prior_lambda * 1.5
    else:
        adaptive_prior_mean = prior_log_q_mean
        adaptive_lambda = prior_lambda
    
    # Use expanding window walk-forward cross-validation (proper time-series CV)
    # Each fold trains on [0, train_end] and tests on [train_end+1, test_end]
    min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
    test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
    
    fold_splits = []
    train_end = min_train
    while train_end + test_window <= n:
        test_end = min(train_end + test_window, n)
        if test_end - train_end >= 20:  # Minimum test size
            fold_splits.append((0, train_end, train_end, test_end))
        train_end += test_window
    
    if not fold_splits:
        split_idx = int(n * train_frac)
        fold_splits = [(0, split_idx, split_idx, n)]
    
    def negative_penalized_ll_cv_student_t(params: np.ndarray) -> float:
        """Objective: negative penalized CV log-likelihood for Student-t."""
        log_q, log_c, log_nu = params
        q = 10 ** log_q
        c = 10 ** log_c
        nu = 10 ** log_nu
        
        # Validate parameters
        if q <= 0 or c <= 0 or nu < nu_min or nu > nu_max:
            return 1e12
        if not (np.isfinite(q) and np.isfinite(c) and np.isfinite(nu)):
            return 1e12
        
        # Compute CV log-likelihood
        total_ll_oos = 0.0
        total_obs = 0
        
        for train_start, train_end, test_start, test_end in fold_splits:
            try:
                ret_train = returns_robust[train_start:train_end]
                vol_train = vol[train_start:train_end]
                
                if len(ret_train) < 3:
                    continue
                
                # Train Student-t filter
                mu_filt_train, P_filt_train, _ = kalman_filter_drift_student_t(returns_robust[train_start:train_end], vol_train, q, c, nu)
                
                mu_final = float(mu_filt_train[-1])
                P_final = float(P_filt_train[-1])
                
                # One-step-ahead predictions on test set
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
                    
                    # Student-t forecast
                    forecast_scale = np.sqrt(P_pred + c * (vol_t ** 2))
                    
                    if forecast_scale > 1e-12:
                        ll_contrib = student_t_logpdf(ret_t, nu, mu_pred, forecast_scale)
                        if np.isfinite(ll_contrib):
                            ll_fold += ll_contrib
                    
                    # Update state for next prediction
                    R = c * (vol_t ** 2)
                    nu_adjust = min(nu / (nu + 3), 1.0)
                    if (P_pred + R) > 1e-12:
                        K = nu_adjust * P_pred / (P_pred + R)
                    else:
                        K = 0.0
                    innovation = ret_t - mu_pred
                    mu_pred = mu_pred + K * innovation
                    P_pred = (1.0 - K) * P_pred
                
                total_ll_oos += ll_fold
                total_obs += (test_end - test_start)
                
            except Exception:
                continue
        
        if total_obs == 0:
            return 1e12
        
        avg_ll_oos = total_ll_oos / max(total_obs, 1)
        
        # Priors
        prior_scale = 1.0 / max(total_obs, 100)
        log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
        log_prior_c = -0.1 * prior_scale * (log_c - np.log10(0.9)) ** 2
        
        # Weak prior on nu: prefer moderate values (nu ~ 6)
        log_prior_nu = -0.05 * prior_scale * (log_nu - np.log10(6.0)) ** 2
        
        penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + log_prior_nu
        
        if not np.isfinite(penalized_ll):
            return 1e12
        
        return -penalized_ll
    
    # Grid search in 3D log-space (coarser than 2D)
    log_q_min = np.log10(q_min)
    log_q_max = np.log10(q_max)
    log_c_min = np.log10(c_min)
    log_c_max = np.log10(c_max)
    log_nu_min = np.log10(nu_min)
    log_nu_max = np.log10(nu_max)
    
    # Coarse grid: 5×4×4 = 80 evaluations
    log_q_grid = np.linspace(log_q_min, log_q_max, 5)
    log_c_grid = np.linspace(log_c_min, log_c_max, 4)
    log_nu_grid = np.linspace(log_nu_min, log_nu_max, 4)
    
    best_neg_ll = float('inf')
    best_log_q_grid = adaptive_prior_mean
    best_log_c_grid = np.log10(0.9)
    best_log_nu_grid = np.log10(6.0)
    
    for lq in log_q_grid:
        for lc in log_c_grid:
            for lnu in log_nu_grid:
                try:
                    neg_ll = negative_penalized_ll_cv_student_t(np.array([lq, lc, lnu]))
                    if neg_ll < best_neg_ll:
                        best_neg_ll = neg_ll
                        best_log_q_grid = lq
                        best_log_c_grid = lc
                        best_log_nu_grid = lnu
                except Exception:
                    continue
    
    grid_best_q = 10 ** best_log_q_grid
    grid_best_c = 10 ** best_log_c_grid
    grid_best_nu = 10 ** best_log_nu_grid
    
    # Fine optimization
    bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (log_nu_min, log_nu_max)]
    
    best_result = None
    best_fun = float('inf')
    
    start_points = [
        np.array([best_log_q_grid, best_log_c_grid, best_log_nu_grid]),
        np.array([adaptive_prior_mean, np.log10(0.9), np.log10(6.0)]),
        np.array([adaptive_prior_mean, np.log10(0.85), np.log10(4.0)]),
        np.array([adaptive_prior_mean, np.log10(1.2), np.log10(8.0)]),
        np.array([best_log_q_grid - 0.5, best_log_c_grid, best_log_nu_grid]),
        np.array([best_log_q_grid + 0.5, best_log_c_grid, best_log_nu_grid]),
    ]
    
    for x0 in start_points:
        try:
            result = minimize(
                negative_penalized_ll_cv_student_t,
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
        log_q_opt, log_c_opt, log_nu_opt = best_result.x
        q_optimal = 10 ** log_q_opt
        c_optimal = 10 ** log_c_opt
        nu_optimal = 10 ** log_nu_opt
        ll_optimal = -best_result.fun
    else:
        q_optimal = grid_best_q
        c_optimal = grid_best_c
        nu_optimal = grid_best_nu
        ll_optimal = -best_neg_ll
    
    # Clip nu to bounds
    nu_optimal = float(np.clip(nu_optimal, nu_min, nu_max))
    
    diagnostics = {
        'grid_best_q': float(grid_best_q),
        'grid_best_c': float(grid_best_c),
        'grid_best_nu': float(grid_best_nu),
        'refined_best_q': float(q_optimal),
        'refined_best_c': float(c_optimal),
        'refined_best_nu': float(nu_optimal),
        'prior_applied': adaptive_lambda > 0,
        'prior_log_q_mean': float(adaptive_prior_mean),
        'prior_lambda': float(adaptive_lambda),
        'vol_cv': float(vol_cv),
        'rv_ratio': float(rv_ratio),
        'n_folds': int(len(fold_splits)),
        'optimization_successful': best_result is not None and (best_result.success if best_result else False)
    }
    
    return q_optimal, c_optimal, nu_optimal, ll_optimal, diagnostics


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
    """Jointly optimize (q, c, φ) via maximum likelihood for φ-Kalman filter."""
    n = len(returns)
    
    # Winsorize returns for robustness
    ret_p005 = np.percentile(returns, 0.5)
    ret_p995 = np.percentile(returns, 99.5)
    returns_robust = np.clip(returns, ret_p005, ret_p995)
    
    # Compute statistics for adaptive prior
    ret_std = float(np.std(returns_robust))
    ret_mean = float(np.mean(returns_robust))
    vol_mean = float(np.mean(vol))
    vol_std = float(np.std(vol))
    
    if vol_mean > 0:
        vol_cv = vol_std / vol_mean  # Coefficient of variation
    else:
        vol_cv = 0.0
    if ret_std > 0:
        rv_ratio = abs(ret_mean) / ret_std
    else:
        rv_ratio = 0.0

    # Adaptive prior
    if vol_cv > 0.5 or rv_ratio > 0.15:
        adaptive_prior_mean = prior_log_q_mean + 0.5
        adaptive_lambda = prior_lambda * 0.5
    elif vol_cv < 0.2 and rv_ratio < 0.05:
        adaptive_prior_mean = prior_log_q_mean - 0.3
        adaptive_lambda = prior_lambda * 1.5
    else:
        adaptive_prior_mean = prior_log_q_mean
        adaptive_lambda = prior_lambda
    
    # Use expanding window walk-forward cross-validation (proper time-series CV)
    # Each fold trains on [0, train_end] and tests on [train_end+1, test_end]
    min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
    test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
    
    fold_splits = []
    train_end = min_train
    while train_end + test_window <= n:
        test_end = min(train_end + test_window, n)
        if test_end - train_end >= 20:  # Minimum test size
            fold_splits.append((0, train_end, train_end, test_end))
        train_end += test_window
    
    if not fold_splits:
        split_idx = int(n * train_frac)
        fold_splits = [(0, split_idx, split_idx, n)]
    
    def negative_penalized_ll_cv_phi(params: np.ndarray) -> float:
        # params = [log_q, log_c, phi]
        log_q, log_c, phi = params
        q = 10 ** log_q
        c = 10 ** log_c
        phi = float(np.clip(phi, phi_min, phi_max))
        
        # Validate parameters
        if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
            return 1e12
        
        # Compute average out-of-sample log-likelihood across folds
        total_ll_oos = 0.0
        total_obs = 0
        
        # Track standardized innovations for calibration check
        all_standardized = []
        
        for train_start, train_end, test_start, test_end in fold_splits:
            try:
                # Run Kalman filter ONLY on training data (no look-ahead)
                ret_train = returns_robust[train_start:train_end]
                vol_train = vol[train_start:train_end]
                
                if len(ret_train) < 3:
                    continue
                
                mu_filt_train, P_filt_train, _ = kalman_filter_drift_phi(returns_robust[train_start:train_end], vol_train, q, c, phi)
                
                # Get final state from training period
                mu_final = float(mu_filt_train[-1])
                P_final = float(P_filt_train[-1])
                
                # Compute one-step-ahead predictions on test set
                ll_fold = 0.0
                mu_pred = mu_final
                P_pred = P_final
                
                for t in range(test_start, test_end):
                    # One-step prediction
                    P_pred = P_pred + q
                    
                    # Extract scalar values safely
                    ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                    vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())
                    
                    R = c * (vol_t ** 2)
                    innovation = ret_t - mu_pred
                    forecast_var = P_pred + R
                    
                    # Safe log-likelihood computation with numerical stability
                    if forecast_var > 1e-12:
                        # Robust likelihood: downweight extreme innovations (Student-t-like behavior)
                        # Use Huber-like loss for extreme deviations
                        standardized_innov = innovation / np.sqrt(forecast_var)
                        standardized_innov_abs = abs(standardized_innov)
                        
                        # Store for calibration check (limit to reasonable range)
                        if len(all_standardized) < 1000:  # Limit memory
                            all_standardized.append(float(standardized_innov))
                        
                        if standardized_innov_abs > 5.0:  # Extreme outlier
                            # Use linear penalty beyond 5 sigma instead of quadratic
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 12.5 - 5.0 * (standardized_innov_abs - 5.0)
                        else:
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
                        
                        ll_fold += ll_contrib
                    
                    # Update state with observation (for next prediction)
                    K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                    mu_pred = mu_pred + K * innovation
                    P_pred = (1.0 - K) * P_pred
                
                total_ll_oos += ll_fold
                total_obs += (test_end - test_start)
                
            except Exception:
                # Skip problematic folds
                continue
        
        if total_obs == 0:
            return 1e12
        
        # Average likelihood per observation
        avg_ll_oos = total_ll_oos / max(total_obs, 1)
        
        # PRODUCTION FIX: Add explicit calibration penalty based on PIT distribution
        # Well-calibrated forecasts should have standardized innovations ~ N(0,1)
        # which means PIT ~ Uniform(0,1)
        calibration_penalty = 0.0
        if len(all_standardized) >= 30:  # Need sufficient samples
            try:
                # Compute PIT values from standardized innovations
                pit_values = norm.cdf(all_standardized)
                
                # KS test against uniform distribution
                ks_result = kstest(pit_values, 'uniform')
                ks_stat = float(ks_result.statistic)
                
                # STRENGTHENED: Much heavier penalty for poor calibration
                # Penalize deviations from uniform distribution aggressively
                # KS statistic ranges from 0 (perfect) to 1 (worst)
                # Target: KS stat < 0.05 for well-calibrated model
                if ks_stat > 0.05:
                    # Apply aggressive quadratic penalty
                    # Scale to make this comparable to likelihood (typically in range -1000 to 0)
                    calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)
                    
                    # Additional linear penalty for moderate miscalibration
                    if ks_stat > 0.10:
                        calibration_penalty -= 100.0 * (ks_stat - 0.10)
                    
                    # Severe penalty for extreme miscalibration
                    if ks_stat > 0.15:
                        calibration_penalty -= 200.0 * (ks_stat - 0.15)
            except Exception:
                pass  # Skip calibration penalty if computation fails
        
        # Add adaptive Bayesian prior regularization on q
        # Scale priors by 1/n to make them weak relative to data (standard Bayesian practice)
        prior_scale = 1.0 / max(total_obs, 100)  # Normalize by number of observations
        log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
        
        # Add weak regularization on c to prefer values near 0.9-1.0
        # EWMA typically underestimates volatility by ~5-15%, so c should be around 0.85-1.0
        c_target = 0.9
        log_c_target = np.log10(c_target)
        log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2  # Weak prior on c
        
        # Total penalized likelihood with calibration term
        penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + calibration_penalty
        
        return -penalized_ll if np.isfinite(penalized_ll) else 1e12

    # Grid search (phi in linear space)
    log_q_min = np.log10(q_min)
    log_q_max = np.log10(q_max)
    log_c_min = np.log10(c_min)
    log_c_max = np.log10(c_max)
    phi_grid = np.linspace(phi_min, phi_max, 5)
    log_q_grid = np.linspace(log_q_min, log_q_max, 5)
    log_c_grid = np.linspace(log_c_min, log_c_max, 4)

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

    # Fine optimization
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
            # Fallback: direct download
            df = _download_prices(asset, start_date, end_date)
            if df.empty:
                raise RuntimeError(f"No data for {asset}")
            px = df['Close']
            title = asset
        
        # Allow very small histories; tune will still run cross-validation with short splits
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
        print(f"  🔧 Fitting φ-Kalman model...")
        q_phi, c_phi, phi_opt, ll_phi_cv, opt_diag_phi = optimize_q_c_phi_mle(
            returns_arr, vol_arr,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        mu_phi, P_phi, ll_phi_full = kalman_filter_drift_phi(returns_arr, vol_arr, q_phi, c_phi, phi_opt)
        ks_phi, pit_p_phi = compute_pit_ks_pvalue(returns_arr, mu_phi, vol_arr, P_phi, c_phi)
        aic_phi = compute_aic(ll_phi_full, n_params=3)
        bic_phi = compute_bic(ll_phi_full, n_params=3, n_obs=n_obs)
        print(f"     φ-Kalman: q={q_phi:.2e}, c={c_phi:.3f}, φ={phi_opt:+.3f}, LL={ll_phi_full:.1f}, BIC={bic_phi:.1f}, PIT p={pit_p_phi:.4f}")
        
        # =================================================================
        # STEP 2: Fit Student-t Model (q, c, ν)
        # =================================================================
        print(f"  🔧 Fitting Student-t model...")
        try:
            q_student, c_student, nu_student, ll_student_cv, opt_diag_student = optimize_q_c_nu_mle(
                returns_arr, vol_arr,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            # Run full Student-t Kalman filter
            mu_student, P_student, ll_student_full = kalman_filter_drift_student_t(
                returns_arr, vol_arr, q_student, c_student, nu_student
            )
            
            # Compute Student-t PIT calibration
            ks_student, pit_p_student = compute_pit_ks_pvalue_student_t(
                returns_arr, mu_student, vol_arr, P_student, c_student, nu_student
            )
            
            # Student-t has 3 parameters: q, c, ν
            aic_student = compute_aic(ll_student_full, n_params=3)
            bic_student = compute_bic(ll_student_full, n_params=3, n_obs=n_obs)
            
            print(f"     Student-t: q={q_student:.2e}, c={c_student:.3f}, ν={nu_student:.1f}, LL={ll_student_full:.1f}, BIC={bic_student:.1f}, PIT p={pit_p_student:.4f}")
            
            student_t_fit_success = True
            
        except Exception as e:
            print(f"  ⚠️  Student-t optimization failed: {e}")
            student_t_fit_success = False
            q_student = None
            c_student = None
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
            candidate_models.append(("student_t", bic_student, aic_student, ll_student_full, mu_student, P_student, ks_student, pit_p_student, q_student, c_student, nu_student, opt_diag_student))

        candidate_models = [m for m in candidate_models if np.isfinite(m[1])]
        best_entry = min(candidate_models, key=lambda x: x[1])
        noise_model, bic_final, aic_final, ll_full, mu_filtered, P_filtered, ks_statistic, ks_pvalue, q_optimal, c_optimal, extra_param, opt_diagnostics = best_entry

        nu_optimal = None
        phi_selected = None
        if noise_model == "student_t":
            nu_optimal = extra_param
        elif noise_model == "phi_gaussian":
            phi_selected = extra_param

        print(f"  ✓ Selected {noise_model} (BIC={bic_final:.1f})")
        if noise_model == "student_t":
            print(f"    (ΔBIC vs Gaussian = {bic_gauss - bic_student:+.1f}, ΔBIC vs φ-Kalman = {bic_phi - bic_student:+.1f})")
        elif noise_model == "phi_gaussian":
            print(f"    (ΔBIC vs Gaussian = {bic_gauss - bic_phi:+.1f})")
        else:
            print(f"    (ΔBIC vs φ-Kalman = {bic_phi - bic_gauss:+.1f})")
        
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
            print(f"     Kalman-Student-t: LL={ll_student_full:.1f}, AIC={aic_student:.1f}, BIC={bic_student:.1f}, ν={nu_student:.1f}")
        
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
            model_comparison['kalman_student_t'] = {'ll': ll_student_full, 'aic': aic_student, 'bic': bic_student, 'n_params': 3}
        
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
            'noise_model': noise_model,  # "gaussian" or "student_t"

            # Parameters
            'q': float(q_optimal),
            'c': float(c_optimal),
            'nu': float(nu_optimal) if nu_optimal is not None else None,
            'phi': float(phi_selected) if phi_selected is not None else None,

            # Likelihood and model comparison
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

        # Add Student-t specific diagnostics if applicable
        if noise_model == "student_t":
            result['grid_best_nu'] = opt_diagnostics.get('grid_best_nu')
            result['refined_best_nu'] = opt_diagnostics.get('refined_best_nu')
        if noise_model == "phi_gaussian":
            result['refined_best_phi'] = float(phi_selected)

        # Add Gaussian comparison if Student-t was selected
        if noise_model == "student_t" and student_t_fit_success:
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


def main():
    parser = argparse.ArgumentParser(
        description="Estimate optimal Kalman drift parameters with Student-t noise support",
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
    
    args = parser.parse_args()
    
    # Enable debug mode
    if args.debug:
        os.environ['DEBUG'] = '1'
    
    print("=" * 80)
    print("Kalman Drift MLE Tuning Pipeline - Student-t Upgrade")
    print("=" * 80)
    print(f"Prior: log10(q) ~ N({args.prior_mean:.1f}, λ={args.prior_lambda:.1f})")
    print("Model selection: Gaussian vs Student-t via BIC")
    print("Enhancements: Fat-tail support, Multi-fold CV, Adaptive regularization")
    
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

    # Process each asset (parallel by default)
    new_estimates = 0
    reused_cached = 0
    failed = 0
    calibration_warnings = 0
    student_t_count = 0
    gaussian_count = 0

    assets_to_process: List[str] = []
    failure_reasons: Dict[str, str] = {}

    for i, asset in enumerate(assets, 1):
        print(f"\n[{i}/{len(assets)}] {asset}")

        # Check cache
        if not args.force and asset in cache:
            cached_q = cache[asset].get('q', float('nan'))
            cached_c = cache[asset].get('c', 1.0)
            cached_model = cache[asset].get('noise_model', 'gaussian')
            if cached_model == 'student_t':
                cached_nu = cache[asset].get('nu', float('nan'))
                print(f"  ✓ Using cached estimate ({cached_model}: q={cached_q:.2e}, c={cached_c:.3f}, ν={cached_nu:.1f})")
            else:
                print(f"  ✓ Using cached estimate ({cached_model}: q={cached_q:.2e}, c={cached_c:.3f})")
            reused_cached += 1
            continue

        assets_to_process.append(asset)

    if assets_to_process:
        worker_count = min(max(1, os.cpu_count() or 1), len(assets_to_process))
        print(f"\nRunning {len(assets_to_process)} assets with {worker_count} workers...")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    tune_asset_q,
                    asset,
                    args.start,
                    args.end,
                    args.prior_mean,
                    args.prior_lambda
                ): asset
                for asset in assets_to_process
            }
            for idx, future in enumerate(as_completed(future_map), 1):
                asset = future_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  ❌ {asset}: Failed - {e}")
                    failure_reasons[asset] = str(e)
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    failed += 1
                    continue

                if result:
                    cache[asset] = result
                    new_estimates += 1

                    if result.get('noise_model') == 'student_t':
                        student_t_count += 1
                    else:
                        gaussian_count += 1

                    if result.get('calibration_warning'):
                        calibration_warnings += 1
                else:
                    failed += 1
                    failure_reasons[asset] = "tune_asset_q returned None"
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
    
    if cache:
        print(f"\nBest-fit parameters (sorted by model type, then q) — ALL ASSETS:")
        print(f"{'Asset':<20} {'Model':<10} {'log10(q)':<10} {'c':<8} {'ν':<6} {'ΔLL_0':<8} {'ΔLL_c':<8} {'ΔLL_e':<8} {'BestModel':<12} {'BIC':<10} {'PIT p':<10}")
        print("-" * 145)
        
        # Sort by model type, then q
        sorted_assets = sorted(cache.items(), key=lambda x: (
            x[1].get('noise_model', 'gaussian'),
            -x[1].get('q', 0)  # Descending q
        ))
        
        for asset, data in sorted_assets:
            q_val = data.get('q', float('nan'))
            c_val = data.get('c', 1.0)
            nu_val = data.get('nu')
            delta_ll_zero = data.get('delta_ll_vs_zero', float('nan'))
            delta_ll_const = data.get('delta_ll_vs_const', float('nan'))
            delta_ll_ewma = data.get('delta_ll_vs_ewma', float('nan'))
            bic_val = data.get('bic', float('nan'))
            pit_p = data.get('pit_ks_pvalue', float('nan'))
            model = data.get('noise_model', 'gaussian')
            best_model = data.get('best_model_by_bic', 'kalman_drift')
            
            log10_q = np.log10(q_val) if q_val > 0 else float('nan')
            
            nu_str = f"{nu_val:.1f}" if nu_val is not None else "-"
            model_abbr = "Student-t" if model == "student_t" else "Gaussian"
            
            # Shorten best model name for display
            best_model_abbr = {
                'zero_drift': 'Zero',
                'constant_drift': 'Const',
                'ewma_drift': 'EWMA',
                'kalman_drift': 'Kalman',
                'phi_kalman_drift': 'PhiKal'
            }.get(best_model, best_model[:8])

            warn_marker = " ⚠️" if data.get('calibration_warning') else ""
            
            print(f"{asset:<20} {model_abbr:<10} {log10_q:>8.2f}   {c_val:>6.3f}  {nu_str:<6} {delta_ll_zero:>6.1f}  {delta_ll_const:>6.1f}  {delta_ll_ewma:>6.1f}  {best_model_abbr:<12} {bic_val:>9.1f}  {pit_p:.4f}{warn_marker}")
        
        # Add legend
        print("\nColumn Legend:")
        print("  ΔLL_0: ΔLL vs zero-drift baseline")
        print("  ΔLL_c: ΔLL vs constant-drift baseline")
        print("  ΔLL_e: ΔLL vs EWMA-drift baseline")
        print("  BestModel: Best model by BIC (Zero/Const/EWMA/Kalman)")
        
        print(f"\nCache file:")
        print(f"  JSON: {args.cache_json}")
    
    if failure_reasons:
        print("\nFailed tickers and reasons:")
        for a, msg in failure_reasons.items():
            print(f"  {a}: {msg}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
