#!/usr/bin/env python3
"""
===============================================================================
OPTIONS VOLATILITY TUNING WITH BAYESIAN MODEL AVERAGING
===============================================================================

Extends the equity BMA framework to options volatility surfaces.

Architecture: Hierarchical Belief Propagation with Confidence-Bounded Envelopes

Implements the governing equation:
    p(σ_{t+H} | r, S) = Σ_m p(σ_{t+H} | r, m, θ_{r,m}) · p(m | r, S)

Where:
- r = volatility regime (derived from equity regime)
- S = equity signal (BUY/SELL conviction acts as prior)
- m ∈ {constant_vol, mean_reverting, regime_vol, regime_skew_vol, variance_swap}
- θ_{r,m} = regime-conditional volatility parameters

The equity BMA posterior provides an informative prior:
- Strong BUY → prior favors call-favorable volatility scenarios
- Strong SELL → prior favors put-favorable volatility scenarios

CORE PRINCIPLE: "Volatility beliefs inherit from, but do not override, 
directional beliefs. Options signals require BOTH dimensions to align."

-------------------------------------------------------------------------------
MODEL COMPETITION FRAMEWORK (Mirrors tune.py)
-------------------------------------------------------------------------------

Volatility models compete via combined BIC + Hyvärinen scoring:
    w_combined(m) = w_bic(m)^α * w_hyvarinen(m)^(1-α)

Models:
1. constant_vol:        σ (constant implied volatility)
2. mean_reverting_vol:  σ̄, κ, η (Ornstein-Uhlenbeck vol dynamics)
3. regime_vol:          σ_r per regime (regime-conditional constant)
4. regime_skew_vol:     σ_r, ξ_r per regime (vol + skew)
5. variance_swap:       Model-free variance swap fair value

TEMPORAL SMOOTHING:
    w_smooth(m|r) = (prev_p(m|r))^α * w_raw(m|r)

HIERARCHICAL PRIOR FROM EQUITY:
    Equity conviction → Volatility prior mean adjustment
    Equity regime → Volatility regime assignment guidance

-------------------------------------------------------------------------------
GOVERNANCE PRINCIPLES
-------------------------------------------------------------------------------

1. Uncertainty is mandatory — every estimate carries confidence bounds
2. Liquidity is prerequisite — untradeable options never enter tuning
3. Expiry demands respect — near-expiry data receives reduced weight
4. Provenance is preserved — every result traceable to inputs
5. Isolation is enforced — option tuning failures cannot affect equity

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache

import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.special import gammaln


# =============================================================================
# IMPORT OPTIONS MODEL REGISTRY AND MODELS
# =============================================================================
# These models implement momentum-coupled volatility estimation matching
# the architecture of equity tune.py model competition.
# =============================================================================
import sys
import os as _os

# Ensure src is in path for imports
_src_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from models.options import (
        # Registry
        OPTION_MODEL_REGISTRY,
        OptionModelFamily,
        get_option_model_spec,
        get_all_option_model_names,
        get_option_models_for_tuning,
        make_option_momentum_gaussian_name,
        make_option_momentum_phi_gaussian_name,
        make_option_momentum_student_t_name,
        OPTION_STUDENT_T_NU_GRID,
        # Momentum Bridge
        MomentumBridge,
        MomentumDistributionType,
        MomentumParameters,
        extract_momentum_parameters,
        compute_sabr_priors_from_momentum,
        compute_ensemble_weights_from_momentum,
        # Models
        OptionMomentumGaussianModel,
        OptionMomentumPhiGaussianModel,
        OptionMomentumPhiStudentTModel,
    )
    OPTION_MODEL_REGISTRY_AVAILABLE = True
except ImportError as e:
    OPTION_MODEL_REGISTRY_AVAILABLE = False
    _import_error = str(e)
    warnings.warn(f"Options model registry not available: {e}")
    OPTION_STUDENT_T_NU_GRID = [4, 8, 20]
    _import_error = str(e)
else:
    _import_error = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# Volatility model registry (legacy - kept for backward compatibility)
VOL_MODEL_CLASSES = [
    "constant_vol",
    "mean_reverting_vol",
    "regime_vol",
    "regime_skew_vol",
    "variance_swap_anchor",
]

# NEW: Momentum-coupled volatility models (preferred)
MOMENTUM_VOL_MODEL_CLASSES = [
    "option_momentum_gaussian",
    "option_momentum_phi_gaussian",
] + [f"option_momentum_phi_student_t_nu_{nu}" for nu in OPTION_STUDENT_T_NU_GRID]

# Enable momentum models by default
USE_MOMENTUM_MODELS = True

# Prior configuration (matching equity tune.py structure)
DEFAULT_VOL_PRIOR_MEAN = 0.25  # 25% annualized vol prior
DEFAULT_VOL_PRIOR_LAMBDA = 1.0  # Regularization strength
DEFAULT_KAPPA_PRIOR = 0.5  # Mean reversion speed prior
DEFAULT_SKEW_PRIOR = 0.0  # Neutral skew prior

# Regime-conditional settings
MIN_VOL_SAMPLES = 15  # Minimum samples for regime-specific fitting
DEFAULT_TEMPORAL_ALPHA = 0.7  # Temporal smoothing for posteriors

# Model selection (matching equity scoring)
DEFAULT_MODEL_SELECTION_METHOD = 'combined'
DEFAULT_BIC_WEIGHT = 0.5
DEFAULT_ENTROPY_LAMBDA = 0.1

# Confidence bounds configuration
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_CONFIDENCE_WIDTH = 0.01  # Minimum 1% confidence interval
MAX_CONFIDENCE_WIDTH = 0.50  # Maximum 50% confidence interval

# Cache directory
OPTION_TUNE_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "option_tune"
)

# Expiry stratification thresholds (days)
EXPIRY_NEAR = 7      # Near-expiry: 0-7 days
EXPIRY_SHORT = 21    # Short-term: 8-21 days
EXPIRY_MEDIUM = 45   # Medium-term: 22-45 days
# Anything beyond 45 days is long-term


# =============================================================================
# VOLATILITY REGIME LABELS (derived from equity regimes)
# =============================================================================

VOL_REGIME_LABELS = {
    0: "Low_Vol",       # Low IV, stable surface
    1: "Normal_Vol",    # Typical IV levels
    2: "Elevated_Vol",  # Above-average IV
    3: "High_Vol",      # Significantly elevated IV
    4: "Extreme_Vol",   # Crisis-level IV
}

# Equity regime to volatility regime mapping
EQUITY_TO_VOL_REGIME = {
    0: 2,  # LOW_VOL_TREND → Elevated_Vol (calm spot but elevated options)
    1: 3,  # HIGH_VOL_TREND → High_Vol
    2: 0,  # LOW_VOL_RANGE → Low_Vol
    3: 2,  # HIGH_VOL_RANGE → Elevated_Vol
    4: 4,  # CRISIS_JUMP → Extreme_Vol
}


def assign_vol_regime_labels(
    iv_series: np.ndarray,
    realized_vol: np.ndarray,
    equity_regime: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Assign volatility regime labels based on IV and realized vol relationship.
    
    Uses equity regime as informative prior when available.
    
    Args:
        iv_series: Implied volatility time series
        realized_vol: Realized volatility time series  
        equity_regime: Optional equity regime labels (0-4)
        
    Returns:
        Array of regime labels (0-4)
    """
    n = len(iv_series)
    labels = np.zeros(n, dtype=int)
    
    # Handle edge cases
    finite_mask = np.isfinite(iv_series)
    if np.sum(finite_mask) < 5:
        return np.ones(n, dtype=int)  # Default to Normal_Vol
    
    # Compute volatility risk premium (IV - RV)
    vol_premium = iv_series - realized_vol
    
    # Percentile-based regime assignment
    iv_finite = iv_series[finite_mask]
    iv_percentiles = np.percentile(iv_finite, [20, 40, 60, 80])
    
    for i in range(n):
        iv = iv_series[i]
        if not np.isfinite(iv):
            labels[i] = 1  # Default to Normal_Vol
            continue
            
        if iv <= iv_percentiles[0]:
            labels[i] = 0  # Low_Vol
        elif iv <= iv_percentiles[1]:
            labels[i] = 1  # Normal_Vol
        elif iv <= iv_percentiles[2]:
            labels[i] = 2  # Elevated_Vol
        elif iv <= iv_percentiles[3]:
            labels[i] = 3  # High_Vol
        else:
            labels[i] = 4  # Extreme_Vol
    
    # If equity regime provided, use as tiebreaker/prior
    if equity_regime is not None and len(equity_regime) == n:
        for i in range(n):
            eq_r = equity_regime[i]
            if eq_r in EQUITY_TO_VOL_REGIME:
                # Bias toward equity-implied vol regime
                implied_vol_regime = EQUITY_TO_VOL_REGIME[eq_r]
                # Take maximum of pure-vol and equity-implied
                labels[i] = max(labels[i], implied_vol_regime)
    
    return labels


def get_expiry_stratum(days_to_expiry: int) -> str:
    """
    Classify option by expiry stratum for stratified processing.
    
    Returns:
        Stratum name: 'near', 'short', 'medium', 'long'
    """
    if days_to_expiry <= EXPIRY_NEAR:
        return 'near'
    elif days_to_expiry <= EXPIRY_SHORT:
        return 'short'
    elif days_to_expiry <= EXPIRY_MEDIUM:
        return 'medium'
    else:
        return 'long'


def get_expiry_weight(days_to_expiry: int) -> float:
    """
    Compute observation weight based on expiry proximity.
    
    Near-expiry observations receive reduced weight due to
    non-stationary dynamics (gamma explosion, liquidity withdrawal).
    
    Returns:
        Weight in (0, 1]
    """
    if days_to_expiry <= 3:
        return 0.3  # Heavy discount for very near expiry
    elif days_to_expiry <= EXPIRY_NEAR:
        return 0.5  # Moderate discount for near expiry
    elif days_to_expiry <= EXPIRY_SHORT:
        return 0.8  # Light discount for short term
    else:
        return 1.0  # Full weight for medium+ term


# =============================================================================
# VOLATILITY MODEL FITTING
# =============================================================================

def _fit_constant_vol(
    iv_data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
    prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
) -> Dict[str, Any]:
    """
    Fit constant volatility model with regularization.
    
    Model: σ_t = σ (constant)
    
    Returns dict with: sigma, log_likelihood, bic, aic, fit_success, confidence_bounds
    """
    valid_mask = np.isfinite(iv_data)
    valid_data = iv_data[valid_mask]
    n = len(valid_data)
    
    if n < 5:
        return {"fit_success": False, "error": "insufficient_data"}
    
    # Apply weights if provided
    if weights is not None:
        w = weights[valid_mask]
        w = w / np.sum(w)  # Normalize
    else:
        w = np.ones(n) / n
    
    # Weighted MLE with prior regularization
    sigma_mle = np.sum(w * valid_data)
    
    # Posterior mean with Gaussian prior
    effective_n = n  # Could use effective sample size from weights
    sigma_posterior = (
        (effective_n * sigma_mle + prior_lambda * prior_mean) / 
        (effective_n + prior_lambda)
    )
    
    # Log-likelihood (weighted Gaussian observation model)
    residuals = valid_data - sigma_posterior
    sigma_resid = np.sqrt(np.sum(w * residuals ** 2)) + 1e-8
    ll = np.sum(w * norm.logpdf(residuals, 0, sigma_resid))
    
    # Information criteria
    k = 1  # Number of parameters
    aic = 2 * k - 2 * ll * n
    bic = k * np.log(n) - 2 * ll * n
    
    # Confidence bounds (using weighted standard error)
    se = sigma_resid / np.sqrt(effective_n)
    z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
    ci_lower = max(0.01, sigma_posterior - z * se)
    ci_upper = min(2.0, sigma_posterior + z * se)
    
    return {
        "model_class": "constant_vol",
        "sigma": float(sigma_posterior),
        "sigma_mle": float(sigma_mle),
        "log_likelihood": float(ll),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": k,
        "n_obs": n,
        "residual_std": float(sigma_resid),
        "confidence_bounds": {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
            "level": DEFAULT_CONFIDENCE_LEVEL,
        },
        "fit_success": True,
    }


def _fit_mean_reverting_vol(
    iv_data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
    prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
    prior_kappa: float = DEFAULT_KAPPA_PRIOR,
) -> Dict[str, Any]:
    """
    Fit mean-reverting volatility model (Ornstein-Uhlenbeck).
    
    Model: dσ_t = κ(σ̄ - σ_t)dt + η·dW_t
    Discrete: σ_{t+1} = σ_t + κ(σ̄ - σ_t) + ε_t
    """
    valid_mask = np.isfinite(iv_data)
    valid_data = iv_data[valid_mask]
    n = len(valid_data)
    
    if n < 10:
        return {"fit_success": False, "error": "insufficient_data"}
    
    # Apply weights
    if weights is not None:
        w = weights[valid_mask][:-1]  # Weights for transitions
        w = w / np.sum(w)
    else:
        w = np.ones(n - 1) / (n - 1)
    
    def neg_log_likelihood(params):
        sigma_bar, kappa, eta = params
        if kappa <= 0 or kappa > 1 or eta <= 1e-8:
            return 1e10
        if sigma_bar <= 0.01 or sigma_bar > 2.0:
            return 1e10
        
        predicted = valid_data[:-1] + kappa * (sigma_bar - valid_data[:-1])
        residuals = valid_data[1:] - predicted
        
        ll = np.sum(w * norm.logpdf(residuals, 0, eta))
        
        # Priors
        ll -= prior_lambda * (sigma_bar - prior_mean) ** 2
        ll -= 0.1 * (kappa - prior_kappa) ** 2
        
        return -ll
    
    # Initial guess
    x0 = [np.mean(valid_data), 0.1, np.std(np.diff(valid_data))]
    
    try:
        result = optimize.minimize(
            neg_log_likelihood, x0,
            method='L-BFGS-B',
            bounds=[(0.01, 1.0), (0.001, 0.99), (0.001, 1.0)]
        )
        
        sigma_bar, kappa, eta = result.x
        ll = -result.fun
        
        k = 3
        aic = 2 * k - 2 * ll * (n - 1)
        bic = k * np.log(n - 1) - 2 * ll * (n - 1)
        
        # Confidence bounds on long-run mean
        se = eta / np.sqrt(2 * kappa * n)
        z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
        ci_lower = max(0.01, sigma_bar - z * se)
        ci_upper = min(2.0, sigma_bar + z * se)
        
        return {
            "model_class": "mean_reverting_vol",
            "sigma_bar": float(sigma_bar),
            "kappa": float(kappa),
            "eta": float(eta),
            "half_life_days": float(np.log(2) / kappa) if kappa > 0 else float('inf'),
            "log_likelihood": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "n_params": k,
            "n_obs": n,
            "confidence_bounds": {
                "sigma_bar_lower": float(ci_lower),
                "sigma_bar_upper": float(ci_upper),
                "level": DEFAULT_CONFIDENCE_LEVEL,
            },
            "fit_success": True,
        }
    except Exception as e:
        return {"fit_success": False, "error": str(e), "model_class": "mean_reverting_vol"}


def _fit_regime_vol(
    iv_data: np.ndarray,
    regime_labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
    prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
) -> Dict[str, Any]:
    """
    Fit regime-conditional volatility model.
    
    Model: σ_t | r_t ~ N(σ_r, η_r) for each regime r
    Analogous to kalman_gaussian in equity tuning.
    """
    n = len(iv_data)
    valid_mask = np.isfinite(iv_data)
    
    if np.sum(valid_mask) < 10:
        return {"fit_success": False, "error": "insufficient_data"}
    
    # Apply weights
    if weights is not None:
        w = weights.copy()
    else:
        w = np.ones(n)
    
    regime_params = {}
    total_ll = 0
    total_params = 0
    
    # Global fallback
    global_sigma = np.average(iv_data[valid_mask], weights=w[valid_mask])
    global_eta = np.sqrt(np.average(
        (iv_data[valid_mask] - global_sigma) ** 2, 
        weights=w[valid_mask]
    ))
    
    for r in np.unique(regime_labels):
        mask = (regime_labels == r) & valid_mask
        n_r = np.sum(mask)
        
        if n_r < MIN_VOL_SAMPLES:
            # Fallback to global with shrinkage
            regime_params[int(r)] = {
                "sigma": float(global_sigma),
                "eta": float(global_eta),
                "n_obs": int(n_r),
                "fallback": True,
                "regime_name": VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
            }
            continue
        
        regime_data = iv_data[mask]
        regime_weights = w[mask]
        regime_weights = regime_weights / np.sum(regime_weights)
        
        # Posterior mean with shrinkage toward global
        sigma_r_mle = np.sum(regime_weights * regime_data)
        shrinkage = min(1.0, MIN_VOL_SAMPLES / n_r)
        sigma_r = shrinkage * global_sigma + (1 - shrinkage) * sigma_r_mle
        
        eta_r = np.sqrt(np.sum(regime_weights * (regime_data - sigma_r) ** 2)) + 1e-8
        
        # Regime log-likelihood
        ll_r = np.sum(regime_weights * norm.logpdf(regime_data, sigma_r, eta_r)) * n_r
        total_ll += ll_r
        total_params += 2
        
        # Confidence bounds
        se = eta_r / np.sqrt(n_r)
        z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
        
        regime_params[int(r)] = {
            "sigma": float(sigma_r),
            "eta": float(eta_r),
            "n_obs": int(n_r),
            "fallback": False,
            "regime_name": VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
            "confidence_bounds": {
                "lower": float(max(0.01, sigma_r - z * se)),
                "upper": float(min(2.0, sigma_r + z * se)),
            },
        }
    
    k = total_params
    n_total = np.sum(valid_mask)
    aic = 2 * k - 2 * total_ll
    bic = k * np.log(n_total) - 2 * total_ll
    
    return {
        "model_class": "regime_vol",
        "regime_params": regime_params,
        "global_sigma": float(global_sigma),
        "global_eta": float(global_eta),
        "log_likelihood": float(total_ll),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": k,
        "n_obs": int(n_total),
        "fit_success": True,
    }


def _fit_regime_skew_vol(
    iv_data: np.ndarray,
    skew_data: np.ndarray,
    regime_labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
    prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
) -> Dict[str, Any]:
    """
    Fit regime-conditional volatility with skew model.
    
    Model: (σ_t, ξ_t) | r_t ~ Bivariate normal per regime
    
    Analogous to kalman_phi_gaussian in equity tuning.
    ξ represents volatility skew (put-call IV difference).
    """
    n = len(iv_data)
    valid_mask = np.isfinite(iv_data) & np.isfinite(skew_data)
    
    if np.sum(valid_mask) < 15:
        return {"fit_success": False, "error": "insufficient_data"}
    
    # Apply weights
    if weights is not None:
        w = weights.copy()
    else:
        w = np.ones(n)
    
    regime_params = {}
    total_ll = 0
    total_params = 0
    
    # Global fallbacks
    global_sigma = np.average(iv_data[valid_mask], weights=w[valid_mask])
    global_skew = np.average(skew_data[valid_mask], weights=w[valid_mask])
    global_eta = np.sqrt(np.average(
        (iv_data[valid_mask] - global_sigma) ** 2, 
        weights=w[valid_mask]
    )) + 1e-8
    global_skew_eta = np.sqrt(np.average(
        (skew_data[valid_mask] - global_skew) ** 2,
        weights=w[valid_mask]
    )) + 1e-8
    
    for r in np.unique(regime_labels):
        mask = (regime_labels == r) & valid_mask
        n_r = np.sum(mask)
        
        if n_r < MIN_VOL_SAMPLES:
            regime_params[int(r)] = {
                "sigma": float(global_sigma),
                "skew": float(global_skew),
                "eta": float(global_eta),
                "skew_eta": float(global_skew_eta),
                "n_obs": int(n_r),
                "fallback": True,
                "regime_name": VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
            }
            continue
        
        regime_iv = iv_data[mask]
        regime_skew = skew_data[mask]
        regime_weights = w[mask]
        regime_weights = regime_weights / np.sum(regime_weights)
        
        # Posterior mean with shrinkage
        shrinkage = min(1.0, MIN_VOL_SAMPLES / n_r)
        
        sigma_r_mle = np.sum(regime_weights * regime_iv)
        sigma_r = shrinkage * global_sigma + (1 - shrinkage) * sigma_r_mle
        
        skew_r_mle = np.sum(regime_weights * regime_skew)
        skew_r = shrinkage * global_skew + (1 - shrinkage) * skew_r_mle
        
        eta_r = np.sqrt(np.sum(regime_weights * (regime_iv - sigma_r) ** 2)) + 1e-8
        skew_eta_r = np.sqrt(np.sum(regime_weights * (regime_skew - skew_r) ** 2)) + 1e-8
        
        # Joint log-likelihood (independent for simplicity)
        ll_r = np.sum(regime_weights * norm.logpdf(regime_iv, sigma_r, eta_r)) * n_r
        ll_r += np.sum(regime_weights * norm.logpdf(regime_skew, skew_r, skew_eta_r)) * n_r
        total_ll += ll_r
        total_params += 4
        
        # Confidence bounds
        se_sigma = eta_r / np.sqrt(n_r)
        se_skew = skew_eta_r / np.sqrt(n_r)
        z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
        
        regime_params[int(r)] = {
            "sigma": float(sigma_r),
            "skew": float(skew_r),
            "eta": float(eta_r),
            "skew_eta": float(skew_eta_r),
            "n_obs": int(n_r),
            "fallback": False,
            "regime_name": VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
            "confidence_bounds": {
                "sigma_lower": float(max(0.01, sigma_r - z * se_sigma)),
                "sigma_upper": float(min(2.0, sigma_r + z * se_sigma)),
                "skew_lower": float(skew_r - z * se_skew),
                "skew_upper": float(skew_r + z * se_skew),
            },
        }
    
    k = total_params
    n_total = np.sum(valid_mask)
    aic = 2 * k - 2 * total_ll
    bic = k * np.log(n_total) - 2 * total_ll
    
    return {
        "model_class": "regime_skew_vol",
        "regime_params": regime_params,
        "global_sigma": float(global_sigma),
        "global_skew": float(global_skew),
        "log_likelihood": float(total_ll),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": k,
        "n_obs": int(n_total),
        "fit_success": True,
    }


def _fit_variance_swap_anchor(
    iv_data: np.ndarray,
    strikes: np.ndarray,
    current_price: float,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute model-free variance swap fair value from option chain.
    
    Uses the replication formula:
    VarSwap = 2/T * [∫_{0}^{F} P(K)/K² dK + ∫_{F}^{∞} C(K)/K² dK]
    
    Simplified: weighted average of squared IVs across strikes.
    """
    valid_mask = np.isfinite(iv_data) & np.isfinite(strikes) & (strikes > 0)
    
    if np.sum(valid_mask) < 5:
        return {"fit_success": False, "error": "insufficient_strikes"}
    
    valid_iv = iv_data[valid_mask]
    valid_strikes = strikes[valid_mask]
    
    if weights is not None:
        w = weights[valid_mask]
    else:
        # Weight by distance from ATM (closer = higher weight)
        moneyness = np.abs(np.log(valid_strikes / current_price))
        w = np.exp(-2 * moneyness)  # Decay away from ATM
    
    w = w / np.sum(w)
    
    # Variance swap fair value (as volatility)
    variance_swap_var = np.sum(w * valid_iv ** 2)
    variance_swap_vol = np.sqrt(variance_swap_var)
    
    # Standard error using delta method
    iv_std = np.sqrt(np.sum(w * (valid_iv - np.mean(valid_iv)) ** 2))
    se = iv_std / np.sqrt(len(valid_iv))
    
    z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
    
    return {
        "model_class": "variance_swap_anchor",
        "variance_swap_vol": float(variance_swap_vol),
        "variance_swap_var": float(variance_swap_var),
        "n_strikes": int(np.sum(valid_mask)),
        "atm_weight": float(w[np.argmin(np.abs(valid_strikes - current_price))]),
        "confidence_bounds": {
            "lower": float(max(0.01, variance_swap_vol - z * se)),
            "upper": float(min(2.0, variance_swap_vol + z * se)),
        },
        # Model-free approach doesn't have traditional likelihood
        "log_likelihood": float(-0.5 * np.sum(w * (valid_iv - variance_swap_vol) ** 2) * len(valid_iv)),
        "bic": float(np.log(len(valid_iv)) - 2 * (-0.5 * np.sum(w * (valid_iv - variance_swap_vol) ** 2) * len(valid_iv))),
        "aic": 2 - 2 * (-0.5 * np.sum(w * (valid_iv - variance_swap_vol) ** 2) * len(valid_iv)),
        "n_params": 1,
        "n_obs": int(np.sum(valid_mask)),
        "fit_success": True,
    }


# =============================================================================
# HYVÄRINEN SCORE FOR VOLATILITY MODELS (matching tune.py pattern)
# =============================================================================

def compute_vol_hyvarinen_score(
    model_result: Dict,
    iv_data: np.ndarray,
    regime_labels: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Hyvärinen score for volatility model.
    
    For volatility models, this measures how well the model captures
    the score function of the implied volatility distribution.
    
    Higher is better (matching equity convention).
    """
    model_class = model_result.get("model_class", "")
    
    valid_mask = np.isfinite(iv_data)
    valid_data = iv_data[valid_mask]
    n = len(valid_data)
    
    if n < 10:
        return float('-inf')
    
    try:
        if model_class == "constant_vol":
            sigma = model_result.get("sigma", DEFAULT_VOL_PRIOR_MEAN)
            residual_std = model_result.get("residual_std", 0.05)
            residuals = valid_data - sigma
            score = residuals / (residual_std ** 2 + 1e-8)
            
        elif model_class == "mean_reverting_vol":
            sigma_bar = model_result.get("sigma_bar", DEFAULT_VOL_PRIOR_MEAN)
            eta = model_result.get("eta", 0.05)
            kappa = model_result.get("kappa", 0.1)
            
            # Score for mean-reverting process
            predicted = valid_data[:-1] + kappa * (sigma_bar - valid_data[:-1])
            residuals = valid_data[1:] - predicted
            score = residuals / (eta ** 2 + 1e-8)
            
        elif model_class == "regime_vol":
            regime_params = model_result.get("regime_params", {})
            global_sigma = model_result.get("global_sigma", DEFAULT_VOL_PRIOR_MEAN)
            global_eta = model_result.get("global_eta", 0.05)
            
            if regime_labels is None:
                regime_labels = np.zeros(n, dtype=int)
            
            score = np.zeros(n)
            for i, (iv, r) in enumerate(zip(valid_data, regime_labels[valid_mask] if regime_labels is not None else np.zeros(n))):
                params = regime_params.get(int(r), {"sigma": global_sigma, "eta": global_eta})
                sigma_r = params.get("sigma", global_sigma)
                eta_r = params.get("eta", global_eta)
                score[i] = (iv - sigma_r) / (eta_r ** 2 + 1e-8)
                
        elif model_class == "regime_skew_vol":
            regime_params = model_result.get("regime_params", {})
            global_sigma = model_result.get("global_sigma", DEFAULT_VOL_PRIOR_MEAN)
            global_eta = model_result.get("global_eta", 0.05)
            
            if regime_labels is None:
                regime_labels = np.zeros(n, dtype=int)
            
            score = np.zeros(n)
            for i, (iv, r) in enumerate(zip(valid_data, regime_labels[valid_mask] if regime_labels is not None else np.zeros(n))):
                params = regime_params.get(int(r), {"sigma": global_sigma, "eta": global_eta})
                sigma_r = params.get("sigma", global_sigma)
                eta_r = params.get("eta", global_eta)
                score[i] = (iv - sigma_r) / (eta_r ** 2 + 1e-8)
                
        elif model_class == "variance_swap_anchor":
            vs_vol = model_result.get("variance_swap_vol", DEFAULT_VOL_PRIOR_MEAN)
            residuals = valid_data - vs_vol
            score = residuals / (np.std(residuals) ** 2 + 1e-8)
            
        else:
            return float('-inf')
        
        # Hyvärinen score approximation
        score_sq = np.mean(score ** 2)
        hyvarinen = -score_sq  # Higher (less negative) is better
        
        return float(hyvarinen)
        
    except Exception:
        return float('-inf')


# =============================================================================
# BMA MODEL AVERAGING (matching tune.py structure)
# =============================================================================

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1."""
    total = sum(w for w in weights.values() if np.isfinite(w) and w > 0)
    if total <= 0:
        n = len(weights)
        return {k: 1.0/n for k in weights}
    return {k: max(0, v)/total for k, v in weights.items()}


def compute_vol_model_posterior(
    model_results: Dict[str, Dict],
    method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
    entropy_lambda: float = DEFAULT_ENTROPY_LAMBDA,
) -> Dict[str, float]:
    """
    Compute posterior probabilities over volatility models.
    
    Uses combined BIC + Hyvärinen scoring (matching equity tune.py).
    """
    successful_models = {
        m: r for m, r in model_results.items() 
        if r.get("fit_success", False)
    }
    
    if not successful_models:
        n = len(VOL_MODEL_CLASSES)
        return {m: 1.0/n for m in VOL_MODEL_CLASSES}
    
    # Extract scores
    bic_scores = {}
    hyv_scores = {}
    
    for m, r in successful_models.items():
        bic_scores[m] = r.get("bic", float('inf'))
        hyv_scores[m] = r.get("hyvarinen_score", float('-inf'))
    
    # BIC-based posterior (lower is better)
    if bic_scores:
        finite_bics = [b for b in bic_scores.values() if np.isfinite(b)]
        if finite_bics:
            min_bic = min(finite_bics)
            bic_weights = {
                m: np.exp(-0.5 * (bic - min_bic)) if np.isfinite(bic) else 0
                for m, bic in bic_scores.items()
            }
        else:
            bic_weights = {m: 1.0 for m in successful_models}
    else:
        bic_weights = {m: 1.0 for m in successful_models}
    
    # Hyvärinen-based posterior (higher is better)
    if hyv_scores and any(np.isfinite(h) for h in hyv_scores.values()):
        finite_hyvs = [h for h in hyv_scores.values() if np.isfinite(h)]
        if finite_hyvs:
            max_hyv = max(finite_hyvs)
            hyv_weights = {
                m: np.exp(h - max_hyv) if np.isfinite(h) else 0
                for m, h in hyv_scores.items()
            }
        else:
            hyv_weights = {m: 1.0 for m in successful_models}
    else:
        hyv_weights = {m: 1.0 for m in successful_models}
    
    # Combined scoring
    if method == 'combined':
        combined_weights = {}
        for m in successful_models:
            bic_w = bic_weights.get(m, 0)
            hyv_w = hyv_weights.get(m, 0)
            combined_weights[m] = bic_weight * bic_w + (1 - bic_weight) * hyv_w
    elif method == 'bic':
        combined_weights = bic_weights
    else:  # hyvarinen
        combined_weights = hyv_weights
    
    # Entropy regularization for robustness
    if entropy_lambda > 0:
        n_models = len(combined_weights)
        if n_models > 0:
            uniform = {m: 1.0/n_models for m in combined_weights}
            for m in combined_weights:
                combined_weights[m] = (
                    (1 - entropy_lambda) * combined_weights[m] + 
                    entropy_lambda * uniform[m]
                )
    
    return normalize_weights(combined_weights)


# =============================================================================
# HIERARCHICAL PRIOR FROM EQUITY SIGNAL
# =============================================================================

def equity_signal_to_vol_prior(
    equity_signal: Dict,
    base_vol_prior: float = DEFAULT_VOL_PRIOR_MEAN,
) -> Dict[str, float]:
    """
    Transform equity BMA signal into informative prior for volatility.
    
    Strong BUY (high p_up) → expect volatility compression (lower vol favorable)
    Strong SELL (low p_up) → expect volatility expansion (higher vol favorable)
    
    This implements the hierarchical structure:
        equity regime → equity posterior → options vol prior
    """
    p_up = equity_signal.get("probability_up", equity_signal.get("p_up", 0.5))
    exp_ret = equity_signal.get("expected_return_pct", equity_signal.get("exp_ret", 0.0))
    if isinstance(exp_ret, (int, float)) and abs(exp_ret) < 1:
        exp_ret = exp_ret * 100  # Convert to percentage if needed
    
    model_posterior = equity_signal.get("model_posterior", {})
    
    # Conviction strength
    conviction = abs(p_up - 0.5) * 2  # 0 to 1
    
    # Adjust vol prior based on conviction
    # High conviction → lower vol prior (expect calm, vol mean reversion)
    # Low conviction → higher vol prior (expect turbulence)
    vol_prior_adj = base_vol_prior * (1 + 0.2 * (1 - conviction))
    
    # Skew prior based on direction
    # Strong BUY → expect negative skew (calls relatively cheaper)
    # Strong SELL → expect positive skew (puts relatively cheaper)
    if p_up > 0.5:
        skew_prior = -0.05 * (p_up - 0.5) * 2  # Negative skew for bullish
    else:
        skew_prior = 0.05 * (0.5 - p_up) * 2  # Positive skew for bearish
    
    # Model preference based on equity model posterior
    equity_regime_weight = 0
    if model_posterior:
        regime_models = ["kalman_gaussian", "kalman_phi_gaussian"] + \
                       [f"phi_student_t_nu_{nu}" for nu in [4, 8, 20]]
        equity_regime_weight = sum(
            model_posterior.get(m, 0) 
            for m in regime_models
        )
    
    return {
        "vol_prior_mean": vol_prior_adj,
        "skew_prior_mean": skew_prior,
        "kappa_prior": DEFAULT_KAPPA_PRIOR * (1 + 0.5 * conviction),
        "regime_model_bonus": 0.1 * equity_regime_weight,
        "equity_conviction": conviction,
        "equity_direction": "bullish" if p_up > 0.5 else "bearish",
    }


# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================

def tune_options_volatility(
    iv_data: np.ndarray,
    returns: np.ndarray,
    skew_data: Optional[np.ndarray] = None,
    strikes: Optional[np.ndarray] = None,
    current_price: Optional[float] = None,
    days_to_expiry: Optional[np.ndarray] = None,
    regime_labels: Optional[np.ndarray] = None,
    equity_signal: Optional[Dict] = None,
    prior_vol_mean: float = DEFAULT_VOL_PRIOR_MEAN,
    prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[str, float]] = None,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
) -> Dict[str, Any]:
    """
    Perform Bayesian Model Averaging over volatility models.
    
    Implements:
        p(σ_{t+H} | r, S) = Σ_m p(σ_{t+H} | r, m, θ_m) · p(m | r, S)
    
    Where S is the equity signal providing an informative prior.
    
    Args:
        iv_data: Implied volatility time series or cross-section
        returns: Underlying returns (for regime detection)
        skew_data: Optional volatility skew time series
        strikes: Optional strike prices (for variance swap calculation)
        current_price: Optional current underlying price
        days_to_expiry: Optional days to expiry per observation
        regime_labels: Optional volatility regime labels
        equity_signal: Optional equity signal dict
        prior_vol_mean: Prior mean for volatility
        prior_lambda: Regularization strength
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous model posteriors (for smoothing)
        model_selection_method: 'bic', 'hyvarinen', or 'combined'
        bic_weight: Weight for BIC in combined method
        
    Returns:
        Dict with model posteriors, parameters, and confidence bounds
    """
    n_obs = len(iv_data)
    
    # Compute expiry-based weights
    if days_to_expiry is not None:
        weights = np.array([get_expiry_weight(d) for d in days_to_expiry])
    else:
        weights = None
    
    # Apply equity signal prior if available
    if equity_signal is not None:
        equity_prior = equity_signal_to_vol_prior(equity_signal, prior_vol_mean)
        prior_vol_mean = equity_prior["vol_prior_mean"]
        skew_prior = equity_prior["skew_prior_mean"]
    else:
        equity_prior = None
        skew_prior = DEFAULT_SKEW_PRIOR
    
    # Assign regime labels if not provided
    if regime_labels is None:
        finite_mask = np.isfinite(iv_data)
        if np.sum(finite_mask) > 5:
            # Compute realized vol from returns
            if len(returns) > 5:
                realized_vol = np.std(returns) * np.sqrt(252)
            else:
                realized_vol = np.nanmean(iv_data)
            
            regime_labels = assign_vol_regime_labels(
                iv_data,
                np.full_like(iv_data, realized_vol),
                None
            )
        else:
            regime_labels = np.ones(n_obs, dtype=int)
    
    # Initialize skew data if not provided
    if skew_data is None:
        skew_data = np.zeros_like(iv_data)
    
    # Extract momentum parameters from equity signal
    momentum_params = None
    if equity_signal is not None and OPTION_MODEL_REGISTRY_AVAILABLE:
        try:
            momentum_params = extract_momentum_parameters(equity_signal)
        except Exception:
            momentum_params = None
    
    # Fit all model classes
    global_models = {}
    
    # =========================================================================
    # MOMENTUM-COUPLED MODELS (preferred when available)
    # =========================================================================
    if USE_MOMENTUM_MODELS and OPTION_MODEL_REGISTRY_AVAILABLE:
        # 1. Momentum Gaussian — constant vol with momentum prior
        try:
            model = OptionMomentumGaussianModel(prior_mean=prior_vol_mean)
            result = model.fit(iv_data, weights, momentum_params)
            global_models[make_option_momentum_gaussian_name()] = model.to_dict(result)
        except Exception as e:
            global_models[make_option_momentum_gaussian_name()] = {
                "fit_success": False, "error": str(e), "model_class": "option_momentum_gaussian"
            }
        
        # 2. Momentum Phi-Gaussian — mean-reverting vol
        try:
            model = OptionMomentumPhiGaussianModel(prior_sigma_bar=prior_vol_mean)
            result = model.fit(iv_data, weights, momentum_params)
            global_models[make_option_momentum_phi_gaussian_name()] = model.to_dict(result)
        except Exception as e:
            global_models[make_option_momentum_phi_gaussian_name()] = {
                "fit_success": False, "error": str(e), "model_class": "option_momentum_phi_gaussian"
            }
        
        # 3. Momentum Phi-Student-t — regime-switching with heavy tails
        for nu in OPTION_STUDENT_T_NU_GRID:
            model_name = make_option_momentum_student_t_name(nu)
            try:
                model = OptionMomentumPhiStudentTModel(nu=nu, prior_sigma=prior_vol_mean)
                result = model.fit(
                    iv_data, regime_labels, weights, skew_data, momentum_params
                )
                global_models[model_name] = model.to_dict(result)
            except Exception as e:
                global_models[model_name] = {
                    "fit_success": False, "error": str(e), "model_class": model_name
                }
    
    # =========================================================================
    # LEGACY MODELS (kept for backward compatibility)
    # =========================================================================
    # 1. Constant volatility
    global_models["constant_vol"] = _fit_constant_vol(
        iv_data, weights, prior_vol_mean, prior_lambda
    )
    
    # 2. Mean-reverting volatility
    global_models["mean_reverting_vol"] = _fit_mean_reverting_vol(
        iv_data, weights, prior_vol_mean, prior_lambda
    )
    
    # 3. Regime volatility
    global_models["regime_vol"] = _fit_regime_vol(
        iv_data, regime_labels, weights, prior_vol_mean, prior_lambda
    )
    
    # 4. Regime skew volatility
    global_models["regime_skew_vol"] = _fit_regime_skew_vol(
        iv_data, skew_data, regime_labels, weights, prior_vol_mean, prior_lambda
    )
    
    # 5. Variance swap anchor (if strikes available)
    if strikes is not None and current_price is not None:
        global_models["variance_swap_anchor"] = _fit_variance_swap_anchor(
            iv_data, strikes, current_price, weights
        )
    else:
        global_models["variance_swap_anchor"] = {
            "fit_success": False, 
            "error": "strikes_or_price_not_provided"
        }
    
    # Compute Hyvärinen scores for all models
    for m, result in global_models.items():
        if result.get("fit_success", False):
            hyv_score = compute_vol_hyvarinen_score(result, iv_data, regime_labels)
            result["hyvarinen_score"] = hyv_score
    
    # Compute model posterior
    global_posterior = compute_vol_model_posterior(
        global_models, 
        method=model_selection_method,
        bic_weight=bic_weight,
    )
    
    # Apply equity prior bonus to regime models if applicable
    if equity_prior is not None:
        regime_bonus = equity_prior.get("regime_model_bonus", 0)
        if regime_bonus > 0:
            for m in ["regime_vol", "regime_skew_vol"]:
                if m in global_posterior:
                    global_posterior[m] *= (1 + regime_bonus)
            global_posterior = normalize_weights(global_posterior)
    
    # Temporal smoothing with previous posteriors
    if previous_posteriors is not None and temporal_alpha < 1.0:
        for m in global_posterior:
            prev_p = previous_posteriors.get(m, 1.0 / len(global_posterior))
            if np.isfinite(prev_p) and prev_p > 0:
                global_posterior[m] = (
                    temporal_alpha * global_posterior[m] + 
                    (1 - temporal_alpha) * prev_p
                )
        global_posterior = normalize_weights(global_posterior)
    
    # Compute ensemble volatility forecast with confidence bounds
    ensemble_vol = 0
    ensemble_var = 0  # For confidence interval
    
    for m, weight in global_posterior.items():
        if weight <= 0:
            continue
        model = global_models.get(m, {})
        if not model.get("fit_success", False):
            continue
        
        if m == "constant_vol":
            model_vol = model.get("sigma", prior_vol_mean)
            model_se = model.get("residual_std", 0.05) / np.sqrt(model.get("n_obs", 1))
        elif m == "mean_reverting_vol":
            model_vol = model.get("sigma_bar", prior_vol_mean)
            eta = model.get("eta", 0.05)
            kappa = model.get("kappa", 0.1)
            model_se = eta / np.sqrt(2 * kappa * model.get("n_obs", 1))
        elif m == "regime_vol":
            # Use current regime's estimate
            current_regime = int(regime_labels[-1]) if len(regime_labels) > 0 else 1
            regime_params = model.get("regime_params", {})
            r_params = regime_params.get(current_regime, {})
            model_vol = r_params.get("sigma", model.get("global_sigma", prior_vol_mean))
            model_se = r_params.get("eta", 0.05) / np.sqrt(r_params.get("n_obs", 1) + 1)
        elif m == "regime_skew_vol":
            current_regime = int(regime_labels[-1]) if len(regime_labels) > 0 else 1
            regime_params = model.get("regime_params", {})
            r_params = regime_params.get(current_regime, {})
            model_vol = r_params.get("sigma", model.get("global_sigma", prior_vol_mean))
            model_se = r_params.get("eta", 0.05) / np.sqrt(r_params.get("n_obs", 1) + 1)
        elif m == "variance_swap_anchor":
            model_vol = model.get("variance_swap_vol", prior_vol_mean)
            cb = model.get("confidence_bounds", {})
            model_se = (cb.get("upper", model_vol) - cb.get("lower", model_vol)) / (2 * 1.96)
        else:
            continue
        
        ensemble_vol += weight * model_vol
        ensemble_var += weight ** 2 * model_se ** 2
    
    ensemble_se = np.sqrt(ensemble_var) if ensemble_var > 0 else 0.05
    z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
    
    # Build result
    result = {
        "global": {
            "model_posterior": global_posterior,
            "models": global_models,
            "model_selection_method": model_selection_method,
            "bic_weight": bic_weight,
            "ensemble_forecast": {
                "volatility": float(ensemble_vol),
                "confidence_lower": float(max(0.01, ensemble_vol - z * ensemble_se)),
                "confidence_upper": float(min(2.0, ensemble_vol + z * ensemble_se)),
                "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            },
        },
        "regime": {},
        "meta": {
            "n_obs": n_obs,
            "temporal_alpha": temporal_alpha,
            "equity_prior_applied": equity_prior is not None,
            "timestamp": datetime.now().isoformat(),
        },
    }
    
    # Add equity prior info
    if equity_prior is not None:
        result["equity_prior"] = equity_prior
    
    return result


def tune_ticker_options(
    ticker: str,
    options_chain: Dict,
    price_history: Dict,
    equity_signal: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Tune volatility models for a single ticker's options.
    
    Args:
        ticker: Ticker symbol
        options_chain: Options chain data from high_conviction_storage
        price_history: Price history data from high_conviction_storage
        equity_signal: Equity signal dict
        
    Returns:
        Tuned options parameters with model posteriors
    """
    if options_chain.get("skipped") or options_chain.get("error"):
        return None
    
    options = options_chain.get("options", [])
    if len(options) < 5:
        return None
    
    # Extract IV data
    iv_data = np.array([opt.get("implied_volatility_pct", 0) / 100 for opt in options])
    strikes = np.array([opt.get("strike", 0) for opt in options])
    current_price = options_chain.get("underlying_price", 0)
    dte = np.array([opt.get("days_to_expiration", 30) for opt in options])
    
    # Extract skew from IV surface metrics if available
    iv_surface = options_chain.get("iv_surface_metrics", {})
    if iv_surface and not iv_surface.get("error"):
        # Use per-expiry skew data
        by_expiry = iv_surface.get("by_expiry", {})
        skew_data = np.zeros_like(iv_data)
        for i, opt in enumerate(options):
            exp = opt.get("expiration", "")
            if exp in by_expiry:
                skew_data[i] = by_expiry[exp].get("skew", 0)
    else:
        skew_data = np.zeros_like(iv_data)
    
    # Get returns from price history
    prices = price_history.get("prices", [])
    if len(prices) > 1:
        closes = np.array([p.get("close", 0) for p in prices])
        valid_closes = closes[closes > 0]
        if len(valid_closes) > 1:
            returns = np.diff(np.log(valid_closes))
        else:
            returns = np.array([0.0])
    else:
        returns = np.array([0.0])
    
    try:
        result = tune_options_volatility(
            iv_data=iv_data,
            returns=returns,
            skew_data=skew_data,
            strikes=strikes,
            current_price=current_price,
            days_to_expiry=dte,
            equity_signal=equity_signal,
        )
        
        if result:
            result["ticker"] = ticker
            result["meta"]["timestamp"] = datetime.now().isoformat()
            
        return result
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "fit_success": False}


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def load_option_tune_cache() -> Dict[str, Dict]:
    """Load cached options tuning results."""
    cache = {}
    if not os.path.exists(OPTION_TUNE_CACHE_DIR):
        os.makedirs(OPTION_TUNE_CACHE_DIR, exist_ok=True)
        return cache
    
    for filename in os.listdir(OPTION_TUNE_CACHE_DIR):
        if filename.endswith(".json") and filename != "manifest.json":
            ticker = filename[:-5]
            filepath = os.path.join(OPTION_TUNE_CACHE_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    cache[ticker] = json.load(f)
            except Exception:
                continue
    
    return cache


def save_option_tune_cache(cache: Dict[str, Dict]) -> None:
    """Save options tuning results to cache."""
    os.makedirs(OPTION_TUNE_CACHE_DIR, exist_ok=True)
    
    for ticker, result in cache.items():
        # Sanitize ticker for filename
        safe_ticker = ticker.replace("^", "_").replace("=", "_").replace(".", "_")
        filepath = os.path.join(OPTION_TUNE_CACHE_DIR, f"{safe_ticker}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception:
            continue
    
    # Write manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "n_tickers": len(cache),
        "tickers": sorted(cache.keys()),
    }
    with open(os.path.join(OPTION_TUNE_CACHE_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# CLI - STANDALONE OPTIONS TUNING PIPELINE
# =============================================================================

def run_options_tuning_pipeline(
    force_retune: bool = False,
    max_workers: int = 8,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Run the full options volatility tuning pipeline.
    
    This is the main entry point for `make options-tune`.
    
    Reads high conviction signals from src/data/high_conviction/
    Tunes volatility models for each ticker
    Saves results to src/data/option_tune/
    
    Args:
        force_retune: Force re-tuning even if cached
        max_workers: Number of parallel workers
        dry_run: Preview only, don't process
        
    Returns:
        Summary dict with counts
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import json
    
    console = Console()
    
    # High conviction directory
    HIGH_CONVICTION_DIR = os.path.join(
        os.path.dirname(__file__), "..", "data", "high_conviction"
    )
    
    # Clean header
    console.print()
    header_text = Text()
    header_text.append("OPTIONS VOLATILITY TUNING", style="bold bright_white")
    header_text.append("  —  ", style="dim")
    header_text.append("Bayesian Model Averaging", style="dim italic")
    
    console.print(Panel(
        Align.center(header_text),
        box=box.DOUBLE,
        border_style="bright_magenta",
        padding=(0, 2),
    ))
    console.print()
    
    # Configuration section
    config_table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=False,
    )
    config_table.add_column("Key", style="dim", width=20)
    config_table.add_column("Value", style="white")
    
    # Show momentum model availability
    if OPTION_MODEL_REGISTRY_AVAILABLE and USE_MOMENTUM_MODELS:
        model_list = "[bright_cyan]Mom-Gaussian, Mom-φ-Gaussian, Mom-φ-t(ν=4,6,8,12,20)[/] + legacy"
        config_table.add_row("Models", model_list)
    else:
        config_table.add_row("Models", "[bright_magenta]constant, mean_reverting, regime, regime_skew, variance_swap[/]")
        if not OPTION_MODEL_REGISTRY_AVAILABLE:
            config_table.add_row("Momentum", f"[red]DISABLED[/] [dim]({_import_error or 'import failed'})[/]")
    
    config_table.add_row("Selection", f"Combined BIC + Hyvärinen (α={DEFAULT_BIC_WEIGHT})")
    config_table.add_row("Temporal Smoothing", f"α={DEFAULT_TEMPORAL_ALPHA}")
    if force_retune:
        config_table.add_row("Mode", "[yellow]Force re-tune[/]")
    
    console.print(config_table)
    console.print()
    
    # Check if high conviction directory exists
    if not os.path.exists(HIGH_CONVICTION_DIR):
        console.print("  [red]ERROR:[/] High conviction directory not found")
        console.print("  [dim]Run `make stocks` first to generate equity signals.[/]")
        return {"error": "no_high_conviction_data", "processed": 0}
    
    # Load existing tune cache
    tune_cache = load_option_tune_cache()
    
    # Find all signal files with options data
    signal_files = []
    for subdir in ["buy", "sell"]:
        dir_path = os.path.join(HIGH_CONVICTION_DIR, subdir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".json") and filename != "manifest.json":
                    signal_files.append(os.path.join(dir_path, filename))
    
    # Count by type
    buy_count = sum(1 for f in signal_files if "/buy/" in f)
    sell_count = sum(1 for f in signal_files if "/sell/" in f)
    
    # Status line
    status_parts = []
    status_parts.append(f"[dim]Cache:[/] [bold]{len(tune_cache)}[/] tuned")
    status_parts.append(f"[dim]Signals:[/] [green]{buy_count} buy[/] · [red]{sell_count} sell[/]")
    console.print("  " + "    ".join(status_parts))
    console.print()
    
    if not signal_files:
        console.print("  [yellow]No high conviction signals found.[/] Run `make stocks` first.")
        return {"processed": 0, "tuned": 0, "cached": 0, "errors": 0}
    
    if dry_run:
        console.print("  [bold yellow]DRY RUN[/] — No processing")
        console.print()
        for f in signal_files[:10]:
            console.print(f"  [dim]Would tune:[/] {os.path.basename(f)}")
        if len(signal_files) > 10:
            console.print(f"  [dim]... and {len(signal_files) - 10} more[/]")
        return {"processed": 0, "tuned": 0, "cached": 0, "errors": 0, "dry_run": True}
    
    # Process signals
    tuned_count = 0
    cached_count = 0
    error_count = 0
    skipped_count = 0
    
    # Track unique tickers
    tickers_seen = set()
    tickers_tuned = []
    tickers_cached = []
    tickers_error = []
    
    console.print("  [bold]Tuning volatility models...[/]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Tuning options", total=len(signal_files))
        
        for signal_path in signal_files:
            filename = os.path.basename(signal_path)
            ticker = filename.split("_")[0]
            
            progress.update(task, advance=1)
            
            # Skip if already seen this ticker
            if ticker in tickers_seen and not force_retune:
                skipped_count += 1
                continue
            tickers_seen.add(ticker)
            
            # Check cache
            safe_ticker = ticker.replace("^", "_").replace("=", "_").replace(".", "_")
            if safe_ticker in tune_cache and not force_retune:
                cached_count += 1
                tickers_cached.append(ticker)
                continue
            
            # Load signal data
            try:
                with open(signal_path, 'r') as f:
                    signal_data = json.load(f)
            except Exception as e:
                error_count += 1
                tickers_error.append(ticker)
                continue
            
            # Extract options chain and price history
            options_chain = signal_data.get("options_chain", {})
            price_history = signal_data.get("price_history", {})
            
            # Skip if no valid options data
            if options_chain.get("skipped") or options_chain.get("error"):
                skipped_count += 1
                continue
            
            if not options_chain.get("options"):
                skipped_count += 1
                continue
            
            # Build equity signal dict for prior
            equity_signal = {
                "probability_up": signal_data.get("probability_up", 0.5),
                "expected_return_pct": signal_data.get("expected_return_pct", 0),
                "signal_type": signal_data.get("signal_type", "HOLD"),
            }
            
            # Tune volatility models
            try:
                result = tune_ticker_options(
                    ticker=ticker,
                    options_chain=options_chain,
                    price_history=price_history,
                    equity_signal=equity_signal,
                )
                
                if result and result.get("global", {}).get("model_posterior"):
                    tune_cache[safe_ticker] = result
                    tuned_count += 1
                    tickers_tuned.append(ticker)
                else:
                    error_count += 1
                    tickers_error.append(ticker)
                    
            except Exception as e:
                error_count += 1
                tickers_error.append(ticker)
    
    # Save updated cache
    save_option_tune_cache(tune_cache)
    
    # Summary
    console.print()
    
    # Build summary line
    summary_parts = []
    if tickers_tuned:
        summary_parts.append(f"[green]Tuned:[/] {', '.join(tickers_tuned[:10])}" + 
                           (f" +{len(tickers_tuned)-10}" if len(tickers_tuned) > 10 else ""))
    if tickers_cached:
        summary_parts.append(f"[cyan]Cached:[/] {len(tickers_cached)}")
    if tickers_error:
        summary_parts.append(f"[red]Errors:[/] {', '.join(tickers_error[:5])}" +
                           (f" +{len(tickers_error)-5}" if len(tickers_error) > 5 else ""))
    
    console.print("  " + "  ·  ".join(summary_parts))
    console.print()
    
    # Display model selection summary
    _render_tuning_summary(tune_cache, console)
    
    stats = {
        "processed": len(signal_files),
        "tuned": tuned_count,
        "cached": cached_count,
        "skipped": skipped_count,
        "errors": error_count,
        "total_in_cache": len(tune_cache),
    }
    
    return stats


def _render_tuning_summary(tune_cache: Dict[str, Dict], console) -> None:
    """Render summary of tuned volatility models."""
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    
    if not tune_cache:
        return
    
    # Build comprehensive model list including momentum models
    all_model_classes = set(VOL_MODEL_CLASSES)
    if USE_MOMENTUM_MODELS:
        all_model_classes.update(MOMENTUM_VOL_MODEL_CLASSES)
    
    # Count model selections AND compute average weights
    model_counts = {m: 0 for m in all_model_classes}
    model_avg_weights = {m: [] for m in all_model_classes}
    total_with_results = 0
    momentum_selected = 0
    legacy_selected = 0
    
    for ticker, result in tune_cache.items():
        global_result = result.get("global", {})
        posterior = global_result.get("model_posterior", {})
        
        if posterior:
            total_with_results += 1
            # Find winning model
            best_model = max(posterior.items(), key=lambda x: x[1])[0]
            if best_model not in model_counts:
                model_counts[best_model] = 0
            model_counts[best_model] += 1
            
            # Track momentum vs legacy
            if best_model.startswith("option_momentum"):
                momentum_selected += 1
            else:
                legacy_selected += 1
            
            # Collect weights for all models
            for m, w in posterior.items():
                if m not in model_avg_weights:
                    model_avg_weights[m] = []
                model_avg_weights[m].append(w)
    
    if total_with_results == 0:
        return
    
    # Compute average weights
    avg_weights = {}
    for m, weights in model_avg_weights.items():
        if weights:
            avg_weights[m] = sum(weights) / len(weights)
        else:
            avg_weights[m] = 0.0
    
    # Summary panel - show ALL models
    summary_table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    summary_table.add_column("Model", style="white", width=32)
    summary_table.add_column("Won", justify="right", style="bright_magenta", width=5)
    summary_table.add_column("Avg Wt", justify="right", style="cyan", width=7)
    summary_table.add_column("Bar", style="magenta", width=20);
    
    model_display_names = {
        # Legacy models
        "constant_vol": "Constant Volatility",
        "mean_reverting_vol": "Mean-Reverting Volatility",
        "regime_vol": "Regime Volatility",
        "regime_skew_vol": "Regime + Skew Volatility",
        "variance_swap_anchor": "Variance Swap Anchor",
        # Momentum models - full names
        "option_momentum_gaussian": "Momentum Gaussian",
        "option_momentum_phi_gaussian": "Momentum φ-Gaussian",
        "option_momentum_phi_student_t_nu_4": "Momentum φ-Student-t (ν=4)",
        "option_momentum_phi_student_t_nu_6": "Momentum φ-Student-t (ν=6)",
        "option_momentum_phi_student_t_nu_8": "Momentum φ-Student-t (ν=8)",
        "option_momentum_phi_student_t_nu_12": "Momentum φ-Student-t (ν=12)",
        "option_momentum_phi_student_t_nu_20": "Momentum φ-Student-t (ν=20)",
    }
    
    # Sort by average weight (show all models)
    sorted_models = sorted(
        [(m, model_counts.get(m, 0), avg_weights.get(m, 0)) for m in avg_weights.keys()], 
        key=lambda x: (-x[2], -x[1])  # Sort by avg weight, then by wins
    )
    
    for model, count, avg_w in sorted_models:
        pct = avg_w * 100
        bar_len = int(pct / 5)  # Max 20 chars
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # Style momentum models differently
        if model.startswith("option_momentum"):
            style = "bright_cyan"
            marker = "●"
        else:
            style = "white"
            marker = "○"
        
        summary_table.add_row(
            f"[{style}]{marker} {model_display_names.get(model, model[:16])}[/]",
            str(count) if count > 0 else "[dim]—[/]",
            f"{pct:.1f}%",
            bar,
        )
    
    console.print(Panel(
        summary_table,
        title="[bold]Volatility Model Competition[/]",
        subtitle=f"[dim]{len(avg_weights)} models competing  ·  {total_with_results} tickers[/]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    ))
    
    # Show momentum vs legacy summary
    mom_pct = momentum_selected / total_with_results * 100 if total_with_results > 0 else 0
    console.print(f"  [bright_cyan]● Momentum models:[/] {momentum_selected} wins ({mom_pct:.1f}%)  ·  [dim]○ Legacy models:[/] {legacy_selected} wins")
    
    console.print()
    
    # Display cache directory info
    console.print(f"  [dim]Cache directory:[/] [bold]{OPTION_TUNE_CACHE_DIR}[/]")
    console.print(f"  [dim]Total tickers tuned:[/] [bold]{len(tune_cache)}[/]")
    console.print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tune volatility models for high conviction options"
    )
    parser.add_argument("--force", action="store_true", help="Force re-tuning all tickers")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't process")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    
    args = parser.parse_args()
    
    result = run_options_tuning_pipeline(
        force_retune=args.force,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )
