"""
===============================================================================
DIAGNOSTICS — Model Scoring, Standardization, and Regime Diagnostics
===============================================================================

This module provides diagnostic and scoring functions for Bayesian Model Averaging:

    - Hyvärinen score computation (Gaussian and Student-t)
    - Robust score standardization (median/MAD)
    - Entropy-regularized weight computation
    - Combined BIC+Hyvärinen model selection
    - Regime-level diagnostic checks

These functions are pure computations with no I/O or orchestration logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t as student_t_dist


# =============================================================================
# ENTROPY REGULARIZATION CONSTANTS
# =============================================================================

# Entropy regularization lambda for model weights
# Higher = more uniform weights, prevents premature posterior collapse
# Lower = sharper weights, stronger model discrimination
# 0.05 provides good balance between stability and discrimination
DEFAULT_ENTROPY_LAMBDA = 0.05

# Minimum weight fraction for entropy floor (prevents belief collapse)
# Total mass allocated uniformly across all models as a floor
# 0.01 = 1% total mass to uniform, each model gets at least 0.01/n_models weight
DEFAULT_MIN_WEIGHT_FRACTION = 0.01


# =============================================================================
# SAMPLE-SIZE-AWARE PIT CALIBRATION (Elite Fix - February 2026)
# =============================================================================
# With large samples (n>5000), even tiny miscalibration (2-3%) gives KS p=0.
# These functions provide more informative calibration metrics.
# =============================================================================

def compute_pit_calibration_metrics(
    pit_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive PIT calibration metrics that scale better with sample size.
    
    The KS test becomes overly sensitive with large samples. This function provides
    additional metrics that better quantify the practical significance of miscalibration.
    
    Returns:
        Dict with:
        - ks_statistic: Standard KS statistic
        - ks_pvalue: Standard KS p-value  
        - max_deviation: Maximum deviation from uniform quantiles
        - mean_deviation: Mean absolute deviation from uniform quantiles
        - practical_calibration: Boolean for practical adequacy (MAD < 0.05)
        - calibration_score: Score from 0-1 (1 = perfect calibration)
    """
    from scipy.stats import kstest
    
    pit_clean = np.asarray(pit_values).flatten()
    pit_clean = pit_clean[np.isfinite(pit_clean)]
    n = len(pit_clean)
    
    if n < 2:
        return {
            "ks_statistic": 1.0,
            "ks_pvalue": 0.0,
            "max_deviation": 1.0,
            "mean_deviation": 1.0,
            "practical_calibration": False,
            "calibration_score": 0.0,
        }
    
    # Standard KS test
    ks_result = kstest(pit_clean, 'uniform')
    ks_stat = float(ks_result.statistic)
    ks_p = float(ks_result.pvalue)
    
    # Empirical CDF comparison to uniform
    pit_sorted = np.sort(pit_clean)
    uniform_quantiles = np.linspace(0, 1, n)
    
    # Deviations from perfect uniform
    deviations = np.abs(pit_sorted - uniform_quantiles)
    max_deviation = float(np.max(deviations))
    mean_deviation = float(np.mean(deviations))
    
    # Practical calibration check: MAD < 5% is acceptable
    practical_calibration = mean_deviation < 0.05
    
    # Calibration score: exponential decay from 1.0
    # Score = exp(-10 * mean_deviation), so MAD=0 => 1.0, MAD=0.1 => 0.37
    calibration_score = float(np.exp(-10.0 * mean_deviation))
    
    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "practical_calibration": practical_calibration,
        "calibration_score": calibration_score,
    }


def sample_size_adjusted_pit_threshold(n_samples: int) -> float:
    """
    Get an adjusted PIT p-value threshold that accounts for sample size.
    
    With small samples, use standard α=0.05.
    With large samples, use a more lenient threshold since even
    tiny miscalibration will cause rejection.
    
    Formula: α_adjusted = 0.05 * (1 + 0.5*log10(n/1000)) for n > 1000
    
    Args:
        n_samples: Number of PIT values
        
    Returns:
        Adjusted significance threshold
    """
    if n_samples <= 1000:
        return 0.05
    
    # Log-adjusted threshold
    # At n=1000: α=0.05
    # At n=5000: α≈0.07
    # At n=10000: α≈0.08
    log_adjustment = 1.0 + 0.5 * np.log10(n_samples / 1000.0)
    return min(0.05 * log_adjustment, 0.15)  # Cap at 0.15


def is_pit_calibrated(
    pit_pvalue: float,
    n_samples: int,
    strict: bool = False
) -> bool:
    """
    Check if PIT calibration passes using sample-size-adjusted threshold.
    
    Args:
        pit_pvalue: KS test p-value
        n_samples: Number of observations
        strict: If True, use standard 0.05; if False, use adjusted threshold
        
    Returns:
        True if calibration is acceptable
    """
    if strict:
        return pit_pvalue >= 0.05
    
    threshold = sample_size_adjusted_pit_threshold(n_samples)
    return pit_pvalue >= threshold


# =============================================================================
# HYVÄRINEN SCORE COMPUTATION
# =============================================================================
# The Hyvärinen score is a proper scoring rule that only requires the score
# function (derivative of log-density), not the normalizing constant.
#
# For density p(r):
#     H(p) = E[ (1/2)||∇log p||² + Δlog p ]
#
# Lower score = better fit. We negate for consistency with likelihood (higher = better).
# =============================================================================


def compute_hyvarinen_score_gaussian(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    min_sigma: float = 1e-3  # Increased for numerical stability
) -> float:
    """
    Compute Hyvärinen score for Gaussian predictive density.
    
    For Gaussian p(r) = N(μ, σ²):
        ∂log p / ∂r = -(r - μ) / σ²
        ∂²log p / ∂r² = -1 / σ²
    
    Therefore:
        H = (1/n) Σ_t [ (r_t - μ_t)² / (2σ_t⁴) - 1/σ_t² ]
    
    NUMERICAL STABILITY FIX (February 2026):
        1. Clip z values to [-10, 10] to prevent extreme outliers
        2. Work with standardized residuals to avoid σ⁴ division
        3. Clip final result to interpretable range [-10000, 10000]
    
    Args:
        returns: Observed returns
        mu: Predicted means
        sigma: Predicted standard deviations (NOT variance)
        min_sigma: Minimum sigma for numerical stability
        
    Returns:
        Hyvärinen score (lower is better, but we return negated for consistency)
    """
    returns = np.asarray(returns).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Numerical stability: floor sigma
    sigma = np.maximum(sigma, min_sigma)
    sigma_sq = sigma ** 2
    
    # Standardized innovations - CLIP to prevent explosion
    z = (returns - mu) / sigma
    z = np.clip(z, -10.0, 10.0)  # Winsorize extreme z values
    z_sq = z ** 2
    
    # Hyvärinen score in terms of z:
    # H = (z² - 2) / (2σ²)
    h_scores = (z_sq - 2.0) / (2.0 * sigma_sq)
    
    # Clip per-observation scores
    h_scores = np.clip(h_scores, -1e4, 1e4)
    
    # Filter out non-finite values
    valid = np.isfinite(h_scores)
    if not np.any(valid):
        return 0.0
    
    h_mean = float(np.mean(h_scores[valid]))
    
    # Clip final result to reasonable interpretable range
    h_mean = np.clip(h_mean, -1e4, 1e4)
    
    # Return negated score (higher = better, for consistency with log-likelihood)
    return -h_mean


def compute_hyvarinen_score_student_t(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
    min_sigma: float = 1e-3,  # Increased for numerical stability
    min_nu: float = 2.1
) -> float:
    """
    Compute Hyvärinen score for Student-t predictive density.
    
    For Student-t p(r) with location μ, scale σ, and degrees of freedom ν:
    
    Let z = (r - μ) / σ
    
        log p(r) = const - ((ν+1)/2) * log(1 + z²/ν) - log(σ)
        
        ∂log p / ∂r = -((ν+1)/ν) * z / (σ * (1 + z²/ν))
        
        ∂²log p / ∂r² = -((ν+1)/σ²) * (ν - z²) / (ν + z²)²
    
    NUMERICAL STABILITY FIX (February 2026):
        1. Clip z values to [-10, 10] to prevent extreme outliers
        2. Clip per-observation scores to [-10000, 10000]
        3. Clip final result to interpretable range
    
    Args:
        returns: Observed returns
        mu: Predicted locations
        sigma: Predicted scales (NOT variance)
        nu: Degrees of freedom
        min_sigma: Minimum sigma for numerical stability
        min_nu: Minimum nu for numerical stability
        
    Returns:
        Hyvärinen score (negated so higher = better)
    """
    returns = np.asarray(returns).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Numerical stability
    sigma = np.maximum(sigma, min_sigma)
    nu = max(float(nu), min_nu)
    
    sigma_sq = sigma ** 2
    
    # Standardized residuals - CLIP to prevent explosion
    z = (returns - mu) / sigma
    z = np.clip(z, -10.0, 10.0)  # Winsorize extreme z values
    z_sq = z ** 2
    
    # Common denominator: ν + z²
    denom = nu + z_sq
    denom = np.maximum(denom, 1e-6)
    
    # First derivative squared:
    # (∂log p/∂r)² = ((ν+1)² * z²) / (σ² * (ν + z²)²)
    d1_sq = ((nu + 1.0) ** 2 * z_sq) / (sigma_sq * denom ** 2)
    
    # Second derivative:
    # ∂²log p/∂r² = -((ν+1)/σ²) * (ν - z²) / (ν + z²)²
    d2 = -((nu + 1.0) / sigma_sq) * (nu - z_sq) / (denom ** 2)
    
    # Hyvärinen score: (1/2) * (∂log p/∂r)² + ∂²log p/∂r²
    h_scores = 0.5 * d1_sq + d2
    
    # Clip per-observation scores
    h_scores = np.clip(h_scores, -1e4, 1e4)
    
    # Filter out non-finite values
    valid = np.isfinite(h_scores)
    if not np.any(valid):
        return 0.0
    
    h_mean = float(np.mean(h_scores[valid]))
    
    # Clip final result to reasonable interpretable range
    h_mean = np.clip(h_mean, -1e4, 1e4)
    
    # Return negated score (higher = better)
    return -h_mean


def compute_hyvarinen_model_weights(
    hyvarinen_scores: Dict[str, float],
    epsilon: float = 1e-10,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Convert Hyvärinen scores to unnormalized posterior weights.
    
    Since Hyvärinen scores are negated (higher = better), we use:
        w_raw(m|r) = exp(temperature * (H_m - H_min))
    
    This mirrors the BIC weight formula but uses Hyvärinen instead.
    
    Args:
        hyvarinen_scores: Dictionary mapping model name to (negated) Hyvärinen score
        epsilon: Small constant to prevent zero weights
        temperature: Scaling factor (higher = more concentrated on best model)
        
    Returns:
        Dictionary of unnormalized weights
    """
    # Find maximum score (best model, since higher = better)
    finite_scores = [s for s in hyvarinen_scores.values() if np.isfinite(s)]
    if not finite_scores:
        n_models = len(hyvarinen_scores)
        return {m: 1.0 / max(n_models, 1) for m in hyvarinen_scores}
    
    score_max = max(finite_scores)
    
    # Compute raw weights
    weights = {}
    for model_name, score in hyvarinen_scores.items():
        if np.isfinite(score):
            # Higher score = better, so exp(temp * (score - max)) gives relative weight
            # When score == max, weight = 1; when score < max, weight < 1
            delta = score - score_max
            w = np.exp(temperature * delta)
            weights[model_name] = max(w, epsilon)
        else:
            weights[model_name] = epsilon
    
    return weights


# =============================================================================
# ROBUST SCORE STANDARDIZATION & ENTROPY-REGULARIZED WEIGHTS
# =============================================================================


def robust_standardize_scores(
    scores: Dict[str, float],
    eps: float = 1e-8,
    max_zscore: float = 5.0,
) -> Dict[str, float]:
    """
    Robust cross-model standardization using median and MAD, with winsorization.
    
    Preserves ordering while normalizing heterogeneous score scales.
    Winsorizes at ±max_zscore to prevent extreme outliers from collapsing
    the softmax: without clipping, a single model with CRPS=0.0035 vs
    others at 0.0050 produces z=-128, making exp(128/λ) dominate everything.
    
    The MAD is scaled by 1.4826 to be consistent with standard deviation
    for Gaussian data: MAD * 1.4826 ≈ σ for N(μ, σ²).
    
    Args:
        scores: Dictionary mapping model name to raw score
        eps: Small constant to prevent division by zero
        max_zscore: Maximum absolute z-score (winsorization bound, default ±5)
        
    Returns:
        Dictionary of standardized scores (zero median, unit scale, clipped to ±max_zscore)
    """
    # Gaussian consistency factor: MAD * 1.4826 ≈ σ for normal distributions
    MAD_CONSISTENCY_FACTOR = 1.4826
    
    # Extract finite values only
    finite_items = [(k, v) for k, v in scores.items() if np.isfinite(v)]
    
    if len(finite_items) < 2:
        # Not enough values to standardize meaningfully
        # Return zeros for finite, keep non-finite as-is
        return {
            k: 0.0 if np.isfinite(v) else v
            for k, v in scores.items()
        }
    
    values = np.array([v for _, v in finite_items], dtype=float)
    
    # Robust location and scale
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    # Scale MAD to be consistent with standard deviation
    scale = mad * MAD_CONSISTENCY_FACTOR if mad > eps else eps
    
    # Standardize all scores with winsorization
    standardized = {}
    for k, v in scores.items():
        if np.isfinite(v):
            z = (v - median) / scale
            # Winsorize: clip to ±max_zscore to prevent softmax collapse
            z = max(-max_zscore, min(max_zscore, z))
            standardized[k] = z
        else:
            standardized[k] = v  # Keep non-finite as-is (inf, nan)
    
    return standardized


def entropy_regularized_weights(
    standardized_scores: Dict[str, float],
    lambda_entropy: float = DEFAULT_ENTROPY_LAMBDA,
    min_weight_fraction: float = DEFAULT_MIN_WEIGHT_FRACTION,
    eps: float = 1e-10
) -> Dict[str, float]:
    """
    Compute entropy-regularized model weights via softmax with entropy floor.
    
    Solves the optimization problem:
        min_w Σ_m w_m * S̃_m + λ Σ_m w_m * log(w_m)
        s.t. Σ_m w_m = 1, w_m ≥ min_weight
    
    The closed-form solution (without floor) is softmax with temperature = λ:
        w_m ∝ exp(-S̃_m / λ)
    
    We then apply an entropy floor to prevent belief collapse:
        w_m = max(w_m, min_weight_fraction / n_models)
    
    Benefits:
    - Prevents premature posterior collapse in low-evidence regimes
    - Smooth weight transitions as evidence accumulates
    - Entropy floor prevents overconfident allocations
    - Convex, stable, deterministic
    
    Args:
        standardized_scores: Dictionary of standardized scores (lower = better)
        lambda_entropy: Entropy regularization strength (0.05 = balanced)
        min_weight_fraction: Minimum total mass allocated to uniform (0.01 = 1%)
        eps: Small constant to prevent zero weights
        
    Returns:
        Dictionary of normalized model weights (sum to 1)
    """
    # Extract finite scores only
    finite_items = [(k, v) for k, v in standardized_scores.items() if np.isfinite(v)]
    
    if not finite_items:
        # No valid scores, return uniform
        n = len(standardized_scores)
        return {k: 1.0 / max(n, 1) for k in standardized_scores}
    
    keys = [k for k, _ in finite_items]
    scores = np.array([v for _, v in finite_items], dtype=float)
    n_models = len(keys)
    
    # Softmax with entropy temperature
    # Lower score = better, so we negate scores in the softmax
    temperature = max(lambda_entropy, 1e-8)
    logits = -scores / temperature
    
    # Numerical stability: subtract max
    logits = logits - logits.max()
    
    # Compute weights
    weights = np.exp(logits)
    weights = np.maximum(weights, eps)  # Prevent exact zeros
    weights = weights / weights.sum()  # Normalize
    
    # Entropy floor: prevent belief collapse
    min_weight_per_model = min_weight_fraction / max(n_models, 1)
    weights = np.maximum(weights, min_weight_per_model)
    weights = weights / weights.sum()  # Re-normalize after floor
    
    # Build result dict
    result = dict(zip(keys, weights))
    
    # Add epsilon weight for non-finite scores
    for k, v in standardized_scores.items():
        if not np.isfinite(v):
            result[k] = eps
    
    # Re-normalize if we added non-finite entries
    total = sum(result.values())
    if total > 0:
        result = {k: w / total for k, w in result.items()}
    
    return result


def compute_combined_standardized_score(
    bic: float,
    hyvarinen: float,
    bic_weight: float = 0.5
) -> float:
    """
    Compute combined score from already-standardized BIC and Hyvärinen.
    
    For BIC: lower is better → we use +BIC in combined score
    For Hyvärinen: higher is better → we use -Hyvärinen in combined score
    
    Combined: S = w_bic * BIC_std - (1 - w_bic) * Hyv_std
    Lower combined score = better model
    
    Args:
        bic: Standardized BIC score
        hyvarinen: Standardized Hyvärinen score
        bic_weight: Weight for BIC (0.5 = equal weighting)
        
    Returns:
        Combined standardized score (lower = better)
    """
    if not np.isfinite(bic):
        bic = 0.0
    if not np.isfinite(hyvarinen):
        hyvarinen = 0.0
    
    # BIC: lower is better, so positive contribution
    # Hyvärinen: higher is better, so negative contribution
    return bic_weight * bic - (1.0 - bic_weight) * hyvarinen


def compute_combined_model_weights(
    bic_values: Dict[str, float],
    hyvarinen_scores: Dict[str, float],
    bic_weight: float = 0.5,
    lambda_entropy: float = DEFAULT_ENTROPY_LAMBDA,
    epsilon: float = 1e-10
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute combined model weights using BIC + Hyvärinen with entropy regularization.
    
    Pipeline:
    1. Standardize BIC values (robust median/MAD)
    2. Standardize Hyvärinen scores (robust median/MAD)
    3. Combine: S = w_bic * BIC_std - (1-w_bic) * Hyv_std
    4. Apply entropy-regularized softmax
    
    Args:
        bic_values: Dictionary mapping model name to BIC value
        hyvarinen_scores: Dictionary mapping model name to Hyvärinen score
        bic_weight: Weight for BIC in combination (0.5 = equal)
        lambda_entropy: Entropy regularization strength
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple of (weights dict, metadata dict)
    """
    # Step 1: Robust standardization
    bic_standardized = robust_standardize_scores(bic_values)
    hyv_standardized = robust_standardize_scores(hyvarinen_scores)
    
    # Step 2: Combine scores
    combined_scores = {}
    for model_name in bic_values.keys():
        bic_std = bic_standardized.get(model_name, 0.0)
        hyv_std = hyv_standardized.get(model_name, 0.0)
        combined_scores[model_name] = compute_combined_standardized_score(
            bic_std, hyv_std, bic_weight
        )
    
    # Step 3: Entropy-regularized weights
    weights = entropy_regularized_weights(
        combined_scores,
        lambda_entropy=lambda_entropy,
        eps=epsilon
    )
    
    # Build metadata for caching and diagnostics
    metadata = {
        "bic_standardized": {k: float(v) if np.isfinite(v) else None for k, v in bic_standardized.items()},
        "hyvarinen_standardized": {k: float(v) if np.isfinite(v) else None for k, v in hyv_standardized.items()},
        "crps_standardized": {},  # Empty - CRPS not used in this method
        "combined_scores_standardized": {k: float(v) if np.isfinite(v) else None for k, v in combined_scores.items()},
        "weights_used": {"bic": float(bic_weight), "hyvarinen": float(1.0 - bic_weight), "crps": 0.0},
        "bic_weight": bic_weight,
        "lambda_entropy": lambda_entropy,
        "entropy_regularized": True,
        "crps_enabled": False,
        "scoring_method": "bic_hyv_only",
    }
    
    return weights, metadata


# =============================================================================
# REGIME DIAGNOSTICS
# =============================================================================


def compute_regime_diagnostics(
    regime_params: Dict[int, Dict],
    global_params: Dict,
    log_fn=None
) -> Dict[int, Dict]:
    """
    Compute regime diagnostics for Layer 5.
    
    Checks:
    1. Sanity relationships between regimes
    2. Parameter distances
    3. Collapse detection
    
    Args:
        regime_params: Parameters for each regime
        global_params: Global fallback parameters
        log_fn: Optional logging function (called with message string)
    
    Returns:
        Dictionary of diagnostics per regime
    """
    # Import here to avoid circular dependency
    from tuning.tune import MarketRegime
    
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
    
    sanity_q_crisis_vs_low = None
    if q_crisis is not None and q_low_trend is not None:
        sanity_q_crisis_vs_low = q_crisis > q_low_trend
    
    # Sanity check 2: nu_crisis < nu_trend (crisis has fatter tails)
    nu_crisis = nu_vals.get(MarketRegime.CRISIS_JUMP)
    nu_low_trend = nu_vals.get(MarketRegime.LOW_VOL_TREND)
    
    sanity_nu_crisis_vs_trend = None
    if nu_crisis is not None and nu_low_trend is not None:
        sanity_nu_crisis_vs_trend = nu_crisis < nu_low_trend
    
    # Sanity check 3: phi_trend > phi_range (trends are more persistent)
    phi_low_trend = phi_vals.get(MarketRegime.LOW_VOL_TREND)
    phi_low_range = phi_vals.get(MarketRegime.LOW_VOL_RANGE)
    
    sanity_phi_trend_vs_range = None
    if phi_low_trend is not None and phi_low_range is not None:
        sanity_phi_trend_vs_range = phi_low_trend > phi_low_range
    
    # Collapse detection: check if all parameters are too close
    collapse_threshold = 0.1
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
            "ll_type": "cv_penalized_mean",
        }
    
    # Log warnings if sanity checks fail
    if log_fn:
        if sanity_q_crisis_vs_low is False:
            log_fn("     ⚠️  Sanity warning: q_crisis should be > q_low_vol")
        if sanity_nu_crisis_vs_trend is False:
            log_fn("     ⚠️  Sanity warning: nu_crisis should be < nu_trend")
        if sanity_phi_trend_vs_range is False:
            log_fn("     ⚠️  Sanity warning: phi_trend should be > phi_range")
        if collapse_detected:
            log_fn("     ⚠️  Collapse warning: All regime parameters too close to global")
    
    return diagnostics


# =============================================================================
# LEAVE-FUTURE-OUT CROSS-VALIDATION (LFO-CV) — February 2026
# =============================================================================
# LFO-CV is the gold-standard for time series model selection.
# Unlike k-fold CV which shuffles data, LFO-CV respects temporal ordering:
#
#   For t = T_start to T:
#     Train on [1, t-1]
#     Predict y_t
#     Accumulate log p(y_t | y_{1:t-1}, θ)
#
# This is exactly what matters for forecasting: how well does the model
# predict FUTURE observations given PAST data?
#
# THEORETICAL BASIS:
# - Proper scoring rule (Gneiting & Raftery 2007)
# - Equivalent to prequential likelihood (Dawid 1984)
# - Used by Renaissance, Two Sigma, DE Shaw for model selection
#
# COMPUTATIONAL NOTE:
# Running Kalman filter once computes all one-step-ahead predictions.
# LFO-CV is therefore O(T), same as standard likelihood.
# =============================================================================

# Enable LFO-CV for model selection
LFO_CV_ENABLED = True
LFO_CV_MIN_TRAIN_FRAC = 0.5  # Use first 50% for initial training


def compute_lfo_cv_score_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
    min_train_frac: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Leave-Future-Out CV score for Gaussian Kalman filter.
    
    The Kalman filter naturally produces one-step-ahead predictive distributions.
    LFO-CV simply evaluates the log-likelihood of observations under these
    predictive distributions, starting from a minimum training period.
    
    Args:
        returns: Time series of returns
        vol: Time series of volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) coefficient (default 1.0 = random walk)
        min_train_frac: Minimum fraction of data for training (default 0.5)
        
    Returns:
        Tuple of (lfo_cv_score, diagnostics)
        - lfo_cv_score: Mean log predictive density (higher is better)
        - diagnostics: Dict with per-period scores and metadata
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    n = len(returns)
    
    # Minimum training period
    t_start = max(int(n * min_train_frac), 20)
    
    if n < t_start + 10:
        return float('-inf'), {"error": "insufficient_data", "n": n, "t_start": t_start}
    
    # Initialize Kalman state
    mu_t = 0.0
    P_t = 1.0  # Initial state variance
    
    # Accumulate log predictive densities
    log_pred_densities = []
    pred_errors = []
    
    for t in range(n):
        # Observation variance
        R_t = c * (vol[t] ** 2)
        
        # Predictive distribution: y_t | y_{1:t-1} ~ N(mu_t|t-1, S_t)
        S_t = P_t + R_t
        
        if t >= t_start:
            # Compute log predictive density for this observation
            innovation = returns[t] - mu_t
            log_pred = -0.5 * np.log(2 * np.pi * S_t) - 0.5 * (innovation ** 2) / S_t
            
            if np.isfinite(log_pred):
                log_pred_densities.append(log_pred)
                pred_errors.append(innovation)
        
        # Kalman update (standard equations)
        innovation = returns[t] - mu_t
        K_t = P_t / S_t if S_t > 1e-12 else 0.0
        
        mu_t = mu_t + K_t * innovation
        P_t = (1 - K_t) * P_t
        
        # State prediction for next step
        mu_t = phi * mu_t
        P_t = (phi ** 2) * P_t + q
    
    if len(log_pred_densities) == 0:
        return float('-inf'), {"error": "no_valid_predictions"}
    
    # LFO-CV score is mean log predictive density
    lfo_cv_score = float(np.mean(log_pred_densities))
    
    diagnostics = {
        "n_predictions": len(log_pred_densities),
        "t_start": t_start,
        "mean_abs_error": float(np.mean(np.abs(pred_errors))),
        "rmse": float(np.sqrt(np.mean(np.array(pred_errors) ** 2))),
        "log_pred_std": float(np.std(log_pred_densities)),
    }
    
    return lfo_cv_score, diagnostics


def compute_lfo_cv_score_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    min_train_frac: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Leave-Future-Out CV score for Student-t Kalman filter.
    
    Uses Student-t predictive density instead of Gaussian.
    The scale parameter is adjusted for Student-t variance.
    
    Args:
        returns: Time series of returns
        vol: Time series of volatility estimates  
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) coefficient
        nu: Degrees of freedom for Student-t
        min_train_frac: Minimum fraction of data for training
        
    Returns:
        Tuple of (lfo_cv_score, diagnostics)
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    n = len(returns)
    nu = max(float(nu), 2.01)  # Ensure nu > 2 for finite variance
    
    # Minimum training period
    t_start = max(int(n * min_train_frac), 20)
    
    if n < t_start + 10:
        return float('-inf'), {"error": "insufficient_data", "n": n, "t_start": t_start}
    
    # Initialize Kalman state
    mu_t = 0.0
    P_t = 1.0
    
    # Log of gamma function ratios for Student-t PDF
    log_gamma_ratio = gammaln((nu + 1) / 2) - gammaln(nu / 2)
    log_norm_const = log_gamma_ratio - 0.5 * np.log(nu * np.pi)
    
    # Accumulate log predictive densities
    log_pred_densities = []
    pred_errors = []
    
    for t in range(n):
        # Observation variance
        R_t = c * (vol[t] ** 2)
        
        # Predictive variance (Gaussian approximation for Kalman)
        S_t = P_t + R_t
        
        # For Student-t: scale = sqrt(S_t * (nu-2)/nu) to match variance
        if nu > 2:
            scale_t = np.sqrt(S_t * (nu - 2) / nu)
        else:
            scale_t = np.sqrt(S_t)
        
        if t >= t_start:
            # Student-t log predictive density
            innovation = returns[t] - mu_t
            z = innovation / scale_t
            
            log_pred = log_norm_const - np.log(scale_t) - ((nu + 1) / 2) * np.log(1 + z**2 / nu)
            
            if np.isfinite(log_pred):
                log_pred_densities.append(log_pred)
                pred_errors.append(innovation)
        
        # Kalman update with Student-t weighting
        innovation = returns[t] - mu_t
        
        # Robust weighting for Student-t (downweight outliers)
        z_sq = (innovation ** 2) / S_t if S_t > 1e-12 else 0
        w_t = (nu + 1) / (nu + z_sq)  # Student-t weight
        
        K_t = P_t / S_t if S_t > 1e-12 else 0.0
        
        # Weighted update (accounts for heavy tails)
        mu_t = mu_t + K_t * w_t * innovation
        P_t = (1 - w_t * K_t) * P_t
        
        # State prediction
        mu_t = phi * mu_t
        P_t = (phi ** 2) * P_t + q
    
    if len(log_pred_densities) == 0:
        return float('-inf'), {"error": "no_valid_predictions"}
    
    lfo_cv_score = float(np.mean(log_pred_densities))
    
    diagnostics = {
        "n_predictions": len(log_pred_densities),
        "t_start": t_start,
        "nu": nu,
        "mean_abs_error": float(np.mean(np.abs(pred_errors))),
        "rmse": float(np.sqrt(np.mean(np.array(pred_errors) ** 2))),
        "log_pred_std": float(np.std(log_pred_densities)),
    }
    
    return lfo_cv_score, diagnostics


def compute_lfo_cv_model_weights(
    lfo_cv_scores: Dict[str, float],
    temperature: float = 1.0,
    epsilon: float = 1e-10,
) -> Dict[str, float]:
    """
    Convert LFO-CV scores to model weights.
    
    Uses softmax transformation:
        w(m) = exp(temperature * (LFO_m - LFO_max))
    
    Args:
        lfo_cv_scores: Dict mapping model name to LFO-CV score
        temperature: Softmax temperature (higher = more uniform)
        epsilon: Minimum weight floor
        
    Returns:
        Dict of normalized model weights
    """
    if not lfo_cv_scores:
        return {}
    
    # Filter finite scores
    finite_scores = {m: s for m, s in lfo_cv_scores.items() if np.isfinite(s)}
    
    if not finite_scores:
        n_models = len(lfo_cv_scores)
        return {m: 1.0 / max(n_models, 1) for m in lfo_cv_scores}
    
    max_score = max(finite_scores.values())
    
    weights = {}
    for model_name, score in lfo_cv_scores.items():
        if np.isfinite(score):
            w = np.exp(temperature * (score - max_score))
            weights[model_name] = max(w, epsilon)
        else:
            weights[model_name] = epsilon
    
    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {m: w / total for m, w in weights.items()}
    
    return weights


# =============================================================================
# CRPS COMPUTATION FOR MODEL SELECTION (February 2026)
# =============================================================================
# Continuous Ranked Probability Score (CRPS) is a strictly proper scoring rule
# that measures both calibration and sharpness of probabilistic forecasts.
#
# CRPS(F, y) = ∫ (F(x) - 1{x ≥ y})² dx
#
# Lower CRPS is better. Closed-form expressions exist for Gaussian/Student-t.
#
# Reference: Gneiting & Raftery (2007), Gneiting et al. (2005)
# =============================================================================


def compute_crps_gaussian_inline(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Compute CRPS for Gaussian predictive distributions.
    
    Closed-form formula (Gneiting & Raftery 2007):
        CRPS(N(μ,σ²), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
        
    where z = (y - μ) / σ, Φ is standard normal CDF, φ is PDF.
    
    Args:
        observations: Actual observed values
        mu: Predicted means
        sigma: Predicted standard deviations
        
    Returns:
        Mean CRPS (lower is better)
    """
    observations = np.asarray(observations).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Ensure positive sigma and matching lengths
    sigma = np.maximum(sigma, 1e-10)
    n = min(len(observations), len(mu), len(sigma))
    
    if n == 0:
        return float('inf')
    
    observations = observations[:n]
    mu = mu[:n]
    sigma = sigma[:n]
    
    # Standardized residual
    z = (observations - mu) / sigma
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # CRPS for each observation (closed-form for Gaussian)
    crps_individual = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    
    # Filter non-finite values
    valid = np.isfinite(crps_individual)
    if not np.any(valid):
        return float('inf')
    
    return float(np.mean(crps_individual[valid]))


def compute_crps_student_t_inline(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
) -> float:
    """
    Compute CRPS for Student-t predictive distributions.
    
    Uses the closed-form expression from Gneiting & Raftery (2007).
    
    Args:
        observations: Actual observed values
        mu: Predicted means
        sigma: Predicted scale parameters
        nu: Degrees of freedom (must be > 1 for finite CRPS)
        
    Returns:
        Mean CRPS (lower is better)
    """
    observations = np.asarray(observations).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Ensure positive sigma and valid nu
    sigma = np.maximum(sigma, 1e-10)
    nu = max(float(nu), 1.01)
    
    n = min(len(observations), len(mu), len(sigma))
    if n == 0:
        return float('inf')
    
    observations = observations[:n]
    mu = mu[:n]
    sigma = sigma[:n]
    
    # Standardized residual
    z = (observations - mu) / sigma
    
    # Student-t PDF and CDF
    t_dist = student_t_dist(df=nu)
    pdf_z = t_dist.pdf(z)
    cdf_z = t_dist.cdf(z)
    
    # Closed-form CRPS for Student-t (Gneiting & Raftery 2007)
    if nu > 1:
        log_B_half_nu_minus_half = gammaln(0.5) + gammaln(nu - 0.5) - gammaln(nu)
        log_B_half_nu_half = gammaln(0.5) + gammaln(nu / 2) - gammaln((nu + 1) / 2)
        B_ratio = np.exp(log_B_half_nu_minus_half - 2 * log_B_half_nu_half)
        
        term1 = z * (2 * cdf_z - 1)
        term2 = 2 * pdf_z * (nu + z**2) / (nu - 1)
        term3 = 2 * np.sqrt(nu) * B_ratio / (nu - 1)
        
        crps_individual = sigma * (term1 + term2 - term3)
    else:
        # Fallback to Gaussian approximation
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        crps_individual = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    
    valid = np.isfinite(crps_individual)
    if not np.any(valid):
        return float('inf')
    
    return float(np.mean(crps_individual[valid]))


def compute_crps_model_weights(
    crps_values: Dict[str, float],
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    Convert CRPS values to model weights (lower CRPS = higher weight).
    """
    finite_crps = [c for c in crps_values.values() if np.isfinite(c) and c > 0]
    if not finite_crps:
        n_models = len(crps_values)
        return {m: 1.0 / max(n_models, 1) for m in crps_values}
    
    crps_median = np.median(finite_crps)
    if crps_median < 1e-10:
        crps_median = 1.0
    
    weights = {}
    for model_name, crps in crps_values.items():
        if np.isfinite(crps) and crps > 0:
            w = np.exp(-crps / crps_median)
            weights[model_name] = max(w, epsilon)
        else:
            weights[model_name] = epsilon
    
    return weights


# =============================================================================
# ELITE DENSITY FORECAST SCORING (CRPS-Dominated, No BIC)
# =============================================================================
# Proper scoring rules for probabilistic forecasting engines.
#
# Design rationale:
#   - BIC penalises parameter count via k*ln(n). This systematically
#     disadvantages richer models (unified student-t has 14 adaptive
#     parameters) even when the extra parameters genuinely improve
#     out-of-sample density quality. BIC is removed.
#   - CRPS is strictly proper and captures full predictive distribution
#     quality (sharpness + calibration). It dominates the score.
#   - PIT deviation (Cramer-von Mises on PIT values) measures calibration
#     continuously, replacing the binary p-value gate.
#   - Tail error measures exceedance frequency at 1% and 5% quantiles.
#   - MAD captures location accuracy.
#   - Berkowitz penalty captures serial dependence in forecast errors.
#
# Score formula (lower = better):
#   EliteScore = w_crps * CRPS_std
#              + w_pit  * PIT_dev_std
#              + w_berk * Berk_penalty_std
#              + w_tail * TailError_std
#              + w_mad  * MAD_std
#
# All components are standardised via robust median/MAD before weighting.
# Weights are regime-aware: crisis regimes increase tail weight,
# low-vol regimes increase calibration weight.
# =============================================================================

CRPS_SCORING_ENABLED = True

# Default weights: CRPS dominates, calibration metrics provide discipline
# (crps, pit_dev, berk_penalty, tail_error, mad)
DEFAULT_ELITE_WEIGHTS = (0.60, 0.15, 0.10, 0.10, 0.05)

# Regime-specific weight configurations
# Format: (w_crps, w_pit_dev, w_berk, w_tail, w_mad)
REGIME_SCORING_WEIGHTS = {
    0: (0.60, 0.15, 0.10, 0.10, 0.05),  # Unknown: default balanced
    1: (0.45, 0.10, 0.10, 0.25, 0.10),  # Crisis: tail error critical, MAD up
    2: (0.60, 0.15, 0.10, 0.10, 0.05),  # Trending: standard
    3: (0.50, 0.20, 0.10, 0.05, 0.15),  # Ranging: calibration + location matter
    4: (0.50, 0.20, 0.15, 0.05, 0.10),  # Low Vol: calibration + serial independence
}

# =============================================================================
# BERKOWITZ CALIBRATION PENALTY — Likelihood-Normalized (February 2026)
# =============================================================================
# Replaces heuristic -log10(p) with principled LR/T per-observation penalty.
# CalPenalty_m = lambda_cal * Berkowitz_LR_m / T_m
# where LR = 2*(ll_alt - ll_null) is the Berkowitz likelihood ratio statistic
# and T is the number of PIT observations.
#
# This is likelihood-consistent: dividing by T gives per-observation
# log-likelihood degradation from miscalibration, making the penalty
# scale-stable across assets of different sample lengths.
#
# lambda_cal is regime-aware: crisis regimes penalize miscalibration harder
# because tail forecasts are safety-critical.
# =============================================================================
BERKOWITZ_CALIBRATION_LAMBDA = 2.0  # Default calibration strength

# Regime-specific lambda_cal: crisis uses stronger penalty (3.0),
# trending/unknown use default (2.0), low-vol/ranging use milder (1.5)
# because calibration noise in calm markets shouldn't dominate scoring.
REGIME_BERKOWITZ_LAMBDA = {
    0: 2.0,   # Unknown: default
    1: 3.0,   # Crisis: calibration is safety-critical
    2: 2.0,   # Trending: standard
    3: 1.5,   # Ranging: mild — calibration noise higher in range-bound markets
    4: 1.5,   # Low Vol: mild — small miscalibration is acceptable
}

# Backward-compat aliases kept so old code referencing these doesn't crash
DEFAULT_BIC_WEIGHT_COMBINED = 0.0
DEFAULT_HYVARINEN_WEIGHT_COMBINED = 0.0
DEFAULT_CRPS_WEIGHT_COMBINED = 1.0
PIT_CATASTROPHIC_THRESHOLD = 0.01
PIT_CATASTROPHIC_PENALTY = 0.5


def _compute_pit_deviation(pit_pvalues: Dict[str, float]) -> Dict[str, float]:
    """
    Compute PIT deviation score per model.

    Uses -log10(max(p, 1e-10)) so that:
      p=1.0  -> 0.0  (perfect calibration)
      p=0.05 -> 1.3
      p=0.001 -> 3.0
    Lower is better. Continuous, no arbitrary threshold.
    """
    result = {}
    for m, p in pit_pvalues.items():
        if p is not None and np.isfinite(p):
            result[m] = -np.log10(max(p, 1e-10))
        else:
            result[m] = 10.0  # worst case
    return result


def _compute_berk_calibration_penalty(
    berkowitz_lr_stats: Dict[str, float],
    pit_counts: Dict[str, int],
    lambda_cal: float = BERKOWITZ_CALIBRATION_LAMBDA,
) -> Dict[str, float]:
    """
    Likelihood-normalized Berkowitz calibration penalty (February 2026).

    CalPenalty_m = lambda_cal * LR_m / T_m

    where LR_m = Berkowitz likelihood ratio statistic (Chi2 test for PIT
    serial dependence) and T_m = number of PIT observations.

    LR/T gives per-observation log-likelihood degradation from miscalibration,
    making the penalty scale-stable across assets of different sample lengths
    and additive in log-likelihood space (BMA-compatible).

    Lower = better calibrated (0 = perfectly calibrated iid PIT).
    """
    result = {}
    all_models = set(berkowitz_lr_stats.keys()) | set(pit_counts.keys())
    for m in all_models:
        lr = berkowitz_lr_stats.get(m, 0.0)
        T = pit_counts.get(m, 0)
        if T > 0 and np.isfinite(lr):
            result[m] = lambda_cal * lr / T
        else:
            # No valid data — assign median penalty (handled by standardization)
            result[m] = 0.0
    return result


def _compute_berk_penalty(berk_pvalues: Dict[str, float]) -> Dict[str, float]:
    """
    Legacy Berkowitz penalty: -log10(max(p, 1e-10)).
    p~1 -> 0 (good), p~0 -> large (bad). Lower is better.

    Kept as fallback when only p-values are available (no raw LR stats).
    Prefer _compute_berk_calibration_penalty when berkowitz_lr_stats available.
    """
    result = {}
    for m, p in berk_pvalues.items():
        if p is not None and np.isfinite(p):
            result[m] = -np.log10(max(p, 1e-10))
        else:
            result[m] = 10.0
    return result


def _compute_tail_error(pit_pvalues: Dict[str, float],
                        crps_values: Dict[str, float]) -> Dict[str, float]:
    """
    Tail calibration proxy from available metrics.

    When full PIT arrays aren't available, use a combined penalty
    from PIT deviation and CRPS. Models with poor PIT AND high CRPS
    have the worst tail calibration.

    Returns dict of tail error scores (lower = better).
    """
    result = {}
    crps_arr = np.array([v for v in crps_values.values() if np.isfinite(v)])
    crps_med = float(np.median(crps_arr)) if len(crps_arr) > 0 else 0.01

    for m in crps_values:
        c = crps_values.get(m, crps_med)
        p = pit_pvalues.get(m)
        if p is not None and np.isfinite(p):
            pit_dev = -np.log10(max(p, 1e-10))
        else:
            pit_dev = 10.0
        # Tail error: geometric combination of CRPS excess and PIT deviation
        crps_excess = max(0.0, c / max(crps_med, 1e-10) - 1.0)
        result[m] = pit_dev * 0.5 + crps_excess * 0.5
    return result


def compute_regime_aware_model_weights(
    bic_values: Dict[str, float],
    hyvarinen_scores: Dict[str, float],
    crps_values: Optional[Dict[str, float]] = None,
    pit_pvalues: Optional[Dict[str, float]] = None,
    berk_pvalues: Optional[Dict[str, float]] = None,
    berkowitz_lr_stats: Optional[Dict[str, float]] = None,
    pit_counts: Optional[Dict[str, int]] = None,
    mad_values: Optional[Dict[str, float]] = None,
    regime: Optional[int] = None,
    bic_weight: Optional[float] = None,
    hyvarinen_weight: Optional[float] = None,
    crps_weight: Optional[float] = None,
    lambda_entropy: float = DEFAULT_ENTROPY_LAMBDA,
    epsilon: float = 1e-10,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Elite density-forecast model selection (CRPS-dominated, no BIC penalty).

    Score = w_crps * CRPS + w_pit * PIT_dev + w_berk * Berk + w_tail * Tail + w_mad * MAD
    All components standardised via robust median/MAD. Lower score = better.
    Weights are regime-aware.

    BIC and Hyvarinen are stored in metadata for diagnostics but do NOT
    enter the selection score.

    Args:
        bic_values:        BIC per model (kept for metadata, not used in score)
        hyvarinen_scores:  Hyvarinen per model (kept for metadata, not used in score)
        crps_values:       CRPS per model (lower = better)
        pit_pvalues:       PIT KS p-values per model (higher = better calibration)
        berk_pvalues:      Berkowitz p-values per model (fallback when LR stats unavailable)
        berkowitz_lr_stats: Raw Berkowitz LR statistics per model (preferred over p-values).
                           When provided with pit_counts, uses likelihood-normalized penalty
                           CalPenalty = lambda_cal * LR/T instead of heuristic -log10(p).
        pit_counts:        Number of PIT observations per model (for LR normalization)
        mad_values:        Histogram MAD per model (lower = better)
        regime:            Regime index for adaptive weights and lambda_cal (0-4)
        bic_weight:        Ignored (backward compat)
        hyvarinen_weight:  Ignored (backward compat)
        crps_weight:       Ignored (backward compat)
        lambda_entropy:    Entropy regularisation strength
        epsilon:           Minimum weight floor
    """
    all_models = set(bic_values.keys())

    has_crps = crps_values is not None and len(crps_values) > 0
    has_pit = pit_pvalues is not None and len(pit_pvalues) > 0
    has_berk = berk_pvalues is not None and len(berk_pvalues) > 0
    has_berk_lr = (berkowitz_lr_stats is not None and len(berkowitz_lr_stats) > 0
                   and pit_counts is not None and len(pit_counts) > 0)
    has_mad = mad_values is not None and len(mad_values) > 0

    # Berkowitz component is available if we have either LR stats or p-values
    has_any_berk = has_berk_lr or has_berk

    # Regime-aware lambda_cal for Berkowitz calibration penalty
    if regime is not None and regime in REGIME_BERKOWITZ_LAMBDA:
        lambda_cal = REGIME_BERKOWITZ_LAMBDA[regime]
    else:
        lambda_cal = BERKOWITZ_CALIBRATION_LAMBDA

    # Select regime-aware weights
    if regime is not None and regime in REGIME_SCORING_WEIGHTS:
        w_crps, w_pit, w_berk, w_tail, w_mad = REGIME_SCORING_WEIGHTS[regime]
    else:
        w_crps, w_pit, w_berk, w_tail, w_mad = DEFAULT_ELITE_WEIGHTS

    # If components are missing, redistribute their weight to CRPS
    if not has_pit:
        w_crps += w_pit; w_pit = 0.0
    if not has_any_berk:
        w_crps += w_berk; w_berk = 0.0
    if not has_mad:
        w_crps += w_mad; w_mad = 0.0
    if not has_crps:
        # Extreme fallback: use BIC only (should never happen)
        w_total = 1.0
        bic_std = robust_standardize_scores(bic_values)
        combined_scores = {m: bic_std.get(m, 0.0) for m in all_models}

        weights = entropy_regularized_weights(combined_scores, lambda_entropy=lambda_entropy, eps=epsilon)
        metadata = {
            "combined_scores_standardized": {k: float(v) for k, v in combined_scores.items()},
            "weights_used": {"crps": 0.0, "pit_dev": 0.0, "berk": 0.0, "tail": 0.0, "mad": 0.0, "bic_fallback": 1.0},
            "scoring_method": "bic_fallback",
            "crps_enabled": False, "pit_enabled": False, "regime": regime,
            "lambda_entropy": lambda_entropy,
        }
        return weights, metadata

    # Normalise weights to sum to 1
    w_total = w_crps + w_pit + w_berk + w_tail + w_mad
    if w_total > 0:
        w_crps /= w_total; w_pit /= w_total; w_berk /= w_total
        w_tail /= w_total; w_mad /= w_total

    # ── Build per-component raw scores (all: lower = better) ──

    # CRPS: already lower = better
    crps_std = robust_standardize_scores(crps_values) if has_crps else {}

    # PIT deviation: -log10(p), lower = better (= well calibrated)
    pit_dev_raw = _compute_pit_deviation(pit_pvalues) if has_pit else {}
    pit_dev_std = robust_standardize_scores(pit_dev_raw) if pit_dev_raw else {}

    # Berkowitz calibration penalty: LR/T (preferred) or -log10(p) (fallback)
    # When raw LR stats are available, use likelihood-normalized penalty:
    #   CalPenalty_m = lambda_cal * LR_m / T_m
    # This is scale-stable across assets and likelihood-consistent.
    if has_berk_lr:
        berk_raw = _compute_berk_calibration_penalty(berkowitz_lr_stats, pit_counts, lambda_cal)
        berk_method = "lr_normalized"
    elif has_berk:
        berk_raw = _compute_berk_penalty(berk_pvalues)
        berk_method = "log10_pvalue"
    else:
        berk_raw = {}
        berk_method = "none"
    berk_std = robust_standardize_scores(berk_raw) if berk_raw else {}

    # Tail error: combined PIT+CRPS proxy, lower = better
    tail_raw = _compute_tail_error(pit_pvalues or {}, crps_values) if has_crps else {}
    tail_std = robust_standardize_scores(tail_raw) if tail_raw else {}

    # MAD: already lower = better
    mad_std = robust_standardize_scores(mad_values) if has_mad else {}

    # BIC/Hyvarinen: stored for metadata only
    bic_std = robust_standardize_scores(bic_values)
    hyv_std = robust_standardize_scores(hyvarinen_scores)

    # ── Combine ──
    combined_scores = {}
    for model_name in all_models:
        s = 0.0
        s += w_crps * crps_std.get(model_name, 0.0)
        s += w_pit  * pit_dev_std.get(model_name, 0.0)
        s += w_berk * berk_std.get(model_name, 0.0)
        s += w_tail * tail_std.get(model_name, 0.0)
        s += w_mad  * mad_std.get(model_name, 0.0)
        combined_scores[model_name] = s

    # ── Compute initial weights from combined scores ──
    weights = entropy_regularized_weights(combined_scores, lambda_entropy=lambda_entropy, eps=epsilon)

    # ── Calibration veto gate (February 2026) ──
    # A density forecast with catastrophic PIT or Berkowitz miscalibration is
    # structurally wrong — its CRPS measures sharpness of a WRONG distribution.
    # No amount of CRPS advantage can justify selecting such a model.
    #
    # Post-softmax veto: after computing weights via the normal scoring pipeline,
    # force miscalibrated models to floor weight and redistribute their excess
    # to well-calibrated models.
    #
    # Veto conditions (any triggers veto):
    #   PIT_p < 0.01   when a model with PIT_p >= 0.05 exists
    #   Berk_p < 0.01  when a model with Berk_p >= 0.05 exists
    n_models_total = len(weights)
    veto_floor = max(epsilon, 0.01 / max(n_models_total, 1))

    def _pit_ok(m):
        if not has_pit:
            return True
        p = pit_pvalues.get(m)
        return p is not None and np.isfinite(p) and p >= 0.01

    def _berk_ok(m):
        if not has_berk:
            return True
        p = berk_pvalues.get(m)
        return p is not None and np.isfinite(p) and p >= 0.01

    def _passes_calibration(m):
        return _pit_ok(m) and _berk_ok(m)

    any_cal_passes = any(_passes_calibration(m) for m in weights)

    if any_cal_passes:
        vetoed = set()
        passed = set()
        for m in weights:
            if _passes_calibration(m):
                passed.add(m)
            else:
                vetoed.add(m)

        if vetoed and passed:
            vetoed_excess = 0.0
            for m in vetoed:
                excess = weights[m] - veto_floor
                if excess > 0:
                    vetoed_excess += excess
                    weights[m] = veto_floor

            if vetoed_excess > 0:
                pass_total = sum(weights[m] for m in passed)
                if pass_total > 0:
                    for m in passed:
                        weights[m] += vetoed_excess * (weights[m] / pass_total)

            w_sum = sum(weights.values())
            if w_sum > 0:
                weights = {m: w / w_sum for m, w in weights.items()}

    metadata = {
        "bic_standardized": {k: float(v) if np.isfinite(v) else None for k, v in bic_std.items()},
        "hyvarinen_standardized": {k: float(v) if np.isfinite(v) else None for k, v in hyv_std.items()},
        "crps_standardized": {k: float(v) if np.isfinite(v) else None for k, v in crps_std.items()},
        "pit_dev_standardized": {k: float(v) if np.isfinite(v) else None for k, v in pit_dev_std.items()},
        "berk_standardized": {k: float(v) if np.isfinite(v) else None for k, v in berk_std.items()},
        "berk_raw": {k: float(v) if np.isfinite(v) else None for k, v in berk_raw.items()},
        "tail_standardized": {k: float(v) if np.isfinite(v) else None for k, v in tail_std.items()},
        "mad_standardized": {k: float(v) if np.isfinite(v) else None for k, v in mad_std.items()},
        "combined_scores_standardized": {k: float(v) if np.isfinite(v) else None for k, v in combined_scores.items()},
        "weights_used": {
            "crps": float(w_crps), "pit_dev": float(w_pit), "berk": float(w_berk),
            "tail": float(w_tail), "mad": float(w_mad),
        },
        "berkowitz_method": berk_method,
        "berkowitz_lambda_cal": float(lambda_cal),
        "regime": regime,
        "lambda_entropy": lambda_entropy,
        "crps_enabled": has_crps,
        "pit_enabled": has_pit,
        "scoring_method": "elite_crps_dominated",
    }

    return weights, metadata
