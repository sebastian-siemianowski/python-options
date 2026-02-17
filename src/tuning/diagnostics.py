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
    min_sigma: float = 1e-8
) -> float:
    """
    Compute Hyvärinen score for Gaussian predictive density.
    
    For Gaussian p(r) = N(μ, σ²):
        ∂log p / ∂r = -(r - μ) / σ²
        ∂²log p / ∂r² = -1 / σ²
    
    Therefore:
        H = (1/n) Σ_t [ (r_t - μ_t)² / (2σ_t⁴) - 1/σ_t² ]
    
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
    sigma_4 = sigma ** 4
    
    # Innovations
    innovation = returns - mu
    innovation_sq = innovation ** 2
    
    # Hyvärinen score components:
    # Term 1: (1/2) * (∂log p / ∂r)² = (r - μ)² / (2σ⁴)
    # Term 2: ∂²log p / ∂r² = -1/σ²
    term1 = innovation_sq / (2.0 * sigma_4)
    term2 = -1.0 / sigma_sq
    
    # Per-observation score
    h_scores = term1 + term2
    
    # Filter out non-finite values
    valid = np.isfinite(h_scores)
    if not np.any(valid):
        return -1e12  # Return very bad score
    
    h_mean = float(np.mean(h_scores[valid]))
    
    # Return negated score (higher = better, for consistency with log-likelihood)
    return -h_mean


def compute_hyvarinen_score_student_t(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
    min_sigma: float = 1e-8,
    min_nu: float = 2.1
) -> float:
    """
    Compute Hyvärinen score for Student-t predictive density.
    
    For Student-t p(r) with location μ, scale σ, and degrees of freedom ν:
    
    Let z = (r - μ) / σ
    
        log p(r) = const - ((ν+1)/2) * log(1 + z²/ν) - log(σ)
        
        ∂log p / ∂r = -((ν+1)/ν) * z / (σ * (1 + z²/ν))
                    = -((ν+1) * (r-μ)) / (σ² * (ν + z²))
        
        ∂²log p / ∂r² = -((ν+1)/σ²) * (ν - z²) / (ν + z²)²
    
    Therefore:
        H = (1/n) Σ_t [ (1/2)(∂log p/∂r)² + ∂²log p/∂r² ]
    
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
    
    # Standardized residuals
    z = (returns - mu) / sigma
    z_sq = z ** 2
    
    # Common denominator: ν + z²
    denom = nu + z_sq
    
    # First derivative: ∂log p / ∂r = -((ν+1) * (r-μ)) / (σ² * (ν + z²))
    # Squared: ((ν+1)² * (r-μ)²) / (σ⁴ * (ν + z²)²)
    #        = ((ν+1)² * z²) / (σ² * (ν + z²)²)
    d1_sq = ((nu + 1.0) ** 2 * z_sq) / (sigma_sq * denom ** 2)
    
    # Second derivative: ∂²log p / ∂r² = -((ν+1)/σ²) * (ν - z²) / (ν + z²)²
    d2 = -((nu + 1.0) / sigma_sq) * (nu - z_sq) / (denom ** 2)
    
    # Hyvärinen score: (1/2) * (∂log p/∂r)² + ∂²log p/∂r²
    h_scores = 0.5 * d1_sq + d2
    
    # Filter out non-finite values
    valid = np.isfinite(h_scores)
    if not np.any(valid):
        return -1e12  # Return very bad score
    
    h_mean = float(np.mean(h_scores[valid]))
    
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
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Robust cross-model standardization using median and MAD.
    
    Preserves ordering while normalizing heterogeneous score scales.
    This ensures BIC and Hyvärinen can be meaningfully combined without
    one dominating due to raw scale differences.
    
    The MAD is scaled by 1.4826 to be consistent with standard deviation
    for Gaussian data: MAD * 1.4826 ≈ σ for N(μ, σ²).
    
    Why median/MAD:
    - Robust to Hyvärinen spikes
    - Stable in low-n regimes
    - No Gaussian assumptions (but calibrated to be consistent with σ)
    
    Args:
        scores: Dictionary mapping model name to raw score
        eps: Small constant to prevent division by zero
        
    Returns:
        Dictionary of standardized scores (zero median, unit scale)
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
    
    # Standardize all scores
    standardized = {}
    for k, v in scores.items():
        if np.isfinite(v):
            standardized[k] = (v - median) / scale
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
# REGIME-AWARE COMBINED MODEL WEIGHTS (BIC + Hyvärinen + CRPS)
# =============================================================================

# Regime-specific weight configurations: (bic_weight, hyvarinen_weight, crps_weight)
REGIME_SCORING_WEIGHTS = {
    0: (0.30, 0.30, 0.40),  # Unknown: balanced with structural geometry checks
    1: (0.25, 0.20, 0.55),  # Crisis: CRPS critical, but maintain stability pressure
    2: (0.30, 0.25, 0.45),  # Trending: forecast quality > curvature purity
    3: (0.45, 0.30, 0.25),  # Ranging: BIC heavy, less CRPS (noise-dominated)
    4: (0.30, 0.40, 0.30),  # Low Vol: curvature misspecification visible, Hyv high
}

DEFAULT_BIC_WEIGHT_COMBINED = 0.35
DEFAULT_HYVARINEN_WEIGHT_COMBINED = 0.30
DEFAULT_CRPS_WEIGHT_COMBINED = 0.35
CRPS_SCORING_ENABLED = True


def compute_regime_aware_model_weights(
    bic_values: Dict[str, float],
    hyvarinen_scores: Dict[str, float],
    crps_values: Optional[Dict[str, float]] = None,
    regime: Optional[int] = None,
    bic_weight: Optional[float] = None,
    hyvarinen_weight: Optional[float] = None,
    crps_weight: Optional[float] = None,
    lambda_entropy: float = DEFAULT_ENTROPY_LAMBDA,
    epsilon: float = 1e-10,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute regime-aware model weights using BIC + Hyvärinen + CRPS.
    """
    has_crps = crps_values is not None and len(crps_values) > 0 and CRPS_SCORING_ENABLED
    
    # Determine weights
    if bic_weight is not None and hyvarinen_weight is not None:
        w_bic, w_hyv = bic_weight, hyvarinen_weight
        w_crps = crps_weight if crps_weight is not None else 0.0
    elif regime is not None and regime in REGIME_SCORING_WEIGHTS and has_crps:
        w_bic, w_hyv, w_crps = REGIME_SCORING_WEIGHTS[regime]
    elif has_crps:
        w_bic, w_hyv, w_crps = DEFAULT_BIC_WEIGHT_COMBINED, DEFAULT_HYVARINEN_WEIGHT_COMBINED, DEFAULT_CRPS_WEIGHT_COMBINED
    else:
        w_bic, w_hyv, w_crps = 0.5, 0.5, 0.0
    
    # Normalize
    w_total = w_bic + w_hyv + w_crps
    if w_total > 0:
        w_bic, w_hyv, w_crps = w_bic / w_total, w_hyv / w_total, w_crps / w_total
    
    # Standardize
    bic_std = robust_standardize_scores(bic_values)
    hyv_std = robust_standardize_scores(hyvarinen_scores)
    crps_std = robust_standardize_scores(crps_values) if has_crps else {}
    
    # Combine: BIC/CRPS lower=better (+), Hyv higher=better (-)
    combined_scores = {}
    for model_name in bic_values.keys():
        b = bic_std.get(model_name, 0.0)
        h = hyv_std.get(model_name, 0.0)
        c = crps_std.get(model_name, 0.0) if has_crps else 0.0
        combined_scores[model_name] = w_bic * b - w_hyv * h + w_crps * c
    
    weights = entropy_regularized_weights(combined_scores, lambda_entropy=lambda_entropy, eps=epsilon)
    
    metadata = {
        "bic_standardized": {k: float(v) if np.isfinite(v) else None for k, v in bic_std.items()},
        "hyvarinen_standardized": {k: float(v) if np.isfinite(v) else None for k, v in hyv_std.items()},
        "crps_standardized": {k: float(v) if np.isfinite(v) else None for k, v in crps_std.items()} if has_crps else {},
        "combined_scores_standardized": {k: float(v) if np.isfinite(v) else None for k, v in combined_scores.items()},
        "weights_used": {"bic": float(w_bic), "hyvarinen": float(w_hyv), "crps": float(w_crps)},
        "regime": regime,
        "lambda_entropy": lambda_entropy,
        "crps_enabled": has_crps,
        "scoring_method": "regime_aware_bic_hyv_crps" if has_crps else "bic_hyv_only",
    }
    
    return weights, metadata
