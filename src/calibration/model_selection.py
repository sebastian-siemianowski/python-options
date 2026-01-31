"""
===============================================================================
MODEL SELECTION — Information Criteria and Bayesian Model Averaging Utilities
===============================================================================

This module provides functions for model selection and Bayesian Model Averaging:

    - compute_aic(): Akaike Information Criterion
    - compute_bic(): Bayesian Information Criterion
    - compute_kurtosis(): Sample excess kurtosis
    - compute_bic_model_weights(): Convert BIC to posterior weights
    - compute_bic_model_weights_from_scores(): Convert scores to weights
    - apply_temporal_smoothing(): Smooth model weights over time
    - normalize_weights(): Normalize weights to sum to 1

These functions implement the core model selection logic used in the
Bayesian Model Averaging (BMA) framework.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    AIC = -2*LL + 2*k

    Lower AIC indicates better model fit with penalty for complexity.
    AIC tends to select more complex models than BIC.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters (k)

    Returns:
        AIC value
    """
    return -2.0 * log_likelihood + 2.0 * n_params


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    BIC = -2*LL + k*ln(n)

    Lower BIC indicates better model fit with penalty for complexity.
    BIC penalizes complexity more heavily than AIC for large samples.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters (k)
        n_obs: Number of observations (n)

    Returns:
        BIC value
    """
    if n_obs <= 0:
        n_obs = 1  # Prevent log(0)
    return -2.0 * log_likelihood + n_params * np.log(n_obs)


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

    mean = float(np.mean(data_clean))
    std = float(np.std(data_clean, ddof=1))

    if std < 1e-12:
        return 0.0

    # Fourth central moment / variance^2
    m4 = np.mean(((data_clean - mean) / std) ** 4)

    # Fisher's definition: excess kurtosis = kurtosis - 3
    excess_kurtosis = m4 - 3.0

    return float(excess_kurtosis)


# Default temporal smoothing alpha for model posterior evolution
DEFAULT_TEMPORAL_ALPHA = 0.3

# Default model selection method: 'bic', 'hyvarinen', or 'combined'
DEFAULT_MODEL_SELECTION_METHOD = 'combined'

# Default BIC weight when using combined method (0.5 = equal weighting)
DEFAULT_BIC_WEIGHT = 0.5


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


def compute_bic_model_weights_from_scores(
    scores: Dict[str, float],
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    Convert scores to weights using softmax (higher score = higher weight).
    
    This is used internally after MAD standardization.
    
    Args:
        scores: Dictionary mapping model name to score (higher = better)
        epsilon: Small constant to prevent zero weights
        
    Returns:
        Dictionary of unnormalized weights
    """
    finite_scores = [s for s in scores.values() if np.isfinite(s)]
    if not finite_scores:
        n_models = len(scores)
        return {m: 1.0 / max(n_models, 1) for m in scores}
    
    score_max = max(finite_scores)
    
    weights = {}
    for model_name, score in scores.items():
        if np.isfinite(score):
            delta = score - score_max
            w = np.exp(delta)
            weights[model_name] = max(w, epsilon)
        else:
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
