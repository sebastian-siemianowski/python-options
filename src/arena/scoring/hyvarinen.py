"""
===============================================================================
HYVARINEN SCORE — Robust Scoring Rule
===============================================================================

The Hyvärinen score is a proper local scoring rule based on the score function
(gradient of log-density). It is robust to model misspecification and does not
require normalizing constants.

For density p(x) and observation y:
    H(p, y) = -2 * ∂log p(y)/∂y - (∂log p(y)/∂y)²

This is equivalent to:
    H(p, y) = Δlog p(y) + (1/2)|∇log p(y)|²

Key Properties:
1. Does not require computation of normalizing constant
2. Robust to heavy tails (unlike log-score)
3. Favors smooth, well-behaved densities

Reference: Hyvärinen (2005) "Estimation of Non-Normalized Statistical Models"

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Optional
import numpy as np
from scipy.stats import norm, t as student_t


def compute_hyvarinen_score_gaussian(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Compute Hyvärinen score for Gaussian predictive distribution.
    
    For N(μ, σ²):
        ∂log p(y)/∂y = -(y - μ) / σ²
        ∂²log p(y)/∂y² = -1 / σ²
        
    Hyvärinen score:
        H = -2 * (-1/σ²) - ((y-μ)/σ²)²
          = 2/σ² - (y-μ)²/σ⁴
    
    Args:
        observations: Actual observed values (n,)
        mu: Predicted means (n,)
        sigma: Predicted standard deviations (n,)
        
    Returns:
        Average Hyvärinen score (lower is better, can be negative)
    """
    observations = np.asarray(observations).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Ensure positive sigma
    sigma = np.maximum(sigma, 1e-10)
    sigma2 = sigma ** 2
    
    # Standardized residual
    z = (observations - mu) / sigma
    
    # Hyvärinen score for Gaussian
    # H = 2/σ² - z²/σ² = (2 - z²) / σ²
    hyvarinen = (2.0 - z**2) / sigma2
    
    return float(np.mean(hyvarinen))


def compute_hyvarinen_score_student_t(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
) -> float:
    """
    Compute Hyvärinen score for Student-t predictive distribution.
    
    For Student-t with location μ, scale σ, df ν:
        log p(y) ∝ -((ν+1)/2) * log(1 + ((y-μ)/σ)²/ν)
        
        ∂log p/∂y = -(ν+1) * (y-μ) / (σ² * (ν + z²))
        
        ∂²log p/∂y² = -(ν+1) * (ν - z²) / (σ² * (ν + z²)²)
        
    where z = (y - μ) / σ
    
    Args:
        observations: Actual observed values (n,)
        mu: Predicted means (n,)
        sigma: Predicted scale parameters (n,)
        nu: Degrees of freedom
        
    Returns:
        Average Hyvärinen score
    """
    observations = np.asarray(observations).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Ensure positive sigma and valid nu
    sigma = np.maximum(sigma, 1e-10)
    nu = max(nu, 2.01)
    
    sigma2 = sigma ** 2
    
    # Standardized residual
    z = (observations - mu) / sigma
    z2 = z ** 2
    
    # Denominator: ν + z²
    denom = nu + z2
    
    # First derivative: ∂log p/∂y
    d1 = -(nu + 1) * z / (sigma * denom)
    
    # Second derivative: ∂²log p/∂y²
    d2 = -(nu + 1) * (nu - z2) / (sigma2 * denom**2)
    
    # Hyvärinen score: -2*d2 - d1²
    hyvarinen = -2 * d2 - d1**2
    
    return float(np.mean(hyvarinen))


def compute_hyvarinen_score_mixture(
    observations: np.ndarray,
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    weight1: float,
) -> float:
    """
    Compute Hyvärinen score for 2-component Gaussian mixture.
    
    For mixture p(y) = w₁*N(μ₁,σ₁²) + w₂*N(μ₂,σ₂²):
    
    Uses numerical differentiation for the score function.
    
    Args:
        observations: Actual observed values (n,)
        mu1, sigma1: First component parameters
        mu2, sigma2: Second component parameters
        weight1: Weight of first component (weight2 = 1 - weight1)
        
    Returns:
        Average Hyvärinen score
    """
    observations = np.asarray(observations).flatten()
    mu1 = np.asarray(mu1).flatten()
    sigma1 = np.asarray(sigma1).flatten()
    mu2 = np.asarray(mu2).flatten()
    sigma2 = np.asarray(sigma2).flatten()
    
    weight2 = 1.0 - weight1
    
    def mixture_pdf(y, m1, s1, m2, s2):
        """Mixture density."""
        return weight1 * norm.pdf(y, m1, s1) + weight2 * norm.pdf(y, m2, s2)
    
    def log_mixture(y, m1, s1, m2, s2):
        """Log mixture density."""
        p = mixture_pdf(y, m1, s1, m2, s2)
        return np.log(np.maximum(p, 1e-300))
    
    # Numerical differentiation
    eps = 1e-5
    hyvarinen_scores = []
    
    for i, y in enumerate(observations):
        m1, s1, m2, s2 = mu1[i], sigma1[i], mu2[i], sigma2[i]
        
        # First derivative (central difference)
        d1 = (log_mixture(y + eps, m1, s1, m2, s2) - 
              log_mixture(y - eps, m1, s1, m2, s2)) / (2 * eps)
        
        # Second derivative
        d2 = (log_mixture(y + eps, m1, s1, m2, s2) - 
              2 * log_mixture(y, m1, s1, m2, s2) +
              log_mixture(y - eps, m1, s1, m2, s2)) / (eps ** 2)
        
        # Hyvärinen score
        h = -2 * d2 - d1**2
        hyvarinen_scores.append(h)
    
    return float(np.mean(hyvarinen_scores))


def compute_hyvarinen_model_weights(
    scores: dict,
    temperature: float = 1.0,
) -> dict:
    """
    Convert Hyvärinen scores to model weights via softmax.
    
    Lower scores are better, so we negate before softmax.
    
    Args:
        scores: Dictionary mapping model names to Hyvärinen scores
        temperature: Softmax temperature (lower = more concentrated)
        
    Returns:
        Dictionary mapping model names to weights (sum to 1)
    """
    if not scores:
        return {}
    
    names = list(scores.keys())
    values = np.array([scores[n] for n in names])
    
    # Negate because lower is better
    neg_values = -values
    
    # Softmax with temperature
    exp_values = np.exp((neg_values - np.max(neg_values)) / temperature)
    weights = exp_values / np.sum(exp_values)
    
    return {name: float(w) for name, w in zip(names, weights)}
