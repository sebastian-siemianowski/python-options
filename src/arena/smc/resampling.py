"""
===============================================================================
RESAMPLING — SMC Resampling Algorithms
===============================================================================

Implements resampling strategies for Sequential Monte Carlo:

1. Systematic Resampling (recommended)
   - Low variance
   - O(N) complexity
   - Maintains diversity better than multinomial

2. Multinomial Resampling
   - Simple but higher variance
   - Standard baseline

3. Residual Resampling
   - Combines deterministic and stochastic components

Reference: Douc & Cappe (2005) "Comparison of Resampling Schemes"

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import List, Tuple
import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size from normalized weights.
    
    ESS = 1 / Σ w_i²
    
    ESS = N means all particles have equal weight (maximum diversity)
    ESS = 1 means one particle has all the weight (degeneracy)
    
    Args:
        weights: Normalized particle weights (sum to 1)
        
    Returns:
        Effective sample size
    """
    weights = np.asarray(weights).flatten()
    
    # Normalize if not already
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        return 1.0
    
    # ESS formula
    ess = 1.0 / np.sum(weights ** 2)
    
    return float(ess)


def systematic_resample(
    weights: np.ndarray,
    n_samples: int = None,
) -> np.ndarray:
    """
    Systematic resampling with low variance.
    
    Uses a single random offset and deterministic spacing to select
    particles. This is the recommended resampling method for SMC.
    
    Algorithm:
        1. Compute cumulative weights C_i = Σ_{j≤i} w_j
        2. Draw single uniform u ~ U[0, 1/N]
        3. Select particle i where C_{i-1} < (u + (j-1)/N) ≤ C_i
    
    Args:
        weights: Normalized particle weights
        n_samples: Number of samples (default: same as len(weights))
        
    Returns:
        Array of selected particle indices
    """
    weights = np.asarray(weights).flatten()
    n = len(weights)
    n_samples = n_samples or n
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(n) / n
    
    # Cumulative sum
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # Ensure exact 1.0 due to floating point
    
    # Generate systematic points
    u0 = np.random.uniform(0, 1.0 / n_samples)
    u = u0 + np.arange(n_samples) / n_samples
    
    # Find indices via searchsorted
    indices = np.searchsorted(cumsum, u)
    
    # Clip to valid range (handles floating point edge cases)
    indices = np.clip(indices, 0, n - 1)
    
    return indices


def multinomial_resample(
    weights: np.ndarray,
    n_samples: int = None,
) -> np.ndarray:
    """
    Multinomial resampling (standard but higher variance).
    
    Draw n_samples from categorical distribution defined by weights.
    Simple but introduces more variance than systematic resampling.
    
    Args:
        weights: Normalized particle weights
        n_samples: Number of samples (default: same as len(weights))
        
    Returns:
        Array of selected particle indices
    """
    weights = np.asarray(weights).flatten()
    n = len(weights)
    n_samples = n_samples or n
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(n) / n
    
    # Draw from categorical
    indices = np.random.choice(n, size=n_samples, replace=True, p=weights)
    
    return indices


def residual_resample(
    weights: np.ndarray,
    n_samples: int = None,
) -> np.ndarray:
    """
    Residual resampling (combines deterministic and stochastic).
    
    Algorithm:
        1. Compute n_i = floor(N * w_i) for each particle
        2. Deterministically select n_i copies of particle i
        3. Resample remaining R = N - Σ n_i using residual weights
    
    This reduces variance compared to pure multinomial while
    maintaining unbiasedness.
    
    Args:
        weights: Normalized particle weights
        n_samples: Number of samples (default: same as len(weights))
        
    Returns:
        Array of selected particle indices
    """
    weights = np.asarray(weights).flatten()
    n = len(weights)
    n_samples = n_samples or n
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(n) / n
    
    # Deterministic part: floor(N * w_i) copies
    n_copies = np.floor(n_samples * weights).astype(int)
    
    # Build indices for deterministic part
    indices = []
    for i in range(n):
        indices.extend([i] * n_copies[i])
    
    # Residual part
    n_remaining = n_samples - len(indices)
    
    if n_remaining > 0:
        # Residual weights
        residual_weights = n_samples * weights - n_copies
        residual_weights = residual_weights / np.sum(residual_weights)
        
        # Multinomial resample for residuals
        residual_indices = np.random.choice(n, size=n_remaining, replace=True, p=residual_weights)
        indices.extend(residual_indices.tolist())
    
    return np.array(indices)


def stratified_resample(
    weights: np.ndarray,
    n_samples: int = None,
) -> np.ndarray:
    """
    Stratified resampling (alternative low-variance method).
    
    Similar to systematic but uses independent uniform draws in each stratum.
    
    Algorithm:
        1. Divide [0,1] into N equal strata
        2. Draw u_i ~ U[(i-1)/N, i/N] for each stratum
        3. Select particle based on cumulative weights
    
    Args:
        weights: Normalized particle weights
        n_samples: Number of samples (default: same as len(weights))
        
    Returns:
        Array of selected particle indices
    """
    weights = np.asarray(weights).flatten()
    n = len(weights)
    n_samples = n_samples or n
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(n) / n
    
    # Cumulative sum
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    
    # Generate stratified points
    u = np.zeros(n_samples)
    for i in range(n_samples):
        u[i] = (i + np.random.uniform()) / n_samples
    
    # Find indices
    indices = np.searchsorted(cumsum, u)
    indices = np.clip(indices, 0, n - 1)
    
    return indices


def should_resample(ess: float, n_particles: int, threshold_ratio: float = 0.5) -> bool:
    """
    Determine if resampling should be triggered.
    
    Standard criterion: resample when ESS < N/2
    
    Args:
        ess: Current effective sample size
        n_particles: Total number of particles
        threshold_ratio: ESS/N ratio below which to resample
        
    Returns:
        True if resampling should be performed
    """
    threshold = threshold_ratio * n_particles
    return ess < threshold


def compute_resampling_statistics(
    weights_before: np.ndarray,
    indices: np.ndarray,
) -> dict:
    """
    Compute statistics about resampling operation.
    
    Args:
        weights_before: Weights before resampling
        indices: Selected particle indices
        
    Returns:
        Dictionary with resampling statistics
    """
    n = len(weights_before)
    
    # Count how many times each particle was selected
    counts = np.bincount(indices, minlength=n)
    
    # Statistics
    n_unique = np.sum(counts > 0)
    n_dead = np.sum(counts == 0)
    max_copies = np.max(counts)
    
    # Effective diversity
    diversity = n_unique / n
    
    return {
        "n_particles": n,
        "n_unique": int(n_unique),
        "n_dead": int(n_dead),
        "max_copies": int(max_copies),
        "diversity_ratio": float(diversity),
        "ess_before": float(effective_sample_size(weights_before)),
    }
