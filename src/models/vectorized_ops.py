"""
Story 7.1: Vectorized Filter Operations with NumPy.

Vectorized implementations for common signal generation operations:
  - phi^H computation across horizons
  - BMA softmax weights
  - Batch Monte Carlo draws

Usage:
    from models.vectorized_ops import vectorized_phi_forecast, vectorized_bma_weights
"""
import numpy as np
from typing import Optional


def vectorized_phi_forecast(
    mu: float,
    phi: float,
    horizons: np.ndarray,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Vectorized multi-horizon forecast using phi^H.
    
    mu_H = mu * phi^H for all H simultaneously.
    
    Args:
        mu: Current level estimate.
        phi: Autoregressive parameter.
        horizons: 1D array of horizon values (e.g., [1,3,7,30,90,180,365]).
        sigma: Observation noise (for interval computation).
    
    Returns:
        Array of forecasts for each horizon.
    """
    horizons = np.asarray(horizons, dtype=float)
    phi_powers = np.power(phi, horizons)
    return mu * phi_powers


def vectorized_phi_variance(
    phi: float,
    q: float,
    R: float,
    horizons: np.ndarray,
) -> np.ndarray:
    """
    Vectorized forecast variance for Kalman filter across horizons.
    
    Var(y_{t+H}) = phi^{2H} * P_t + sum_{j=0}^{H-1} phi^{2j} * q + R
    
    For large H with |phi| < 1:
    sum_{j=0}^{H-1} phi^{2j} = (1 - phi^{2H}) / (1 - phi^2)
    
    Args:
        phi: AR coefficient.
        q: Process noise.
        R: Observation noise.
        horizons: Array of horizons.
    
    Returns:
        Array of forecast variances.
    """
    horizons = np.asarray(horizons, dtype=float)
    phi_sq = phi * phi
    
    if abs(phi_sq - 1.0) < 1e-10:
        # Random walk case
        state_var = q * horizons
    else:
        phi_2H = np.power(phi_sq, horizons)
        state_var = q * (1 - phi_2H) / (1 - phi_sq)
    
    return state_var + R


def vectorized_bma_weights(
    bic_scores: np.ndarray,
    penalty: float = 0.0,
) -> np.ndarray:
    """
    Vectorized BMA weights from BIC scores using softmax.
    
    w_i = exp(-0.5 * BIC_i) / sum(exp(-0.5 * BIC_j))
    
    Numerically stable via log-sum-exp trick.
    
    Args:
        bic_scores: Array of BIC scores (lower is better).
        penalty: Additional penalty per model (for complexity).
    
    Returns:
        Array of BMA weights (sum to 1).
    """
    bic = np.asarray(bic_scores, dtype=float) + penalty
    
    if len(bic) == 0:
        return np.array([])
    
    # Log-sum-exp trick for numerical stability
    log_weights = -0.5 * bic
    max_lw = np.max(log_weights)
    exp_shifted = np.exp(log_weights - max_lw)
    
    total = np.sum(exp_shifted)
    if total <= 0:
        return np.ones(len(bic)) / len(bic)
    
    return exp_shifted / total


def batch_monte_carlo_sample(
    means: np.ndarray,
    variances: np.ndarray,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Batched Monte Carlo draws for multiple horizons simultaneously.
    
    Instead of looping per-horizon, draws all at once.
    
    Args:
        means: Array of means (one per horizon).
        variances: Array of variances (one per horizon).
        n_samples: Number of MC draws per horizon.
        rng: Random generator.
    
    Returns:
        Shape (n_horizons, n_samples) array of samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    stds = np.sqrt(np.maximum(variances, 0.0))
    
    n_horizons = len(means)
    
    # Batch draw: shape (n_horizons, n_samples)
    z = rng.standard_normal((n_horizons, n_samples))
    samples = means[:, np.newaxis] + stds[:, np.newaxis] * z
    
    return samples


def vectorized_quantiles(
    samples: np.ndarray,
    quantiles: np.ndarray = None,
) -> np.ndarray:
    """
    Vectorized quantile computation across horizons.
    
    Args:
        samples: Shape (n_horizons, n_samples).
        quantiles: Array of quantile levels (e.g., [0.10, 0.25, 0.50, 0.75, 0.90]).
    
    Returns:
        Shape (n_horizons, n_quantiles) array.
    """
    if quantiles is None:
        quantiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    
    return np.quantile(samples, quantiles, axis=1).T
