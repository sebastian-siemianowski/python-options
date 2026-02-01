"""
===============================================================================
2-STATE GAUSSIAN MIXTURE MODEL — BMA Candidate for Return Distribution
===============================================================================

Implements a 2-component Gaussian Mixture Model (GMM) for return distributions,
integrated into the Bayesian Model Averaging framework.

MATHEMATICAL FOUNDATION:
    p(z_t) = π₁ · N(z_t; μ₁, σ₁²) + π₂ · N(z_t; μ₂, σ₂²)

Where:
    z_t = r_t / σ_vol_t  (volatility-standardized returns)
    π₁, π₂ = mixing proportions (π₁ + π₂ = 1)
    μ₁, μ₂ = component means
    σ₁², σ₂² = component variances

INTERPRETATION:
    Component 1 ("Momentum"): Captures trending/momentum behavior
        - Typically μ₁ > 0 (positive drift)
        - Moderate variance σ₁²
    
    Component 2 ("Reversal/Crisis"): Captures mean-reversion/crash behavior
        - Typically μ₂ < 0 (negative drift) or μ₂ ≈ 0
        - Higher variance σ₂² (fat tails)

WHY GMM IN BMA:
    1. Captures BIMODAL return distributions that single Gaussian misses
    2. Provides INTRA-REGIME heterogeneity beyond HMM regimes
    3. Naturally models MOMENTUM vs REVERSAL dynamics
    4. Mixing weights provide TIME-VARYING regime probabilities

FITTING APPROACH:
    GMM is fit to volatility-adjusted returns to orthogonalize volatility
    and distributional shape. This allows GMM to capture conditional mean
    dynamics independent of GARCH volatility clustering.

BMA INTEGRATION:
    GMM parameters are stored in tuned_params and used in signal.py:
    - For Monte Carlo simulation, samples are drawn from GMM mixture
    - Mixing weights can be modulated by current regime probabilities
    - Provides fallback to single Gaussian when GMM is degenerate

CORE PRINCIPLE:
    "Bimodality is a hypothesis, not a certainty."
    GMM competes with simpler distributions via BIC weights.

===============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest, norm
from scipy.special import logsumexp


# =============================================================================
# GMM CONSTANTS AND DEFAULTS
# =============================================================================

# Minimum weight for any component (prevents degenerate mixtures)
GMM_MIN_WEIGHT = 0.05
GMM_MAX_WEIGHT = 0.95

# Minimum variance for any component (numerical stability)
GMM_MIN_VARIANCE = 1e-6
GMM_MAX_VARIANCE = 10.0

# EM convergence criteria
GMM_EM_MAX_ITER = 100
GMM_EM_TOL = 1e-6

# Minimum observations for fitting
GMM_MIN_OBS = 100

# Component separation threshold (Mahalanobis distance)
GMM_MIN_SEPARATION = 0.5

# Degenerate threshold (one component captures > this proportion)
GMM_DEGENERATE_THRESHOLD = 0.95


class GaussianMixtureModel:
    """
    2-State Gaussian Mixture Model for return distribution modeling.
    
    Fits a mixture of two Gaussian components to volatility-adjusted returns,
    providing parameters for Monte Carlo simulation in the BMA framework.
    
    The model captures bimodal return distributions that arise from:
    - Alternating momentum and reversal regimes
    - Crash risk (left-tail fattening)
    - Euphoria/melt-up (right-tail fattening)
    """
    
    def __init__(
        self,
        weights: Tuple[float, float] = (0.5, 0.5),
        means: Tuple[float, float] = (0.0, 0.0),
        variances: Tuple[float, float] = (1.0, 1.0)
    ):
        """
        Initialize GMM with component parameters.
        
        Args:
            weights: (π₁, π₂) mixing proportions
            means: (μ₁, μ₂) component means
            variances: (σ₁², σ₂²) component variances
        """
        self.weights = np.array(weights)
        self.means = np.array(means)
        self.variances = np.array(variances)
        self.n_components = 2
        
        # Ensure weights sum to 1
        self.weights = self.weights / np.sum(self.weights)
    
    @property
    def stds(self) -> np.ndarray:
        """Component standard deviations."""
        return np.sqrt(self.variances)
    
    @property
    def is_degenerate(self) -> bool:
        """Check if one component dominates (> DEGENERATE_THRESHOLD)."""
        return np.max(self.weights) > GMM_DEGENERATE_THRESHOLD
    
    @property
    def separation(self) -> float:
        """
        Compute component separation (Mahalanobis distance between means).
        
        separation = |μ₁ - μ₂| / √((σ₁² + σ₂²) / 2)
        """
        mean_diff = abs(self.means[0] - self.means[1])
        avg_var = (self.variances[0] + self.variances[1]) / 2
        return mean_diff / np.sqrt(avg_var) if avg_var > 0 else 0.0
    
    @property
    def is_well_separated(self) -> bool:
        """Check if components are sufficiently separated."""
        return self.separation > GMM_MIN_SEPARATION
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute GMM probability density.
        
        Args:
            x: Array of values
            
        Returns:
            Array of density values
        """
        x = np.asarray(x)
        density = np.zeros_like(x, dtype=float)
        
        for k in range(self.n_components):
            density += self.weights[k] * norm.pdf(
                x, loc=self.means[k], scale=self.stds[k]
            )
        
        return density
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute GMM log-probability density (numerically stable).
        
        Args:
            x: Array of values
            
        Returns:
            Array of log-density values
        """
        x = np.asarray(x)
        
        # Compute log-component densities
        log_components = np.zeros((len(x), self.n_components))
        for k in range(self.n_components):
            log_components[:, k] = (
                np.log(self.weights[k] + 1e-10) +
                norm.logpdf(x, loc=self.means[k], scale=self.stds[k])
            )
        
        # Log-sum-exp for numerical stability
        return logsumexp(log_components, axis=1)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute GMM cumulative distribution function.
        
        Args:
            x: Array of values
            
        Returns:
            Array of CDF values
        """
        x = np.asarray(x)
        cdf_vals = np.zeros_like(x, dtype=float)
        
        for k in range(self.n_components):
            cdf_vals += self.weights[k] * norm.cdf(
                x, loc=self.means[k], scale=self.stds[k]
            )
        
        return cdf_vals
    
    def sample(
        self,
        size: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample from GMM distribution.
        
        Uses two-stage sampling:
        1. Sample component indicator from Categorical(weights)
        2. Sample value from selected Gaussian component
        
        Args:
            size: Number of samples
            rng: Random generator
            
        Returns:
            Array of samples
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample component indicators
        component_idx = rng.choice(
            self.n_components,
            size=size,
            p=self.weights
        )
        
        # Sample from each component
        samples = np.zeros(size)
        for k in range(self.n_components):
            mask = (component_idx == k)
            n_k = np.sum(mask)
            if n_k > 0:
                samples[mask] = rng.normal(
                    loc=self.means[k],
                    scale=self.stds[k],
                    size=n_k
                )
        
        return samples
    
    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        """
        Compute posterior responsibilities γ(z_nk) = p(k | x_n).
        
        These are the posterior probabilities that each observation
        belongs to each component.
        
        Args:
            x: Array of observations
            
        Returns:
            Array of shape (n, k) with responsibilities
        """
        x = np.asarray(x).flatten()
        n = len(x)
        
        # Compute log-likelihoods for each component
        log_probs = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            log_probs[:, k] = (
                np.log(self.weights[k] + 1e-10) +
                norm.logpdf(x, loc=self.means[k], scale=self.stds[k])
            )
        
        # Normalize via log-sum-exp
        log_sum = logsumexp(log_probs, axis=1, keepdims=True)
        responsibilities = np.exp(log_probs - log_sum)
        
        return responsibilities
    
    def log_likelihood(self, x: np.ndarray) -> float:
        """Compute total log-likelihood."""
        return float(np.sum(self.logpdf(x)))
    
    def bic(self, x: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion.
        
        BIC = -2 * log_likelihood + k * log(n)
        
        For 2-component GMM: k = 5 (2 means + 2 variances + 1 weight)
        """
        n = len(x)
        k = 5  # 2 means + 2 variances + 1 free weight (other is 1 - π₁)
        ll = self.log_likelihood(x)
        return -2 * ll + k * np.log(n)
    
    def aic(self, x: np.ndarray) -> float:
        """Compute Akaike Information Criterion."""
        k = 5
        ll = self.log_likelihood(x)
        return -2 * ll + 2 * k
    
    @classmethod
    def fit_em(
        cls,
        x: np.ndarray,
        max_iter: int = GMM_EM_MAX_ITER,
        tol: float = GMM_EM_TOL,
        init_method: str = 'kmeans'
    ) -> Tuple['GaussianMixtureModel', Dict]:
        """
        Fit GMM parameters via Expectation-Maximization.
        
        Args:
            x: Array of observations (volatility-adjusted returns)
            max_iter: Maximum EM iterations
            tol: Convergence tolerance (relative change in log-likelihood)
            init_method: Initialization method ('kmeans', 'random', 'quantile')
            
        Returns:
            Tuple of (fitted model, diagnostics dict)
        """
        x = np.asarray(x).flatten()
        x = x[np.isfinite(x)]
        n = len(x)
        
        if n < GMM_MIN_OBS:
            # Insufficient data: return single Gaussian
            return cls._fallback_single_gaussian(x), {
                "fit_success": False,
                "error": "insufficient_data",
                "n_obs": n,
                "fallback": "single_gaussian"
            }
        
        # Initialize parameters
        weights, means, variances = cls._initialize_params(x, init_method)
        
        prev_ll = float('-inf')
        converged = False
        
        for iteration in range(max_iter):
            # E-step: Compute responsibilities
            resp = np.zeros((n, 2))
            for k in range(2):
                resp[:, k] = weights[k] * norm.pdf(x, loc=means[k], scale=np.sqrt(variances[k]))
            
            # Normalize responsibilities
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp_sum = np.maximum(resp_sum, 1e-10)  # Prevent division by zero
            resp = resp / resp_sum
            
            # M-step: Update parameters
            N_k = resp.sum(axis=0)
            N_k = np.maximum(N_k, 1e-10)  # Prevent division by zero
            
            # Update weights
            weights = N_k / n
            weights = np.clip(weights, GMM_MIN_WEIGHT, GMM_MAX_WEIGHT)
            weights = weights / weights.sum()
            
            # Update means
            means = (resp.T @ x) / N_k
            
            # Update variances
            for k in range(2):
                diff = x - means[k]
                variances[k] = (resp[:, k] @ (diff ** 2)) / N_k[k]
            variances = np.clip(variances, GMM_MIN_VARIANCE, GMM_MAX_VARIANCE)
            
            # Compute log-likelihood
            ll = 0.0
            for k in range(2):
                ll += np.sum(resp[:, k] * (
                    np.log(weights[k] + 1e-10) +
                    norm.logpdf(x, loc=means[k], scale=np.sqrt(variances[k]))
                ))
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
                if rel_change < tol:
                    converged = True
                    break
            
            prev_ll = ll
        
        # Create model
        model = cls(
            weights=tuple(weights),
            means=tuple(means),
            variances=tuple(variances)
        )
        
        # Order components by mean (component 0 = lower mean)
        if model.means[0] > model.means[1]:
            model = cls(
                weights=(model.weights[1], model.weights[0]),
                means=(model.means[1], model.means[0]),
                variances=(model.variances[1], model.variances[0])
            )
        
        diagnostics = {
            "fit_success": True,
            "converged": converged,
            "n_iterations": iteration + 1,
            "final_log_likelihood": float(ll),
            "n_obs": n,
            "separation": float(model.separation),
            "is_well_separated": model.is_well_separated,
            "is_degenerate": model.is_degenerate,
            "component_0_weight": float(model.weights[0]),
            "component_1_weight": float(model.weights[1]),
            "component_0_mean": float(model.means[0]),
            "component_1_mean": float(model.means[1]),
            "component_0_std": float(model.stds[0]),
            "component_1_std": float(model.stds[1]),
        }
        
        return model, diagnostics
    
    @classmethod
    def _initialize_params(
        cls,
        x: np.ndarray,
        method: str = 'kmeans'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize GMM parameters."""
        n = len(x)
        
        if method == 'kmeans':
            # Simple k-means++ style initialization
            # Pick first center randomly
            idx1 = np.random.randint(n)
            c1 = x[idx1]
            
            # Pick second center far from first
            distances = (x - c1) ** 2
            probs = distances / distances.sum()
            idx2 = np.random.choice(n, p=probs)
            c2 = x[idx2]
            
            means = np.array([min(c1, c2), max(c1, c2)])
            
            # Assign points to nearest center
            d1 = np.abs(x - means[0])
            d2 = np.abs(x - means[1])
            labels = (d2 < d1).astype(int)
            
            # Compute initial variances
            variances = np.array([
                np.var(x[labels == 0]) if np.sum(labels == 0) > 1 else 1.0,
                np.var(x[labels == 1]) if np.sum(labels == 1) > 1 else 1.0
            ])
            variances = np.clip(variances, GMM_MIN_VARIANCE, GMM_MAX_VARIANCE)
            
            weights = np.array([
                np.mean(labels == 0),
                np.mean(labels == 1)
            ])
            weights = np.clip(weights, GMM_MIN_WEIGHT, GMM_MAX_WEIGHT)
            weights = weights / weights.sum()
            
        elif method == 'quantile':
            # Initialize based on quantiles
            q25, q75 = np.percentile(x, [25, 75])
            means = np.array([q25, q75])
            variances = np.array([np.var(x)] * 2)
            weights = np.array([0.5, 0.5])
            
        else:  # random
            means = np.random.choice(x, size=2, replace=False)
            means = np.sort(means)
            variances = np.array([np.var(x)] * 2)
            weights = np.array([0.5, 0.5])
        
        return weights, means, variances
    
    @classmethod
    def _fallback_single_gaussian(cls, x: np.ndarray) -> 'GaussianMixtureModel':
        """Create a degenerate GMM equivalent to single Gaussian."""
        mean = float(np.mean(x))
        var = float(np.var(x))
        var = max(var, GMM_MIN_VARIANCE)
        
        return cls(
            weights=(0.99, 0.01),
            means=(mean, mean),
            variances=(var, var)
        )
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary for JSON serialization."""
        return {
            "weights": [float(w) for w in self.weights],
            "means": [float(m) for m in self.means],
            "variances": [float(v) for v in self.variances],
            "stds": [float(s) for s in self.stds],
            "separation": float(self.separation),
            "is_degenerate": bool(self.is_degenerate),
            "is_well_separated": bool(self.is_well_separated),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GaussianMixtureModel':
        """Create model from dictionary."""
        return cls(
            weights=tuple(d["weights"]),
            means=tuple(d["means"]),
            variances=tuple(d["variances"])
        )


def fit_gmm_to_returns(
    returns: np.ndarray,
    vol: np.ndarray,
    min_obs: int = GMM_MIN_OBS
) -> Tuple[Optional[GaussianMixtureModel], Dict]:
    """
    Fit 2-State GMM to volatility-adjusted returns.
    
    This is the main entry point for GMM fitting in the tuning layer.
    
    Args:
        returns: Array of raw returns
        vol: Array of EWMA/GARCH volatility
        min_obs: Minimum observations required
        
    Returns:
        Tuple of (model or None, diagnostics dict)
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    
    # Ensure same length
    n = min(len(returns), len(vol))
    returns = returns[:n]
    vol = vol[:n]
    
    # Remove invalid observations
    valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 1e-10)
    returns_valid = returns[valid_mask]
    vol_valid = vol[valid_mask]
    
    if len(returns_valid) < min_obs:
        return None, {
            "fit_success": False,
            "error": "insufficient_valid_data",
            "n_valid": len(returns_valid),
            "min_required": min_obs
        }
    
    # Standardize returns by volatility
    z = returns_valid / vol_valid
    
    # Winsorize extreme values
    z_p01 = np.percentile(z, 1)
    z_p99 = np.percentile(z, 99)
    z_clipped = np.clip(z, z_p01, z_p99)
    
    # Fit GMM
    model, diag = GaussianMixtureModel.fit_em(z_clipped)
    
    # Add standardization info to diagnostics
    diag["standardized"] = True
    diag["z_mean"] = float(np.mean(z_clipped))
    diag["z_std"] = float(np.std(z_clipped))
    diag["vol_mean"] = float(np.mean(vol_valid))
    
    return model, diag


def compute_gmm_pit(
    returns: np.ndarray,
    vol: np.ndarray,
    gmm: GaussianMixtureModel
) -> Tuple[float, float]:
    """
    Compute PIT calibration for GMM forecasts.
    
    Args:
        returns: Raw returns
        vol: Volatility
        gmm: Fitted GMM model
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    
    n = min(len(returns), len(vol))
    valid_mask = np.isfinite(returns[:n]) & np.isfinite(vol[:n]) & (vol[:n] > 1e-10)
    
    z = returns[:n][valid_mask] / vol[:n][valid_mask]
    
    pit_values = gmm.cdf(z)
    
    if len(pit_values) < 2:
        return 1.0, 0.0
    
    ks_result = kstest(pit_values, 'uniform')
    return float(ks_result.statistic), float(ks_result.pvalue)


# =============================================================================
# UTILITY FUNCTIONS FOR BMA INTEGRATION
# =============================================================================

def get_gmm_model_name() -> str:
    """Generate model name for BMA ensemble."""
    return "phi_gmm_k2"


def is_gmm_model(model_name: str) -> bool:
    """Check if model name is a GMM variant."""
    return model_name.startswith("phi_gmm_") or model_name == "gmm_k2"
