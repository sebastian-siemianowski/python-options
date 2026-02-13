"""
===============================================================================
OPTION MOMENTUM GAUSSIAN MODEL — Constant Vol with Momentum Prior
===============================================================================

Implements constant volatility model with momentum-informed prior.

MATHEMATICAL FORMULATION:
    σ_t = σ (constant)
    σ ~ N(μ_σ, τ²)
    
    Where:
    - μ_σ is derived from momentum scale parameter
    - τ² is derived from momentum confidence

SABR COUPLING:
    The model's volatility prior is coupled to SABR parameters:
    - When momentum shows negative skewness, bias prior upward (crash risk)
    - When momentum shows high kurtosis, widen the prior (regime uncertainty)

USAGE:
    model = OptionMomentumGaussianModel()
    result = model.fit(iv_data, weights, momentum_params)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .momentum_bridge import MomentumParameters, compute_cross_entropy_vol_prior


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VOL_PRIOR_MEAN = 0.25
DEFAULT_VOL_PRIOR_STD = 0.10
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SIGMA = 0.01
MAX_SIGMA = 2.0


@dataclass
class GaussianVolResult:
    """Result of Gaussian volatility model fitting."""
    sigma: float                      # Fitted constant volatility
    sigma_mle: float                  # MLE estimate (before regularization)
    log_likelihood: float             # Log-likelihood
    bic: float                        # Bayesian Information Criterion
    aic: float                        # Akaike Information Criterion
    hyvarinen_score: float           # Hyvärinen score for model comparison
    n_params: int                     # Number of parameters
    n_obs: int                        # Number of observations
    residual_std: float              # Residual standard deviation
    confidence_bounds: Dict[str, float]  # Confidence interval
    prior_mean: float                 # Prior mean used
    prior_std: float                  # Prior std used
    momentum_coupled: bool            # Whether momentum prior was applied
    fit_success: bool                 # Whether fit was successful


class OptionMomentumGaussianModel:
    """
    Constant volatility model with momentum-coupled prior.
    
    This is the simplest option volatility model, suitable for:
    - Normal market conditions
    - High-confidence equity signals
    - Short-dated options with stable surfaces
    """
    
    MODEL_NAME = "option_momentum_gaussian"
    N_PARAMS = 2  # sigma, regularization_weight
    
    def __init__(
        self,
        prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
        prior_std: float = DEFAULT_VOL_PRIOR_STD,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ):
        """
        Initialize model.
        
        Args:
            prior_mean: Prior mean for volatility
            prior_std: Prior standard deviation
            confidence_level: Confidence level for bounds (0.95 = 95%)
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.confidence_level = confidence_level
    
    def fit(
        self,
        iv_data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        momentum_params: Optional[MomentumParameters] = None,
    ) -> GaussianVolResult:
        """
        Fit constant volatility model with momentum-informed prior.
        
        Args:
            iv_data: Implied volatility data (array)
            weights: Observation weights (optional, from expiry stratification)
            momentum_params: Momentum parameters from equity signal
            
        Returns:
            GaussianVolResult with fitted parameters and diagnostics
        """
        # Validate input
        valid_mask = np.isfinite(iv_data)
        valid_data = iv_data[valid_mask]
        n = len(valid_data)
        
        if n < 5:
            return self._failed_result("insufficient_data")
        
        # Apply weights
        if weights is not None:
            w = weights[valid_mask]
            w = w / np.sum(w)
        else:
            w = np.ones(n) / n
        
        # Compute momentum-adjusted prior
        if momentum_params is not None:
            prior_mean, prior_std = compute_cross_entropy_vol_prior(
                momentum_params, self.prior_mean
            )
            momentum_coupled = True
        else:
            prior_mean = self.prior_mean
            prior_std = self.prior_std
            momentum_coupled = False
        
        # Weighted MLE
        sigma_mle = np.sum(w * valid_data)
        
        # Posterior mean with Gaussian prior (conjugate update)
        effective_n = n
        prior_precision = 1 / (prior_std ** 2)
        data_precision = effective_n / (np.var(valid_data) + 1e-8)
        
        sigma_posterior = (
            (prior_precision * prior_mean + data_precision * sigma_mle) /
            (prior_precision + data_precision)
        )
        
        # Clamp to valid range
        sigma_posterior = max(MIN_SIGMA, min(MAX_SIGMA, sigma_posterior))
        
        # Compute residuals and likelihood
        residuals = valid_data - sigma_posterior
        sigma_resid = np.sqrt(np.sum(w * residuals ** 2)) + 1e-8
        
        # Log-likelihood
        ll = np.sum(w * norm.logpdf(residuals, 0, sigma_resid)) * n
        
        # Information criteria
        k = self.N_PARAMS
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        
        # Hyvärinen score (matching tune.py pattern)
        hyvarinen_score = self._compute_hyvarinen_score(valid_data, sigma_posterior, sigma_resid)
        
        # Confidence bounds
        se = sigma_resid / np.sqrt(effective_n)
        z = norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = max(MIN_SIGMA, sigma_posterior - z * se)
        ci_upper = min(MAX_SIGMA, sigma_posterior + z * se)
        
        return GaussianVolResult(
            sigma=float(sigma_posterior),
            sigma_mle=float(sigma_mle),
            log_likelihood=float(ll),
            bic=float(bic),
            aic=float(aic),
            hyvarinen_score=float(hyvarinen_score),
            n_params=k,
            n_obs=n,
            residual_std=float(sigma_resid),
            confidence_bounds={
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": self.confidence_level,
            },
            prior_mean=float(prior_mean),
            prior_std=float(prior_std),
            momentum_coupled=momentum_coupled,
            fit_success=True,
        )
    
    def _compute_hyvarinen_score(
        self,
        data: np.ndarray,
        sigma: float,
        sigma_resid: float,
    ) -> float:
        """
        Compute Hyvärinen score for model comparison.
        
        The score measures how well the model captures the score function
        of the data distribution. Higher (less negative) is better.
        
        Clamps extreme values to ensure finite results.
        """
        try:
            residuals = data - sigma
            # Use variance instead of sigma_resid^2 to avoid numerical issues
            var = max(sigma_resid ** 2, 1e-8)
            score = residuals / var
            score_sq = np.mean(score ** 2)
            # Clamp to reasonable range for BMA
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> GaussianVolResult:
        """Return a failed result with error message."""
        return GaussianVolResult(
            sigma=DEFAULT_VOL_PRIOR_MEAN,
            sigma_mle=DEFAULT_VOL_PRIOR_MEAN,
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=self.N_PARAMS,
            n_obs=0,
            residual_std=0.0,
            confidence_bounds={
                "lower": DEFAULT_VOL_PRIOR_MEAN,
                "upper": DEFAULT_VOL_PRIOR_MEAN,
                "level": self.confidence_level,
                "error": error,
            },
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            momentum_coupled=False,
            fit_success=False,
        )
    
    def to_dict(self, result: GaussianVolResult) -> Dict[str, Any]:
        """Convert result to dictionary for caching."""
        return {
            "model_class": self.MODEL_NAME,
            "sigma": result.sigma,
            "sigma_mle": result.sigma_mle,
            "log_likelihood": result.log_likelihood,
            "bic": result.bic,
            "aic": result.aic,
            "hyvarinen_score": result.hyvarinen_score,
            "n_params": result.n_params,
            "n_obs": result.n_obs,
            "residual_std": result.residual_std,
            "confidence_bounds": result.confidence_bounds,
            "prior_mean": result.prior_mean,
            "prior_std": result.prior_std,
            "momentum_coupled": result.momentum_coupled,
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GaussianVolResult:
        """Reconstruct result from cached dictionary."""
        return GaussianVolResult(
            sigma=data.get("sigma", DEFAULT_VOL_PRIOR_MEAN),
            sigma_mle=data.get("sigma_mle", DEFAULT_VOL_PRIOR_MEAN),
            log_likelihood=data.get("log_likelihood", float('-inf')),
            bic=data.get("bic", float('inf')),
            aic=data.get("aic", float('inf')),
            hyvarinen_score=data.get("hyvarinen_score", float('-inf')),
            n_params=data.get("n_params", cls.N_PARAMS),
            n_obs=data.get("n_obs", 0),
            residual_std=data.get("residual_std", 0.0),
            confidence_bounds=data.get("confidence_bounds", {}),
            prior_mean=data.get("prior_mean", DEFAULT_VOL_PRIOR_MEAN),
            prior_std=data.get("prior_std", DEFAULT_VOL_PRIOR_STD),
            momentum_coupled=data.get("momentum_coupled", False),
            fit_success=data.get("fit_success", False),
        )
