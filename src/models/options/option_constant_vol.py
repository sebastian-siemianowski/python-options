"""
===============================================================================
OPTION CONSTANT VOLATILITY MODEL — Simple Constant Vol Estimation
===============================================================================

Implements constant volatility model with regularization toward prior.

MATHEMATICAL FORMULATION:
    σ_t = σ (constant)
    
    MLE with Gaussian prior:
    σ_posterior = (n·σ_mle + λ·σ_prior) / (n + λ)

USAGE:
    model = OptionConstantVolModel()
    result = model.fit(iv_data, weights)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import norm


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VOL_PRIOR_MEAN = 0.25
DEFAULT_VOL_PRIOR_LAMBDA = 1.0
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SIGMA = 0.01
MAX_SIGMA = 2.0


@dataclass
class ConstantVolResult:
    """Result of constant volatility model fitting."""
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
    fit_success: bool                 # Whether fit was successful


class OptionConstantVolModel:
    """
    Constant volatility model with prior regularization.
    
    This is the simplest option volatility model, suitable for:
    - Stable market conditions
    - Short estimation windows
    - When regime effects are minimal
    """
    
    MODEL_NAME = "constant_vol"
    N_PARAMS = 1
    
    def __init__(
        self,
        prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
        prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ):
        """
        Initialize model.
        
        Args:
            prior_mean: Prior mean for volatility
            prior_lambda: Regularization strength
            confidence_level: Confidence level for bounds (0.95 = 95%)
        """
        self.prior_mean = prior_mean
        self.prior_lambda = prior_lambda
        self.confidence_level = confidence_level
    
    def fit(
        self,
        iv_data: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> ConstantVolResult:
        """
        Fit constant volatility model.
        
        Args:
            iv_data: Implied volatility data (array)
            weights: Observation weights (optional)
            
        Returns:
            ConstantVolResult with fitted parameters and diagnostics
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
        
        # Weighted MLE
        sigma_mle = np.sum(w * valid_data)
        
        # Posterior mean with Gaussian prior
        effective_n = n
        sigma_posterior = (
            (effective_n * sigma_mle + self.prior_lambda * self.prior_mean) /
            (effective_n + self.prior_lambda)
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
        
        # Hyvärinen score
        hyvarinen_score = self._compute_hyvarinen_score(valid_data, sigma_posterior, sigma_resid)
        
        # Confidence bounds
        se = sigma_resid / np.sqrt(effective_n)
        z = norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = max(MIN_SIGMA, sigma_posterior - z * se)
        ci_upper = min(MAX_SIGMA, sigma_posterior + z * se)
        
        return ConstantVolResult(
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
            fit_success=True,
        )
    
    def _compute_hyvarinen_score(
        self,
        data: np.ndarray,
        sigma: float,
        sigma_resid: float,
    ) -> float:
        """Compute Hyvärinen score for model comparison."""
        try:
            residuals = data - sigma
            var = max(sigma_resid ** 2, 1e-8)
            score = residuals / var
            score_sq = np.mean(score ** 2)
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> ConstantVolResult:
        """Return a failed result with error message."""
        return ConstantVolResult(
            sigma=self.prior_mean,
            sigma_mle=self.prior_mean,
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=self.N_PARAMS,
            n_obs=0,
            residual_std=0.0,
            confidence_bounds={
                "lower": self.prior_mean,
                "upper": self.prior_mean,
                "level": self.confidence_level,
                "error": error,
            },
            fit_success=False,
        )
    
    def to_dict(self, result: ConstantVolResult) -> Dict[str, Any]:
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
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstantVolResult:
        """Reconstruct result from cached dictionary."""
        return ConstantVolResult(
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
            fit_success=data.get("fit_success", False),
        )
