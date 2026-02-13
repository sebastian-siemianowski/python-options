"""
===============================================================================
OPTION MEAN-REVERTING VOLATILITY MODEL — Ornstein-Uhlenbeck Vol Dynamics
===============================================================================

Implements mean-reverting volatility model (Ornstein-Uhlenbeck process).

MATHEMATICAL FORMULATION:
    Ornstein-Uhlenbeck process for volatility:
    dσ_t = κ(σ̄ - σ_t)dt + η·dW_t
    
    Discrete approximation:
    σ_{t+1} = σ_t + κ(σ̄ - σ_t) + ε_t,  ε_t ~ N(0, η²)
    
    Where:
    - σ̄ (sigma_bar): Long-run mean volatility
    - κ (kappa): Mean-reversion speed (higher = faster reversion)
    - η (eta): Volatility of volatility

USAGE:
    model = OptionMeanRevertingVolModel()
    result = model.fit(iv_data, weights)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VOL_PRIOR_MEAN = 0.25
DEFAULT_VOL_PRIOR_LAMBDA = 1.0
DEFAULT_KAPPA_PRIOR = 0.5
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_KAPPA = 0.001
MAX_KAPPA = 0.99
MIN_SIGMA = 0.01
MAX_SIGMA = 2.0


@dataclass
class MeanRevertingVolResult:
    """Result of mean-reverting volatility model fitting."""
    sigma_bar: float                  # Long-run mean volatility
    kappa: float                      # Mean-reversion speed
    eta: float                        # Volatility of volatility
    half_life_days: float            # Half-life in days (ln(2)/κ)
    log_likelihood: float             # Log-likelihood
    bic: float                        # Bayesian Information Criterion
    aic: float                        # Akaike Information Criterion
    hyvarinen_score: float           # Hyvärinen score
    n_params: int                     # Number of parameters
    n_obs: int                        # Number of observations
    confidence_bounds: Dict[str, float]
    fit_success: bool


class OptionMeanRevertingVolModel:
    """
    Mean-reverting volatility model (Ornstein-Uhlenbeck).
    
    This model is suitable for:
    - Range-bound market conditions
    - Mean-reverting volatility regimes
    - Longer estimation windows
    """
    
    MODEL_NAME = "mean_reverting_vol"
    N_PARAMS = 3  # sigma_bar, kappa, eta
    
    def __init__(
        self,
        prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
        prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
        prior_kappa: float = DEFAULT_KAPPA_PRIOR,
    ):
        """
        Initialize model.
        
        Args:
            prior_mean: Prior for long-run volatility
            prior_lambda: Regularization strength
            prior_kappa: Prior for mean-reversion speed
        """
        self.prior_mean = prior_mean
        self.prior_lambda = prior_lambda
        self.prior_kappa = prior_kappa
    
    def fit(
        self,
        iv_data: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> MeanRevertingVolResult:
        """
        Fit mean-reverting volatility model.
        
        Args:
            iv_data: Implied volatility time series
            weights: Observation weights (optional)
            
        Returns:
            MeanRevertingVolResult with fitted parameters
        """
        # Validate input
        valid_mask = np.isfinite(iv_data)
        valid_data = iv_data[valid_mask]
        n = len(valid_data)
        
        if n < 10:
            return self._failed_result("insufficient_data")
        
        # Apply weights
        if weights is not None:
            w = weights[valid_mask][:-1]  # Weights for transitions
            w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(w)) / len(w)
        else:
            w = np.ones(n - 1) / (n - 1)
        
        # Define negative log-likelihood with regularization
        def neg_log_likelihood(params):
            sigma_bar, kappa, eta = params
            
            # Parameter bounds
            if kappa <= MIN_KAPPA or kappa >= MAX_KAPPA:
                return 1e10
            if eta <= 1e-8 or eta > 1.0:
                return 1e10
            if sigma_bar <= MIN_SIGMA or sigma_bar > MAX_SIGMA:
                return 1e10
            
            # Predicted volatility (O-U dynamics)
            predicted = valid_data[:-1] + kappa * (sigma_bar - valid_data[:-1])
            residuals = valid_data[1:] - predicted
            
            # Weighted log-likelihood
            ll = np.sum(w * norm.logpdf(residuals, 0, eta))
            
            # Priors (soft regularization)
            ll -= self.prior_lambda * (sigma_bar - self.prior_mean) ** 2
            ll -= 0.1 * (kappa - self.prior_kappa) ** 2
            
            return -ll
        
        # Initial guess
        x0 = [np.mean(valid_data), 0.1, np.std(np.diff(valid_data))]
        
        try:
            result = minimize(
                neg_log_likelihood, x0,
                method='L-BFGS-B',
                bounds=[
                    (MIN_SIGMA, MAX_SIGMA),
                    (MIN_KAPPA, MAX_KAPPA),
                    (0.001, 1.0),
                ]
            )
            
            sigma_bar, kappa, eta = result.x
            ll = -result.fun
            
            # Information criteria
            k = self.N_PARAMS
            aic = 2 * k - 2 * ll * (n - 1)
            bic = k * np.log(n - 1) - 2 * ll * (n - 1)
            
            # Hyvärinen score
            hyvarinen_score = self._compute_hyvarinen_score(
                valid_data, sigma_bar, kappa, eta
            )
            
            # Confidence bounds (using asymptotic standard errors)
            se = eta / np.sqrt(2 * kappa * n)
            z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
            ci_lower = max(MIN_SIGMA, sigma_bar - z * se)
            ci_upper = min(MAX_SIGMA, sigma_bar + z * se)
            
            # Half-life in days
            half_life = np.log(2) / kappa if kappa > 0 else float('inf')
            
            return MeanRevertingVolResult(
                sigma_bar=float(sigma_bar),
                kappa=float(kappa),
                eta=float(eta),
                half_life_days=float(half_life),
                log_likelihood=float(ll),
                bic=float(bic),
                aic=float(aic),
                hyvarinen_score=float(hyvarinen_score),
                n_params=k,
                n_obs=n,
                confidence_bounds={
                    "sigma_bar_lower": float(ci_lower),
                    "sigma_bar_upper": float(ci_upper),
                    "level": DEFAULT_CONFIDENCE_LEVEL,
                },
                fit_success=True,
            )
            
        except Exception as e:
            return self._failed_result(str(e))
    
    def _compute_hyvarinen_score(
        self,
        data: np.ndarray,
        sigma_bar: float,
        kappa: float,
        eta: float,
    ) -> float:
        """Compute Hyvärinen score for model comparison."""
        try:
            predicted = data[:-1] + kappa * (sigma_bar - data[:-1])
            residuals = data[1:] - predicted
            var = max(eta ** 2, 1e-8)
            score = residuals / var
            score_sq = np.mean(score ** 2)
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> MeanRevertingVolResult:
        """Return a failed result."""
        return MeanRevertingVolResult(
            sigma_bar=self.prior_mean,
            kappa=self.prior_kappa,
            eta=0.05,
            half_life_days=float('inf'),
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=self.N_PARAMS,
            n_obs=0,
            confidence_bounds={"error": error},
            fit_success=False,
        )
    
    def to_dict(self, result: MeanRevertingVolResult) -> Dict[str, Any]:
        """Convert result to dictionary for caching."""
        return {
            "model_class": self.MODEL_NAME,
            "sigma_bar": result.sigma_bar,
            "kappa": result.kappa,
            "eta": result.eta,
            "half_life_days": result.half_life_days,
            "log_likelihood": result.log_likelihood,
            "bic": result.bic,
            "aic": result.aic,
            "hyvarinen_score": result.hyvarinen_score,
            "n_params": result.n_params,
            "n_obs": result.n_obs,
            "confidence_bounds": result.confidence_bounds,
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MeanRevertingVolResult:
        """Reconstruct result from cached dictionary."""
        return MeanRevertingVolResult(
            sigma_bar=data.get("sigma_bar", DEFAULT_VOL_PRIOR_MEAN),
            kappa=data.get("kappa", DEFAULT_KAPPA_PRIOR),
            eta=data.get("eta", 0.05),
            half_life_days=data.get("half_life_days", float('inf')),
            log_likelihood=data.get("log_likelihood", float('-inf')),
            bic=data.get("bic", float('inf')),
            aic=data.get("aic", float('inf')),
            hyvarinen_score=data.get("hyvarinen_score", float('-inf')),
            n_params=data.get("n_params", cls.N_PARAMS),
            n_obs=data.get("n_obs", 0),
            confidence_bounds=data.get("confidence_bounds", {}),
            fit_success=data.get("fit_success", False),
        )
