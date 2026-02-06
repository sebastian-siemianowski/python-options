"""
===============================================================================
OPTION MOMENTUM PHI-GAUSSIAN MODEL — Mean-Reverting Vol with Bounded Dynamics
===============================================================================

Implements mean-reverting volatility model with momentum-coupled parameters.

MATHEMATICAL FORMULATION:
    Ornstein-Uhlenbeck process for volatility:
    dσ_t = κ(σ̄ - σ_t)dt + η·dW_t
    
    Discrete approximation:
    σ_{t+1} = σ_t + κ(σ̄ - σ_t) + ε_t,  ε_t ~ N(0, η²)
    
    Where:
    - σ̄ (sigma_bar): Long-run mean volatility
    - κ (kappa): Mean-reversion speed (higher = faster reversion)
    - η (eta): Volatility of volatility
    
MOMENTUM COUPLING:
    - κ is derived from momentum persistence φ (bounded dynamics)
    - σ̄ prior is informed by momentum scale
    - When momentum shows mean-reversion (Phi-Gaussian), this model is preferred

SABR ANALOGY:
    This model corresponds to SABR with deterministic vol-of-vol,
    suitable for stable mean-reverting volatility regimes.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from .momentum_bridge import MomentumParameters, compute_cross_entropy_vol_prior


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VOL_PRIOR_MEAN = 0.25
DEFAULT_KAPPA_PRIOR = 0.5   # Mean reversion half-life ~14 days
DEFAULT_ETA_PRIOR = 0.05    # Vol-of-vol ~5%
MIN_KAPPA = 0.01
MAX_KAPPA = 0.99
MIN_SIGMA = 0.01
MAX_SIGMA = 2.0


@dataclass
class PhiGaussianVolResult:
    """Result of Phi-Gaussian volatility model fitting."""
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
    prior_sigma_bar: float            # Prior for sigma_bar
    prior_kappa: float                # Prior for kappa
    momentum_coupled: bool
    fit_success: bool


class OptionMomentumPhiGaussianModel:
    """
    Mean-reverting volatility model with momentum-coupled parameters.
    
    This model is suitable for:
    - Range-bound market conditions
    - Mean-reverting volatility regimes
    - Phi-Gaussian momentum distributions (bounded dynamics)
    """
    
    MODEL_NAME = "option_momentum_phi_gaussian"
    N_PARAMS = 4  # sigma_bar, kappa, eta, regularization
    
    def __init__(
        self,
        prior_sigma_bar: float = DEFAULT_VOL_PRIOR_MEAN,
        prior_kappa: float = DEFAULT_KAPPA_PRIOR,
        prior_eta: float = DEFAULT_ETA_PRIOR,
    ):
        """
        Initialize model.
        
        Args:
            prior_sigma_bar: Prior for long-run volatility
            prior_kappa: Prior for mean-reversion speed
            prior_eta: Prior for vol-of-vol
        """
        self.prior_sigma_bar = prior_sigma_bar
        self.prior_kappa = prior_kappa
        self.prior_eta = prior_eta
    
    def fit(
        self,
        iv_data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        momentum_params: Optional[MomentumParameters] = None,
    ) -> PhiGaussianVolResult:
        """
        Fit mean-reverting volatility model.
        
        Args:
            iv_data: Implied volatility time series
            weights: Observation weights (optional)
            momentum_params: Momentum parameters from equity signal
            
        Returns:
            PhiGaussianVolResult with fitted parameters
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
        
        # Compute momentum-adjusted priors
        if momentum_params is not None:
            prior_sigma_bar, _ = compute_cross_entropy_vol_prior(
                momentum_params, self.prior_sigma_bar
            )
            # Adjust kappa based on momentum persistence
            if momentum_params.phi is not None:
                # Higher phi → lower kappa (slower mean reversion)
                prior_kappa = self.prior_kappa * (1 - 0.5 * abs(momentum_params.phi))
            else:
                prior_kappa = self.prior_kappa * momentum_params.ensemble_weight_phi_gaussian
            momentum_coupled = True
        else:
            prior_sigma_bar = self.prior_sigma_bar
            prior_kappa = self.prior_kappa
            momentum_coupled = False
        
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
            ll -= 0.1 * (sigma_bar - prior_sigma_bar) ** 2
            ll -= 0.1 * (kappa - prior_kappa) ** 2
            ll -= 0.01 * (eta - self.prior_eta) ** 2
            
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
            z = norm.ppf(0.975)
            ci_lower = max(MIN_SIGMA, sigma_bar - z * se)
            ci_upper = min(MAX_SIGMA, sigma_bar + z * se)
            
            # Half-life in days
            half_life = np.log(2) / kappa if kappa > 0 else float('inf')
            
            return PhiGaussianVolResult(
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
                    "level": 0.95,
                },
                prior_sigma_bar=float(prior_sigma_bar),
                prior_kappa=float(prior_kappa),
                momentum_coupled=momentum_coupled,
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
        """Compute Hyvärinen score for model comparison.
        
        Clamps extreme values to ensure finite results for BMA.
        """
        try:
            predicted = data[:-1] + kappa * (sigma_bar - data[:-1])
            residuals = data[1:] - predicted
            var = max(eta ** 2, 1e-8)
            score = residuals / var
            score_sq = np.mean(score ** 2)
            # Clamp to reasonable range
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> PhiGaussianVolResult:
        """Return a failed result."""
        return PhiGaussianVolResult(
            sigma_bar=self.prior_sigma_bar,
            kappa=self.prior_kappa,
            eta=self.prior_eta,
            half_life_days=float('inf'),
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=self.N_PARAMS,
            n_obs=0,
            confidence_bounds={"error": error},
            prior_sigma_bar=self.prior_sigma_bar,
            prior_kappa=self.prior_kappa,
            momentum_coupled=False,
            fit_success=False,
        )
    
    def to_dict(self, result: PhiGaussianVolResult) -> Dict[str, Any]:
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
            "prior_sigma_bar": result.prior_sigma_bar,
            "prior_kappa": result.prior_kappa,
            "momentum_coupled": result.momentum_coupled,
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhiGaussianVolResult:
        """Reconstruct result from cached dictionary."""
        return PhiGaussianVolResult(
            sigma_bar=data.get("sigma_bar", DEFAULT_VOL_PRIOR_MEAN),
            kappa=data.get("kappa", DEFAULT_KAPPA_PRIOR),
            eta=data.get("eta", DEFAULT_ETA_PRIOR),
            half_life_days=data.get("half_life_days", float('inf')),
            log_likelihood=data.get("log_likelihood", float('-inf')),
            bic=data.get("bic", float('inf')),
            aic=data.get("aic", float('inf')),
            hyvarinen_score=data.get("hyvarinen_score", float('-inf')),
            n_params=data.get("n_params", cls.N_PARAMS),
            n_obs=data.get("n_obs", 0),
            confidence_bounds=data.get("confidence_bounds", {}),
            prior_sigma_bar=data.get("prior_sigma_bar", DEFAULT_VOL_PRIOR_MEAN),
            prior_kappa=data.get("prior_kappa", DEFAULT_KAPPA_PRIOR),
            momentum_coupled=data.get("momentum_coupled", False),
            fit_success=data.get("fit_success", False),
        )
