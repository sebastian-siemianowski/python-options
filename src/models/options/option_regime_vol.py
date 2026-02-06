"""
===============================================================================
OPTION REGIME VOLATILITY MODEL — Regime-Conditional Volatility Estimation
===============================================================================

Implements regime-conditional volatility model with hierarchical fallback.

MATHEMATICAL FORMULATION:
    Per-regime volatility with shrinkage:
    σ_t | r_t ~ N(σ_r, η_r) for each regime r
    
    Shrinkage toward global mean:
    σ_r = α·σ_global + (1-α)·σ_r_mle
    α = min(1, n_min / n_r)
    
    Where:
    - r_t: Regime label (Low_Vol, Normal_Vol, Elevated_Vol, High_Vol, Extreme_Vol)
    - σ_r: Regime-specific volatility level
    - η_r: Regime-specific vol-of-vol

USAGE:
    model = OptionRegimeVolModel()
    result = model.fit(iv_data, regime_labels, weights)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import norm


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VOL_PRIOR_MEAN = 0.25
DEFAULT_VOL_PRIOR_LAMBDA = 1.0
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SAMPLES_PER_REGIME = 10
MIN_SIGMA = 0.01
MAX_SIGMA = 2.0

# Regime labels
VOL_REGIME_LABELS = {
    0: "Low_Vol",
    1: "Normal_Vol",
    2: "Elevated_Vol",
    3: "High_Vol",
    4: "Extreme_Vol",
}


@dataclass
class RegimeParams:
    """Parameters for a single volatility regime."""
    sigma: float              # Regime-specific vol level
    eta: float                # Regime-specific vol-of-vol
    n_obs: int                # Number of observations in regime
    fallback: bool = False    # Whether global fallback was used
    regime_name: str = ""
    confidence_bounds: Optional[Dict[str, float]] = None


@dataclass
class RegimeVolResult:
    """Result of regime volatility model fitting."""
    regime_params: Dict[int, RegimeParams]
    global_sigma: float
    global_eta: float
    log_likelihood: float
    bic: float
    aic: float
    hyvarinen_score: float
    n_params: int
    n_obs: int
    fit_success: bool


class OptionRegimeVolModel:
    """
    Regime-conditional volatility model.
    
    This model is suitable for:
    - Variable market conditions with distinct regimes
    - Adapting volatility estimates to current market state
    - When volatility differs meaningfully across regimes
    """
    
    MODEL_NAME = "regime_vol"
    
    def __init__(
        self,
        prior_mean: float = DEFAULT_VOL_PRIOR_MEAN,
        prior_lambda: float = DEFAULT_VOL_PRIOR_LAMBDA,
        min_samples: int = MIN_SAMPLES_PER_REGIME,
    ):
        """
        Initialize model.
        
        Args:
            prior_mean: Prior for global volatility
            prior_lambda: Regularization strength
            min_samples: Minimum samples before using regime-specific estimate
        """
        self.prior_mean = prior_mean
        self.prior_lambda = prior_lambda
        self.min_samples = min_samples
    
    def fit(
        self,
        iv_data: np.ndarray,
        regime_labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> RegimeVolResult:
        """
        Fit regime-conditional volatility model.
        
        Args:
            iv_data: Implied volatility data
            regime_labels: Regime labels for each observation
            weights: Observation weights (optional)
            
        Returns:
            RegimeVolResult with per-regime parameters
        """
        n = len(iv_data)
        valid_mask = np.isfinite(iv_data)
        
        if np.sum(valid_mask) < 10:
            return self._failed_result("insufficient_data")
        
        # Apply weights
        if weights is not None:
            w = weights.copy()
        else:
            w = np.ones(n)
        
        # Global estimates (fallback)
        global_sigma = np.average(iv_data[valid_mask], weights=w[valid_mask])
        global_eta = np.sqrt(np.average(
            (iv_data[valid_mask] - global_sigma) ** 2,
            weights=w[valid_mask]
        )) + 1e-8
        
        # Fit per-regime parameters
        regime_params = {}
        total_ll = 0.0
        total_params = 0
        
        for r in np.unique(regime_labels):
            mask = (regime_labels == r) & valid_mask
            n_r = np.sum(mask)
            
            if n_r < self.min_samples:
                # Fallback to global with shrinkage
                regime_params[int(r)] = RegimeParams(
                    sigma=float(global_sigma),
                    eta=float(global_eta),
                    n_obs=int(n_r),
                    fallback=True,
                    regime_name=VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
                    confidence_bounds=None,
                )
                continue
            
            regime_data = iv_data[mask]
            regime_weights = w[mask]
            regime_weights = regime_weights / np.sum(regime_weights)
            
            # Posterior mean with shrinkage toward global
            shrinkage = min(1.0, self.min_samples / n_r)
            sigma_r_mle = np.sum(regime_weights * regime_data)
            sigma_r = shrinkage * global_sigma + (1 - shrinkage) * sigma_r_mle
            
            # Regime-specific vol-of-vol
            eta_r = np.sqrt(np.sum(
                regime_weights * (regime_data - sigma_r) ** 2
            )) + 1e-8
            
            # Regime log-likelihood
            ll_r = np.sum(
                regime_weights * norm.logpdf(regime_data, sigma_r, eta_r)
            ) * n_r
            total_ll += ll_r
            total_params += 2  # sigma_r, eta_r per regime
            
            # Confidence bounds
            se = eta_r / np.sqrt(n_r)
            z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
            
            regime_params[int(r)] = RegimeParams(
                sigma=float(sigma_r),
                eta=float(eta_r),
                n_obs=int(n_r),
                fallback=False,
                regime_name=VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
                confidence_bounds={
                    "lower": float(max(MIN_SIGMA, sigma_r - z * se)),
                    "upper": float(min(MAX_SIGMA, sigma_r + z * se)),
                },
            )
        
        # Information criteria
        k = total_params if total_params > 0 else 2
        n_total = np.sum(valid_mask)
        aic = 2 * k - 2 * total_ll
        bic = k * np.log(n_total) - 2 * total_ll
        
        # Hyvärinen score
        hyvarinen_score = self._compute_hyvarinen_score(
            iv_data[valid_mask], regime_labels[valid_mask], regime_params
        )
        
        return RegimeVolResult(
            regime_params=regime_params,
            global_sigma=float(global_sigma),
            global_eta=float(global_eta),
            log_likelihood=float(total_ll),
            bic=float(bic),
            aic=float(aic),
            hyvarinen_score=float(hyvarinen_score),
            n_params=k,
            n_obs=int(n_total),
            fit_success=True,
        )
    
    def _compute_hyvarinen_score(
        self,
        data: np.ndarray,
        regime_labels: np.ndarray,
        regime_params: Dict[int, RegimeParams],
    ) -> float:
        """Compute Hyvärinen score for model comparison."""
        try:
            scores = []
            for iv, r in zip(data, regime_labels):
                params = regime_params.get(int(r))
                if params is None:
                    continue
                sigma_r = params.sigma
                eta_r = params.eta
                var = max(eta_r ** 2, 1e-8)
                score = (iv - sigma_r) / var
                scores.append(score)
            
            if not scores:
                return float('-inf')
            
            score_sq = np.mean(np.array(scores) ** 2)
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> RegimeVolResult:
        """Return a failed result."""
        return RegimeVolResult(
            regime_params={},
            global_sigma=self.prior_mean,
            global_eta=0.05,
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=2,
            n_obs=0,
            fit_success=False,
        )
    
    def to_dict(self, result: RegimeVolResult) -> Dict[str, Any]:
        """Convert result to dictionary for caching."""
        regime_params_dict = {}
        for r, params in result.regime_params.items():
            regime_params_dict[r] = {
                "sigma": params.sigma,
                "eta": params.eta,
                "n_obs": params.n_obs,
                "fallback": params.fallback,
                "regime_name": params.regime_name,
                "confidence_bounds": params.confidence_bounds,
            }
        
        return {
            "model_class": self.MODEL_NAME,
            "regime_params": regime_params_dict,
            "global_sigma": result.global_sigma,
            "global_eta": result.global_eta,
            "log_likelihood": result.log_likelihood,
            "bic": result.bic,
            "aic": result.aic,
            "hyvarinen_score": result.hyvarinen_score,
            "n_params": result.n_params,
            "n_obs": result.n_obs,
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegimeVolResult:
        """Reconstruct result from cached dictionary."""
        regime_params = {}
        for r_str, params_dict in data.get("regime_params", {}).items():
            r = int(r_str)
            regime_params[r] = RegimeParams(
                sigma=params_dict.get("sigma", DEFAULT_VOL_PRIOR_MEAN),
                eta=params_dict.get("eta", 0.05),
                n_obs=params_dict.get("n_obs", 0),
                fallback=params_dict.get("fallback", True),
                regime_name=params_dict.get("regime_name", ""),
                confidence_bounds=params_dict.get("confidence_bounds"),
            )
        
        return RegimeVolResult(
            regime_params=regime_params,
            global_sigma=data.get("global_sigma", DEFAULT_VOL_PRIOR_MEAN),
            global_eta=data.get("global_eta", 0.05),
            log_likelihood=data.get("log_likelihood", float('-inf')),
            bic=data.get("bic", float('inf')),
            aic=data.get("aic", float('inf')),
            hyvarinen_score=data.get("hyvarinen_score", float('-inf')),
            n_params=data.get("n_params", 2),
            n_obs=data.get("n_obs", 0),
            fit_success=data.get("fit_success", False),
        )
