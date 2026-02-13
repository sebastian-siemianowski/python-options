"""
===============================================================================
OPTION REGIME SKEW VOLATILITY MODEL — Regime-Conditional Vol + Skew
===============================================================================

Implements regime-conditional volatility model with volatility skew component.

MATHEMATICAL FORMULATION:
    Joint modeling of volatility and skew per regime:
    (σ_t, ξ_t) | r_t ~ Bivariate normal per regime
    
    Where:
    - σ_t: ATM implied volatility at time t
    - ξ_t: Volatility skew (put-call IV difference) at time t
    - r_t: Regime label
    
    Per-regime parameters:
    - σ_r, η_r: Volatility level and vol-of-vol
    - ξ_r, η_ξr: Skew level and skew volatility

USAGE:
    model = OptionRegimeSkewVolModel()
    result = model.fit(iv_data, skew_data, regime_labels, weights)

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
MIN_SAMPLES_PER_REGIME = 15
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
class RegimeSkewParams:
    """Parameters for a single volatility+skew regime."""
    sigma: float              # Regime-specific vol level
    skew: float               # Regime-specific skew level
    eta: float                # Vol-of-vol
    skew_eta: float           # Skew volatility
    n_obs: int                # Observations in regime
    fallback: bool = False    # Whether global fallback was used
    regime_name: str = ""
    confidence_bounds: Optional[Dict[str, float]] = None


@dataclass
class RegimeSkewVolResult:
    """Result of regime skew volatility model fitting."""
    regime_params: Dict[int, RegimeSkewParams]
    global_sigma: float
    global_skew: float
    log_likelihood: float
    bic: float
    aic: float
    hyvarinen_score: float
    n_params: int
    n_obs: int
    fit_success: bool


class OptionRegimeSkewVolModel:
    """
    Regime-conditional volatility model with skew.
    
    This model is suitable for:
    - Markets with significant skew dynamics
    - When put-call skew varies by regime
    - Options strategy selection based on skew
    """
    
    MODEL_NAME = "regime_skew_vol"
    
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
        skew_data: np.ndarray,
        regime_labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> RegimeSkewVolResult:
        """
        Fit regime-conditional volatility model with skew.
        
        Args:
            iv_data: Implied volatility data
            skew_data: Volatility skew data (put IV - call IV)
            regime_labels: Regime labels for each observation
            weights: Observation weights (optional)
            
        Returns:
            RegimeSkewVolResult with per-regime parameters
        """
        n = len(iv_data)
        valid_mask = np.isfinite(iv_data) & np.isfinite(skew_data)
        
        if np.sum(valid_mask) < 15:
            return self._failed_result("insufficient_data")
        
        # Apply weights
        if weights is not None:
            w = weights.copy()
        else:
            w = np.ones(n)
        
        # Global estimates (fallback)
        global_sigma = np.average(iv_data[valid_mask], weights=w[valid_mask])
        global_skew = np.average(skew_data[valid_mask], weights=w[valid_mask])
        global_eta = np.sqrt(np.average(
            (iv_data[valid_mask] - global_sigma) ** 2,
            weights=w[valid_mask]
        )) + 1e-8
        global_skew_eta = np.sqrt(np.average(
            (skew_data[valid_mask] - global_skew) ** 2,
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
                # Fallback to global
                regime_params[int(r)] = RegimeSkewParams(
                    sigma=float(global_sigma),
                    skew=float(global_skew),
                    eta=float(global_eta),
                    skew_eta=float(global_skew_eta),
                    n_obs=int(n_r),
                    fallback=True,
                    regime_name=VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
                    confidence_bounds=None,
                )
                continue
            
            regime_iv = iv_data[mask]
            regime_skew = skew_data[mask]
            regime_weights = w[mask]
            regime_weights = regime_weights / np.sum(regime_weights)
            
            # Posterior mean with shrinkage toward global
            shrinkage = min(1.0, self.min_samples / n_r)
            
            sigma_r_mle = np.sum(regime_weights * regime_iv)
            sigma_r = shrinkage * global_sigma + (1 - shrinkage) * sigma_r_mle
            
            skew_r_mle = np.sum(regime_weights * regime_skew)
            skew_r = shrinkage * global_skew + (1 - shrinkage) * skew_r_mle
            
            # Regime-specific dispersions
            eta_r = np.sqrt(np.sum(
                regime_weights * (regime_iv - sigma_r) ** 2
            )) + 1e-8
            skew_eta_r = np.sqrt(np.sum(
                regime_weights * (regime_skew - skew_r) ** 2
            )) + 1e-8
            
            # Joint log-likelihood (independent for simplicity)
            ll_r = np.sum(
                regime_weights * norm.logpdf(regime_iv, sigma_r, eta_r)
            ) * n_r
            ll_r += np.sum(
                regime_weights * norm.logpdf(regime_skew, skew_r, skew_eta_r)
            ) * n_r
            total_ll += ll_r
            total_params += 4  # sigma_r, skew_r, eta_r, skew_eta_r
            
            # Confidence bounds
            se_sigma = eta_r / np.sqrt(n_r)
            se_skew = skew_eta_r / np.sqrt(n_r)
            z = norm.ppf((1 + DEFAULT_CONFIDENCE_LEVEL) / 2)
            
            regime_params[int(r)] = RegimeSkewParams(
                sigma=float(sigma_r),
                skew=float(skew_r),
                eta=float(eta_r),
                skew_eta=float(skew_eta_r),
                n_obs=int(n_r),
                fallback=False,
                regime_name=VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
                confidence_bounds={
                    "sigma_lower": float(max(MIN_SIGMA, sigma_r - z * se_sigma)),
                    "sigma_upper": float(min(MAX_SIGMA, sigma_r + z * se_sigma)),
                    "skew_lower": float(skew_r - z * se_skew),
                    "skew_upper": float(skew_r + z * se_skew),
                },
            )
        
        # Information criteria
        k = total_params if total_params > 0 else 4
        n_total = np.sum(valid_mask)
        aic = 2 * k - 2 * total_ll
        bic = k * np.log(n_total) - 2 * total_ll
        
        # Hyvärinen score
        hyvarinen_score = self._compute_hyvarinen_score(
            iv_data[valid_mask], skew_data[valid_mask],
            regime_labels[valid_mask], regime_params
        )
        
        return RegimeSkewVolResult(
            regime_params=regime_params,
            global_sigma=float(global_sigma),
            global_skew=float(global_skew),
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
        iv_data: np.ndarray,
        skew_data: np.ndarray,
        regime_labels: np.ndarray,
        regime_params: Dict[int, RegimeSkewParams],
    ) -> float:
        """Compute Hyvärinen score for model comparison."""
        try:
            scores = []
            for iv, skew, r in zip(iv_data, skew_data, regime_labels):
                params = regime_params.get(int(r))
                if params is None:
                    continue
                
                # Score for vol component
                var_sigma = max(params.eta ** 2, 1e-8)
                score_sigma = (iv - params.sigma) / var_sigma
                scores.append(score_sigma)
                
                # Score for skew component
                var_skew = max(params.skew_eta ** 2, 1e-8)
                score_skew = (skew - params.skew) / var_skew
                scores.append(score_skew)
            
            if not scores:
                return float('-inf')
            
            score_sq = np.mean(np.array(scores) ** 2)
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> RegimeSkewVolResult:
        """Return a failed result."""
        return RegimeSkewVolResult(
            regime_params={},
            global_sigma=self.prior_mean,
            global_skew=0.0,
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=4,
            n_obs=0,
            fit_success=False,
        )
    
    def to_dict(self, result: RegimeSkewVolResult) -> Dict[str, Any]:
        """Convert result to dictionary for caching."""
        regime_params_dict = {}
        for r, params in result.regime_params.items():
            regime_params_dict[r] = {
                "sigma": params.sigma,
                "skew": params.skew,
                "eta": params.eta,
                "skew_eta": params.skew_eta,
                "n_obs": params.n_obs,
                "fallback": params.fallback,
                "regime_name": params.regime_name,
                "confidence_bounds": params.confidence_bounds,
            }
        
        return {
            "model_class": self.MODEL_NAME,
            "regime_params": regime_params_dict,
            "global_sigma": result.global_sigma,
            "global_skew": result.global_skew,
            "log_likelihood": result.log_likelihood,
            "bic": result.bic,
            "aic": result.aic,
            "hyvarinen_score": result.hyvarinen_score,
            "n_params": result.n_params,
            "n_obs": result.n_obs,
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegimeSkewVolResult:
        """Reconstruct result from cached dictionary."""
        regime_params = {}
        for r_str, params_dict in data.get("regime_params", {}).items():
            r = int(r_str)
            regime_params[r] = RegimeSkewParams(
                sigma=params_dict.get("sigma", DEFAULT_VOL_PRIOR_MEAN),
                skew=params_dict.get("skew", 0.0),
                eta=params_dict.get("eta", 0.05),
                skew_eta=params_dict.get("skew_eta", 0.01),
                n_obs=params_dict.get("n_obs", 0),
                fallback=params_dict.get("fallback", True),
                regime_name=params_dict.get("regime_name", ""),
                confidence_bounds=params_dict.get("confidence_bounds"),
            )
        
        return RegimeSkewVolResult(
            regime_params=regime_params,
            global_sigma=data.get("global_sigma", DEFAULT_VOL_PRIOR_MEAN),
            global_skew=data.get("global_skew", 0.0),
            log_likelihood=data.get("log_likelihood", float('-inf')),
            bic=data.get("bic", float('inf')),
            aic=data.get("aic", float('inf')),
            hyvarinen_score=data.get("hyvarinen_score", float('-inf')),
            n_params=data.get("n_params", 4),
            n_obs=data.get("n_obs", 0),
            fit_success=data.get("fit_success", False),
        )
