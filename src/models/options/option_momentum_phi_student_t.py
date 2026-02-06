"""
===============================================================================
OPTION MOMENTUM PHI-STUDENT-T MODEL — Regime-Switching Vol with Heavy Tails
===============================================================================

Implements regime-conditional volatility with Student-t innovations.

MATHEMATICAL FORMULATION:
    Regime-switching volatility with heavy-tailed innovations:
    
    σ_{t+1} | r_t = σ_r + κ_r(σ̄_r - σ_t) + ε_t,  ε_t ~ t(ν)
    
    Where:
    - r_t ∈ {Low_Vol, Normal_Vol, Elevated_Vol, High_Vol, Extreme_Vol}
    - σ̄_r: Regime-specific long-run vol
    - κ_r: Regime-specific mean-reversion
    - ν: Degrees of freedom (fixed per model instance)

SABR COUPLING:
    This model maps to SABR with stochastic vol-of-vol:
    - ρ (correlation) ← derived from momentum skewness
    - ν (vol-of-vol) ← derived from momentum kurtosis
    - α (initial vol) ← regime-conditional

MOMENTUM COUPLING:
    - When momentum is Student-t, this model receives elevated BMA weight
    - SABR ρ is calibrated from momentum distribution skewness
    - Regime assignments use equity regime labels as prior

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize
from scipy.special import gammaln

from .momentum_bridge import MomentumParameters, compute_sabr_priors_from_momentum


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_VOL_PRIOR_MEAN = 0.25
DEFAULT_KAPPA_PRIOR = 0.3
DEFAULT_ETA_PRIOR = 0.08
MIN_SAMPLES_PER_REGIME = 10
MIN_SIGMA = 0.01
MAX_SIGMA = 2.0

# Volatility regime labels
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
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0


@dataclass
class PhiStudentTVolResult:
    """Result of Phi-Student-t volatility model fitting."""
    regime_params: Dict[int, RegimeParams]  # Per-regime parameters
    global_sigma: float                      # Global mean vol
    global_eta: float                        # Global vol-of-vol
    sabr_rho: float                          # SABR correlation (from momentum)
    nu: int                                  # Degrees of freedom
    log_likelihood: float
    bic: float
    aic: float
    hyvarinen_score: float
    n_params: int
    n_obs: int
    momentum_coupled: bool
    momentum_kurtosis: Optional[float]       # Original momentum kurtosis
    momentum_skewness: Optional[float]       # Original momentum skewness
    fit_success: bool


class OptionMomentumPhiStudentTModel:
    """
    Regime-switching volatility model with Student-t innovations.
    
    This model is suitable for:
    - Crisis/high-volatility regimes
    - Heavy-tailed return distributions
    - Student-t momentum signals (fat tails detected)
    
    SABR Integration:
    The SABR correlation ρ is derived from momentum skewness:
    - Negative momentum skewness → more negative ρ (crash correlation)
    - This creates principled connection to vol surface shape
    """
    
    MODEL_NAME_TEMPLATE = "option_momentum_phi_student_t_nu_{nu}"
    BASE_N_PARAMS = 5  # sigma, eta, kappa, rho, + regime-specific
    
    def __init__(
        self,
        nu: int,
        prior_sigma: float = DEFAULT_VOL_PRIOR_MEAN,
        prior_kappa: float = DEFAULT_KAPPA_PRIOR,
        prior_eta: float = DEFAULT_ETA_PRIOR,
    ):
        """
        Initialize model with fixed degrees of freedom.
        
        Args:
            nu: Degrees of freedom for Student-t (typically 4, 6, 8, 12, or 20)
            prior_sigma: Prior for vol level
            prior_kappa: Prior for mean-reversion speed
            prior_eta: Prior for vol-of-vol
        """
        self.nu = nu
        self.prior_sigma = prior_sigma
        self.prior_kappa = prior_kappa
        self.prior_eta = prior_eta
        self.model_name = self.MODEL_NAME_TEMPLATE.format(nu=nu)
    
    def fit(
        self,
        iv_data: np.ndarray,
        regime_labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
        skew_data: Optional[np.ndarray] = None,
        momentum_params: Optional[MomentumParameters] = None,
    ) -> PhiStudentTVolResult:
        """
        Fit regime-switching volatility model with Student-t innovations.
        
        Args:
            iv_data: Implied volatility data
            regime_labels: Volatility regime labels (0-4)
            weights: Observation weights (optional)
            skew_data: Volatility skew data (optional)
            momentum_params: Momentum parameters from equity signal
            
        Returns:
            PhiStudentTVolResult with fitted parameters
        """
        n = len(iv_data)
        valid_mask = np.isfinite(iv_data)
        
        if np.sum(valid_mask) < 15:
            return self._failed_result("insufficient_data")
        
        # Apply weights
        if weights is not None:
            w = weights.copy()
        else:
            w = np.ones(n)
        
        # Compute SABR priors from momentum
        if momentum_params is not None:
            sabr_priors = compute_sabr_priors_from_momentum(momentum_params)
            sabr_rho = sabr_priors["rho"]
            momentum_kurtosis = momentum_params.kurtosis
            momentum_skewness = momentum_params.skewness
            momentum_coupled = True
            
            # Adjust prior based on momentum distribution type
            if momentum_params.ensemble_weight_student_t > 0.4:
                # Strong Student-t signal → expect higher vol
                prior_sigma = self.prior_sigma * 1.1
            else:
                prior_sigma = self.prior_sigma
        else:
            sabr_rho = -0.5  # Default negative correlation
            momentum_kurtosis = None
            momentum_skewness = None
            momentum_coupled = False
            prior_sigma = self.prior_sigma
        
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
            
            if n_r < MIN_SAMPLES_PER_REGIME:
                # Fallback to global with shrinkage
                regime_params[int(r)] = RegimeParams(
                    sigma=float(global_sigma),
                    eta=float(global_eta),
                    n_obs=int(n_r),
                    fallback=True,
                    regime_name=VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
                )
                continue
            
            regime_data = iv_data[mask]
            regime_weights = w[mask]
            regime_weights = regime_weights / np.sum(regime_weights)
            
            # Posterior mean with shrinkage toward global
            shrinkage = min(1.0, MIN_SAMPLES_PER_REGIME / n_r)
            sigma_r_mle = np.sum(regime_weights * regime_data)
            sigma_r = shrinkage * global_sigma + (1 - shrinkage) * sigma_r_mle
            
            # Regime-specific vol-of-vol
            eta_r = np.sqrt(np.sum(
                regime_weights * (regime_data - sigma_r) ** 2
            )) + 1e-8
            
            # Student-t log-likelihood
            standardized = (regime_data - sigma_r) / eta_r
            ll_r = np.sum(
                regime_weights * self._student_t_logpdf(standardized, self.nu)
            ) * n_r
            total_ll += ll_r
            total_params += 2  # sigma_r, eta_r per regime
            
            # Confidence bounds
            se = eta_r / np.sqrt(n_r)
            z = norm.ppf(0.975)
            
            regime_params[int(r)] = RegimeParams(
                sigma=float(sigma_r),
                eta=float(eta_r),
                n_obs=int(n_r),
                fallback=False,
                regime_name=VOL_REGIME_LABELS.get(r, f"Regime_{r}"),
                confidence_lower=float(max(MIN_SIGMA, sigma_r - z * se)),
                confidence_upper=float(min(MAX_SIGMA, sigma_r + z * se)),
            )
        
        # Information criteria
        k = total_params + self.BASE_N_PARAMS
        n_total = np.sum(valid_mask)
        aic = 2 * k - 2 * total_ll
        bic = k * np.log(n_total) - 2 * total_ll
        
        # Hyvärinen score
        hyvarinen_score = self._compute_hyvarinen_score(
            iv_data[valid_mask], regime_labels[valid_mask], regime_params
        )
        
        return PhiStudentTVolResult(
            regime_params=regime_params,
            global_sigma=float(global_sigma),
            global_eta=float(global_eta),
            sabr_rho=float(sabr_rho),
            nu=self.nu,
            log_likelihood=float(total_ll),
            bic=float(bic),
            aic=float(aic),
            hyvarinen_score=float(hyvarinen_score),
            n_params=k,
            n_obs=int(n_total),
            momentum_coupled=momentum_coupled,
            momentum_kurtosis=momentum_kurtosis,
            momentum_skewness=momentum_skewness,
            fit_success=True,
        )
    
    def _student_t_logpdf(self, x: np.ndarray, nu: int) -> np.ndarray:
        """Compute Student-t log-PDF."""
        return student_t.logpdf(x, df=nu)
    
    def _compute_hyvarinen_score(
        self,
        data: np.ndarray,
        regime_labels: np.ndarray,
        regime_params: Dict[int, RegimeParams],
    ) -> float:
        """Compute Hyvärinen score for model comparison.
        
        Clamps extreme values to ensure finite results for BMA.
        """
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
            # Clamp to reasonable range for BMA
            score_sq = min(score_sq, 1e6)
            return -score_sq
        except Exception:
            return float('-inf')
    
    def _failed_result(self, error: str) -> PhiStudentTVolResult:
        """Return a failed result."""
        return PhiStudentTVolResult(
            regime_params={},
            global_sigma=self.prior_sigma,
            global_eta=self.prior_eta,
            sabr_rho=-0.5,
            nu=self.nu,
            log_likelihood=float('-inf'),
            bic=float('inf'),
            aic=float('inf'),
            hyvarinen_score=float('-inf'),
            n_params=self.BASE_N_PARAMS,
            n_obs=0,
            momentum_coupled=False,
            momentum_kurtosis=None,
            momentum_skewness=None,
            fit_success=False,
        )
    
    def to_dict(self, result: PhiStudentTVolResult) -> Dict[str, Any]:
        """Convert result to dictionary for caching."""
        regime_params_dict = {}
        for r, params in result.regime_params.items():
            regime_params_dict[r] = {
                "sigma": params.sigma,
                "eta": params.eta,
                "n_obs": params.n_obs,
                "fallback": params.fallback,
                "regime_name": params.regime_name,
                "confidence_lower": params.confidence_lower,
                "confidence_upper": params.confidence_upper,
            }
        
        return {
            "model_class": self.model_name,
            "regime_params": regime_params_dict,
            "global_sigma": result.global_sigma,
            "global_eta": result.global_eta,
            "sabr_rho": result.sabr_rho,
            "nu": result.nu,
            "log_likelihood": result.log_likelihood,
            "bic": result.bic,
            "aic": result.aic,
            "hyvarinen_score": result.hyvarinen_score,
            "n_params": result.n_params,
            "n_obs": result.n_obs,
            "momentum_coupled": result.momentum_coupled,
            "momentum_kurtosis": result.momentum_kurtosis,
            "momentum_skewness": result.momentum_skewness,
            "fit_success": result.fit_success,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhiStudentTVolResult:
        """Reconstruct result from cached dictionary."""
        regime_params = {}
        for r_str, params_dict in data.get("regime_params", {}).items():
            r = int(r_str)
            regime_params[r] = RegimeParams(
                sigma=params_dict.get("sigma", DEFAULT_VOL_PRIOR_MEAN),
                eta=params_dict.get("eta", DEFAULT_ETA_PRIOR),
                n_obs=params_dict.get("n_obs", 0),
                fallback=params_dict.get("fallback", True),
                regime_name=params_dict.get("regime_name", ""),
                confidence_lower=params_dict.get("confidence_lower", 0),
                confidence_upper=params_dict.get("confidence_upper", 1),
            )
        
        return PhiStudentTVolResult(
            regime_params=regime_params,
            global_sigma=data.get("global_sigma", DEFAULT_VOL_PRIOR_MEAN),
            global_eta=data.get("global_eta", DEFAULT_ETA_PRIOR),
            sabr_rho=data.get("sabr_rho", -0.5),
            nu=data.get("nu", 8),
            log_likelihood=data.get("log_likelihood", float('-inf')),
            bic=data.get("bic", float('inf')),
            aic=data.get("aic", float('inf')),
            hyvarinen_score=data.get("hyvarinen_score", float('-inf')),
            n_params=data.get("n_params", cls.BASE_N_PARAMS),
            n_obs=data.get("n_obs", 0),
            momentum_coupled=data.get("momentum_coupled", False),
            momentum_kurtosis=data.get("momentum_kurtosis"),
            momentum_skewness=data.get("momentum_skewness"),
            fit_success=data.get("fit_success", False),
        )
