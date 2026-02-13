"""
===============================================================================
MOMENTUM BRIDGE — Cross-Domain Information Transfer Layer
===============================================================================

This module bridges the equity momentum distribution to volatility model priors.
It extracts actionable parameters from the equity signal's momentum distribution
and transforms them into SABR-compatible volatility priors.

CORE INSIGHT:
    The equity signal pipeline has already computed rich momentum distributions:
    - Gaussian for normal regimes
    - Phi-Gaussian for bounded mean-reverting states
    - Student-t for heavy-tailed regimes
    
    This information should DRIVE volatility beliefs, not merely inform them.

INFORMATION FLOW:
    Equity Signal Pipeline
         │
         ├── momentum_distribution_type: "student_t"
         ├── momentum_mu: 0.02
         ├── momentum_sigma: 0.15
         ├── momentum_df: 4.5  (for Student-t)
         ├── momentum_skewness: -0.3
         └── momentum_kurtosis: 7.2
                    │
                    ▼
         ┌─────────────────────────┐
         │  Momentum Bridge        │
         │  - Extract parameters   │
         │  - Compute SABR priors  │
         │  - Set ensemble weights │
         └─────────────────────────┘

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, Tuple

import numpy as np
from scipy.stats import norm, skew, kurtosis


class MomentumDistributionType(Enum):
    """
    Classification of momentum distribution type from equity signal.
    
    Maps to volatility model preference:
    - GAUSSIAN → favor constant_vol, momentum_gaussian
    - PHI_GAUSSIAN → favor mean_reverting, momentum_phi_gaussian
    - STUDENT_T → favor regime_switching, momentum_phi_student_t
    """
    GAUSSIAN = "gaussian"
    PHI_GAUSSIAN = "phi_gaussian"
    STUDENT_T = "student_t"
    UNKNOWN = "unknown"


@dataclass
class MomentumParameters:
    """
    Extracted momentum parameters for volatility prior construction.
    
    These parameters are transformed from the equity signal's
    momentum distribution to inform SABR-style volatility priors.
    """
    distribution_type: MomentumDistributionType
    
    # Location and scale
    mu: float              # Momentum location (drift direction)
    sigma: float           # Momentum scale (dispersion)
    
    # Shape parameters
    skewness: float        # Momentum skewness (asymmetry)
    kurtosis: float        # Momentum kurtosis (tail heaviness)
    
    # Distribution-specific parameters
    phi: Optional[float] = None       # AR(1) persistence (Phi-Gaussian)
    nu: Optional[float] = None        # Degrees of freedom (Student-t)
    
    # Derived SABR priors
    sabr_alpha_prior: float = 0.25    # Initial vol prior
    sabr_rho_prior: float = -0.5      # Correlation prior (typically negative)
    sabr_nu_prior: float = 0.4        # Vol-of-vol prior
    
    # Ensemble weighting
    ensemble_weight_gaussian: float = 0.33
    ensemble_weight_phi_gaussian: float = 0.33
    ensemble_weight_student_t: float = 0.34
    
    # Confidence in parameters
    confidence: float = 0.5           # 0-1, higher = more trust in parameters
    
    def __post_init__(self):
        """Validate and compute derived quantities."""
        # Clamp values to reasonable ranges
        self.sigma = max(0.01, min(2.0, self.sigma))
        self.skewness = max(-3.0, min(3.0, self.skewness))
        self.kurtosis = max(0.0, min(50.0, self.kurtosis))


class MomentumBridge:
    """
    Bridge class for transforming equity momentum to volatility priors.
    
    USAGE:
        bridge = MomentumBridge()
        params = bridge.extract_from_equity_signal(equity_signal)
        sabr_priors = bridge.compute_sabr_priors(params)
        ensemble_weights = bridge.compute_ensemble_weights(params)
    """
    
    # Mapping from excess kurtosis to Student-t nu
    KURTOSIS_TO_NU = {
        (0, 3): 20,     # Normal-like
        (3, 6): 12,     # Mild fat tails
        (6, 12): 8,     # Moderate fat tails
        (12, 20): 6,    # Heavy fat tails
        (20, 100): 4,   # Very heavy fat tails
    }
    
    def __init__(
        self,
        base_vol_prior: float = 0.25,
        base_rho_prior: float = -0.5,
        base_nu_prior: float = 0.4,
    ):
        """
        Initialize momentum bridge.
        
        Args:
            base_vol_prior: Baseline volatility prior (25%)
            base_rho_prior: Baseline SABR correlation (-0.5)
            base_nu_prior: Baseline SABR vol-of-vol (0.4)
        """
        self.base_vol_prior = base_vol_prior
        self.base_rho_prior = base_rho_prior
        self.base_nu_prior = base_nu_prior
    
    def detect_distribution_type(
        self,
        equity_signal: Dict[str, Any],
    ) -> MomentumDistributionType:
        """
        Detect the momentum distribution type from equity signal.
        
        Uses model_posterior weights to determine dominant distribution.
        """
        model_posterior = equity_signal.get("model_posterior", {})
        
        if not model_posterior:
            return MomentumDistributionType.UNKNOWN
        
        # Sum weights by family
        gaussian_weight = 0.0
        phi_gaussian_weight = 0.0
        student_t_weight = 0.0
        
        for model_name, weight in model_posterior.items():
            name_lower = model_name.lower()
            
            if "student_t" in name_lower or "phi_t" in name_lower:
                student_t_weight += weight
            elif "phi_gaussian" in name_lower or "phi_gau" in name_lower:
                phi_gaussian_weight += weight
            elif "gaussian" in name_lower and "phi" not in name_lower:
                gaussian_weight += weight
        
        # Determine dominant type
        max_weight = max(gaussian_weight, phi_gaussian_weight, student_t_weight)
        
        if max_weight == student_t_weight and student_t_weight > 0.2:
            return MomentumDistributionType.STUDENT_T
        elif max_weight == phi_gaussian_weight and phi_gaussian_weight > 0.2:
            return MomentumDistributionType.PHI_GAUSSIAN
        elif max_weight == gaussian_weight and gaussian_weight > 0.2:
            return MomentumDistributionType.GAUSSIAN
        else:
            return MomentumDistributionType.UNKNOWN
    
    def extract_from_equity_signal(
        self,
        equity_signal: Dict[str, Any],
    ) -> MomentumParameters:
        """
        Extract momentum parameters from equity signal.
        
        Args:
            equity_signal: Dictionary from equity signal pipeline containing:
                - probability_up (or p_up)
                - expected_return_pct (or exp_ret)
                - model_posterior (dict of model → weight)
                - Optional: momentum_features, regime_label
                
        Returns:
            MomentumParameters with all derived quantities
        """
        # Detect distribution type
        dist_type = self.detect_distribution_type(equity_signal)
        
        # Extract basic parameters
        p_up = equity_signal.get("probability_up", equity_signal.get("p_up", 0.5))
        exp_ret = equity_signal.get("expected_return_pct", equity_signal.get("exp_ret", 0.0))
        
        # Normalize expected return
        if isinstance(exp_ret, (int, float)) and abs(exp_ret) < 1:
            exp_ret = exp_ret * 100  # Convert to percentage
        
        # Compute momentum location and scale
        mu = (p_up - 0.5) * 0.1  # Map p_up to drift direction
        sigma = abs(exp_ret) / 100 + 0.01  # Scale from expected return
        
        # Compute skewness from probability asymmetry
        skewness = (p_up - 0.5) * 2  # -1 to +1 range
        
        # Estimate kurtosis from distribution type
        if dist_type == MomentumDistributionType.STUDENT_T:
            # Extract nu from model posterior
            nu = self._extract_nu_from_posterior(equity_signal.get("model_posterior", {}))
            if nu and nu > 4:
                kurtosis_excess = 6 / (nu - 4)
            else:
                kurtosis_excess = 10.0  # Heavy tails
        elif dist_type == MomentumDistributionType.PHI_GAUSSIAN:
            kurtosis_excess = 0.5  # Slightly above normal
            nu = None
        else:
            kurtosis_excess = 0.0  # Normal
            nu = None
        
        kurtosis_val = 3.0 + kurtosis_excess  # Total kurtosis
        
        # Extract phi for Phi-Gaussian
        phi = equity_signal.get("phi", None)
        
        # Compute SABR priors
        sabr_priors = self.compute_sabr_priors_internal(mu, sigma, skewness, kurtosis_excess)
        
        # Compute ensemble weights
        ensemble_weights = self.compute_ensemble_weights_internal(dist_type, p_up)
        
        # Compute confidence from conviction
        conviction = abs(p_up - 0.5) * 2
        confidence = 0.3 + 0.5 * conviction  # 0.3 to 0.8
        
        return MomentumParameters(
            distribution_type=dist_type,
            mu=mu,
            sigma=sigma,
            skewness=skewness,
            kurtosis=kurtosis_val,
            phi=phi,
            nu=nu,
            sabr_alpha_prior=sabr_priors["alpha"],
            sabr_rho_prior=sabr_priors["rho"],
            sabr_nu_prior=sabr_priors["nu"],
            ensemble_weight_gaussian=ensemble_weights["gaussian"],
            ensemble_weight_phi_gaussian=ensemble_weights["phi_gaussian"],
            ensemble_weight_student_t=ensemble_weights["student_t"],
            confidence=confidence,
        )
    
    def _extract_nu_from_posterior(
        self,
        model_posterior: Dict[str, float],
    ) -> Optional[float]:
        """Extract weighted average nu from Student-t model posterior."""
        total_weight = 0.0
        weighted_nu = 0.0
        
        for model_name, weight in model_posterior.items():
            if "student_t" in model_name.lower() and "nu_" in model_name:
                # Extract nu value
                try:
                    nu_str = model_name.split("nu_")[-1].split("_")[0]
                    nu = float(nu_str)
                    weighted_nu += nu * weight
                    total_weight += weight
                except (ValueError, IndexError):
                    continue
        
        if total_weight > 0:
            return weighted_nu / total_weight
        return None
    
    def compute_sabr_priors_internal(
        self,
        mu: float,
        sigma: float,
        skewness: float,
        kurtosis_excess: float,
    ) -> Dict[str, float]:
        """
        Compute SABR-style priors from momentum characteristics.
        
        SABR parameters:
        - α (alpha): Initial volatility level
        - ρ (rho): Correlation between spot and vol
        - ν (nu): Vol-of-vol
        
        Mapping:
        - Alpha ← Base vol adjusted by momentum scale
        - Rho ← Derived from momentum skewness (negative skew → negative rho)
        - Nu ← Derived from momentum kurtosis (high kurtosis → high nu)
        """
        # Alpha: scale-adjusted volatility prior
        alpha = self.base_vol_prior * (1 + 0.5 * sigma)
        alpha = max(0.05, min(1.0, alpha))
        
        # Rho: skewness-adjusted correlation
        # Negative skewness (crash risk) → more negative rho
        rho = self.base_rho_prior - 0.3 * skewness
        rho = max(-0.99, min(0.99, rho))
        
        # Nu: kurtosis-adjusted vol-of-vol
        # Higher kurtosis → higher vol-of-vol
        nu = self.base_nu_prior + 0.1 * kurtosis_excess
        nu = max(0.1, min(2.0, nu))
        
        return {
            "alpha": alpha,
            "rho": rho,
            "nu": nu,
        }
    
    def compute_ensemble_weights_internal(
        self,
        dist_type: MomentumDistributionType,
        p_up: float,
    ) -> Dict[str, float]:
        """
        Compute ensemble weights based on momentum distribution type.
        
        When momentum is Student-t → upweight Student-t vol models
        When momentum is Phi-Gaussian → upweight mean-reverting vol models
        When momentum is Gaussian → upweight constant vol models
        """
        # Base weights
        w_gaussian = 0.33
        w_phi_gaussian = 0.33
        w_student_t = 0.34
        
        # Adjust based on distribution type
        if dist_type == MomentumDistributionType.STUDENT_T:
            # Heavy tails detected → favor Student-t vol
            w_student_t = 0.50
            w_phi_gaussian = 0.30
            w_gaussian = 0.20
        elif dist_type == MomentumDistributionType.PHI_GAUSSIAN:
            # Mean-reversion detected → favor Phi-Gaussian vol
            w_phi_gaussian = 0.50
            w_gaussian = 0.30
            w_student_t = 0.20
        elif dist_type == MomentumDistributionType.GAUSSIAN:
            # Normal dynamics → favor constant vol
            w_gaussian = 0.50
            w_phi_gaussian = 0.30
            w_student_t = 0.20
        
        # Additional adjustment for conviction
        conviction = abs(p_up - 0.5) * 2
        if conviction > 0.5:
            # High conviction → slightly favor simpler models
            w_gaussian *= 1.1
        
        # Normalize
        total = w_gaussian + w_phi_gaussian + w_student_t
        
        return {
            "gaussian": w_gaussian / total,
            "phi_gaussian": w_phi_gaussian / total,
            "student_t": w_student_t / total,
        }


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_bridge = MomentumBridge()


def extract_momentum_parameters(equity_signal: Dict[str, Any]) -> MomentumParameters:
    """Extract momentum parameters from equity signal using default bridge."""
    return _default_bridge.extract_from_equity_signal(equity_signal)


def compute_sabr_priors_from_momentum(
    momentum_params: MomentumParameters,
) -> Dict[str, float]:
    """Compute SABR priors from momentum parameters."""
    return {
        "alpha": momentum_params.sabr_alpha_prior,
        "rho": momentum_params.sabr_rho_prior,
        "nu": momentum_params.sabr_nu_prior,
    }


def compute_ensemble_weights_from_momentum(
    momentum_params: MomentumParameters,
) -> Dict[str, float]:
    """Compute ensemble weights from momentum parameters."""
    return {
        "gaussian": momentum_params.ensemble_weight_gaussian,
        "phi_gaussian": momentum_params.ensemble_weight_phi_gaussian,
        "student_t": momentum_params.ensemble_weight_student_t,
    }


def compute_cross_entropy_vol_prior(
    momentum_params: MomentumParameters,
    base_vol: float = 0.25,
) -> Tuple[float, float]:
    """
    Compute cross-entropy optimized volatility prior.
    
    The prior is chosen to minimize KL divergence between the implied
    volatility distribution and the distribution induced by the momentum signal.
    
    Returns:
        Tuple of (prior_mean, prior_std)
    """
    # Use momentum scale to adjust prior mean
    vol_adjustment = 0.2 * momentum_params.sigma * momentum_params.confidence
    prior_mean = base_vol + vol_adjustment
    
    # Use kurtosis to adjust prior uncertainty
    # Higher kurtosis → wider prior
    kurtosis_excess = momentum_params.kurtosis - 3.0
    prior_std = 0.05 * (1 + 0.2 * min(kurtosis_excess, 10.0))
    
    return prior_mean, prior_std
