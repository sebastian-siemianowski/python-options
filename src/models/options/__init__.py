"""
===============================================================================
OPTIONS VOLATILITY MODELS — Momentum-Coupled Bayesian Framework
===============================================================================

This module implements sophisticated volatility models for options tuning.
The models leverage momentum distributions from the equity signal pipeline
to inform volatility beliefs through hierarchical Bayesian structure.

CORE ARCHITECTURE:
    Momentum Distribution → Cross-Entropy Prior → SABR-Coupled Vol Model
    
    The equity BMA posterior provides an informative prior:
    - Gaussian momentum → constant/mean-reverting vol preferred
    - Phi-Gaussian momentum → bounded regime vol preferred  
    - Student-t momentum → regime-switching/jump vol preferred

MODEL FAMILIES:
    1. OPTION_MOMENTUM_GAUSSIAN:
       - Constant vol with momentum-informed prior
       - SABR correlation ρ derived from momentum skewness
       - Suitable for normal market conditions
       
    2. OPTION_MOMENTUM_PHI_GAUSSIAN:
       - Mean-reverting vol with bounded dynamics
       - Kappa (mean reversion speed) linked to momentum persistence
       - Suitable for range-bound markets
       
    3. OPTION_MOMENTUM_PHI_STUDENT_T:
       - Regime-switching vol with heavy tails
       - Vol-of-vol ν derived from momentum kurtosis
       - Suitable for crisis/high-vol regimes

SABR INTEGRATION:
    The models implement SABR-style dynamics where:
    - α (initial vol) ← momentum scale parameter
    - ρ (correlation) ← momentum skewness
    - ν (vol-of-vol) ← momentum kurtosis excess
    
    This creates principled connection between observed price dynamics
    and implied volatility surface characteristics.

SELECTION MECHANISM:
    Models compete via combined BIC + Hyvärinen scoring (matching equity tune.py):
    w_combined(m) = w_bic(m)^α * w_hyvarinen(m)^(1-α)
    
    With momentum-conditional weighting:
    - When momentum is Student-t → upweight regime_student_t model
    - When momentum is Phi-Gaussian → upweight phi_gaussian model
    - When momentum is Gaussian → upweight gaussian model

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from .option_model_registry import (
    OptionModelFamily,
    OptionModelSpec,
    OPTION_MODEL_REGISTRY,
    get_option_model_spec,
    get_all_option_model_names,
    get_option_models_for_tuning,
    make_option_momentum_gaussian_name,
    make_option_momentum_phi_gaussian_name,
    make_option_momentum_student_t_name,
    OPTION_STUDENT_T_NU_GRID,
)

from .momentum_bridge import (
    MomentumBridge,
    MomentumDistributionType,
    MomentumParameters,
    extract_momentum_parameters,
    compute_sabr_priors_from_momentum,
    compute_ensemble_weights_from_momentum,
    compute_cross_entropy_vol_prior,
)

from .option_momentum_gaussian import OptionMomentumGaussianModel
from .option_momentum_phi_gaussian import OptionMomentumPhiGaussianModel
from .option_momentum_phi_student_t import OptionMomentumPhiStudentTModel

__all__ = [
    # Registry
    "OptionModelFamily",
    "OptionModelSpec", 
    "OPTION_MODEL_REGISTRY",
    "get_option_model_spec",
    "get_all_option_model_names",
    "get_option_models_for_tuning",
    "make_option_momentum_gaussian_name",
    "make_option_momentum_phi_gaussian_name",
    "make_option_momentum_student_t_name",
    "OPTION_STUDENT_T_NU_GRID",
    # Momentum Bridge
    "MomentumBridge",
    "MomentumDistributionType",
    "MomentumParameters",
    "extract_momentum_parameters",
    "compute_sabr_priors_from_momentum",
    "compute_ensemble_weights_from_momentum",
    "compute_cross_entropy_vol_prior",
    # Momentum-coupled Models
    "OptionMomentumGaussianModel",
    "OptionMomentumPhiGaussianModel",
    "OptionMomentumPhiStudentTModel",
    # Legacy Volatility Models
    "OptionConstantVolModel",
    "OptionMeanRevertingVolModel",
    "OptionRegimeVolModel",
    "OptionRegimeSkewVolModel",
]
