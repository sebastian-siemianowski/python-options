"""
===============================================================================
OPTIONS MODEL REGISTRY — Single Source of Truth for Volatility Models
===============================================================================

This module defines the CANONICAL registry of all volatility models used in
options tuning and signal generation.

ARCHITECTURAL LAW:
    - option_tune.py ITERATES the registry to fit models
    - option_signal.py DISPATCHES via registry to generate signals
    - No model can exist in one without existing in the other
    - Model names are generated ONLY by registry functions

MODEL FAMILIES:
    1. OPTION_MOMENTUM_GAUSSIAN — Constant vol with momentum priors
    2. OPTION_MOMENTUM_PHI_GAUSSIAN — Mean-reverting vol with bounded dynamics
    3. OPTION_MOMENTUM_PHI_STUDENT_T — Regime-switching vol with heavy tails

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class OptionModelFamily(Enum):
    """Option volatility model family classification."""
    # Momentum-coupled models
    MOMENTUM_GAUSSIAN = "momentum_gaussian"
    MOMENTUM_PHI_GAUSSIAN = "momentum_phi_gaussian"
    MOMENTUM_PHI_STUDENT_T = "momentum_phi_student_t"
    # Legacy models (non-momentum)
    CONSTANT = "constant"
    MEAN_REVERTING = "mean_reverting"
    REGIME = "regime"
    REGIME_SKEW = "regime_skew"
    # Special models
    SABR = "sabr"  # Future: full SABR model
    VARIANCE_SWAP = "variance_swap"  # Model-free anchor


class OptionSupportType(Enum):
    """
    Predictive support type for options models.
    
    SURFACE: Models full volatility surface (strike/expiry)
    SCALAR: Models single ATM volatility
    TERM_STRUCTURE: Models term structure only
    """
    SURFACE = "surface"
    SCALAR = "scalar"
    TERM_STRUCTURE = "term_structure"


@dataclass(frozen=True)
class OptionModelSpec:
    """
    Immutable specification for a single volatility model.
    
    This is the CANONICAL contract between option_tune.py and option_signal.py.
    """
    name: str                               # Unique model identifier
    family: OptionModelFamily               # Model family for dispatch
    support: OptionSupportType              # Predictive support type
    n_params: int                           # Number of free parameters (for BIC)
    requires_skew: bool = False             # Whether model requires skew data
    requires_term_structure: bool = False   # Whether model requires term structure
    momentum_coupled: bool = True           # Whether model uses momentum priors
    nu: Optional[int] = None                # Degrees of freedom (for Student-t)
    description: str = ""                   # Human-readable description
    
    def __post_init__(self):
        """Validate specification."""
        if self.n_params < 1:
            raise ValueError(f"n_params must be >= 1: {self.n_params}")


# =============================================================================
# PARAMETER GRIDS
# =============================================================================

# Student-t degrees of freedom grid (matching equity tune.py)
OPTION_STUDENT_T_NU_GRID = [4, 8, 20]

# Mean-reversion kappa grid
OPTION_KAPPA_GRID = [0.1, 0.3, 0.5, 0.7]

# SABR correlation grid (for future SABR implementation)
SABR_RHO_GRID = [-0.9, -0.7, -0.5, -0.3, 0.0]

# SABR vol-of-vol grid
SABR_NU_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


# =============================================================================
# MODEL NAME GENERATORS (Canonical)
# =============================================================================

def make_option_momentum_gaussian_name() -> str:
    """Generate canonical name for momentum-coupled Gaussian vol model."""
    return "option_momentum_gaussian"


def make_option_momentum_phi_gaussian_name() -> str:
    """Generate canonical name for momentum-coupled Phi-Gaussian vol model."""
    return "option_momentum_phi_gaussian"


def make_option_momentum_student_t_name(nu: int) -> str:
    """Generate canonical name for momentum-coupled Phi-Student-t vol model."""
    return f"option_momentum_phi_student_t_nu_{nu}"


def make_option_sabr_name(rho: float, nu: float) -> str:
    """Generate canonical name for SABR model (future)."""
    return f"option_sabr_rho_{rho:.1f}_nu_{nu:.1f}"


def make_option_variance_swap_name() -> str:
    """Generate canonical name for variance swap anchor model."""
    return "option_variance_swap"


# =============================================================================
# MODEL REGISTRY
# =============================================================================

def _build_option_model_registry() -> Dict[str, OptionModelSpec]:
    """Build the complete option model registry."""
    registry = {}
    
    # 1. Momentum Gaussian — constant vol with momentum prior
    name = make_option_momentum_gaussian_name()
    registry[name] = OptionModelSpec(
        name=name,
        family=OptionModelFamily.MOMENTUM_GAUSSIAN,
        support=OptionSupportType.SCALAR,
        n_params=2,  # sigma, sigma_prior_weight
        requires_skew=False,
        requires_term_structure=False,
        momentum_coupled=True,
        description="Constant volatility with momentum-informed prior from equity signal"
    )
    
    # 2. Momentum Phi-Gaussian — mean-reverting vol with bounded dynamics
    name = make_option_momentum_phi_gaussian_name()
    registry[name] = OptionModelSpec(
        name=name,
        family=OptionModelFamily.MOMENTUM_PHI_GAUSSIAN,
        support=OptionSupportType.SCALAR,
        n_params=4,  # sigma_bar, kappa, eta, phi
        requires_skew=False,
        requires_term_structure=True,
        momentum_coupled=True,
        description="Mean-reverting volatility with momentum-derived kappa"
    )
    
    # 3. Momentum Phi-Student-t — regime-switching with heavy tails
    for nu in OPTION_STUDENT_T_NU_GRID:
        name = make_option_momentum_student_t_name(nu)
        registry[name] = OptionModelSpec(
            name=name,
            family=OptionModelFamily.MOMENTUM_PHI_STUDENT_T,
            support=OptionSupportType.SURFACE,
            n_params=5,  # sigma_bar, kappa, eta, rho, (nu fixed)
            requires_skew=True,
            requires_term_structure=True,
            momentum_coupled=True,
            nu=nu,
            description=f"Regime-switching vol with Student-t(ν={nu}) innovations"
        )
    
    # 4. Variance Swap Anchor — model-free benchmark
    name = make_option_variance_swap_name()
    registry[name] = OptionModelSpec(
        name=name,
        family=OptionModelFamily.VARIANCE_SWAP,
        support=OptionSupportType.SCALAR,
        n_params=1,  # variance_swap_vol
        requires_skew=False,
        requires_term_structure=False,
        momentum_coupled=False,
        description="Model-free variance swap fair value from option chain"
    )
    
    # =========================================================================
    # LEGACY VOLATILITY MODELS (non-momentum)
    # =========================================================================
    
    # 5. Constant Volatility — simple constant vol estimation
    registry["constant_vol"] = OptionModelSpec(
        name="constant_vol",
        family=OptionModelFamily.CONSTANT,
        support=OptionSupportType.SCALAR,
        n_params=1,  # sigma
        requires_skew=False,
        requires_term_structure=False,
        momentum_coupled=False,
        description="Constant volatility with prior regularization"
    )
    
    # 6. Mean-Reverting Volatility — Ornstein-Uhlenbeck process
    registry["mean_reverting_vol"] = OptionModelSpec(
        name="mean_reverting_vol",
        family=OptionModelFamily.MEAN_REVERTING,
        support=OptionSupportType.SCALAR,
        n_params=3,  # sigma_bar, kappa, eta
        requires_skew=False,
        requires_term_structure=True,
        momentum_coupled=False,
        description="Mean-reverting vol (Ornstein-Uhlenbeck dynamics)"
    )
    
    # 7. Regime Volatility — per-regime vol estimation
    registry["regime_vol"] = OptionModelSpec(
        name="regime_vol",
        family=OptionModelFamily.REGIME,
        support=OptionSupportType.SCALAR,
        n_params=2,  # sigma_r, eta_r per regime
        requires_skew=False,
        requires_term_structure=False,
        momentum_coupled=False,
        description="Regime-conditional volatility with hierarchical fallback"
    )
    
    # 8. Regime Skew Volatility — per-regime vol + skew
    registry["regime_skew_vol"] = OptionModelSpec(
        name="regime_skew_vol",
        family=OptionModelFamily.REGIME_SKEW,
        support=OptionSupportType.SURFACE,
        n_params=4,  # sigma_r, skew_r, eta_r, skew_eta_r per regime
        requires_skew=True,
        requires_term_structure=False,
        momentum_coupled=False,
        description="Regime-conditional volatility with skew component"
    )
    
    return registry


# Global registry instance
OPTION_MODEL_REGISTRY: Dict[str, OptionModelSpec] = _build_option_model_registry()


# =============================================================================
# REGISTRY ACCESS FUNCTIONS
# =============================================================================

def get_option_model_spec(model_name: str) -> Optional[OptionModelSpec]:
    """Get specification for a model by name."""
    return OPTION_MODEL_REGISTRY.get(model_name)


def get_all_option_model_names() -> List[str]:
    """Get all registered model names."""
    return list(OPTION_MODEL_REGISTRY.keys())


def get_option_models_for_tuning(
    include_momentum: bool = True,
    include_legacy: bool = True,
    include_variance_swap: bool = True,
    student_t_nu_grid: Optional[List[int]] = None,
) -> List[str]:
    """
    Get list of models to include in tuning.
    
    Args:
        include_momentum: Include momentum-coupled models
        include_legacy: Include legacy (non-momentum) models
        include_variance_swap: Include model-free variance swap
        student_t_nu_grid: Custom nu grid (default: OPTION_STUDENT_T_NU_GRID)
        
    Returns:
        List of model names for BMA competition
    """
    if student_t_nu_grid is None:
        student_t_nu_grid = OPTION_STUDENT_T_NU_GRID
    
    models = []
    
    if include_momentum:
        # Momentum Gaussian
        models.append(make_option_momentum_gaussian_name())
        
        # Momentum Phi-Gaussian
        models.append(make_option_momentum_phi_gaussian_name())
        
        # Momentum Phi-Student-t for each nu
        for nu in student_t_nu_grid:
            models.append(make_option_momentum_student_t_name(nu))
    
    if include_legacy:
        # Legacy models
        models.append("constant_vol")
        models.append("mean_reverting_vol")
        models.append("regime_vol")
        models.append("regime_skew_vol")
    
    if include_variance_swap:
        models.append(make_option_variance_swap_name())
    
    return models


def get_option_models_by_family(family: OptionModelFamily) -> List[str]:
    """Get all models belonging to a specific family."""
    return [
        name for name, spec in OPTION_MODEL_REGISTRY.items()
        if spec.family == family
    ]


def extract_option_model_params_for_sampling(
    model_name: str,
    fitted_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract parameters needed for signal generation from fitted params.
    
    Args:
        model_name: Model name from registry
        fitted_params: Parameters from option_tune
        
    Returns:
        Dictionary with sampling parameters
    """
    spec = get_option_model_spec(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Extract common parameters
    params = {
        "model_name": model_name,
        "family": spec.family.value,
        "momentum_coupled": spec.momentum_coupled,
    }
    
    # Family-specific extraction
    if spec.family == OptionModelFamily.MOMENTUM_GAUSSIAN:
        params["sigma"] = fitted_params.get("sigma", 0.25)
        params["confidence_lower"] = fitted_params.get("confidence_bounds", {}).get("lower", 0.15)
        params["confidence_upper"] = fitted_params.get("confidence_bounds", {}).get("upper", 0.35)
        
    elif spec.family == OptionModelFamily.MOMENTUM_PHI_GAUSSIAN:
        params["sigma_bar"] = fitted_params.get("sigma_bar", 0.25)
        params["kappa"] = fitted_params.get("kappa", 0.5)
        params["eta"] = fitted_params.get("eta", 0.05)
        params["half_life_days"] = fitted_params.get("half_life_days", 14)
        
    elif spec.family == OptionModelFamily.MOMENTUM_PHI_STUDENT_T:
        params["sigma_bar"] = fitted_params.get("sigma_bar", 0.25)
        params["kappa"] = fitted_params.get("kappa", 0.5)
        params["eta"] = fitted_params.get("eta", 0.05)
        params["rho"] = fitted_params.get("sabr_rho", -0.5)
        params["nu"] = spec.nu
        # Regime-specific parameters if available
        if "regime_params" in fitted_params:
            params["regime_params"] = fitted_params["regime_params"]
            
    elif spec.family == OptionModelFamily.VARIANCE_SWAP:
        params["variance_swap_vol"] = fitted_params.get("variance_swap_vol", 0.25)
        
    return params


# =============================================================================
# VALIDATION
# =============================================================================

def validate_option_model_registry() -> bool:
    """
    Validate registry integrity.
    
    Checks:
    - All names are unique
    - All specs are valid
    - All Student-t models have valid nu
    
    Returns:
        True if registry is valid
    """
    names = set()
    
    for name, spec in OPTION_MODEL_REGISTRY.items():
        # Check uniqueness
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name)
        
        # Check name consistency
        if spec.name != name:
            raise ValueError(f"Spec name mismatch: {spec.name} vs {name}")
        
        # Check Student-t has nu
        if spec.family == OptionModelFamily.MOMENTUM_PHI_STUDENT_T:
            if spec.nu is None:
                raise ValueError(f"Student-t model missing nu: {name}")
            if spec.nu not in OPTION_STUDENT_T_NU_GRID:
                raise ValueError(f"Invalid nu for Student-t: {spec.nu}")
    
    return True


# Validate on import
validate_option_model_registry()