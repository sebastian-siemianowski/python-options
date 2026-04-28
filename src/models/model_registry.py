#!/usr/bin/env python3
"""
===============================================================================
MODEL REGISTRY — Single Source of Truth for All Distributional Models
===============================================================================

This module defines the CANONICAL registry of all models used in the system.
Both tune.py and signals.py MUST use this registry to ensure synchronisation.

ARCHITECTURAL LAW:
    - tune.py ITERATES the registry to fit models
    - signals.py DISPATCHES via registry to sample from fitted models
    - No model can exist in one without existing in the other
    - Model names are generated ONLY by registry functions

This prevents the #1 silent failure mode in quant systems:
    Model name mismatch → dropped from BMA without error → distorted posteriors

===============================================================================
MODEL FAMILIES
===============================================================================

1. GAUSSIAN FAMILY (symmetric, light tails)
   - kalman_gaussian: Standard Kalman filter
   - kalman_phi_gaussian: AR(1) drift persistence

2. STUDENT-T FAMILY (symmetric, heavy tails)
   - phi_student_t_nu_{ν}: Discrete ν grid (4, 6, 8, 12, 20)
   - phi_student_t_nu_{ν}_refined: Adaptive refinement (3, 5, 7, 10, 14, 16, 25)

3. HANSEN SKEW-T FAMILY (asymmetric, heavy tails)
   - hansen_skew_t_nu_{ν}_lambda_{λ}: Hansen (1994) parameterisation
   - λ < 0: left-skewed (crash risk)
   - λ > 0: right-skewed (recovery potential)
   - λ = 0: reduces to symmetric Student-t

4. EVT/GPD FAMILY (tail-only, extreme value theory)
   - evt_gpd_xi_{ξ}_sigma_{σ}: Generalised Pareto for tail extrapolation
   - SUPPORT: tail-only (threshold exceedances)
   - Used for E[loss] estimation, NOT full predictive

5. CONTAMINATED STUDENT-T FAMILY (regime-dependent tails)
   - cst_eps_{ε}_nu_normal_{ν₁}_nu_crisis_{ν₂}: Mixture of two Student-t
   - With probability ε, use crisis ν (heavier tails)
   - SUPPORT: mixture (requires latent state sampling)

===============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
import numpy as np


class ModelFamily(Enum):
    """Model family classification for dispatch logic."""
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    HANSEN_SKEW_T = "hansen_skew_t"
    EVT_GPD = "evt_gpd"
    CONTAMINATED_T = "contaminated_t"
    RV_Q = "rv_q"  # RV-adaptive process noise variants (Tune.md Story 1.3)


class SupportType(Enum):
    """
    Predictive support type — determines how to sample from the model.
    
    FULL: Can generate samples across entire real line (Gaussian, Student-t, etc.)
    TAIL: Only models exceedances above threshold (EVT/GPD)
    MIXTURE: Requires sampling latent component indicators (CST)
    """
    FULL = "full"
    TAIL = "tail"
    MIXTURE = "mixture"


@dataclass(frozen=True)
class ModelSpec:
    """
    Immutable specification for a single distributional model.
    
    This is the CANONICAL contract between tune.py and signals.py.
    """
    name: str                           # Unique model identifier (e.g., "phi_student_t_nu_8")
    family: ModelFamily                 # Model family for dispatch
    support: SupportType                # Predictive support type
    n_params: int                       # Number of free parameters (for BIC)
    param_names: Tuple[str, ...]        # Parameter names in canonical order
    default_params: Dict[str, float]    # Default parameter values
    description: str                    # Human-readable description
    
    # Optional: grid values for tuning (if applicable)
    grid_values: Optional[Dict[str, List[float]]] = None
    
    # Whether this is an "augmentation layer" (applied on top of base models)
    is_augmentation: bool = False
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Check if provided params contain all required parameter names."""
        return all(p in params for p in self.param_names)
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelSpec):
            return False
        return self.name == other.name


# =============================================================================
# MODEL NAME GENERATORS — The ONLY way to construct model names
# =============================================================================
# These functions are the single source of truth for model naming.
# Both tune.py and signals.py MUST use these functions.
# =============================================================================

def make_gaussian_name() -> str:
    """Generate canonical name for Gaussian model."""
    return "kalman_gaussian"


def make_phi_gaussian_name() -> str:
    """Generate canonical name for AR(1) Gaussian model."""
    return "kalman_phi_gaussian"


def make_student_t_name(nu: int) -> str:
    """Generate canonical name for Student-t model with given ν."""
    return f"phi_student_t_nu_{nu}"


def make_student_t_improved_name(nu: int) -> str:
    """Generate canonical name for improved Student-t model with given ν."""
    return f"phi_student_t_improved_nu_{nu}"


def make_student_t_mle_name() -> str:
    """Generate canonical name for continuous-ν Student-t profile model."""
    return "phi_student_t_nu_mle"


def make_student_t_improved_mle_name() -> str:
    """Generate canonical name for improved continuous-ν Student-t profile model."""
    return "phi_student_t_improved_nu_mle"


def make_unified_student_t_name(nu: int) -> str:
    """Generate canonical name for unified Student-t model with given ν seed."""
    return f"phi_student_t_unified_nu_{nu}"


def make_unified_student_t_improved_name(nu: int) -> str:
    """Generate canonical name for improved unified Student-t model with given ν seed."""
    return f"phi_student_t_unified_improved_nu_{nu}"


def make_gaussian_unified_name(phi_mode: bool = False) -> str:
    """Generate canonical name for unified Gaussian models."""
    return "kalman_phi_gaussian_unified" if phi_mode else "kalman_gaussian_unified"


def make_hansen_skew_t_name(nu: float, lambda_: float) -> str:
    """
    Generate canonical name for Hansen Skew-t model.
    
    Args:
        nu: Degrees of freedom (tail heaviness)
        lambda_: Skewness parameter in (-1, 1)
    """
    # Round to avoid floating point issues in name matching
    nu_str = f"{nu:.0f}" if nu == int(nu) else f"{nu:.1f}"
    lambda_str = f"{lambda_:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
    return f"hansen_skew_t_nu_{nu_str}_lambda_{lambda_str}"


def parse_hansen_skew_t_name(name: str) -> Optional[Tuple[float, float]]:
    """
    Parse Hansen Skew-t model name to extract (nu, lambda).
    
    Returns None if name doesn't match expected pattern.
    """
    import re
    # Match pattern: hansen_skew_t_nu_{nu}_lambda_{lambda}
    # Lambda is encoded as p/m for +/- and no decimal point
    pattern = r"hansen_skew_t_nu_(\d+(?:\.\d+)?)?_lambda_([pm]\d+)"
    match = re.match(pattern, name)
    if not match:
        return None
    
    nu = float(match.group(1))
    lambda_encoded = match.group(2)
    
    # Decode lambda: p02 -> +0.2, m03 -> -0.3
    sign = 1.0 if lambda_encoded[0] == 'p' else -1.0
    lambda_val = sign * int(lambda_encoded[1:]) / 100.0
    
    return (nu, lambda_val)


def make_evt_gpd_name(xi: float, sigma: float) -> str:
    """
    Generate canonical name for EVT/GPD model.
    
    Args:
        xi: Shape parameter (tail index)
        sigma: Scale parameter
    """
    xi_str = f"{xi:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
    sigma_str = f"{sigma:.3f}".replace(".", "")
    return f"evt_gpd_xi_{xi_str}_sigma_{sigma_str}"


def make_cst_name(epsilon: float, nu_normal: float, nu_crisis: float) -> str:
    """
    Generate canonical name for Contaminated Student-t model.
    
    Args:
        epsilon: Crisis contamination probability
        nu_normal: ν for normal regime
        nu_crisis: ν for crisis regime (typically smaller = heavier)
    """
    eps_str = f"{epsilon:.2f}".replace(".", "")
    return f"cst_eps_{eps_str}_nu_normal_{nu_normal:.0f}_nu_crisis_{nu_crisis:.0f}"


def make_rv_q_gaussian_name() -> str:
    """Generate canonical name for RV-Q Gaussian model."""
    return "rv_q_gaussian"


def make_rv_q_phi_gaussian_name() -> str:
    """Generate canonical name for RV-Q phi-Gaussian model."""
    return "rv_q_phi_gaussian"


def make_rv_q_student_t_name(nu: int) -> str:
    """Generate canonical name for RV-Q Student-t model with given nu."""
    return f"rv_q_student_t_nu_{nu}"


def is_rv_q_model(name: str) -> bool:
    """Check if model name is an RV-Q variant."""
    return name.startswith("rv_q_")


def is_student_t_family_model(name: str) -> bool:
    """Check if a model name is any registered Student-t family variant."""
    if not name:
        return False
    base_name = name[:-9] if name.endswith("_momentum") else name
    return (
        base_name.startswith("phi_student_t_nu_")
        or base_name.startswith("phi_student_t_improved_nu_")
        or base_name.startswith("phi_student_t_unified_nu_")
        or base_name.startswith("phi_student_t_unified_improved_nu_")
    )


def parse_cst_name(name: str) -> Optional[Tuple[float, float, float]]:
    """Parse CST model name to extract (epsilon, nu_normal, nu_crisis)."""
    import re
    pattern = r"cst_eps_(\d+)_nu_normal_(\d+)_nu_crisis_(\d+)"
    match = re.match(pattern, name)
    if not match:
        return None
    
    epsilon = int(match.group(1)) / 100.0
    nu_normal = float(match.group(2))
    nu_crisis = float(match.group(3))
    
    return (epsilon, nu_normal, nu_crisis)


# =============================================================================
# PARAMETER GRIDS — Canonical grids for tuning
# =============================================================================

# Student-t ν grid (4 BMA flavours: extreme-fat, fat, moderate, near-Gaussian)
# ν=3 added February 2026: EVT analysis shows MSTR ξ=0.302 → ν≈3.3;
# the previous grid's closest (ν=4, kurtosis=6) under-represents power-law tails.
STUDENT_T_NU_GRID = [3, 4, 8, 20]

# Student-t ν grid (adaptive refinement candidates)
STUDENT_T_NU_REFINED_GRID = [5, 7, 10, 14, 16, 25]

# Hansen λ grid (skewness)
HANSEN_LAMBDA_GRID = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]  # λ=0.0 removed (identical to base Student-t)

# Hansen ν grid (same as Student-t for consistency)
HANSEN_NU_GRID = [4, 8, 20]

# CST ε grid (contamination probability)
CST_EPSILON_GRID = [0.05, 0.10, 0.15]

# CST ν pairs (normal, crisis) - crisis has heavier tails
CST_NU_PAIRS = [
    (20, 4),   # Mild: normal near-Gaussian, crisis heavy
    (12, 4),   # Moderate
    (8, 3),    # Severe: both heavy, crisis extreme
]


# =============================================================================
# MODEL SPECIFICATIONS — The complete registry
# =============================================================================

def build_model_registry() -> Dict[str, ModelSpec]:
    """
    Build the complete model registry.
    
    This function constructs all ModelSpec entries.
    Called once at module load time.
    """
    registry: Dict[str, ModelSpec] = {}
    
    # =========================================================================
    # GAUSSIAN FAMILY
    # =========================================================================
    registry[make_gaussian_name()] = ModelSpec(
        name=make_gaussian_name(),
        family=ModelFamily.GAUSSIAN,
        support=SupportType.FULL,
        n_params=2,  # q, c
        param_names=("q", "c"),
        default_params={"q": 1e-6, "c": 1.0},
        description="Standard Kalman filter with Gaussian innovations",
    )
    
    registry[make_phi_gaussian_name()] = ModelSpec(
        name=make_phi_gaussian_name(),
        family=ModelFamily.GAUSSIAN,
        support=SupportType.FULL,
        n_params=3,  # q, c, phi
        param_names=("q", "c", "phi"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95},
        description="AR(1) Kalman filter with Gaussian innovations",
    )

    for phi_mode in (False, True):
        name = make_gaussian_unified_name(phi_mode)
        registry[name] = ModelSpec(
            name=name,
            family=ModelFamily.GAUSSIAN,
            support=SupportType.FULL,
            n_params=8,
            param_names=("q", "c", "phi"),
            default_params={"q": 1e-6, "c": 1.0, "phi": 0.95 if phi_mode else 1.0},
            description=(
                "Unified AR(1) Gaussian calibration pipeline"
                if phi_mode else
                "Unified Gaussian calibration pipeline"
            ),
        )
    
    # =========================================================================
    # STUDENT-T FAMILY (discrete ν grid)
    # =========================================================================
    for nu in STUDENT_T_NU_GRID:
        name = make_student_t_name(nu)
        registry[name] = ModelSpec(
            name=name,
            family=ModelFamily.STUDENT_T,
            support=SupportType.FULL,
            n_params=4,  # q, c, phi, nu (nu is fixed per model)
            param_names=("q", "c", "phi", "nu"),
            default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu": float(nu)},
            description=f"AR(1) Kalman with Student-t(ν={nu}) innovations",
            grid_values={"nu": [float(nu)]},
        )
    
    # Student-t with refined ν (adaptive refinement candidates)
    for nu in STUDENT_T_NU_REFINED_GRID:
        if nu not in STUDENT_T_NU_GRID:  # Don't duplicate
            name = make_student_t_name(nu)
            registry[name] = ModelSpec(
                name=name,
                family=ModelFamily.STUDENT_T,
                support=SupportType.FULL,
                n_params=4,
                param_names=("q", "c", "phi", "nu"),
                default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu": float(nu)},
                description=f"AR(1) Kalman with Student-t(ν={nu}) innovations [refined]",
                grid_values={"nu": [float(nu)]},
            )

    registry[make_student_t_mle_name()] = ModelSpec(
        name=make_student_t_mle_name(),
        family=ModelFamily.STUDENT_T,
        support=SupportType.FULL,
        n_params=4,
        param_names=("q", "c", "phi", "nu"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu": 8.0},
        description="AR(1) Kalman with profile-MLE Student-t degrees of freedom",
    )

    # Improved Student-t implementation: fitted side by side with the canonical
    # implementation so BMA can decide which likelihood/calibration path wins.
    for nu in STUDENT_T_NU_GRID:
        name = make_student_t_improved_name(nu)
        registry[name] = ModelSpec(
            name=name,
            family=ModelFamily.STUDENT_T,
            support=SupportType.FULL,
            n_params=4,
            param_names=("q", "c", "phi", "nu"),
            default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu": float(nu)},
            description=f"Improved AR(1) Kalman with Student-t(ν={nu}) innovations",
            grid_values={"nu": [float(nu)]},
        )

    registry[make_student_t_improved_mle_name()] = ModelSpec(
        name=make_student_t_improved_mle_name(),
        family=ModelFamily.STUDENT_T,
        support=SupportType.FULL,
        n_params=4,
        param_names=("q", "c", "phi", "nu"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu": 8.0},
        description="Improved AR(1) Kalman with profile-MLE Student-t degrees of freedom",
    )

    # Unified Student-t implementations. The original and improved pipelines
    # share the same parameter contract but remain separate hypotheses.
    unified_defaults = {
        "q": 1e-6,
        "c": 1.0,
        "phi": 0.95,
        "nu": 8.0,
        "gamma_vov": 0.3,
        "alpha_asym": 0.0,
        "ms_sensitivity": 2.0,
        "q_stress_ratio": 10.0,
    }
    unified_param_names = (
        "q", "c", "phi", "nu", "gamma_vov", "alpha_asym",
        "ms_sensitivity", "q_stress_ratio",
    )
    for nu in STUDENT_T_NU_GRID:
        for name, description in (
            (
                make_unified_student_t_name(nu),
                f"Unified Student-t pipeline seeded at ν={nu}",
            ),
            (
                make_unified_student_t_improved_name(nu),
                f"Improved unified Student-t pipeline seeded at ν={nu}",
            ),
        ):
            defaults = dict(unified_defaults)
            defaults["nu"] = float(nu)
            registry[name] = ModelSpec(
                name=name,
                family=ModelFamily.STUDENT_T,
                support=SupportType.FULL,
                n_params=14,
                param_names=unified_param_names,
                default_params=defaults,
                description=description,
                grid_values={"nu": [float(nu)]},
            )
    
    # =========================================================================
    # HANSEN SKEW-T FAMILY (augmentation layer)
    # =========================================================================
    # Hansen skew-t is an AUGMENTATION LAYER on top of Student-t
    # It adds asymmetry via λ parameter without changing the base model structure
    for nu in HANSEN_NU_GRID:
        for lambda_ in HANSEN_LAMBDA_GRID:
            name = make_hansen_skew_t_name(nu, lambda_)
            registry[name] = ModelSpec(
                name=name,
                family=ModelFamily.HANSEN_SKEW_T,
                support=SupportType.FULL,
                n_params=5,  # q, c, phi, nu, lambda
                param_names=("q", "c", "phi", "nu", "lambda"),
                default_params={
                    "q": 1e-6, "c": 1.0, "phi": 0.95,
                    "nu": float(nu), "lambda": float(lambda_)
                },
                description=f"Hansen Skew-t(ν={nu}, λ={lambda_:+.1f})",
                grid_values={"nu": [float(nu)], "lambda": [float(lambda_)]},
                is_augmentation=True,
            )
    
    # =========================================================================
    # CONTAMINATED STUDENT-T FAMILY (augmentation layer)
    # =========================================================================
    for epsilon in CST_EPSILON_GRID:
        for nu_normal, nu_crisis in CST_NU_PAIRS:
            name = make_cst_name(epsilon, nu_normal, nu_crisis)
            registry[name] = ModelSpec(
                name=name,
                family=ModelFamily.CONTAMINATED_T,
                support=SupportType.MIXTURE,
                n_params=3,  # epsilon, nu_normal, nu_crisis
                param_names=("cst_epsilon", "cst_nu_normal", "cst_nu_crisis"),
                default_params={
                    "cst_epsilon": float(epsilon),
                    "cst_nu_normal": float(nu_normal),
                    "cst_nu_crisis": float(nu_crisis),
                },
                description=f"Contaminated-t(ε={epsilon:.0%}, ν_n={nu_normal}, ν_c={nu_crisis})",
                grid_values={
                    "cst_epsilon": [float(epsilon)],
                    "cst_nu_normal": [float(nu_normal)],
                    "cst_nu_crisis": [float(nu_crisis)],
                },
                is_augmentation=True,
            )
    
    # =========================================================================
    # EVT/GPD FAMILY (augmentation layer, tail-only)
    # =========================================================================
    # EVT is fitted dynamically based on data, so we register a template
    # The actual ξ and σ come from fitting
    registry["evt_gpd"] = ModelSpec(
        name="evt_gpd",
        family=ModelFamily.EVT_GPD,
        support=SupportType.TAIL,
        n_params=2,  # xi, sigma (threshold is hyperparameter)
        param_names=("evt_xi", "evt_sigma", "evt_threshold"),
        default_params={
            "evt_xi": 0.1,
            "evt_sigma": 0.01,
            "evt_threshold": 0.0,
        },
        description="EVT/GPD tail model for extreme loss estimation",
        is_augmentation=True,
    )
    
    # =========================================================================
    # RV-Q FAMILY (RV-adaptive process noise variants) -- Tune.md Story 1.3
    # =========================================================================
    # RV-Q models compete with static-q and GAS-Q via BMA.
    # q_t = q_base * exp(gamma * delta_log(vol_t^2))
    # Proactive regime adaptation (vs GAS-Q reactive).
    
    # RV-Q Gaussian (no phi)
    registry[make_rv_q_gaussian_name()] = ModelSpec(
        name=make_rv_q_gaussian_name(),
        family=ModelFamily.RV_Q,
        support=SupportType.FULL,
        n_params=4,  # q_base, gamma, c, phi=1 fixed -> effectively 3 + gamma
        param_names=("q_base", "gamma", "c", "phi"),
        default_params={"q_base": 1e-6, "gamma": 1.0, "c": 1.0, "phi": 1.0},
        description="Kalman filter with RV-adaptive process noise (Gaussian)",
    )
    
    # RV-Q phi-Gaussian
    registry[make_rv_q_phi_gaussian_name()] = ModelSpec(
        name=make_rv_q_phi_gaussian_name(),
        family=ModelFamily.RV_Q,
        support=SupportType.FULL,
        n_params=4,  # q_base, gamma, c, phi
        param_names=("q_base", "gamma", "c", "phi"),
        default_params={"q_base": 1e-6, "gamma": 1.0, "c": 1.0, "phi": 0.98},
        description="AR(1) Kalman with RV-adaptive process noise (Gaussian)",
    )
    
    # RV-Q Student-t family (same nu grid as base Student-t)
    for nu in STUDENT_T_NU_GRID:
        name = make_rv_q_student_t_name(nu)
        registry[name] = ModelSpec(
            name=name,
            family=ModelFamily.RV_Q,
            support=SupportType.FULL,
            n_params=5,  # q_base, gamma, c, phi, nu
            param_names=("q_base", "gamma", "c", "phi", "nu"),
            default_params={
                "q_base": 1e-6, "gamma": 1.0, "c": 1.0, "phi": 0.98,
                "nu": float(nu),
            },
            description=f"AR(1) Kalman with RV-adaptive q, Student-t(nu={nu})",
            grid_values={"nu": [float(nu)]},
        )
    
    # =========================================================================
    
    return registry


# Build registry at module load time
MODEL_REGISTRY: Dict[str, ModelSpec] = build_model_registry()


# =============================================================================
# REGISTRY ACCESS FUNCTIONS
# =============================================================================

def get_model_spec(name: str) -> Optional[ModelSpec]:
    """Get ModelSpec by name. Returns None if not found."""
    return MODEL_REGISTRY.get(name)


def get_all_model_names() -> Set[str]:
    """Get set of all registered model names."""
    return set(MODEL_REGISTRY.keys())


def get_base_model_names() -> Set[str]:
    """Get set of base model names (not augmentation layers)."""
    return {name for name, spec in MODEL_REGISTRY.items() if not spec.is_augmentation}


def get_augmentation_model_names() -> Set[str]:
    """Get set of augmentation layer model names."""
    return {name for name, spec in MODEL_REGISTRY.items() if spec.is_augmentation}


def get_models_by_family(family: ModelFamily) -> Dict[str, ModelSpec]:
    """Get all models belonging to a specific family."""
    return {name: spec for name, spec in MODEL_REGISTRY.items() if spec.family == family}


def get_models_by_support(support: SupportType) -> Dict[str, ModelSpec]:
    """Get all models with a specific support type."""
    return {name: spec for name, spec in MODEL_REGISTRY.items() if spec.support == support}


def validate_tuned_models(tuned_model_names: Set[str]) -> Tuple[bool, Set[str], Set[str]]:
    """
    Validate that tuned models match registry.
    
    Args:
        tuned_model_names: Set of model names from tuning output
        
    Returns:
        Tuple of:
        - is_valid: True if all models are in registry
        - missing_from_registry: Models in tuned but not in registry
        - extra_in_registry: Models in registry but not tuned (info only)
    """
    registry_names = get_all_model_names()
    
    missing_from_registry = tuned_model_names - registry_names
    extra_in_registry = registry_names - tuned_model_names
    
    is_valid = len(missing_from_registry) == 0
    
    return is_valid, missing_from_registry, extra_in_registry


def assert_models_synchronised(tuned_model_names: Set[str], context: str = "") -> None:
    """
    Assert that tuned models are synchronised with registry.
    
    Raises AssertionError with detailed message if validation fails.
    This is the sanity check that prevents silent posterior loss.
    
    Args:
        tuned_model_names: Set of model names from tuning output
        context: Optional context string for error message
    """
    is_valid, missing, extra = validate_tuned_models(tuned_model_names)
    
    if not is_valid:
        msg = f"MODEL REGISTRY DESYNC{f' ({context})' if context else ''}\n"
        msg += f"Models in tuned output but NOT in registry: {missing}\n"
        msg += "This will cause silent posterior mass loss!\n"
        msg += "Fix: Add missing models to model_registry.py"
        raise AssertionError(msg)


# =============================================================================
# PREDICTIVE SAMPLING DISPATCH
# =============================================================================

def get_sampler_for_model(spec: ModelSpec) -> str:
    """
    Get the sampling strategy for a model based on its support type.
    
    Returns a strategy identifier that signals.py uses to dispatch sampling.
    """
    if spec.support == SupportType.FULL:
        if spec.family == ModelFamily.GAUSSIAN:
            return "gaussian_mc"
        elif spec.family == ModelFamily.STUDENT_T:
            return "student_t_mc"
        elif spec.family == ModelFamily.HANSEN_SKEW_T:
            return "hansen_skew_t_mc"
        elif spec.family == ModelFamily.RV_Q:
            # RV-Q uses same sampling as base model (Gaussian or Student-t)
            params = spec.param_names
            if "nu" in params:
                return "student_t_mc"
            return "gaussian_mc"
        else:
            return "gaussian_mc"  # fallback
    
    elif spec.support == SupportType.MIXTURE:
        if spec.family == ModelFamily.CONTAMINATED_T:
            return "contaminated_t_mc"
        else:
            return "mixture_mc"  # generic
    
    elif spec.support == SupportType.TAIL:
        # EVT/GPD cannot generate full predictive samples
        # It's used for E[loss] correction, not direct sampling
        return "evt_tail_correction"
    
    return "unknown"


# =============================================================================
# MODEL PARAMETER EXTRACTION
# =============================================================================

def extract_model_params_for_sampling(
    model_name: str,
    fitted_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract and validate parameters needed for sampling from fitted params.
    
    This ensures the parameter interface between tune.py and signals.py is clean.
    
    Args:
        model_name: Canonical model name
        fitted_params: Parameters from tuning output
        
    Returns:
        Dictionary with standardised parameter names for sampling
    """
    spec = get_model_spec(model_name)
    if spec is None:
        # Unknown model - return fitted params as-is with warning flag
        return {"_unknown_model": True, **fitted_params}
    
    result = {"_model_spec": spec}
    
    # Extract common parameters
    result["q"] = fitted_params.get("q", spec.default_params.get("q", 1e-6))
    result["c"] = fitted_params.get("c", spec.default_params.get("c", 1.0))
    result["phi"] = fitted_params.get("phi", spec.default_params.get("phi", 1.0))
    
    # Family-specific parameters
    if spec.family == ModelFamily.STUDENT_T:
        result["nu"] = fitted_params.get("nu", spec.default_params.get("nu"))
        
    elif spec.family == ModelFamily.HANSEN_SKEW_T:
        result["nu"] = fitted_params.get("nu", spec.default_params.get("nu"))
        result["hansen_lambda"] = fitted_params.get("lambda", fitted_params.get("hansen_lambda", spec.default_params.get("lambda", 0.0)))
        
    elif spec.family == ModelFamily.CONTAMINATED_T:
        result["cst_epsilon"] = fitted_params.get("cst_epsilon", fitted_params.get("epsilon", spec.default_params.get("cst_epsilon")))
        result["cst_nu_normal"] = fitted_params.get("cst_nu_normal", fitted_params.get("nu_normal", spec.default_params.get("cst_nu_normal")))
        result["cst_nu_crisis"] = fitted_params.get("cst_nu_crisis", fitted_params.get("nu_crisis", spec.default_params.get("cst_nu_crisis")))
        
    elif spec.family == ModelFamily.EVT_GPD:
        result["evt_xi"] = fitted_params.get("evt_xi", fitted_params.get("xi", spec.default_params.get("evt_xi")))
        result["evt_sigma"] = fitted_params.get("evt_sigma", fitted_params.get("sigma", spec.default_params.get("evt_sigma")))
        result["evt_threshold"] = fitted_params.get("evt_threshold", fitted_params.get("threshold", spec.default_params.get("evt_threshold")))
    
    elif spec.family == ModelFamily.RV_Q:
        result["q_base"] = fitted_params.get("q_base", spec.default_params.get("q_base", 1e-6))
        result["gamma"] = fitted_params.get("gamma", spec.default_params.get("gamma", 1.0))
        if "nu" in spec.param_names:
            result["nu"] = fitted_params.get("nu", spec.default_params.get("nu"))

    passthrough_params = (
        # Unified Student-t and Gaussian calibration transport.
        "nu_base", "gamma_vov", "alpha_asym", "k_asym",
        "ms_sensitivity", "ms_ewm_lambda", "q_stress_ratio", "vov_damping",
        "variance_inflation", "mu_drift", "risk_premium_sensitivity",
        "skew_score_sensitivity", "skew_persistence",
        "garch_omega", "garch_alpha", "garch_beta", "garch_leverage",
        "garch_unconditional_var", "rough_hurst",
        "jump_intensity", "jump_variance", "jump_sensitivity", "jump_mean",
        "crps_ewm_lambda", "crps_sigma_shrinkage",
        "rho_leverage", "kappa_mean_rev", "theta_long_var",
        "sigma_eta", "t_df_asym", "regime_switch_prob",
        "garch_kalman_weight", "q_vol_coupling",
        "loc_bias_var_coeff", "loc_bias_drift_coeff",
        "leverage_dynamic_decay", "liq_stress_coeff", "entropy_sigma_lambda",
        "chisq_ewm_lambda", "pit_var_lambda", "pit_var_dz_lo", "pit_var_dz_hi",
        "calibrated_gw", "calibrated_nu_pit", "calibrated_nu_crps",
        "calibrated_beta_probit_corr", "calibrated_lambda_rho",
        "momentum_weight", "gas_q_omega", "gas_q_alpha", "gas_q_beta",
        "hansen_activated", "hansen_lambda",
        "cst_activated", "cst_nu_crisis", "cst_epsilon",
    )
    for key in passthrough_params:
        if key not in result and key in fitted_params:
            value = fitted_params[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, bool, str)) or value is None:
                result[key] = value
    
    return result


# =============================================================================
# TUNING GRID GENERATION
# =============================================================================

def get_base_models_for_tuning() -> List[str]:
    """
    Get ordered list of base model names for tuning.
    
    This is what tune.py should iterate over.
    Augmentation layers are applied separately after base model selection.
    """
    # Order: Gaussian → Student-t (by ν)
    models = []
    
    # Gaussian family first
    models.append(make_gaussian_name())
    models.append(make_phi_gaussian_name())
    models.append(make_gaussian_unified_name(False))
    models.append(make_gaussian_unified_name(True))
    
    # Student-t family (original grid)
    for nu in STUDENT_T_NU_GRID:
        models.append(make_student_t_name(nu))
        models.append(make_student_t_improved_name(nu))
        models.append(make_unified_student_t_name(nu))
        models.append(make_unified_student_t_improved_name(nu))
    models.append(make_student_t_mle_name())
    models.append(make_student_t_improved_mle_name())
    
    return models


def get_augmentation_layers_for_tuning() -> Dict[str, List[str]]:
    """
    Get augmentation layers organised by type.
    
    Returns dict mapping augmentation type to list of model names.
    tune.py applies these conditionally after base model selection.
    """
    return {
        "hansen_skew_t": [
            make_hansen_skew_t_name(nu, lambda_)
            for nu in HANSEN_NU_GRID
            for lambda_ in HANSEN_LAMBDA_GRID
            if lambda_ != 0.0  # Skip symmetric (redundant with base Student-t)
        ],
        "contaminated_t": [
            make_cst_name(eps, nu_n, nu_c)
            for eps in CST_EPSILON_GRID
            for nu_n, nu_c in CST_NU_PAIRS
        ],
        "evt_gpd": ["evt_gpd"],  # Single template, fitted dynamically
    }


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_registry_summary() -> None:
    """Print a summary of the model registry for debugging."""
    print("\n" + "=" * 60)
    print("MODEL REGISTRY SUMMARY")
    print("=" * 60)
    
    by_family = {}
    for name, spec in MODEL_REGISTRY.items():
        family = spec.family.value
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(spec)
    
    for family, specs in sorted(by_family.items()):
        print(f"\n{family.upper()} ({len(specs)} models)")
        print("-" * 40)
        for spec in specs[:5]:  # Show first 5
            aug_marker = " [AUG]" if spec.is_augmentation else ""
            print(f"  {spec.name}{aug_marker}")
        if len(specs) > 5:
            print(f"  ... and {len(specs) - 5} more")
    
    print(f"\nTotal: {len(MODEL_REGISTRY)} models")
    print(f"  Base models: {len(get_base_model_names())}")
    print(f"  Augmentation layers: {len(get_augmentation_model_names())}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Print registry summary when run directly
    print_registry_summary()
