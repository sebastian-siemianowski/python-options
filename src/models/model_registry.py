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

4. NIG FAMILY (semi-heavy tails, asymmetric, Lévy process compatible)
   - phi_nig_alpha_{α}_beta_{β}: Normal-Inverse Gaussian
   - α controls tail heaviness
   - β controls asymmetry

5. GMM FAMILY (bimodal, regime mixture)
   - gmm_k2: 2-component Gaussian mixture

6. EVT/GPD FAMILY (tail-only, extreme value theory)
   - evt_gpd_xi_{ξ}_sigma_{σ}: Generalised Pareto for tail extrapolation
   - SUPPORT: tail-only (threshold exceedances)
   - Used for E[loss] estimation, NOT full predictive

7. CONTAMINATED STUDENT-T FAMILY (regime-dependent tails)
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
    NIG = "nig"
    GMM = "gmm"
    EVT_GPD = "evt_gpd"
    CONTAMINATED_T = "contaminated_t"
    AIGF_NF = "aigf_nf"  # Adaptive Implicit Generative Filter (Normalizing Flow)


class SupportType(Enum):
    """
    Predictive support type — determines how to sample from the model.
    
    FULL: Can generate samples across entire real line (Gaussian, Student-t, etc.)
    TAIL: Only models exceedances above threshold (EVT/GPD)
    MIXTURE: Requires sampling latent component indicators (CST, GMM)
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


def make_nig_name(alpha: float, beta: float) -> str:
    """
    Generate canonical name for NIG model.
    
    Args:
        alpha: Tail heaviness (smaller = heavier)
        beta: Asymmetry (negative = left-skewed)
    """
    alpha_str = f"{alpha:.1f}".replace(".", "")
    beta_str = f"{beta:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
    return f"phi_nig_alpha_{alpha_str}_beta_{beta_str}"


def parse_nig_name(name: str) -> Optional[Tuple[float, float]]:
    """Parse NIG model name to extract (alpha, beta)."""
    import re
    pattern = r"phi_nig_alpha_(\d+)_beta_([pm]\d+)"
    match = re.match(pattern, name)
    if not match:
        return None
    
    alpha = int(match.group(1)) / 10.0
    beta_encoded = match.group(2)
    sign = 1.0 if beta_encoded[0] == 'p' else -1.0
    beta = sign * int(beta_encoded[1:]) / 100.0
    
    return (alpha, beta)


def make_gmm_name(n_components: int = 2) -> str:
    """Generate canonical name for Gaussian Mixture Model."""
    return f"gmm_k{n_components}"


def make_aigf_nf_name() -> str:
    """Generate canonical name for AIGF-NF (Adaptive Implicit Generative Filter)."""
    return "aigf_nf"


def is_aigf_nf_model(name: str) -> bool:
    """Check if a model name corresponds to AIGF-NF."""
    if not name:
        return False
    return name.lower() == "aigf_nf"


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

# Student-t ν grid (original discrete grid)
STUDENT_T_NU_GRID = [4, 6, 8, 12, 20]

# Student-t ν grid (adaptive refinement candidates)
STUDENT_T_NU_REFINED_GRID = [3, 5, 7, 10, 14, 16, 25]

# Hansen λ grid (skewness)
HANSEN_LAMBDA_GRID = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

# Hansen ν grid (same as Student-t for consistency)
HANSEN_NU_GRID = [4, 6, 8, 12, 20]

# NIG α grid (tail heaviness)
NIG_ALPHA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]

# NIG β grid (asymmetry)
NIG_BETA_GRID = [-0.3, -0.15, 0.0, 0.15, 0.3]

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
    # NIG FAMILY
    # =========================================================================
    for alpha in NIG_ALPHA_GRID:
        for beta in NIG_BETA_GRID:
            # Constraint: |β| < α
            if abs(beta) >= alpha:
                continue
            name = make_nig_name(alpha, beta)
            registry[name] = ModelSpec(
                name=name,
                family=ModelFamily.NIG,
                support=SupportType.FULL,
                n_params=5,  # q, c, phi, nig_alpha, nig_beta (delta derived)
                param_names=("q", "c", "phi", "nig_alpha", "nig_beta", "nig_delta"),
                default_params={
                    "q": 1e-6, "c": 1.0, "phi": 0.95,
                    "nig_alpha": float(alpha),
                    "nig_beta": float(beta),
                    "nig_delta": 1.0,
                },
                description=f"NIG(α={alpha:.1f}, β={beta:+.2f})",
                grid_values={"nig_alpha": [float(alpha)], "nig_beta": [float(beta)]},
            )
    
    # =========================================================================
    # GMM FAMILY
    # =========================================================================
    registry[make_gmm_name(2)] = ModelSpec(
        name=make_gmm_name(2),
        family=ModelFamily.GMM,
        support=SupportType.MIXTURE,
        n_params=5,  # π, μ1, μ2, σ1, σ2
        param_names=("weights", "means", "variances"),
        default_params={
            "weights": [0.7, 0.3],
            "means": [0.0, -0.02],
            "variances": [0.0001, 0.0004],
        },
        description="2-component Gaussian Mixture (momentum/reversal)",
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
    # AIGF-NF FAMILY — Adaptive Implicit Generative Filter (Normalizing Flow)
    # =========================================================================
    # AIGF-NF is a bounded, non-parametric belief model per spec v1.0.
    # It competes in BMA via log-likelihood.
    # Flow parameters θ are frozen; only latent z_t evolves online.
    registry[make_aigf_nf_name()] = ModelSpec(
        name=make_aigf_nf_name(),
        family=ModelFamily.AIGF_NF,
        support=SupportType.FULL,
        n_params=10,  # latent_dim (8) + EWMA state (2)
        param_names=(
            "latent_z", "latent_dim", "ewma_lambda", "clip_threshold",
            "step_size_alpha", "trust_radius", "predictive_mean", "predictive_std",
            "tail_heaviness", "novelty_score"
        ),
        default_params={
            "latent_dim": 8,
            "ewma_lambda": 0.95,
            "clip_threshold": 5.0,
            "step_size_alpha": 0.01,
            "trust_radius": 3.0,
        },
        description="AIGF-NF: Bounded non-parametric belief model via normalizing flow",
        is_augmentation=False,
    )
    
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
        elif spec.family == ModelFamily.NIG:
            return "nig_mc"
        elif spec.family == ModelFamily.AIGF_NF:
            return "aigf_nf_mc"  # AIGF-NF has its own sampling via flow
        else:
            return "gaussian_mc"  # fallback
    
    elif spec.support == SupportType.MIXTURE:
        if spec.family == ModelFamily.GMM:
            return "gmm_mc"
        elif spec.family == ModelFamily.CONTAMINATED_T:
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
        
    elif spec.family == ModelFamily.NIG:
        result["nig_alpha"] = fitted_params.get("nig_alpha", spec.default_params.get("nig_alpha"))
        result["nig_beta"] = fitted_params.get("nig_beta", spec.default_params.get("nig_beta"))
        result["nig_delta"] = fitted_params.get("nig_delta", spec.default_params.get("nig_delta", 1.0))
        
    elif spec.family == ModelFamily.GMM:
        result["gmm_weights"] = fitted_params.get("weights", spec.default_params.get("weights"))
        result["gmm_means"] = fitted_params.get("means", spec.default_params.get("means"))
        result["gmm_variances"] = fitted_params.get("variances", spec.default_params.get("variances"))
        
    elif spec.family == ModelFamily.CONTAMINATED_T:
        result["cst_epsilon"] = fitted_params.get("cst_epsilon", fitted_params.get("epsilon", spec.default_params.get("cst_epsilon")))
        result["cst_nu_normal"] = fitted_params.get("cst_nu_normal", fitted_params.get("nu_normal", spec.default_params.get("cst_nu_normal")))
        result["cst_nu_crisis"] = fitted_params.get("cst_nu_crisis", fitted_params.get("nu_crisis", spec.default_params.get("cst_nu_crisis")))
        
    elif spec.family == ModelFamily.EVT_GPD:
        result["evt_xi"] = fitted_params.get("evt_xi", fitted_params.get("xi", spec.default_params.get("evt_xi")))
        result["evt_sigma"] = fitted_params.get("evt_sigma", fitted_params.get("sigma", spec.default_params.get("evt_sigma")))
        result["evt_threshold"] = fitted_params.get("evt_threshold", fitted_params.get("threshold", spec.default_params.get("evt_threshold")))
    
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
    
    # Student-t family (original grid)
    for nu in STUDENT_T_NU_GRID:
        models.append(make_student_t_name(nu))
    
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
