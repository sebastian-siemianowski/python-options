"""
===============================================================================
EXPERIMENTAL MODELS — Arena Model Competition Framework
===============================================================================

This package contains experimental distributional models that compete against
production baselines in the arena. Each model is a standalone file.

Models here are NOT used in production tuning (make tune) — only in arena
competition (make arena-tune).

STANDARD MODELS (Baselines to beat):
    - kalman_gaussian_momentum
    - kalman_phi_gaussian_momentum
    - phi_student_t_nu_{4,6,8,12,20}_momentum

EXPERIMENTAL MODELS:
    - momentum_student_t_v2: Upgraded Student-t with adaptive tail coupling
    - momentum_student_t_regime_coupled: Regime-aware tail dynamics

PROMOTION CRITERIA:
    To graduate from arena to production, an experimental model must:
    1. Beat average standard model score by >5%
    2. Pass PIT calibration on all benchmark symbols
    3. Show consistent performance across cap categories
    4. Panel review and approval

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from .momentum_student_t_v2 import MomentumStudentTV2
from .momentum_student_t_regime_coupled import MomentumStudentTRegimeCoupled
from .base import ExperimentalModelSpec, ExperimentalModelFamily

# Registry of all experimental models
EXPERIMENTAL_MODELS = {
    "momentum_student_t_v2": MomentumStudentTV2,
    "momentum_student_t_regime_coupled": MomentumStudentTRegimeCoupled,
}

# Model specifications
EXPERIMENTAL_MODEL_SPECS = {
    "momentum_student_t_v2": ExperimentalModelSpec(
        name="momentum_student_t_v2",
        family=ExperimentalModelFamily.STUDENT_T_V2,
        n_params=5,
        param_names=("q", "c", "phi", "nu_base", "alpha"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu_base": 6.0, "alpha": 0.5},
        description="Adaptive tail Student-t with momentum-coupled ν",
    ),
    "momentum_student_t_regime_coupled": ExperimentalModelSpec(
        name="momentum_student_t_regime_coupled",
        family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=3,
        param_names=("q", "c", "phi"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95},
        description="Regime-coupled Student-t with preset ν per regime",
    ),
}


def get_experimental_model_specs():
    """Get all experimental model specifications."""
    return list(EXPERIMENTAL_MODEL_SPECS.values())


def create_experimental_model(name: str):
    """Create an instance of an experimental model."""
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown experimental model: {name}. Available: {list(EXPERIMENTAL_MODELS.keys())}")
    return EXPERIMENTAL_MODELS[name]()


__all__ = [
    'EXPERIMENTAL_MODELS',
    'EXPERIMENTAL_MODEL_SPECS',
    'ExperimentalModelSpec',
    'ExperimentalModelFamily',
    'MomentumStudentTV2',
    'MomentumStudentTRegimeCoupled',
    'get_experimental_model_specs',
    'create_experimental_model',
]
