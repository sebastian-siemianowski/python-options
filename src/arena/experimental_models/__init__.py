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

EXPERIMENTAL MODELS (implementing panel recommendations):
    
    From Professor Wei Chen (Citadel Asia):
    - asymmetric_loss: Downside-weighted prediction (Score: 78/100)
    - ensemble_distillation: Knowledge transfer from standards (Score: 82/100)
    
    From Professor Liu Xiaoming (D.E. Shaw):
    - pit_constrained: Calibration-guaranteed optimization (Score: 86/100)
    
    From Professor Zhang Yifan (Millennium):
    - multi_horizon: Temporal consistency across horizons (Score: 86/100)
    
    Original experimental models:
    - momentum_student_t_v2: Adaptive tail coupling
    - momentum_student_t_regime_coupled: Regime-aware tail dynamics

PROMOTION CRITERIA:
    To graduate from arena to production, an experimental model must:
    1. Beat average standard model CRPS by >5%
    2. Pass PIT calibration on all benchmark symbols
    3. Show consistent performance across cap categories
    4. Panel review and approval

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Original experimental models
from .momentum_student_t_v2 import MomentumStudentTV2
from .momentum_student_t_regime_coupled import MomentumStudentTRegimeCoupled

# Panel recommendation models
from .asymmetric_loss import AsymmetricLossModel
from .multi_horizon import MultiHorizonModel
from .ensemble_distillation import EnsembleDistillationModel
from .pit_constrained import PITConstrainedModel


# Registry of all experimental models
EXPERIMENTAL_MODELS = {
    # Original models
    "momentum_student_t_v2": MomentumStudentTV2,
    "momentum_student_t_regime_coupled": MomentumStudentTRegimeCoupled,
    # Panel recommendation models
    "asymmetric_loss": AsymmetricLossModel,
    "multi_horizon": MultiHorizonModel,
    "ensemble_distillation": EnsembleDistillationModel,
    "pit_constrained": PITConstrainedModel,
}

# Model specifications
EXPERIMENTAL_MODEL_SPECS = {
    "momentum_student_t_v2": ExperimentalModelSpec(
        name="momentum_student_t_v2",
        family=ExperimentalModelFamily.STUDENT_T_V2,
        n_params=5,
        param_names=("q", "c", "phi", "nu_base", "alpha"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu_base": 6.0, "alpha": 0.5},
        description="Adaptive tail Student-t with momentum-coupled nu",
    ),
    "momentum_student_t_regime_coupled": ExperimentalModelSpec(
        name="momentum_student_t_regime_coupled",
        family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=3,
        param_names=("q", "c", "phi"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95},
        description="Regime-coupled Student-t with preset nu per regime",
    ),
    "asymmetric_loss": ExperimentalModelSpec(
        name="asymmetric_loss",
        family=ExperimentalModelFamily.TAIL_SWITCHING,
        n_params=4,
        param_names=("q", "c", "phi", "alpha"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "alpha": 2.0},
        description="Asymmetric loss with heavier downside penalty (Wei Chen)",
    ),
    "multi_horizon": ExperimentalModelSpec(
        name="multi_horizon",
        family=ExperimentalModelFamily.STUDENT_T_V2,
        n_params=3,
        param_names=("q", "c", "phi"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95},
        description="Multi-horizon objective (1d, 5d, 20d) for temporal consistency (Zhang Yifan)",
    ),
    "ensemble_distillation": ExperimentalModelSpec(
        name="ensemble_distillation",
        family=ExperimentalModelFamily.STUDENT_T_V2,
        n_params=4,
        param_names=("q", "c", "phi", "lambda_reg"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "lambda_reg": 1.0},
        description="Knowledge distillation from standard models (Wei Chen)",
    ),
    "pit_constrained": ExperimentalModelSpec(
        name="pit_constrained",
        family=ExperimentalModelFamily.STUDENT_T_V2,
        n_params=4,
        param_names=("q", "c", "phi", "nu"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.95, "nu": 8.0},
        description="PIT-constrained optimization for guaranteed calibration (Liu Xiaoming)",
    ),
}


def get_experimental_model_specs():
    """Get all experimental model specifications."""
    return list(EXPERIMENTAL_MODEL_SPECS.values())


def create_experimental_model(name: str, **kwargs):
    """
    Create an instance of an experimental model.
    
    Args:
        name: Model name from EXPERIMENTAL_MODELS
        **kwargs: Arguments passed to model constructor
        
    Returns:
        Instantiated model
    """
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown experimental model: {name}. Available: {list(EXPERIMENTAL_MODELS.keys())}")
    return EXPERIMENTAL_MODELS[name](**kwargs)


__all__ = [
    # Base classes
    'BaseExperimentalModel',
    'ExperimentalModelSpec',
    'ExperimentalModelFamily',
    # Original models
    'MomentumStudentTV2',
    'MomentumStudentTRegimeCoupled',
    # Panel recommendation models
    'AsymmetricLossModel',
    'MultiHorizonModel',
    'EnsembleDistillationModel',
    'PITConstrainedModel',
    # Registry
    'EXPERIMENTAL_MODELS',
    'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs',
    'create_experimental_model',
]
