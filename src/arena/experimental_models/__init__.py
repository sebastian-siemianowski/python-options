"""
===============================================================================
EXPERIMENTAL MODELS — Arena Model Competition Framework
===============================================================================

This package contains experimental distributional models that compete against
production baselines in the arena. Each model is a standalone file.

STANDARD MODELS (Baselines to beat):
    - kalman_gaussian_momentum
    - kalman_phi_gaussian_momentum
    - phi_student_t_nu_{4,6,8,12,20}_momentum

EXPERIMENTAL MODELS (Active):
    - online_bayesian_ma: Dynamic model averaging (Liu Xiaoming, 94/100) — +5.7% vs std
      STATUS: PROMOTION CANDIDATE

PROMOTION CRITERIA:
    To graduate from arena to production, an experimental model must:
    1. Beat average standard model CRPS by >5%
    2. Pass PIT calibration on all benchmark symbols
    3. Show consistent performance across cap categories
    4. Panel review and approval

Author: Chinese Staff Professor Panel
Date: February 2026
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Active experimental models
from .online_bayesian_ma import OnlineBayesianModelAvgModel


# Registry of active experimental models
EXPERIMENTAL_MODELS = {
    "online_bayesian_ma": OnlineBayesianModelAvgModel,
}

# Model specifications
EXPERIMENTAL_MODEL_SPECS = {
    "online_bayesian_ma": ExperimentalModelSpec(
        name="online_bayesian_ma",
        family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=3,
        param_names=("forgetting_factor", "model_weights"),
        default_params={"forgetting_factor": 0.99},
        description="Online Bayesian model averaging with dynamic weights (Liu Xiaoming, 94/100)",
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
    # Active models
    'OnlineBayesianModelAvgModel',
    # Registry
    'EXPERIMENTAL_MODELS',
    'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs',
    'create_experimental_model',
]