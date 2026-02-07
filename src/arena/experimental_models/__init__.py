"""
===============================================================================
EXPERIMENTAL MODELS — Arena Model Competition Framework
===============================================================================

Staff Professor Implementation — Chinese Academy of Quantitative Finance

STANDARD MODELS (Baselines to beat):
    #1 kalman_gaussian_momentum:    Score 0.6970 | BIC -5658 | CRPS 0.0182 | PIT PASS
    #2 phi_student_t_nu_4_momentum: Score 0.5456 | BIC -5593 | CRPS 0.0203 | PIT 25%
    #3-7 phi variants:              Score ~0.52  | BIC ~-5640 | CRPS ~0.02 | PIT 0%

CURRENT CHAMPION (PROMOTION CANDIDATE):
    online_bayesian_ma: Score 0.6992 | BIC -5665 | CRPS 0.0186 | PIT PASS | +0.3% vs std

Model achieves:
    - Best BIC (-5665 vs -5658 standard)
    - Competitive CRPS (0.0186)
    - 100% PIT calibration pass
    - Consistent #1 ranking across all tests

SCORING METRICS:
    - BIC (Bayesian Information Criterion) — model complexity penalty
    - CRPS (Continuous Ranked Probability Score) — calibration + sharpness
    - Hyvärinen Score — robustness to misspecification
    - PIT Calibration — distributional correctness

PROMOTION CRITERIA:
    Beat best standard by >5%, 100% PIT pass, consistent across categories.

Author: Staff Professor, Chinese Academy of Quantitative Finance
Date: February 2026
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Current champion and only active experimental model
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
    # Current champion
    'OnlineBayesianModelAvgModel',
    # Registry
    'EXPERIMENTAL_MODELS',
    'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs',
    'create_experimental_model',
]