"""
===============================================================================
EXPERIMENTAL MODELS — Arena Model Competition Framework
===============================================================================

Staff Professor Implementation — Chinese Academy of Quantitative Finance
Panel: Chen Wei (Tsinghua), Liu Xiaoming (Peking), Zhang Yifan (Fudan)

PROMOTION CANDIDATES:
    wavelet_kalman: Score 0.7250 | BIC -13083 | CRPS 0.0071 | PIT PASS | +117.6% vs STD
    online_bayesian_ma: Score 0.3340 | BIC -7729 | CRPS 0.0072 | PIT PASS | +0.2% vs STD

These are the top 2 experimental models that have passed all criteria.

Author: Staff Professor, Chinese Academy of Quantitative Finance
Date: February 2026
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Top performing models
from .online_bayesian_ma import OnlineBayesianModelAvgModel
from .m06_wavelet_kalman import WaveletKalmanModel


# Registry of active experimental models
EXPERIMENTAL_MODELS = {
    "online_bayesian_ma": OnlineBayesianModelAvgModel,
    "wavelet_kalman": WaveletKalmanModel,
}

# Model specifications
EXPERIMENTAL_MODEL_SPECS = {
    "online_bayesian_ma": ExperimentalModelSpec(
        name="online_bayesian_ma",
        family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=3,
        param_names=("forgetting_factor", "model_weights"),
        default_params={"forgetting_factor": 0.99},
        description="Online Bayesian model averaging with dynamic weights (+0.2% vs std)",
    ),
    "wavelet_kalman": ExperimentalModelSpec(
        name="wavelet_kalman",
        family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=9,
        param_names=("q", "c", "phi"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0},
        description="Multi-scale wavelet Kalman filter (+117.6% vs std) - PROMOTION CANDIDATE",
    ),
}


def get_experimental_model_specs():
    return list(EXPERIMENTAL_MODEL_SPECS.values())


def create_experimental_model(name: str, **kwargs):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown: {name}. Available: {list(EXPERIMENTAL_MODELS.keys())}")
    return EXPERIMENTAL_MODELS[name](**kwargs)


__all__ = [
    'BaseExperimentalModel', 'ExperimentalModelSpec', 'ExperimentalModelFamily',
    'OnlineBayesianModelAvgModel', 'WaveletKalmanModel',
    'EXPERIMENTAL_MODELS', 'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs', 'create_experimental_model',
]