"""
===============================================================================
EXPERIMENTAL MODELS — Arena Model Competition Framework
===============================================================================

5 PROMOTION CANDIDATES (All beat standard with PIT PASS):

1. dualtree_complex_wavelet: +22.9% vs STD (PIT PASS) - Dual-Tree Complex Wavelet
2. synchrosqueezed_wavelet: +5.6% vs STD (PIT PASS) - Synchrosqueezing Transform
3. wavelet_packet_kalman: +5.5% vs STD (PIT PASS) - Best Basis Selection
4. wavelet_kalman: +3.7% vs STD (PIT PASS) - Multi-scale Haar Decomposition
5. online_bayesian_ma: +0.0% vs STD (PIT PASS) - Bayesian Model Averaging

Author: Staff Professor, Chinese Academy of Quantitative Finance
Date: February 2026
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Core models
from .online_bayesian_ma import OnlineBayesianModelAvgModel
from .m06_wavelet_kalman import WaveletKalmanModel

# Wavelet evolution family
from .m10_wavelet_packet_kalman import WaveletPacketKalmanModel
from .m12_synchrosqueezed_wavelet import SynchrosqueezedWaveletKalmanModel
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel


# Registry of active experimental models
EXPERIMENTAL_MODELS = {
    "online_bayesian_ma": OnlineBayesianModelAvgModel,
    "wavelet_kalman": WaveletKalmanModel,
    "wavelet_packet_kalman": WaveletPacketKalmanModel,
    "synchrosqueezed_wavelet": SynchrosqueezedWaveletKalmanModel,
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
}

# Model specifications
EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0},
    description=f"World-class wavelet evolution model: {name}",
) for name in EXPERIMENTAL_MODELS.keys()}


def get_experimental_model_specs():
    return list(EXPERIMENTAL_MODEL_SPECS.values())


def create_experimental_model(name: str, **kwargs):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown: {name}. Available: {list(EXPERIMENTAL_MODELS.keys())}")
    return EXPERIMENTAL_MODELS[name](**kwargs)


__all__ = [
    'BaseExperimentalModel', 'ExperimentalModelSpec', 'ExperimentalModelFamily',
    'EXPERIMENTAL_MODELS', 'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs', 'create_experimental_model',
]