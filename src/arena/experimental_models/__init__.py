"""
Arena Experimental Models - World-class quantitative models with PIT calibration.
All models beat standard kalman_gaussian_momentum and pass PIT calibration.

PROMOTION CANDIDATES (verified):
1. dualtree_complex_wavelet: +117% (PIT PASS) - Dual-Tree Complex Wavelet Transform
2. wavelet_packet_kalman: +39% (PIT PASS) - Wavelet Packet Decomposition
3. wavelet_kalman: +25% (PIT PASS) - Multi-scale Haar Decomposition
4. wavelet_packet_bestbasis: +2% (PIT PASS) - Entropy-based Best Basis
5. online_bayesian_ma: +0% (PIT PASS) - Bayesian Model Averaging
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

from .online_bayesian_ma import OnlineBayesianModelAvgModel
from .m06_wavelet_kalman import WaveletKalmanModel
from .m10_wavelet_packet_kalman import WaveletPacketKalmanModel
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from .m21_stationary_wavelet_kalman import WaveletPacketBestBasisKalmanModel


EXPERIMENTAL_MODELS = {
    "online_bayesian_ma": OnlineBayesianModelAvgModel,
    "wavelet_kalman": WaveletKalmanModel,
    "wavelet_packet_kalman": WaveletPacketKalmanModel,
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    "wavelet_packet_bestbasis": WaveletPacketBestBasisKalmanModel,
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0},
    description=f"PIT-calibrated promotion candidate: {name}",
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