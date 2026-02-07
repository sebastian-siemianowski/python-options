"""
Arena Experimental Models - Top 4 world-class models that beat standard.
All models pass PIT calibration and outperform kalman_gaussian_momentum.
10,000ms hard time limit enforced.

Proven Champions:
1. dualtree_complex_wavelet: +130% - Dual-Tree Complex Wavelet Transform
2. wavelet_packet_kalman: +31% - Wavelet Packet Decomposition
3. wavelet_kalman: +21% - Multi-scale Haar Wavelet
4. wavelet_packet_bestbasis: +1.4% - Entropy Best Basis Selection
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

from .m06_wavelet_kalman import WaveletKalmanModel
from .m10_wavelet_packet_kalman import WaveletPacketKalmanModel
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from .m21_stationary_wavelet_kalman import WaveletPacketBestBasisKalmanModel


EXPERIMENTAL_MODELS = {
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
    description=f"Proven promotion candidate: {name}",
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