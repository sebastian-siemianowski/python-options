"""
Arena Experimental Models - Top DTCWT models that pass all hard gates.
10,000ms hard time limit enforced.

Promotion Hard Gates:
- CSS >= 0.65 (Calibration Stability Under Stress)
- FEC >= 0.75 (Forecast Entropy Consistency)

Current Champions (passing all gates):
- dtcwt_deep_triple_boost: FINAL 63.94, CSS 0.75, FEC 0.81
- dtcwt_deep_double_boost: FINAL 59.61, CSS 0.75, FEC 0.81
- dualtree_complex_wavelet: FINAL 55.93, CSS 0.77, FEC 0.81
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Champion model
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel

# Gen2 DTCWT evolution models (CSS >= 0.65 and FEC >= 0.75 only)
from .m80_dtcwt_gen2 import (
    DTCWTDeepDoubleBoostModel,
    DTCWTDeepTripleBoostModel,
)


EXPERIMENTAL_MODELS = {
    # Original champion (CSS 0.77, FEC 0.81)
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    # Gen2 DTCWT evolution (passing CSS >= 0.65 and FEC >= 0.75)
    "dtcwt_deep_double_boost": DTCWTDeepDoubleBoostModel,  # CSS 0.75, FEC 0.81
    "dtcwt_deep_triple_boost": DTCWTDeepTripleBoostModel,  # CSS 0.75, FEC 0.81
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0},
    description=f"World-class wavelet model: {name}",
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