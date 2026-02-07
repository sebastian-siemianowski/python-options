"""
Arena Experimental Models - Top DTCWT models + Next-Gen evolutions.
10,000ms hard time limit enforced.

Current Champions (sorted by Final Score):
- dtcwt_deep_triple_boost: FINAL 67.11 (+10.6)
- dtcwt_deep_double_boost: FINAL 63.00 (+6.5)
- dtcwt_quad_boost: FINAL 61.10 (+4.6)
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Champion model
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel

# Gen2 DTCWT evolution models
from .m80_dtcwt_gen2 import (
    DTCWTTripleBoostModel,
    DTCWTQuadBoostModel,
    DTCWTDeepDoubleBoostModel,
    DTCWTDeepTripleBoostModel,
)


EXPERIMENTAL_MODELS = {
    # Original champion (+1.6)
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    # Gen2 DTCWT evolution (all positive vs STD)
    "dtcwt_triple_boost": DTCWTTripleBoostModel,          # +2.9
    "dtcwt_quad_boost": DTCWTQuadBoostModel,              # +4.6
    "dtcwt_deep_double_boost": DTCWTDeepDoubleBoostModel, # +6.5
    "dtcwt_deep_triple_boost": DTCWTDeepTripleBoostModel, # +10.6
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