"""
Arena Experimental Models - Top DTCWT models + Next-Gen evolutions.
10,000ms hard time limit enforced.

Current Champions (sorted by BIC):
- dtcwt_double_boost: BIC -75068 (+133%)
- dtcwt_enhanced_boost: BIC -58171 (+99%)
- dtcwt_deep_levels: BIC -54022 (+91%)
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Champion model
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel

# Top wavelet family models
from .m77_wavelet_families import (
    DB4ComplexWaveletModel,
    Sym8ComplexWaveletModel,
)

# Gen1 DTCWT evolution models
from .m79_dtcwt_next_gen import (
    DTCWTEnhancedBoostModel,
    DTCWTDeepLevelsModel,
    DTCWTPhaseAwareModel,
    DTCWTScaleWeightedModel,
    DTCWTDoubleBoostModel,
)

# Gen2 DTCWT evolution models
from .m80_dtcwt_gen2 import (
    DTCWTTripleBoostModel,
    DTCWTQuadBoostModel,
    DTCWTDeepDoubleBoostModel,
    DTCWTDeepTripleBoostModel,
)


EXPERIMENTAL_MODELS = {
    # Original champion
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    # Top wavelet families
    "db4_complex_cwt": DB4ComplexWaveletModel,
    "sym8_complex_cwt": Sym8ComplexWaveletModel,
    # Gen1 DTCWT evolution
    "dtcwt_enhanced_boost": DTCWTEnhancedBoostModel,
    "dtcwt_deep_levels": DTCWTDeepLevelsModel,
    "dtcwt_phase_aware": DTCWTPhaseAwareModel,
    "dtcwt_scale_weighted": DTCWTScaleWeightedModel,
    "dtcwt_double_boost": DTCWTDoubleBoostModel,
    # Gen2 DTCWT evolution
    "dtcwt_triple_boost": DTCWTTripleBoostModel,
    "dtcwt_quad_boost": DTCWTQuadBoostModel,
    "dtcwt_deep_double_boost": DTCWTDeepDoubleBoostModel,
    "dtcwt_deep_triple_boost": DTCWTDeepTripleBoostModel,
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