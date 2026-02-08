"""Arena Experimental Models - Top 5 Elite Winners (Generation 9)

Winners from arena competition (Feb 2026):
1. dtcwt_vol_regime         67.84  +10.6 vs STD
2. dtcwt_qshift             67.69  +10.4 vs STD  
3. dualtree_complex_wavelet 67.02  +9.8 vs STD
4. entropy_matching         66.02  +8.8 vs STD
5. dtcwt_adaptive_levels    65.83  +8.6 vs STD
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from .m85_dtcwt_gen3_batch5 import DTCWTVolRegimeModel

from .gen9_batch1_dtcwt import (
    DTCWTQShiftModel,
    DTCWTAdaptiveLevelsModel,
)

from .gen9_batch3_entropy import (
    EntropyMatchingModel,
)

EXPERIMENTAL_MODELS = {
    "dtcwt_vol_regime": DTCWTVolRegimeModel,
    "dtcwt_qshift": DTCWTQShiftModel,
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    "entropy_matching": EntropyMatchingModel,
    "dtcwt_adaptive_levels": DTCWTAdaptiveLevelsModel,
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi", "cw"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
    description=f"Gen9 elite winner: {name}",
) for name in EXPERIMENTAL_MODELS.keys()}

def get_experimental_model_specs():
    return list(EXPERIMENTAL_MODEL_SPECS.values())

def create_experimental_model(name, **kwargs):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown experimental model: {name}")
    return EXPERIMENTAL_MODELS[name](**kwargs)

__all__ = [
    "BaseExperimentalModel", "ExperimentalModelSpec", "ExperimentalModelFamily",
    "EXPERIMENTAL_MODELS", "EXPERIMENTAL_MODEL_SPECS",
    "get_experimental_model_specs", "create_experimental_model",
]