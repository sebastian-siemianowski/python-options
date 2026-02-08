"""Arena Experimental Models - 5 Elite Models (Promotion Candidates)"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from .m85_dtcwt_gen3_batch5 import DTCWTVolRegimeModel
from .m99_gen8_fec_batch2 import FECVarianceTargetModel
from .m100_gen8_combined_batch3 import CombinedCalibConstrainedModel, CombinedHybridStressModel

EXPERIMENTAL_MODELS = {
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    "dtcwt_vol_regime": DTCWTVolRegimeModel,
    "combined_calib_constrained": CombinedCalibConstrainedModel,
    "combined_hybrid_stress": CombinedHybridStressModel,
    "fec_variance_target": FECVarianceTargetModel,
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi", "cw"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
    description=f"Elite model: {name}",
) for name in EXPERIMENTAL_MODELS.keys()}

def get_experimental_model_specs():
    return list(EXPERIMENTAL_MODEL_SPECS.values())

def create_experimental_model(name, **kwargs):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown experimental model: {name}")
    return EXPERIMENTAL_MODELS[name](**kwargs)

__all__ = [
    'BaseExperimentalModel', 'ExperimentalModelSpec', 'ExperimentalModelFamily',
    'EXPERIMENTAL_MODELS', 'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs', 'create_experimental_model',
]
