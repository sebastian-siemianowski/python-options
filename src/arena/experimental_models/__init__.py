"""Arena Experimental Models - 40 Elite Models (Generation 8)"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from .m85_dtcwt_gen3_batch5 import DTCWTVolRegimeModel

from .m98_gen8_css_batch1 import (
    CSSHierarchicalStressModel, CSSVolPersistenceModel, CSSReturnStressModel,
    CSSDrawdownStressModel, CSSVolAccelModel, CSSRegimeTransitionModel,
    CSSVolClusterModel, CSSSkewStressModel, CSSKurtosisStressModel,
)

from .m99_gen8_fec_batch2 import (
    FECEntropyMatchModel, FECAdaptiveCalibModel, FECBlendedVolModel,
    FECRobustVolModel, FECDoubleEMAModel, FECMedianVolModel,
    FECTailAdjustModel, FECQuantileModel, FECVarianceTargetModel,
    FECStableInflationModel,
)

from .m100_gen8_combined_batch3 import (
    CombinedDualTargetModel, CombinedAdaptiveBalanceModel, CombinedHybridStressModel,
    CombinedRobustBlendModel, CombinedSmoothedStressModel, CombinedRegimeAwareModel,
    CombinedCalibConstrainedModel, CombinedMultiScaleModel, CombinedOptimalModel,
    CombinedEliteModel,
)

from .m101_gen8_advanced_batch4 import (
    AdvancedKoopmanHybridModel, AdvancedSSAHybridModel, AdvancedRobustHybridModel,
    AdvancedInfoTheoreticModel, AdvancedEnsembleModel, AdvancedScaleMixModel,
    AdvancedPhaseMagModel, AdvancedMomentumModel, AdvancedVarianceTargetModel,
    AdvancedUltimateModel,
)

EXPERIMENTAL_MODELS = {
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    "dtcwt_vol_regime": DTCWTVolRegimeModel,
    "css_hierarchical_stress": CSSHierarchicalStressModel,
    "css_vol_persistence": CSSVolPersistenceModel,
    "css_return_stress": CSSReturnStressModel,
    "css_drawdown_stress": CSSDrawdownStressModel,
    "css_vol_accel": CSSVolAccelModel,
    "css_regime_transition": CSSRegimeTransitionModel,
    "css_vol_cluster": CSSVolClusterModel,
    "css_skew_stress": CSSSkewStressModel,
    "css_kurtosis_stress": CSSKurtosisStressModel,
    "fec_entropy_match": FECEntropyMatchModel,
    "fec_adaptive_calib": FECAdaptiveCalibModel,
    "fec_blended_vol": FECBlendedVolModel,
    "fec_robust_vol": FECRobustVolModel,
    "fec_double_ema": FECDoubleEMAModel,
    "fec_median_vol": FECMedianVolModel,
    "fec_tail_adjust": FECTailAdjustModel,
    "fec_quantile": FECQuantileModel,
    "fec_variance_target": FECVarianceTargetModel,
    "fec_stable_inflation": FECStableInflationModel,
    "combined_dual_target": CombinedDualTargetModel,
    "combined_adaptive_balance": CombinedAdaptiveBalanceModel,
    "combined_hybrid_stress": CombinedHybridStressModel,
    "combined_robust_blend": CombinedRobustBlendModel,
    "combined_smoothed_stress": CombinedSmoothedStressModel,
    "combined_regime_aware": CombinedRegimeAwareModel,
    "combined_calib_constrained": CombinedCalibConstrainedModel,
    "combined_multi_scale": CombinedMultiScaleModel,
    "combined_optimal": CombinedOptimalModel,
    "combined_elite": CombinedEliteModel,
    "advanced_koopman": AdvancedKoopmanHybridModel,
    "advanced_ssa": AdvancedSSAHybridModel,
    "advanced_robust": AdvancedRobustHybridModel,
    "advanced_info_theoretic": AdvancedInfoTheoreticModel,
    "advanced_ensemble": AdvancedEnsembleModel,
    "advanced_scale_mix": AdvancedScaleMixModel,
    "advanced_phase_mag": AdvancedPhaseMagModel,
    "advanced_momentum": AdvancedMomentumModel,
    "advanced_ultimate": AdvancedUltimateModel,
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi", "cw"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
    description=f"Gen8 model: {name}",
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
