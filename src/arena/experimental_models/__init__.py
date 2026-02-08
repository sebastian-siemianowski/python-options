"""Arena Experimental Models - Generation 10 (60 models)

Panel of Three Chinese Professors Evaluation:
- Professor Wei Chen (Tsinghua): 95/100
- Professor Liu Xiaoming (PKU): 93/100
- Professor Zhang Yifan (Fudan): 92/100

Staff Professor Decision: Implement 60 models in 6 batches.

Hard Gates:
- CSS >= 0.65 (Calibration Stability Under Stress)
- FEC >= 0.75 (Forecast Entropy Consistency)
- vs STD >= 3 points
- PIT >= 75%
- 10000ms time limit
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from .m85_dtcwt_gen3_batch5 import DTCWTVolRegimeModel

from .gen9_batch1_dtcwt import DTCWTQShiftModel, DTCWTAdaptiveLevelsModel
from .gen9_batch3_entropy import EntropyMatchingModel

from .gen10_batch1_dtcwt import (
    DTCWTDeepScaleModel, DTCWTAdaptiveScaleWeightModel, DTCWTPhaseRegimeModel,
    DTCWTMagnitudeThresholdModel, DTCWTDirectionalFilterModel, DTCWTCrossScaleCorrelationModel,
    DTCWTEnergyConcentrationModel, DTCWTTemporalSmoothingModel, DTCWTBandpassFilterModel,
    DTCWTHybridScaleModel,
)

from .gen10_batch2_stress import (
    StressHierarchicalDeepModel, StressDrawdownIntegrationModel, StressRegimePersistenceModel,
    StressVolAccelerationModel, StressTailRiskModel, StressReturnMagnitudeModel,
    StressClusteringModel, StressExponentialWeightedModel, StressAdaptiveInflationModel,
    StressCombinedEliteModel,
)

from .gen10_batch3_entropy import (
    EntropyTrackingModel, EntropyBlendedVolModel, EntropyCalibrationFeedbackModel,
    EntropyRobustVolModel, EntropyDoubleEMAModel, EntropyMedianVolModel,
    EntropyVarianceTargetModel, EntropyQuantileModel, EntropyStableInflationModel,
    EntropyCombinedEliteModel,
)

from .gen10_batch4_regime import (
    RegimeSmoothTransitionModel, RegimeAdaptiveParamsModel, RegimeMultiHorizonModel,
    RegimePersistenceModel, RegimeVolClusterModel, RegimeTrendAwareModel,
    RegimeStabilityModel, RegimeEnhancedModel, RegimeCalibrationModel,
    RegimeCombinedEliteModel,
)

from .gen10_batch5_robust import (
    RobustHuberModel, RobustTukeyModel, RobustWinsorizedModel, RobustTrimmedModel,
    RobustMADModel, RobustIQRModel, RobustAdaptiveModel, RobustExponentialModel,
    RobustHybridModel, RobustCombinedEliteModel,
)

from .gen10_batch6_elite import (
    EliteHybridAlphaModel, EliteHybridBetaModel, EliteHybridGammaModel,
    EliteHybridDeltaModel, EliteHybridEpsilonModel, EliteHybridZetaModel,
    EliteHybridEtaModel, EliteHybridThetaModel, EliteHybridIotaModel,
    EliteUltimateModel,
)

EXPERIMENTAL_MODELS = {
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    "dtcwt_vol_regime": DTCWTVolRegimeModel,
    "dtcwt_qshift": DTCWTQShiftModel,
    "dtcwt_adaptive_levels": DTCWTAdaptiveLevelsModel,
    "entropy_matching": EntropyMatchingModel,
    "dtcwt_deep_scale": DTCWTDeepScaleModel,
    "dtcwt_adaptive_scale_weight": DTCWTAdaptiveScaleWeightModel,
    "dtcwt_phase_regime": DTCWTPhaseRegimeModel,
    "dtcwt_magnitude_threshold": DTCWTMagnitudeThresholdModel,
    "dtcwt_directional_filter": DTCWTDirectionalFilterModel,
    "dtcwt_cross_scale_correlation": DTCWTCrossScaleCorrelationModel,
    "dtcwt_energy_concentration": DTCWTEnergyConcentrationModel,
    "dtcwt_temporal_smoothing": DTCWTTemporalSmoothingModel,
    "dtcwt_bandpass_filter": DTCWTBandpassFilterModel,
    "dtcwt_hybrid_scale": DTCWTHybridScaleModel,
    "stress_hierarchical_deep": StressHierarchicalDeepModel,
    "stress_drawdown_integration": StressDrawdownIntegrationModel,
    "stress_regime_persistence": StressRegimePersistenceModel,
    "stress_vol_acceleration": StressVolAccelerationModel,
    "stress_tail_risk": StressTailRiskModel,
    "stress_return_magnitude": StressReturnMagnitudeModel,
    "stress_clustering": StressClusteringModel,
    "stress_exponential_weighted": StressExponentialWeightedModel,
    "stress_adaptive_inflation": StressAdaptiveInflationModel,
    "stress_combined_elite": StressCombinedEliteModel,
    "entropy_tracking": EntropyTrackingModel,
    "entropy_blended_vol": EntropyBlendedVolModel,
    "entropy_calibration_feedback": EntropyCalibrationFeedbackModel,
    "entropy_robust_vol": EntropyRobustVolModel,
    "entropy_double_ema": EntropyDoubleEMAModel,
    "entropy_median_vol": EntropyMedianVolModel,
    "entropy_variance_target": EntropyVarianceTargetModel,
    "entropy_quantile": EntropyQuantileModel,
    "entropy_stable_inflation": EntropyStableInflationModel,
    "entropy_combined_elite": EntropyCombinedEliteModel,
    "regime_smooth_transition": RegimeSmoothTransitionModel,
    "regime_adaptive_params": RegimeAdaptiveParamsModel,
    "regime_multi_horizon": RegimeMultiHorizonModel,
    "regime_persistence": RegimePersistenceModel,
    "regime_vol_cluster": RegimeVolClusterModel,
    "regime_trend_aware": RegimeTrendAwareModel,
    "regime_stability": RegimeStabilityModel,
    "regime_enhanced": RegimeEnhancedModel,
    "regime_calibration": RegimeCalibrationModel,
    "regime_combined_elite": RegimeCombinedEliteModel,
    "robust_huber": RobustHuberModel,
    "robust_tukey": RobustTukeyModel,
    "robust_winsorized": RobustWinsorizedModel,
    "robust_trimmed": RobustTrimmedModel,
    "robust_mad": RobustMADModel,
    "robust_iqr": RobustIQRModel,
    "robust_adaptive": RobustAdaptiveModel,
    "robust_exponential": RobustExponentialModel,
    "robust_hybrid": RobustHybridModel,
    "robust_combined_elite": RobustCombinedEliteModel,
    "elite_hybrid_alpha": EliteHybridAlphaModel,
    "elite_hybrid_beta": EliteHybridBetaModel,
    "elite_hybrid_gamma": EliteHybridGammaModel,
    "elite_hybrid_delta": EliteHybridDeltaModel,
    "elite_hybrid_epsilon": EliteHybridEpsilonModel,
    "elite_hybrid_zeta": EliteHybridZetaModel,
    "elite_hybrid_eta": EliteHybridEtaModel,
    "elite_hybrid_theta": EliteHybridThetaModel,
    "elite_hybrid_iota": EliteHybridIotaModel,
    "elite_ultimate": EliteUltimateModel,
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi", "cw"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
    description=f"Gen10 experimental model: {name}",
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
