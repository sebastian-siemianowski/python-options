"""
Arena Experimental Models - Generation 3 DTCWT Elite Models.
10,000ms hard time limit enforced.

Promotion Hard Gates:
- CSS >= 0.65 (Calibration Stability Under Stress)
- FEC >= 0.75 (Forecast Entropy Consistency)

All models use dualtree_complex_wavelet as genetic base.
Total: 20 experimental models
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Champion model - genetic base for all others
from .m14_dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel

# Generation 3 Batch 1: Regime-Adaptive Variants
from .m81_dtcwt_gen3_batch1 import (
    DTCWTRegimeAdaptiveModel,
    DTCWTExpWeightedModel,
)

# Generation 3 Batch 2: Score-Driven & Copula Variants
from .m82_dtcwt_gen3_batch2 import (
    DTCWTScoreDrivenModel,
    DTCWTCopulaTailModel,
    DTCWTSpectralRiskModel,
)

# Generation 3 Batch 3: Ensemble & Particle Filter Variants
from .m83_dtcwt_gen3_batch3 import (
    DTCWTOrthogonalEnsembleModel,
    DTCWTParticleFilterModel,
)

# Generation 3 Batch 4: Advanced Variants
from .m84_dtcwt_gen3_batch4 import (
    DTCWTInfoTheoreticModel,
    DTCWTAdaptiveVolModel,
)

# Generation 3 Batch 5: Top Performers
from .m85_dtcwt_gen3_batch5 import (
    DTCWTVolRegimeModel,
    DTCWTScaleWeightedModel,
    DTCWTMomentumEnhancedModel,
)

# Generation 3 Batch 6: Final Elite Models
from .m86_dtcwt_gen3_batch6 import (
    DTCWTAdaptiveDecayModel,
    DTCWTDeepLevelsModel,
    DTCWTVolClusteringModel,
    DTCWTTrendFollowingModel,
    DTCWTMeanReversionModel,
    DTCWTLiquidityAwareModel,
    DTCWTSkewAwareModel,
)


EXPERIMENTAL_MODELS = {
    # Original champion (CSS 0.77, FEC 0.81) - #1
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    
    # Gen3 Batch 1: Regime-Adaptive - #2-3
    "dtcwt_regime_adaptive": DTCWTRegimeAdaptiveModel,
    "dtcwt_exp_weighted": DTCWTExpWeightedModel,
    
    # Gen3 Batch 2: Score-Driven & Copula - #4-6
    "dtcwt_score_driven": DTCWTScoreDrivenModel,
    "dtcwt_copula_tail": DTCWTCopulaTailModel,
    "dtcwt_spectral_risk": DTCWTSpectralRiskModel,
    
    # Gen3 Batch 3: Ensemble & Particle Filter - #7-8
    "dtcwt_orthogonal_ensemble": DTCWTOrthogonalEnsembleModel,
    "dtcwt_particle_filter": DTCWTParticleFilterModel,
    
    # Gen3 Batch 4: Advanced - #9-10
    "dtcwt_info_theoretic": DTCWTInfoTheoreticModel,
    "dtcwt_adaptive_vol": DTCWTAdaptiveVolModel,
    
    # Gen3 Batch 5: Top Performers - #11-13
    "dtcwt_vol_regime": DTCWTVolRegimeModel,
    "dtcwt_scale_weighted": DTCWTScaleWeightedModel,
    "dtcwt_momentum_enhanced": DTCWTMomentumEnhancedModel,
    
    # Gen3 Batch 6: Final Elite - #14-20
    "dtcwt_adaptive_decay": DTCWTAdaptiveDecayModel,
    "dtcwt_deep_levels": DTCWTDeepLevelsModel,
    "dtcwt_vol_clustering": DTCWTVolClusteringModel,
    "dtcwt_trend_following": DTCWTTrendFollowingModel,
    "dtcwt_mean_reversion": DTCWTMeanReversionModel,
    "dtcwt_liquidity_aware": DTCWTLiquidityAwareModel,
    "dtcwt_skew_aware": DTCWTSkewAwareModel,
}

EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=4,
    param_names=("q", "c", "phi"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0},
    description=f"Gen3 DTCWT model: {name}",
) for name in EXPERIMENTAL_MODELS.keys()}


def get_experimental_model_specs():
    return list(EXPERIMENTAL_MODEL_SPECS.values())


def create_experimental_model(name: str, **kwargs):
    if name not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unknown experimental model: {name}")
    return EXPERIMENTAL_MODELS[name](**kwargs)


__all__ = [
    'BaseExperimentalModel', 'ExperimentalModelSpec', 'ExperimentalModelFamily',
    'EXPERIMENTAL_MODELS', 'EXPERIMENTAL_MODEL_SPECS',
    'get_experimental_model_specs', 'create_experimental_model',
]