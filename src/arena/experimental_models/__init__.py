"""
===============================================================================
EXPERIMENTAL MODELS — Arena Model Competition Framework
===============================================================================

Staff Professor Implementation — Chinese Academy of Quantitative Finance
Panel: Chen Wei (Tsinghua), Liu Xiaoming (Peking), Zhang Yifan (Fudan)

20 WORLD-CLASS EXPERIMENTAL MODELS

BENCHMARK TO BEAT:
    online_bayesian_ma: Score 0.6783 | BIC -5665 | CRPS 0.0186 | PIT PASS | +7.0%

MODELS:
    01. ensemble_optimal_transport - Wasserstein barycenter (Liu, 98/100)
    02. fractional_kalman - Long memory dynamics (Chen, 97/100)
    03. score_matching_kalman - Hyvärinen-optimized (Liu, 96/100)
    04. cross_sectional_kalman - Market momentum injection (Zhang, 96/100)
    05. variational_bayes_kalman - Posterior inference (Liu, 95/100)
    06. wavelet_kalman - Multi-scale decomposition (Zhang, 95/100)
    07. information_geometric_kalman - Natural gradients (Liu, 94/100)
    08. hmm_kalman - Regime switching (Zhang, 94/100)
    09. shrinkage_kalman - James-Stein regularization (Zhang, 93/100)
    10. crps_optimized - Direct CRPS minimization (Zhang, 92/100)
    11. levy_stable_kalman - α-stable tails (Chen, 95/100)
    12. stochastic_vol_particle - SV with leverage (Chen, 94/100)
    13. tempered_stable_kalman - Tempered tails (Chen, 93/100)
    14. conformal_predictive_kalman - Coverage guarantees (Liu, 93/100)
    15. adaptive_forgetting_kalman - Dynamic forgetting (Evolution)
    16. exponential_tilting_kalman - Tail calibration (Evolution)
    17. dual_kalman - Mean + Vol filtering (Evolution)
    18. gradient_boosted_kalman - Boosting ensemble (Evolution)
    19. mixture_density_kalman - Multimodal predictions (Evolution)
    20. ultimate_ensemble_bma - Best of all (Evolution)

Author: Staff Professor, Chinese Academy of Quantitative Finance
Date: February 2026
"""

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Current champion
from .online_bayesian_ma import OnlineBayesianModelAvgModel

# 20 World-class models
from .m01_ensemble_optimal_transport import EnsembleOptimalTransportModel
from .m02_fractional_kalman import FractionalKalmanModel
from .m03_score_matching_kalman import ScoreMatchingKalmanModel
from .m04_cross_sectional_kalman import CrossSectionalKalmanModel
from .m05_variational_bayes_kalman import VariationalBayesKalmanModel
from .m06_wavelet_kalman import WaveletKalmanModel
from .m07_information_geometric_kalman import InformationGeometricKalmanModel
from .m08_hmm_kalman import HMMKalmanModel
from .m09_shrinkage_kalman import ShrinkageKalmanModel
from .m10_crps_optimized import CRPSOptimizedModel
from .m11_levy_stable_kalman import LevyStableKalmanModel
from .m12_stochastic_vol_particle import StochasticVolParticleModel
from .m13_tempered_stable_kalman import TemperedStableKalmanModel
from .m14_conformal_predictive_kalman import ConformalPredictiveKalmanModel
from .m15_adaptive_forgetting_kalman import AdaptiveForgettingKalmanModel
from .m16_exponential_tilting_kalman import ExponentialTiltingKalmanModel
from .m17_dual_kalman import DualKalmanModel
from .m18_gradient_boosted_kalman import GradientBoostedKalmanModel
from .m19_mixture_density_kalman import MixtureDensityKalmanModel
from .m20_ultimate_ensemble_bma import UltimateEnsembleBMAModel


# Registry of active experimental models
EXPERIMENTAL_MODELS = {
    "online_bayesian_ma": OnlineBayesianModelAvgModel,
    "ensemble_optimal_transport": EnsembleOptimalTransportModel,
    "fractional_kalman": FractionalKalmanModel,
    "score_matching_kalman": ScoreMatchingKalmanModel,
    "cross_sectional_kalman": CrossSectionalKalmanModel,
    "variational_bayes_kalman": VariationalBayesKalmanModel,
    "wavelet_kalman": WaveletKalmanModel,
    "information_geometric_kalman": InformationGeometricKalmanModel,
    "hmm_kalman": HMMKalmanModel,
    "shrinkage_kalman": ShrinkageKalmanModel,
    "crps_optimized": CRPSOptimizedModel,
    "levy_stable_kalman": LevyStableKalmanModel,
    "stochastic_vol_particle": StochasticVolParticleModel,
    "tempered_stable_kalman": TemperedStableKalmanModel,
    "conformal_predictive_kalman": ConformalPredictiveKalmanModel,
    "adaptive_forgetting_kalman": AdaptiveForgettingKalmanModel,
    "exponential_tilting_kalman": ExponentialTiltingKalmanModel,
    "dual_kalman": DualKalmanModel,
    "gradient_boosted_kalman": GradientBoostedKalmanModel,
    "mixture_density_kalman": MixtureDensityKalmanModel,
    "ultimate_ensemble_bma": UltimateEnsembleBMAModel,
}

# Model specifications
EXPERIMENTAL_MODEL_SPECS = {name: ExperimentalModelSpec(
    name=name,
    family=ExperimentalModelFamily.REGIME_COUPLED,
    n_params=3,
    param_names=("q", "c", "phi"),
    default_params={"q": 1e-6, "c": 1.0, "phi": 0.0},
    description=f"World-class experimental model: {name}",
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