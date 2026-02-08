"""Arena Experimental Models - 7 Core Models (Feb 2026)

After extensive arena competition, only 7 proven models remain active.
All models are stored as standalone files in safe_storage/.

Active Models (vs STD > 3):
1. dtcwt_qshift (+7.2%) - Q-shift filters for improved frequency selectivity
2. dtcwt_magnitude_threshold (+7.1%) - Magnitude-based noise reduction
3. dualtree_complex_wavelet (+7.1%) - Core DTCWT with phase-aware filtering
4. elite_hybrid_eta (+5.5%) - Full ensemble combination
5. dtcwt_adaptive_levels (+5.4%) - Adaptive decomposition levels
6. dtcwt_vol_regime (+4.6%) - Volatility regime conditioning
7. stress_adaptive_inflation (+3.0%) - Calibration-based inflation

Hard Gates:
- CSS >= 0.65 (Calibration Stability Under Stress)
- FEC >= 0.75 (Forecast Entropy Consistency)
- vs STD >= 3 points
- PIT >= 75%
- 10000ms time limit
"""

import sys
import os

# Add safe_storage to path for imports
_safe_storage_path = os.path.join(os.path.dirname(__file__), '..', 'safe_storage')
if _safe_storage_path not in sys.path:
    sys.path.insert(0, _safe_storage_path)

from .base import ExperimentalModelSpec, ExperimentalModelFamily, BaseExperimentalModel

# Import models from safe_storage
from ..safe_storage.dtcwt_qshift import DTCWTQShiftModel
from ..safe_storage.dtcwt_magnitude_threshold import DTCWTMagnitudeThresholdModel
from ..safe_storage.dualtree_complex_wavelet import DualTreeComplexWaveletKalmanModel
from ..safe_storage.elite_hybrid_eta import EliteHybridEtaModel
from ..safe_storage.dtcwt_adaptive_levels import DTCWTAdaptiveLevelsModel
from ..safe_storage.dtcwt_vol_regime import DTCWTVolRegimeModel
from ..safe_storage.stress_adaptive_inflation import StressAdaptiveInflationModel

EXPERIMENTAL_MODELS = {
    "dtcwt_qshift": DTCWTQShiftModel,
    "dtcwt_magnitude_threshold": DTCWTMagnitudeThresholdModel,
    "dualtree_complex_wavelet": DualTreeComplexWaveletKalmanModel,
    "elite_hybrid_eta": EliteHybridEtaModel,
    "dtcwt_adaptive_levels": DTCWTAdaptiveLevelsModel,
    "dtcwt_vol_regime": DTCWTVolRegimeModel,
    "stress_adaptive_inflation": StressAdaptiveInflationModel,
}

# Model specs with scores from arena competition
EXPERIMENTAL_MODEL_SPECS = {
    "dtcwt_qshift": ExperimentalModelSpec(
        name="dtcwt_qshift", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Q-shift DTCWT (+7.2% vs STD, CSS: 0.44, FEC: 0.79)",
    ),
    "dtcwt_magnitude_threshold": ExperimentalModelSpec(
        name="dtcwt_magnitude_threshold", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Magnitude thresholding (+7.1% vs STD, CSS: 0.73, FEC: 0.85)",
    ),
    "dualtree_complex_wavelet": ExperimentalModelSpec(
        name="dualtree_complex_wavelet", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Core DTCWT (+7.1% vs STD, CSS: 0.77, FEC: 0.81)",
    ),
    "elite_hybrid_eta": ExperimentalModelSpec(
        name="elite_hybrid_eta", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Hybrid ensemble (+5.5% vs STD, CSS: 0.69, FEC: 0.84)",
    ),
    "dtcwt_adaptive_levels": ExperimentalModelSpec(
        name="dtcwt_adaptive_levels", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Adaptive levels (+5.4% vs STD, CSS: 0.56, FEC: 0.82)",
    ),
    "dtcwt_vol_regime": ExperimentalModelSpec(
        name="dtcwt_vol_regime", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Vol regime conditioning (+4.6% vs STD, CSS: 0.66, FEC: 0.80)",
    ),
    "stress_adaptive_inflation": ExperimentalModelSpec(
        name="stress_adaptive_inflation", family=ExperimentalModelFamily.REGIME_COUPLED,
        n_params=4, param_names=("q", "c", "phi", "cw"),
        default_params={"q": 1e-6, "c": 1.0, "phi": 0.0, "cw": 1.0},
        description="Adaptive inflation (+3.0% vs STD, CSS: 0.51, FEC: 0.80)",
    ),
}


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