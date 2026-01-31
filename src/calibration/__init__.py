"""
===============================================================================
CALIBRATION — Probability Calibration and Model Selection
===============================================================================

This package provides calibration utilities for the tuning system:

    - model_selection: AIC, BIC, kurtosis, model weights, temporal smoothing
    - pit_calibration: PIT uniformity tests
    - isotonic_recalibration: Probability transport operators
    - calibrated_trust: Trust authority with regime penalties
    - adaptive_nu_refinement: ν refinement for Student-t models
    - gh_distribution: Generalized Hyperbolic distribution
    - tvvm_model: Time-varying volatility multiplier
"""

from calibration.model_selection import (
    compute_aic,
    compute_bic,
    compute_kurtosis,
    compute_bic_model_weights,
    compute_bic_model_weights_from_scores,
    apply_temporal_smoothing,
    normalize_weights,
    DEFAULT_TEMPORAL_ALPHA,
    DEFAULT_MODEL_SELECTION_METHOD,
    DEFAULT_BIC_WEIGHT,
)

__all__ = [
    # Model selection
    'compute_aic',
    'compute_bic',
    'compute_kurtosis',
    'compute_bic_model_weights',
    'compute_bic_model_weights_from_scores',
    'apply_temporal_smoothing',
    'normalize_weights',
    'DEFAULT_TEMPORAL_ALPHA',
    'DEFAULT_MODEL_SELECTION_METHOD',
    'DEFAULT_BIC_WEIGHT',
]
