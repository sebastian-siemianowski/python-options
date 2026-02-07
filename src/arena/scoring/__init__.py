"""
===============================================================================
ARENA SCORING — Proper Scoring Rules for Model Evaluation
===============================================================================

Implements theoretically justified scoring rules for probabilistic forecasts:

1. CRPS (Continuous Ranked Probability Score)
   - Strictly proper scoring rule
   - Decomposes into reliability (calibration) and sharpness (confidence)
   - Closed-form for Gaussian and Student-t distributions

2. Hyvärinen Score
   - Robust to model misspecification
   - Based on score function (gradient of log-density)

3. Combined Scoring
   - Unified score combining multiple metrics
   - Configurable weights with regime adaptation

Reference: Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from .crps import (
    compute_crps_gaussian,
    compute_crps_student_t,
    compute_crps_empirical,
    CRPSResult,
    decompose_crps,
)

from .hyvarinen import (
    compute_hyvarinen_score_gaussian,
    compute_hyvarinen_score_student_t,
)

from .combined import (
    compute_combined_score,
    CombinedScoreResult,
    ScoringConfig,
    DEFAULT_SCORING_CONFIG,
)

# Advanced metrics: CSS, FEC, DIG
from .advanced import (
    compute_css,
    compute_fec,
    compute_dig,
    compute_advanced_scores,
    CSSResult,
    FECResult,
    DIGResult,
    AdvancedScoringResult,
    StressRegime,
    classify_stress_regime,
)

__all__ = [
    # CRPS
    'compute_crps_gaussian',
    'compute_crps_student_t',
    'compute_crps_empirical',
    'CRPSResult',
    'decompose_crps',
    # Hyvarinen
    'compute_hyvarinen_score_gaussian',
    'compute_hyvarinen_score_student_t',
    # Combined
    'compute_combined_score',
    'CombinedScoreResult',
    'ScoringConfig',
    'DEFAULT_SCORING_CONFIG',
    # Advanced (CSS, FEC, DIG)
    'compute_css',
    'compute_fec',
    'compute_dig',
    'compute_advanced_scores',
    'CSSResult',
    'FECResult',
    'DIGResult',
    'AdvancedScoringResult',
    'StressRegime',
    'classify_stress_regime',
]
