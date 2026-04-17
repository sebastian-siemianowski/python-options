"""
tune_modules/utilities.py -- Utility functions and constants extracted from tune.py.

Contains:
  - Verbose output control (_is_quiet, _log)
  - Model class definitions (ModelClass enum, labels, param counts)
  - Model classification (is_student_t_model, is_heavy_tailed_model)
  - GH / TVVM / Isotonic / Adaptive-nu config constants and getter functions
  - Regime-conditional process noise floor (Q_FLOOR_BY_REGIME, compute_vol_proportional_q_floor)
  - Cross-asset phi pooling (apply_cross_asset_phi_pooling)
  - DEFAULT_TEMPORAL_ALPHA, MIN_HYVARINEN_SAMPLES
  - MarketRegime, REGIME_LABELS, MIN_REGIME_SAMPLES (re-exported from models.regime)
"""

import os
from enum import IntEnum
from typing import Dict, Optional

import numpy as np

# Import config flags and types from config.py (same package)
from tuning.tune_modules.config import (
    GH_MODEL_AVAILABLE,
    GHModelConfig,
    TVVM_AVAILABLE,
    TVVMConfig,
    ISOTONIC_RECALIBRATION_AVAILABLE,
    IsotonicRecalibrationConfig,
    ADAPTIVE_NU_AVAILABLE,
    AdaptiveNuConfig,
)


# =============================================================================
# VERBOSE OUTPUT CONTROL
# =============================================================================
# When running through tune_pretty.py with Rich progress display, we suppress
# verbose print statements to avoid cluttering the animated progress bar.
# Set TUNING_QUIET=1 environment variable to suppress detailed tuning messages.
# =============================================================================

def _is_quiet() -> bool:
    """Check if verbose output should be suppressed."""
    return os.environ.get('TUNING_QUIET', '').lower() in ('1', 'true', 'yes')


def _log(msg: str) -> None:
    """Print message only if not in quiet mode."""
    if not _is_quiet():
        print(msg)


# =============================================================================
# REGIME DEFINITIONS FOR HIERARCHICAL BAYESIAN PARAMETER TUNING
# =============================================================================
# These 5 latent regimes represent distinct market dynamics.
# Regime assignment is provided EXTERNALLY - this file only learns parameters.
# =============================================================================

# =============================================================================
# MODEL CLASS DEFINITIONS FOR BAYESIAN MODEL AVERAGING
# =============================================================================
# These model classes represent competing physical hypotheses about market dynamics.
# Model averaging preserves uncertainty across physics rather than selecting one.
# =============================================================================

class ModelClass(IntEnum):
    """
    Model class definitions for Bayesian model averaging within each regime.

    Model 0: KALMAN_GAUSSIAN
        - Standard Kalman filter with Gaussian observation noise
        - Parameters: q, c
        - Best for stable, well-behaved markets

    Model 1: PHI_GAUSSIAN
        - Kalman filter with AR(1) drift persistence
        - Parameters: q, c, phi
        - Best for trending markets

    Model 2: PHI_STUDENT_T
        - Kalman filter with AR(1) drift and Student-t tails
        - Parameters: q, c, phi, nu
        - Best for fat-tailed, trending markets
    """
    KALMAN_GAUSSIAN = 0
    PHI_GAUSSIAN = 1
    PHI_STUDENT_T = 2


# Model class labels for display (base names - actual models use phi_student_t_nu_{nu})
MODEL_CLASS_LABELS = {
    ModelClass.KALMAN_GAUSSIAN: "kalman_gaussian",
    ModelClass.PHI_GAUSSIAN: "kalman_phi_gaussian",
    ModelClass.PHI_STUDENT_T: "phi_student_t",  # Base name; actual models are phi_student_t_nu_{4,6,8,12,20}
}

# Model class parameter counts for BIC/AIC computation
MODEL_CLASS_N_PARAMS = {
    ModelClass.KALMAN_GAUSSIAN: 2,   # q, c
    ModelClass.PHI_GAUSSIAN: 3,      # q, c, phi
    ModelClass.PHI_STUDENT_T: 3,     # q, c, phi (nu is FIXED per sub-model, not estimated)
}


def is_student_t_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to a Student-t model.
    
    Student-t models follow the naming pattern:
    - 'phi_student_t_nu_{nu}' (e.g., 'phi_student_t_nu_4', 'phi_student_t_nu_8')
    - 'student_t' (legacy)
    - 'phi_student_t' (legacy)
    
    Args:
        model_name: Model identifier string
        
    Returns:
        True if model is a Student-t variant, False otherwise
    """
    if not model_name:
        return False
    model_lower = model_name.lower()
    return 'student_t' in model_lower or 'student-t' in model_lower


def is_heavy_tailed_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to any heavy-tailed distribution.
    
    Heavy-tailed models include:
    - Student-t: 'phi_student_t_nu_{nu}'
    - Skew-t: 'phi_skew_t_nu_{nu}_gamma_{gamma}'
    
    This is used for tail-aware signal generation and Monte Carlo sampling.
    
    Args:
        model_name: Model identifier string
        
    Returns:
        True if model has heavy tails, False otherwise
    """
    if not model_name:
        return False
    model_lower = model_name.lower()
    return (
        'student_t' in model_lower or 
        'student-t' in model_lower or 
        'skew_t' in model_lower
    )


# =============================================================================
# DISCRETE ν GRID FOR STUDENT-T MODELS
# =============================================================================
# DISCRETE ν GRID AND φ SHRINKAGE PRIOR — NOW IN src/models/base.py
# =============================================================================
# The following constants and functions have been moved to src/models/base.py
# for modularity and reuse across the codebase:
#
#   STUDENT_T_NU_GRID              - Discrete ν grid [4, 8, 20]
#   PHI_SHRINKAGE_TAU_MIN          - Minimum τ for numerical stability
#   PHI_SHRINKAGE_GLOBAL_DEFAULT   - Center of shrinkage prior (0.0)
#   PHI_SHRINKAGE_LAMBDA_DEFAULT   - Prior strength (0.05)
#   phi_shrinkage_log_prior()      - Gaussian shrinkage log-prior
#   lambda_to_tau()                - Convert legacy λ to τ
#   tau_to_lambda()                - Convert τ to legacy λ
#   compute_phi_prior_diagnostics() - Audit diagnostics
#
# They are imported at the top of this file from models.base
# =============================================================================





# =============================================================================
# GENERALIZED HYPERBOLIC (GH) DISTRIBUTION CONFIGURATION
# =============================================================================
# GH distribution is a fallback model when Student-t fails PIT calibration.
# GH captures SKEWNESS that symmetric Student-t cannot.
#
# GH is a 5-parameter family that includes Student-t and Variance-Gamma
# as special cases. The key addition is β (beta) for skewness.
#
# When enabled:
#   - GH is attempted ONLY when other escalation methods fail
#   - GH must improve PIT p-value to be selected
#   - Small BIC penalty is acceptable if calibration improves
#
# ESCALATION ORDER:
#   1. φ-Gaussian / φ-Student-t (baseline)
#   2. Adaptive ν refinement (if boundary ν)
#   3. K=2 mixture (if still failing)
#   4. GH distribution (last resort for skewed assets)
# =============================================================================

# Feature toggle
GH_MODEL_ENABLED = True

# PIT threshold to attempt GH
GH_PIT_THRESHOLD = 0.05

# BIC threshold: allow GH even if BIC is slightly worse (for calibration)
# Negative value means we accept up to 10 BIC worse if PIT improves
GH_BIC_THRESHOLD = -10.0

# PIT improvement factor: must at least double p-value to justify GH
GH_PIT_IMPROVEMENT_FACTOR = 2.0


def get_gh_config() -> Optional['GHModelConfig']:
    """
    Get GH model configuration based on global settings.
    
    Returns:
        GHModelConfig if GH is available and enabled, None otherwise.
    """
    if not GH_MODEL_AVAILABLE or not GH_MODEL_ENABLED:
        return None
    
    return GHModelConfig(
        enabled=GH_MODEL_ENABLED,
        pit_threshold=GH_PIT_THRESHOLD,
        bic_threshold=GH_BIC_THRESHOLD,
        pit_improvement_factor=GH_PIT_IMPROVEMENT_FACTOR,
    )


# =============================================================================
# TIME-VARYING VOLATILITY MULTIPLIER (TVVM) CONFIGURATION
# =============================================================================
# TVVM addresses the volatility-of-volatility effect by making c time-varying.
#
# Standard model: r_t = μ_t + √(c·σ_t²)  (static c)
# TVVM model:     r_t = μ_t + √(c_t·σ_t²)  (dynamic c_t)
#
# where: c_t = c_base * (1 + γ * |Δσ_t/σ_t|)
#
# ESCALATION ORDER (TVVM is after GH):
#   1. φ-Gaussian / φ-Student-t (baseline)
#   2. Adaptive ν refinement
#   3. K=2 mixture
#   4. GH distribution (skewness)
#   5. TVVM (volatility-of-volatility)
# =============================================================================

# Feature toggle
TVVM_ENABLED = True

# PIT threshold to attempt TVVM
TVVM_PIT_THRESHOLD = 0.05

# Volatility-of-volatility threshold
TVVM_VOL_OF_VOL_THRESHOLD = 0.1

# PIT improvement factor
TVVM_PIT_IMPROVEMENT_FACTOR = 1.5

# Gamma grid for optimization
TVVM_GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def get_tvvm_config() -> Optional['TVVMConfig']:
    """
    Get TVVM model configuration based on global settings.
    
    Returns:
        TVVMConfig if TVVM is available and enabled, None otherwise.
    """
    if not TVVM_AVAILABLE or not TVVM_ENABLED:
        return None
    
    return TVVMConfig(
        enabled=TVVM_ENABLED,
        pit_threshold=TVVM_PIT_THRESHOLD,
        vol_of_vol_threshold=TVVM_VOL_OF_VOL_THRESHOLD,
        pit_improvement_factor=TVVM_PIT_IMPROVEMENT_FACTOR,
        gamma_grid=TVVM_GAMMA_GRID,
    )


# =============================================================================
# ISOTONIC RECALIBRATION CONFIGURATION
# =============================================================================
# Isotonic recalibration is a FIRST-CLASS PROBABILISTIC TRANSPORT OPERATOR.
# It is NOT a patch, validator, or escalation trigger.
#
# CORE DOCTRINE:
#   "Inference generates beliefs. Regimes provide context.
#    Calibration aligns beliefs with reality. Trust is updated continuously."
#
# ARCHITECTURE:
#   Model Inference → Raw PIT → Transport Map g → Calibrated PIT
#                                    ↓
#               Regime-Conditioned Diagnostics → Weight Updates
#
# KEY RULE: Raw PIT is NEVER used by regimes, diagnostics, or escalation.
#           Regimes see CALIBRATED probability, not raw belief.
#
# The transport map g: [0,1] → [0,1] is:
#   - Monotone (preserves probability ranking)
#   - Learned from data (via isotonic regression)
#   - Persisted with model parameters
#   - Applied BEFORE any downstream processing
# =============================================================================

# Feature toggle
ISOTONIC_RECALIBRATION_ENABLED = True

# Minimum observations for fitting
ISOTONIC_MIN_OBSERVATIONS = 50

# Validation split for out-of-sample check
ISOTONIC_VALIDATION_SPLIT = 0.2

# PIT bounds (numerical stability)
ISOTONIC_PIT_MIN = 0.001
ISOTONIC_PIT_MAX = 0.999


def get_recalibration_config():
    """
    Get isotonic recalibration configuration based on global settings.
    
    Returns:
        IsotonicRecalibrationConfig if available and enabled, None otherwise.
    """
    # Check if isotonic recalibration module is available
    try:
        if not ISOTONIC_RECALIBRATION_AVAILABLE or not ISOTONIC_RECALIBRATION_ENABLED:
            return None
    except NameError:
        # Variable not defined, module not available
        return None
    
    return IsotonicRecalibrationConfig(
        enabled=ISOTONIC_RECALIBRATION_ENABLED,
        min_observations=ISOTONIC_MIN_OBSERVATIONS,
        validation_split=ISOTONIC_VALIDATION_SPLIT,
        pit_min=ISOTONIC_PIT_MIN,
        pit_max=ISOTONIC_PIT_MAX,
    )


# =============================================================================
# ADAPTIVE ν REFINEMENT CONFIGURATION
# =============================================================================
# When calibration fails (PIT p < 0.05) for φ-T models at boundary ν values,
# we locally refine the ν grid to test intermediate values.
#
# CORE PRINCIPLE: Add resolution only where truth demands it.
#
# EXPANDED (Jan 2026): Now covers ALL ν values, not just boundaries.
# For severe failures (PIT < 0.01), always attempt refinement.
#
# Detection criteria (OR logic for severe, boundary OR flat for moderate):
#   1. Best ν has refinement candidates available
#   2. PIT KS p-value < 0.05 (calibration failure)
#   3. Model is φ-t variant (not Gaussian or mixture)
#   4. SEVERE (PIT < 0.01): Always refine
#   5. MODERATE: Boundary ν OR likelihood is locally flat
#
# Refinement candidates (EXPANDED):
#   - ν = 4  → test [3, 5] (extreme fat tails)
#   - ν = 6  → test [5, 7]
#   - ν = 8  → test [6, 10]
#   - ν = 12 → test [10, 14]
#   - ν = 20 → test [16, 25]
# =============================================================================

# Feature toggle
ADAPTIVE_NU_ENABLED = True

# Boundary ν values that trigger refinement check (EXPANDED to all values)
ADAPTIVE_NU_BOUNDARY_VALUES = (4.0, 6.0, 8.0, 12.0, 20.0)

# PIT p-value threshold for calibration failure
ADAPTIVE_NU_PIT_THRESHOLD = 0.05

# Severe PIT threshold - always attempt refinement regardless of other criteria
ADAPTIVE_NU_PIT_SEVERE_THRESHOLD = 0.01

# Log-likelihood flatness threshold (increased for more aggressive refinement)
ADAPTIVE_NU_FLATNESS_THRESHOLD = 2.0

# Refinement candidates for each ν value (EXPANDED)
ADAPTIVE_NU_CANDIDATES = {
    4.0: [3.0, 5.0],      # For extreme fat tails
    6.0: [5.0, 7.0],      # Fill gap between 4 and 8
    8.0: [6.0, 10.0],     # Fill gap between 6 and 12
    12.0: [10.0, 14.0],   # Test between 8-12 and 12-20
    20.0: [16.0, 25.0],   # Both directions
}


def get_adaptive_nu_config() -> Optional['AdaptiveNuConfig']:
    """
    Get adaptive ν refinement configuration based on global settings.
    
    Returns:
        AdaptiveNuConfig if available and enabled, None otherwise.
    """
    if not ADAPTIVE_NU_AVAILABLE or not ADAPTIVE_NU_ENABLED:
        return None
    
    return AdaptiveNuConfig(
        enabled=ADAPTIVE_NU_ENABLED,
        boundary_nu_values=ADAPTIVE_NU_BOUNDARY_VALUES,
        pit_threshold=ADAPTIVE_NU_PIT_THRESHOLD,
        pit_severe_threshold=ADAPTIVE_NU_PIT_SEVERE_THRESHOLD,
        likelihood_flatness_threshold=ADAPTIVE_NU_FLATNESS_THRESHOLD,
        refinement_candidates=ADAPTIVE_NU_CANDIDATES,
    )


# =============================================================================
# φ SHRINKAGE PRIOR FUNCTIONS — NOW IN src/models/base.py
# =============================================================================
# The following functions have been moved to src/models/base.py:
#   - phi_shrinkage_log_prior()
#   - lambda_to_tau()
#   - tau_to_lambda()
#   - compute_phi_prior_diagnostics()
#
# They are imported at the top of this file from models.base.
# =============================================================================


# Default temporal smoothing alpha for model posterior evolution
DEFAULT_TEMPORAL_ALPHA = 0.3


# Story 4.2: Import regime definitions from shared module
from models.regime import MarketRegime, REGIME_LABELS, MIN_REGIME_SAMPLES


# Minimum sample size for reliable Hyvärinen score computation
# Below this threshold, Hyvärinen is DISABLED to prevent illusory smoothness
# from small-n Student-t fits creating artificially good scores
MIN_HYVARINEN_SAMPLES = 100


# =============================================================================
# REGIME-CONDITIONAL PROCESS NOISE FLOOR (Story 1.1, April 2026)
# =============================================================================
# The Kalman filter's MLE-optimal q converges to ~1e-6, which collapses drift
# estimates to zero. A regime-conditional floor ensures the filter maintains
# meaningful sensitivity to drift in each market regime.
#
# Rationale for floor values:
#   - LOW_VOL_TREND:  drift matters, need moderate sensitivity  -> 5e-5
#   - HIGH_VOL_TREND: fast-moving drift, need high sensitivity  -> 1e-4
#   - LOW_VOL_RANGE:  small drift, low but nonzero floor        -> 2e-5
#   - HIGH_VOL_RANGE: choppy, moderate sensitivity              -> 5e-5
#   - CRISIS_JUMP:    maximum adaptivity needed                  -> 5e-4
#
# BIC adjustment when floor binds:
#   BIC_adj = BIC + Q_FLOOR_BIC_LAMBDA * log(q_floor / q_mle)
# This penalizes the deviation from MLE to prevent the floor from
# distorting model selection. Lambda = 2.0 means ~4.6 BIC penalty
# when floor is 10x above q_mle (moderate penalty).
# =============================================================================

Q_FLOOR_BY_REGIME = {
    0: 5e-5,   # LOW_VOL_TREND
    1: 1e-4,   # HIGH_VOL_TREND
    2: 2e-5,   # LOW_VOL_RANGE
    3: 5e-5,   # HIGH_VOL_RANGE
    4: 5e-4,   # CRISIS_JUMP
}

# -------------------------------------------------------------------------
# Vol-Proportional Q Floor (Story 1.7)
# -------------------------------------------------------------------------
# Q_FLOOR_BY_REGIME are calibrated for a reference equity with ~16%
# annualized vol (~1%/day).  For assets with different volatility, the
# floor scales quadratically because q is in variance (returns^2) units:
#
#   q_floor(regime, vol) = Q_FLOOR_BY_REGIME[regime] * (vol / REF_VOL)^2
#
# Clamped to [Q_FLOOR_ABS_MIN, inf) for numerical stability.
# -------------------------------------------------------------------------
REFERENCE_VOL = 0.16   # annualized vol of reference equity (~1%/day)
Q_FLOOR_ABS_MIN = 1e-8  # absolute minimum for numerical stability


def compute_vol_proportional_q_floor(regime: int, asset_vol: float) -> float:
    """
    Compute vol-proportional q floor for a given regime and asset vol.

    Args:
        regime: Regime index (0-4).
        asset_vol: Annualized asset volatility.

    Returns:
        q_floor scaled by (asset_vol / REFERENCE_VOL)^2, floored at Q_FLOOR_ABS_MIN.
    """
    q_base = Q_FLOOR_BY_REGIME.get(regime, 0.0)
    if q_base <= 0 or asset_vol <= 0:
        return max(q_base, Q_FLOOR_ABS_MIN)
    vol_ratio = asset_vol / REFERENCE_VOL
    q_scaled = q_base * vol_ratio * vol_ratio
    return max(q_scaled, Q_FLOOR_ABS_MIN)

# BIC penalty multiplier when floor overrides MLE-optimal q
Q_FLOOR_BIC_LAMBDA = 2.0

# =============================================================================
# CROSS-ASSET PHI POOLING (Story 1.3, April 2026)
# =============================================================================
# Phi (drift persistence) estimated independently per asset can produce
# nonsensical values when regime windows are short (60-200 samples):
#   - GOOGL: phi=-0.016 (anti-persistent -- floored to 0)
#   - MSFT:  phi=0.124  (fast decay -- drift gone in 7 days)
#   - SPY:   phi=0.994  (very persistent -- good)
#
# Hierarchical Bayesian pooling shrinks pathological phi values toward
# the cross-asset population median using precision-weighted averaging:
#
#   phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_pop_median) / (tau_asset + tau_pop)
#
# where tau_asset = n_samples / se_phi^2  (more data = higher precision)
#       tau_pop   = 1 / phi_pop_std^2     (tighter population = stronger pull)
#
# Default phi prior when insufficient cross-asset data:
DEFAULT_PHI_PRIOR = 0.85         # Conservative default (persistent but not unit root)
DEFAULT_PHI_PRIOR_STD = 0.15     # Moderate prior uncertainty
PHI_POOL_MIN_ASSETS = 5          # Minimum assets to form cross-asset prior

# -------------------------------------------------------------------------
# Phi ACF Lower Bound (Story 1.8)
# -------------------------------------------------------------------------
# A principled phi lower bound from the lag-1 autocorrelation:
#   phi_floor = max(0, acf_1 * PHI_ACF_SCALE)
# PHI_ACF_SCALE > 1 because acf_1 underestimates true AR(1) persistence
# due to noise contamination (Yule-Walker bias).
# -------------------------------------------------------------------------
PHI_ACF_SCALE = 2.0


def apply_cross_asset_phi_pooling(cache: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Post-processing step: shrink per-asset phi toward cross-asset population median.

    Story 3.1 upgrade: Asset-class stratified pooling. Each asset class
    (index, large_cap, small_cap, crypto, metals, forex, high_vol) gets its
    own population median. Assets pool within their class.

    Two-pass approach:
      Pass 1: Collect all asset phi values, sample counts, and asset classes.
      Pass 2: Compute per-class population prior and shrink within class.

    Hierarchical precision-weighted shrinkage formula:
      tau_asset = n_regime_samples (proxy for estimation precision)
      tau_pop   = 1 / phi_pop_std^2
      phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_class_median) / (tau_asset + tau_pop)

    Assets with many regime samples (high tau_asset) barely shrink.
    Assets with few samples (low tau_asset) shrink strongly toward class median.

    Modifies cache in-place and returns it.
    """
    # Try to import asset classification (Story 3.1)
    try:
        from models.phi_student_t_unified import _classify_asset_for_phi
        _HAS_CLASSIFY = True
    except ImportError:
        _HAS_CLASSIFY = False

    # Pass 1: collect phi values across all assets
    phi_entries = []  # (asset, phi_mle, n_samples, asset_class)
    for asset, data in cache.items():
        if not isinstance(data, dict):
            continue
        global_data = data.get("global", data)
        phi_val = global_data.get("phi")
        if phi_val is None or not np.isfinite(phi_val):
            continue
        # Estimate n_samples from regime_counts or data_length
        n_samples = 252  # Default
        regime_counts = data.get("regime_counts", {})
        if regime_counts:
            n_samples = max(sum(int(v) for v in regime_counts.values() if isinstance(v, (int, float))), 100)
        elif "data_length" in global_data:
            n_samples = int(global_data["data_length"])
        # Story 3.1: classify asset
        asset_class = _classify_asset_for_phi(asset) if _HAS_CLASSIFY else 'default'
        phi_entries.append((asset, float(phi_val), n_samples, asset_class))

    # Compute global fallback population stats
    if len(phi_entries) < PHI_POOL_MIN_ASSETS:
        global_phi_median = DEFAULT_PHI_PRIOR
        global_phi_std = DEFAULT_PHI_PRIOR_STD
    else:
        all_phi = np.array([e[1] for e in phi_entries])
        global_phi_median = float(np.median(all_phi))
        mad = float(np.median(np.abs(all_phi - global_phi_median)))
        global_phi_std = max(mad * 1.4826, 0.05)

    # Story 3.1: Compute per-class population stats
    from collections import defaultdict
    class_entries = defaultdict(list)
    for asset, phi_mle, n_samples, asset_class in phi_entries:
        class_entries[asset_class].append(phi_mle)

    class_stats = {}  # class -> (median, std)
    for cls_name, phi_list in class_entries.items():
        if len(phi_list) >= PHI_POOL_MIN_ASSETS:
            arr = np.array(phi_list)
            cls_median = float(np.median(arr))
            cls_mad = float(np.median(np.abs(arr - cls_median)))
            cls_std = max(cls_mad * 1.4826, 0.05)
            class_stats[cls_name] = (cls_median, cls_std)
        else:
            # Too few assets in this class: fall back to global
            class_stats[cls_name] = (global_phi_median, global_phi_std)

    # Pass 2: shrink each asset's phi toward its class population
    n_shrunk = 0
    for asset, phi_mle, n_samples, asset_class in phi_entries:
        phi_pop_median, phi_pop_std = class_stats.get(asset_class, (global_phi_median, global_phi_std))
        tau_pop = 1.0 / (phi_pop_std ** 2)

        # Precision proportional to sample count
        tau_asset = float(n_samples)
        tau_total = tau_asset + tau_pop

        phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_pop_median) / tau_total
        phi_shrunk = float(np.clip(phi_shrunk, -0.999, 0.999))

        # Update cache
        data = cache[asset]
        global_data = data.get("global", data)
        phi_original = global_data.get("phi")

        if abs(phi_shrunk - phi_mle) > 0.001:
            n_shrunk += 1

        global_data["phi_mle_original"] = phi_original
        global_data["phi"] = phi_shrunk

        # Also update per-regime phi values
        regime_data = data.get("regime", {})
        if isinstance(regime_data, dict):
            for r_key, r_params in regime_data.items():
                if isinstance(r_params, dict):
                    models = r_params.get("models", {})
                    if isinstance(models, dict):
                        for m_name, m_info in models.items():
                            if isinstance(m_info, dict) and "phi" in m_info:
                                m_phi = m_info["phi"]
                                if m_phi is not None and np.isfinite(m_phi):
                                    m_tau_asset = float(n_samples)
                                    m_phi_shrunk = (m_tau_asset * m_phi + tau_pop * phi_pop_median) / (m_tau_asset + tau_pop)
                                    m_info["phi_mle_original"] = m_phi
                                    m_info["phi"] = float(np.clip(m_phi_shrunk, -0.999, 0.999))

    # Store population prior in metadata (now includes per-class info)
    _pool_meta = {
        "phi_population_median": global_phi_median,
        "phi_population_std": global_phi_std,
        "n_assets_pooled": len(phi_entries),
        "n_assets_shrunk": n_shrunk,
        "class_stats": {k: {"median": v[0], "std": v[1], "n": len(class_entries[k])}
                        for k, v in class_stats.items()},
    }
    for asset in cache:
        if isinstance(cache[asset], dict):
            ht = cache[asset].setdefault("hierarchical_tuning", {})
            ht["phi_prior"] = _pool_meta

    return cache
