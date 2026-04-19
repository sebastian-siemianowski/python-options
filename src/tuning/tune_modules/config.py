"""
tune_modules/config.py -- Feature flags, conditional imports, and constants.

Extracted from tune.py to provide a single, readable configuration module.
Every feature toggle and optional dependency lives here.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup (mirrors tune.py so that conditional imports resolve)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# tune_modules/ is inside src/tuning/, so TUNING_DIR is one level up
TUNING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(TUNING_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TUNING_DIR not in sys.path:
    sys.path.insert(0, TUNING_DIR)


# =============================================================================
# MOMENTUM AUGMENTATION (February 2026)
# =============================================================================
# Momentum-augmented models compete as conditional hypotheses in BMA.
# Momentum enters model selection, not filter equations - preserves identifiability.
# Set to False to disable momentum augmentation entirely.
# =============================================================================
MOMENTUM_AUGMENTATION_ENABLED = True

# =============================================================================
# PURE GAUSSIAN MODEL DISABLING (February 2026)
# =============================================================================
# Empirical evidence shows momentum-augmented models dominate:
#   - Pure Gaussian: 3 selections (0.7%)
#   - Pure φ-Gaussian: 0 selections (0.0%)
#   - Momentum models: 432 selections (94.9%)
# Disabling pure Gaussian/φ-Gaussian reduces computational overhead and
# focuses BMA on models that actually contribute to posterior probability.
# 
# IMPORTANT: This flag is INDEPENDENT of MOMENTUM_AUGMENTATION_ENABLED.
# When both are True:
#   - Pure Gaussian and φ-Gaussian are DISABLED
#   - Gaussian+Mom and φ-Gaussian+Mom are ENABLED
#   - All Student-t variants remain enabled
# =============================================================================

# Import Adaptive ν Refinement for calibration improvement
try:
    from calibration.adaptive_nu_refinement import (
        AdaptiveNuConfig,
        AdaptiveNuRefiner,
        NuRefinementResult,
        needs_nu_refinement,
        get_refinement_candidates,
        is_nu_likelihood_flat,
        is_phi_t_model,
    )
    ADAPTIVE_NU_AVAILABLE = True
except ImportError:
    ADAPTIVE_NU_AVAILABLE = False

# =============================================================================
# GARMAN-KLASS REALIZED VOLATILITY (February 2026)
# =============================================================================
# Range-based volatility estimator using OHLC data.
# 7.4x more efficient than close-to-close EWMA.
# Improves PIT calibration without adding parameters.
# =============================================================================
try:
    from calibration.realized_volatility import (
        compute_gk_volatility,
        compute_hybrid_volatility,
        compute_hybrid_volatility_har,
        compute_volatility_from_df,
        VolatilityEstimator,
        # HAR configuration constants (February 2026)
        HAR_WEIGHT_DAILY,
        HAR_WEIGHT_WEEKLY,
        HAR_WEIGHT_MONTHLY,
        # GK c prior (Story 2.2)
        gk_c_prior,
    )
    GK_VOLATILITY_AVAILABLE = True
    HAR_VOLATILITY_AVAILABLE = True
    GK_C_PRIOR_AVAILABLE = True
except ImportError:
    GK_VOLATILITY_AVAILABLE = False
    HAR_VOLATILITY_AVAILABLE = False
    GK_C_PRIOR_AVAILABLE = False

# =============================================================================
# MARKET CONDITIONING LAYER (February 2026)
# =============================================================================
# Cross-sectional and VIX-based model enhancement for systemic risk awareness.
# σ²_composite = σ²_asset + (coupling × β²) × σ²_market
# ν_t = max(ν_min, ν_base - κ × VIX_normalized)
# =============================================================================
MARKET_CONDITIONING_ENABLED = True  # Set to False to disable

try:
    from calibration.market_conditioning import (
        compute_composite_volatility,
        compute_vix_nu_adjustment,
        condition_model_on_market,
        MarketConditioningResult,
        VIXConditioningResult,
        VIX_KAPPA_DEFAULT,
        MARKET_VOL_COUPLING_DEFAULT,
    )
    MARKET_CONDITIONING_AVAILABLE = True
except ImportError:
    MARKET_CONDITIONING_AVAILABLE = False

# =============================================================================
# UNIFIED STUDENT-T ARCHITECTURE (February 2026 - Elite Consolidation)
# =============================================================================
# When UNIFIED_STUDENT_T_ONLY is True:
#   - Only phi_student_t_nu_8 (base reference) and phi_student_t_unified_nu_{4,8,20} are fitted
#   - Disables: VoV grid, Two-Piece grid, Mixture grid, MS-q, GAS-Q Student-T
#   - Reduces model explosion from 48+ variants to 4 canonical models
#   - All enhancements (VoV, asymmetry, MS-q) are incorporated INTO unified models
#
# This is the production configuration after architectural consolidation.
# Set to False to restore legacy model grid for comparison testing.
# =============================================================================
UNIFIED_STUDENT_T_ONLY = True  # Set to False to enable legacy 48+ model variants

# =============================================================================
# GAS-Q SCORE-DRIVEN PARAMETER DYNAMICS (February 2026)
# =============================================================================
# Implements Creal, Koopman & Lucas (2013) GAS dynamics for process noise q.
# q_t = omega + alpha * s_{t-1} + beta * q_{t-1}
# where s_t is the score (derivative of log-likelihood with respect to q).
# Enables adaptive process noise during regime transitions.
# =============================================================================
GAS_Q_ENABLED = True  # Set to False to disable GAS-Q augmentation

try:
    from models.gas_q import (
        GASQConfig,
        GASQResult,
        DEFAULT_GAS_Q_CONFIG,
        gas_q_filter_gaussian,
        gas_q_filter_student_t,
        optimize_gas_q_params,
        get_gas_q_model_name,
        is_gas_q_model,
        is_gas_q_enabled,
    )
    GAS_Q_AVAILABLE = True
    
    # =========================================================================
    # GAS-Q FITTING WRAPPER FUNCTIONS
    # =========================================================================
    # Convenient wrapper functions for fitting GAS-Q models.
    # These are exported for testing and external use.
    # =========================================================================
    
    @dataclass
    class GASQFitResult:
        """Result of GAS-Q parameter optimization."""
        omega: float
        alpha: float
        beta: float
        log_likelihood: float
        ll_improvement: float
        ll_static: float
        fit_success: bool
        q_mean: float
        q_std: float
        n_obs: int
    
    def fit_gas_q_gaussian(y, sigma, c, phi, train_frac=0.7):
        """
        Fit GAS-Q parameters for Gaussian Kalman filter.
        
        Args:
            y: Observations (returns)
            sigma: Volatility series
            c: Observation scale parameter
            phi: AR(1) coefficient for drift
            train_frac: Fraction of data for training
            
        Returns:
            GASQFitResult with optimized parameters
        """
        config, diag = optimize_gas_q_params(y, sigma, c, phi, nu=None, train_frac=train_frac)
        
        # Run filter with optimized params to get log-likelihood
        result = gas_q_filter_gaussian(y, sigma, c, phi, config)
        
        # Compare with static q filter
        ll_static = diag.get("ll_static", result.log_likelihood)
        ll_improvement = result.log_likelihood - ll_static
        
        return GASQFitResult(
            omega=config.omega,
            alpha=config.alpha,
            beta=config.beta,
            log_likelihood=result.log_likelihood,
            ll_improvement=ll_improvement,
            ll_static=ll_static,
            fit_success=diag.get("fit_success", True),
            q_mean=result.q_mean,
            q_std=result.q_std,
            n_obs=len(y),
        )
    
    def fit_gas_q_student_t(y, sigma, c, phi, nu, train_frac=0.7):
        """
        Fit GAS-Q parameters for Student-t Kalman filter.
        
        Args:
            y: Observations (returns)
            sigma: Volatility series
            c: Observation scale parameter
            phi: AR(1) coefficient for drift
            nu: Degrees of freedom for Student-t
            train_frac: Fraction of data for training
            
        Returns:
            GASQFitResult with optimized parameters
        """
        config, diag = optimize_gas_q_params(y, sigma, c, phi, nu=nu, train_frac=train_frac)
        
        # Run filter with optimized params to get log-likelihood
        result = gas_q_filter_student_t(y, sigma, c, phi, nu, config)
        
        # Compare with static q filter
        ll_static = diag.get("ll_static", result.log_likelihood)
        ll_improvement = result.log_likelihood - ll_static
        
        return GASQFitResult(
            omega=config.omega,
            alpha=config.alpha,
            beta=config.beta,
            log_likelihood=result.log_likelihood,
            ll_improvement=ll_improvement,
            ll_static=ll_static,
            fit_success=diag.get("fit_success", True),
            q_mean=result.q_mean,
            q_std=result.q_std,
            n_obs=len(y),
        )
        
except ImportError as e:
    GAS_Q_AVAILABLE = False
    warnings.warn(f"GAS-Q module not available: {e}")

# =============================================================================
# RV-Q ADAPTIVE PROCESS NOISE (Tune.md Epic 1, Story 1.3)
# =============================================================================
# Proactive process noise: q_t = q_base * exp(gamma * delta_log(vol_t^2))
# Competes with static-q and GAS-Q via BMA. Data decides which wins per asset.
# =============================================================================
RV_Q_ENABLED = True

try:
    from models.rv_adaptive_q import (
        RVAdaptiveQConfig,
        RVAdaptiveQResult,
        rv_adaptive_q_filter_gaussian,
        rv_adaptive_q_filter_student_t,
        optimize_rv_q_params,
    )
    from models.model_registry import (
        make_rv_q_gaussian_name,
        make_rv_q_phi_gaussian_name,
        make_rv_q_student_t_name,
        is_rv_q_model,
    )
    RV_Q_AVAILABLE = True

    @dataclass
    class RVQFitResult:
        """Result of RV-Q parameter optimization."""
        q_base: float
        gamma: float
        log_likelihood: float
        ll_improvement: float
        ll_static: float
        bic: float
        delta_bic: float
        fit_success: bool
        n_obs: int
        oos_delta_ll: float = 0.0

    def fit_rv_q_gaussian(y, sigma, c, phi, train_frac=0.7):
        """
        Fit RV-Q parameters for phi-Gaussian Kalman filter.

        Returns RVQFitResult with optimized (q_base, gamma).
        """
        config, diag = optimize_rv_q_params(y, sigma, c, phi, nu=None, train_frac=train_frac)
        result = rv_adaptive_q_filter_gaussian(y, sigma, c, phi, config)

        return RVQFitResult(
            q_base=config.q_base,
            gamma=config.gamma,
            log_likelihood=result.log_likelihood,
            ll_improvement=diag.get("delta_ll", 0.0),
            ll_static=diag.get("ll_static", result.log_likelihood),
            bic=diag.get("bic_rv", float("inf")),
            delta_bic=diag.get("delta_bic", 0.0),
            fit_success=True,
            n_obs=len(y),
            oos_delta_ll=diag.get("oos_delta_ll", 0.0),
        )

    def fit_rv_q_student_t(y, sigma, c, phi, nu, train_frac=0.7):
        """
        Fit RV-Q parameters for phi-Student-t Kalman filter.

        Returns RVQFitResult with optimized (q_base, gamma).
        """
        config, diag = optimize_rv_q_params(y, sigma, c, phi, nu=nu, train_frac=train_frac)
        result = rv_adaptive_q_filter_student_t(y, sigma, c, phi, nu, config)

        return RVQFitResult(
            q_base=config.q_base,
            gamma=config.gamma,
            log_likelihood=result.log_likelihood,
            ll_improvement=diag.get("delta_ll", 0.0),
            ll_static=diag.get("ll_static", result.log_likelihood),
            bic=diag.get("bic_rv", float("inf")),
            delta_bic=diag.get("delta_bic", 0.0),
            fit_success=True,
            n_obs=len(y),
            oos_delta_ll=diag.get("oos_delta_ll", 0.0),
        )

except ImportError as e:
    RV_Q_AVAILABLE = False
    warnings.warn(f"RV-Q module not available: {e}")

# Import Generalized Hyperbolic (GH) distribution for calibration improvement
# GH is a fallback model when Student-t fails - captures skewness that t cannot
try:
    from calibration.gh_distribution import (
        GHModel,
        GHModelConfig,
        GHModelResult,
        should_attempt_gh,
        should_select_gh,
        compute_gh_probability,
        gh_cdf,
    )
    GH_MODEL_AVAILABLE = True
except ImportError:
    GH_MODEL_AVAILABLE = False

# Import Time-Varying Volatility Multiplier (TVVM) for calibration improvement
# TVVM addresses volatility-of-volatility effect with dynamic c_t
try:
    from calibration.tvvm_model import (
        TVVMModel,
        TVVMConfig,
        TVVMResult,
        compute_vol_of_vol,
        compute_dynamic_c,
        should_attempt_tvvm,
        should_select_tvvm,
    )
    TVVM_AVAILABLE = True
except ImportError:
    TVVM_AVAILABLE = False

# Import Isotonic Recalibration — Probability Transport Operator
# This is the CORE calibration layer - applied to ALL models BEFORE diagnostics
# Calibration is NOT a validator/patch/escalation trigger
# Calibration IS a learned probability transport map g: [0,1] → [0,1]
try:
    from calibration.isotonic_recalibration import (
        IsotonicRecalibrationConfig,
        TransportMapResult,
        IsotonicRecalibrator,
        fit_recalibrator_for_asset,
        apply_recalibration,
        compute_calibration_diagnostics,
        classify_calibration_failure,
        compute_raw_pit_gaussian,
        compute_raw_pit_student_t,
    )
    ISOTONIC_RECALIBRATION_AVAILABLE = True
except ImportError:
    ISOTONIC_RECALIBRATION_AVAILABLE = False

# Import Calibrated Trust Authority Module
# ARCHITECTURAL LAW: Trust = Calibration Authority -- Governed, Bounded Regime Penalty
# This is the SINGLE AUTHORITY for trust decisions. No other path is allowed.
try:
    from calibration.calibrated_trust import (
        CalibratedTrust,
        TrustConfig,
        compute_calibrated_trust,
        compute_drift_weight,
        create_isotonic_transport,
        MAX_REGIME_PENALTY,
        MAX_MODEL_PENALTY,
        DEFAULT_REGIME_PENALTY_SCHEDULE,
        DEFAULT_MODEL_PENALTY_SCHEDULE,
        REGIME_NAMES,
        verify_trust_architecture,
    )
    CALIBRATED_TRUST_AVAILABLE = True
except ImportError:
    CALIBRATED_TRUST_AVAILABLE = False

# Import Control Policy — Authority Boundary Layer (Counter-Proposal v1.0)
# ARCHITECTURAL LAW: Diagnostics RECOMMEND, Policy DECIDES, Models OBEY
# This is the missing layer identified in the institutional audit.
try:
    from calibration.control_policy import (
        EscalationDecision,
        CalibrationDiagnostics,
        ControlPolicy,
        AdaptiveRefinementConfig,
        TuningAuditRecord,
        EscalationStatistics,
        DECISION_NAMES,
        DEFAULT_CONTROL_POLICY,
        DEFAULT_REFINEMENT_CONFIG,
        create_diagnostics_from_result,
        verify_control_policy_architecture,
    )
    CONTROL_POLICY_AVAILABLE = True
except ImportError:
    CONTROL_POLICY_AVAILABLE = False

# =============================================================================
# IMPORT ASYMMETRIC PIT VIOLATION PENALTY MODULE (February 2026)
# =============================================================================
# PIT must only act as a penalty, never as a reward.
# This module provides regime-conditional, one-sided PIT penalties.
# =============================================================================
try:
    from calibration.pit_penalty import (
        apply_pit_penalties_to_weights,
        compute_model_pit_penalty,
        PITViolationResult,
        PITPenaltyReport,
    )
    PIT_PENALTY_AVAILABLE = True
except ImportError:
    PIT_PENALTY_AVAILABLE = False

# =============================================================================
# IMPORT FILTER RESULT CACHE (February 2026)
# =============================================================================
# Deterministic cache for Kalman filter results to eliminate redundant executions
# during CV optimization and regime-conditional tuning.
# Cache sits ABOVE Numba dispatch - works regardless of Numba availability.
# =============================================================================
try:
    from models.filter_cache import (
        get_filter_cache,
        clear_filter_cache,
        get_cache_stats,
        reset_cache_stats,
        set_filter_cache_enabled,
        FilterCacheKey,
        FilterCacheValue,
        compute_cv_likelihood_from_cache,
        FILTER_CACHE_ENABLED,
    )
    FILTER_CACHE_AVAILABLE = True
except ImportError:
    FILTER_CACHE_AVAILABLE = False

# =============================================================================
# EPIC 7: VECTORIZED OPERATIONS AND COMPUTATION CACHE (April 2026)
# =============================================================================
# Performance optimization modules for BMA weight computation and regime caching.
# vectorized_ops: NumPy-vectorized BMA weights (log-sum-exp stable softmax).
# computation_cache: Content-hash based caching for regime/vol computations.
# =============================================================================
try:
    from models.vectorized_ops import vectorized_bma_weights
    VECTORIZED_OPS_AVAILABLE = True
except ImportError:
    VECTORIZED_OPS_AVAILABLE = False

try:
    from models.computation_cache import ComputationCache
    COMPUTATION_CACHE_AVAILABLE = True
    _computation_cache = ComputationCache(max_size=500)
except ImportError:
    COMPUTATION_CACHE_AVAILABLE = False
    _computation_cache = None

# =============================================================================
# IMPORT ELITE TUNING MODULE (February 2026)
# =============================================================================
# Plateau-optimal parameter selection with:
#   1. Hessian-informed curvature penalties (stability-seeking optimization)
#   2. Cross-fold coherence scoring (temporal consistency)
#   3. Fragility index computation for early warning
#
# INSTITUTIONAL ALIGNMENT: This methodology mirrors Renaissance/DE Shaw/Two Sigma:
#   - Never optimize for peaks; optimize for stable regions
#   - Parameters that degrade gracefully > parameters that maximize in-sample
# =============================================================================
try:
    from tuning.elite_tuning import (
        EliteTuningConfig,
        EliteTuningDiagnostics,
        EliteOptimizer,
        create_elite_tuning_config,
        compute_hessian_finite_diff,
        compute_curvature_penalty,
        compute_coherence_penalty,
        compute_fragility_index,
        format_elite_diagnostics_summary,
        # v2.0 Top 0.001% upgrades
        compute_directional_curvature_penalty,
        compute_asymmetric_coherence_penalty,
        evaluate_connected_plateau,
        COUPLING_DANGER_WEIGHTS,
    )
    ELITE_TUNING_AVAILABLE = True
except ImportError:
    ELITE_TUNING_AVAILABLE = False

# Elite tuning configuration - global setting
# Set to 'balanced', 'conservative', 'aggressive', or 'diagnostic'
# v2.0 presets now include ridge detection and drift penalty
ELITE_TUNING_PRESET = 'balanced'
ELITE_TUNING_ENABLED = True  # Set to False to disable elite tuning

# =============================================================================
# IMPORT UNIFIED RISK CONTEXT (February 2026)
# =============================================================================
# Unified risk context for signal-risk architecture integration.
# Provides copula-based tail dependence and smooth scale factors.
# =============================================================================
try:
    from calibration.copula_correlation import (
        compute_unified_risk_context,
        UnifiedRiskContext,
        compute_smooth_scale_factor,
        COPULA_CORRELATION_AVAILABLE,
    )
    UNIFIED_RISK_AVAILABLE = True
except ImportError:
    UNIFIED_RISK_AVAILABLE = False
    COPULA_CORRELATION_AVAILABLE = False

# =============================================================================
# IMPORT TUNE VALIDATOR (Story 3.5 - Quality Control Gate)
# =============================================================================
try:
    from calibration.tune_validator import validate_tune_result, validate_model_params
    TUNE_VALIDATOR_AVAILABLE = True
except ImportError:
    TUNE_VALIDATOR_AVAILABLE = False

# =============================================================================
# IMPORT ENSEMBLE VALIDATOR (Story 2.10 - Benchmark Validation)
# =============================================================================
try:
    from calibration.ensemble_validator import evaluate_metrics, compare_metrics, load_baseline, save_baseline
    ENSEMBLE_VALIDATOR_AVAILABLE = True
except ImportError:
    ENSEMBLE_VALIDATOR_AVAILABLE = False

# =============================================================================
# IMPORT MODEL SELECTION UTILITIES FROM CALIBRATION
# =============================================================================
# AIC, BIC, kurtosis, and model weight functions are now in src/calibration/
# for better modularity and reuse across the codebase.
# =============================================================================
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

# =============================================================================
# MODEL REGISTRY -- Single Source of Truth
# =============================================================================
# The model registry ensures tune.py and signals.py are ALWAYS synchronised.
# This prevents the #1 silent failure: model name mismatch -> dropped from BMA.
#
# ARCHITECTURAL LAW: All model names MUST be generated via registry functions.
# Never use string literals for model names except through the registry.
# =============================================================================
try:
    from models.model_registry import (
        MODEL_REGISTRY,
        ModelFamily,
        SupportType,
        get_model_spec,
        get_all_model_names,
        get_base_model_names,
        get_augmentation_model_names,
        get_base_models_for_tuning,
        get_augmentation_layers_for_tuning,
        assert_models_synchronised,
        extract_model_params_for_sampling,
        # Name generators (CANONICAL model name construction)
        make_gaussian_name,
        make_phi_gaussian_name,
        make_student_t_name,
        # Grids
        STUDENT_T_NU_GRID,
        STUDENT_T_NU_REFINED_GRID,
        HANSEN_LAMBDA_GRID,
        HANSEN_NU_GRID,
        CST_EPSILON_GRID,
        CST_NU_PAIRS,
    )
    MODEL_REGISTRY_AVAILABLE = True
except ImportError as e:
    MODEL_REGISTRY_AVAILABLE = False
    warnings.warn(f"Model registry not available: {e}. Using legacy string-based model names.")
    
    # Fallback definitions for when registry is not available
    def make_gaussian_name() -> str:
        return "kalman_gaussian"
    
    def make_phi_gaussian_name() -> str:
        return "kalman_phi_gaussian"
    
    def make_student_t_name(nu: int) -> str:
        return f"phi_student_t_nu_{nu}"

    STUDENT_T_NU_GRID = [3, 4, 8, 20]

# =============================================================================
# CONTINUOUS NU OPTIMIZATION CONSTANTS (Story 1.5)
# =============================================================================
# After discrete grid search, refine nu via profile log-likelihood maximization.
# Holds (q, c, phi) at MLE and optimises nu alone using bounded Brent.
#
# Standard error from observed Fisher information:
#   se(nu) = 1 / sqrt( -d^2 LL / d nu^2 )
# computed via central finite differences at the MLE.
# =============================================================================
NU_CONTINUOUS_BOUNDS = (2.5, 60.0)   # search domain for nu
NU_SE_FD_STEP = 0.25                 # finite-difference step for Hessian

# =============================================================================
# IMPORT DIAGNOSTICS & REPORTING
# =============================================================================
# Scoring functions, standardization, and CLI reporting are now in separate
# modules for better separation of concerns.
# =============================================================================
from tuning.diagnostics import (
    compute_hyvarinen_score_gaussian,
    compute_hyvarinen_score_student_t,
    compute_hyvarinen_model_weights,
    robust_standardize_scores,
    entropy_regularized_weights,
    compute_combined_standardized_score,
    compute_combined_model_weights,
    compute_regime_diagnostics,
    DEFAULT_ENTROPY_LAMBDA,
    DEFAULT_MIN_WEIGHT_FRACTION,
    # CRPS computation (February 2026)
    compute_crps_gaussian_inline,
    compute_crps_student_t_inline,
    compute_crps_model_weights,
    compute_regime_aware_model_weights,
    REGIME_SCORING_WEIGHTS,
    CRPS_SCORING_ENABLED,
    # LFO-CV computation (February 2026)
    compute_lfo_cv_score_gaussian,
    compute_lfo_cv_score_student_t,
    compute_lfo_cv_model_weights,
    LFO_CV_ENABLED,
    LFO_CV_MIN_TRAIN_FRAC,
)

# PIT calibration metrics (Elite Fix - February 2026)
from tuning.pit_calibration import (
    compute_pit_calibration_metrics,
    sample_size_adjusted_pit_threshold,
    is_pit_calibrated,
)

from tuning.reporting import render_calibration_issues_table

# =============================================================================
# IMPORT MODULAR DRIFT MODELS
# =============================================================================
# Model classes are now in separate files under src/models/ for modularity.
# Each model is SELF-CONTAINED with no cross-dependencies.
#
# BMA ARCHITECTURE:
# The BMA ensemble includes the following candidate distributions:
#   - Gaussian:           mu, sigma (baseline)
#   - Symmetric Student-t: mu, sigma, nu (fat tails)
#   - phi-Skew-t:           mu, sigma, nu, gamma (fat tails + asymmetry, Fernandez-Steel)
#   - Hansen Skew-t:      mu, sigma, nu, lambda (fat tails + asymmetry, regime-conditional)
#
# CORE PRINCIPLE: "Heavy tails and asymmetry are hypotheses, not certainties."
# Complex distributions compete with simpler alternatives via BIC weights.
# =============================================================================
from models import (
    # Constants
    PHI_SHRINKAGE_TAU_MIN,
    PHI_SHRINKAGE_GLOBAL_DEFAULT,
    PHI_SHRINKAGE_LAMBDA_DEFAULT,
    STUDENT_T_NU_GRID,
    # Hansen Skew-t (Stage U-H uses local import for fit_hansen_skew_t_mle)
    # Contaminated Student-t (Stage U-C uses local import)
    # Model classes
    GaussianDriftModel,
    GaussianUnifiedConfig,
    PhiGaussianDriftModel,
    PhiStudentTDriftModel,
    # Unified Student-T Architecture (February 2026)
    UnifiedStudentTConfig,
)

# =============================================================================
# MOMENTUM AUGMENTATION MODULE (February 2026)
# =============================================================================
# Momentum-augmented drift models using compositional wrapper architecture.
# Momentum enters model selection via BMA, not filter equations.
# =============================================================================
try:
    from models.momentum_augmented import (
        MomentumConfig,
        MomentumAugmentedDriftModel,
        DEFAULT_MOMENTUM_CONFIG,
        compute_momentum_features,
        compute_momentum_signal,
        is_momentum_augmented_model,
        get_base_model_name,
        compute_momentum_model_bic_adjustment,
        compute_ablation_result,
        MomentumAblationResult,
        MOMENTUM_BMA_PRIOR_PENALTY,
        apply_phi_shrinkage_for_mr,
    )
    MOMENTUM_AUGMENTATION_AVAILABLE = True
except ImportError as e:
    MOMENTUM_AUGMENTATION_AVAILABLE = False
    warnings.warn(f"Momentum augmentation not available: {e}")

# =============================================================================
# EVT (EXTREME VALUE THEORY) FOR TAIL RISK MODELING
# =============================================================================
# Import GPD/POT for EVT-based expected loss estimation during tuning.
# This allows pre-computing optimal threshold parameters per asset.
# =============================================================================
try:
    from calibration.evt_tail import (
        fit_gpd_pot,
        compute_evt_expected_loss,
        GPDFitResult,
        EVT_THRESHOLD_PERCENTILE_DEFAULT,
        EVT_MIN_EXCEEDANCES,
        check_student_t_consistency,
        # EVT Tail Splice for PIT Calibration (February 2026)
        compute_evt_spliced_pit,
        test_evt_splice_improvement,
        EVT_SPLICE_ENABLED,
        EVT_SPLICE_PIT_IMPROVEMENT_THRESHOLD,
    )
    EVT_AVAILABLE = True
    EVT_SPLICE_AVAILABLE = True
except ImportError:
    EVT_AVAILABLE = False
    EVT_SPLICE_AVAILABLE = False
    EVT_THRESHOLD_PERCENTILE_DEFAULT = 0.90
    EVT_MIN_EXCEEDANCES = 30
    EVT_SPLICE_ENABLED = False

# =============================================================================
# REGIME-CONDITIONAL OBSERVATION NOISE (Story 2.1)
# =============================================================================
try:
    from models.regime_c import fit_regime_c, RegimeCResult
    REGIME_C_AVAILABLE = True
except ImportError:
    REGIME_C_AVAILABLE = False

# =============================================================================
# ROLLING PHI ESTIMATION (Story 3.2)
# =============================================================================
try:
    from calibration.rolling_phi import rolling_phi_estimate, RollingPhiResult
    ROLLING_PHI_AVAILABLE = True
except ImportError:
    ROLLING_PHI_AVAILABLE = False

# =============================================================================
# PHI-NU IDENTIFIABILITY CHECK (Story 3.3)
# =============================================================================
try:
    from calibration.phi_nu_identifiability import check_phi_nu_identifiability, IdentifiabilityResult
    PHI_NU_IDENTIFIABILITY_AVAILABLE = True
except ImportError:
    PHI_NU_IDENTIFIABILITY_AVAILABLE = False

# =============================================================================
# LOO-CRPS MODEL EVALUATION (Story 4.1)
# =============================================================================
try:
    from calibration.loo_crps import loo_crps_gaussian, loo_crps_student_t
    LOO_CRPS_AVAILABLE = True
except ImportError:
    LOO_CRPS_AVAILABLE = False

# =============================================================================
# CRPS STACKING WEIGHTS (Stories 4.2, 4.3)
# =============================================================================
try:
    from calibration.crps_stacking import (
        crps_stacking_weights,
        temporal_crps_stacking,
        StackingResult,
        TemporalStackingResult,
    )
    CRPS_STACKING_AVAILABLE = True
except ImportError:
    CRPS_STACKING_AVAILABLE = False

# =============================================================================
# ENTROPY-REGULARIZED BMA (Story 6.1) + MDL WEIGHTS (Story 6.2)
# =============================================================================
try:
    from calibration.entropy_bma import entropy_regularized_bma, EntropyBMAResult
    ENTROPY_BMA_AVAILABLE = True
except ImportError:
    ENTROPY_BMA_AVAILABLE = False

try:
    from calibration.entropy_bma import mdl_weights
    MDL_WEIGHTS_AVAILABLE = True
except ImportError:
    MDL_WEIGHTS_AVAILABLE = False

# =============================================================================
# VOL FUSION & HAR-GK HYBRID (Stories 7.1, 7.2, 7.3)
# =============================================================================
try:
    from calibration.realized_volatility import (
        vol_fusion_kernel,
        har_gk_hybrid,
        detect_overnight_gap,
        VolFusionResult,
        HarGkResult,
        GapDetectionResult,
    )
    VOL_FUSION_AVAILABLE = True
except ImportError:
    VOL_FUSION_AVAILABLE = False

# =============================================================================
# CONTINUOUS NU REFINEMENT (Story 8.1)
# =============================================================================
try:
    from calibration.continuous_nu import refine_nu_continuous, NuRefinementResult
    CONTINUOUS_NU_AVAILABLE = True
except ImportError:
    CONTINUOUS_NU_AVAILABLE = False

# =============================================================================
# VIX-CONDITIONAL NU (Story 8.3)
# =============================================================================
try:
    from calibration.continuous_nu import vix_conditional_nu
    VIX_CONDITIONAL_NU_AVAILABLE = True
except ImportError:
    VIX_CONDITIONAL_NU_AVAILABLE = False

# =============================================================================
# INNOVATION DIAGNOSTICS (Stories 9.2, 9.3)
# =============================================================================
try:
    from calibration.innovation_diagnostics import (
        innovation_variance_ratio,
        innovation_cusum,
        VarianceRatioResult,
        CUSUMResult,
    )
    INNOVATION_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    INNOVATION_DIAGNOSTICS_AVAILABLE = False

# =============================================================================
# GJR-GARCH MODEL (Stories 16.1, 16.2)
# =============================================================================
try:
    from models.gjr_garch import (
        fit_gjr_garch_innovations,
        iterated_filter_garch,
        IteratedFilterGARCHResult,
    )
    GJR_GARCH_AVAILABLE = True
except ImportError:
    GJR_GARCH_AVAILABLE = False

# =============================================================================
# HANSEN SKEW-T REGIME LAMBDA (Story 17.2)
# =============================================================================
try:
    from models.hansen_skew_t import regime_lambda_estimates, RegimeLambdaResult
    REGIME_LAMBDA_AVAILABLE = True
except ImportError:
    REGIME_LAMBDA_AVAILABLE = False

# =============================================================================
# REGIME CLASSIFICATION (Stories 19.1, 19.2, 19.3)
# =============================================================================
try:
    from calibration.regime_classification import (
        soft_regime_membership,
        hmm_regime_fit,
        regime_forecast_quality,
        SoftRegimeMembership,
        HMMRegimeResult,
        RegimeForecastQuality,
    )
    REGIME_CLASSIFICATION_AVAILABLE = True
except ImportError:
    REGIME_CLASSIFICATION_AVAILABLE = False

# =============================================================================
# OU MEAN REVERSION (Stories 20.1, 20.2)
# =============================================================================
try:
    from models.ou_mean_reversion import (
        multi_scale_kappa,
        detect_equilibrium_shift,
        MultiScaleKappaResult,
        ChangePointResult,
    )
    OU_MEAN_REVERSION_AVAILABLE = True
except ImportError:
    OU_MEAN_REVERSION_AVAILABLE = False

# =============================================================================
# RTS SMOOTHER & EM (Stories 21.1, 21.2, 21.3)
# =============================================================================
try:
    from models.rts_smoother import (
        rts_smoother_backward,
        em_parameter_update,
        smoothed_innovations,
        InnovationDiagnostics,
    )
    RTS_SMOOTHER_AVAILABLE = True
except ImportError:
    RTS_SMOOTHER_AVAILABLE = False

# =============================================================================
# WALK-FORWARD VALIDATION (Stories 25.2, 25.3)
# =============================================================================
try:
    from calibration.walk_forward import (
        expanding_window_train,
        detect_overfitting,
    )
    WALK_FORWARD_AVAILABLE = True
except ImportError:
    WALK_FORWARD_AVAILABLE = False

# =============================================================================
# NUMERICAL STABILITY (Story 28.3)
# =============================================================================
try:
    from calibration.numerical_stability import kahan_sum, kahan_sum_value
    KAHAN_SUM_AVAILABLE = True
except ImportError:
    KAHAN_SUM_AVAILABLE = False

# Note: Tuning presentation functions (create_tuning_console, render_tuning_header, etc.)
# are now defined in tune_ux.py to avoid circular imports. tune.py is the core tuning
# logic module and should not depend on UX presentation functions.
