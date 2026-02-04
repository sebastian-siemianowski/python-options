#!/usr/bin/env python3
"""
===============================================================================
SYSTEM DNA — TUNING LAYER (Bayesian Model Averaging Edition)
===============================================================================

This file implements the *epistemic core* of the quant system:
regime-conditional Bayesian model averaging with temporal smoothing.

The system is governed by the following probabilistic law:

    p(r_{t+H} | r)
        = Σ_m  p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)

Where:

    r          = regime label (LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE,
                              HIGH_VOL_RANGE, CRISIS_JUMP)
    m          = model class (kalman_gaussian, kalman_phi_gaussian,
                              phi_student_t_nu_{4,6,8,12,20},
                              phi_skew_t_nu_{ν}_gamma_{γ},
                              phi_nig_alpha_{α}_beta_{β})
    θ_{r,m}    = parameters of model m in regime r
    p(m | r)   = posterior probability of model m in regime r

-------------------------------------------------------------------------------
WHAT THIS FILE DOES

For EACH regime r:

    1. Fits ALL candidate model classes m independently:
       - kalman_gaussian:       q, c           (2 params) [DISABLED - use momentum version]
       - kalman_phi_gaussian:   q, c, φ        (3 params) [DISABLED - use momentum version]
       - kalman_gaussian_momentum:     q, c           (2 params) [ENABLED]
       - kalman_phi_gaussian_momentum: q, c, φ        (3 params) [ENABLED]
       - phi_student_t_nu_4:    q, c, φ        (3 params, ν=4 FIXED)
       - phi_student_t_nu_6:    q, c, φ        (3 params, ν=6 FIXED)
       - phi_student_t_nu_8:    q, c, φ        (3 params, ν=8 FIXED)
       - phi_student_t_nu_12:   q, c, φ        (3 params, ν=12 FIXED)
       - phi_student_t_nu_20:   q, c, φ        (3 params, ν=20 FIXED)
       - phi_student_t_nu_{ν}_momentum: q, c, φ (3 params, ν FIXED) [ENABLED]
       - phi_skew_t_nu_{ν}_gamma_{γ}: q, c, φ  (3 params, ν and γ FIXED)
       - phi_nig_alpha_{α}_beta_{β}: q, c, φ   (3 params, α and β FIXED)

    NOTE: Pure Gaussian and φ-Gaussian are DISABLED by default (February 2026).
    Empirical evidence shows momentum-augmented models dominate (94.9% selection rate).
    Set DISABLE_PURE_GAUSSIAN_MODELS = False to re-enable them.

    NOTE: Student-t, Skew-t, and NIG use DISCRETE grids (not continuous optimization).
    Each parameter combination is treated as a separate sub-model in BMA.

    SKEW-T ADDITION (Proposal 5 — φ-Skew-t with BMA):
    The Fernández-Steel skew-t distribution captures asymmetric return distributions:
       - γ = 1: Symmetric (reduces to Student-t)
       - γ < 1: Left-skewed (heavier left tail) — crash risk
       - γ > 1: Right-skewed (heavier right tail) — euphoria risk

    NIG ADDITION (Solution 2 — NIG as Parallel BMA Candidate):
    The Normal-Inverse Gaussian distribution provides:
       - α: Tail heaviness (smaller α = heavier tails, α→∞ = Gaussian)
       - β: Asymmetry (-α < β < α; β=0 symmetric, β<0 left-skewed, β>0 right-skewed)
    
    NIG differs from Student-t/Skew-t by having semi-heavy tails (between
    Gaussian and Cauchy) and being infinitely divisible (Lévy process compatible).

    CORE PRINCIPLE: "Heavy tails and asymmetry are hypotheses, not certainties."
    All distributional models compete via BIC weights; if extra parameters don't
    improve fit, model weight collapses toward simpler alternatives.

    2. Computes for each (r, m):
       - mean_log_likelihood
       - BIC, AIC
       - PIT calibration diagnostics

    3. Converts BIC into posterior weights:
       w_raw(m|r) = exp(-0.5 * (BIC_{m,r} - BIC_min_r))

    4. Applies temporal smoothing:
       w_smooth(m|r) = (prev_p(m|r))^α * w_raw(m|r)
       (Uses uniform prior if no previous posterior exists)

    5. Normalizes to get p(m|r)

    6. Applies hierarchical shrinkage toward global (optional)

-------------------------------------------------------------------------------
φ SHRINKAGE PRIOR (AR(1) COEFFICIENT REGULARIZATION)

Models with autoregressive drift (Phi-Gaussian, Phi-Student-t) include an
explicit Gaussian shrinkage prior on φ:

    φ_r ~ N(φ_global, τ²)
    
    log p(φ_r) = -0.5 * (φ_r - φ_global)² / τ²

Where:
    φ_global = 0 (shrinkage toward full mean reversion)
    τ = 1/√(2λ_effective)  where λ_effective = 0.05 * prior_scale

This prior:
    - Prevents unit-root instability (φ → 1)
    - Stabilizes small-sample estimation
    - Is numerically equivalent to legacy implicit regularization
    - Is explicitly auditable via phi_prior_logp diagnostics

See PHI_SHRINKAGE_* constants and phi_shrinkage_log_prior() for implementation.

-------------------------------------------------------------------------------
HIERARCHICAL FALLBACK

When regime r has insufficient samples:

    p(m|r) = p(m|global)
    θ_{r,m} = θ_{global,m}

This is correct hierarchical Bayesian shrinkage.
The regime block is marked: borrowed_from_global = True
Never returns empty models.

-------------------------------------------------------------------------------
OUTPUT FORMAT

For each regime r:

    "regime": {
        "r": {
            "model_posterior": { m: p(m|r) },
            "models": {
                m: {
                    "q", "phi", "nu", "c",
                    "mean_log_likelihood",
                    "bic", "aic",
                    "ks_statistic", "pit_ks_pvalue",
                    "fit_success", ...
                }
            },
            "regime_meta": {
                "temporal_alpha": α,
                "n_samples": N,
                "regime_name": str,
                "fallback": bool,
                "borrowed_from_global": bool,
                "shrinkage_applied": bool
            }
        }
    }

Global block remains available as fallback and for backward compatibility.

-------------------------------------------------------------------------------
CONTRACT WITH SIGNAL LAYER

The signal layer consumes this structure WITHOUT reinterpretation:

    regime_models = regime["models"]
    model_posterior = regime["model_posterior"]

Fallback is transparent. No special-case logic needed downstream.

-------------------------------------------------------------------------------
CRITICAL RULES

    • Never select a single best model per regime
    • Never discard models
    • Never force weights to zero
    • Never return empty models for a regime
    • Never mix tuning with signal logic
    • Preserve all priors, shrinkage, diagnostics
    • Preserve Bayesian coherence

-------------------------------------------------------------------------------
PHILOSOPHY

The system does NOT assume a single true market physics.

It maintains a *population of competing physics models* inside each regime.
Those models evolve in probability over time via temporal smoothing.

Regimes are ontological contexts.
Models are hypotheses about physics inside those contexts.
This file defines the epistemology of the system.

===============================================================================

tune.py

Automatic per-asset Kalman drift parameter estimation via MLE with:
- Bayesian Model Averaging across model classes
- Regime-conditional parameter tuning
- Temporal smoothing of model posteriors
- Hierarchical shrinkage toward global

Caches results persistently (JSON) for reuse across runs.

IMPORTANT AI AGENT INSTRUCTIONS: DO NOT REPLACE ELSE STATEMENTS WITH TERNARY : EXPRESSIONS
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm, kstest, t as student_t
from scipy.special import gammaln
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from enum import IntEnum

# Add repository root (parent of scripts) and scripts directory to sys.path for imports
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ingestion.data_utils import fetch_px, _download_prices, get_default_asset_universe

# K=2 Mixture Model REMOVED - empirically falsified (206 attempts, 0 selections)
# The HMM regime-switching + Student-t already captures regime heterogeneity.
# See: docs/CALIBRATION_SOLUTIONS_ANALYSIS.md for decision rationale.
MIXTURE_MODEL_AVAILABLE = False

# =============================================================================
# TEMPORARILY DISABLED MODELS (can be re-enabled by setting to True)
# =============================================================================
# These models are disabled for simplification/debugging. Set to True to re-enable.

# φ-NIG (Normal-Inverse Gaussian) - disabled for simplicity
# NIG adds α/β asymmetry parameters but rarely improves calibration over Student-t
PHI_NIG_ENABLED = False

# φ-Skew-t (Fernández-Steel) - disabled for simplicity  
# Skew-t adds γ skewness parameter but Hansen Skew-t is preferred
PHI_SKEW_T_ENABLED = False

# GMM (2-State Gaussian Mixture) - disabled for simplicity
# GMM captures bimodality but HMM regime-switching handles this
GMM_ENABLED = False

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
DISABLE_PURE_GAUSSIAN_MODELS = True

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
# ARCHITECTURAL LAW: Trust = Calibration Authority − Governed, Bounded Regime Penalty
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
# MODEL REGISTRY — Single Source of Truth
# =============================================================================
# The model registry ensures tune.py and signals.py are ALWAYS synchronised.
# This prevents the #1 silent failure: model name mismatch → dropped from BMA.
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
        make_hansen_skew_t_name,
        make_nig_name,
        make_cst_name,
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
    import warnings
    warnings.warn(f"Model registry not available: {e}. Using legacy string-based model names.")
    
    # Fallback definitions for when registry is not available
    def make_gaussian_name() -> str:
        return "kalman_gaussian"
    
    def make_phi_gaussian_name() -> str:
        return "kalman_phi_gaussian"
    
    def make_student_t_name(nu: int) -> str:
        return f"phi_student_t_nu_{nu}"
    
    STUDENT_T_NU_GRID = [4, 6, 8, 12, 20]

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
#   - Gaussian:           μ, σ (baseline)
#   - Symmetric Student-t: μ, σ, ν (fat tails)
#   - φ-Skew-t:           μ, σ, ν, γ (fat tails + asymmetry, Fernández-Steel)
#   - Hansen Skew-t:      μ, σ, ν, λ (fat tails + asymmetry, regime-conditional)
#   - NIG:                μ, σ, α, β (semi-heavy tails + asymmetry)
#   - GMM:                2-component Gaussian mixture (bimodality)
#
# CORE PRINCIPLE: "Heavy tails, asymmetry, and bimodality are hypotheses, not certainties."
# Complex distributions compete with simpler alternatives via BIC weights.
# =============================================================================
from models import (
    # Constants
    PHI_SHRINKAGE_TAU_MIN,
    PHI_SHRINKAGE_GLOBAL_DEFAULT,
    PHI_SHRINKAGE_LAMBDA_DEFAULT,
    STUDENT_T_NU_GRID,
    # NIG constants (Normal-Inverse Gaussian)
    NIG_ALPHA_GRID,
    NIG_BETA_RATIO_GRID,
    NIG_ALPHA_DEFAULT,
    NIG_BETA_DEFAULT,
    NIG_DELTA_DEFAULT,
    is_nig_model,
    get_nig_model_name,
    parse_nig_model_name,
    # GMM (2-State Gaussian Mixture)
    GaussianMixtureModel,
    fit_gmm_to_returns,
    compute_gmm_pit,
    get_gmm_model_name,
    is_gmm_model,
    GMM_MIN_OBS,
    GMM_MIN_SEPARATION,
    # Hansen Skew-t (regime-conditional asymmetry)
    HansenSkewTParams,
    fit_hansen_skew_t_mle,
    compare_symmetric_vs_hansen,
    hansen_skew_t_rvs,
    hansen_skew_t_cdf,
    HANSEN_NU_MIN,
    HANSEN_NU_MAX,
    HANSEN_NU_DEFAULT,
    HANSEN_LAMBDA_MIN,
    HANSEN_LAMBDA_MAX,
    HANSEN_LAMBDA_DEFAULT,
    HANSEN_MLE_MIN_OBS,
    # Contaminated Student-t Mixture (regime-dependent tails)
    ContaminatedStudentTParams,
    fit_contaminated_student_t_profile,
    contaminated_student_t_rvs,
    compare_contaminated_vs_single,
    compute_crisis_probability_from_vol,
    CST_NU_NORMAL_DEFAULT,
    CST_NU_CRISIS_DEFAULT,
    CST_EPSILON_DEFAULT,
    CST_MIN_OBS,
    # Model classes
    GaussianDriftModel,
    PhiGaussianDriftModel,
    PhiStudentTDriftModel,
    PhiNIGDriftModel,
    # AIGF-NF (Adaptive Implicit Generative Filter - Normalizing Flow)
    AIGFNFModel,
    AIGFNFConfig,
    DEFAULT_AIGF_NF_CONFIG,
    AIGF_NF_MODEL_NAME,
    get_aigf_nf_model_name,
    is_aigf_nf_model,
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
        DISABLED_MOMENTUM_CONFIG,
        compute_momentum_features,
        compute_momentum_signal,
        get_momentum_augmented_model_name,
        is_momentum_augmented_model,
        get_base_model_name,
        compute_momentum_model_bic_adjustment,
        compute_ablation_result,
        MomentumAblationResult,
        MOMENTUM_BMA_PRIOR_PENALTY,
    )
    MOMENTUM_AUGMENTATION_AVAILABLE = True
except ImportError as e:
    MOMENTUM_AUGMENTATION_AVAILABLE = False
    import warnings
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
    )
    EVT_AVAILABLE = True
except ImportError:
    EVT_AVAILABLE = False
    EVT_THRESHOLD_PERCENTILE_DEFAULT = 0.90
    EVT_MIN_EXCEEDANCES = 30

# Note: Tuning presentation functions (create_tuning_console, render_tuning_header, etc.)
# are now defined in tune_ux.py to avoid circular imports. tune.py is the core tuning
# logic module and should not depend on UX presentation functions.


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
    - NIG: 'phi_nig_alpha_{alpha}_beta_{beta}'
    
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
        'skew_t' in model_lower or
        'phi_nig' in model_lower
    )


# =============================================================================
# DISCRETE ν GRID FOR STUDENT-T MODELS
# =============================================================================
# DISCRETE ν GRID AND φ SHRINKAGE PRIOR — NOW IN src/models/base.py
# =============================================================================
# The following constants and functions have been moved to src/models/base.py
# for modularity and reuse across the codebase:
#
#   STUDENT_T_NU_GRID              - Discrete ν grid [4, 6, 8, 12, 20]
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
# K=2 MIXTURE MODEL - REMOVED (Empirically Falsified)
# =============================================================================
# The K=2 mixture model was removed after empirical evaluation:
#   - 206 attempts across assets, 0 selections
#   - 0% success rate indicates model misspecification
#   - Returns are fat-tailed unimodal, not bimodal
#   - HMM regime-switching + Student-t already captures regime heterogeneity
#
# Decision rationale documented in docs/CALIBRATION_SOLUTIONS_ANALYSIS.md
# Panel scoring: 92.3/100 for removal option
# =============================================================================

# Feature toggle (DISABLED - feature removed)
MIXTURE_MODEL_ENABLED = False


def get_mixture_config():
    """
    K=2 mixture model has been removed - always returns None.
    
    Reason: 206 attempts, 0 selections. The existing HMM regime-switching
    with Student-t tail modeling provides superior calibration.
    """
    return None


# =============================================================================
# GENERALIZED HYPERBOLIC (GH) DISTRIBUTION CONFIGURATION
# =============================================================================
# GH distribution is a fallback model when Student-t fails PIT calibration.
# GH captures SKEWNESS that symmetric Student-t cannot.
#
# GH is a 5-parameter family that includes Student-t, NIG, and Variance-Gamma
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
# Standard model: r_t = μ_t + √(c·σ_t²)·ε_t  (static c)
# TVVM model:     r_t = μ_t + √(c_t·σ_t²)·ε_t  (dynamic c_t)
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


class MarketRegime(IntEnum):
    """
    Market regime definitions for conditional parameter estimation.

    Regime 0: LOW_VOL_TREND
        - Low EWMA volatility
        - Strong drift persistence
        - Positive or negative trend

    Regime 1: HIGH_VOL_TREND
        - High volatility
        - Strong drift persistence
        - Large trend amplitude

    Regime 2: LOW_VOL_RANGE
        - Low volatility
        - Drift near zero
        - Mean reversion dominant

    Regime 3: HIGH_VOL_RANGE
        - High volatility
        - Drift near zero
        - Whipsaw / choppy behavior

    Regime 4: CRISIS_JUMP
        - Extreme volatility
        - Tail events / jumps
        - Correlation breakdown
    """
    LOW_VOL_TREND = 0
    HIGH_VOL_TREND = 1
    LOW_VOL_RANGE = 2
    HIGH_VOL_RANGE = 3
    CRISIS_JUMP = 4


# Regime labels for display
REGIME_LABELS = {
    MarketRegime.LOW_VOL_TREND: "LOW_VOL_TREND",
    MarketRegime.HIGH_VOL_TREND: "HIGH_VOL_TREND",
    MarketRegime.LOW_VOL_RANGE: "LOW_VOL_RANGE",
    MarketRegime.HIGH_VOL_RANGE: "HIGH_VOL_RANGE",
    MarketRegime.CRISIS_JUMP: "CRISIS_JUMP",
}

# Minimum sample size per regime for stable parameter estimation
MIN_REGIME_SAMPLES = 60

# Minimum sample size for reliable Hyvärinen score computation
# Below this threshold, Hyvärinen is DISABLED to prevent illusory smoothness
# from small-n Student-t fits creating artificially good scores
MIN_HYVARINEN_SAMPLES = 100


# =============================================================================
# REGIME CLASSIFICATION FUNCTION
# =============================================================================

def assign_regime_labels(returns: np.ndarray, vol: np.ndarray, lookback: int = 21) -> np.ndarray:
    """
    Assign market regime labels to each observation.
    
    Classification Logic:
    - CRISIS_JUMP (4): vol_relative > 2.0 OR tail_indicator > 4.0
    - HIGH_VOL_TREND (1): vol_relative > 1.3 AND drift_abs > threshold
    - HIGH_VOL_RANGE (3): vol_relative > 1.3 AND drift_abs <= threshold
    - LOW_VOL_TREND (0): vol_relative < 0.85 AND drift_abs > threshold
    - LOW_VOL_RANGE (2): vol_relative < 0.85 AND drift_abs <= threshold
    - Normal vol: based on drift threshold
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        lookback: Rolling window for feature computation (default 21 days)
        
    Returns:
        Array of regime labels (0-4) for each observation
    """
    n = len(returns)
    regime_labels = np.zeros(n, dtype=int)
    
    # Drift threshold
    drift_threshold = 0.0005  # ~0.05% daily drift threshold
    
    # Compute expanding median for volatility normalization
    vol_series = pd.Series(vol)
    vol_median_expanding = vol_series.expanding(min_periods=lookback).median().values
    
    # Handle early periods where expanding median is NaN
    vol_median_expanding[:lookback] = np.nanmedian(vol[:lookback]) if lookback <= n else np.nanmedian(vol)
    
    for t in range(n):
        # Current volatility and return
        vol_now = vol[t]
        ret_now = returns[t]
        
        # Volatility relative to expanding median
        vol_median = vol_median_expanding[t] if vol_median_expanding[t] > 1e-12 else vol_now
        vol_relative = vol_now / vol_median if vol_median > 1e-12 else 1.0
        
        # Rolling mean absolute return (drift proxy)
        start_idx = max(0, t - lookback + 1)
        drift_abs = abs(np.mean(returns[start_idx:t+1]))
        
        # Tail indicator: |return| / vol
        tail_indicator = abs(ret_now) / vol_now if vol_now > 1e-12 else 0.0
        
        # Classification logic
        # Crisis/Jump: extreme volatility or tail events
        if vol_relative > 2.0 or tail_indicator > 4.0:
            regime_labels[t] = MarketRegime.CRISIS_JUMP
        # High volatility regimes
        elif vol_relative > 1.3:
            if drift_abs > drift_threshold:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE
        # Low volatility regimes
        elif vol_relative < 0.85:
            if drift_abs > drift_threshold:
                regime_labels[t] = MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.LOW_VOL_RANGE
        # Normal volatility (between 0.85 and 1.3)
        else:
            if drift_abs > drift_threshold * 1.5:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND if vol_relative > 1.0 else MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE if vol_relative > 1.0 else MarketRegime.LOW_VOL_RANGE
    
    return regime_labels


# =============================================================================
# BASIC ASSET TUNING (WITHOUT FULL REGIME BMA)
# =============================================================================

def tune_asset_q(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
) -> Optional[Dict]:
    """
    Tune Kalman parameters for a single asset using global model fitting.
    
    This is the basic tuning function that fits all model classes globally
    (without regime-conditional BMA). Used as:
    1. Fallback when data is insufficient for regime tuning
    2. To get global parameters for hierarchical shrinkage
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        
    Returns:
        Dictionary with tuned parameters and diagnostics, or None if failed
    """
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            df = _download_prices(asset, start_date, end_date)
            if df is None or df.empty:
                _log(f"     ⚠️  No price data for {asset}")
                return None
            px = df['Close']
        
        if px is None or len(px) < 20:
            _log(f"     ⚠️  Insufficient data for {asset}")
            return None
        
        # Compute returns and volatility
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values
        
        vol_ewma = log_ret.ewm(span=21, adjust=False).std()
        vol = vol_ewma.values
        
        # Remove NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns = returns[valid_mask]
        vol = vol[valid_mask]
        
        if len(returns) < 20:
            _log(f"     ⚠️  Insufficient valid data for {asset}")
            return None
        
        n_obs = len(returns)
        
        # Fit all model classes globally
        models = fit_all_models_for_regime(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
        )
        
        # Compute model weights using BIC
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        model_weights = compute_bic_model_weights(bic_values)
        
        # Find best model by BIC
        best_model = min(bic_values.items(), key=lambda x: x[1])[0]
        best_params = models[best_model]
        
        # =====================================================================
        # FIT 2-STATE GAUSSIAN MIXTURE MODEL (GMM)
        # =====================================================================
        # GMM is fit to volatility-adjusted returns to capture bimodal behavior:
        #   - Component 1: "Momentum" regime (typically positive mean)
        #   - Component 2: "Reversal/Crisis" regime (typically negative mean)
        #
        # The GMM parameters are stored separately and used in Monte Carlo
        # simulation to improve tail behavior in Expected Utility estimation.
        #
        # CORE PRINCIPLE: "Bimodality is a hypothesis, not a certainty."
        # If GMM is degenerate (one component dominates), it falls back to
        # single Gaussian behavior.
        # =====================================================================
        gmm_result = None
        gmm_diagnostics = None
        
        # TEMPORARILY DISABLED - set GMM_ENABLED = True to re-enable
        if GMM_ENABLED:
            try:
                gmm_model, gmm_diag = fit_gmm_to_returns(returns, vol, min_obs=GMM_MIN_OBS)
                
                if gmm_model is not None and gmm_diag.get("fit_success", False):
                    # Check for degeneracy
                    if not gmm_model.is_degenerate:
                        gmm_result = gmm_model.to_dict()
                        gmm_diagnostics = gmm_diag
                        _log(f"     ✓ GMM fitted: π=[{gmm_model.weights[0]:.2f}, {gmm_model.weights[1]:.2f}], "
                             f"μ=[{gmm_model.means[0]:.3f}, {gmm_model.means[1]:.3f}], "
                             f"sep={gmm_model.separation:.2f}")
                    else:
                        _log(f"     ⚠️ GMM degenerate (one component dominates) - using single Gaussian fallback")
                        gmm_diagnostics = {"fit_success": False, "reason": "degenerate"}
                else:
                    error_msg = gmm_diag.get("error", "unknown") if gmm_diag else "fit_returned_none"
                    _log(f"     ⚠️ GMM fit failed: {error_msg} - Monte Carlo will use single Gaussian")
                    gmm_diagnostics = gmm_diag or {"fit_success": False, "error": error_msg}
            except Exception as gmm_err:
                _log(f"     ⚠️ GMM fitting exception: {gmm_err}")
                gmm_diagnostics = {"fit_success": False, "error": str(gmm_err)}
        
        # =====================================================================
        # FIT HANSEN SKEW-T DISTRIBUTION (Regime-Conditional Asymmetric Tails)
        # =====================================================================
        # Hansen (1994) skew-t captures directional asymmetry via λ parameter:
        #   - λ > 0: Right-skewed (recovery potential)
        #   - λ < 0: Left-skewed (crash risk)
        #   - λ = 0: Symmetric Student-t
        #
        # This is estimated globally and used as fallback when regime-specific
        # estimation has insufficient data.
        # =====================================================================
        hansen_skew_t_result = None
        hansen_skew_t_diagnostics = None
        hansen_comparison = None
        
        try:
            nu_hansen, lambda_hansen, ll_hansen, hansen_diag = fit_hansen_skew_t_mle(
                returns,
                nu_init=HANSEN_NU_DEFAULT,
                lambda_init=HANSEN_LAMBDA_DEFAULT,
                nu_bounds=(HANSEN_NU_MIN, HANSEN_NU_MAX),
                lambda_bounds=(HANSEN_LAMBDA_MIN, HANSEN_LAMBDA_MAX)
            )
            
            if hansen_diag.get("fit_success", False):
                # Compute comparison with symmetric Student-t
                best_nu = best_params.get("nu", HANSEN_NU_DEFAULT)
                hansen_comparison = compare_symmetric_vs_hansen(
                    returns, 
                    nu_symmetric=best_nu if best_nu else HANSEN_NU_DEFAULT,
                    nu_hansen=nu_hansen,
                    lambda_hansen=lambda_hansen
                )
                
                hansen_skew_t_result = {
                    "nu": float(nu_hansen),
                    "lambda": float(lambda_hansen),
                    "log_likelihood": float(ll_hansen),
                    "skew_direction": "left" if lambda_hansen < -0.01 else ("right" if lambda_hansen > 0.01 else "symmetric"),
                }
                hansen_skew_t_diagnostics = hansen_diag
                
                # Log the result with color-coded direction
                skew_indicator = "←" if lambda_hansen < -0.01 else ("→" if lambda_hansen > 0.01 else "—")
                preference = hansen_comparison.get("preference", "unknown")
                _log(f"     ✓ Hansen Skew-t: ν={nu_hansen:.1f}, λ={lambda_hansen:+.3f} {skew_indicator} "
                     f"| ΔAic={hansen_comparison.get('delta_aic', 0):.1f} [{preference}]")
            else:
                error_msg = hansen_diag.get("error", "unknown")
                _log(f"     ⚠️ Hansen Skew-t fit failed: {error_msg}")
                hansen_skew_t_diagnostics = hansen_diag
        except Exception as hansen_err:
            _log(f"     ⚠️ Hansen Skew-t fitting exception: {hansen_err}")
            hansen_skew_t_diagnostics = {"fit_success": False, "error": str(hansen_err)}
        
        # =====================================================================
        # FIT EVT/GPD DISTRIBUTION (Extreme Value Theory for Tail Risk)
        # =====================================================================
        # The Pickands–Balkema–de Haan theorem provides theoretical foundation:
        # exceedances over high threshold u converge to GPD distribution.
        #
        # This pre-computes optimal EVT parameters per asset for use in signals.py
        # where EVT-corrected expected loss is used for position sizing.
        #
        # Key outputs:
        #   - ξ (xi): GPD shape parameter (ξ > 0 = heavy tails, ξ = 1/ν for Student-t)
        #   - σ (sigma): GPD scale parameter
        #   - u (threshold): POT threshold (90th percentile default)
        #   - CTE: Conditional Tail Expectation = E[Loss | Loss > u]
        # =====================================================================
        evt_result = None
        evt_diagnostics = None
        evt_consistency = None
        
        if EVT_AVAILABLE:
            try:
                # Compute losses (positive values)
                losses = -returns[returns < 0]
                
                if len(losses) >= EVT_MIN_EXCEEDANCES:
                    gpd_result = fit_gpd_pot(
                        losses,
                        threshold_percentile=EVT_THRESHOLD_PERCENTILE_DEFAULT,
                        method='auto'
                    )
                    
                    if gpd_result.fit_success:
                        evt_result = gpd_result.to_dict()
                        evt_diagnostics = {
                            "fit_success": True,
                            "n_losses": len(losses),
                            "n_total_obs": n_obs,
                        }
                        
                        # Check consistency with Student-t ν
                        best_nu = best_params.get("nu")
                        if best_nu is not None:
                            evt_consistency = check_student_t_consistency(best_nu, gpd_result.xi)
                            evt_diagnostics["student_t_consistency"] = evt_consistency
                        
                        # Log result
                        xi = gpd_result.xi
                        implied_nu = gpd_result.implied_student_t_nu
                        tail_type = "heavy" if xi > 0.2 else ("moderate" if xi > 0.05 else "light")
                        nu_str = f"(≈ν={implied_nu:.0f})" if implied_nu and implied_nu < 100 else ""
                        _log(f"     ✓ EVT/GPD: ξ={xi:.3f} {nu_str} [{tail_type} tails], "
                             f"CTE={gpd_result.cte:.4f}, n_exc={gpd_result.n_exceedances}")
                    else:
                        error_msg = gpd_result.diagnostics.get("error", "unknown")
                        _log(f"     ⚠️ EVT/GPD fit failed: {error_msg}")
                        evt_diagnostics = {
                            "fit_success": False,
                            "error": error_msg,
                            "n_losses": len(losses),
                        }
                else:
                    _log(f"     ⚠️ EVT/GPD skipped: insufficient losses ({len(losses)} < {EVT_MIN_EXCEEDANCES})")
                    evt_diagnostics = {
                        "fit_success": False,
                        "error": "insufficient_losses",
                        "n_losses": len(losses),
                    }
            except Exception as evt_err:
                _log(f"     ⚠️ EVT/GPD fitting exception: {evt_err}")
                evt_diagnostics = {"fit_success": False, "error": str(evt_err)}
        else:
            evt_diagnostics = {"fit_success": False, "error": "evt_not_available"}
        
        # =====================================================================
        # FIT CONTAMINATED STUDENT-T MIXTURE (Regime-Dependent Tails)
        # =====================================================================
        # The contaminated model captures distinct fat-tail behavior in normal
        # versus stressed market conditions:
        #
        #   p(r) = (1-ε) × t(ν_normal) + ε × t(ν_crisis)
        #
        # Where:
        #   - ν_normal: Degrees of freedom for calm periods (lighter tails)
        #   - ν_crisis: Degrees of freedom for stress (heavier tails, ν_crisis < ν_normal)
        #   - ε: Contamination probability (linked to vol_regime)
        #
        # CORE PRINCIPLE: "5% of the time we're in crisis mode with ν=4,
        #                  95% of time we're normal with ν=12"
        # =====================================================================
        cst_result = None
        cst_diagnostics = None
        cst_comparison = None
        
        try:
            if n_obs >= CST_MIN_OBS:
                # Identify high-volatility observations for regime labeling
                vol_threshold = np.percentile(vol, 80)
                vol_regime_labels = (vol > vol_threshold).astype(int)
                
                # Fit contaminated Student-t mixture
                cst_params, cst_diag = fit_contaminated_student_t_profile(
                    returns,
                    vol_regime_labels=vol_regime_labels,
                )
                
                if cst_diag.get("fit_success", False):
                    cst_result = cst_params.to_dict()
                    cst_diagnostics = cst_diag
                    
                    # Compare with single Student-t
                    best_nu = best_params.get("nu", CST_NU_NORMAL_DEFAULT)
                    cst_comparison = compare_contaminated_vs_single(
                        returns,
                        cst_params,
                        single_nu=best_nu if best_nu else CST_NU_NORMAL_DEFAULT
                    )
                    
                    # Log result
                    preference = "mixture" if cst_comparison.get("mixture_preferred_bic", False) else "single"
                    _log(f"     ✓ Contaminated-t: ν_normal={cst_params.nu_normal:.0f}, "
                         f"ν_crisis={cst_params.nu_crisis:.0f}, ε={cst_params.epsilon:.1%} "
                         f"| ΔBIC={cst_diag.get('delta_bic', 0):.1f} [{preference}]")
                else:
                    error_msg = cst_diag.get("error", "unknown")
                    _log(f"     ⚠️ Contaminated-t fit failed: {error_msg}")
                    cst_diagnostics = cst_diag
            else:
                _log(f"     ⚠️ Contaminated-t skipped: insufficient data ({n_obs} < {CST_MIN_OBS})")
                cst_diagnostics = {
                    "fit_success": False,
                    "error": "insufficient_data",
                    "n_obs": n_obs,
                }
        except Exception as cst_err:
            _log(f"     ⚠️ Contaminated-t fitting exception: {cst_err}")
            cst_diagnostics = {"fit_success": False, "error": str(cst_err)}
        
        # =====================================================================
        # PIT-DRIVEN ESCALATION: ν-REFINEMENT (L1 → L2)
        # =====================================================================
        nu_refinement_result = None
        nu_refinement_attempted = False
        nu_refinement_improved = False
        
        if ADAPTIVE_NU_AVAILABLE and ADAPTIVE_NU_ENABLED:
            pit_pvalue = best_params.get("pit_ks_pvalue", 1.0)
            best_nu = best_params.get("nu")
            is_student_t = best_model.startswith("phi_student_t") if best_model else False
            pit_fails = pit_pvalue < ADAPTIVE_NU_PIT_THRESHOLD
            pit_severe = pit_pvalue < ADAPTIVE_NU_PIT_SEVERE_THRESHOLD
            
            if is_student_t and best_nu and (pit_fails or pit_severe):
                nu_refinement_attempted = True
                try:
                    candidates = ADAPTIVE_NU_CANDIDATES.get(float(best_nu), [])
                    if candidates:
                        _log(f"     🔄 ν-refinement: PIT p={pit_pvalue:.4f} → testing ν={candidates}")
                        best_refined_nu = best_nu
                        best_refined_pit = pit_pvalue
                        
                        for nu_candidate in candidates:
                            model_key = f"phi_student_t_nu_{int(nu_candidate)}"
                            if model_key in models and models[model_key].get("fit_success"):
                                cand_pit = models[model_key].get("pit_ks_pvalue", 0)
                                if cand_pit > best_refined_pit:
                                    best_refined_nu = nu_candidate
                                    best_refined_pit = cand_pit
                                    best_params = models[model_key]
                                    best_model = model_key
                                    _log(f"        ✓ ν={nu_candidate}: PIT p={cand_pit:.4f} (improved)")
                        
                        if best_refined_pit > pit_pvalue:
                            nu_refinement_improved = True
                            _log(f"     ✓ ν-refinement SUCCESS: PIT {pit_pvalue:.4f}→{best_refined_pit:.4f}")
                        
                        nu_refinement_result = {
                            "refinement_attempted": True,
                            "nu_original": best_nu,
                            "nu_final": best_refined_nu,
                            "improvement_achieved": nu_refinement_improved,
                            "pit_before": pit_pvalue,
                            "pit_after": best_refined_pit,
                        }
                except Exception as nu_err:
                    _log(f"     ⚠️ ν-refinement error: {nu_err}")
                    nu_refinement_result = {"error": str(nu_err)}
        
        # Build result structure - BMA-compatible format
        # signals.py expects: {"global": {...}, "has_bma": True}
        global_data = {
            "asset": asset,
            "q": float(best_params.get("q", 1e-6)),
            "c": float(best_params.get("c", 1.0)),
            "phi": best_params.get("phi"),
            "nu": best_params.get("nu"),
            "noise_model": best_model,
            "best_model_by_bic": best_model,
            "bic": float(best_params.get("bic", float('inf'))),
            "aic": float(best_params.get("aic", float('inf'))),
            "log_likelihood": float(best_params.get("log_likelihood", float('-inf'))),
            "mean_log_likelihood": float(best_params.get("mean_log_likelihood", float('-inf'))),
            "ks_statistic": float(best_params.get("ks_statistic", 0.0)),
            "pit_ks_pvalue": float(best_params.get("pit_ks_pvalue", 0.0)),
            "calibration_warning": best_params.get("pit_ks_pvalue", 1.0) < 0.05,
            "n_obs": n_obs,
            "model_weights": model_weights,
            "model_posterior": model_weights,  # BMA expects model_posterior
            "models": models,  # Full model details for BMA
            "model_comparison": {m: {
                "ll": models[m].get("log_likelihood", float('-inf')),
                "bic": models[m].get("bic", float('inf')),
                "aic": models[m].get("aic", float('inf')),
                "fit_success": models[m].get("fit_success", False),
            } for m in models},
            # GMM parameters for Monte Carlo simulation
            "gmm": gmm_result,
            "gmm_diagnostics": gmm_diagnostics,
            # Hansen Skew-t parameters for asymmetric tail modeling
            "hansen_skew_t": hansen_skew_t_result,
            "hansen_skew_t_diagnostics": hansen_skew_t_diagnostics,
            "hansen_vs_symmetric_comparison": hansen_comparison,
            # EVT/GPD parameters for tail risk modeling
            "evt": evt_result,
            "evt_diagnostics": evt_diagnostics,
            "evt_student_t_consistency": evt_consistency,
            # Contaminated Student-t Mixture for regime-dependent tails
            "contaminated_student_t": cst_result,
            "contaminated_student_t_diagnostics": cst_diagnostics,
            "contaminated_vs_single_comparison": cst_comparison,
            # PIT-driven ν-refinement results (L2 escalation)
            "nu_refinement": nu_refinement_result,
            "nu_refinement_attempted": nu_refinement_attempted,
            "nu_refinement_improved": nu_refinement_improved,
        }
        
        result = {
            "asset": asset,
            "has_bma": True,  # CRITICAL: signals.py checks this flag
            "global": global_data,  # BMA-compatible structure
            "regime": None,  # No regime data for basic tuning
            "use_regime_tuning": False,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        return result
        
    except Exception as e:
        import traceback
        _log(f"     ❌ {asset}: Failed - {e}")
        traceback.print_exc()
        return None


# =============================================================================
# ENTROPY REGULARIZATION CONSTANTS — IMPORTED FROM tuning.diagnostics
# =============================================================================
# DEFAULT_ENTROPY_LAMBDA and DEFAULT_MIN_WEIGHT_FRACTION are now imported
# from tuning.diagnostics to avoid duplication.
# =============================================================================


def load_asset_list(assets_arg: Optional[str], assets_file: Optional[str]) -> List[str]:
    """Load list of assets from command-line argument or file."""
    if assets_arg:
        return [a.strip() for a in assets_arg.split(',') if a.strip()]
    
    if assets_file and os.path.exists(assets_file):
        with open(assets_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Default asset list: use centralized universe from fx_data_utils
    return get_default_asset_universe()


# =============================================================================
# CACHE MANAGEMENT - Per-Asset Architecture
# =============================================================================
# Cache is now stored in individual files per asset under:
#   src/data/tune/{SYMBOL}.json
# 
# This enables:
#   - Git-friendly storage (small individual files)
#   - Parallel-safe tuning (no file lock contention)
#   - Incremental updates (re-tune one asset without touching others)
#
# Legacy single-file cache is supported for backward compatibility during migration.
# =============================================================================

# Import per-asset cache module
try:
    from tuning.kalman_cache import (
        load_tuned_params as _load_per_asset,
        save_tuned_params as _save_per_asset,
        load_full_cache as _load_full_cache,
        list_cached_symbols,
        get_cache_stats,
        TUNE_CACHE_DIR,
    )
    PER_ASSET_CACHE_AVAILABLE = True
except ImportError:
    PER_ASSET_CACHE_AVAILABLE = False


def load_cache(cache_json: str) -> Dict[str, Dict]:
    """
    Load existing cache from per-asset files or legacy single JSON file.
    
    The cache_json parameter is kept for backward compatibility but is ignored
    when per-asset cache is available. It falls back to the legacy behavior
    if the per-asset module is not found.
    
    Args:
        cache_json: Path to legacy cache file (used as fallback)
        
    Returns:
        Dict mapping symbol -> params for all cached assets
    """
    # Try per-asset cache first
    if PER_ASSET_CACHE_AVAILABLE:
        cache = _load_full_cache()
        if cache:
            return cache
    
    # Fallback to legacy single-file cache
    if os.path.exists(cache_json):
        try:
            with open(cache_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return {}
    return {}


def load_single_asset_cache(symbol: str, cache_json: str = None) -> Optional[Dict]:
    """
    Load cached parameters for a single asset.
    
    This is more efficient than load_cache() when you only need one asset.
    
    Args:
        symbol: Asset symbol
        cache_json: Path to cache directory or legacy cache file (used as fallback)
        
    Returns:
        Dict with tuned parameters or None if not cached
    """
    if PER_ASSET_CACHE_AVAILABLE:
        return _load_per_asset(symbol)
    
    # Fallback to legacy cache (only if it's a file, not a directory)
    if cache_json and os.path.isfile(cache_json):
        try:
            with open(cache_json, 'r') as f:
                cache = json.load(f)
            return cache.get(symbol)
        except Exception:
            pass
    return None


def save_cache_json(cache: Dict[str, Dict], cache_json: str) -> None:
    """
    Persist cache to per-asset files.
    
    Each asset is saved to its own JSON file in the cache directory.
    The cache_json path is used as the cache directory.
    
    Args:
        cache: Dict mapping symbol -> params
        cache_json: Path to cache directory
    """
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles numpy types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super().default(obj)
    
    # Use per-asset cache
    if PER_ASSET_CACHE_AVAILABLE:
        saved_count = 0
        for symbol, params in cache.items():
            try:
                _save_per_asset(symbol, params)
                saved_count += 1
            except Exception as e:
                print(f"Warning: Failed to save {symbol} to per-asset cache: {e}")
        return
    
    # Fallback to legacy single-file cache (only if per-asset not available)
    # Note: This should never run in normal operation since PER_ASSET_CACHE_AVAILABLE=True
    # Skip if cache_json is a directory (new architecture)
    if os.path.isdir(cache_json):
        print(f"Warning: Legacy cache save skipped - {cache_json} is a directory")
        return
        
    os.makedirs(os.path.dirname(cache_json) if os.path.dirname(cache_json) else '.', exist_ok=True)
    json_temp = cache_json + '.tmp'
    try:
        with open(json_temp, 'w') as f:
            json.dump(cache, f, indent=2, cls=NumpyEncoder)
        os.replace(json_temp, cache_json)
    finally:
        # Clean up temp file if it exists
        if os.path.exists(json_temp):
            try:
                os.remove(json_temp)
            except Exception:
                pass


# =============================================================================
# DRIFT MODEL CLASSES — NOW IN src/models/
# =============================================================================
# The following model classes have been moved to separate files for modularity:
#
#   GaussianDriftModel     → src/models/gaussian.py
#   PhiGaussianDriftModel  → src/models/phi_gaussian.py
#   PhiStudentTDriftModel  → src/models/phi_student_t.py
#
# They are imported at the top of this file from the models package.
#
# Each model class implements:
#   - filter() / filter_phi(): Run Kalman filter with model-specific dynamics
#   - optimize_params() / optimize_params_fixed_nu(): Joint parameter optimization
#   - pit_ks(): PIT/KS calibration test
#
# This refactoring:
#   - Preserves 100% backward compatibility via compatibility wrappers below
#   - Enables easier testing of individual model classes
#   - Allows extension without modifying this file
#   - Reduces tune.py file size by ~1400 lines
# =============================================================================


# Compatibility wrappers to preserve existing API surface

def kalman_filter_drift(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for GaussianDriftModel.filter()"""
    return GaussianDriftModel.filter(returns, vol, q, c)


def kalman_filter_drift_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for GaussianDriftModel.filter_phi()"""
    return GaussianDriftModel.filter_phi(returns, vol, q, c, phi)


def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
    """Compatibility wrapper for GaussianDriftModel.pit_ks()"""
    return GaussianDriftModel.pit_ks(returns, mu_filtered, vol, P_filtered, c)


def optimize_q_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-1,
    c_min: float = 0.3,
    c_max: float = 3.0,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, Dict]:
    """Delegate Gaussian q/c optimization to GaussianDriftModel for modularity."""
    return GaussianDriftModel.optimize_params(
        returns=returns,
        vol=vol,
        train_frac=train_frac,
        q_min=q_min,
        q_max=q_max,
        c_min=c_min,
        c_max=c_max,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )


def kalman_filter_drift_phi_student_t(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compatibility wrapper for PhiStudentTDriftModel.filter_phi()"""
    return PhiStudentTDriftModel.filter_phi(returns, vol, q, c, phi, nu)


def compute_pit_ks_pvalue_student_t(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
    """Compatibility wrapper for PhiStudentTDriftModel.pit_ks()"""
    return PhiStudentTDriftModel.pit_ks(returns, mu_filtered, vol, P_filtered, c, nu)


def optimize_q_c_phi_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-1,
    c_min: float = 0.3,
    c_max: float = 3.0,
    phi_min: float = -0.999,
    phi_max: float = 0.999,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, float, Dict]:
    """Delegate φ-Gaussian optimization to PhiGaussianDriftModel for modularity."""
    return PhiGaussianDriftModel.optimize_params(
        returns=returns,
        vol=vol,
        train_frac=train_frac,
        q_min=q_min,
        q_max=q_max,
        c_min=c_min,
        c_max=c_max,
        phi_min=phi_min,
        phi_max=phi_max,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )


# =============================================================================
# MODEL SELECTION FUNCTIONS — IMPORTED FROM src/calibration/model_selection.py
# =============================================================================
# The following functions are imported at the top of this file:
#   - compute_bic_model_weights_from_scores()
#   - compute_bic_model_weights()
#   - apply_temporal_smoothing()
#   - normalize_weights()
#   - DEFAULT_MODEL_SELECTION_METHOD
#   - DEFAULT_BIC_WEIGHT
#   - DEFAULT_TEMPORAL_ALPHA
# =============================================================================


def fit_all_models_for_regime(
    returns: np.ndarray,
    vol: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
) -> Dict[str, Dict]:
    """
    Fit ALL candidate model classes for a single regime's data.
    
    For each model m, computes:
        - Tuned parameters θ_{r,m}
        - Full log-likelihood
        - BIC, AIC
        - PIT calibration diagnostics
    
    DISCRETE ν GRID FOR STUDENT-T:
    We fit separate Student-t sub-models for each ν in STUDENT_T_NU_GRID.
    Each participates independently in BMA, eliminating ν-σ identifiability issues.
    
    Args:
        returns: Regime-specific returns
        vol: Regime-specific volatility
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        
    Returns:
        Dictionary with fitted models:
        {
            "kalman_gaussian": {...},
            "kalman_phi_gaussian": {...},
            "phi_student_t_nu_4": {...},
            "phi_student_t_nu_6": {...},
            "phi_student_t_nu_8": {...},
            "phi_student_t_nu_12": {...},
            "phi_student_t_nu_20": {...},
        }
    """
    n_obs = len(returns)
    models = {}
    
    # =========================================================================
    # Model 0: Kalman Gaussian (q, c)
    # =========================================================================
    # NOTE: Pure Gaussian is disabled by default (February 2026).
    # Empirical evidence shows Gaussian+Mom dominates with 40.9% selection rate
    # vs pure Gaussian at 0.7%. We still fit it to provide parameters for
    # momentum-augmented version, but mark it as disabled in BMA.
    # =========================================================================
    try:
        q_gauss, c_gauss, ll_cv_gauss, diag_gauss = GaussianDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full filter
        mu_gauss, P_gauss, ll_full_gauss = GaussianDriftModel.filter(returns, vol, q_gauss, c_gauss)
        
        # Compute PIT calibration
        ks_gauss, pit_p_gauss = GaussianDriftModel.pit_ks(returns, mu_gauss, vol, P_gauss, c_gauss)
        
        # Compute information criteria
        n_params_gauss = MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN]
        aic_gauss = compute_aic(ll_full_gauss, n_params_gauss)
        bic_gauss = compute_bic(ll_full_gauss, n_params_gauss, n_obs)
        mean_ll_gauss = ll_full_gauss / max(n_obs, 1)
        
        # Compute Hyvärinen score for robust model selection
        # Forecast std = sqrt(c * vol^2 + P) includes both observation and state uncertainty
        forecast_std_gauss = np.sqrt(c_gauss * (vol ** 2) + P_gauss)
        hyvarinen_gauss = compute_hyvarinen_score_gaussian(returns, mu_gauss, forecast_std_gauss)
        
        # Mark as disabled in BMA if DISABLE_PURE_GAUSSIAN_MODELS is True
        # Parameters are still stored for momentum-augmented version
        is_disabled = DISABLE_PURE_GAUSSIAN_MODELS
        
        models["kalman_gaussian"] = {
            "q": float(q_gauss),
            "c": float(c_gauss),
            "phi": None,
            "nu": None,
            "log_likelihood": float(ll_full_gauss),
            "mean_log_likelihood": float(mean_ll_gauss),
            "cv_penalized_ll": float(ll_cv_gauss),
            "bic": float('inf') if is_disabled else float(bic_gauss),  # Effectively remove from BMA
            "bic_original": float(bic_gauss),  # Store original for diagnostics
            "aic": float(aic_gauss),
            "hyvarinen_score": float(hyvarinen_gauss),
            "n_params": int(n_params_gauss),
            "ks_statistic": float(ks_gauss),
            "pit_ks_pvalue": float(pit_p_gauss),
            "fit_success": True,
            "disabled_in_bma": is_disabled,  # Flag for UX display
            "diagnostics": diag_gauss,
        }
    except Exception as e:
        models["kalman_gaussian"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
            "hyvarinen_score": float('-inf'),
            "disabled_in_bma": DISABLE_PURE_GAUSSIAN_MODELS,
        }
    
    # =========================================================================
    # Model 1: Phi-Gaussian (q, c, phi)
    # =========================================================================
    # NOTE: Pure φ-Gaussian is disabled by default (February 2026).
    # Empirical evidence shows φ-Gaussian+Mom dominates with 15.2% selection rate
    # vs pure φ-Gaussian at 0.0%. We still fit it to provide parameters for
    # momentum-augmented version, but mark it as disabled in BMA.
    # =========================================================================
    try:
        q_phi, c_phi, phi_opt, ll_cv_phi, diag_phi = PhiGaussianDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full filter
        mu_phi, P_phi, ll_full_phi = GaussianDriftModel.filter_phi(returns, vol, q_phi, c_phi, phi_opt)
        
        # Compute PIT calibration
        ks_phi, pit_p_phi = GaussianDriftModel.pit_ks(returns, mu_phi, vol, P_phi, c_phi)
        
        # Compute information criteria
        n_params_phi = MODEL_CLASS_N_PARAMS[ModelClass.PHI_GAUSSIAN]
        aic_phi = compute_aic(ll_full_phi, n_params_phi)
        bic_phi = compute_bic(ll_full_phi, n_params_phi, n_obs)
        mean_ll_phi = ll_full_phi / max(n_obs, 1)
        
        # Compute Hyvärinen score for robust model selection
        forecast_std_phi = np.sqrt(c_phi * (vol ** 2) + P_phi)
        hyvarinen_phi = compute_hyvarinen_score_gaussian(returns, mu_phi, forecast_std_phi)
        
        # Mark as disabled in BMA if DISABLE_PURE_GAUSSIAN_MODELS is True
        # Parameters are still stored for momentum-augmented version
        is_disabled = DISABLE_PURE_GAUSSIAN_MODELS
        
        models["kalman_phi_gaussian"] = {
            "q": float(q_phi),
            "c": float(c_phi),
            "phi": float(phi_opt),
            "nu": None,
            "log_likelihood": float(ll_full_phi),
            "mean_log_likelihood": float(mean_ll_phi),
            "cv_penalized_ll": float(ll_cv_phi),
            "bic": float('inf') if is_disabled else float(bic_phi),  # Effectively remove from BMA
            "bic_original": float(bic_phi),  # Store original for diagnostics
            "aic": float(aic_phi),
            "hyvarinen_score": float(hyvarinen_phi),
            "n_params": int(n_params_phi),
            "ks_statistic": float(ks_phi),
            "pit_ks_pvalue": float(pit_p_phi),
            "fit_success": True,
            "disabled_in_bma": is_disabled,  # Flag for UX display
            "diagnostics": diag_phi,
        }
    except Exception as e:
        models["kalman_phi_gaussian"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
            "hyvarinen_score": float('-inf'),
            "disabled_in_bma": DISABLE_PURE_GAUSSIAN_MODELS,
        }
    
    # =========================================================================
    # Model 2: Phi-Student-t with DISCRETE ν GRID
    # =========================================================================
    # Instead of continuous ν optimization, we fit separate sub-models for
    # each ν in STUDENT_T_NU_GRID. Each sub-model participates independently
    # in BMA, eliminating ν-σ identifiability issues.
    #
    # Model naming: "phi_student_t_nu_{nu}" (e.g., "phi_student_t_nu_4")
    # =========================================================================
    
    n_params_st = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T]  # 3 (q, c, phi)
    
    for nu_fixed in STUDENT_T_NU_GRID:
        model_name = f"phi_student_t_nu_{nu_fixed}"
        
        try:
            # Optimize q, c, phi with FIXED nu
            q_st, c_st, phi_st, ll_cv_st, diag_st = PhiStudentTDriftModel.optimize_params_fixed_nu(
                returns, vol,
                nu=nu_fixed,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            # Run full filter with fixed nu
            mu_st, P_st, ll_full_st = PhiStudentTDriftModel.filter_phi(
                returns, vol, q_st, c_st, phi_st, nu_fixed
            )
            
            # Compute PIT calibration
            ks_st, pit_p_st = PhiStudentTDriftModel.pit_ks(
                returns, mu_st, vol, P_st, c_st, nu_fixed
            )
            
            # Compute information criteria
            aic_st = compute_aic(ll_full_st, n_params_st)
            bic_st = compute_bic(ll_full_st, n_params_st, n_obs)
            mean_ll_st = ll_full_st / max(n_obs, 1)
            
            # Compute Hyvärinen score for robust model selection (Student-t)
            forecast_scale_st = np.sqrt(c_st * (vol ** 2) + P_st)
            hyvarinen_st = compute_hyvarinen_score_student_t(
                returns, mu_st, forecast_scale_st, nu_fixed
            )
            
            models[model_name] = {
                "q": float(q_st),
                "c": float(c_st),
                "phi": float(phi_st),
                "nu": float(nu_fixed),  # FIXED, not estimated
                "log_likelihood": float(ll_full_st),
                "mean_log_likelihood": float(mean_ll_st),
                "cv_penalized_ll": float(ll_cv_st),
                "bic": float(bic_st),
                "aic": float(aic_st),
                "hyvarinen_score": float(hyvarinen_st),
                "n_params": int(n_params_st),
                "ks_statistic": float(ks_st),
                "pit_ks_pvalue": float(pit_p_st),
                "fit_success": True,
                "diagnostics": diag_st,
                "nu_fixed": True,  # Flag indicating ν was fixed, not estimated
            }
        except Exception as e:
            models[model_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "nu": float(nu_fixed),
                "nu_fixed": True,
            }
    
    # =========================================================================
    # Model 3: Phi-NIG with DISCRETE α and β GRID (Normal-Inverse Gaussian)
    # =========================================================================
    # NIG (Normal-Inverse Gaussian) distribution captures both:
    #   - Semi-heavy tails via α parameter (smaller α = heavier tails)
    #   - Asymmetry via β parameter (β < 0 = left-skewed, β > 0 = right-skewed)
    #
    # NIG is a 4-parameter family that includes Student-t-like behavior but
    # with more flexible tail decay and native asymmetry support.
    #
    # CORE PRINCIPLE: "Heavy tails and asymmetry are hypotheses, not certainties."
    # NIG competes with simpler models via BIC — if the extra parameters don't
    # improve fit, the model weight collapses toward Gaussian/Student-t.
    #
    # DISCRETE GRID APPROACH:
    # Use discrete grids for α and β to avoid continuous optimization instability.
    # Each (α, β) combination is a separate sub-model in BMA.
    #
    # Model naming: "phi_nig_alpha_{α}_beta_{β}"
    # =========================================================================
    
    # TEMPORARILY DISABLED - set PHI_NIG_ENABLED = True to re-enable
    if PHI_NIG_ENABLED:
        # Number of parameters: q, c, phi (α, β, δ are FIXED per sub-model)
        n_params_nig = 3
        
        # Estimate baseline δ from data volatility
        delta_baseline = float(np.std(returns)) * 0.5
        delta_baseline = max(delta_baseline, 0.001)
        
        for alpha_fixed in NIG_ALPHA_GRID:
            for beta_ratio in NIG_BETA_RATIO_GRID:
                # Convert ratio to actual β (β = ratio * α, ensuring |β| < α)
                beta_fixed = beta_ratio * alpha_fixed
                
                model_name = get_nig_model_name(alpha_fixed, beta_fixed)
                
                try:
                    # Optimize q, c, phi with FIXED α, β, δ
                    q_nig, c_nig, phi_nig, ll_cv_nig, diag_nig = PhiNIGDriftModel.optimize_params_fixed_nig(
                        returns, vol,
                        alpha=alpha_fixed,
                        beta=beta_fixed,
                        delta=delta_baseline,
                        prior_log_q_mean=prior_log_q_mean,
                        prior_lambda=prior_lambda
                    )
                    
                    # Run full filter with fixed NIG parameters
                    mu_nig, P_nig, ll_full_nig = PhiNIGDriftModel.filter_phi(
                        returns, vol, q_nig, c_nig, phi_nig,
                        alpha_fixed, beta_fixed, delta_baseline
                    )
                    
                    # Compute PIT calibration using NIG CDF
                    ks_nig, pit_p_nig = PhiNIGDriftModel.pit_ks(
                        returns, mu_nig, vol, P_nig, c_nig,
                        alpha_fixed, beta_fixed, delta_baseline
                    )
                    
                    # Compute information criteria
                    aic_nig = compute_aic(ll_full_nig, n_params_nig)
                    bic_nig = compute_bic(ll_full_nig, n_params_nig, n_obs)
                    mean_ll_nig = ll_full_nig / max(n_obs, 1)
                    
                    # Compute Hyvärinen score (use Gaussian approximation for NIG)
                    # Full NIG Hyvärinen requires additional derivation
                    forecast_scale_nig = np.sqrt(c_nig * (vol ** 2) + P_nig)
                    hyvarinen_nig = compute_hyvarinen_score_gaussian(
                        returns, mu_nig, forecast_scale_nig
                    )
                    
                    models[model_name] = {
                        "q": float(q_nig),
                        "c": float(c_nig),
                        "phi": float(phi_nig),
                        "nu": None,  # NIG doesn't use ν
                        "nig_alpha": float(alpha_fixed),   # NIG tail parameter
                        "nig_beta": float(beta_fixed),    # NIG asymmetry parameter
                        "nig_delta": float(delta_baseline), # NIG scale parameter
                        "log_likelihood": float(ll_full_nig),
                        "mean_log_likelihood": float(mean_ll_nig),
                        "cv_penalized_ll": float(ll_cv_nig),
                        "bic": float(bic_nig),
                        "aic": float(aic_nig),
                        "hyvarinen_score": float(hyvarinen_nig),
                        "n_params": int(n_params_nig),
                        "ks_statistic": float(ks_nig),
                        "pit_ks_pvalue": float(pit_p_nig),
                        "fit_success": True,
                        "diagnostics": diag_nig,
                        "nig_params_fixed": True,  # Flag indicating NIG params were fixed
                        "model_type": "phi_nig",
                    }
                except Exception as e:
                    models[model_name] = {
                        "fit_success": False,
                        "error": str(e),
                        "bic": float('inf'),
                        "aic": float('inf'),
                        "hyvarinen_score": float('-inf'),
                        "nig_alpha": float(alpha_fixed),
                        "nig_beta": float(beta_fixed),
                        "nig_delta": float(delta_baseline),
                        "nig_params_fixed": True,
                        "model_type": "phi_nig",
                    }
    
    # =========================================================================
    # Model 4: AIGF-NF (Adaptive Implicit Generative Filter - Normalizing Flow)
    # =========================================================================
    # AIGF-NF is a bounded, non-parametric belief model per spec v1.0.
    # 
    # DESIGN PHILOSOPHY:
    #   Financial returns are generated by a non-stationary stochastic process
    #   whose distributional shape evolves over time. AIGF-NF removes the 
    #   functional-form commitment via normalizing flow.
    #
    # ARCHITECTURE:
    #   - Offline: Pretrained flow (frozen θ)
    #   - Online: Adaptive latent z_t via projected updates
    #
    # CONSTRAINTS (Section 4.5):
    #   - Boundedness: sup_t |z_t| < ∞
    #   - Gradient clipping: |clip(x)|_∞ ≤ 5.0
    #   - EWMA scaling: λ ∈ [0.90, 0.99]
    #   - Trust region projection
    #
    # INTEGRATION:
    #   - Competes in BMA via log-likelihood
    #   - Binary calibration pass/fail rules
    #   - No control authority, no regime logic, no governance exceptions
    #
    # Model naming: "aigf_nf"
    # =========================================================================
    
    # AIGF-NF is DISABLED - set to True to re-enable
    AIGF_NF_ENABLED = False
    
    if AIGF_NF_ENABLED:
        model_name = get_aigf_nf_model_name()
        
        try:
            # Create AIGF-NF model with default config
            aigf_model = AIGFNFModel(config=DEFAULT_AIGF_NF_CONFIG)
            
            # Fit model via filtering
            aigf_result = aigf_model.optimize_params(returns)
            
            # Extract results
            ll_full_aigf = aigf_result['mean_log_likelihood'] * n_obs
            mean_ll_aigf = aigf_result['mean_log_likelihood']
            bic_aigf = aigf_result['bic']
            aic_aigf = aigf_result['aic']
            n_params_aigf = aigf_result['n_params']
            
            # Compute Hyvärinen score approximation
            # AIGF-NF uses its predictive std as forecast scale
            forecast_std_aigf = aigf_result.get('predictive_std', 0.01) * np.ones(n_obs)
            # Use Gaussian approximation for Hyvärinen (flow is mixture)
            mu_aigf = aigf_result.get('predictive_mean', 0.0) * np.ones(n_obs)
            hyvarinen_aigf = compute_hyvarinen_score_gaussian(returns, mu_aigf, forecast_std_aigf)
            
            # PIT calibration (computed during filtering)
            # Use simple empirical PIT for now
            samples_for_pit = aigf_model.sample_predictive(n_samples=min(n_obs * 10, 10000))
            pit_values = np.array([np.mean(samples_for_pit <= r) for r in returns])
            pit_values = np.clip(pit_values, 0.001, 0.999)
            from scipy.stats import kstest
            ks_aigf, pit_p_aigf = kstest(pit_values, 'uniform')
            
            models[model_name] = {
                "q": None,  # AIGF-NF doesn't use q/c/phi/nu parameterization
                "c": None,
                "phi": None,
                "nu": None,
                "latent_z": aigf_result.get('latent_z', [0.0] * 8),
                "latent_dim": aigf_result.get('latent_dim', 8),
                "log_likelihood": float(ll_full_aigf),
                "mean_log_likelihood": float(mean_ll_aigf),
                "cv_penalized_ll": float(ll_full_aigf),  # No CV for flow
                "bic": float(bic_aigf),
                "aic": float(aic_aigf),
                "hyvarinen_score": float(hyvarinen_aigf),
                "n_params": int(n_params_aigf),
                "ks_statistic": float(ks_aigf),
                "pit_ks_pvalue": float(pit_p_aigf),
                "fit_success": True,
                "model_type": "aigf_nf",
                # AIGF-NF specific fields
                "predictive_mean": float(aigf_result.get('predictive_mean', 0.0)),
                "predictive_std": float(aigf_result.get('predictive_std', 0.01)),
                "tail_heaviness": aigf_result.get('tail_heaviness'),
                "novelty_score": aigf_result.get('novelty_score'),
                "clipping_frequency": aigf_result.get('clipping_frequency', 0.0),
                "stability_violations": aigf_result.get('stability_violations', 0),
                "flow_artifact_id": aigf_result.get('flow_artifact_id', 'default_init'),
                "flow_version": aigf_result.get('flow_version', '0.0.1'),
                "config": aigf_result.get('config', {}),
            }
        except Exception as e:
            models[model_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "model_type": "aigf_nf",
            }
    
    # =========================================================================
    # Model 5: MOMENTUM-AUGMENTED MODELS (February 2026)
    # =========================================================================
    # Momentum-augmented variants of base models compete as conditional
    # hypotheses in BMA. Momentum enters model selection, not filter equations.
    #
    # DESIGN PHILOSOPHY:
    #   "Momentum is a hypothesis, not a certainty."
    #   Momentum-augmented models receive a prior penalty in BMA.
    #   They must earn their weight through superior predictive likelihood.
    #
    # MODELS:
    #   - kalman_gaussian_momentum
    #   - kalman_phi_gaussian_momentum
    #   - phi_student_t_nu_{nu}_momentum (for each nu in grid)
    #
    # BIC ADJUSTMENT:
    #   Momentum models receive a prior penalty implemented as BIC adjustment.
    #   This prevents slow drift toward momentum dominance in noisy assets.
    # =========================================================================
    
    if MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE:
        # Create momentum wrapper
        momentum_wrapper = MomentumAugmentedDriftModel(DEFAULT_MOMENTUM_CONFIG)
        momentum_wrapper.precompute_momentum(returns)
        
        # Momentum-augmented Gaussian
        base_name = "kalman_gaussian"
        mom_name = get_momentum_augmented_model_name(base_name)
        
        if base_name in models and models[base_name].get("fit_success", False):
            try:
                base_model = models[base_name]
                q_mom = base_model["q"]
                c_mom = base_model["c"]
                
                # Run filter with momentum augmentation
                mu_mom, P_mom, ll_mom = momentum_wrapper.filter(
                    returns, vol, q_mom, c_mom, phi=1.0, base_model='gaussian'
                )
                
                # Compute PIT calibration
                ks_mom, pit_p_mom = GaussianDriftModel.pit_ks(returns, mu_mom, vol, P_mom, c_mom)
                
                # Compute information criteria with prior penalty
                n_params_mom = MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN]
                aic_mom = compute_aic(ll_mom, n_params_mom)
                bic_raw_mom = compute_bic(ll_mom, n_params_mom, n_obs)
                bic_mom = compute_momentum_model_bic_adjustment(bic_raw_mom, MOMENTUM_BMA_PRIOR_PENALTY)
                mean_ll_mom = ll_mom / max(n_obs, 1)
                
                # Compute Hyvärinen score
                forecast_std_mom = np.sqrt(c_mom * (vol ** 2) + P_mom)
                hyvarinen_mom = compute_hyvarinen_score_gaussian(returns, mu_mom, forecast_std_mom)
                
                models[mom_name] = {
                    "q": float(q_mom),
                    "c": float(c_mom),
                    "phi": None,
                    "nu": None,
                    "log_likelihood": float(ll_mom),
                    "mean_log_likelihood": float(mean_ll_mom),
                    "cv_penalized_ll": float(ll_mom),
                    "bic": float(bic_mom),
                    "bic_raw": float(bic_raw_mom),
                    "aic": float(aic_mom),
                    "hyvarinen_score": float(hyvarinen_mom),
                    "n_params": int(n_params_mom),
                    "ks_statistic": float(ks_mom),
                    "pit_ks_pvalue": float(pit_p_mom),
                    "fit_success": True,
                    "momentum_augmented": True,
                    "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                    "base_model": base_name,
                    "diagnostics": momentum_wrapper.get_diagnostics(),
                }
            except Exception as e:
                models[mom_name] = {
                    "fit_success": False,
                    "error": str(e),
                    "bic": float('inf'),
                    "aic": float('inf'),
                    "hyvarinen_score": float('-inf'),
                    "momentum_augmented": True,
                    "base_model": base_name,
                }
        
        # Momentum-augmented Phi-Gaussian
        base_name = "kalman_phi_gaussian"
        mom_name = get_momentum_augmented_model_name(base_name)
        
        if base_name in models and models[base_name].get("fit_success", False):
            try:
                base_model = models[base_name]
                q_mom = base_model["q"]
                c_mom = base_model["c"]
                phi_mom = base_model["phi"]
                
                # Run filter with momentum augmentation
                mu_mom, P_mom, ll_mom = momentum_wrapper.filter(
                    returns, vol, q_mom, c_mom, phi=phi_mom, base_model='phi_gaussian'
                )
                
                # Compute PIT calibration
                ks_mom, pit_p_mom = GaussianDriftModel.pit_ks(returns, mu_mom, vol, P_mom, c_mom)
                
                # Compute information criteria with prior penalty
                n_params_mom = MODEL_CLASS_N_PARAMS[ModelClass.PHI_GAUSSIAN]
                aic_mom = compute_aic(ll_mom, n_params_mom)
                bic_raw_mom = compute_bic(ll_mom, n_params_mom, n_obs)
                bic_mom = compute_momentum_model_bic_adjustment(bic_raw_mom, MOMENTUM_BMA_PRIOR_PENALTY)
                mean_ll_mom = ll_mom / max(n_obs, 1)
                
                # Compute Hyvärinen score
                forecast_std_mom = np.sqrt(c_mom * (vol ** 2) + P_mom)
                hyvarinen_mom = compute_hyvarinen_score_gaussian(returns, mu_mom, forecast_std_mom)
                
                models[mom_name] = {
                    "q": float(q_mom),
                    "c": float(c_mom),
                    "phi": float(phi_mom),
                    "nu": None,
                    "log_likelihood": float(ll_mom),
                    "mean_log_likelihood": float(mean_ll_mom),
                    "cv_penalized_ll": float(ll_mom),
                    "bic": float(bic_mom),
                    "bic_raw": float(bic_raw_mom),
                    "aic": float(aic_mom),
                    "hyvarinen_score": float(hyvarinen_mom),
                    "n_params": int(n_params_mom),
                    "ks_statistic": float(ks_mom),
                    "pit_ks_pvalue": float(pit_p_mom),
                    "fit_success": True,
                    "momentum_augmented": True,
                    "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                    "base_model": base_name,
                    "diagnostics": momentum_wrapper.get_diagnostics(),
                }
            except Exception as e:
                models[mom_name] = {
                    "fit_success": False,
                    "error": str(e),
                    "bic": float('inf'),
                    "aic": float('inf'),
                    "hyvarinen_score": float('-inf'),
                    "momentum_augmented": True,
                    "base_model": base_name,
                }
        
        # Momentum-augmented Student-t (for each nu in grid)
        for nu_fixed in STUDENT_T_NU_GRID:
            base_name = f"phi_student_t_nu_{nu_fixed}"
            mom_name = get_momentum_augmented_model_name(base_name)
            
            if base_name in models and models[base_name].get("fit_success", False):
                try:
                    base_model = models[base_name]
                    q_mom = base_model["q"]
                    c_mom = base_model["c"]
                    phi_mom = base_model["phi"]
                    
                    # Run filter with momentum augmentation
                    mu_mom, P_mom, ll_mom = momentum_wrapper.filter(
                        returns, vol, q_mom, c_mom, phi=phi_mom, nu=nu_fixed, 
                        base_model='phi_student_t'
                    )
                    
                    # Compute PIT calibration
                    ks_mom, pit_p_mom = PhiStudentTDriftModel.pit_ks(
                        returns, mu_mom, vol, P_mom, c_mom, nu_fixed
                    )
                    
                    # Compute information criteria with prior penalty
                    n_params_mom = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T]
                    aic_mom = compute_aic(ll_mom, n_params_mom)
                    bic_raw_mom = compute_bic(ll_mom, n_params_mom, n_obs)
                    bic_mom = compute_momentum_model_bic_adjustment(bic_raw_mom, MOMENTUM_BMA_PRIOR_PENALTY)
                    mean_ll_mom = ll_mom / max(n_obs, 1)
                    
                    # Compute Hyvärinen score (Student-t)
                    forecast_scale_mom = np.sqrt(c_mom * (vol ** 2) + P_mom)
                    hyvarinen_mom = compute_hyvarinen_score_student_t(
                        returns, mu_mom, forecast_scale_mom, nu_fixed
                    )
                    
                    models[mom_name] = {
                        "q": float(q_mom),
                        "c": float(c_mom),
                        "phi": float(phi_mom),
                        "nu": float(nu_fixed),
                        "log_likelihood": float(ll_mom),
                        "mean_log_likelihood": float(mean_ll_mom),
                        "cv_penalized_ll": float(ll_mom),
                        "bic": float(bic_mom),
                        "bic_raw": float(bic_raw_mom),
                        "aic": float(aic_mom),
                        "hyvarinen_score": float(hyvarinen_mom),
                        "n_params": int(n_params_mom),
                        "ks_statistic": float(ks_mom),
                        "pit_ks_pvalue": float(pit_p_mom),
                        "fit_success": True,
                        "momentum_augmented": True,
                        "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                        "base_model": base_name,
                        "nu_fixed": True,
                        "diagnostics": momentum_wrapper.get_diagnostics(),
                    }
                except Exception as e:
                    models[mom_name] = {
                        "fit_success": False,
                        "error": str(e),
                        "bic": float('inf'),
                        "aic": float('inf'),
                        "hyvarinen_score": float('-inf'),
                        "momentum_augmented": True,
                        "base_model": base_name,
                        "nu": float(nu_fixed),
                    }
    
    return models


def fit_regime_model_posterior(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    global_models: Optional[Dict[str, Dict]] = None,
    global_posterior: Optional[Dict[str, float]] = None,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
) -> Dict[int, Dict]:
    """
    Compute regime-conditional Bayesian model averaging with temporal smoothing.
    
    This function implements the core epistemic law:
    
        p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
    
    For EACH regime r:
    1. Fit EACH candidate model class m independently
    2. Compute mean_log_likelihood, BIC, AIC, Hyvärinen score for each (r, m)
    3. Convert scores to posterior weights using specified method:
       - 'bic': w_raw(m|r) = exp(-0.5 * ΔBIC)
       - 'hyvarinen': w_raw(m|r) = exp(ΔH) where H is negated Hyvärinen score
       - 'combined': geometric mean of BIC and Hyvärinen weights
    4. Apply temporal smoothing: w_smooth = prev_p^alpha * w_raw
    5. Normalize to get p(m|r)
    
    HIERARCHICAL FALLBACK:
    When a regime r has insufficient samples:
    - Use global_models as the regime's models (hierarchical borrowing)
    - Use global_posterior as the regime's model_posterior
    - Mark as fallback with borrowed_from_global=True
    
    This is correct hierarchical Bayesian shrinkage:
        p(m|r) = p(m|global) when data is insufficient
        θ_{r,m} = θ_{global,m} when data is insufficient
    
    CRITICAL RULES:
    - Never select a single best model per regime
    - Never discard models
    - Never force weights to zero
    - Never return empty models for a regime
    - Never mix tuning with signal logic
    - Preserve all priors, shrinkage, diagnostics
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility  
        regime_labels: Array of regime labels (0-4) for each time step
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum samples required per regime
        temporal_alpha: Smoothing exponent for model posterior evolution
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        global_models: Global model fits (for hierarchical fallback)
        global_posterior: Global model posterior (for hierarchical fallback)
        model_selection_method: Method for computing model weights:
            - 'bic': Use BIC only (traditional)
            - 'hyvarinen': Use Hyvärinen score only (robust to misspecification)
            - 'combined': Geometric mean of BIC and Hyvärinen weights (default)
        bic_weight: Weight for BIC when using 'combined' method (0-1)
        
    Returns:
        Dictionary with regime-conditional model posteriors and parameters:
        {
            r: {
                "model_posterior": { m: p(m|r) },
                "models": {
                    m: {
                        "q", "phi", "nu", "c",
                        "mean_log_likelihood",
                        "bic", "aic",
                        "ks_statistic", "pit_ks_pvalue",
                        "fit_success", ...
                    }
                },
                "regime_meta": {
                    "temporal_alpha": alpha,
                    "n_samples": n,
                    "regime_name": str,
                    "fallback": bool,
                    "borrowed_from_global": bool,
                    "shrinkage_applied": bool
                }
            }
        }
    """
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    if len(returns) != len(regime_labels):
        raise ValueError(f"Length mismatch: returns={len(returns)}, regime_labels={len(regime_labels)}")
    
    # Initialize result structure
    regime_results = {}
    
    # Process each regime
    for regime in range(5):
        regime_name = REGIME_LABELS.get(regime, f"REGIME_{regime}")
        mask = (regime_labels == regime)
        n_samples = int(np.sum(mask))
        
        _log(f"  📊 Fitting all models for {regime_name} (n={n_samples})...")
        
        # Get previous posterior for this regime (for temporal smoothing)
        prev_posterior = None
        if previous_posteriors is not None and regime in previous_posteriors:
            prev_posterior = previous_posteriors[regime]
        
        # Check if we have enough samples
        if n_samples < min_samples:
            _log(f"     ⚠️  Insufficient samples ({n_samples} < {min_samples}), using hierarchical fallback from global")
            # =========================================================================
            # HIERARCHICAL BAYESIAN FALLBACK
            # =========================================================================
            # When regime r has insufficient samples, we borrow from global:
            #   p(m|r) = p(m|global)
            #   θ_{r,m} = θ_{global,m}
            #
            # This is correct hierarchical Bayesian shrinkage, not parameter invention.
            # Never return empty models - always provide usable fallback.
            # =========================================================================
            if global_models is not None and global_posterior is not None:
                # Use global as hierarchical fallback
                regime_results[regime] = {
                    "model_posterior": global_posterior.copy(),
                    "models": global_models.copy(),
                    "regime_meta": {
                        "temporal_alpha": temporal_alpha,
                        "n_samples": n_samples,
                        "regime_name": regime_name,
                        "fallback": True,
                        "borrowed_from_global": True,
                        "fallback_reason": f"insufficient_samples_{n_samples}_lt_{min_samples}",
                    }
                }
            else:
                # =====================================================================
                # CRITICAL: No global available - this should NOT happen in normal flow
                # =====================================================================
                # tune_regime_model_averaging() always computes global first.
                # If we reach here, it's a programming error or corrupt state.
                #
                # DO NOT synthesize fake models. That violates Bayesian integrity.
                # Instead: skip this regime and let it be handled upstream.
                #
                # The correct response to missing evidence is ignorance, not invention.
                # =====================================================================
                _log(f"     ⚠️  CRITICAL: No global models for regime {regime} fallback - skipping")
                # Skip this regime - it will be missing from regime_results
                # Downstream must handle missing regimes by using global directly
            continue
        
        # Extract regime-specific data
        ret_regime = returns[mask]
        vol_regime = vol[mask]
        
        # =====================================================================
        # Step 1: Fit ALL models for this regime
        # =====================================================================
        models = fit_all_models_for_regime(
            ret_regime, vol_regime,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
        )
        
        # =====================================================================
        # Step 2: Extract BIC and Hyvärinen scores
        # =====================================================================
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        hyvarinen_scores = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models}
        
        # Print model fits
        for m, info in models.items():
            if info.get("fit_success", False):
                bic_val = info.get("bic", float('nan'))
                hyv_val = info.get("hyvarinen_score", float('nan'))
                mean_ll = info.get("mean_log_likelihood", float('nan'))
                _log(f"     {m}: BIC={bic_val:.1f}, H={hyv_val:.4f}, mean_LL={mean_ll:.4f}")
            else:
                _log(f"     {m}: FAILED - {info.get('error', 'unknown')}")
        
        # =====================================================================
        # Step 3: Compute raw weights using specified method
        # =====================================================================
        # Model selection methods:
        #   'bic'       - Traditional BIC-based weights (consistent but not robust)
        #   'hyvarinen' - Hyvärinen score weights (robust under misspecification)
        #   'combined'  - Entropy-regularized weights from BIC and Hyvärinen
        #
        # HARD GATE: Disable Hyvärinen for small-sample regimes
        # Small-n Student-t fits can produce illusory smoothness → artificially 
        # good Hyvärinen scores. This is epistemically incorrect.
        # =====================================================================
        hyvarinen_disabled = n_samples < MIN_HYVARINEN_SAMPLES
        effective_method = model_selection_method
        weight_metadata = None
        
        if hyvarinen_disabled and model_selection_method in ('hyvarinen', 'combined'):
            effective_method = 'bic'
            _log(f"     ⚠️  Hyvärinen disabled for regime {regime} (n={n_samples} < {MIN_HYVARINEN_SAMPLES}) → using BIC-only")
        
        if effective_method == 'bic':
            raw_weights = compute_bic_model_weights(bic_values)
            _log(f"     → Using BIC-only model selection")
        elif effective_method == 'hyvarinen':
            raw_weights = compute_hyvarinen_model_weights(hyvarinen_scores)
            _log(f"     → Using Hyvärinen-only model selection (robust)")
        else:
            # Default: combined method with entropy regularization
            raw_weights, weight_metadata = compute_combined_model_weights(
                bic_values, hyvarinen_scores, bic_weight=bic_weight,
                lambda_entropy=DEFAULT_ENTROPY_LAMBDA
            )
            _log(f"     → Using entropy-regularized BIC+Hyvärinen selection (α={bic_weight:.2f}, λ={DEFAULT_ENTROPY_LAMBDA:.3f})")
        
        # Store combined_score and entropy-regularized weights in each model
        for m in models:
            w = raw_weights.get(m, 1e-10)
            if weight_metadata is not None:
                # Use standardized combined score (lower = better)
                models[m]['combined_score'] = float(weight_metadata['combined_scores_standardized'].get(m, 0.0))
                models[m]['model_weight_entropy'] = float(w)
                models[m]['standardized_bic'] = float(weight_metadata['bic_standardized'].get(m, 0.0)) if weight_metadata['bic_standardized'].get(m) is not None else None
                models[m]['standardized_hyvarinen'] = float(weight_metadata['hyvarinen_standardized'].get(m, 0.0)) if weight_metadata['hyvarinen_standardized'].get(m) is not None else None
                models[m]['entropy_lambda'] = DEFAULT_ENTROPY_LAMBDA
            else:
                # Legacy: log of weight
                models[m]['combined_score'] = float(np.log(w)) if w > 0 else float('-inf')
        
        # =====================================================================
        # Step 3b: Apply Asymmetric PIT Violation Penalties (February 2026)
        # =====================================================================
        # CORE DESIGN CONSTRAINT: PIT must only act as a PENALTY, never a reward.
        # Good PIT → neutral (P=1.0, no effect)
        # Bad PIT → penalized (P<1.0, model demoted)
        # =====================================================================
        pit_penalty_report = None
        weights_pre_pit = raw_weights.copy()
        
        if PIT_PENALTY_AVAILABLE:
            # Extract PIT p-values for each model
            model_pit_pvalues = {
                m: models[m].get('pit_ks_pvalue') for m in models
            }
            
            # Apply asymmetric PIT penalties
            raw_weights, pit_penalty_report = apply_pit_penalties_to_weights(
                raw_weights=raw_weights,
                model_pit_pvalues=model_pit_pvalues,
                regime=regime,
                n_samples=n_samples,
            )
            
            # Store PIT penalty info in each model
            for m in models:
                if pit_penalty_report and m in pit_penalty_report.model_penalties:
                    penalty_result = pit_penalty_report.model_penalties[m]
                    models[m]['pit_violation_severity'] = float(penalty_result.violation_severity)
                    models[m]['pit_penalty_raw'] = float(penalty_result.raw_penalty)
                    models[m]['pit_penalty_effective'] = float(penalty_result.effective_penalty)
                    models[m]['pit_triggers_exit'] = penalty_result.triggers_exit
                    models[m]['model_weight_pre_pit'] = float(weights_pre_pit.get(m, 0.0))
                    models[m]['model_weight_post_pit'] = float(raw_weights.get(m, 0.0))
            
            # Log if PIT penalty changed model selection
            if pit_penalty_report and pit_penalty_report.selection_diverged:
                _log(f"     ⚠️  PIT penalty changed selection: {pit_penalty_report.best_model_by_fit} → {pit_penalty_report.best_model_after_penalty}")
            
            # Count and log violations
            if pit_penalty_report and pit_penalty_report.n_violated > 0:
                _log(f"     → PIT violations: {pit_penalty_report.n_violated} models penalized")
        
        # =====================================================================
        # Step 4: Apply temporal smoothing
        # =====================================================================
        smoothed_weights = apply_temporal_smoothing(raw_weights, prev_posterior, temporal_alpha)
        
        # =====================================================================
        # Step 5: Normalize to get posterior p(m|r)
        # =====================================================================
        model_posterior = normalize_weights(smoothed_weights)
        
        # Print posterior
        posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in model_posterior.items()])
        _log(f"     → Posterior: {posterior_str}")
        
        # =====================================================================
        # Build regime result
        # =====================================================================
        # Compute best scores for metadata
        finite_bics = [b for b in bic_values.values() if np.isfinite(b)]
        finite_hyvs = [h for h in hyvarinen_scores.values() if np.isfinite(h)]
        finite_combined = [models[m].get('combined_score', float('inf')) for m in models if np.isfinite(models[m].get('combined_score', float('inf')))]
        
        # Best model by combined score (lowest = best for standardized scores)
        best_model_by_combined = min(models.items(), key=lambda kv: kv[1].get('combined_score', float('inf')))[0] if models else None
        
        # Build PIT penalty metadata
        pit_penalty_meta = None
        if pit_penalty_report is not None:
            pit_penalty_meta = {
                "n_violated": pit_penalty_report.n_violated,
                "n_exit_triggered": pit_penalty_report.n_exit_triggered,
                "selection_diverged": pit_penalty_report.selection_diverged,
                "best_model_by_fit": pit_penalty_report.best_model_by_fit,
                "best_model_after_penalty": pit_penalty_report.best_model_after_penalty,
                "max_penalty_model": pit_penalty_report.max_penalty_model,
                "max_penalty_value": float(pit_penalty_report.max_penalty_value),
            }
        
        regime_results[regime] = {
            "model_posterior": model_posterior,
            "models": models,
            "regime_meta": {
                "temporal_alpha": temporal_alpha,
                "n_samples": n_samples,
                "regime_name": regime_name,
                "fallback": False,
                "borrowed_from_global": False,
                "bic_min": float(min(finite_bics)) if finite_bics else None,
                "hyvarinen_max": float(max(finite_hyvs)) if finite_hyvs else None,
                "combined_score_min": float(min(finite_combined)) if finite_combined else None,
                "best_model_by_combined": best_model_by_combined,
                "model_selection_method": model_selection_method,
                "effective_selection_method": effective_method,
                "hyvarinen_disabled": hyvarinen_disabled,
                "bic_weight": bic_weight if model_selection_method == 'combined' else None,
                "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
                "smoothing_applied": prev_posterior is not None and temporal_alpha > 0,
                # PIT Penalty metadata (February 2026)
                "pit_penalty_applied": pit_penalty_report is not None,
                "pit_penalty": pit_penalty_meta,
            }
        }
    
    return regime_results


def tune_regime_model_averaging(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    lambda_regime: float = 0.05,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
) -> Dict:
    """
    Full regime-conditional Bayesian model averaging pipeline.
    
    This is the main entry point for the upgraded tuning system.
    It combines:
    1. Global model fitting (fallback)
    2. Regime-conditional model fitting with BMA
    3. Temporal smoothing of model posteriors
    4. Hierarchical shrinkage toward global
    5. Robust model selection via Hyvärinen score (optional)
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        regime_labels: Array of regime labels (0-4)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum samples per regime
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        lambda_regime: Hierarchical shrinkage strength
        model_selection_method: 'bic', 'hyvarinen', or 'combined' (default)
        bic_weight: Weight for BIC in combined method (0-1, default 0.5)
        
    Returns:
        Dictionary with:
        {
            "global": { global model fits },
            "regime": {
                r: {
                    "model_posterior": { m: p(m|r) },
                    "models": { m: {...} },
                    "regime_meta": {...}
                }
            },
            "meta": {
                "temporal_alpha": ...,
                "lambda_regime": ...,
                ...
            }
        }
    """
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    n_obs = len(returns)
    _log(f"  📊 Bayesian Model Averaging: {n_obs} observations, α={temporal_alpha:.2f}")
    _log(f"  📊 Model selection method: {model_selection_method}" + 
         (f" (BIC weight={bic_weight:.2f})" if model_selection_method == 'combined' else ""))
    
    # =========================================================================
    # Step 1: Fit global models (fallback)
    # =========================================================================
    _log(f"  🔧 Fitting global models...")
    global_models = fit_all_models_for_regime(
        returns, vol,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )
    
    # Compute global model posterior using specified method
    global_bic = {m: global_models[m].get("bic", float('inf')) for m in global_models}
    global_hyvarinen = {m: global_models[m].get("hyvarinen_score", float('-inf')) for m in global_models}
    fallback_weight_metadata = None
    
    if model_selection_method == 'bic':
        global_raw_weights = compute_bic_model_weights(global_bic)
    elif model_selection_method == 'hyvarinen':
        global_raw_weights = compute_hyvarinen_model_weights(global_hyvarinen)
    else:
        # Default: combined with entropy regularization
        global_raw_weights, fallback_weight_metadata = compute_combined_model_weights(
            global_bic, global_hyvarinen, bic_weight=bic_weight,
            lambda_entropy=DEFAULT_ENTROPY_LAMBDA
        )
    
    # Store combined_score and entropy-regularized weights in each global model
    for m in global_models:
        w = global_raw_weights.get(m, 1e-10)
        if fallback_weight_metadata is not None:
            global_models[m]['combined_score'] = float(fallback_weight_metadata['combined_scores_standardized'].get(m, 0.0))
            global_models[m]['model_weight_entropy'] = float(w)
            global_models[m]['standardized_bic'] = float(fallback_weight_metadata['bic_standardized'].get(m, 0.0)) if fallback_weight_metadata['bic_standardized'].get(m) is not None else None
            global_models[m]['standardized_hyvarinen'] = float(fallback_weight_metadata['hyvarinen_standardized'].get(m, 0.0)) if fallback_weight_metadata['hyvarinen_standardized'].get(m) is not None else None
            global_models[m]['entropy_lambda'] = DEFAULT_ENTROPY_LAMBDA
        else:
            global_models[m]['combined_score'] = float(np.log(w)) if w > 0 else float('-inf')
    
    global_posterior = normalize_weights(global_raw_weights)
    
    _log(f"     Global posterior: " + ", ".join([f"{m}={p:.3f}" for m, p in global_posterior.items()]))
    
    # =========================================================================
    # Step 2: Fit regime-conditional models with BMA
    # =========================================================================
    _log(f"  🔄 Fitting regime-conditional models...")
    regime_results = fit_regime_model_posterior(
        returns, vol, regime_labels,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
        min_samples=MIN_REGIME_SAMPLES,
        temporal_alpha=DEFAULT_TEMPORAL_ALPHA,
        previous_posteriors=previous_posteriors,
        global_models=global_models,
        global_posterior=global_posterior,
        model_selection_method=model_selection_method,
        bic_weight=bic_weight,
    )
    
    # =========================================================================
    # Step 3: Apply hierarchical shrinkage to regime posteriors (optional)
    # =========================================================================
    if lambda_regime > 0:
        _log(f"  📐 Applying hierarchical shrinkage (λ={lambda_regime:.3f})...")
        for r, r_result in regime_results.items():
            if r_result.get("regime_meta", {}).get("fallback", False):
                continue
            
            n_samples = r_result.get("regime_meta", {}).get("n_samples", 0)
            if n_samples < min_samples:
                continue
            
            # Shrinkage factor
            sf = 1.0 / (1.0 + lambda_regime * min_samples / max(n_samples, 1.0))
            
            # Shrink model posteriors toward global
            shrunk_posterior = {}
            for m in r_result["model_posterior"]:
                p_regime = r_result["model_posterior"][m]
                p_global = global_posterior.get(m, 1.0 / 3.0)
                p_shrunk = sf * p_regime + (1 - sf) * p_global
                shrunk_posterior[m] = p_shrunk
            
            # Renormalize
            shrunk_posterior = normalize_weights(shrunk_posterior)
            r_result["model_posterior_unshrunk"] = r_result["model_posterior"]
            r_result["model_posterior"] = shrunk_posterior
            r_result["regime_meta"]["shrinkage_applied"] = True
            r_result["regime_meta"]["shrinkage_factor"] = float(sf)
    
    # =========================================================================
    # Build final result
    # =========================================================================
    # Compute global Hyvärinen max for metadata
    global_hyvarinen_scores = [
        global_models[m].get("hyvarinen_score", float('-inf')) 
        for m in global_models 
        if global_models[m].get("fit_success", False)
    ]
    global_hyvarinen_max = max(global_hyvarinen_scores) if global_hyvarinen_scores else None
    
    global_bic_scores = [
        global_models[m].get("bic", float('inf')) 
        for m in global_models 
        if global_models[m].get("fit_success", False)
    ]
    global_bic_min = min(global_bic_scores) if global_bic_scores else None
    
    # Compute global combined_score_min for metadata (lower = better for standardized scores)
    global_combined_scores = [
        global_models[m].get("combined_score", float('inf')) 
        for m in global_models 
        if global_models[m].get("fit_success", False) and np.isfinite(global_models[m].get("combined_score", float('inf')))
    ]
    global_combined_score_min = min(global_combined_scores) if global_combined_scores else None
    
    result = {
        "global": {
            "model_posterior": global_posterior,
            "models": global_models,
            "hyvarinen_max": float(global_hyvarinen_max) if global_hyvarinen_max is not None and np.isfinite(global_hyvarinen_max) else None,
            "combined_score_min": float(global_combined_score_min) if global_combined_score_min is not None and np.isfinite(global_combined_score_min) else None,
            "bic_min": float(global_bic_min) if global_bic_min is not None and np.isfinite(global_bic_min) else None,
            "model_selection_method": model_selection_method,
            "bic_weight": bic_weight if model_selection_method == 'combined' else None,
            "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
        },
        "regime": regime_results,
        "meta": {
            "temporal_alpha": temporal_alpha,
            "lambda_regime": lambda_regime,
            "n_obs": n_obs,
            "min_samples": min_samples,
            "model_selection_method": model_selection_method,
            "bic_weight": bic_weight if model_selection_method == 'combined' else None,
            "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
            "n_regimes_active": sum(1 for r in regime_results.values() 
                                    if not r.get("regime_meta", {}).get("fallback", False)),
        }
    }
    
    return result


def tune_asset_with_bma(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    lambda_regime: float = 0.05,
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
    model_selection_method: str = DEFAULT_MODEL_SELECTION_METHOD,
    bic_weight: float = DEFAULT_BIC_WEIGHT,
) -> Optional[Dict]:
    """
    Tune asset parameters using full Bayesian Model Averaging.
    
    This is the upgraded entry point that implements:
    
        p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
    
    For EACH regime r:
    - Fits ALL candidate model classes independently
    - Computes model posteriors with temporal smoothing
    - Uses robust Hyvärinen score for model selection (optional)
    - Preserves full uncertainty across models
    
    NEVER selects a single best model — maintains full posterior.
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        lambda_regime: Hierarchical shrinkage strength
        temporal_alpha: Smoothing exponent for model posteriors
        previous_posteriors: Previous model posteriors per regime (for smoothing)
        model_selection_method: 'bic', 'hyvarinen', or 'combined' (default)
        bic_weight: Weight for BIC in combined method (0-1, default 0.5)
        
    Returns:
        Dictionary with structure:
        {
            "asset": str,
            "global": {
                "model_posterior": { m: p(m) },
                "models": { m: {...} }
            },
            "regime": {
                r: {
                    "model_posterior": { m: p(m|r) },
                    "models": { m: {...} },
                    "regime_meta": {...}
                }
            },
            "regime_counts": {...},
            "meta": {...},
            "timestamp": str
        }
    """
    # Minimum data thresholds
    MIN_DATA_FOR_REGIME = 100
    MIN_DATA_FOR_GLOBAL = 20
    
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            df = _download_prices(asset, start_date, end_date)
            if df is None or df.empty:
                _log(f"     ⚠️  No price data for {asset}")
                return None
            px = df['Close']
        
        n_points = len(px) if px is not None else 0
        
        # For very small datasets, fall back directly to global-only tuning
        if n_points < MIN_DATA_FOR_GLOBAL:
            _log(f"     ⚠️  Insufficient data for {asset} ({n_points} points) - need at least {MIN_DATA_FOR_GLOBAL}")
            return None
        
        # For small-to-medium datasets (20-100 points), skip regime tuning but do global
        if n_points < MIN_DATA_FOR_REGIME:
            _log(f"     ⚠️  Insufficient data for {asset} ({n_points} points) for regime tuning")
            _log(f"     ↩️  Falling back to global-only model tuning...")
            
            # Do global tuning only
            global_result = tune_asset_q(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            if global_result is None:
                _log(f"     ⚠️  Global tuning also failed for {asset}")
                return None
            
            # Return result with explicit markers that regime tuning was skipped
            # Note: tune_asset_q now returns {"has_bma": True, "global": {...}}
            # We need to extract the inner global data
            global_data = global_result.get('global', global_result)
            return {
                "asset": asset,
                "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
                "global": global_data,
                "regime": None,  # Explicitly None - no regime params available
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_for_regime_bma",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        # Compute returns and volatility
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values

        vol_ewma = log_ret.ewm(span=21, adjust=False).std()
        vol = vol_ewma.values

        # Remove NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns = returns[valid_mask]
        vol = vol[valid_mask]

        # After cleaning, check if we still have enough data for regime tuning
        if len(returns) < MIN_DATA_FOR_REGIME:
            if len(returns) < MIN_DATA_FOR_GLOBAL:
                _log(f"     ⚠️  Insufficient valid data for {asset} after cleaning ({len(returns)} returns)")
                return None
            
            _log(f"     ⚠️  Insufficient data for {asset} after cleaning ({len(returns)} returns) for regime tuning")
            _log(f"     ↩️  Falling back to global-only model tuning...")
            
            # Do global tuning only
            global_result = tune_asset_q(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            if global_result is None:
                _log(f"     ⚠️  Global tuning also failed for {asset}")
                return None
            
            # Note: tune_asset_q now returns {"has_bma": True, "global": {...}}
            # We need to extract the inner global data
            global_data = global_result.get('global', global_result)
            return {
                "asset": asset,
                "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
                "global": global_data,
                "regime": None,
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_after_cleaning",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        # Assign regime labels
        _log(f"     📊 Assigning regime labels for {len(returns)} observations...")
        regime_labels = assign_regime_labels(returns, vol)

        # Count samples per regime
        regime_counts = {r: int(np.sum(regime_labels == r)) for r in range(5)}
        _log(f"     Regime distribution: " + ", ".join([f"{REGIME_LABELS[r]}={c}" for r, c in sorted(regime_counts.items()) if c > 0]))

        # First get global params (for backward compatibility)
        _log(f"     🔧 Estimating global parameters...")
        global_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )

        if global_result is None:
            return None

        # =================================================================
        # BAYESIAN MODEL AVERAGING: Fit ALL models for each regime
        # =================================================================
        # This implements the governing law:
        #     p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
        # =================================================================
        _log(f"     🔄 Bayesian Model Averaging (λ_regime={lambda_regime})...")
        if previous_posteriors is not None:
            _log(f"        ↪ Using previous posteriors for temporal smoothing (α={DEFAULT_TEMPORAL_ALPHA})")
        bma_result = tune_regime_model_averaging(
            returns=returns,
            vol=vol,
            regime_labels=regime_labels,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            min_samples=MIN_REGIME_SAMPLES,
            temporal_alpha=DEFAULT_TEMPORAL_ALPHA,
            previous_posteriors=previous_posteriors,  # Use provided previous posteriors
            lambda_regime=lambda_regime,
        )

        # Collect diagnostics summary
        regime_results = bma_result.get("regime", {})
        n_active = sum(1 for r, p in regime_results.items() 
                       if not p.get("regime_meta", {}).get("fallback", False))
        n_shrunk = sum(1 for r, p in regime_results.items() 
                       if p.get("regime_meta", {}).get("shrinkage_applied", False))
        collapse_warnings = 0
        for r_data in regime_results.values():
            if r_data.get("regime_meta", {}).get("collapse_warning", False):
                collapse_warnings += 1

        # Build combined result with BMA structure
        # Note: tune_asset_q now returns {"has_bma": True, "global": {...}}
        # We need to extract the inner global data
        global_data = global_result.get('global', global_result)  # Backward compatible
        
        result = {
            "asset": asset,
            "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
            "global": {
                # Keep backward-compatible global result
                **global_data,
                # Add BMA global model posterior
                "model_posterior": bma_result.get("global", {}).get("model_posterior", {}),
                "models": bma_result.get("global", {}).get("models", {}),
            },
            "regime": regime_results,  # Now contains model_posterior and models per regime
            "use_regime_tuning": True,
            "regime_counts": regime_counts,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hierarchical_tuning": {
                "lambda_regime": lambda_regime,
                "temporal_alpha": DEFAULT_TEMPORAL_ALPHA,
                "n_active_regimes": n_active,
                "n_shrunk_regimes": n_shrunk,
                "collapse_warning": collapse_warnings,
            },
            "meta": bma_result.get("meta", {}),
        }

        # Print summary
        global_posterior = result["global"].get("model_posterior", {})
        if global_posterior:
            posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in global_posterior.items()])
            _log(f"     ✓ Global model posterior: {posterior_str}")
        else:
            _log(f"     ✓ Global: q={global_result['q']:.2e}, φ={global_result.get('phi', 'N/A')}")
        
        for r, r_data in regime_results.items():
            regime_meta = r_data.get("regime_meta", {})
            if not regime_meta.get("fallback", False):
                model_posterior = r_data.get("model_posterior", {})
                posterior_str = ", ".join([f"{m}={p:.3f}" for m, p in model_posterior.items()])
                shrunk_marker = " [shrunk]" if regime_meta.get("shrinkage_applied", False) else ""
                _log(f"     ✓ {REGIME_LABELS[int(r)]}: {posterior_str}{shrunk_marker}")

        if collapse_warnings > 0:
            _log(f"     ⚠️  Collapse warnings: regime parameters too close to global")

        return result

    except Exception as e:
        import traceback
        _log(f"     ❌ {asset}: Failed - {e}")
        traceback.print_exc()
        raise  # Re-raise so caller can handle it


def _tune_worker(args_tuple: Tuple[str, str, Optional[str], float, float, float, Optional[Dict]]) -> Tuple[str, Optional[Dict], Optional[str], Optional[str]]:
    """
    Worker function for parallel asset tuning.
    Must be defined at module level for ProcessPoolExecutor pickling.
    
    Args:
        args_tuple: (asset, start_date, end_date, prior_log_q_mean, prior_lambda, lambda_regime, previous_posteriors)
        
    Returns:
        Tuple of (asset, result_dict, error_message, traceback_str)
        - If success: (asset, result, None, None)
        - If failure: (asset, None, error_string, traceback_string)
    """
    asset, start_date, end_date, prior_log_q_mean, prior_lambda, lambda_regime, previous_posteriors = args_tuple
    
    try:
        result = tune_asset_with_bma(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            lambda_regime=lambda_regime,
            previous_posteriors=previous_posteriors,
        )
        
        if result:
            return (asset, result, None, None)

        # Fallback to standard tuning when regime tuning fails (insufficient data for regime estimation)
        _log(f"  ↩️  {asset}: Falling back to standard model tuning...")
        fallback_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )

        if fallback_result:
            # tune_asset_q now returns BMA-compatible structure with global wrapper
            # Just mark as fallback for diagnostic purposes
            fallback_result['regime_fallback'] = True
            fallback_result['regime_fallback_reason'] = 'tune_asset_with_bma_returned_none'
            return (asset, fallback_result, None, None)
        else:
            return (asset, None, "both regime and standard tuning failed", None)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return (asset, None, str(e), tb_str)


def _extract_previous_posteriors(cached_entry: Optional[Dict]) -> Optional[Dict[int, Dict[str, float]]]:
    """
    Extract previous model posteriors from a cached entry for temporal smoothing.
    
    Args:
        cached_entry: Cached result for an asset (may be old or new structure)
        
    Returns:
        Dictionary mapping regime index to model posteriors, or None if not available
    """
    if cached_entry is None:
        return None
    
    regime_data = cached_entry.get("regime")
    if regime_data is None or not isinstance(regime_data, dict):
        return None
    
    previous_posteriors = {}
    for r_str, r_data in regime_data.items():
        try:
            r = int(r_str)
            model_posterior = r_data.get("model_posterior")
            if model_posterior is not None and isinstance(model_posterior, dict):
                # Validate it has expected model keys
                # Models: kalman_gaussian, kalman_phi_gaussian, phi_student_t_nu_{4,6,8,12,20}
                has_gaussian = "kalman_gaussian" in model_posterior
                has_phi_gaussian = "kalman_phi_gaussian" in model_posterior
                has_student_t = any(is_student_t_model(k) for k in model_posterior)
                
                if has_gaussian or has_phi_gaussian or has_student_t:
                    previous_posteriors[r] = model_posterior
        except (ValueError, TypeError):
            continue
    
    # Return None if no valid posteriors found
    if not previous_posteriors:
        return None
    
    return previous_posteriors


def main():
    parser = argparse.ArgumentParser(
        description="Estimate optimal Kalman drift parameters with Kalman Phi Student-t noise support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --force                          # Re-estimate all assets
  %(prog)s --max-assets 10 --dry-run        # Preview first 10 assets
  %(prog)s --prior-lambda 2.0 --prior-mean -5.5  # Custom regularization
  %(prog)s --debug                          # Enable debug output
        """
    )
    parser.add_argument('--assets', type=str, help='Comma-separated list of asset symbols')
    parser.add_argument('--assets-file', type=str, help='Path to file with asset list (one per line)')
    parser.add_argument('--cache-json', type=str, default='src/data/tune',
                       help='Path to cache directory (per-asset) or legacy JSON file')
    parser.add_argument('--force', action='store_true',
                       help='Force re-estimation even if cached values exist')
    parser.add_argument('--start', type=str, default='2015-01-01',
                       help='Start date for data fetching')
    parser.add_argument('--end', type=str, default=None,
                       help='End date for data fetching (default: today)')

    # CLI enhancements
    parser.add_argument('--max-assets', type=int, default=None,
                       help='Maximum number of assets to process (useful for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be done without actually processing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (stack traces on errors)')
    # Cache is always preserved; legacy flag kept for compatibility
    parser.add_argument('--no-clear-cache', action='store_true',
                       help='Deprecated: cache is always preserved; flag is ignored')

    # Bayesian regularization parameters
    parser.add_argument('--prior-mean', type=float, default=-6.0,
                       help='Prior mean for log10(q) (default: -6.0)')
    parser.add_argument('--prior-lambda', type=float, default=1.0,
                       help='Regularization strength (default: 1.0, set to 0 to disable)')

    # Hierarchical regime tuning parameters
    parser.add_argument('--lambda-regime', type=float, default=0.05,
                       help='Hierarchical shrinkage toward global (default: 0.05, set to 0 for original behavior)')

    args = parser.parse_args()

    # Enable debug mode
    if args.debug:
        os.environ['DEBUG'] = '1'

    print("=" * 80)
    print("Kalman Drift MLE Tuning Pipeline - Hierarchical Regime-Conditional BMA")
    print("=" * 80)
    print(f"Prior on q: log10(q) ~ N({args.prior_mean:.1f}, λ={args.prior_lambda:.1f})")
    print(f"Prior on φ: φ ~ N(0, τ) with λ_φ=0.05 (explicit Gaussian shrinkage)")
    print(f"Hierarchical shrinkage: λ_regime={args.lambda_regime:.3f}")
    print("Models: Gaussian, φ-Gaussian, φ-Student-t (ν ∈ {4, 6, 8, 12, 20})")
    if MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE:
        print(f"Momentum: ENABLED (prior penalty={args.momentum_penalty:.2f})")
    else:
        print("Momentum: DISABLED")
    print("Selection: BIC + Hyvärinen combined scoring")
    print("Regime-conditional: Fits (q, c, φ) per regime; ν is discrete grid (not optimized)")

    # Cache is always preserved; no automatic clearing

    # Load asset list
    assets = load_asset_list(args.assets, args.assets_file)

    # Apply max-assets limit
    if args.max_assets:
        assets = assets[:args.max_assets]
        print(f"\nLimited to first {args.max_assets} assets")

    print(f"Assets to process: {len(assets)}")

    # Dry-run mode
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual processing]")
        print("Would process:")
        for i, asset in enumerate(assets[:10], 1):
            print(f"  {i}. {asset}")
        if len(assets) > 10:
            print(f"  ... and {len(assets) - 10} more")
        return

    # Load existing cache
    cache = load_cache(args.cache_json)
    print(f"Loaded cache with {len(cache)} existing entries")

    # Process each asset with regime-conditional tuning
    new_estimates = 0
    reused_cached = 0
    failed = 0
    calibration_warnings = 0
    student_t_count = 0
    gaussian_count = 0
    regime_tuning_count = 0

    assets_to_process: List[str] = []
    failure_reasons: Dict[str, str] = {}
    failure_tracebacks: Dict[str, str] = {}  # Full tracebacks for failed assets
    regime_distributions: Dict[str, Dict[int, int]] = {}  # Per-asset regime counts
    processing_warnings: List[str] = []  # Collect all warnings
    model_comparisons: Dict[str, Dict] = {}  # Per-asset model comparison results

    for i, asset in enumerate(assets, 1):
        print(f"\n[{i}/{len(assets)}] {asset}")

        # Check cache - handle both old and new structure
        if not args.force and asset in cache:
            cached_entry = cache[asset]
            # Get q from either new structure or old structure
            if 'global' in cached_entry:
                cached_q = cached_entry['global'].get('q', float('nan'))
                cached_c = cached_entry['global'].get('c', 1.0)
                cached_model = cached_entry['global'].get('noise_model', 'gaussian')
                cached_nu = cached_entry['global'].get('nu')
                has_regime = 'regime' in cached_entry
            else:
                cached_q = cached_entry.get('q', float('nan'))
                cached_c = cached_entry.get('c', 1.0)
                cached_model = cached_entry.get('noise_model', 'gaussian')
                cached_nu = cached_entry.get('nu')
                has_regime = False

            if is_student_t_model(cached_model) and cached_nu is not None:
                print(f"  ✓ Using cached estimate ({cached_model}: q={cached_q:.2e}, c={cached_c:.3f}, ν={cached_nu:.1f})")
            else:
                print(f"  ✓ Using cached estimate ({cached_model}: q={cached_q:.2e}, c={cached_c:.3f})")
            if has_regime:
                print(f"     + Regime-conditional params available")
            reused_cached += 1
            continue

        assets_to_process.append(asset)

    if assets_to_process:
        # Parallel processing using all available CPU cores
        import multiprocessing
        n_workers = multiprocessing.cpu_count()
        print(f"\n🚀 Running {len(assets_to_process)} assets with parallel regime-conditional tuning ({n_workers} workers)...")

        # Prepare arguments for workers, extracting previous posteriors from cache for temporal smoothing
        worker_args = []
        for asset in assets_to_process:
            # Extract previous posteriors from cache if available (for temporal smoothing)
            prev_posteriors = _extract_previous_posteriors(cache.get(asset))
            worker_args.append(
                (asset, args.start, args.end, args.prior_mean, args.prior_lambda, args.lambda_regime, prev_posteriors)
            )

        # Process in parallel using all CPU cores
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_tune_worker, arg): arg[0] for arg in worker_args}

            for future in as_completed(futures):
                asset = futures[future]
                try:
                    asset_name, result, error, traceback_str = future.result()

                    if result:
                        cache[asset_name] = result
                        new_estimates += 1
                        regime_tuning_count += 1

                        # Collect regime distribution if available
                        if result.get('regime_counts'):
                            regime_distributions[asset_name] = result['regime_counts']

                        # Collect model comparison data if available
                        global_result = result.get('global', result)
                        if global_result.get('model_comparison'):
                            model_comparisons[asset_name] = {
                                'model_comparison': global_result['model_comparison'],
                                'selected_model': global_result.get('noise_model', 'unknown'),
                                'best_model_by_bic': global_result.get('best_model_by_bic', 'unknown'),
                                'q': global_result.get('q'),
                                'c': global_result.get('c'),
                                'phi': global_result.get('phi'),
                                'nu': global_result.get('nu'),
                                'bic': global_result.get('bic'),
                                'aic': global_result.get('aic'),
                                'log_likelihood': global_result.get('log_likelihood'),
                            }

                        # Count model type from global params
                        noise_model = global_result.get('noise_model', '')
                        if is_student_t_model(noise_model):
                            student_t_count += 1
                        else:
                            gaussian_count += 1

                        if global_result.get('calibration_warning'):
                            calibration_warnings += 1
                            processing_warnings.append(f"{asset_name}: calibration warning")

                        # Collect collapse warnings
                        if result.get('hierarchical_tuning', {}).get('collapse_warning', False):
                            processing_warnings.append(f"{asset_name}: regime collapse warning (params too close to global)")

                        # Print success summary
                        q_val = global_result.get('q', float('nan'))
                        phi_val = global_result.get('phi')
                        phi_str = f", φ={phi_val:.3f}" if phi_val is not None else ""
                        print(f"  ✓ {asset_name}: q={q_val:.2e}{phi_str}")
                    else:
                        failed += 1
                        failure_reasons[asset_name] = error or "tuning returned None"
                        if traceback_str:
                            failure_tracebacks[asset_name] = traceback_str
                        print(f"  ❌ {asset_name}: {error or 'tuning returned None'}")

                except Exception as e:
                    import traceback
                    failed += 1
                    failure_reasons[asset] = str(e)
                    failure_tracebacks[asset] = traceback.format_exc()
                    print(f"  ❌ {asset}: {e}")
    else:
        print("\nNo assets to process (all reused from cache).")

    # Save updated cache (JSON only)
    if new_estimates > 0:
        save_cache_json(cache, args.cache_json)
        print(f"\n✓ Cache updated: {args.cache_json}")

    # Summary report
    print("\n" + "=" * 80)
    print("Kalman Drift MLE Tuning Summary")
    print("=" * 80)
    print(f"Assets processed:       {len(assets)}")
    print(f"New estimates:          {new_estimates}")
    print(f"Reused cached:          {reused_cached}")
    print(f"Failed:                 {failed}")
    print(f"Calibration warnings:   {calibration_warnings}")
    print(f"\nModel Selection (BIC + Hyvärinen combined scoring):")
    print(f"  Gaussian/φ-Gaussian:  {gaussian_count}")
    print(f"  φ-Student-t:          {student_t_count} (discrete ν ∈ {{4, 6, 8, 12, 20}})")
    print(f"\nPrior Configuration:")
    print(f"  q prior:              log₁₀(q) ~ N({args.prior_mean:.1f}, λ={args.prior_lambda:.1f})")
    print(f"  φ prior:              φ ~ N(0, τ) with λ_φ=0.05 (explicit shrinkage)")
    print(f"\nRegime-Conditional Tuning (Hierarchical Bayesian):")
    print(f"  Hierarchical shrinkage λ: {args.lambda_regime:.3f}")
    print(f"  Assets with regime params: {regime_tuning_count}")
    # Count regimes with actual params (not fallback) and shrinkage stats
    regime_fit_counts = {r: 0 for r in range(5)}
    regime_shrunk_counts = {r: 0 for r in range(5)}
    collapse_warnings = 0
    for asset, data in cache.items():
        regime_data = data.get('regime')
        if regime_data is not None and isinstance(regime_data, dict):
            for r, params in regime_data.items():
                if isinstance(params, dict):
                    # Handle both old structure (fallback at top level) and new BMA structure (in regime_meta)
                    is_fallback = params.get('fallback', False) or params.get('regime_meta', {}).get('fallback', False)
                    if not is_fallback:
                        regime_fit_counts[int(r)] += 1
                        # Check for shrinkage in both old and new structures
                        is_shrunk = params.get('shrinkage_applied', False) or params.get('regime_meta', {}).get('shrinkage_applied', False)
                        if is_shrunk:
                            regime_shrunk_counts[int(r)] += 1
        if 'hierarchical_tuning' in data:
            if data['hierarchical_tuning'].get('collapse_warning', False):
                collapse_warnings += 1
    print(f"  Regime-specific fits:")
    for r in range(5):
        shrunk_str = f" ({regime_shrunk_counts[r]} shrunk)" if regime_shrunk_counts[r] > 0 else ""
        print(f"    {REGIME_LABELS[r]}: {regime_fit_counts[r]} assets{shrunk_str}")
    if collapse_warnings > 0:
        print(f"  ⚠️  Collapse warnings: {collapse_warnings} assets")
    
    if cache:
        print("\nBest-fit parameters (grouped by model family, then q) — ALL ASSETS:")

        def _model_label(data: dict) -> str:
            # Handle new regime-conditional structure
            if 'global' in data:
                data = data['global']
            phi_val = data.get('phi')
            noise_model = data.get('noise_model', 'gaussian')
            # Check for Student-t model (phi_student_t_nu_* naming)
            if is_student_t_model(noise_model) and phi_val is not None:
                return 'Phi-Student-t'
            if is_student_t_model(noise_model):
                return 'Student-t'
            if noise_model == 'kalman_phi_gaussian' or phi_val is not None:
                return 'Phi-Gaussian'
            return 'Gaussian'
        
        col_specs = [
            ("Asset", 18), ("Model", 14), ("log₁₀(q)", 9), ("c", 7), ("ν", 7), ("φ", 7),
            ("ΔLL0", 8), ("ΔLLc", 8), ("ΔLLe", 8), ("BestModel", 12), ("BIC", 10), ("PIT p", 8)
        ]

        def fmt_row(values):
            parts = []
            for (val, (_, width)) in zip(values, col_specs):
                parts.append(f"{val:<{width}}")
            return "| " + " | ".join(parts) + " |"

        sep_line = "+" + "+".join(["-" * (w + 2) for _, w in col_specs]) + "+"
        header_line = fmt_row([name for name, _ in col_specs])

        print(sep_line)
        print(header_line)
        print(sep_line)

        # Sort by model family, then descending q
        def _get_q_for_sort(data):
            if 'global' in data:
                return data['global'].get('q', 0)
            return data.get('q', 0)
        
        sorted_assets = sorted(
            cache.items(),
            key=lambda x: (
                _model_label(x[1]),
                -_get_q_for_sort(x[1])
            )
        )

        last_group = None
        for asset, raw_data in sorted_assets:
            # Handle regime-conditional structure
            if 'global' in raw_data:
                data = raw_data['global']
            else:
                data = raw_data
            
            q_val = data.get('q', float('nan'))
            c_val = data.get('c', 1.0)
            nu_val = data.get('nu')
            phi_val = data.get('phi')
            delta_ll_zero = data.get('delta_ll_vs_zero', float('nan'))
            delta_ll_const = data.get('delta_ll_vs_const', float('nan'))
            delta_ll_ewma = data.get('delta_ll_vs_ewma', float('nan'))
            bic_val = data.get('bic', float('nan'))
            pit_p = data.get('pit_ks_pvalue', float('nan'))
            model = _model_label(raw_data)
            best_model = data.get('best_model_by_bic', 'kalman_drift')

            log10_q = np.log10(q_val) if q_val > 0 else float('nan')

            nu_str = f"{nu_val:.1f}" if nu_val is not None else "-"
            phi_str = f"{phi_val:.3f}" if phi_val is not None else "-"

            best_model_abbr = {
                'zero_drift': 'Zero',
                'constant_drift': 'Const',
                'ewma_drift': 'EWMA',
                'kalman_drift': 'Kalman',
                'kalman_gaussian': 'Gaussian',
                'kalman_phi_gaussian': 'PhiGauss',
            }
            # Add entries for discrete nu grid models
            for nu in STUDENT_T_NU_GRID:
                best_model_abbr[f"phi_student_t_nu_{nu}"] = f'PhiT-ν{nu}'
            best_model_abbr = best_model_abbr.get(best_model, best_model[:8])

            warn_marker = " ⚠️" if data.get('calibration_warning') else ""

            if model != last_group:
                if last_group is not None:
                    print(sep_line)
                print(f"| Group: {model:<{sum(w+3 for _, w in col_specs)-9}}|")
                print(sep_line)
                last_group = model

            row = fmt_row([
                asset,
                model,
                f"{log10_q:>7.2f}",
                f"{c_val:>5.3f}",
                nu_str,
                phi_str,
                f"{delta_ll_zero:>6.1f}",
                f"{delta_ll_const:>6.1f}",
                f"{delta_ll_ewma:>6.1f}",
                best_model_abbr,
                f"{bic_val:>8.1f}",
                f"{pit_p:.4f}{warn_marker}"
            ])
            print(row)

        print(sep_line)

        print("\nColumn Legend:")
        print("  Model: Gaussian / Phi-Gaussian / Phi-Student-t (φ from cache)")
        print("  φ: Drift persistence (if AR(1) model)")
        print("  ΔLL_0: ΔLL vs zero-drift baseline")
        print("  ΔLL_c: ΔLL vs constant-drift baseline")
        print("  ΔLL_e: ΔLL vs EWMA-drift baseline")
        print("  BestModel: Best model by BIC (Zero/Const/EWMA/Kalman/PhiKal)")
 
        print("\nCache file:")
        print(f"  JSON: {args.cache_json}")

    if failure_reasons:
        print("\nFailed tickers and reasons:")
        for a, msg in failure_reasons.items():
            print(f"  {a}: {msg}")
    
    print("=" * 80)

    # ==========================================================================
    # END-OF-RUN SUMMARY: Regime Distributions, Warnings, and Errors
    # ==========================================================================
    print("\n" + "=" * 80)
    print("END-OF-RUN SUMMARY")
    print("=" * 80)

    # Model Comparison Summary (per asset)
    if model_comparisons:
        print("\n🔬 MODEL COMPARISON RESULTS (per asset):")
        print("-" * 80)
        for asset_name in sorted(model_comparisons.keys()):
            mc = model_comparisons[asset_name]
            model_comp = mc.get('model_comparison', {})
            selected = mc.get('selected_model', 'unknown')
            best_bic = mc.get('best_model_by_bic', 'unknown')
            model_sel_method = mc.get('model_selection_method', 'combined')
            
            print(f"\n  {asset_name} (selection: {model_sel_method}):")
            
            # Print each baseline/model with Hyvärinen score where available
            if 'zero_drift' in model_comp:
                m = model_comp['zero_drift']
                print(f"     Zero-drift:     LL={m['ll']:.1f}, AIC={m['aic']:.1f}, BIC={m['bic']:.1f}")
            
            if 'constant_drift' in model_comp:
                m = model_comp['constant_drift']
                mu_str = f", μ={m.get('mu', 0):.6f}" if 'mu' in m else ""
                print(f"     Constant-drift: LL={m['ll']:.1f}, AIC={m['aic']:.1f}, BIC={m['bic']:.1f}{mu_str}")
            
            if 'ewma_drift' in model_comp:
                m = model_comp['ewma_drift']
                print(f"     EWMA-drift:     LL={m['ll']:.1f}, AIC={m['aic']:.1f}, BIC={m['bic']:.1f}")
            
            if 'kalman_gaussian' in model_comp:
                m = model_comp['kalman_gaussian']
                hyv_str = f", H={m['hyvarinen_score']:.1f}" if m.get('hyvarinen_score') is not None else ""
                print(f"     Kalman-Gaussian: LL={m['ll']:.1f}, AIC={m['aic']:.1f}, BIC={m['bic']:.1f}{hyv_str}")
            
            if 'kalman_phi_gaussian' in model_comp:
                m = model_comp['kalman_phi_gaussian']
                phi_str = f", φ={m.get('phi', 0):+.3f}" if 'phi' in m else ""
                hyv_str = f", H={m['hyvarinen_score']:.1f}" if m.get('hyvarinen_score') is not None else ""
                print(f"     Kalman-φ-Gaussian: LL={m['ll']:.1f}, AIC={m['aic']:.1f}, BIC={m['bic']:.1f}{phi_str}{hyv_str}")
            
            # Display discrete nu grid Student-t models
            for nu_val in STUDENT_T_NU_GRID:
                model_key = f"phi_student_t_nu_{nu_val}"
                if model_key in model_comp:
                    m = model_comp[model_key]
                    phi_str = f", φ={m.get('phi', 0):+.3f}" if 'phi' in m else ""
                    hyv_str = f", H={m['hyvarinen_score']:.1f}" if m.get('hyvarinen_score') is not None else ""
                    print(f"     Phi-Student-t (ν={nu_val}): LL={m['ll']:.1f}, AIC={m['aic']:.1f}, BIC={m['bic']:.1f}{phi_str}{hyv_str}")
            
            # Selected model with Hyvärinen score
            ll_sel = mc.get('log_likelihood', float('nan'))
            bic_sel = mc.get('bic', float('nan'))
            hyv_sel = mc.get('hyvarinen_score')
            hyv_summary = f", H={hyv_sel:.1f}" if hyv_sel is not None else ""
            print(f"     Selected:        LL={ll_sel:.1f}, BIC={bic_sel:.1f}{hyv_summary} ({selected})")

    # Regime Distributions Summary
    if regime_distributions:
        print("\n📊 REGIME DISTRIBUTIONS (per asset):")
        print("-" * 80)
        for asset_name in sorted(regime_distributions.keys()):
            counts = regime_distributions[asset_name]
            total = sum(counts.values())
            dist_str = ", ".join([f"{REGIME_LABELS[r]}={c}" for r, c in sorted(counts.items()) if c > 0])
            print(f"  {asset_name} ({total} obs): {dist_str}")
        
        # Aggregate statistics
        print("\n  Aggregate regime counts across all processed assets:")
        aggregate = {r: 0 for r in range(5)}
        for counts in regime_distributions.values():
            for r, c in counts.items():
                aggregate[r] += c
        total_obs = sum(aggregate.values())
        for r in range(5):
            pct = 100.0 * aggregate[r] / total_obs if total_obs > 0 else 0
            print(f"    {REGIME_LABELS[r]}: {aggregate[r]:,} ({pct:.1f}%)")

    # Warnings Summary
    if processing_warnings:
        print("\n⚠️  PROCESSING WARNINGS:")
        print("-" * 80)
        for warning in processing_warnings:
            print(f"  ⚠️  {warning}")

    # Failures Summary with Full Tracebacks
    if failure_reasons:
        print("\n❌ FAILED TICKERS AND REASONS:")
        print("-" * 80)
        for asset_name, msg in sorted(failure_reasons.items()):
            print(f"\n  {asset_name}: {msg}")
            if asset_name in failure_tracebacks:
                print("  Full traceback:")
                for line in failure_tracebacks[asset_name].split('\n'):
                    print(f"    {line}")
    
    # ══════════════════════════════════════════════════════════════════════════════
    # CALIBRATION ISSUES SUMMARY - Apple-quality Rich table
    # ══════════════════════════════════════════════════════════════════════════════
    render_calibration_issues_table(cache, failure_reasons)
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
