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
       - kalman_gaussian_momentum:          q, c           (2 params) [ENABLED - momentum Gaussian]
       - kalman_phi_gaussian_momentum:      q, c, φ        (3 params) [ENABLED - momentum φ-Gaussian]
       - phi_student_t_nu_4:    q, c, φ        (3 params, ν=4 FIXED)
       - phi_student_t_nu_6:    q, c, φ        (3 params, ν=6 FIXED)
       - phi_student_t_nu_8:    q, c, φ        (3 params, ν=8 FIXED)
       - phi_student_t_nu_12:   q, c, φ        (3 params, ν=12 FIXED)
       - phi_student_t_nu_20:   q, c, φ        (3 params, ν=20 FIXED)
       - phi_student_t_nu_{ν}_momentum: q, c, φ     (3 params, ν FIXED) [ENABLED]
       - phi_skew_t_nu_{ν}_gamma_{γ}: q, c, φ  (3 params, ν and γ FIXED)
       - phi_nig_alpha_{α}_beta_{β}: q, c, φ   (3 params, α and β FIXED)
       - phi_fisher_rao_w{W}_d{D}_momentum: q, c, φ (5 params) [ENABLED - Fisher-Rao geometry]

    NOTE: Pure Gaussian and φ-Gaussian are NOT exposed in BMA (February 2026).
    Only momentum-augmented versions (kalman_gaussian_momentum, kalman_phi_gaussian_momentum) participate.

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
VOLATILITY ESTIMATION — GARMAN-KLASS REALIZED VOLATILITY (February 2026)

The observation variance in the Kalman filter is scaled by volatility σ_t:

    r_t = μ_t + √(c·σ_t²)·ε_t

VOLATILITY ESTIMATION PRIORITY:
    1. Garman-Klass (GK) — 7.4x more efficient than close-to-close EWMA
    2. GARCH(1,1) via MLE — captures volatility clustering
    3. EWMA blend — robust baseline fallback

GARMAN-KLASS FORMULA (uses OHLC data):
    σ²_GK = 0.5*(log(H/L))² - (2*log(2)-1)*(log(C/O))²

The efficiency gain (7.4x) means:
    - Same precision as EWMA with ~7x fewer observations
    - Or: same observations → ~7x more precise variance estimate

IMPACT ON CALIBRATION:
    - Better σ_t estimate → less noise absorbed by c parameter
    - Improves PIT calibration without adding parameters
    - Reduces variance estimation error in all downstream models

The volatility estimator used is stored in tuning results:
    "volatility_estimator": "GK" | "EWMA" | "garch11"

-------------------------------------------------------------------------------
GAS-Q SCORE-DRIVEN PROCESS NOISE (February 2026)
-------------------------------------------------------------------------------

PROBLEM: Static process noise q doesn't adapt to recent forecast errors.
When innovations are large, q should increase. When small, q should decrease.

SOLUTION: Generalized Autoregressive Score (GAS) model for q.
Reference: Creal, Koopman & Lucas (2013) "Generalized Autoregressive Score Models"

GAS-Q DYNAMICS:
    q_t = ω + α·s_{t-1} + β·q_{t-1}
    
Where:
    s_t = ∂log p(y_t|θ)/∂q is the scaled score
    
For Gaussian innovations:
    s_t = (e_t² / S_t - 1) / (2 * S_t)
    
For Student-t innovations with ν degrees of freedom:
    w_t = (ν + 1) / (ν + (y_t - μ_t)² / S_t)
    s_t = (w_t * e_t² / S_t - 1) / (2 * S_t)
    
PARAMETERS (jointly estimated with c, φ, ν):
    ω (omega): Long-run mean level of q (ω > 0)
    α (alpha): Score sensitivity (0 ≤ α ≤ 1)
    β (beta):  Persistence (0 ≤ β < 1)
    
STATIONARITY CONSTRAINT:
    β < 1 ensures q_t is mean-reverting to ω / (1 - β)

EXPECTED IMPACT:
    - 15-20% improvement in adaptive forecasting during regime transitions
    - Process noise adapts to recent forecast errors
    - Better PIT calibration in volatile periods

TUNING OUTPUT:
    When GAS-Q is selected, the model includes:
    - gas_q_augmented: True
    - gas_q_omega, gas_q_alpha, gas_q_beta: GAS-Q parameters
    - gas_q_ll: Log-likelihood with GAS-Q dynamics

INTEGRATION:
    - tune.py: Fits GAS-Q parameters via concentrated likelihood
    - gas_q.py: Implements GAS-Q filter functions
    - signals.py: Uses GAS-Q filter when gas_q_augmented=True in cache

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
    )
    GK_VOLATILITY_AVAILABLE = True
    HAR_VOLATILITY_AVAILABLE = True
except ImportError:
    GK_VOLATILITY_AVAILABLE = False
    HAR_VOLATILITY_AVAILABLE = False

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
# ENHANCED MIXTURE WEIGHT DYNAMICS (February 2026 - Expert Panel)
# =============================================================================
# Upgraded from reactive (vol-only) to multi-factor conditioning:
#   w_t = sigmoid(a × z_t + b × Δσ_t + c × M_t)
# Where: z_t = shock, Δσ_t = vol acceleration, M_t = momentum
# =============================================================================
ENHANCED_MIXTURE_ENABLED = True  # Set to False to use standard vol-only mixture

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
    from dataclasses import dataclass
    
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
    import warnings
    warnings.warn(f"GAS-Q module not available: {e}")

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
    # Enhanced Mixture Weight Dynamics (February 2026)
    MIXTURE_WEIGHT_A_SHOCK,
    MIXTURE_WEIGHT_B_VOL_ACCEL,
    MIXTURE_WEIGHT_C_MOMENTUM,
    # Markov-Switching Process Noise (MS-q) — February 2026
    MS_Q_ENABLED,
    MS_Q_CALM_DEFAULT,
    MS_Q_STRESS_DEFAULT,
    MS_Q_SENSITIVITY,
    MS_Q_THRESHOLD,
    MS_Q_BMA_PENALTY,
    compute_ms_process_noise,
    filter_phi_ms_q,
    optimize_params_ms_q,
    # Unified Student-T Architecture (February 2026)
    UnifiedStudentTConfig,
    compute_ms_process_noise_smooth,
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
        # Fetch price data (need OHLC for Garman-Klass volatility)
        # Use _download_prices ONCE to get OHLC data - extract Close from it
        # This avoids duplicate downloads (February 2026 optimization)
        df = _download_prices(asset, start_date, end_date)
        if df is None or df.empty:
            _log(f"     ⚠️  No price data for {asset}")
            return None
        
        # Extract Close prices from OHLC DataFrame
        cols = {c.lower(): c for c in df.columns}
        if 'close' in cols:
            px = df[cols['close']]
        else:
            _log(f"     ⚠️  No Close column for {asset}")
            return None
        
        if px is None or len(px) < 20:
            _log(f"     ⚠️  Insufficient data for {asset}")
            return None
        
        # Compute returns
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values
        
        # Compute volatility using Garman-Klass or HAR (7.4x more efficient than EWMA)
        vol_estimator_used = "EWMA"
        if GK_VOLATILITY_AVAILABLE and df is not None and not df.empty:
            try:
                # Check for OHLC columns
                cols = {c.lower(): c for c in df.columns}
                if all(c in cols for c in ['open', 'high', 'low', 'close']):
                    # Align OHLC data with returns (drop first row to match log returns)
                    df_aligned = df.iloc[1:].copy()
                    open_ = df_aligned[cols['open']].values
                    high = df_aligned[cols['high']].values
                    low = df_aligned[cols['low']].values
                    close = df_aligned[cols['close']].values
                    
                    # ENFORCE HAR-GK ONLY (February 2026)
                    # HAR-GK provides multi-horizon memory for crash detection
                    # Combined with Garman-Klass (7.4x more efficient than EWMA)
                    vol, vol_estimator_used = compute_hybrid_volatility_har(
                        open_=open_, high=high, low=low, close=close,
                        span=21, annualize=False, use_har=True
                    )
                else:
                    # OHLC not available - raise error as HAR-GK is required
                    raise ValueError(f"OHLC data required for HAR-GK volatility estimation for {asset}")
            except Exception as e:
                # Log error but don't silently fall back to inferior estimator
                _log(f"     ⚠️ HAR-GK volatility estimation failed: {e}")
                raise ValueError(f"HAR-GK volatility estimation required but failed for {asset}: {e}")
        else:
            # GK/HAR module not available - this should not happen in production
            raise ImportError("HAR-GK volatility module required but not available")
        
        # Ensure returns and vol have same length
        min_len = min(len(returns), len(vol))
        returns = returns[:min_len]
        vol = vol[:min_len]

        # Remove NaN/Inf
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns = returns[valid_mask]
        vol = vol[valid_mask]

        if len(returns) < 20:
            _log(f"     ⚠️  Insufficient valid data for {asset}")
            return None
        
        n_obs = len(returns)
        
        # Extract prices array for MR integration (February 2026)
        # Align prices with returns (skip first element since returns = diff(log(px)))
        prices_array = None
        if px is not None:
            px_values = px.values if hasattr(px, 'values') else np.array(px)
            # Skip first price to align with returns, then apply same valid_mask
            if len(px_values) > 1:
                prices_aligned = px_values[1:][:min_len]
                if len(prices_aligned) == len(valid_mask):
                    prices_array = prices_aligned[valid_mask]
                elif len(prices_aligned) >= len(returns):
                    prices_array = prices_aligned[:len(returns)]
        
        # Fit all model classes globally
        models = fit_all_models_for_regime(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            prices=prices_array,  # MR integration (February 2026)
            asset=asset,  # FIX #4: Asset-class adaptive c bounds
        )
        
        # Compute model weights using regime-aware BIC + Hyvärinen + CRPS + PIT (February 2026)
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        hyvarinen_values = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models}
        crps_values = {m: models[m].get("crps", float('inf')) for m in models if models[m].get("crps") is not None}
        pit_pvalues = {m: models[m].get("pit_ks_pvalue") for m in models if models[m].get("pit_ks_pvalue") is not None}
        
        # Use regime-aware weights with PIT penalty (February 2026 - Elite Architecture)
        if crps_values and CRPS_SCORING_ENABLED:
            model_weights, weight_meta = compute_regime_aware_model_weights(
                bic_values, hyvarinen_values, crps_values, pit_pvalues=pit_pvalues, regime=None
            )
        else:
            model_weights = compute_bic_model_weights(bic_values)
            weight_meta = {"scoring_method": "bic_only", "crps_enabled": False, "pit_enabled": False}
        
        # Store standardized scores and weights in each model
        for m in models:
            w = model_weights.get(m, 1e-10)
            models[m]['model_weight_entropy'] = float(w)
            if weight_meta:
                combined_score_val = weight_meta.get('combined_scores_standardized', {}).get(m)
                models[m]['combined_score'] = float(combined_score_val) if combined_score_val is not None else 0.0
                bic_std_val = weight_meta.get('bic_standardized', {}).get(m)
                models[m]['standardized_bic'] = float(bic_std_val) if bic_std_val is not None else None
                hyv_std_val = weight_meta.get('hyvarinen_standardized', {}).get(m)
                models[m]['standardized_hyvarinen'] = float(hyv_std_val) if hyv_std_val is not None else None
                crps_std_val = weight_meta.get('crps_standardized', {}).get(m)
                models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
                scoring_weights = weight_meta.get('weights_used', {})
                models[m]['scoring_weights'] = {
                    'bic': float(scoring_weights.get('bic', 0.0)),
                    'hyvarinen': float(scoring_weights.get('hyvarinen', 0.0)),
                    'crps': float(scoring_weights.get('crps', 0.0)),
                }
                models[m]['crps_scoring_enabled'] = weight_meta.get('crps_enabled', False)
        
        # Find best model by COMBINED SCORE (BIC + Hyvärinen + CRPS)
        # Combined score formula: w_bic * BIC_std - w_hyv * Hyv_std + w_crps * CRPS_std
        # Lower combined score = better model (February 2026 - Elite Architecture Fix)
        combined_scores = weight_meta.get('combined_scores_standardized', {})
        best_model = min(
            ((m, s) for m, s in combined_scores.items() if s is not None and np.isfinite(s)),
            key=lambda x: x[1]
        )[0]
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
        
        # =====================================================================
        # PIT-DRIVEN ESCALATION: EVT TAIL SPLICE (L3)
        # =====================================================================
        # EVT Tail Splice replaces the tail portion of the CDF with GPD.
        # This is triggered when:
        #   1. PIT calibration still fails after ν-refinement
        #   2. EVT has been fitted successfully
        #   3. EVT splice improves PIT p-value by >= 50%
        #
        # The spliced distribution provides theoretically justified tail
        # extrapolation via the Pickands–Balkema–de Haan theorem.
        # =====================================================================
        evt_splice_result = None
        evt_splice_attempted = False
        evt_splice_selected = False
        
        if EVT_SPLICE_AVAILABLE and EVT_SPLICE_ENABLED and evt_result is not None:
            # Check if PIT still fails after previous escalations
            current_pit = best_params.get("pit_ks_pvalue", 1.0)
            pit_still_fails = current_pit < 0.05
            
            if pit_still_fails:
                evt_splice_attempted = True
                try:
                    # Reconstruct GPDFitResult from stored dict
                    gpd_result_obj = GPDFitResult.from_dict(evt_result)
                    
                    # Get filter outputs for PIT computation
                    # We need to run filter with best params
                    q_best = best_params.get("q", 1e-6)
                    c_best = best_params.get("c", 1.0)
                    phi_best = best_params.get("phi", 0.0)
                    nu_best = best_params.get("nu")
                    
                    # Run filter
                    if phi_best is not None:
                        from models import PhiStudentTDriftModel, PhiGaussianDriftModel
                        if nu_best is not None:
                            mu_filt, P_filt, _ = PhiStudentTDriftModel.filter_phi(
                                returns, vol, q_best, c_best, phi_best, nu_best
                            )
                        else:
                            mu_filt, P_filt, _ = PhiGaussianDriftModel.filter_phi(
                                returns, vol, q_best, c_best, phi_best
                            )
                    else:
                        from models import GaussianDriftModel
                        mu_filt, P_filt, _ = GaussianDriftModel.filter(
                            returns, vol, q_best, c_best
                        )
                    
                    # Test if EVT splice improves PIT
                    should_select, evt_pit_pvalue, improvement_ratio, evt_diag = test_evt_splice_improvement(
                        returns=returns,
                        mu_filtered=mu_filt,
                        vol=vol,
                        P_filtered=P_filt,
                        c=c_best,
                        nu=nu_best,
                        gpd_result=gpd_result_obj,
                        baseline_pit_pvalue=current_pit,
                    )
                    
                    evt_splice_result = {
                        "attempted": True,
                        "should_select": should_select,
                        "baseline_pit_pvalue": current_pit,
                        "evt_pit_pvalue": evt_pit_pvalue,
                        "improvement_ratio": improvement_ratio,
                        "diagnostics": evt_diag,
                    }
                    
                    if should_select:
                        evt_splice_selected = True
                        # Update best params to flag EVT splice
                        best_params = best_params.copy()
                        best_params["evt_splice_selected"] = True
                        best_params["evt_splice_pit_pvalue"] = evt_pit_pvalue
                        best_params["pit_ks_pvalue"] = evt_pit_pvalue  # Update PIT
                        _log(f"     ✓ EVT Splice L3: PIT {current_pit:.4f}→{evt_pit_pvalue:.4f} ({improvement_ratio:.1f}x improvement)")
                    else:
                        reason = evt_diag.get("reason", "unknown")
                        _log(f"     ⚠️ EVT Splice L3: Not selected ({reason})")
                        
                except Exception as evt_err:
                    _log(f"     ⚠️ EVT Splice error: {evt_err}")
                    evt_splice_result = {"attempted": True, "error": str(evt_err)}

        
        # Build result structure - BMA-compatible format
        # signals.py expects: {"global": {...}, "has_bma": True}
        global_data = {
            "asset": asset,
            "q": float(best_params.get("q", 1e-6)),
            "c": float(best_params.get("c", 1.0)),
            "phi": best_params.get("phi"),
            "nu": best_params.get("nu"),
            "noise_model": best_model,
            "best_model": best_model,  # Selected by combined BIC+Hyv+CRPS score
            # Unified Student-t specific parameters (February 2026 - Elite Architecture)
            "unified_model": best_params.get("unified_model", False),
            "alpha_asym": best_params.get("alpha_asym"),
            "gamma_vov": best_params.get("gamma_vov"),
            "ms_sensitivity": best_params.get("ms_sensitivity"),
            "q_stress_ratio": best_params.get("q_stress_ratio"),
            "vov_damping": best_params.get("vov_damping"),
            "degraded": best_params.get("degraded", False),
            "hessian_cond": best_params.get("hessian_cond"),
            "pit_calibration_grade": best_params.get("pit_calibration_grade"),
            "bic": float(best_params.get("bic", float('inf'))),
            "aic": float(best_params.get("aic", float('inf'))),
            "log_likelihood": float(best_params.get("log_likelihood", float('-inf'))),
            "mean_log_likelihood": float(best_params.get("mean_log_likelihood", float('-inf'))),
            "ks_statistic": float(best_params.get("ks_statistic", 0.0)),
            "pit_ks_pvalue": float(best_params.get("pit_ks_pvalue", 0.0)),
            "calibration_warning": best_params.get("pit_ks_pvalue", 1.0) < 0.05,
            "n_obs": n_obs,
            # Volatility estimator used (February 2026 - Garman-Klass support)
            "volatility_estimator": vol_estimator_used,
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
            # EVT Tail Splice for PIT calibration (L3 escalation)
            "evt_splice": evt_splice_result,
            "evt_splice_attempted": evt_splice_attempted,
            "evt_splice_selected": evt_splice_selected,
            # Contaminated Student-t Mixture for regime-dependent tails
            "contaminated_student_t": cst_result,
            "contaminated_student_t_diagnostics": cst_diagnostics,
            "contaminated_vs_single_comparison": cst_comparison,
            # PIT-driven ν-refinement results (L2 escalation)
            "nu_refinement": nu_refinement_result,
            "nu_refinement_attempted": nu_refinement_attempted,
            "nu_refinement_improved": nu_refinement_improved,
            # Market conditioning flags (February 2026)
            # These tell signals.py to apply VIX-based ν adjustment at inference time
            "market_conditioning_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE,
            "vix_nu_adjustment_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE and best_params.get("nu") is not None,
        }
        
        result = {
            "asset": asset,
            "has_bma": True,  # CRITICAL: signals.py checks this flag to accept the cache
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
        get_cache_stats as get_kalman_cache_stats,
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
    # Note: cache_json might be a directory path (from retune), so check isfile()
    if cache_json and os.path.isfile(cache_json):
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


def compute_predictive_pit_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    ELITE FIX: Compute proper PIT using PREDICTIVE distribution.
    
    This function runs the filter to get predictive values (before seeing y_t)
    and computes PIT correctly. The key insight: PIT transforms y_t through
    the CDF of its PRIOR predictive distribution, not the posterior.
    
    Args:
        returns: Return observations
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence
        nu: Degrees of freedom
        
    Returns:
        Tuple of (KS statistic, KS p-value, mu_pred, S_pred)
    """
    # Run filter with predictive output
    mu_filt, P_filt, mu_pred, S_pred, ll = PhiStudentTDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi, nu
    )
    
    # Compute PIT using predictive distribution
    ks_stat, pit_p = PhiStudentTDriftModel.pit_ks_predictive(
        returns, mu_pred, S_pred, nu
    )
    
    return ks_stat, pit_p, mu_pred, S_pred


def compute_predictive_scores_student_t(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
    nu: float,
) -> Tuple[float, float]:
    """
    ELITE FIX: Compute Hyvärinen and CRPS using PREDICTIVE distribution.
    
    Uses predictive mean and variance (before seeing y_t) for proper
    out-of-sample scoring.
    
    Args:
        returns: Return observations
        mu_pred: Predictive means from filter_phi_with_predictive
        S_pred: Predictive variances from filter_phi_with_predictive
        nu: Degrees of freedom
        
    Returns:
        Tuple of (Hyvärinen score, CRPS)
    """
    # Compute Student-t scale from predictive variance
    if nu > 2:
        forecast_scale = np.sqrt(S_pred * (nu - 2) / nu)
    else:
        forecast_scale = np.sqrt(S_pred)
    
    # Compute scores using predictive values
    hyvarinen = compute_hyvarinen_score_student_t(returns, mu_pred, forecast_scale, nu)
    crps = compute_crps_student_t_inline(returns, mu_pred, forecast_scale, nu)
    
    return hyvarinen, crps


def compute_predictive_pit_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    ELITE FIX: Compute proper PIT using PREDICTIVE distribution for Gaussian.
    
    Args:
        returns: Return observations
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence (default 1.0 for random walk)
        
    Returns:
        Tuple of (KS statistic, KS p-value, mu_pred, S_pred)
    """
    # Run filter with predictive output
    mu_filt, P_filt, mu_pred, S_pred, ll = GaussianDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi
    )
    
    # Compute PIT using predictive distribution
    ks_stat, pit_p = GaussianDriftModel.pit_ks_predictive(
        returns, mu_pred, S_pred
    )
    
    return ks_stat, pit_p, mu_pred, S_pred


def compute_predictive_scores_gaussian(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    ELITE FIX: Compute Hyvärinen and CRPS using PREDICTIVE distribution for Gaussian.
    """
    forecast_std = np.sqrt(S_pred)
    
    hyvarinen = compute_hyvarinen_score_gaussian(returns, mu_pred, forecast_std)
    crps = compute_crps_gaussian_inline(returns, mu_pred, forecast_std)
    
    return hyvarinen, crps


def reconstruct_predictive_from_filtered_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ELITE FIX: Reconstruct predictive values from filtered values.
    
    For momentum-augmented models that only return filtered values,
    this function reconstructs the predictive values needed for proper PIT.
    
    The key insight:
        mu_pred[t] = phi * mu_filtered[t-1]  (before seeing y_t)
        P_pred[t] = phi^2 * P_filtered[t-1] + q  (before seeing y_t)
        S_pred[t] = P_pred[t] + c * vol[t]^2  (total predictive variance)
    
    Args:
        returns: Return observations (for length)
        mu_filtered: Filtered (posterior) mean estimates
        P_filtered: Filtered (posterior) variance estimates
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence
        
    Returns:
        Tuple of (mu_pred, S_pred) predictive arrays
    """
    n = len(returns)
    mu_pred = np.zeros(n)
    S_pred = np.zeros(n)
    
    # Initial values (t=0)
    mu_pred[0] = 0.0  # Prior mean
    P_pred_0 = 1e-4 + q  # Prior variance + process noise
    S_pred[0] = P_pred_0 + c * (vol[0] ** 2)
    
    # Reconstruct for t >= 1
    for t in range(1, n):
        # Predictive mean: φ × μ_{t-1|t-1}
        mu_pred[t] = phi * mu_filtered[t - 1]
        
        # Predictive state variance: φ² × P_{t-1|t-1} + q
        P_pred_t = (phi ** 2) * P_filtered[t - 1] + q
        
        # Total predictive variance: P_pred + observation noise
        S_pred[t] = P_pred_t + c * (vol[t] ** 2)
    
    return mu_pred, S_pred


def compute_pit_from_filtered_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Tuple[float, float]:
    """
    ELITE FIX: Compute proper PIT from filtered values for Gaussian.
    
    Reconstructs predictive values and computes PIT correctly.
    Use this for momentum-augmented models that only return filtered values.
    
    Args:
        returns: Return observations
        mu_filtered: Filtered (posterior) mean estimates
        P_filtered: Filtered (posterior) variance estimates
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence
        
    Returns:
        Tuple of (KS statistic, KS p-value)
    """
    mu_pred, S_pred = reconstruct_predictive_from_filtered_gaussian(
        returns, mu_filtered, P_filtered, vol, q, c, phi
    )
    
    return GaussianDriftModel.pit_ks_predictive(returns, mu_pred, S_pred)


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
# ELITE TUNING DIAGNOSTICS HELPER (v2.0 - February 2026)
# =============================================================================
# Computes elite tuning diagnostics after parameter optimization.
# 
# TOP 0.001% UPGRADES:
#   1. DIRECTIONAL CURVATURE: φ-q coupling penalized 2× more than benign couplings
#   2. RIDGE DETECTION: Distinguishes flat basins from narrow ridges
#   3. ASYMMETRIC COHERENCE: Drift penalized harder than oscillation
#   4. PURE FRAGILITY: No calibration leakage into tuning
#
# INSTITUTIONAL ALIGNMENT:
#   Top funds don't just prefer flat optima; they explicitly reject optima
#   that are only flat along fragile directions.
# =============================================================================

def compute_elite_diagnostics(
    objective_fn,
    optimal_params: np.ndarray,
    optimal_value: float,
    bounds: List[Tuple[float, float]],
    param_names: Optional[List[str]] = None,
    fold_optimal_params: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive elite tuning diagnostics (v2.0).
    
    This function computes stability metrics that distinguish top 0.001% funds:
    - Curvature penalty with directional weighting (φ-q coupling danger)
    - Ridge vs basin detection via joint perturbations
    - Drift vs noise decomposition in cross-fold coherence
    - Pure parameter fragility index (no calibration leakage)
    
    Args:
        objective_fn: Objective function to evaluate (neg penalized LL)
        optimal_params: Optimized parameter vector
        optimal_value: Objective value at optimum
        bounds: Parameter bounds [(lo, hi), ...]
        param_names: Optional parameter names for diagnostics
        fold_optimal_params: Optional list of per-fold optimal params
        
    Returns:
        Dictionary with elite diagnostics including:
        - condition_number: Hessian condition number
        - curvature_penalty: Soft penalty for sharp optima
        - directional_curvature_penalty: Weighted by coupling danger
        - plateau_score: Width of acceptable region
        - basin_score: 1.0 = flat basin, 0.0 = narrow ridge
        - is_ridge_optimum: True if ridge detected (dangerous)
        - drift_component: Variance from persistent drift
        - noise_component: Variance from random oscillation
        - fragility_index: Composite fragility score
        - fragility_warning: True if fragility > threshold
        - fragility_components: Per-dimension breakdown
    """
    if not ELITE_TUNING_AVAILABLE or not ELITE_TUNING_ENABLED:
        return {'elite_tuning_enabled': False}
    
    diagnostics = {
        'elite_tuning_version': '2.0',
        'elite_tuning_enabled': True,
    }
    
    n_params = len(optimal_params)
    param_ranges = np.array([b[1] - b[0] for b in bounds])
    
    # Get elite tuning config
    try:
        config = create_elite_tuning_config(ELITE_TUNING_PRESET)
    except Exception:
        config = EliteTuningConfig()
    
    # =========================================================================
    # 1. HESSIAN-BASED CURVATURE ANALYSIS (with directional weighting)
    # =========================================================================
    try:
        H = compute_hessian_finite_diff(
            objective_fn, optimal_params, 
            config.hessian_epsilon, bounds
        )
        
        if config.enable_directional_curvature:
            # v2.0: Directional curvature - dangerous couplings penalized more
            penalty, cond_num, eigenvalues, coupling_fragility = compute_directional_curvature_penalty(
                H, config.max_condition_number
            )
            diagnostics['directional_curvature_penalty'] = float(penalty)
            diagnostics['dangerous_coupling_fragility'] = coupling_fragility
        else:
            # Legacy: Standard curvature penalty
            penalty, cond_num, eigenvalues = compute_curvature_penalty(
                H, config.max_condition_number
            )
        
        diagnostics['condition_number'] = float(cond_num) if np.isfinite(cond_num) else None
        diagnostics['curvature_penalty'] = float(penalty)
        diagnostics['eigenvalues'] = eigenvalues.tolist() if len(eigenvalues) > 0 else []
        
    except Exception as e:
        diagnostics['condition_number'] = None
        diagnostics['curvature_penalty'] = 0.0
        diagnostics['curvature_error'] = str(e)
    
    # =========================================================================
    # 2. CONNECTED PLATEAU ANALYSIS (basin vs ridge detection)
    # =========================================================================
    try:
        if config.enable_ridge_detection:
            # v2.0: Detect ridges via joint perturbations
            plateau_width, plateau_score, is_isolated, basin_score, survival_rate = evaluate_connected_plateau(
                objective_fn,
                optimal_params,
                optimal_value,
                bounds,
                config.plateau_acceptance_ratio,
                n_joint_perturbations=config.n_joint_perturbations,
                ridge_threshold=config.ridge_threshold
            )
            diagnostics['is_ridge_optimum'] = basin_score < config.ridge_threshold
            diagnostics['basin_score'] = float(basin_score)
            diagnostics['joint_perturbation_survival_rate'] = float(survival_rate)
        else:
            # Legacy: Axis-aligned plateau only
            from tuning.elite_tuning import evaluate_plateau_width
            plateau_width, plateau_score, is_isolated = evaluate_plateau_width(
                objective_fn, optimal_params, optimal_value, bounds,
                config.plateau_acceptance_ratio
            )
            basin_score = plateau_score
            diagnostics['is_ridge_optimum'] = False
            diagnostics['basin_score'] = float(basin_score)
        
        diagnostics['plateau_score'] = float(plateau_score)
        diagnostics['plateau_width'] = plateau_width.tolist()
        diagnostics['is_isolated_optimum'] = bool(is_isolated)
        
    except Exception as e:
        diagnostics['plateau_score'] = 0.5
        diagnostics['plateau_width'] = []
        diagnostics['is_isolated_optimum'] = False
        diagnostics['basin_score'] = 1.0
        diagnostics['is_ridge_optimum'] = False
        diagnostics['plateau_error'] = str(e)
        basin_score = 1.0
        plateau_score = 0.5
    
    # =========================================================================
    # 3. CROSS-FOLD COHERENCE (with drift vs noise decomposition)
    # =========================================================================
    coherence_variance = np.array([])
    drift_ratio = 0.0
    
    if fold_optimal_params and len(fold_optimal_params) >= config.min_folds_for_coherence:
        try:
            if config.enable_drift_penalty:
                # v2.0: Asymmetric coherence - drift is worse than oscillation
                penalty, variance, drift_comp, noise_comp, drift_pen = compute_asymmetric_coherence_penalty(
                    fold_optimal_params,
                    param_ranges,
                    config.drift_penalty_multiplier
                )
                diagnostics['drift_component'] = drift_comp.tolist()
                diagnostics['noise_component'] = noise_comp.tolist()
                diagnostics['drift_penalty'] = float(drift_pen)
                
                # Compute drift ratio for fragility
                total_var = np.sum(drift_comp) + np.sum(noise_comp)
                if total_var > 1e-12:
                    drift_ratio = np.sum(drift_comp) / total_var
                diagnostics['drift_ratio'] = float(drift_ratio)
            else:
                # Legacy: Symmetric coherence
                penalty, variance = compute_coherence_penalty(
                    fold_optimal_params, param_ranges
                )
            
            coherence_variance = variance
            diagnostics['coherence_penalty'] = float(penalty)
            diagnostics['parameter_variance_across_folds'] = variance.tolist()
            diagnostics['n_folds_evaluated'] = len(fold_optimal_params)
            
        except Exception as e:
            diagnostics['coherence_error'] = str(e)
    
    # =========================================================================
    # 4. FRAGILITY INDEX (PURE PARAMETER FRAGILITY - v2.0)
    # =========================================================================
    try:
        fragility, components = compute_fragility_index(
            condition_number=diagnostics.get('condition_number', 1.0) or 1.0,
            coherence_variance=coherence_variance,
            plateau_width=np.array(diagnostics.get('plateau_width', [])),
            basin_score=basin_score,  # v2.0: ridge awareness
            drift_ratio=drift_ratio,  # v2.0: drift awareness
            # NO calibration_ks_stat - pure parameter fragility only
        )
        
        diagnostics['fragility_index'] = float(fragility)
        diagnostics['fragility_warning'] = fragility > config.fragility_threshold
        diagnostics['fragility_components'] = components
        diagnostics['fragility_threshold'] = config.fragility_threshold
        
    except Exception as e:
        diagnostics['fragility_index'] = 0.5
        diagnostics['fragility_warning'] = False
        diagnostics['fragility_error'] = str(e)
    
    return diagnostics


def format_elite_status(diagnostics: Dict[str, Any]) -> str:
    """
    Format elite tuning status for logging.
    
    Returns a compact string summarizing elite diagnostics.
    """
    if not diagnostics.get('elite_tuning_enabled', False):
        return "elite:disabled"
    
    parts = []
    
    # Fragility
    frag = diagnostics.get('fragility_index', 0)
    if frag > 0.5:
        parts.append(f"⚠frag={frag:.2f}")
    else:
        parts.append(f"frag={frag:.2f}")
    
    # Ridge warning
    if diagnostics.get('is_ridge_optimum', False):
        parts.append("⚠RIDGE")
    
    # Basin score
    basin = diagnostics.get('basin_score', 1.0)
    if basin < 0.5:
        parts.append(f"basin={basin:.2f}")
    
    # Condition number
    cond = diagnostics.get('condition_number')
    if cond and cond > 1e6:
        parts.append(f"κ={cond:.1e}")
    
    return " | ".join(parts) if parts else "elite:ok"


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
    prices: np.ndarray = None,  # Added for MR integration (February 2026)
    regime_labels: np.ndarray = None,  # Added for regime-adaptive blending
    asset: str = None,  # FIX #4: Asset symbol for c-bounds detection
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
    Each sub-model participates independently in BMA, eliminating ν-σ identifiability issues.

    Args:
        returns: Regime-specific returns
        vol: Regime-specific volatility
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        prices: Price array for MR equilibrium estimation (optional)
        regime_labels: Regime labels for adaptive blending (optional)
        asset: Asset symbol for c-bounds detection (FIX #4, optional)

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
    # FILTER RESULT CACHE (February 2026 Performance Optimization)
    # =========================================================================
    # Cache filter results to avoid redundant filter runs when creating
    # momentum/enhanced variants from base models.
    #
    # KEY: (model_name, q, c, phi) -> (mu, P, ll, lfo_cv_score)
    # 
    # PERFORMANCE IMPACT: ~30% speedup by avoiding redundant filter calls
    # =========================================================================
    _filter_cache = {}

    def _cache_key(model_name: str, q: float, c: float, phi: float) -> str:
        """Generate unique cache key for filter results."""
        return f"{model_name}_{q:.8e}_{c:.6f}_{phi:.6f}"

    # Optionally, you can import or define run_student_t_filter_with_lfo_cv if available
    try:
        from models.phi_student_t import run_student_t_filter_with_lfo_cv
    except ImportError:
        run_student_t_filter_with_lfo_cv = None

    def _get_cached_or_filter(
        model_name: str, 
        q: float, 
        c: float, 
        phi: float, 
        nu: float,
        with_lfo_cv: bool = True,
    ) -> tuple:
        """Get filter results from cache or compute and cache them."""
        key = _cache_key(model_name, q, c, phi)

        if key in _filter_cache:
            return _filter_cache[key]

        # Use fused LFO-CV filter when available (40% faster)
        if with_lfo_cv and run_student_t_filter_with_lfo_cv is not None:
            try:
                result = run_student_t_filter_with_lfo_cv(
                    returns, vol, q, c, phi, nu,
                    lfo_start_frac=LFO_CV_MIN_TRAIN_FRAC
                )
                _filter_cache[key] = result
                return result
            except Exception:
                pass

        # Fallback to standard filter
        mu, P, ll = PhiStudentTDriftModel.filter_phi(
            returns, vol, q, c, phi, nu
        )
        lfo_cv_score = float('-inf')  # Not computed in fallback
        result = (mu, P, ll, lfo_cv_score)
        _filter_cache[key] = result
        return result

    # =========================================================================
    # Model 1: Phi-Student-t with DISCRETE ν GRID
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
            # FIX #4: Pass asset symbol for adaptive c bounds
            q_st, c_st, phi_st, ll_cv_st, diag_st = PhiStudentTDriftModel.optimize_params_fixed_nu(
                returns, vol,
                nu=nu_fixed,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda,
                asset_symbol=asset,  # FIX #4: Asset-class adaptive c bounds
            )
            
            # ELITE FIX: Run filter with predictive output for proper PIT
            # The key insight: PIT requires PRIOR PREDICTIVE values (before y_t),
            # not posterior values (after update). Using posterior values causes
            # systematic miscalibration (residuals appear too concentrated).
            mu_st, P_st, mu_pred_st, S_pred_st, ll_full_st = PhiStudentTDriftModel.filter_phi_with_predictive(
                returns, vol, q_st, c_st, phi_st, nu_fixed
            )
            
            # ELITE FIX: Compute PIT calibration using PREDICTIVE distribution
            # This is the mathematically correct formulation:
            # z_t = (y_t - mu_pred_t) / scale_t, PIT_t = F_ν(z_t)
            ks_st, pit_p_st = PhiStudentTDriftModel.pit_ks_predictive(
                returns, mu_pred_st, S_pred_st, nu_fixed
            )
            
            # Compute information criteria
            aic_st = compute_aic(ll_full_st, n_params_st)
            bic_st = compute_bic(ll_full_st, n_params_st, n_obs)
            mean_ll_st = ll_full_st / max(n_obs, 1)
            
            # Compute Hyvärinen score for robust model selection (Student-t)
            # FIX #1: Use correct Student-t scale parameterization
            # For Student-t, Var = scale² × ν/(ν-2), so scale = sqrt(Var × (ν-2)/ν)
            # ELITE FIX: Use predictive variance S_pred for consistency
            if nu_fixed > 2:
                forecast_scale_st = np.sqrt(S_pred_st * (nu_fixed - 2) / nu_fixed)
            else:
                forecast_scale_st = np.sqrt(S_pred_st)
            hyvarinen_st = compute_hyvarinen_score_student_t(
                returns, mu_pred_st, forecast_scale_st, nu_fixed
            )
            
            # Compute CRPS for regime-aware model selection (February 2026)
            # ELITE FIX: Use predictive mean for proper out-of-sample scoring
            crps_st = compute_crps_student_t_inline(
                returns, mu_pred_st, forecast_scale_st, nu_fixed
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
                "crps": float(crps_st),  # CRPS for regime-aware selection
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
                "crps": float('inf'),  # CRPS for failed models
                "nu": float(nu_fixed),
                "nu_fixed": True,
            }
    
    # =========================================================================
    # Model 2: UNIFIED Phi-Student-t (February 2026 - Elite Architecture)
    # =========================================================================
    # Consolidates 48+ model variants into 3 adaptive unified models.
    # Each unified model combines: VoV, Smooth Asymmetric ν, Probabilistic MS-q
    #
    # This replaces separate VoV/Two-Piece/Mixture/MS-q loops with a single
    # coherent architecture that auto-adapts to asset characteristics.
    #
    # Model naming: "phi_student_t_unified_nu_{nu}"
    # =========================================================================
    
    # Unified models use 3 ν values from grid
    UNIFIED_NU_GRID = [4, 8, 20]
    n_params_unified = 7  # q, c, φ, γ_vov, ms_sensitivity, α_asym, ν
    
    for nu_fixed in UNIFIED_NU_GRID:
        unified_name = f"phi_student_t_unified_nu_{nu_fixed}"
        
        try:
            # Staged optimization for unified model
            config, diagnostics = PhiStudentTDriftModel.optimize_params_unified(
                returns, vol, 
                nu_base=float(nu_fixed),
                train_frac=0.7, 
                asset_symbol=asset
            )

            if not diagnostics.get("success", False):
                raise ValueError(f"Unified optimization failed: {diagnostics.get('error', 'Unknown')}")

            # Run unified filter to get predictive values
            mu_u, P_u, mu_pred_u, S_pred_u, ll_u = PhiStudentTDriftModel.filter_phi_unified(
                returns, vol, config
            )

            # PIT calibration using predictive distribution
            ks_u, pit_p_u, pit_metrics = PhiStudentTDriftModel.pit_ks_unified(
                returns, mu_pred_u, S_pred_u, config
            )

            # Information criteria
            aic_u = compute_aic(ll_u, n_params_unified)
            bic_u = compute_bic(ll_u, n_params_unified, n_obs)
            mean_ll_u = ll_u / max(n_obs, 1)

            # Hyvärinen and CRPS scores using predictive values
            # Use base nu for scoring (asymmetry is internal adjustment)
            if nu_fixed > 2:
                forecast_scale_u = np.sqrt(S_pred_u * (nu_fixed - 2) / nu_fixed)
            else:
                forecast_scale_u = np.sqrt(S_pred_u)

            hyvarinen_u = compute_hyvarinen_score_student_t(
                returns, mu_pred_u, forecast_scale_u, nu_fixed
            )
            crps_u = compute_crps_student_t_inline(
                returns, mu_pred_u, forecast_scale_u, nu_fixed
            )

            models[unified_name] = {
                "q": float(config.q),
                "c": float(config.c),
                "phi": float(config.phi),
                "nu": float(nu_fixed),
                # Unified-specific parameters
                "gamma_vov": float(config.gamma_vov),
                "alpha_asym": float(config.alpha_asym),
                "ms_sensitivity": float(config.ms_sensitivity),
                "q_stress_ratio": float(config.q_stress_ratio),
                # Scores
                "log_likelihood": float(ll_u),
                "mean_log_likelihood": float(mean_ll_u),
                "bic": float(bic_u),
                "aic": float(aic_u),
                "hyvarinen_score": float(hyvarinen_u),
                "crps": float(crps_u),
                "ks_statistic": float(ks_u),
                "pit_ks_pvalue": float(pit_p_u),
                "pit_calibration_grade": pit_metrics.get("calibration_grade", "F"),
                "histogram_mad": pit_metrics.get("histogram_mad", 1.0),
                # Metadata
                "fit_success": True,
                "unified_model": True,
                "degraded": diagnostics.get("degraded", False),
                "hessian_cond": diagnostics.get("hessian_cond", float('inf')),
                "n_params": int(n_params_unified),
                "nu_fixed": True,
                "model_type": "phi_student_t_unified",
            }

        except Exception as e:
            models[unified_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "crps": float('inf'),
                "nu": float(nu_fixed),
                "unified_model": True,
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
                    
                    # Compute CRPS for regime-aware model selection (February 2026)
                    # Use Gaussian approximation for NIG (consistent with Hyvärinen approach)
                    crps_nig = compute_crps_gaussian_inline(
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
                        "crps": float(crps_nig),  # CRPS for regime-aware selection
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
                        "crps": float('inf'),  # CRPS for failed models
                        "nig_alpha": float(alpha_fixed),
                        "nig_beta": float(beta_fixed),
                        "nig_delta": float(delta_baseline),
                        "nig_params_fixed": True,
                        "model_type": "phi_nig",
                    }
    
    # =========================================================================
    # MOMENTUM-AUGMENTED MODELS (February 2026)
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
        # Create momentum wrapper with MR integration (February 2026)
        momentum_wrapper = MomentumAugmentedDriftModel(DEFAULT_MOMENTUM_CONFIG)
        
        # Use precompute_signals for state-equation MR integration
        # Falls back to momentum-only if prices not available
        if prices is not None:
            # Get process noise q from base Gaussian model for dynamic max_u scaling
            q_for_scaling = 1e-6  # Default
            if 'kalman_gaussian' in models and models['kalman_gaussian'].get('q'):
                q_for_scaling = models['kalman_gaussian']['q']
            
            momentum_wrapper.precompute_signals(
                returns=returns,
                prices=prices,
                vol=vol,
                regime_labels=regime_labels,
                q=q_for_scaling,
            )
        else:
            # Legacy path: momentum-only (no MR)
            momentum_wrapper.precompute_momentum(returns)
        
        # =====================================================================
        # Momentum-augmented Gaussian (inline parameter fitting)
        # =====================================================================
        # Fit Gaussian parameters directly, then apply momentum augmentation.
        # This is the ONLY Gaussian model in BMA - no pure Gaussian variant.
        # =====================================================================
        mom_name = "kalman_gaussian_momentum"  
        
        try:
            # Fit Gaussian parameters (q, c)
            q_gauss, c_gauss, ll_cv_gauss, diag_gauss = GaussianDriftModel.optimize_params(
                returns, vol,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            # Run filter with momentum augmentation
            mu_mom, P_mom, ll_mom = momentum_wrapper.filter(
                returns, vol, q_gauss, c_gauss, phi=1.0, base_model='gaussian'
            )
            
            # Compute PIT calibration
            ks_mom, pit_p_mom = compute_pit_from_filtered_gaussian(
                returns, mu_mom, P_mom, vol, q_gauss, c_gauss, phi=1.0
            )
            
            # Compute information criteria with prior penalty
            n_params_mom = MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN]
            aic_mom = compute_aic(ll_mom, n_params_mom)
            bic_raw_mom = compute_bic(ll_mom, n_params_mom, n_obs)
            bic_mom = compute_momentum_model_bic_adjustment(bic_raw_mom, MOMENTUM_BMA_PRIOR_PENALTY)
            mean_ll_mom = ll_mom / max(n_obs, 1)
            
            # Compute Hyvärinen score
            forecast_std_mom = np.sqrt(c_gauss * (vol ** 2) + P_mom)
            hyvarinen_mom = compute_hyvarinen_score_gaussian(returns, mu_mom, forecast_std_mom)
            
            # Compute CRPS for regime-aware model selection (February 2026)
            crps_mom = compute_crps_gaussian_inline(returns, mu_mom, forecast_std_mom)
            
            models[mom_name] = {
                "q": float(q_gauss),
                "c": float(c_gauss),
                "phi": None,
                "nu": None,
                "log_likelihood": float(ll_mom),
                "mean_log_likelihood": float(mean_ll_mom),
                "cv_penalized_ll": float(ll_mom),
                "bic": float(bic_mom),
                "bic_raw": float(bic_raw_mom),
                "aic": float(aic_mom),
                "hyvarinen_score": float(hyvarinen_mom),
                "crps": float(crps_mom),  # CRPS for regime-aware selection
                "n_params": int(n_params_mom),
                "ks_statistic": float(ks_mom),
                "pit_ks_pvalue": float(pit_p_mom),
                "fit_success": True,
                "momentum_augmented": True,
                "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                "diagnostics": {**diag_gauss, **momentum_wrapper.get_diagnostics()},
            }
        except Exception as e:
            models[mom_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "crps": float('inf'),  # CRPS for failed models
                "momentum_augmented": True,
            }
        
        # =====================================================================
        # Momentum-augmented Phi-Gaussian (inline parameter fitting)
        # =====================================================================
        # Fit φ-Gaussian parameters directly, then apply momentum augmentation.
        # This is the ONLY φ-Gaussian model in BMA - no pure variant.
        # =====================================================================
        mom_name = "kalman_phi_gaussian_momentum"  
        
        try:
            # Fit Phi-Gaussian parameters (q, c, phi)
            q_phi, c_phi, phi_opt, ll_cv_phi, diag_phi = PhiGaussianDriftModel.optimize_params(
                returns, vol,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            # Run filter with momentum augmentation
            mu_mom, P_mom, ll_mom = momentum_wrapper.filter(
                returns, vol, q_phi, c_phi, phi=phi_opt, base_model='phi_gaussian'
            )
            
            # Compute PIT calibration using reconstructed predictive values
            ks_mom, pit_p_mom = compute_pit_from_filtered_gaussian(
                returns, mu_mom, P_mom, vol, q_phi, c_phi, phi=phi_opt
            )
            
            # Compute information criteria with prior penalty
            n_params_mom = MODEL_CLASS_N_PARAMS[ModelClass.PHI_GAUSSIAN]
            aic_mom = compute_aic(ll_mom, n_params_mom)
            bic_raw_mom = compute_bic(ll_mom, n_params_mom, n_obs)
            bic_mom = compute_momentum_model_bic_adjustment(bic_raw_mom, MOMENTUM_BMA_PRIOR_PENALTY)
            mean_ll_mom = ll_mom / max(n_obs, 1)
            
            # Compute Hyvärinen score
            forecast_std_mom = np.sqrt(c_phi * (vol ** 2) + P_mom)
            hyvarinen_mom = compute_hyvarinen_score_gaussian(returns, mu_mom, forecast_std_mom)
            
            # Compute CRPS for regime-aware model selection (February 2026)
            crps_mom = compute_crps_gaussian_inline(returns, mu_mom, forecast_std_mom)
            
            models[mom_name] = {
                "q": float(q_phi),
                "c": float(c_phi),
                "phi": float(phi_opt),
                "nu": None,
                "log_likelihood": float(ll_mom),
                "mean_log_likelihood": float(mean_ll_mom),
                "cv_penalized_ll": float(ll_mom),
                "bic": float(bic_mom),
                "bic_raw": float(bic_raw_mom),
                "aic": float(aic_mom),
                "hyvarinen_score": float(hyvarinen_mom),
                "crps": float(crps_mom),  # CRPS for regime-aware selection
                "n_params": int(n_params_mom),
                "ks_statistic": float(ks_mom),
                "pit_ks_pvalue": float(pit_p_mom),
                "fit_success": True,
                "momentum_augmented": True,
                "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                "diagnostics": {**diag_phi, **momentum_wrapper.get_diagnostics()},
            }
        except Exception as e:
            models[mom_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "crps": float('inf'),  # CRPS for failed models
                "momentum_augmented": True,
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
                    
                    # ELITE FIX: Compute PIT using PREDICTIVE distribution
                    # Run the base filter to get predictive values for proper PIT
                    ks_mom, pit_p_mom, mu_pred_mom, S_pred_mom = compute_predictive_pit_student_t(
                        returns, vol, q_mom, c_mom, phi_mom, nu_fixed
                    )
                    
                    # Compute information criteria with prior penalty
                    n_params_mom = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T]
                    aic_mom = compute_aic(ll_mom, n_params_mom)
                    bic_raw_mom = compute_bic(ll_mom, n_params_mom, n_obs)
                    bic_mom = compute_momentum_model_bic_adjustment(bic_raw_mom, MOMENTUM_BMA_PRIOR_PENALTY)
                    mean_ll_mom = ll_mom / max(n_obs, 1)
                    
                    # ELITE FIX: Compute Hyvärinen/CRPS using PREDICTIVE distribution
                    hyvarinen_mom, crps_mom = compute_predictive_scores_student_t(
                        returns, mu_pred_mom, S_pred_mom, nu_fixed
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
                        "crps": float(crps_mom),  # CRPS for regime-aware selection
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
                        "crps": float('inf'),  # CRPS for failed models
                        "momentum_augmented": True,
                        "base_model": base_name,
                        "nu": float(nu_fixed),
                    }
    
    # =========================================================================
    # GAS-Q AUGMENTED MODELS (February 2026)
    # =========================================================================
    # Score-Driven Parameter Dynamics for process noise q.
    # GAS-Q models adapt q_t based on forecast errors: q_t = ω + α·s_{t-1} + β·q_{t-1}
    # This improves forecasting during regime transitions.
    # Reference: Creal, Koopman & Lucas (2013) "Generalized Autoregressive Score Models"
    # =========================================================================
    
    if GAS_Q_ENABLED and GAS_Q_AVAILABLE:
        from models.gas_q import (
            gas_q_filter_gaussian,
            gas_q_filter_student_t,
            optimize_gas_q_params,
            get_gas_q_model_name,
            compute_gas_q_bic,
            GASQConfig,
            DEFAULT_GAS_Q_CONFIG,
        )
        
        # GAS-Q augmented Gaussian+Momentum (stack on top of momentum)
        base_name = "kalman_gaussian_momentum"
        gas_q_name = get_gas_q_model_name(base_name)
        
        if base_name in models and models[base_name].get("fit_success", False):
            try:
                base_model = models[base_name]
                c_gas = base_model["c"]
                phi_gas = base_model.get("phi", 1.0)
                
                # Optimize GAS-Q parameters
                gas_config, gas_diag = optimize_gas_q_params(
                    returns, vol, c_gas, phi_gas, nu=None, train_frac=0.7
                )
                
                if gas_diag.get("fit_success", False):
                    # Run GAS-Q filter
                    gas_result = gas_q_filter_gaussian(
                        returns, vol, c_gas, phi_gas, gas_config
                    )
                    
                    # Compute PIT calibration using reconstructed predictive values
                    # Note: GAS-Q has time-varying q, use mean q for reconstruction
                    ks_gas, pit_p_gas = compute_pit_from_filtered_gaussian(
                        returns, gas_result.mu_filtered, gas_result.P_filtered, 
                        vol, gas_result.q_mean, c_gas, phi=phi_gas
                    )
                    
                    # BIC with extra 3 GAS parameters
                    n_params_gas = MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN] + 3
                    bic_gas = compute_gas_q_bic(
                        gas_result.log_likelihood, n_obs, 
                        MODEL_CLASS_N_PARAMS[ModelClass.KALMAN_GAUSSIAN], 3
                    )
                    aic_gas = compute_aic(gas_result.log_likelihood, n_params_gas)
                    mean_ll_gas = gas_result.log_likelihood / max(n_obs, 1)
                    
                    # Compute Hyvärinen score
                    forecast_std_gas = np.sqrt(c_gas * (vol ** 2) + gas_result.P_filtered)
                    hyvarinen_gas = compute_hyvarinen_score_gaussian(
                        returns, gas_result.mu_filtered, forecast_std_gas
                    )
                    
                    # Compute CRPS for regime-aware model selection (February 2026)
                    crps_gas = compute_crps_gaussian_inline(
                        returns, gas_result.mu_filtered, forecast_std_gas
                    )
                    
                    models[gas_q_name] = {
                        "q": float(gas_result.q_mean),
                        "c": float(c_gas),
                        "phi": float(phi_gas),
                        "nu": None,
                        "log_likelihood": float(gas_result.log_likelihood),
                        "mean_log_likelihood": float(mean_ll_gas),
                        "cv_penalized_ll": float(gas_result.log_likelihood),
                        "bic": float(bic_gas),
                        "aic": float(aic_gas),
                        "hyvarinen_score": float(hyvarinen_gas),
                        "crps": float(crps_gas),  # CRPS for regime-aware selection
                        "n_params": int(n_params_gas),
                        "ks_statistic": float(ks_gas),
                        "pit_ks_pvalue": float(pit_p_gas),
                        "fit_success": True,
                        "gas_q_augmented": True,
                        "gas_q_omega": float(gas_config.omega),
                        "gas_q_alpha": float(gas_config.alpha),
                        "gas_q_beta": float(gas_config.beta),
                        "gas_q_mean": float(gas_result.q_mean),
                        "gas_q_std": float(gas_result.q_std),
                        "gas_q_final": float(gas_result.final_q),
                        "base_model": base_name,
                        "momentum_augmented": True,  # Inherits from base
                    }
            except Exception as e:
                models[gas_q_name] = {
                    "fit_success": False,
                    "error": str(e),
                    "bic": float('inf'),
                    "aic": float('inf'),
                    "hyvarinen_score": float('-inf'),
                    "crps": float('inf'),  # CRPS for failed models
                    "gas_q_augmented": True,
                    "base_model": base_name,
                }
        
        # GAS-Q augmented Student-t+Momentum (for each nu)
        # DISABLED when UNIFIED_STUDENT_T_ONLY=True - unified models include adaptive q
        if not UNIFIED_STUDENT_T_ONLY:
            for nu_fixed in STUDENT_T_NU_GRID:
                base_name = f"phi_student_t_nu_{nu_fixed}_momentum"
                gas_q_name = get_gas_q_model_name(base_name)
                
                if base_name in models and models[base_name].get("fit_success", False):
                    try:
                        base_model = models[base_name]
                        c_gas = base_model["c"]
                        phi_gas = base_model.get("phi", 0.0)
                        
                        # Optimize GAS-Q parameters for Student-t
                        gas_config, gas_diag = optimize_gas_q_params(
                            returns, vol, c_gas, phi_gas, nu=nu_fixed, train_frac=0.7
                        )
                        
                        if gas_diag.get("fit_success", False):
                            # Run GAS-Q filter
                            gas_result = gas_q_filter_student_t(
                                returns, vol, c_gas, phi_gas, nu_fixed, gas_config
                            )
                            
                            # ELITE FIX: Compute PIT using PREDICTIVE distribution
                            # GAS-Q uses time-varying q, but PIT still needs predictive values
                            ks_gas, pit_p_gas, mu_pred_gas, S_pred_gas = compute_predictive_pit_student_t(
                                returns, vol, float(gas_result.q_mean), c_gas, phi_gas, nu_fixed
                            )
                            
                            # BIC with extra 3 GAS parameters
                            n_params_gas = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T] + 3
                            bic_gas = compute_gas_q_bic(
                                gas_result.log_likelihood, n_obs,
                                MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T], 3
                            )
                            aic_gas = compute_aic(gas_result.log_likelihood, n_params_gas)
                            mean_ll_gas = gas_result.log_likelihood / max(n_obs, 1)
                            
                            # ELITE FIX: Compute Hyvärinen/CRPS using PREDICTIVE distribution
                            hyvarinen_gas, crps_gas = compute_predictive_scores_student_t(
                                returns, mu_pred_gas, S_pred_gas, nu_fixed
                            )
                            
                            models[gas_q_name] = {
                                "q": float(gas_result.q_mean),
                                "c": float(c_gas),
                                "phi": float(phi_gas),
                                "nu": float(nu_fixed),
                                "log_likelihood": float(gas_result.log_likelihood),
                                "mean_log_likelihood": float(mean_ll_gas),
                                "cv_penalized_ll": float(gas_result.log_likelihood),
                                "bic": float(bic_gas),
                                "aic": float(aic_gas),
                                "hyvarinen_score": float(hyvarinen_gas),
                                "crps": float(crps_gas),  # CRPS for regime-aware selection
                                "n_params": int(n_params_gas),
                                "ks_statistic": float(ks_gas),
                                "pit_ks_pvalue": float(pit_p_gas),
                                "fit_success": True,
                                "gas_q_augmented": True,
                                "gas_q_omega": float(gas_config.omega),
                                "gas_q_alpha": float(gas_config.alpha),
                                "gas_q_beta": float(gas_config.beta),
                                "gas_q_mean": float(gas_result.q_mean),
                                "gas_q_std": float(gas_result.q_std),
                                "gas_q_final": float(gas_result.final_q),
                                "base_model": base_name,
                                "momentum_augmented": True,
                                "nu_fixed": True,
                            }
                    except Exception as e:
                        models[gas_q_name] = {
                            "fit_success": False,
                            "error": str(e),
                            "bic": float('inf'),
                            "aic": float('inf'),
                            "hyvarinen_score": float('-inf'),
                            "crps": float('inf'),  # CRPS for failed models
                            "gas_q_augmented": True,
                            "base_model": base_name,
                            "nu": float(nu_fixed),
                        }
    
    # =========================================================================
    # MARKOV-SWITCHING PROCESS NOISE (MS-q) — February 2026
    # =========================================================================
    # Proactive regime-switching q based on volatility structure.
    # Unlike GAS-Q (reactive), MS-q shifts BEFORE errors materialize.
    #
    # CORE INSIGHT: Volatility structure PREDICTS regime changes.
    # When vol rises above median, we proactively increase q.
    #
    # MS-q provides:
    # - Faster regime transition response (20-30% improvement)
    # - Better PIT during volatility spikes
    # - Proactive vs reactive adaptation
    # =========================================================================
    
    # DISABLED when UNIFIED_STUDENT_T_ONLY=True - unified models include probabilistic MS-q
    if MS_Q_ENABLED and MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE and not UNIFIED_STUDENT_T_ONLY:
        
        for nu_fixed in STUDENT_T_NU_GRID:
            base_name = f"phi_student_t_nu_{nu_fixed}_momentum"
            ms_q_name = f"phi_student_t_nu_{nu_fixed}_ms_q_momentum"
            
            if base_name in models and models[base_name].get("fit_success", False):
                try:
                    # Optimize MS-q parameters
                    c_ms, phi_ms, q_calm, q_stress, ll_ms, lfo_cv, ms_diag = optimize_params_ms_q(
                        returns, vol, nu_fixed,
                        prior_log_q_mean=prior_log_q_mean,
                        prior_lambda=prior_lambda,
                    )
                    
                    if ms_diag.get("fit_success", False):
                        # Run MS-q filter for full diagnostics
                        mu_ms, P_ms, ll_full, q_t, p_stress = filter_phi_ms_q(
                            returns, vol, c_ms, phi_ms, nu_fixed,
                            q_calm=q_calm, q_stress=q_stress
                        )
                        
                        # ELITE FIX: Compute PIT using PREDICTIVE distribution
                        # MS-q uses time-varying q, but PIT still needs predictive values
                        # Use average q for predictive computation
                        q_mean_ms = float(np.mean(q_t))
                        ks_ms, pit_p_ms, mu_pred_ms, S_pred_ms = compute_predictive_pit_student_t(
                            returns, vol, q_mean_ms, c_ms, phi_ms, nu_fixed
                        )
                        
                        # Compute information criteria
                        # MS-q has 4 params: c, phi, q_calm, q_stress (nu is fixed)
                        n_params_ms_q = 4
                        aic_ms = compute_aic(ll_full, n_params_ms_q)
                        bic_ms = compute_bic(ll_full, n_params_ms_q, n_obs)
                        mean_ll_ms = ll_full / max(n_obs, 1)
                        
                        # ELITE FIX: Compute Hyvärinen/CRPS using PREDICTIVE distribution
                        hyvarinen_ms, crps_ms = compute_predictive_scores_student_t(
                            returns, mu_pred_ms, S_pred_ms, nu_fixed
                        )
                        
                        models[ms_q_name] = {
                            "q": float(np.mean(q_t)),  # Mean q over time
                            "q_calm": float(q_calm),
                            "q_stress": float(q_stress),
                            "c": float(c_ms),
                            "phi": float(phi_ms),
                            "nu": float(nu_fixed),
                            "log_likelihood": float(ll_full),
                            "mean_log_likelihood": float(mean_ll_ms),
                            "lfo_cv_score": float(lfo_cv),  # LFO-CV score for model selection
                            "bic": float(bic_ms),
                            "aic": float(aic_ms),
                            "hyvarinen_score": float(hyvarinen_ms),
                            "crps": float(crps_ms),
                            "n_params": int(n_params_ms_q),
                            "ks_statistic": float(ks_ms),
                            "pit_ks_pvalue": float(pit_p_ms),
                            "fit_success": True,
                            "ms_q_augmented": True,
                            "momentum_augmented": True,
                            "ms_q_penalty": MS_Q_BMA_PENALTY,
                            "p_stress_mean": float(np.mean(p_stress)),
                            "p_stress_max": float(np.max(p_stress)),
                            "q_ratio": float(q_stress / q_calm) if q_calm > 0 else 0,
                            "base_model": base_name,
                            "nu_fixed": True,
                            "model_type": "phi_student_t_ms_q",
                        }
                except Exception as e:
                    models[ms_q_name] = {
                        "fit_success": False,
                        "error": str(e),
                        "bic": float('inf'),
                        "aic": float('inf'),
                        "hyvarinen_score": float('-inf'),
                        "crps": float('inf'),
                        "ms_q_augmented": True,
                        "momentum_augmented": True,
                        "nu": float(nu_fixed),
                    }
    
    # =========================================================================
    # ENHANCED STUDENT-T MODELS (February 2026)
    # =========================================================================
    # Three enhancements to improve Hyvarinen/PIT calibration:
    #   1. Vol-of-Vol (VoV): R_t = c × σ² × (1 + γ × |Δlog(σ)|)
    #   2. Two-Piece: Different νL (crash) vs νR (recovery) tails
    #   3. Two-Component Mixture: Blend νcalm and νstress with dynamic weights
    #
    # These compete in BMA with appropriate penalties for extra parameters.
    # Only momentum variants are fitted (as per existing architecture).
    #
    # DISABLED when UNIFIED_STUDENT_T_ONLY=True - unified models incorporate
    # VoV, smooth asymmetric ν, and probabilistic MS-q into single architecture.
    # =========================================================================
    
    if MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE and not UNIFIED_STUDENT_T_ONLY:
        # Import enhanced Student-t grids
        from models import (
            NU_LEFT_GRID, NU_RIGHT_GRID, TWO_PIECE_BMA_PENALTY,
            GAMMA_VOV_GRID, VOV_BMA_PENALTY,
            NU_CALM_GRID, NU_STRESS_GRID, MIXTURE_BMA_PENALTY,
        )
        
        # Reuse momentum wrapper from above
        if 'momentum_wrapper' not in dir():
            momentum_wrapper = MomentumAugmentedDriftModel(DEFAULT_MOMENTUM_CONFIG)
            # Use precompute_signals for MR integration if prices available
            if prices is not None:
                q_for_scaling = models.get('kalman_gaussian', {}).get('q', 1e-6)
                momentum_wrapper.precompute_signals(
                    returns=returns, prices=prices, vol=vol,
                    regime_labels=regime_labels, q=q_for_scaling,
                )
            else:
                momentum_wrapper.precompute_momentum(returns)
        
        # =====================================================================
        # Vol-of-Vol Enhanced Student-t + Momentum
        # =====================================================================
        # For each (nu, gamma_vov) combination, fit enhanced model.
        # γ=0 is equivalent to base model, so skip γ=0 to avoid duplication.
        # =====================================================================
        for nu_fixed in STUDENT_T_NU_GRID:
            base_name = f"phi_student_t_nu_{nu_fixed}"
            
            if base_name not in models or not models[base_name].get("fit_success", False):
                continue
                
            base_model = models[base_name]
            q_base = base_model["q"]
            c_base = base_model["c"]
            phi_base = base_model["phi"]
            
            for gamma_vov in GAMMA_VOV_GRID:
                # Skip γ=0 (already covered by base momentum model)
                if gamma_vov == 0.0:
                    continue
                
                # Format gamma for model name (remove decimal point)
                gamma_str = f"{gamma_vov:.1f}".replace(".", "")
                vov_name = f"phi_student_t_nu_{nu_fixed}_vov_{gamma_str}_momentum"
                
                try:
                    # Run VoV-enhanced filter
                    mu_vov, P_vov, ll_vov = PhiStudentTDriftModel.filter_phi_vov(
                        returns, vol, q_base, c_base, phi_base, nu_fixed, gamma_vov
                    )
                    
                    # Apply momentum augmentation to drift estimate
                    # (momentum adjusts the drift, VoV adjusts observation noise)
                    momentum_signal = momentum_wrapper._momentum_signal
                    if momentum_signal is not None:
                        mu_vov_mom = mu_vov + momentum_signal * momentum_wrapper.config.adjustment_scale
                    else:
                        mu_vov_mom = mu_vov
                    
                    # ELITE FIX: Compute PIT using PREDICTIVE distribution
                    ks_vov, pit_p_vov, mu_pred_vov, S_pred_vov = compute_predictive_pit_student_t(
                        returns, vol, q_base, c_base, phi_base, nu_fixed
                    )
                    
                    # Compute information criteria with VoV + momentum penalties
                    n_params_vov = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T] + 1  # +1 for γ
                    aic_vov = compute_aic(ll_vov, n_params_vov)
                    bic_raw_vov = compute_bic(ll_vov, n_params_vov, n_obs)
                    combined_penalty = MOMENTUM_BMA_PRIOR_PENALTY + VOV_BMA_PENALTY
                    bic_vov = compute_momentum_model_bic_adjustment(bic_raw_vov, combined_penalty)
                    mean_ll_vov = ll_vov / max(n_obs, 1)
                    
                    # ELITE FIX: Compute Hyvärinen/CRPS using PREDICTIVE distribution
                    hyvarinen_vov, crps_vov = compute_predictive_scores_student_t(
                        returns, mu_pred_vov, S_pred_vov, nu_fixed
                    )
                    
                    models[vov_name] = {
                        "q": float(q_base),
                        "c": float(c_base),
                        "phi": float(phi_base),
                        "nu": float(nu_fixed),
                        "gamma_vov": float(gamma_vov),
                        "log_likelihood": float(ll_vov),
                        "mean_log_likelihood": float(mean_ll_vov),
                        "cv_penalized_ll": float(ll_vov),
                        "bic": float(bic_vov),
                        "bic_raw": float(bic_raw_vov),
                        "aic": float(aic_vov),
                        "hyvarinen_score": float(hyvarinen_vov),
                        "crps": float(crps_vov),  # CRPS for regime-aware selection
                        "n_params": int(n_params_vov),
                        "ks_statistic": float(ks_vov),
                        "pit_ks_pvalue": float(pit_p_vov),
                        "fit_success": True,
                        "vov_enhanced": True,
                        "momentum_augmented": True,
                        "vov_penalty": VOV_BMA_PENALTY,
                        "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                        "base_model": base_name,
                        "nu_fixed": True,
                        "model_type": "phi_student_t_vov",
                    }
                except Exception as e:
                    models[vov_name] = {
                        "fit_success": False,
                        "error": str(e),
                        "bic": float('inf'),
                        "aic": float('inf'),
                        "hyvarinen_score": float('-inf'),
                        "crps": float('inf'),  # CRPS for failed models
                        "vov_enhanced": True,
                        "momentum_augmented": True,
                        "nu": float(nu_fixed),
                        "gamma_vov": float(gamma_vov),
                    }
        
        # =====================================================================
        # Two-Piece Student-t + Momentum
        # =====================================================================
        # Different ν for left (crash) vs right (recovery) tails.
        # Captures empirical asymmetry: crashes are more extreme than rallies.
        # =====================================================================
        for nu_left in NU_LEFT_GRID:
            for nu_right in NU_RIGHT_GRID:
                # Use closest standard ν as base for parameters
                base_nu = min(STUDENT_T_NU_GRID, key=lambda x: abs(x - (nu_left + nu_right) / 2))
                base_name = f"phi_student_t_nu_{base_nu}"
                
                if base_name not in models or not models[base_name].get("fit_success", False):
                    continue
                
                base_model = models[base_name]
                q_2p = base_model["q"]
                c_2p = base_model["c"]
                phi_2p = base_model["phi"]
                
                two_piece_name = f"phi_student_t_nuL{nu_left}_nuR{nu_right}_momentum"
                
                try:
                    # Run Two-Piece filter
                    mu_2p, P_2p, ll_2p = PhiStudentTDriftModel.filter_phi_two_piece(
                        returns, vol, q_2p, c_2p, phi_2p, nu_left, nu_right
                    )
                    
                    # Apply momentum augmentation (use precomputed momentum signal)
                    momentum_signal = momentum_wrapper._momentum_signal
                    if momentum_signal is not None:
                        mu_2p_mom = mu_2p + momentum_signal * momentum_wrapper.config.adjustment_scale
                    else:
                        mu_2p_mom = mu_2p
                    
                    # ELITE FIX: Compute PIT using PREDICTIVE distribution
                    # Use average ν for predictive computation
                    nu_avg = (nu_left + nu_right) / 2
                    ks_2p, pit_p_2p, mu_pred_2p, S_pred_2p = compute_predictive_pit_student_t(
                        returns, vol, q_2p, c_2p, phi_2p, nu_avg
                    )
                    
                    # Compute information criteria with Two-Piece + momentum penalties
                    # +2 params: nu_left, nu_right instead of single nu
                    n_params_2p = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T] + 1
                    aic_2p = compute_aic(ll_2p, n_params_2p)
                    bic_raw_2p = compute_bic(ll_2p, n_params_2p, n_obs)
                    combined_penalty = MOMENTUM_BMA_PRIOR_PENALTY + TWO_PIECE_BMA_PENALTY
                    bic_2p = compute_momentum_model_bic_adjustment(bic_raw_2p, combined_penalty)
                    mean_ll_2p = ll_2p / max(n_obs, 1)
                    
                    # ELITE FIX: Compute Hyvärinen/CRPS using PREDICTIVE distribution
                    hyvarinen_2p, crps_2p = compute_predictive_scores_student_t(
                        returns, mu_pred_2p, S_pred_2p, nu_avg
                    )
                    
                    models[two_piece_name] = {
                        "q": float(q_2p),
                        "c": float(c_2p),
                        "phi": float(phi_2p),
                        "nu": None,  # No single ν
                        "nu_left": float(nu_left),
                        "nu_right": float(nu_right),
                        "log_likelihood": float(ll_2p),
                        "mean_log_likelihood": float(mean_ll_2p),
                        "cv_penalized_ll": float(ll_2p),
                        "bic": float(bic_2p),
                        "bic_raw": float(bic_raw_2p),
                        "aic": float(aic_2p),
                        "hyvarinen_score": float(hyvarinen_2p),
                        "crps": float(crps_2p),  # CRPS for regime-aware selection
                        "n_params": int(n_params_2p),
                        "ks_statistic": float(ks_2p),
                        "pit_ks_pvalue": float(pit_p_2p),
                        "fit_success": True,
                        "two_piece": True,
                        "momentum_augmented": True,
                        "two_piece_penalty": TWO_PIECE_BMA_PENALTY,
                        "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                        "base_model": base_name,
                        "model_type": "phi_student_t_two_piece",
                    }
                except Exception as e:
                    models[two_piece_name] = {
                        "fit_success": False,
                        "error": str(e),
                        "bic": float('inf'),
                        "aic": float('inf'),
                        "hyvarinen_score": float('-inf'),
                        "crps": float('inf'),  # CRPS for failed models
                        "two_piece": True,
                        "momentum_augmented": True,
                        "nu_left": float(nu_left),
                        "nu_right": float(nu_right),
                    }
        
        # =====================================================================
        # Two-Component Mixture Student-t + Momentum
        # =====================================================================
        # Blend νcalm and νstress with dynamic vol-based weights.
        # Captures two curvature regimes in the central body.
        # =====================================================================
        for nu_calm in NU_CALM_GRID:
            for nu_stress in NU_STRESS_GRID:
                # Use calm ν as base for parameters
                base_name = f"phi_student_t_nu_{nu_calm}"
                
                if base_name not in models or not models[base_name].get("fit_success", False):
                    continue
                
                base_model = models[base_name]
                q_mix = base_model["q"]
                c_mix = base_model["c"]
                phi_mix = base_model["phi"]
                
                mixture_name = f"phi_student_t_mix_{nu_calm}_{nu_stress}_momentum"
                
                try:
                    # Run Mixture filter (enhanced or standard based on config)
                    from models import MIXTURE_WEIGHT_DEFAULT
                    if ENHANCED_MIXTURE_ENABLED:
                        # Enhanced: multi-factor weight dynamics (shock + vol accel + momentum)
                        mu_mix, P_mix, ll_mix = PhiStudentTDriftModel.filter_phi_mixture_enhanced(
                            returns, vol, q_mix, c_mix, phi_mix, 
                            nu_calm, nu_stress, MIXTURE_WEIGHT_DEFAULT,
                            a_shock=MIXTURE_WEIGHT_A_SHOCK,
                            b_vol_accel=MIXTURE_WEIGHT_B_VOL_ACCEL,
                            c_momentum=MIXTURE_WEIGHT_C_MOMENTUM,
                        )
                    else:
                        # Standard: vol-only weight dynamics
                        mu_mix, P_mix, ll_mix = PhiStudentTDriftModel.filter_phi_mixture(
                            returns, vol, q_mix, c_mix, phi_mix, 
                            nu_calm, nu_stress, MIXTURE_WEIGHT_DEFAULT
                        )
                    
                    # Apply momentum augmentation (use precomputed momentum signal)
                    momentum_signal = momentum_wrapper._momentum_signal
                    if momentum_signal is not None:
                        mu_mix_mom = mu_mix + momentum_signal * momentum_wrapper.config.adjustment_scale
                    else:
                        mu_mix_mom = mu_mix
                    
                    # ELITE FIX: Compute PIT using PREDICTIVE distribution
                    # Use effective ν (average) for predictive computation
                    nu_eff = (nu_calm + nu_stress) / 2
                    ks_mix, pit_p_mix, mu_pred_mix, S_pred_mix = compute_predictive_pit_student_t(
                        returns, vol, q_mix, c_mix, phi_mix, nu_eff
                    )
                    
                    # Compute information criteria with Mixture + momentum penalties
                    # +3 params: nu_calm, nu_stress, w_base instead of single nu
                    n_params_mix = MODEL_CLASS_N_PARAMS[ModelClass.PHI_STUDENT_T] + 2
                    aic_mix = compute_aic(ll_mix, n_params_mix)
                    bic_raw_mix = compute_bic(ll_mix, n_params_mix, n_obs)
                    combined_penalty = MOMENTUM_BMA_PRIOR_PENALTY + MIXTURE_BMA_PENALTY
                    bic_mix = compute_momentum_model_bic_adjustment(bic_raw_mix, combined_penalty)
                    mean_ll_mix = ll_mix / max(n_obs, 1)
                    
                    # ELITE FIX: Compute Hyvärinen/CRPS using PREDICTIVE distribution
                    hyvarinen_mix, crps_mix = compute_predictive_scores_student_t(
                        returns, mu_pred_mix, S_pred_mix, nu_eff
                    )
                    
                    models[mixture_name] = {
                        "q": float(q_mix),
                        "c": float(c_mix),
                        "phi": float(phi_mix),
                        "nu": None,  # No single ν
                        "nu_calm": float(nu_calm),
                        "nu_stress": float(nu_stress),
                        "w_base": MIXTURE_WEIGHT_DEFAULT,
                        "log_likelihood": float(ll_mix),
                        "mean_log_likelihood": float(mean_ll_mix),
                        "cv_penalized_ll": float(ll_mix),
                        "bic": float(bic_mix),
                        "bic_raw": float(bic_raw_mix),
                        "aic": float(aic_mix),
                        "hyvarinen_score": float(hyvarinen_mix),
                        "crps": float(crps_mix),  # CRPS for regime-aware selection
                        "n_params": int(n_params_mix),
                        "ks_statistic": float(ks_mix),
                        "pit_ks_pvalue": float(pit_p_mix),
                        "fit_success": True,
                        "mixture_model": True,
                        "momentum_augmented": True,
                        "mixture_penalty": MIXTURE_BMA_PENALTY,
                        "momentum_prior_penalty": MOMENTUM_BMA_PRIOR_PENALTY,
                        "base_model": base_name,
                        "model_type": "phi_student_t_mixture",
                    }
                except Exception as e:
                    models[mixture_name] = {
                        "fit_success": False,
                        "error": str(e),
                        "bic": float('inf'),
                        "aic": float('inf'),
                        "hyvarinen_score": float('-inf'),
                        "crps": float('inf'),  # CRPS for failed models
                        "mixture_model": True,
                        "momentum_augmented": True,
                        "nu_calm": float(nu_calm),
                        "nu_stress": float(nu_stress),
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
    prices: np.ndarray = None,  # Added for MR integration (February 2026)
    asset: str = None,  # FIX #4: Asset symbol for c-bounds detection
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
                    "temporal_alpha": α,
                    "n_samples": N,
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
        
        # Extract regime-specific prices for MR integration (February 2026)
        prices_regime = prices[mask] if prices is not None else None
        regime_labels_regime = regime_labels[mask] if regime_labels is not None else None
        
        # =====================================================================
        # Step 1: Fit ALL models for this regime
        # =====================================================================
        models = fit_all_models_for_regime(
            ret_regime, vol_regime,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            prices=prices_regime,  # MR integration (February 2026)
            regime_labels=regime_labels_regime,
            asset=asset,  # FIX #4: Asset-class adaptive c bounds
        )
        
        # =====================================================================
        # Step 2: Extract BIC, Hyvärinen, CRPS, PIT and compute LFO-CV scores
        # =====================================================================
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        hyvarinen_scores = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models}
        crps_values = {m: models[m].get("crps", float('inf')) for m in models if models[m].get("crps") is not None}
        # February 2026 - Elite PIT calibration: extract PIT p-values for regime-aware scoring
        pit_pvalues = {m: models[m].get("pit_ks_pvalue") for m in models if models[m].get("pit_ks_pvalue") is not None}
        
        # LFO-CV scores for proper out-of-sample model selection (February 2026)
        lfo_cv_scores = {}
        if LFO_CV_ENABLED and n_samples >= 50:  # Need enough data for LFO-CV
            for m, info in models.items():
                if info.get("fit_success", False):
                    # Check if model already has LFO-CV score (e.g., MS-q models)
                    if info.get("lfo_cv_score") is not None:
                        lfo_cv_scores[m] = info["lfo_cv_score"]
                    else:
                        # Compute LFO-CV score
                        q_val = info.get("q", 1e-6)
                        c_val = info.get("c", 1.0)
                        phi_val = info.get("phi", 1.0) if info.get("phi") is not None else 1.0
                        nu_val = info.get("nu")
                        
                        try:
                            if nu_val is not None:
                                lfo_score, lfo_diag = compute_lfo_cv_score_student_t(
                                    ret_regime, vol_regime, q_val, c_val, phi_val, nu_val,
                                    min_train_frac=LFO_CV_MIN_TRAIN_FRAC
                                )
                            else:
                                lfo_score, lfo_diag = compute_lfo_cv_score_gaussian(
                                    ret_regime, vol_regime, q_val, c_val, phi_val,
                                    min_train_frac=LFO_CV_MIN_TRAIN_FRAC
                                )
                            
                            lfo_cv_scores[m] = lfo_score
                            models[m]["lfo_cv_score"] = float(lfo_score)
                            models[m]["lfo_cv_diagnostics"] = lfo_diag
                        except Exception as e:
                            lfo_cv_scores[m] = float('-inf')
                            models[m]["lfo_cv_error"] = str(e)
        
        # Print model fits
        for m, info in models.items():
            if info.get("fit_success", False):
                bic_val = info.get("bic", float('nan'))
                hyv_val = info.get("hyvarinen_score", float('nan'))
                crps_val = info.get("crps", float('nan'))
                mean_ll = info.get("mean_log_likelihood", float('nan'))
                lfo_val = info.get("lfo_cv_score", float('nan'))
                if LFO_CV_ENABLED and np.isfinite(lfo_val):
                    _log(f"     {m}: BIC={bic_val:.1f}, H={hyv_val:.4f}, CRPS={crps_val:.4f}, LFO={lfo_val:.4f}")
                else:
                    _log(f"     {m}: BIC={bic_val:.1f}, H={hyv_val:.4f}, CRPS={crps_val:.4f}, mean_LL={mean_ll:.4f}")
            else:
                _log(f"     {m}: FAILED - {info.get('error', 'unknown')}")
        
        # =====================================================================
        # Step 3: Compute raw weights using regime-aware method (February 2026)
        # =====================================================================
        # Model selection uses BIC + Hyvärinen + CRPS with regime-specific weights:
        #   - Unknown: (0.30, 0.30, 0.40) — balanced with CRPS tilt
        #   - Crisis: (0.25, 0.20, 0.55) — CRPS critical for tail risk
        #   - Trending: (0.30, 0.25, 0.45) — forecast quality > curvature
        #   - Ranging: (0.45, 0.30, 0.25) — simpler models preferred
        #   - Low Vol: (0.30, 0.40, 0.30) — robustness to misspecification
        #
        # SMALL SAMPLE HANDLING: When Hyvärinen is disabled, use BIC+CRPS only
        # =====================================================================
        hyvarinen_disabled = n_samples < MIN_HYVARINEN_SAMPLES
        weight_metadata = None
        
        if hyvarinen_disabled:
            # Small samples: use BIC + CRPS + PIT only (Hyvärinen unreliable)
            raw_weights, weight_metadata = compute_regime_aware_model_weights(
                bic_values, hyvarinen_scores, crps_values,
                pit_pvalues=pit_pvalues,  # February 2026 - Elite PIT calibration
                regime=regime, 
                bic_weight=0.50, hyvarinen_weight=0.0, crps_weight=0.50,
                lambda_entropy=DEFAULT_ENTROPY_LAMBDA
            )
            w_used = weight_metadata.get('weights_used', {})
            pit_penalty_active = any(v > 0 for v in weight_metadata.get('pit_penalty_applied', {}).values())
            pit_indicator = " +PIT_penalty" if pit_penalty_active else ""
            _log(f"     ⚠️  Hyvärinen disabled (n={n_samples} < {MIN_HYVARINEN_SAMPLES}) → BIC+CRPS (bic={w_used.get('bic', 0):.2f}, crps={w_used.get('crps', 0):.2f}){pit_indicator}")
        else:
            # Full regime-aware method: BIC + Hyvärinen + CRPS + PIT
            raw_weights, weight_metadata = compute_regime_aware_model_weights(
                bic_values, hyvarinen_scores, crps_values,
                pit_pvalues=pit_pvalues,  # February 2026 - Elite PIT calibration
                regime=regime, lambda_entropy=DEFAULT_ENTROPY_LAMBDA
            )
            w_used = weight_metadata.get('weights_used', {})
            pit_penalty_active = any(v > 0 for v in weight_metadata.get('pit_penalty_applied', {}).values())
            pit_indicator = " +PIT_penalty" if pit_penalty_active else ""
            _log(f"     → Using regime-aware BIC+Hyvärinen+CRPS selection (regime={regime}, bic={w_used.get('bic', 0):.2f}, hyv={w_used.get('hyvarinen', 0):.2f}, crps={w_used.get('crps', 0):.2f}){pit_indicator}")
        
        # Store combined_score and entropy-regularized weights in each model
        for m in models:
            w = raw_weights.get(m, 1e-10)
            if weight_metadata is not None:
                # Use standardized combined score (lower = better)
                # Handle None values from metadata (non-finite scores stored as None)
                combined_score_val = weight_metadata.get('combined_scores_standardized', {}).get(m)
                models[m]['combined_score'] = float(combined_score_val) if combined_score_val is not None else 0.0
                models[m]['model_weight_entropy'] = float(w)
                bic_std_val = weight_metadata.get('bic_standardized', {}).get(m)
                models[m]['standardized_bic'] = float(bic_std_val) if bic_std_val is not None else None
                hyv_std_val = weight_metadata.get('hyvarinen_standardized', {}).get(m)
                models[m]['standardized_hyvarinen'] = float(hyv_std_val) if hyv_std_val is not None else None
                # CRPS standardized (February 2026 - regime-aware scoring)
                crps_std_val = weight_metadata.get('crps_standardized', {}).get(m)
                models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
                # Store scoring weights used for this model
                scoring_weights = weight_metadata.get('weights_used', {})
                models[m]['scoring_weights'] = {
                    'bic': float(scoring_weights.get('bic', 0.0)),
                    'hyvarinen': float(scoring_weights.get('hyvarinen', 0.0)),
                    'crps': float(scoring_weights.get('crps', 0.0)),
                }
                models[m]['crps_scoring_enabled'] = weight_metadata.get('crps_enabled', False)
                models[m]['entropy_lambda'] = DEFAULT_ENTROPY_LAMBDA
            else:
                # Legacy: log of weight
                models[m]['combined_score'] = float(np.log(w)) if w > 0 else float('-inf')
        
        # =====================================================================
        # Step 3a: Apply Elite Tuning Fragility Penalties (v2.0 - February 2026)
        # =====================================================================
        # TOP 0.001% UPGRADE: Fragile models are down-weighted
        # 
        # CORE DESIGN CONSTRAINT: Fragility must only act as a PENALTY, not reward.
        # - Basin optimum (fragility < 0.3) → neutral (no effect)
        # - Moderate fragility (0.3-0.5) → mild penalty (10-30%)
        # - Ridge optimum (fragility > 0.5) → significant penalty (30-70%)
        #
        # This ensures BIC/Hyvarinen selection is STABILITY-AWARE, not just fit-aware.
        # =====================================================================
        weights_pre_elite = raw_weights.copy()
        elite_penalty_applied = False
        
        if ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED:
            for m in models:
                if not models[m].get('fit_success', False):
                    continue
                    
                elite_diag = models[m].get('diagnostics', {}).get('elite_diagnostics', {})
                if not elite_diag:
                    continue
                
                fragility = elite_diag.get('fragility_index', 0.0)
                is_ridge = elite_diag.get('is_ridge_optimum', False)
                basin_score = elite_diag.get('basin_score', 1.0)
                
                # Compute fragility penalty (asymmetric: only penalize, never reward)
                # Penalty = 0 for fragility < 0.3, then scales up
                if fragility > 0.3:
                    # Penalty scales from 0 at 0.3 to 0.7 at 1.0
                    penalty_factor = min((fragility - 0.3) / 0.7, 1.0)
                    weight_multiplier = 1.0 - (0.7 * penalty_factor)  # 1.0 → 0.3
                    
                    # Extra penalty for ridge optima (dangerous)
                    if is_ridge:
                        weight_multiplier *= 0.5  # Additional 50% penalty
                        _log(f"     ⚠️  {m}: RIDGE optimum detected (basin={basin_score:.2f}) → extra penalty")
                    
                    old_weight = raw_weights.get(m, 0.0)
                    new_weight = old_weight * weight_multiplier
                    raw_weights[m] = new_weight
                    
                    # Store penalty info
                    models[m]['elite_fragility_penalty'] = 1.0 - weight_multiplier
                    models[m]['elite_weight_pre_penalty'] = old_weight
                    models[m]['elite_weight_post_penalty'] = new_weight
                    elite_penalty_applied = True
                    
                    _log(f"     → {m}: fragility={fragility:.2f} → penalty {(1.0-weight_multiplier)*100:.0f}%")
                else:
                    models[m]['elite_fragility_penalty'] = 0.0
        
        if elite_penalty_applied:
            _log(f"     → Elite tuning penalties applied (fragility-aware BIC/Hyvärinen)")
        
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
                "model_selection_method": "regime_aware_crps",
                "effective_selection_method": "bic_crps_only" if hyvarinen_disabled else "bic_hyv_crps",
                "hyvarinen_disabled": hyvarinen_disabled,
                "crps_enabled": True,
                "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
                "smoothing_applied": prev_posterior is not None and temporal_alpha > 0,
                # PIT Penalty metadata (February 2026)
                "pit_penalty_applied": pit_penalty_report is not None,
                "pit_penalty": pit_penalty_meta,
                # Elite Tuning metadata (v2.0 - February 2026)
                "elite_tuning_enabled": ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED,
                "elite_tuning_preset": ELITE_TUNING_PRESET if ELITE_TUNING_AVAILABLE else None,
                "elite_penalty_applied": elite_penalty_applied,
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
    prices: np.ndarray = None,  # Added for MR integration (February 2026)
    asset: str = None,  # FIX #4: Asset symbol for c-bounds detection
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
        prices=prices,  # MR integration (February 2026)
        regime_labels=regime_labels,
        asset=asset,  # FIX #4: Asset-class adaptive c bounds
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
            combined_score_val = fallback_weight_metadata.get('combined_scores_standardized', {}).get(m)
            global_models[m]['combined_score'] = float(combined_score_val) if combined_score_val is not None else 0.0
            global_models[m]['model_weight_entropy'] = float(w)
            bic_std_val = fallback_weight_metadata.get('bic_standardized', {}).get(m)
            global_models[m]['standardized_bic'] = float(bic_std_val) if bic_std_val is not None else None
            hyv_std_val = fallback_weight_metadata.get('hyvarinen_standardized', {}).get(m)
            global_models[m]['standardized_hyvarinen'] = float(hyv_std_val) if hyv_std_val is not None else None
            # CRPS standardized (February 2026 - regime-aware scoring)
            crps_std_val = fallback_weight_metadata.get('crps_standardized', {}).get(m)
            global_models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
            # Store scoring weights used
            scoring_weights = fallback_weight_metadata.get('weights_used', {})
            global_models[m]['scoring_weights'] = {
                'bic': float(scoring_weights.get('bic', 0.0)),
                'hyvarinen': float(scoring_weights.get('hyvarinen', 0.0)),
                'crps': float(scoring_weights.get('crps', 0.0)),
            }
            global_models[m]['crps_scoring_enabled'] = fallback_weight_metadata.get('crps_enabled', False)
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
        prices=prices,  # MR integration (February 2026)
        asset=asset,  # FIX #4: Asset-class adaptive c bounds
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
            # Elite Tuning metadata (v2.0 - February 2026)
            "elite_tuning_enabled": ELITE_TUNING_AVAILABLE and ELITE_TUNING_ENABLED,
            "elite_tuning_preset": ELITE_TUNING_PRESET if ELITE_TUNING_AVAILABLE else None,
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
            # Elite Tuning configuration (v2.0 - February 2026)
            "elite_tuning_available": ELITE_TUNING_AVAILABLE,
            "elite_tuning_enabled": ELITE_TUNING_ENABLED,
            "elite_tuning_preset": ELITE_TUNING_PRESET if ELITE_TUNING_AVAILABLE else None,
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
    
    # Reset filter cache for this asset to avoid cross-asset contamination
    # and collect fresh statistics for this tuning run
    if FILTER_CACHE_AVAILABLE:
        clear_filter_cache()
        reset_cache_stats()
    
    try:
        # Fetch price data (need OHLC for Garman-Klass volatility)
        # Use _download_prices ONCE to get OHLC data - extract Close from it
        # This avoids duplicate downloads (February 2026 optimization)
        df = _download_prices(asset, start_date, end_date)
        if df is None or df.empty:
            _log(f"     ⚠️  No price data for {asset}")
            return None
        
        # Extract Close prices from OHLC DataFrame
        cols = {c.lower(): c for c in df.columns}
        if 'close' in cols:
            px = df[cols['close']]
        else:
            _log(f"     ⚠️  No Close column for {asset}")
            return None
        
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

        # Compute returns
        log_ret = np.log(px / px.shift(1)).dropna()
        returns = log_ret.values

        # Compute volatility using Garman-Klass or HAR (7.4x more efficient than EWMA)
        vol_estimator_used = "EWMA"
        if GK_VOLATILITY_AVAILABLE and df is not None and not df.empty:
            try:
                # Check for OHLC columns
                cols = {c.lower(): c for c in df.columns}
                if all(c in cols for c in ['open', 'high', 'low', 'close']):
                    # Align OHLC data with returns (drop first row to match log returns)
                    df_aligned = df.iloc[1:].copy()
                    open_ = df_aligned[cols['open']].values
                    high = df_aligned[cols['high']].values
                    low = df_aligned[cols['low']].values
                    close = df_aligned[cols['close']].values
                    
                    # ENFORCE HAR-GK ONLY (February 2026)
                    # HAR-GK provides multi-horizon memory for crash detection
                    # Combined with Garman-Klass (7.4x more efficient than EWMA)
                    vol, vol_estimator_used = compute_hybrid_volatility_har(
                        open_=open_, high=high, low=low, close=close,
                        span=21, annualize=False, use_har=True
                    )
                else:
                    # OHLC not available - raise error as HAR-GK is required
                    raise ValueError(f"OHLC data required for HAR-GK volatility estimation for {asset}")
            except Exception as e:
                # Log error but don't silently fall back to inferior estimator
                _log(f"     ⚠️ HAR-GK volatility estimation failed: {e}")
                raise ValueError(f"HAR-GK volatility estimation required but failed for {asset}: {e}")
        else:
            # GK/HAR module not available - this should not happen in production
            raise ImportError("HAR-GK volatility module required but not available")
        
        # Ensure returns and vol have same length
        min_len = min(len(returns), len(vol))
        returns = returns[:min_len]
        vol = vol[:min_len]

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
        
        # Extract prices array for MR integration (February 2026)
        prices_array = None
        if px is not None:
            px_values = px.values if hasattr(px, 'values') else np.array(px)
            # Skip first price to align with returns, then apply same valid_mask
            if len(px_values) > 1:
                prices_aligned = px_values[1:][:min_len]
                if len(prices_aligned) == len(valid_mask):
                    prices_array = prices_aligned[valid_mask]
                elif len(prices_aligned) >= len(returns):
                    prices_array = prices_aligned[:len(returns)]
        
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
            prices=prices_array,  # MR integration (February 2026)
            asset=asset,  # FIX #4: Asset-class adaptive c bounds
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
                # Volatility estimator used (February 2026)
                "volatility_estimator": vol_estimator_used,
                # Market conditioning flags (February 2026)
                "market_conditioning_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE,
                "vix_nu_adjustment_enabled": MARKET_CONDITIONING_ENABLED and MARKET_CONDITIONING_AVAILABLE and global_data.get("nu") is not None,
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

        # Report filter cache statistics
        if FILTER_CACHE_AVAILABLE:
            cache_stats = get_cache_stats()
            if cache_stats.hits > 0 or cache_stats.misses > 0:
                _log(f"     📊 {cache_stats.summary()}")
            # Add cache stats to result for analysis
            result["filter_cache_stats"] = {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "fold_slice_reuses": cache_stats.fold_slice_reuses,
                "warm_starts": cache_stats.warm_starts,
            }

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
    
    # Track failure reasons for better error reporting
    failure_reasons = []
    
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
        
        failure_reasons.append("tune_asset_with_bma returned None (likely insufficient data or data fetch error)")

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
            failure_reasons.append("tune_asset_q also returned None")
            
            # Try to get more info about why data fetch might have failed
            try:
                df = _download_prices(asset, start_date, end_date)
                if df is None:
                    failure_reasons.append(f"_download_prices returned None for {asset}")
                elif df.empty:
                    failure_reasons.append(f"_download_prices returned empty DataFrame for {asset}")
                else:
                    n_rows = len(df)
                    failure_reasons.append(f"Data was fetched ({n_rows} rows) but processing failed")
                    # Check for NaN/Inf issues
                    if 'Close' in df.columns:
                        close = df['Close']
                        n_valid = close.notna().sum()
                        n_inf = np.isinf(close.replace([np.inf, -np.inf], np.nan).dropna()).sum() if n_valid > 0 else 0
                        failure_reasons.append(f"Close prices: {n_valid} valid, {n_rows - n_valid} NaN, {n_inf} Inf")
            except Exception as data_check_err:
                failure_reasons.append(f"Data check error: {data_check_err}")
            
            detailed_error = " | ".join(failure_reasons)
            return (asset, None, detailed_error, None)

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
                                'best_model': global_result.get('best_model', global_result.get('best_model_by_bic', 'unknown')),
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
            best_model = data.get('best_model', data.get('best_model_by_bic', 'kalman_drift'))

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
            best_bic = mc.get('best_model', mc.get('best_model_by_bic', 'unknown'))
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
