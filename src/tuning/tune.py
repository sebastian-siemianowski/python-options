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
                              phi_skew_t_nu_{ν}_gamma_{γ})
    θ_{r,m}    = parameters of model m in regime r
    p(m | r)   = posterior probability of model m in regime r

-------------------------------------------------------------------------------
WHAT THIS FILE DOES

For EACH regime r:

    1. Fits ALL candidate model classes m independently:
       - kalman_gaussian_unified:           (8+ params) [ENABLED - unified Gaussian with momentum + GAS-Q]
       - kalman_phi_gaussian_unified:       (9+ params) [ENABLED - unified phi-Gaussian with momentum + GAS-Q]
       - phi_student_t_nu_4:    q, c, φ        (3 params, ν=4 FIXED)
       - phi_student_t_nu_8:    q, c, φ        (3 params, ν=8 FIXED)
       - phi_student_t_nu_20:   q, c, φ        (3 params, ν=20 FIXED)
       NOTE: Momentum augmentation is now an INTERNAL pipeline step within each Student-t model.
             Activated only if it improves CRPS. No separate _momentum model names.
       - phi_skew_t_nu_{ν}_gamma_{γ}: q, c, φ  (3 params, ν and γ FIXED)

    NOTE: Legacy kalman_gaussian_momentum retired (Feb 2026). Unified Gaussian models subsume their functionality.
    Unified models use Stage 1.5 momentum + Stage 4.5 GAS-Q internally.

    NOTE: Student-t and Skew-t use DISCRETE grids (not continuous optimization).
    Each parameter combination is treated as a separate sub-model in BMA.

    SKEW-T ADDITION (Proposal 5 — φ-Skew-t with BMA):
    The Fernández-Steel skew-t distribution captures asymmetric return distributions:
       - γ = 1: Symmetric (reduces to Student-t)
       - γ < 1: Left-skewed (heavier left tail) — crash risk
       - γ > 1: Right-skewed (heavier right tail) — euphoria risk

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
from scipy.stats import norm, t as student_t
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
from ingestion.adaptive_quality import adaptive_data_quality


def _fast_ks_uniform(pit_values):
    """
    Inline KS test against Uniform(0,1) — replaces scipy.stats.kstest.
    Returns (statistic, p_value) using Kolmogorov asymptotic approximation.
    """
    n = len(pit_values)
    if n < 2:
        return 1.0, 0.0
    sorted_pit = np.sort(pit_values)
    ecdf = np.arange(1, n + 1) / n
    D_plus = float(np.max(ecdf - sorted_pit))
    D_minus = float(np.max(sorted_pit - np.arange(0, n) / n))
    D = max(D_plus, D_minus)
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D
    if lam < 0.001:
        p = 1.0
    elif lam > 3.0:
        p = 0.0
    else:
        # 4-term alternating series: P(K>λ) = 2·Σ (-1)^{k+1} exp(-2k²λ²)
        # Single-term (2·exp(-2λ²)) overestimates and saturates at 1.0 for λ<0.59
        lam2 = lam * lam
        p = 2.0 * (math.exp(-2.0 * lam2)
                   - math.exp(-8.0 * lam2)
                   + math.exp(-18.0 * lam2)
                   - math.exp(-32.0 * lam2))
        if p < 0.0:
            p = 0.0
    return D, p



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
#   - Gaussian:           μ, σ (baseline)
#   - Symmetric Student-t: μ, σ, ν (fat tails)
#   - φ-Skew-t:           μ, σ, ν, γ (fat tails + asymmetry, Fernández-Steel)
#   - Hansen Skew-t:      μ, σ, ν, λ (fat tails + asymmetry, regime-conditional)
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

    Two-pass approach:
      Pass 1: Collect all asset phi values and sample counts.
      Pass 2: Compute population prior (median, MAD) and shrink each asset.

    Hierarchical precision-weighted shrinkage formula:
      tau_asset = n_regime_samples (proxy for estimation precision)
      tau_pop   = 1 / phi_pop_std^2
      phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_pop_median) / (tau_asset + tau_pop)

    Assets with many regime samples (high tau_asset) barely shrink.
    Assets with few samples (low tau_asset) shrink strongly toward population.

    Modifies cache in-place and returns it.
    """
    # Pass 1: collect phi values across all assets
    phi_entries = []  # (asset, phi_mle, n_samples)
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
        phi_entries.append((asset, float(phi_val), n_samples))

    if len(phi_entries) < PHI_POOL_MIN_ASSETS:
        # Not enough assets to form cross-asset prior -- use defaults
        phi_pop_median = DEFAULT_PHI_PRIOR
        phi_pop_std = DEFAULT_PHI_PRIOR_STD
    else:
        # Robust location and scale from cross-asset phi distribution
        all_phi = np.array([e[1] for e in phi_entries])
        phi_pop_median = float(np.median(all_phi))
        # MAD-based robust std estimate (1.4826 converts MAD to std for normal)
        mad = float(np.median(np.abs(all_phi - phi_pop_median)))
        phi_pop_std = max(mad * 1.4826, 0.05)  # Floor at 0.05 to prevent over-shrinkage

    tau_pop = 1.0 / (phi_pop_std ** 2)

    # Pass 2: shrink each asset's phi toward population
    n_shrunk = 0
    for asset, phi_mle, n_samples in phi_entries:
        # Precision proportional to sample count
        # Fisher information for AR(1) coefficient scales as O(n)
        tau_asset = float(n_samples)
        tau_total = tau_asset + tau_pop

        phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_pop_median) / tau_total

        # Clamp to valid range
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

    # Store population prior in metadata
    _pool_meta = {
        "phi_population_median": phi_pop_median,
        "phi_population_std": phi_pop_std,
        "n_assets_pooled": len(phi_entries),
        "n_assets_shrunk": n_shrunk,
    }
    for asset in cache:
        if isinstance(cache[asset], dict):
            ht = cache[asset].setdefault("hierarchical_tuning", {})
            ht["phi_prior"] = _pool_meta

    return cache


# =========================================================================
# Story 2.2: EMOS Calibration from Walk-Forward Errors
# =========================================================================
# Train per-horizon EMOS parameters via CRPS minimization on walk-forward
# (forecast, realized) pairs.
#
# EMOS (Gneiting 2005) applies:
#   mu_corrected  = a + b * mu_raw
#   sig_corrected = c + d * sig_raw   (c,d >= 0 enforced)
#
# The CRPS for a Normal(mu, sig) distribution is:
#   CRPS(y; mu, sig) = sig * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
# where z = (y - mu) / sig.
# =========================================================================

EMOS_TRAIN_FRAC = 0.70   # Use first 70% for training, last 30% for validation
EMOS_IDENTITY = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}


def _crps_normal_single(y: float, mu: float, sig: float) -> float:
    """CRPS for a single observation against Normal(mu, sig)."""
    if sig <= 0:
        return abs(y - mu)
    from scipy.stats import norm
    z = (y - mu) / sig
    return sig * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


def _emos_crps_objective(params: np.ndarray, forecasts: np.ndarray,
                         sigmas: np.ndarray, realized: np.ndarray) -> float:
    """Mean CRPS for EMOS-corrected Normal distribution."""
    a, b, c, d = params
    mu_cor = a + b * forecasts
    sig_cor = np.abs(c + d * sigmas)
    sig_cor = np.maximum(sig_cor, 1e-8)
    n = len(realized)
    total = 0.0
    for i in range(n):
        total += _crps_normal_single(realized[i], mu_cor[i], sig_cor[i])
    return total / n


def train_emos_parameters(
    wf_records: list,
    horizons: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Train per-horizon EMOS parameters via CRPS minimization.

    Using walk-forward (forecast, realized) pairs, fits the affine EMOS
    correction (a + b*mu, |c + d*sig|) that minimises the CRPS of the
    corrected Normal predictive distribution.

    Training uses the first EMOS_TRAIN_FRAC of records, validation uses
    the remainder.

    Args:
        wf_records: List of WalkForwardRecord (from Story 2.1).
        horizons: Horizons to calibrate (default [1,3,7,21,63]).

    Returns:
        {horizon: {'a', 'b', 'c', 'd', 'crps_train', 'crps_val',
                   'crps_uncorrected', 'crps_improvement_pct', 'n_train', 'n_val'}}
    """
    from scipy.optimize import minimize as sp_minimize

    if horizons is None:
        horizons = [1, 3, 7, 21, 63]

    results: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        h_recs = [r for r in wf_records if r.horizon == H]
        if len(h_recs) < 20:
            results[H] = dict(EMOS_IDENTITY)
            results[H].update({"n_train": 0, "n_val": 0,
                               "crps_train": float("nan"),
                               "crps_val": float("nan"),
                               "crps_uncorrected": float("nan"),
                               "crps_improvement_pct": 0.0})
            continue

        # Sort by date_idx to respect temporal ordering
        h_recs.sort(key=lambda r: r.date_idx)

        # Train/validation split
        n_train = max(10, int(len(h_recs) * EMOS_TRAIN_FRAC))
        train = h_recs[:n_train]
        val = h_recs[n_train:]

        f_train = np.array([r.forecast_ret for r in train])
        s_train = np.array([r.forecast_sig for r in train])
        y_train = np.array([r.realized_ret for r in train])

        # Optimize EMOS params on training set
        x0 = np.array([0.0, 1.0, 0.0, 1.0])
        try:
            opt = sp_minimize(
                _emos_crps_objective, x0, args=(f_train, s_train, y_train),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
            )
            a, b, c, d = opt.x
        except Exception:
            a, b, c, d = 0.0, 1.0, 0.0, 1.0

        # Ensure sigma correction is non-negative
        if c + d * np.mean(s_train) < 0:
            c, d = 0.0, 1.0

        crps_train = _emos_crps_objective(np.array([a, b, c, d]), f_train, s_train, y_train)
        crps_uncorrected = _emos_crps_objective(x0, f_train, s_train, y_train)

        # Validation
        crps_val = float("nan")
        n_val = len(val)
        if n_val > 0:
            f_val = np.array([r.forecast_ret for r in val])
            s_val = np.array([r.forecast_sig for r in val])
            y_val = np.array([r.realized_ret for r in val])
            crps_val = _emos_crps_objective(np.array([a, b, c, d]), f_val, s_val, y_val)

        improvement = 0.0
        if crps_uncorrected > 0:
            improvement = (crps_uncorrected - crps_train) / crps_uncorrected * 100.0

        results[H] = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "crps_train": float(crps_train),
            "crps_val": float(crps_val),
            "crps_uncorrected": float(crps_uncorrected),
            "crps_improvement_pct": float(improvement),
            "n_train": n_train,
            "n_val": n_val,
        }

    return results


# =========================================================================
# Story 2.3: Realized Volatility Feedback for Sigma Calibration
# =========================================================================
# Compute per-horizon vol calibration ratio from walk-forward data.
#
# vol_ratio_H = realized_std(returns_H) / mean(forecast_sig_H)
#
# If vol_ratio > 1: system is overconfident (underestimates uncertainty)
# If vol_ratio < 1: system is too cautious (overestimates uncertainty)
#
# The ratio is clamped to [0.5, 2.0] for stability.
# =========================================================================

VOL_RATIO_CLAMP_LOW = 0.5
VOL_RATIO_CLAMP_HIGH = 2.0


def compute_vol_calibration_ratios(
    wf_records: list,
    horizons: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute per-horizon volatility calibration ratio from walk-forward data.

    vol_ratio_H = std(realized_ret_H) / mean(forecast_sig_H)

    If vol_ratio > 1.0 the system is overconfident.
    If vol_ratio < 1.0 the system is too cautious.

    Args:
        wf_records: List of WalkForwardRecord from Story 2.1.
        horizons: Horizons (default [1,3,7,21,63]).

    Returns:
        {horizon: vol_ratio} clamped to [0.5, 2.0].
    """
    if horizons is None:
        horizons = [1, 3, 7, 21, 63]

    ratios: Dict[int, float] = {}
    for H in horizons:
        h_recs = [r for r in wf_records if r.horizon == H]
        if len(h_recs) < 10:
            ratios[H] = 1.0
            continue
        realized = np.array([r.realized_ret for r in h_recs])
        forecast_sig = np.array([r.forecast_sig for r in h_recs])
        realized_vol = float(np.std(realized, ddof=1))
        mean_sig = float(np.mean(forecast_sig))
        if mean_sig <= 0:
            ratios[H] = 1.0
            continue
        ratio = realized_vol / mean_sig
        ratios[H] = float(np.clip(ratio, VOL_RATIO_CLAMP_LOW, VOL_RATIO_CLAMP_HIGH))
    return ratios


# ─── Story 2.4: Directional Information Gain (DIG) for BMA Weights ──────
DIG_BASELINE = 0.5          # random walk hit rate
DIG_W_START = 0.10          # DIG weight when data is sparse
DIG_W_MAX = 0.25            # DIG weight at full data accumulation
DIG_MIN_RECORDS = 30        # minimum records per model for meaningful DIG
DIG_FULL_DATA_THRESHOLD = 200  # records at which w_dig reaches DIG_W_MAX


def compute_dig_per_model(
    wf_records: list,
    horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Directional Information Gain (DIG) per model from walk-forward data.

    DIG_m = hit_rate_m - 0.5

    Where hit_rate_m = P(sign(forecast_ret) == sign(realized_ret)).
    Positive DIG means directional edge over random. Negative DIG means
    worse than random (penalized in BMA weighting).

    Args:
        wf_records: List of WalkForwardRecord from Story 2.1.
        horizons: Horizons to include (default all).

    Returns:
        {model_or_horizon_key: DIG}. Uses "ensemble" key for the
        overall ensemble, and "H={h}" for per-horizon.
    """
    if horizons is not None:
        recs = [r for r in wf_records if r.horizon in horizons]
    else:
        recs = list(wf_records)

    if len(recs) < DIG_MIN_RECORDS:
        return {"ensemble": 0.0}

    hit_count = sum(1 for r in recs if r.hit)
    hit_rate = hit_count / len(recs)
    result: Dict[str, float] = {"ensemble": hit_rate - DIG_BASELINE}

    # Per-horizon DIG
    horizon_set = set(r.horizon for r in recs)
    for h in sorted(horizon_set):
        h_recs = [r for r in recs if r.horizon == h]
        if len(h_recs) >= DIG_MIN_RECORDS:
            h_hits = sum(1 for r in h_recs if r.hit)
            result[f"H={h}"] = h_hits / len(h_recs) - DIG_BASELINE
    return result


def compute_dig_weight(n_records: int) -> float:
    """
    Compute adaptive DIG weight that grows from DIG_W_START to DIG_W_MAX
    as walk-forward data accumulates.

    Linear ramp: w_dig = DIG_W_START + (DIG_W_MAX - DIG_W_START) * frac
    where frac = min(1, n_records / DIG_FULL_DATA_THRESHOLD).

    Args:
        n_records: Total walk-forward records available.

    Returns:
        DIG weight in [DIG_W_START, DIG_W_MAX].
    """
    if n_records <= 0:
        return DIG_W_START
    frac = min(1.0, n_records / DIG_FULL_DATA_THRESHOLD)
    return DIG_W_START + (DIG_W_MAX - DIG_W_START) * frac


def adjust_bma_weights_with_dig(
    raw_weights: Dict[str, float],
    dig_values: Dict[str, float],
    n_records: int,
) -> Dict[str, float]:
    """
    Adjust BMA weights using Directional Information Gain (DIG).

    Models with DIG > 0 get a proportional boost. Models with DIG < 0
    get penalized. The adjustment is multiplicative:

    adjusted_w_m = raw_w_m * (1 + w_dig * DIG_m_standardized)

    Where DIG_m_standardized is robust z-score across the ensemble.

    Args:
        raw_weights: BMA weights from the 6-component scoring system.
        dig_values: Per-model DIG values (model_name -> DIG).
        n_records: Walk-forward record count (for adaptive weighting).

    Returns:
        Adjusted and renormalized BMA weights.
    """
    if not dig_values or n_records < DIG_MIN_RECORDS:
        return dict(raw_weights)

    w_dig = compute_dig_weight(n_records)

    # Robust standardize DIG across models present in raw_weights
    model_digs = {}
    for m in raw_weights:
        if m in dig_values:
            model_digs[m] = dig_values[m]
    if not model_digs:
        return dict(raw_weights)

    vals = np.array(list(model_digs.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad < 1e-10:
        mad = float(np.std(vals)) if np.std(vals) > 1e-10 else 1.0

    adjusted = {}
    for m, w in raw_weights.items():
        if m in model_digs:
            dig_std = (model_digs[m] - med) / mad
            dig_std = float(np.clip(dig_std, -3.0, 3.0))  # winsorize
            multiplier = 1.0 + w_dig * dig_std
            multiplier = max(0.1, multiplier)  # floor to prevent zeroing
            adjusted[m] = w * multiplier
        else:
            adjusted[m] = w

    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {m: v / total for m, v in adjusted.items()}
    return adjusted


# ─── Story 2.6: Automated Calibration Pipeline ──────────────────────────
CALIBRATION_REPORT_DIR = "src/data/calibration"
CALIBRATION_REPORT_FILE = "calibration_report.json"
CALIBRATION_DEFAULT_START = "2024-01-01"


def run_calibration_pipeline(
    assets: List[str],
    cache: Dict[str, Dict],
    cache_json: str = "src/data/tune",
    start_date: str = CALIBRATION_DEFAULT_START,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run the full calibration pipeline for a list of assets.

    For each asset:
        1. Run walk-forward backtest (Story 2.1)
        2. Train EMOS parameters (Story 2.2)
        3. Compute volatility calibration ratios (Story 2.3)
        4. Compute DIG (Story 2.4)
        5. Compute P&L attribution (Story 2.5)
        6. Update tune cache with calibration data

    Args:
        assets: List of asset symbols.
        cache: Existing tune cache (modified in-place).
        cache_json: Path to cache directory.
        start_date: Walk-forward start date.
        horizons: Horizons for calibration (default [1,3,7,21,63]).

    Returns:
        Calibration report dict with per-asset results and summary.
    """
    from decision.signals import (
        run_walk_forward_backtest, compute_pnl_attribution,
        WF_HORIZONS,
    )

    if horizons is None:
        horizons = WF_HORIZONS

    report: Dict[str, Any] = {
        "assets": {},
        "summary": {},
    }

    n_nontrivial_emos = 0
    total_assets = len(assets)

    for i, asset in enumerate(assets, 1):
        print(f"\n[Calibrate {i}/{total_assets}] {asset}")
        asset_report: Dict[str, Any] = {"asset": asset, "status": "pending"}

        try:
            # Step 1: Walk-forward
            wf = run_walk_forward_backtest(
                asset, start_date=start_date, horizons=horizons,
            )
            n_records = len(wf.records)
            asset_report["n_records"] = n_records
            asset_report["hit_rate"] = dict(wf.hit_rate)

            if n_records < 10:
                asset_report["status"] = "skipped_insufficient_data"
                report["assets"][asset] = asset_report
                continue

            # Step 2: EMOS
            emos_params = train_emos_parameters(wf.records, horizons=horizons)
            asset_report["emos_params"] = {}
            for h, ep in emos_params.items():
                is_identity = (abs(ep.get("a", 0)) < 0.01
                               and abs(ep.get("b", 1) - 1.0) < 0.01
                               and abs(ep.get("c", 0)) < 0.01
                               and abs(ep.get("d", 1) - 1.0) < 0.01)
                if not is_identity:
                    n_nontrivial_emos += 1
                asset_report["emos_params"][h] = {
                    "a": ep.get("a", 0), "b": ep.get("b", 1),
                    "c": ep.get("c", 0), "d": ep.get("d", 1),
                    "crps_train": ep.get("crps_train", None),
                    "crps_val": ep.get("crps_val", None),
                    "is_identity": is_identity,
                }

            # Step 3: Vol ratios
            vol_ratios = compute_vol_calibration_ratios(wf.records, horizons=horizons)
            asset_report["vol_ratios"] = {str(k): v for k, v in vol_ratios.items()}

            # Step 4: DIG
            dig = compute_dig_per_model(wf.records, horizons=horizons)
            asset_report["dig"] = dig

            # Step 5: P&L attribution
            pnl = compute_pnl_attribution(wf.records, horizons=horizons)
            asset_report["pnl_attribution"] = {
                str(k): v for k, v in pnl.items()
            }

            # Step 6: Update cache
            if asset not in cache:
                cache[asset] = {}
            if "calibration" not in cache[asset]:
                cache[asset]["calibration"] = {}
            cache[asset]["calibration"]["emos"] = {
                str(h): {k: v for k, v in ep.items()
                         if k in ("a", "b", "c", "d")}
                for h, ep in emos_params.items()
            }
            cache[asset]["calibration"]["vol_ratios"] = {
                str(k): v for k, v in vol_ratios.items()
            }
            cache[asset]["calibration"]["dig"] = dig

            asset_report["status"] = "success"

        except Exception as e:
            asset_report["status"] = "failed"
            asset_report["error"] = str(e)

        report["assets"][asset] = asset_report

    # Summary
    n_success = sum(1 for r in report["assets"].values() if r["status"] == "success")
    n_failed = sum(1 for r in report["assets"].values() if r["status"] == "failed")
    n_skipped = sum(1 for r in report["assets"].values() if r["status"].startswith("skipped"))
    report["summary"] = {
        "total_assets": total_assets,
        "success": n_success,
        "failed": n_failed,
        "skipped": n_skipped,
        "n_nontrivial_emos": n_nontrivial_emos,
    }

    return report


def save_calibration_report(report: Dict[str, Any],
                            output_dir: str = CALIBRATION_REPORT_DIR) -> str:
    """Save calibration report to JSON. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CALIBRATION_REPORT_FILE)

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, cls=_Encoder)
    return path


def apply_regime_q_floor(
    models: Dict[str, Dict],
    regime: int,
    returns: np.ndarray,
    vol: np.ndarray,
    asset_vol: float = 0.0,
) -> Tuple[int, int]:
    """
    Apply regime-conditional q floor to all fitted models in-place.

    For each model where q_mle < q_floor:
      1. Store original q as q_mle_original
      2. Set q = q_floor
      3. Re-run filter to get updated log-likelihood
      4. Recompute BIC with penalty term
      5. Set q_floor_applied = True

    This preserves the optimiser's choice of c and phi while only lifting q.
    The BIC adjustment ensures model selection is not distorted by the floor.

    Story 1.7: When asset_vol > 0, uses vol-proportional q floor via
    compute_vol_proportional_q_floor().  Falls back to Q_FLOOR_BY_REGIME
    when asset_vol is not provided.

    Args:
        models: Dict of model_name -> model_dict (modified in-place)
        regime: Regime index (0-4) for floor lookup
        returns: Regime-specific returns (for re-filtering)
        vol: Regime-specific volatility (for re-filtering)
        asset_vol: Annualized asset volatility (for vol-proportional floor)

    Returns:
        (n_floored, n_total): count of models where floor was applied
    """
    if asset_vol > 0:
        q_floor = compute_vol_proportional_q_floor(regime, asset_vol)
    else:
        q_floor = Q_FLOOR_BY_REGIME.get(regime, 0.0)
    if q_floor <= 0:
        return 0, len(models)

    n_floored = 0
    n_total = 0
    n_obs = len(returns)

    for model_name, info in models.items():
        if not info.get("fit_success", False):
            continue
        n_total += 1

        q_mle = info.get("q")
        if q_mle is None or q_mle >= q_floor:
            info["q_floor_applied"] = False
            continue

        # Floor binds: override q
        n_floored += 1
        info["q_mle_original"] = float(q_mle)
        info["q"] = float(q_floor)
        info["q_floor_applied"] = True
        info["q_floor_regime"] = int(regime)
        info["q_floor_value"] = float(q_floor)

        # Re-run filter with floored q to get updated log-likelihood
        c_val = info.get("c", 1.0)
        phi_val = info.get("phi")
        nu_val = info.get("nu")

        try:
            if nu_val is not None and phi_val is not None:
                # Student-t model
                _, _, ll_new = PhiStudentTDriftModel.filter_phi(
                    returns, vol, q_floor, c_val, phi_val, nu_val
                )
            elif phi_val is not None:
                # Phi-Gaussian model
                _, _, ll_new = PhiGaussianDriftModel.filter_phi(
                    returns, vol, q_floor, c_val, phi_val
                )
            else:
                # Gaussian model
                _, _, ll_new = GaussianDriftModel.filter(
                    returns, vol, q_floor, c_val
                )

            # Update log-likelihood
            old_ll = info.get("log_likelihood", ll_new)
            info["log_likelihood"] = float(ll_new)
            info["mean_log_likelihood"] = float(ll_new / max(n_obs, 1))

            # Recompute BIC with penalty for floor deviation
            n_params = info.get("n_params", 3)
            from calibration.model_selection import compute_bic as _compute_bic
            base_bic = _compute_bic(ll_new, n_params, n_obs)

            # BIC penalty: penalise for deviating from MLE-optimal q
            # log(q_floor / q_mle) is always positive when floor binds
            bic_penalty = Q_FLOOR_BIC_LAMBDA * math.log(q_floor / max(q_mle, 1e-15))
            info["bic"] = float(base_bic + bic_penalty)
            info["bic_floor_penalty"] = float(bic_penalty)

        except Exception as e:
            # If re-filtering fails, keep original BIC but still apply floor
            info["q_floor_refilter_error"] = str(e)

    return n_floored, n_total


# =============================================================================
# REGIME CLASSIFICATION FUNCTION
# Story 4.2: Imported from shared module models.regime
# =============================================================================
from models.regime import assign_regime_labels


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
        
        # Adaptive data quality filter (February 2026)
        # Detects phantom/synthetic data via Volume analysis
        df, _dq_report = adaptive_data_quality(df, asset=asset, verbose=(not _is_quiet()))
        if _dq_report.get('rows_purged_leading', 0) > 0 or _dq_report.get('window_applied', False):
            _log(f"     🔬  Data quality: {_dq_report['rows_original']} → {_dq_report['rows_final']} rows")
        
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
        _volume_arr = None  # For Volume-based stale filter
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
                    
                    # Extract Volume for stale-price detection (February 2026)
                    _vol_col = cols.get('volume')
                    if _vol_col is not None:
                        _volume_arr = df_aligned[_vol_col].values
                    
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

        # Remove NaN/Inf and stale-price observations (zero-return days)
        # Stale days (O=H=L=C, vol=0) produce degenerate GK variance ≈ 1e-12
        # and contaminate model parameters. Threshold 1e-10 is well below any
        # genuine trade return but catches exact zeros and float near-zeros.
        # Also filter Volume=0 phantom quotes (February 2026) — catches
        # illiquid OTC assets (GPUS) where prices move without genuine trades.
        _STALE_RETURN_THRESHOLD = 1e-10
        valid_mask = (np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
                      & (np.abs(returns) > _STALE_RETURN_THRESHOLD))
        # Add Volume >= 100 filter if Volume data available
        # Skip for FX pairs (=X) and indices (^) — Yahoo reports Volume=0
        _MIN_GENUINE_VOLUME = 100  # Floor of genuine price discovery
        _skip_vol = (asset.endswith('=X') or asset.startswith('^')) if asset else False
        if _volume_arr is not None and not _skip_vol:
            _vol_aligned = _volume_arr[:min_len]
            _vol_mask = _vol_aligned >= _MIN_GENUINE_VOLUME
            n_zero_vol = int(np.sum(~_vol_mask & valid_mask))
            if n_zero_vol > 0:
                _log(f"     🧹  Filtered {n_zero_vol} additional low-volume phantom rows (Volume<{_MIN_GENUINE_VOLUME})")
            valid_mask = valid_mask & _vol_mask
        n_stale = int(np.sum(np.abs(returns) <= _STALE_RETURN_THRESHOLD))
        if n_stale > 0:
            _log(f"     🧹  Filtered {n_stale}/{len(returns)} stale-price rows ({100*n_stale/len(returns):.1f}%)")
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
        
        # Compute model weights using elite CRPS-dominated scoring (February 2026)
        # Only include successfully fitted models with finite metrics
        bic_values = {m: models[m].get("bic", float('inf')) for m in models if models[m].get("fit_success", False)}
        hyvarinen_values = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models if models[m].get("fit_success", False)}
        crps_values = {m: models[m]["crps"] for m in models
                       if models[m].get("fit_success", False)
                       and models[m].get("crps") is not None
                       and np.isfinite(models[m]["crps"])}
        pit_pvalues = {m: models[m]["pit_ks_pvalue"] for m in models
                       if models[m].get("fit_success", False)
                       and models[m].get("pit_ks_pvalue") is not None}
        berk_pvalues = {m: models[m]["berkowitz_pvalue"] for m in models
                        if models[m].get("fit_success", False)
                        and models[m].get("berkowitz_pvalue") is not None}
        mad_values = {m: models[m]["histogram_mad"] for m in models
                      if models[m].get("fit_success", False)
                      and models[m].get("histogram_mad") is not None}
        berk_lr_values = {m: models[m]["berkowitz_lr"] for m in models
                          if models[m].get("fit_success", False)
                          and models[m].get("berkowitz_lr") is not None}
        pit_count_values = {m: models[m]["pit_count"] for m in models
                            if models[m].get("fit_success", False)
                            and models[m].get("pit_count") is not None}
        ad_pvalues_global = {m: models[m]["ad_pvalue"] for m in models
                             if models[m].get("fit_success", False)
                             and models[m].get("ad_pvalue") is not None
                             and np.isfinite(models[m]["ad_pvalue"])}
        
        # Use elite CRPS-dominated scoring with PIT/Berk/MAD penalties
        if crps_values and CRPS_SCORING_ENABLED:
            model_weights, weight_meta = compute_regime_aware_model_weights(
                bic_values, hyvarinen_values, crps_values,
                pit_pvalues=pit_pvalues, berk_pvalues=berk_pvalues,
                berkowitz_lr_stats=berk_lr_values, pit_counts=pit_count_values,
                mad_values=mad_values, ad_pvalues=ad_pvalues_global, regime=None
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
        
        # Find best model by WEIGHT (after calibration veto gate)
        # The veto gate forces catastrophically miscalibrated models (PIT<0.01 or
        # Berk<0.01) to floor weight, redistributing to well-calibrated models.
        # Selecting by weight ensures the winner is always well-calibrated.
        best_model = max(
            ((m, w) for m, w in model_weights.items() if w is not None),
            key=lambda x: x[1]
        )[0]
        best_params = models[best_model]
        
        # Gate external augmentation layers for unified models (they already have internal calibration)
        is_unified_winner = "unified" in best_model.lower() if best_model else False

        # GMM: Removed (empirically — bimodality hypothesis consistently rejected)
        # Hansen Skew-T: Now handled internally by Stage 7.5 / Stage U-H
        hansen_skew_t_result = None
        hansen_skew_t_diagnostics = None
        hansen_comparison = None
        
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
        
        if EVT_AVAILABLE and not is_unified_winner:
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
        # CONTAMINATED STUDENT-T: Now handled internally by Stage 7.6 / Stage U-C
        # Global fit block removed (was dead code under UNIFIED_STUDENT_T_ONLY)
        # =====================================================================
        cst_result = None
        cst_diagnostics = None
        cst_comparison = None
        
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
            "best_model": best_model,  # Selected by max weight (after calibration veto gate)
            # Unified Student-t specific parameters (February 2026 - Elite Architecture)
            "unified_model": best_params.get("unified_model", False),
            "gaussian_unified": best_params.get("gaussian_unified", False),
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
            # GMM parameters — removed (bimodality hypothesis rejected)
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
# Story 3.1: Content-Based Change Detection for Incremental Tuning
# =============================================================================
# Uses hash of last 20 rows of price CSV to detect new data.
# NOT based on mtime (unreliable across git checkout, rsync, backup restore).
# =============================================================================

def compute_price_data_hash(symbol: str, prices_dir: str = None) -> Optional[str]:
    """
    Story 3.1: Compute content-based hash of last 20 rows of price CSV.
    
    Uses hashlib.sha256 on the raw bytes of the last 20 lines.
    Returns hex digest or None if file not found.
    """
    import hashlib
    
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
    if not os.path.exists(price_file):
        return None
    
    try:
        with open(price_file, 'rb') as f:
            lines = f.readlines()
        # Hash last 20 lines (or all if fewer)
        tail = lines[-20:] if len(lines) > 20 else lines
        return hashlib.sha256(b''.join(tail)).hexdigest()
    except Exception:
        return None


def get_last_price_date(symbol: str, prices_dir: str = None) -> Optional[str]:
    """
    Story 3.1: Get the date of the last price row.
    """
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
    if not os.path.exists(price_file):
        return None
    
    try:
        import csv
        with open(price_file, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        # Last line, first column (date)
        last_line = lines[-1].strip()
        if last_line:
            return last_line.split(',')[0]
    except Exception:
        pass
    return None


def needs_retune(symbol: str, cached_params: Optional[dict], prices_dir: str = None) -> bool:
    """
    Story 3.1: Check if an asset needs re-tuning based on content hash.
    
    Returns True if:
      - No cached params exist
      - Cached params don't have a price_data_hash
      - Price data hash has changed since last tune
    """
    if cached_params is None:
        return True
    
    stored_hash = None
    if isinstance(cached_params, dict):
        global_block = cached_params.get("global", cached_params)
        stored_hash = global_block.get("price_data_hash")
    
    if stored_hash is None:
        return True
    
    current_hash = compute_price_data_hash(symbol, prices_dir)
    if current_hash is None:
        return False  # No price file -> can't tune
    
    return current_hash != stored_hash


def stamp_tune_result(result: dict, symbol: str, prices_dir: str = None) -> dict:
    """
    Story 3.1: Add price_data_hash and last_price_date to tune result.
    """
    if result is None:
        return result
    
    current_hash = compute_price_data_hash(symbol, prices_dir)
    last_date = get_last_price_date(symbol, prices_dir)
    
    global_block = result.get("global", result)
    if current_hash:
        global_block["price_data_hash"] = current_hash
    if last_date:
        global_block["last_price_date"] = last_date
    
    return result


# =============================================================================
# Story 3.2: Parallel Tuning Optimization
# =============================================================================

def estimate_tuning_complexity(symbol: str, prices_dir: str = None) -> float:
    """
    Story 3.2: Estimate tuning complexity for work-stealing optimization.
    
    Higher score = more complex (should be tuned first for load balancing).
    Based on: data length, annualized vol (volatile assets have more models to try).
    
    Returns estimated complexity score (0-100).
    """
    if prices_dir is None:
        prices_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "prices")
    
    price_file = os.path.join(prices_dir, f"{symbol}_1d.csv")
    if not os.path.exists(price_file):
        return 50.0  # Default complexity
    
    try:
        with open(price_file, 'r') as f:
            line_count = sum(1 for _ in f)
        # Length score: more data = more work
        length_score = min(line_count / 30, 50)
        
        # Vol score: approximate from last 20 lines
        with open(price_file, 'r') as f:
            lines = f.readlines()
        if len(lines) < 22:
            return length_score
        
        closes = []
        for line in lines[-21:]:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    closes.append(float(parts[4]))
                except (ValueError, IndexError):
                    continue
        
        if len(closes) >= 10:
            import numpy as np
            rets = np.diff(np.log(np.array(closes)))
            vol = float(np.std(rets) * np.sqrt(252))
            vol_score = min(vol / 0.01, 50)  # High vol = more complexity
            return length_score + vol_score
        
        return length_score
    except Exception:
        return 50.0


def sort_assets_by_complexity(assets: list, prices_dir: str = None) -> list:
    """
    Story 3.2: Sort assets by estimated complexity, slowest first.
    
    This enables work-stealing: slow assets start first, fast ones fill gaps.
    """
    scored = [(a, estimate_tuning_complexity(a, prices_dir)) for a in assets]
    scored.sort(key=lambda x: -x[1])  # Descending: most complex first
    return [a for a, _ in scored]


def get_optimal_worker_count() -> int:
    """Story 3.2: Optimal worker count for parallel tuning."""
    cpu = os.cpu_count() or 4
    return min(cpu - 1, 8)  # Leave 1 core for OS, cap at 8


# =============================================================================
# Story 3.3: GARCH(1,1) MLE Parameter Fitting
# =============================================================================

GARCH_ALPHA_MIN = 1e-6
GARCH_ALPHA_MAX = 0.50
GARCH_BETA_MIN = 1e-6
GARCH_BETA_MAX = 0.999
GARCH_OMEGA_MIN = 1e-12
GARCH_PERSISTENCE_MAX = 0.999

# -------------------------------------------------------------------------
# Multi-Start GARCH (Story 1.6)
# -------------------------------------------------------------------------
# Five (alpha, beta) starting points covering the typical GARCH parameter
# space.  The optimizer runs from each start and selects the highest LL.
# -------------------------------------------------------------------------
GARCH_STARTS = [
    (0.03, 0.93),  # Low reactivity, high persistence
    (0.08, 0.88),  # Standard (legacy)
    (0.15, 0.80),  # High reactivity
    (0.05, 0.90),  # Medium
    (0.10, 0.85),  # Medium-high reactivity
]


def gjr_garch_log_likelihood(params, returns, barrier_lambda=5.0):
    """
    GJR-GARCH(1,1) negative log-likelihood (Glosten-Jagannathan-Runkle 1993).

    h_t = omega + alpha * e_{t-1}^2 + gamma * e_{t-1}^2 * I(e_{t-1}<0) + beta * h_{t-1}

    params: [omega, alpha, beta, gamma]
    returns: array of mean-adjusted returns
    barrier_lambda: log-barrier penalty weight for stationarity (Story 3.1)

    The leverage term gamma captures asymmetric volatility:
      - Negative returns increase variance by (alpha + gamma) * e^2
      - Positive returns increase variance by alpha * e^2
    """
    omega, alpha, beta, gamma = params
    T = len(returns)

    # Effective persistence including half the leverage effect
    eff_pers = alpha + gamma * 0.5 + beta

    # Story 3.1: log-barrier stationarity penalty
    # -lambda * log(1 - persistence) -> +inf as persistence -> 1
    if eff_pers >= 1.0:
        return 1e15
    barrier_penalty = -barrier_lambda * np.log(max(1.0 - eff_pers, 1e-15))

    if eff_pers < 1.0:
        sigma2_init = omega / (1.0 - eff_pers)
    else:
        sigma2_init = np.var(returns)

    sigma2 = max(sigma2_init, 1e-20)
    total_ll = 0.0

    for t in range(T):
        r = returns[t]
        if sigma2 < 1e-20:
            sigma2 = 1e-20
        total_ll += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + r * r / sigma2)
        leverage = gamma * r * r * (1.0 if r < 0.0 else 0.0)
        sigma2 = omega + alpha * r * r + leverage + beta * sigma2

    return -total_ll + barrier_penalty  # Negative for minimization


def garch_log_likelihood(params, returns):
    """
    GARCH(1,1) negative log-likelihood (backward compatible wrapper).

    Delegates to GJR-GARCH with gamma=0.
    """
    omega, alpha, beta = params
    return gjr_garch_log_likelihood([omega, alpha, beta, 0.0], returns)


def fit_garch_mle(returns, max_iter=200):
    """
    Fit GJR-GARCH(1,1) via multi-start MLE (Story 1.6).

    Runs SLSQP from 5 starting points covering the (alpha, beta) space,
    selects the solution with highest log-likelihood.  The GJR leverage
    term gamma captures asymmetric volatility.

    Returns dict with:
      - omega, alpha, beta, gamma (GJR leverage)
      - persistence (alpha + gamma/2 + beta)
      - long_run_vol (annualized)
      - converged: bool
      - se_alpha, se_beta, se_gamma (standard errors via Hessian)
      - best_start: (alpha0, beta0) of the winning start
      - n_starts_tried: number of starts attempted
    """
    from scipy.optimize import minimize

    returns = np.asarray(returns, dtype=np.float64)
    returns = returns - np.mean(returns)
    sample_var = float(np.var(returns))

    if sample_var < 1e-20 or len(returns) < 30:
        return None

    bounds = [
        (GARCH_OMEGA_MIN, sample_var * 10),   # omega
        (GARCH_ALPHA_MIN, GARCH_ALPHA_MAX),    # alpha
        (GARCH_BETA_MIN, GARCH_BETA_MAX),      # beta
        (0.0, 0.30),                           # gamma (leverage)
    ]

    # Stationarity: alpha + gamma/2 + beta < 0.999
    constraints = [{
        'type': 'ineq',
        'fun': lambda p: GARCH_PERSISTENCE_MAX - (p[1] + p[3] * 0.5 + p[2]),
    }]

    best_result = None
    best_ll = float('inf')  # minimising neg-LL
    best_start = None
    n_tried = 0

    for alpha0, beta0 in GARCH_STARTS:
        omega0 = sample_var * max(1.0 - alpha0 - beta0, 0.01)
        x0 = [max(omega0, 1e-10), alpha0, beta0, 0.02]  # gamma0=0.02

        try:
            result = minimize(
                gjr_garch_log_likelihood,
                x0,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-10},
            )
            n_tried += 1
            if result.fun < best_ll:
                best_ll = result.fun
                best_result = result
                best_start = (alpha0, beta0)
        except Exception:
            n_tried += 1
            continue

    if best_result is None:
        return None

    omega, alpha, beta, gamma = best_result.x
    persistence = alpha + gamma * 0.5 + beta

    if persistence >= 1.0:
        return None

    long_run_var = omega / (1.0 - persistence) if persistence < 1 else sample_var
    long_run_vol = float(np.sqrt(max(long_run_var, 0.0) * 252))

    # Standard errors via numerical Hessian (central finite differences)
    se_alpha = float('inf')
    se_beta = float('inf')
    se_gamma = float('inf')
    try:
        h = 1e-5
        x_opt = best_result.x
        n_p = len(x_opt)
        hessian = np.zeros((n_p, n_p))
        f0 = best_result.fun
        for i in range(n_p):
            for j in range(i, n_p):
                xpp = x_opt.copy()
                xpn = x_opt.copy()
                xnp = x_opt.copy()
                xnn = x_opt.copy()
                xpp[i] += h; xpp[j] += h
                xpn[i] += h; xpn[j] -= h
                xnp[i] -= h; xnp[j] += h
                xnn[i] -= h; xnn[j] -= h
                hessian[i, j] = (
                    gjr_garch_log_likelihood(xpp, returns)
                    - gjr_garch_log_likelihood(xpn, returns)
                    - gjr_garch_log_likelihood(xnp, returns)
                    + gjr_garch_log_likelihood(xnn, returns)
                ) / (4.0 * h * h)
                hessian[j, i] = hessian[i, j]
        # Hessian of neg-LL -> information matrix
        # SE = sqrt(diag(inv(Hessian)))
        eigvals = np.linalg.eigvalsh(hessian)
        if np.all(eigvals > 1e-12):
            cov = np.linalg.inv(hessian)
            diag = np.diag(cov)
            if diag[1] > 0:
                se_alpha = float(np.sqrt(diag[1]))
            if diag[2] > 0:
                se_beta = float(np.sqrt(diag[2]))
            if diag[3] > 0:
                se_gamma = float(np.sqrt(diag[3]))
    except Exception:
        pass

    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "garch_leverage": float(gamma),  # alias for downstream
        "persistence": float(persistence),
        "long_run_vol": long_run_vol,
        "converged": bool(best_result.success),
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "se_gamma": se_gamma,
        "best_start": best_start,
        "n_starts_tried": n_tried,
    }


# =============================================================================
# Story 3.4: OU Parameter Estimation in Tuning Pipeline
# =============================================================================

OU_HALF_LIFE_MIN_TUNE = 5
OU_HALF_LIFE_MAX_TUNE = 252


def fit_ou_params(prices):
    """
    Story 3.4: Estimate Ornstein-Uhlenbeck parameters from prices.
    
    AR(1) regression: log_price_t = phi * log_price_{t-1} + c + eps
    kappa = -log(phi) (daily frequency, dt=1)
    
    Returns dict with:
      - kappa: mean reversion speed
      - theta: long-run level (current EWMA of prices)
      - sigma_ou: residual volatility
      - half_life_days: ln(2)/kappa
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    if len(prices_arr) < 60:
        return None
    
    log_p = np.log(prices_arr[prices_arr > 0])
    if len(log_p) < 60:
        return None
    
    # AR(1) regression: y_t = phi * y_{t-1}
    y = log_p[1:]
    x = log_p[:-1]
    
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    
    if den < 1e-20:
        return None
    
    phi = num / den
    phi = max(phi, 0.001)
    phi = min(phi, 0.9999)
    
    kappa = -np.log(phi)
    kappa = np.clip(kappa, np.log(2) / OU_HALF_LIFE_MAX_TUNE, np.log(2) / OU_HALF_LIFE_MIN_TUNE)
    
    half_life = np.log(2) / kappa
    
    # Theta from EWMA with span = half_life
    span = max(int(half_life), 5)
    ew_alpha = 2.0 / (span + 1)
    theta = float(prices_arr[0])
    for p in prices_arr[1:]:
        theta = ew_alpha * float(p) + (1 - ew_alpha) * theta
    
    # Residual vol
    residuals = y - phi * x
    sigma_ou = float(np.std(residuals))
    
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_ou": sigma_ou,
        "half_life_days": float(half_life),
    }


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
    vol_flat = np.asarray(vol).flatten()
    mu_flat = np.asarray(mu_filtered).flatten()
    P_flat = np.asarray(P_filtered).flatten()
    phi_sq = phi * phi

    # Vectorized reconstruction (replaces per-element Python loop)
    mu_pred = np.empty(n, dtype=np.float64)
    S_pred = np.empty(n, dtype=np.float64)

    # t=0: prior
    mu_pred[0] = 0.0
    S_pred[0] = (1e-4 + q) + c * (vol_flat[0] * vol_flat[0])

    # t>=1: vectorized
    if n > 1:
        mu_pred[1:] = phi * mu_flat[:n - 1]
        S_pred[1:] = phi_sq * P_flat[:n - 1] + q + c * (vol_flat[1:] * vol_flat[1:])

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


def compute_extended_pit_metrics_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Dict:
    """PIT + Berkowitz + histogram MAD for Gaussian models."""
    # kstest, norm already imported at module level (line 280)

    mu_pred, S_pred = reconstruct_predictive_from_filtered_gaussian(
        returns, mu_filtered, P_filtered, vol, q, c, phi
    )
    returns_flat = np.asarray(returns).flatten()
    mu_pred_flat = np.asarray(mu_pred).flatten()
    S_pred_flat = np.asarray(S_pred).flatten()
    forecast_std = np.sqrt(np.maximum(S_pred_flat, 1e-20))
    forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
    standardized = (returns_flat - mu_pred_flat) / forecast_std

    # ── Chi² EWM variance correction (causal scale adaptation) ────────
    # Same algorithm as Student-t version but with chi2_target = 1.0
    _n_g = len(standardized)
    _chi2_lam_g = 0.98
    try:
        from models.numba_wrappers import run_chi2_ewm_correction as _numba_chi2_g
        _std_adj = _numba_chi2_g(standardized, 1.0, _chi2_lam_g)
    except (ImportError, Exception):
        import math as _m
        _chi2_1m_g = 0.02
        _chi2_wcap_g = 50.0  # target=1.0 so cap=50
        _ewm_z2_g = 1.0
        _std_adj = np.ones(_n_g)
        for _t in range(_n_g):
            _ratio = _ewm_z2_g  # / 1.0
            _ratio = max(0.3, min(3.0, _ratio))
            _dev = abs(_ratio - 1.0)
            if _ratio >= 1.0:
                _dz_lo, _dz_rng = 0.25, 0.25
            else:
                _dz_lo, _dz_rng = 0.10, 0.15
            if _dev < _dz_lo:
                _adj = 1.0
            elif _dev >= _dz_lo + _dz_rng:
                _adj = _m.sqrt(_ratio)
            else:
                _s = (_dev - _dz_lo) / _dz_rng
                _adj = 1.0 + _s * (_m.sqrt(_ratio) - 1.0)
            _std_adj[_t] = _adj
            _z2 = standardized[_t] ** 2
            _z2w = min(_z2, _chi2_wcap_g)
            _ewm_z2_g = _chi2_lam_g * _ewm_z2_g + _chi2_1m_g * _z2w
    standardized_corrected = standardized / _std_adj

    valid_mask = np.isfinite(standardized_corrected)
    pit_values = norm.cdf(standardized_corrected[valid_mask])

    # ── PIT-Variance stretching (Var[PIT] → 1/12) ────────────────────
    try:
        from models.numba_wrappers import run_pit_var_stretching as _numba_pvs_g
        pit_values = _numba_pvs_g(pit_values)
    except (ImportError, Exception):
        import math as _m
        _pv_tgt_g = 1.0 / 12.0
        _pv_lam_g = 0.97
        _pv_1m_g = 0.03
        _pv_dz_lo_g = 0.30
        _pv_dz_hi_g = 0.55
        _pv_dz_rng_g = _pv_dz_hi_g - _pv_dz_lo_g
        _ewm_pm_g = 0.5
        _ewm_psq_g = 1.0 / 3.0
        for _t in range(len(pit_values)):
            _ov = _ewm_psq_g - _ewm_pm_g * _ewm_pm_g
            if _ov < 0.005:
                _ov = 0.005
            _vr = _ov / _pv_tgt_g
            _vd = abs(_vr - 1.0)
            _rp = float(pit_values[_t])
            if _vd > _pv_dz_lo_g:
                _rs = _m.sqrt(_pv_tgt_g / _ov)
                _rs = max(0.70, min(1.50, _rs))
                if _vd >= _pv_dz_hi_g:
                    _st = _rs
                else:
                    _sg = (_vd - _pv_dz_lo_g) / _pv_dz_rng_g
                    _st = 1.0 + _sg * (_rs - 1.0)
                _c = 0.5 + (_rp - 0.5) * _st
                pit_values[_t] = max(0.001, min(0.999, _c))
            _ewm_pm_g = _pv_lam_g * _ewm_pm_g + _pv_1m_g * _rp
            _ewm_psq_g = _pv_lam_g * _ewm_psq_g + _pv_1m_g * _rp * _rp

    if len(pit_values) < 20:
        return {"ks_statistic": 1.0, "pit_ks_pvalue": 0.0,
                "berkowitz_pvalue": 0.0, "berkowitz_lr": 0.0,
                "pit_count": 0, "histogram_mad": 1.0}
    ks_stat_g, ks_pval_g = _fast_ks_uniform(pit_values)
    hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
    hist_freq = hist / len(pit_values)
    hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
    # Berkowitz on TEST split (last 30%) — in-sample Berk always gives p≈0
    # due to serial dependence from volatility clustering not captured by filter
    n_test_start_g = int(len(pit_values) * 0.7)
    pit_test_g = pit_values[n_test_start_g:]
    if len(pit_test_g) >= 30:
        berkowitz_p, berkowitz_lr_g, pit_count_g = PhiStudentTDriftModel._compute_berkowitz_full(pit_test_g)
    else:
        berkowitz_p, berkowitz_lr_g, pit_count_g = PhiStudentTDriftModel._compute_berkowitz_full(pit_values)
    if not np.isfinite(berkowitz_p):
        berkowitz_p = 0.0
    return {
        "ks_statistic": float(ks_stat_g),
        "pit_ks_pvalue": float(ks_pval_g),
        "berkowitz_pvalue": float(berkowitz_p),
        "berkowitz_lr": float(berkowitz_lr_g),
        "pit_count": int(pit_count_g),
        "histogram_mad": float(hist_mad),
    }


def compute_extended_pit_metrics_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    mu_pred_precomputed: np.ndarray = None,
    S_pred_precomputed: np.ndarray = None,
    scale_already_adapted: bool = False,
    forecast_scale_override: np.ndarray = None,
) -> Dict:
    """PIT + Berkowitz + histogram MAD for Student-t models.

    Performance: if mu_pred_precomputed and S_pred_precomputed are provided,
    skips the expensive filter_phi_with_predictive call entirely.
    PIT CDF computation is vectorized (~5x faster than per-element loop).

    Args:
        scale_already_adapted: if True, S_pred was produced by filter with
            online_scale_adapt=True. Skips chi² EWM and PIT-variance
            stretching to avoid double-correction.
        forecast_scale_override: if provided, use this scale array directly
            instead of computing from S_pred. Used by isotonic scale
            correction (Stage 8) to pass corrected scale.
    """
    # student_t (as t), kstest already imported at module level
    _st_dist = student_t

    if mu_pred_precomputed is not None and S_pred_precomputed is not None:
        mu_pred = mu_pred_precomputed
        S_pred = S_pred_precomputed
    else:
        _, _, mu_pred, S_pred, _ = PhiStudentTDriftModel.filter_phi_with_predictive(
            returns, vol, q, c, phi, nu
        )

    returns_flat = np.asarray(returns).flatten()
    n = min(len(returns_flat), len(mu_pred), len(S_pred))

    # Use override scale (from isotonic scale correction) if provided
    if forecast_scale_override is not None:
        scale_arr = np.maximum(forecast_scale_override[:n], 1e-10)
    else:
        # Vectorized PIT computation (replaces per-element Python loop)
        S_clamped = np.maximum(S_pred[:n], 1e-20)
        if nu > 2:
            scale_arr = np.sqrt(S_clamped * (nu - 2) / nu)
        else:
            scale_arr = np.sqrt(S_clamped)
        scale_arr = np.maximum(scale_arr, 1e-10)

    # ── Chi² EWM variance correction (causal scale adaptation) ────────
    # Tracks E[z²] and corrects scale when filter variance is systematically
    # off. Same algorithm as unified models but applied to base models.
    # This is the #1 fix for systemic PIT < 0.05 across all assets.
    # SKIP when scale_already_adapted (filter did online_scale_adapt) to
    # avoid double-correction that ruins calibration (March 2026 fix).
    if scale_already_adapted:
        # Filter already did chi² EWM in _filter_phi_core — no post-hoc needed
        scale_corrected = scale_arr
    else:
        _z_raw = (returns_flat[:n] - mu_pred[:n]) / scale_arr
        _chi2_tgt = nu / (nu - 2.0) if nu > 2.0 else 1.0
        _chi2_lam = 0.98
        try:
            from models.numba_wrappers import run_chi2_ewm_correction as _numba_chi2_t
            _scale_adj = _numba_chi2_t(_z_raw, _chi2_tgt, _chi2_lam)
        except (ImportError, Exception):
            import math as _m
            _chi2_1m = 0.02
            _chi2_wcap = _chi2_tgt * 50.0
            _ewm_z2 = _chi2_tgt
            _scale_adj = np.ones(n)
            for _t in range(n):
                _ratio = _ewm_z2 / _chi2_tgt
                _ratio = max(0.3, min(3.0, _ratio))
                _dev = abs(_ratio - 1.0)
                if _ratio >= 1.0:
                    _dz_lo, _dz_rng = 0.25, 0.25
                else:
                    _dz_lo, _dz_rng = 0.05, 0.10  # Tighter for under-dispersion
                if _dev < _dz_lo:
                    _adj = 1.0
                elif _dev >= _dz_lo + _dz_rng:
                    _adj = _m.sqrt(_ratio)
                else:
                    _s = (_dev - _dz_lo) / _dz_rng
                    _adj = 1.0 + _s * (_m.sqrt(_ratio) - 1.0)
                _scale_adj[_t] = _adj
                _z2 = _z_raw[_t] ** 2
                _z2w = min(_z2, _chi2_wcap)
                _ewm_z2 = _chi2_lam * _ewm_z2 + _chi2_1m * _z2w
        scale_corrected = scale_arr * _scale_adj

    # Pre-standardize then use Numba CDF (avoids scipy per-element scale dispatch)
    _z_std = (returns_flat[:n] - mu_pred[:n]) / scale_corrected
    try:
        from models.phi_student_t import _fast_t_cdf as _tune_fast_t_cdf
        pit_values = _tune_fast_t_cdf(_z_std, nu)
    except (ImportError, Exception):
        pit_values = _st_dist.cdf(_z_std, df=nu)

    # ── PIT-Variance stretching (Var[PIT] → 1/12) ────────────────────
    # Fixes shape miscalibration not caught by chi² (scale) correction.
    # Also skip when filter already adapted the scale (double-correction).
    if not scale_already_adapted:
        try:
            from models.numba_wrappers import run_pit_var_stretching as _numba_pvs_t
            pit_values = _numba_pvs_t(pit_values)
        except (ImportError, Exception):
            import math as _m
            _pv_tgt = 1.0 / 12.0
            _pv_lam = 0.97
            _pv_1m = 0.03
            _pv_dz_lo = 0.30
            _pv_dz_hi = 0.55
            _pv_dz_rng = _pv_dz_hi - _pv_dz_lo
            _ewm_pm = 0.5
            _ewm_psq = 1.0 / 3.0
            for _t in range(n):
                _ov = _ewm_psq - _ewm_pm * _ewm_pm
                if _ov < 0.005:
                    _ov = 0.005
                _vr = _ov / _pv_tgt
                _vd = abs(_vr - 1.0)
                _rp = float(pit_values[_t])
                if _vd > _pv_dz_lo:
                    _rs = _m.sqrt(_pv_tgt / _ov)
                    _rs = max(0.70, min(1.50, _rs))
                    if _vd >= _pv_dz_hi:
                        _st = _rs
                    else:
                        _sg = (_vd - _pv_dz_lo) / _pv_dz_rng
                        _st = 1.0 + _sg * (_rs - 1.0)
                    _c = 0.5 + (_rp - 0.5) * _st
                    pit_values[_t] = max(0.001, min(0.999, _c))
                _ewm_pm = _pv_lam * _ewm_pm + _pv_1m * _rp
                _ewm_psq = _pv_lam * _ewm_psq + _pv_1m * _rp * _rp

    valid = np.isfinite(pit_values)
    pit_clean = np.clip(pit_values[valid], 0, 1)
    if len(pit_clean) < 20:
        return {"ks_statistic": 1.0, "pit_ks_pvalue": 0.0,
                "berkowitz_pvalue": 0.0, "berkowitz_lr": 0.0,
                "pit_count": 0, "histogram_mad": 1.0}
    ks_stat_st, ks_pval_st = _fast_ks_uniform(pit_clean)

    # ── AD Tail-Correction Pipeline (March 2026) ─────────────────────
    # Apply TWSC + SPTG + Isotonic corrections to PIT values for AD test only.
    # KS and Berkowitz continue using raw pit_clean (no double-dipping).
    _ad_pval_raw = float('nan')
    _ad_correction_diag = {}
    pit_for_ad = pit_clean  # default: uncorrected
    try:
        from calibration.pit_calibration import anderson_darling_uniform
        _ad_stat_raw, _ad_pval_raw = anderson_darling_uniform(pit_clean)
    except Exception:
        pass

    try:
        pit_ad_corrected, _ad_correction_diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns_flat[:n], mu_pred[:n], scale_corrected, nu, pit_clean
        )
        pit_for_ad = pit_ad_corrected
    except Exception:
        pass

    # Anderson-Darling test (tail-sensitive) — on corrected PIT
    try:
        from calibration.pit_calibration import anderson_darling_uniform
        _ad_stat, _ad_pval = anderson_darling_uniform(pit_for_ad)
    except Exception:
        _ad_pval = float('nan')

    hist, _ = np.histogram(pit_clean, bins=10, range=(0, 1))
    hist_freq = hist / len(pit_clean)
    hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
    # Berkowitz on TEST split (last 30%) — in-sample Berk always gives p≈0
    # due to serial dependence from overfitting
    n_test_start = int(len(pit_clean) * 0.7)
    pit_test = pit_clean[n_test_start:]
    if len(pit_test) >= 30:
        berkowitz_p, berkowitz_lr_st, pit_count_st = PhiStudentTDriftModel._compute_berkowitz_full(pit_test)
    else:
        berkowitz_p, berkowitz_lr_st, pit_count_st = PhiStudentTDriftModel._compute_berkowitz_full(pit_clean)
    if not np.isfinite(berkowitz_p):
        berkowitz_p = 0.0
    return {
        "ks_statistic": float(ks_stat_st),
        "pit_ks_pvalue": float(ks_pval_st),
        "ad_pvalue": float(_ad_pval),
        "ad_pvalue_raw": float(_ad_pval_raw),
        "ad_correction": _ad_correction_diag,
        "berkowitz_pvalue": float(berkowitz_p),
        "berkowitz_lr": float(berkowitz_lr_st),
        "pit_count": int(pit_count_st),
        "histogram_mad": float(hist_mad),
        "pit_values": pit_clean,  # For isotonic recalibration (March 2026)
    }


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
    # Phi Lower Bound from Autocorrelation (Story 1.8)
    # =========================================================================
    # Compute empirical lag-1 autocorrelation of returns and derive a
    # principled lower bound for phi.  If autocorrelation is positive,
    # phi should be positive — preventing the anti-persistent phi values
    # that signals.py currently floors at 0.0 post-hoc.
    #
    # phi_floor = max(0, acf_1 * PHI_ACF_SCALE)
    # =========================================================================
    _acf_1 = 0.0
    _phi_acf_floor = 0.0
    if n_obs > 10:
        _r = np.asarray(returns).flatten()
        _acf_corr = np.corrcoef(_r[:-1], _r[1:])
        if _acf_corr.shape == (2, 2) and np.isfinite(_acf_corr[0, 1]):
            _acf_1 = float(_acf_corr[0, 1])
        _phi_acf_floor = max(0.0, _acf_1 * PHI_ACF_SCALE)

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
    # Delegates fully to PhiStudentTDriftModel.tune_and_calibrate() which
    # encapsulates: MLE → filter → PIT/CRPS → GARCH blending → ν-refinement
    # → GAS ν → CRPS shrinkage → momentum augmentation → isotonic recal.
    # =========================================================================

    # VoV precomputation — shared across all ν (data-only, O(n))
    _vov_rolling = PhiStudentTDriftModel._precompute_vov(vol)
    _gamma_vov_fixed = 0.3

    # Pre-create momentum wrapper for internal CRPS comparison (if available)
    _st_momentum_wrapper = None
    if MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE:
        _st_momentum_wrapper = MomentumAugmentedDriftModel(DEFAULT_MOMENTUM_CONFIG)
        if prices is not None:
            _st_momentum_wrapper.precompute_signals(
                returns=returns, prices=prices, vol=vol,
                regime_labels=regime_labels, q=1e-6,
            )
        else:
            _st_momentum_wrapper.precompute_momentum(returns)

    _st_warm_start = None  # Cross-ν warm-starting (March 2026)
    for nu_fixed in STUDENT_T_NU_GRID:
        model_name = f"phi_student_t_nu_{nu_fixed}"
        try:
            models[model_name] = PhiStudentTDriftModel.tune_and_calibrate(
                returns, vol, nu_fixed,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda,
                asset_symbol=asset,
                n_obs=n_obs,
                n_train=int(n_obs * 0.7),
                vov_rolling=_vov_rolling,
                gamma_vov=_gamma_vov_fixed,
                momentum_wrapper=_st_momentum_wrapper,
                prices=prices,
                regime_labels=regime_labels,
                warm_start_params=_st_warm_start,
            )
            # Extract optimised (q, c, φ) for warm-starting next ν
            _m = models[model_name]
            if _m.get("fit_success", False):
                _st_warm_start = (_m.get("q", 1e-5), _m.get("c", 1.0), _m.get("phi", 0.0))
        except Exception as e:
            models[model_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "crps": float('inf'),
                "nu": float(nu_fixed),
                "nu_fixed": True,
            }

    # =========================================================================
    # Model 1b: Continuous Nu Optimization (Story 1.5)
    # =========================================================================
    # Profile-likelihood refinement: hold (q, c, phi) at best discrete MLE,
    # optimise nu over [2.5, 60] via bounded Brent.  The discrete grid remains
    # the initialisation — this finds the true MLE between grid points.
    #
    # Mathematical justification:
    #   L_profile(nu) = L(q_hat, c_hat, phi_hat, nu)
    #   nu_mle = argmax_nu  L_profile(nu)
    #   se(nu) = 1 / sqrt( -d^2 log L_profile / d nu^2 |_{nu=nu_mle} )
    # =========================================================================
    _best_discrete_bic = float('inf')
    _best_discrete_name = None
    for _dn, _dm in models.items():
        if _dn.startswith("phi_student_t_nu_") and _dm.get("fit_success", False):
            _dbic = _dm.get("bic", float('inf'))
            if _dbic < _best_discrete_bic:
                _best_discrete_bic = _dbic
                _best_discrete_name = _dn

    if _best_discrete_name is not None:
        _best_dm = models[_best_discrete_name]
        _prof_q = float(_best_dm.get("q", 1e-5))
        _prof_c = float(_best_dm.get("c", 1.0))
        _prof_phi = float(_best_dm.get("phi", 0.0))
        _prof_nu_init = float(_best_dm.get("nu", 8.0))
        _prof_gamma_vov = float(_best_dm.get("gamma_vov", 0.0))

        # Profile log-likelihood: filter with fixed (q, c, phi), varying nu
        def _profile_neg_ll(nu_candidate):
            nu_c = float(np.clip(nu_candidate, NU_CONTINUOUS_BOUNDS[0], NU_CONTINUOUS_BOUNDS[1]))
            try:
                _, _, _, _, ll = PhiStudentTDriftModel.filter_phi_with_predictive(
                    returns, vol, _prof_q, _prof_c, _prof_phi, nu_c,
                    robust_wt=True,
                    gamma_vov=_prof_gamma_vov,
                    vov_rolling=_vov_rolling if _prof_gamma_vov > 1e-12 else None,
                )
                if not np.isfinite(ll):
                    return 1e12
                return -ll
            except Exception:
                return 1e12

        try:
            _nu_opt_result = minimize_scalar(
                _profile_neg_ll,
                bounds=NU_CONTINUOUS_BOUNDS,
                method='bounded',
                options={'xatol': 0.05, 'maxiter': 40},
            )
            _nu_mle = float(_nu_opt_result.x)

            # Standard error via central finite-difference Hessian
            _h = NU_SE_FD_STEP
            _ll_center = -_profile_neg_ll(_nu_mle)
            _ll_plus = -_profile_neg_ll(_nu_mle + _h)
            _ll_minus = -_profile_neg_ll(_nu_mle - _h)
            _d2ll = (_ll_plus - 2.0 * _ll_center + _ll_minus) / (_h * _h)
            if _d2ll < -1e-12:
                _nu_se = 1.0 / math.sqrt(-_d2ll)
            else:
                _nu_se = float('inf')

            # Profile LL at the optimum
            _ll_mle = -_profile_neg_ll(_nu_mle)

            # Only create the continuous model if it improves on discrete
            _n_params_mle = 4  # q, c, phi, nu (same param count as discrete)
            _bic_mle = -2.0 * _ll_mle + _n_params_mle * math.log(max(n_obs, 1))

            _nu_mle_name = "phi_student_t_nu_mle"

            # Copy best discrete model and update nu-related fields
            _mle_model = dict(_best_dm)
            _mle_model["nu"] = _nu_mle
            _mle_model["nu_mle"] = _nu_mle
            _mle_model["nu_se"] = min(_nu_se, 999.0)
            _mle_model["nu_discrete_init"] = _prof_nu_init
            _mle_model["nu_fixed"] = False
            _mle_model["log_likelihood"] = _ll_mle
            _mle_model["bic"] = _bic_mle
            _mle_model["fit_success"] = True
            _mle_model["continuous_nu_optimization"] = True

            models[_nu_mle_name] = _mle_model
        except Exception:
            pass  # Discrete grid models still available

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
    
    # Unified models use 3 ν flavours matching STUDENT_T_NU_GRID.
    # Internal Stage 5 re-optimizes ν across a finer grid [3..20],
    # so intermediate values (6, 10, 12, 15) are still explored —
    # we just don't spawn separate BMA models for them.
    UNIFIED_NU_GRID = [3, 4, 8, 20]
    n_params_unified = 14  # q, c, φ, γ_vov, ms_sensitivity, α_asym, ν, garch(3), rough_hurst, risk_premium, skew(2), jump(4-cond)
    
    from models.phi_student_t_unified import UnifiedPhiStudentTModel
    
    for nu_fixed in UNIFIED_NU_GRID:
        unified_name = f"phi_student_t_unified_nu_{nu_fixed}"
        
        try:
            # Staged optimization for unified model
            config, diagnostics = UnifiedPhiStudentTModel.optimize_params_unified(
                returns, vol, 
                nu_base=float(nu_fixed),
                train_frac=0.7, 
                asset_symbol=asset
            )

            if not diagnostics.get("success", False):
                raise ValueError(f"Unified optimization failed: {diagnostics.get('error', 'Unknown')}")

            # Run unified filter to get predictive values
            mu_u, P_u, mu_pred_u, S_pred_u, ll_u = UnifiedPhiStudentTModel.filter_phi_unified(
                returns, vol, config
            )

            # PIT calibration using predictive distribution
            ks_u, pit_p_u, pit_metrics = UnifiedPhiStudentTModel.pit_ks_unified(
                returns, mu_pred_u, S_pred_u, config
            )

            # Information criteria — adjust param count if jump/rough layers active
            jump_active = getattr(config, 'jump_intensity', 0.0) > 1e-6 and getattr(config, 'jump_variance', 0.0) > 1e-12
            rough_active = getattr(config, 'rough_hurst', 0.0) > 0.01
            effective_n_params = n_params_unified + (4 if jump_active else 0) + (1 if rough_active else 0)
            aic_u = compute_aic(ll_u, effective_n_params)
            bic_u = compute_bic(ll_u, effective_n_params, n_obs)
            mean_ll_u = ll_u / max(n_obs, 1)

            # Hyvärinen and CRPS scores using calibrated predictive values
            # ─────────────────────────────────────────────────────────────
            # Use filter_and_calibrate to get the same calibrated sigma
            # and effective ν that the PIT test uses. This ensures tuning
            # optimises against the actual predictive distribution, not
            # the raw Kalman S_pred which ignores GARCH blending, β
            # recalibration, and ν refinement.
            # ─────────────────────────────────────────────────────────────
            n_train_u = int(n_obs * 0.7)
            # Default Berk/MAD/LR/count from raw PIT; overridden by filter_and_calibrate
            _calibrated_berk_u = pit_metrics.get("berkowitz_pvalue", 0.0)
            _calibrated_berk_lr_u = pit_metrics.get("berkowitz_lr", 0.0)
            _calibrated_pit_count_u = int(pit_metrics.get("pit_count", 0))
            _calibrated_mad_u = pit_metrics.get("histogram_mad", 1.0)
            _ad_p_u = float('nan')
            try:
                # ── PERFORMANCE: Reuse test-period evaluation from optimize_params_unified
                # instead of calling filter_and_calibrate a second time (saves ~1 full
                # pipeline pass per unified model: filter + GARCH + Stage 6 calibration)
                _calib_diag_u = diagnostics.get('test_calib_diag')
                if (_calib_diag_u is not None
                    and diagnostics.get('test_sigma') is not None
                    and diagnostics.get('test_pit_pvalue') is not None):
                    # Reuse cached results from optimize_params_unified
                    _sigma_cal_u = diagnostics['test_sigma']
                    _pit_p_u = diagnostics['test_pit_pvalue']
                    _pit_cal_u = diagnostics.get('test_pit_values')
                else:
                    # Fallback: call filter_and_calibrate (shouldn't normally happen)
                    _pit_cal_u, _pit_p_u, _sigma_cal_u, _, _calib_diag_u = \
                        UnifiedPhiStudentTModel.filter_and_calibrate(
                            returns, vol, config, train_frac=0.7
                        )
                # Use calibrated PIT p-value from filter_and_calibrate
                # (raw pit_ks_unified runs on full data without GARCH blending;
                #  filter_and_calibrate applies GARCH + beta recalibration on
                #  the test fold, matching what make pit-metals reports)
                if _pit_p_u is not None and np.isfinite(_pit_p_u):
                    pit_p_u = float(_pit_p_u)
                # Extract calibrated Berkowitz/MAD from GARCH path
                _bd = _calib_diag_u.get('berkowitz_pvalue')
                if _bd is not None and np.isfinite(_bd):
                    _calibrated_berk_u = float(_bd)
                _blr = _calib_diag_u.get('berkowitz_lr')
                if _blr is not None and np.isfinite(_blr):
                    _calibrated_berk_lr_u = float(_blr)
                _bpc = _calib_diag_u.get('pit_count')
                if _bpc is not None and _bpc > 0:
                    _calibrated_pit_count_u = int(_bpc)
                _md = _calib_diag_u.get('mad')
                if _md is not None and np.isfinite(_md):
                    _calibrated_mad_u = float(_md)
                _ad_p_u = _calib_diag_u.get('ad_pvalue', float('nan'))
                _nu_eff_u = _calib_diag_u.get('nu_effective', float(nu_fixed))
                returns_test_u = returns[n_train_u:]
                mu_pred_test_u = mu_pred_u[n_train_u:]
                
                if len(_sigma_cal_u) == len(returns_test_u) and np.all(_sigma_cal_u > 0):
                    forecast_scale_u = np.maximum(_sigma_cal_u, 1e-10)
                    nu_for_score = _nu_eff_u
                else:
                    if nu_fixed > 2:
                        forecast_scale_u = np.sqrt(S_pred_u[n_train_u:] * (nu_fixed - 2) / nu_fixed)
                    else:
                        forecast_scale_u = np.sqrt(S_pred_u[n_train_u:])
                    forecast_scale_u = np.maximum(forecast_scale_u, 1e-10)
                    nu_for_score = float(nu_fixed)
                
                mu_effective_u = mu_pred_test_u
            except Exception:
                # Fallback to raw predictions on full data
                if nu_fixed > 2:
                    forecast_scale_u = np.sqrt(S_pred_u * (nu_fixed - 2) / nu_fixed)
                else:
                    forecast_scale_u = np.sqrt(S_pred_u)
                forecast_scale_u = np.maximum(forecast_scale_u, 1e-10)
                mu_effective_u = mu_pred_u
                nu_for_score = float(nu_fixed)
                returns_test_u = returns

            hyvarinen_u = compute_hyvarinen_score_student_t(
                returns_test_u, mu_effective_u, forecast_scale_u, nu_for_score
            )
            crps_u = compute_crps_student_t_inline(
                returns_test_u, mu_effective_u, forecast_scale_u, nu_for_score
            )

            # =============================================================
            # STAGE U-H: Hansen Skew-t augmentation for unified model
            # =============================================================
            # Same CRPS-gated pattern as base phi_student_t (Stage 7.5).
            # Fits Hansen λ on training residuals, re-runs filter with
            # Hansen observation noise using the unified config's core
            # parameters (q, c, phi, nu).
            # =============================================================
            _u_hansen_activated = False
            _u_hansen_lambda = 0.0
            _u_hansen_diag = None
            try:
                if (forecast_scale_u is not None and len(forecast_scale_u) >= 50
                    and n_obs >= 100):
                    _fs_valid_u = forecast_scale_u[forecast_scale_u > 1e-12]
                    if len(_fs_valid_u) >= 50:
                        from models.hansen_skew_t import fit_hansen_skew_t_mle
                        # Compute standardised residuals for Hansen MLE
                        _resid_u = (returns_test_u - mu_effective_u) / np.maximum(forecast_scale_u, 1e-12)
                        _resid_train_u = _resid_u[:n_train_u] if len(_resid_u) > n_train_u else _resid_u[:len(_resid_u)//2]
                        _nu_h_u, _lam_h_u, _ll_h_u, _h_diag_u = fit_hansen_skew_t_mle(
                            _resid_train_u, nu_hint=nu_for_score)
                        if (_h_diag_u.get('fit_success', False) and
                            abs(_lam_h_u) > 0.01 and np.isfinite(_ll_h_u)):
                            from models.numba_wrappers import run_phi_hansen_skew_t_filter
                            mu_fh_u, P_fh_u, mu_ph_u, S_ph_u, ll_h_u = run_phi_hansen_skew_t_filter(
                                returns, vol,
                                float(config.q), float(config.c), float(config.phi),
                                nu_for_score,
                                hansen_lambda=_lam_h_u,
                                online_scale_adapt=True,
                                gamma_vov=float(getattr(config, 'gamma_vov', 0.0)),
                            )
                            if nu_for_score > 2:
                                _fs_h_u = np.sqrt(S_ph_u * (nu_for_score - 2) / nu_for_score)
                            else:
                                _fs_h_u = np.sqrt(S_ph_u)
                            _fs_h_u = np.maximum(_fs_h_u, 1e-10)
                            # CRPS on test fold
                            _crps_h_u = compute_crps_student_t_inline(
                                returns_test_u, mu_ph_u[n_train_u:] if len(mu_ph_u) > n_train_u else mu_ph_u,
                                _fs_h_u[n_train_u:] if len(_fs_h_u) > n_train_u else _fs_h_u,
                                nu_for_score)
                            _n_params_h_u = effective_n_params + 1
                            _bic_h_u = compute_bic(ll_h_u, _n_params_h_u, n_obs)
                            # CRPS gate
                            if (np.isfinite(_crps_h_u) and _crps_h_u < crps_u and
                                np.isfinite(_bic_h_u) and _bic_h_u < bic_u + 2.0):
                                _u_hansen_activated = True
                                _u_hansen_lambda = float(_lam_h_u)
                                _u_hansen_diag = {
                                    'nu_hansen': float(_nu_h_u),
                                    'lambda_hansen': float(_lam_h_u),
                                    'crps_before': float(crps_u),
                                    'crps_after': float(_crps_h_u),
                                    'bic_before': float(bic_u),
                                    'bic_after': float(_bic_h_u),
                                }
                                # Update scoring vars
                                mu_pred_u = mu_ph_u
                                S_pred_u = S_ph_u
                                forecast_scale_u = _fs_h_u
                                crps_u = _crps_h_u
                                ll_u = ll_h_u
                                mean_ll_u = ll_h_u / max(n_obs, 1)
                                effective_n_params = _n_params_h_u
                                bic_u = _bic_h_u
                                aic_u = compute_aic(ll_h_u, _n_params_h_u)
                                # Recompute diagnostics
                                _pit_h_u = compute_extended_pit_metrics_student_t(
                                    returns, vol,
                                    float(config.q), float(config.c), float(config.phi),
                                    nu_for_score,
                                    mu_pred_precomputed=mu_ph_u,
                                    S_pred_precomputed=S_ph_u,
                                    scale_already_adapted=True,
                                )
                                ks_u = _pit_h_u["ks_statistic"]
                                pit_p_u = _pit_h_u["pit_ks_pvalue"]
                                _ad_p_u = _pit_h_u.get("ad_pvalue", float('nan'))
                                _calibrated_berk_u = _pit_h_u["berkowitz_pvalue"]
                                _calibrated_mad_u = _pit_h_u["histogram_mad"]
                                hyvarinen_u = compute_hyvarinen_score_student_t(
                                    returns_test_u,
                                    mu_ph_u[n_train_u:] if len(mu_ph_u) > n_train_u else mu_ph_u,
                                    _fs_h_u[n_train_u:] if len(_fs_h_u) > n_train_u else _fs_h_u,
                                    nu_for_score)
            except Exception:
                pass

            # =============================================================
            # STAGE U-C: Contaminated Student-t augmentation for unified
            # =============================================================
            # Same CRPS-gated pattern as base phi_student_t (Stage 7.6).
            # Profile grid search over (ν_crisis, ε).
            # =============================================================
            _u_cst_activated = False
            _u_cst_nu_crisis = None
            _u_cst_epsilon = 0.0
            _u_cst_diag = None
            try:
                if n_obs >= 100 and nu_for_score >= 4:
                    _U_CST_NU_GRID = [3.0, 4.0, 5.0, 6.0]
                    _U_CST_EPS_GRID = [0.02, 0.05, 0.10]
                    _best_cst_crps_u = crps_u

                    _best_cst_combo_u = None
                    for _nc_u in _U_CST_NU_GRID:
                        if _nc_u >= nu_for_score:
                            continue
                        for _eps_u in _U_CST_EPS_GRID:
                            try:
                                from models.numba_wrappers import run_phi_cst_filter
                                mu_fc_u, P_fc_u, mu_pc_u, S_pc_u, ll_c_u = run_phi_cst_filter(
                                    returns, vol,
                                    float(config.q), float(config.c), float(config.phi),
                                    nu_normal=nu_for_score,
                                    nu_crisis=_nc_u,
                                    epsilon=_eps_u,
                                    online_scale_adapt=True,
                                    gamma_vov=float(getattr(config, 'gamma_vov', 0.0)),
                                )
                                if nu_for_score > 2:
                                    _fs_c_u = np.sqrt(S_pc_u * (nu_for_score - 2) / nu_for_score)
                                else:
                                    _fs_c_u = np.sqrt(S_pc_u)
                                _fs_c_u = np.maximum(_fs_c_u, 1e-10)
                                _crps_c_u = compute_crps_student_t_inline(
                                    returns_test_u,
                                    mu_pc_u[n_train_u:] if len(mu_pc_u) > n_train_u else mu_pc_u,
                                    _fs_c_u[n_train_u:] if len(_fs_c_u) > n_train_u else _fs_c_u,
                                    nu_for_score)
                                _n_params_c_u = effective_n_params + 2
                                _bic_c_u = compute_bic(ll_c_u, _n_params_c_u, n_obs)
                                if (np.isfinite(_crps_c_u) and _crps_c_u < _best_cst_crps_u and
                                    np.isfinite(_bic_c_u) and _bic_c_u < bic_u + 4.0):
                                    _best_cst_crps_u = _crps_c_u
                                    _best_cst_combo_u = (_nc_u, _eps_u, mu_pc_u, S_pc_u,
                                                         _fs_c_u, ll_c_u, _bic_c_u, _n_params_c_u)
                            except Exception:
                                continue

                    if _best_cst_combo_u is not None:
                        _nc_u, _eps_u, mu_pc_u, S_pc_u, _fs_c_u, ll_c_u, _bic_c_u, _n_params_c_u = _best_cst_combo_u
                        _u_cst_activated = True
                        _u_cst_nu_crisis = float(_nc_u)
                        _u_cst_epsilon = float(_eps_u)
                        _u_cst_diag = {
                            'nu_crisis': float(_nc_u),
                            'epsilon': float(_eps_u),
                            'crps_before': float(crps_u),
                            'crps_after': float(_best_cst_crps_u),
                            'bic_before': float(bic_u),
                            'bic_after': float(_bic_c_u),
                        }
                        # Update scoring vars
                        mu_pred_u = mu_pc_u
                        S_pred_u = S_pc_u
                        forecast_scale_u = _fs_c_u
                        crps_u = _best_cst_crps_u
                        ll_u = ll_c_u
                        mean_ll_u = ll_c_u / max(n_obs, 1)
                        effective_n_params = _n_params_c_u
                        bic_u = _bic_c_u
                        aic_u = compute_aic(ll_c_u, _n_params_c_u)
                        # Recompute diagnostics
                        _pit_c_u = compute_extended_pit_metrics_student_t(
                            returns, vol,
                            float(config.q), float(config.c), float(config.phi),
                            nu_for_score,
                            mu_pred_precomputed=mu_pc_u,
                            S_pred_precomputed=S_pc_u,
                            scale_already_adapted=True,
                        )
                        ks_u = _pit_c_u["ks_statistic"]
                        pit_p_u = _pit_c_u["pit_ks_pvalue"]
                        _ad_p_u = _pit_c_u.get("ad_pvalue", float('nan'))
                        _calibrated_berk_u = _pit_c_u["berkowitz_pvalue"]
                        _calibrated_mad_u = _pit_c_u["histogram_mad"]
                        hyvarinen_u = compute_hyvarinen_score_student_t(
                            returns_test_u,
                            mu_pc_u[n_train_u:] if len(mu_pc_u) > n_train_u else mu_pc_u,
                            _fs_c_u[n_train_u:] if len(_fs_c_u) > n_train_u else _fs_c_u,
                            nu_for_score)
            except Exception:
                pass

            models[unified_name] = {
                # Core parameters
                "q": float(config.q),
                "c": float(config.c),
                "phi": float(config.phi),
                "nu": float(nu_for_score),  # Use calibrated ν (from Stage 5/6), not grid ν
                "nu_grid": float(nu_fixed),  # Original grid value preserved for reference
                # Unified-specific parameters
                "gamma_vov": float(config.gamma_vov),
                "alpha_asym": float(config.alpha_asym),
                "k_asym": float(getattr(config, 'k_asym', 1.0)),
                "ms_sensitivity": float(config.ms_sensitivity),
                "ms_ewm_lambda": float(getattr(config, 'ms_ewm_lambda', 0.0)),
                "q_stress_ratio": float(config.q_stress_ratio),
                "vov_damping": float(getattr(config, 'vov_damping', 0.3)),
                # Calibration parameters (February 2026)
                "variance_inflation": float(getattr(config, 'variance_inflation', 1.0)),
                "mu_drift": float(getattr(config, 'mu_drift', 0.0)),
                "risk_premium_sensitivity": float(getattr(config, 'risk_premium_sensitivity', 0.0)),
                # Conditional skew dynamics (February 2026 - GAS Framework)
                "skew_score_sensitivity": float(getattr(config, 'skew_score_sensitivity', 0.0)),
                "skew_persistence": float(getattr(config, 'skew_persistence', 0.97)),
                # GARCH parameters (February 2026)
                "garch_omega": float(getattr(config, 'garch_omega', 0.0)),
                "garch_alpha": float(getattr(config, 'garch_alpha', 0.0)),
                "garch_beta": float(getattr(config, 'garch_beta', 0.0)),
                "garch_leverage": float(getattr(config, 'garch_leverage', 0.0)),
                "garch_unconditional_var": float(getattr(config, 'garch_unconditional_var', 1e-4)),
                # Rough volatility memory (February 2026 - Gatheral-Jaisson-Rosenbaum)
                "rough_hurst": float(getattr(config, 'rough_hurst', 0.0)),
                # Merton jump-diffusion parameters (February 2026)
                "jump_intensity": float(getattr(config, 'jump_intensity', 0.0)),
                "jump_variance": float(getattr(config, 'jump_variance', 0.0)),
                "jump_sensitivity": float(getattr(config, 'jump_sensitivity', 1.0)),
                "jump_mean": float(getattr(config, 'jump_mean', 0.0)),
                # Data-driven bounds
                "c_min": float(getattr(config, 'c_min', 0.01)),
                "c_max": float(getattr(config, 'c_max', 10.0)),
                "q_min": float(getattr(config, 'q_min', 1e-8)),
                # CRPS-optimal EWM location correction (February 2026)
                "crps_ewm_lambda": float(getattr(config, 'crps_ewm_lambda', 0.0)),
                # Heston-DLSV leverage and mean reversion (February 2026)
                "rho_leverage": float(getattr(config, 'rho_leverage', 0.0)),
                "kappa_mean_rev": float(getattr(config, 'kappa_mean_rev', 0.0)),
                "theta_long_var": float(getattr(config, 'theta_long_var', 0.0)),
                "crps_sigma_shrinkage": float(getattr(config, 'crps_sigma_shrinkage', 1.0)),
                # CRPS-enhancement: vol-of-vol noise, asymmetric df, regime switching
                "sigma_eta": float(getattr(config, 'sigma_eta', 0.0)),
                "t_df_asym": float(getattr(config, 't_df_asym', 0.0)),
                "regime_switch_prob": float(getattr(config, 'regime_switch_prob', 0.0)),
                # v7.8 elite MC enhancements: dynamic leverage, liquidity stress, entropy
                "leverage_dynamic_decay": float(getattr(config, 'leverage_dynamic_decay', 0.0)),
                "liq_stress_coeff": float(getattr(config, 'liq_stress_coeff', 0.0)),
                "entropy_sigma_lambda": float(getattr(config, 'entropy_sigma_lambda', 0.0)),
                # GARCH-Kalman reconciliation + Q_t coupling + location bias (February 2026)
                "garch_kalman_weight": float(getattr(config, 'garch_kalman_weight', 0.0)),
                "q_vol_coupling": float(getattr(config, 'q_vol_coupling', 0.0)),
                "loc_bias_var_coeff": float(getattr(config, 'loc_bias_var_coeff', 0.0)),
                "loc_bias_drift_coeff": float(getattr(config, 'loc_bias_drift_coeff', 0.0)),
                # Stage 6: pre-calibrated walk-forward CV params
                "calibrated_gw": float(getattr(config, 'calibrated_gw', 0.50)),
                "calibrated_nu_pit": float(getattr(config, 'calibrated_nu_pit', 0.0)),
                "calibrated_beta_probit_corr": float(getattr(config, 'calibrated_beta_probit_corr', 1.0)),
                "calibrated_lambda_rho": float(getattr(config, 'calibrated_lambda_rho', 0.985)),
                "calibrated_nu_crps": float(getattr(config, 'calibrated_nu_crps', 0.0)),
                # Scores
                "log_likelihood": float(ll_u),
                "mean_log_likelihood": float(mean_ll_u),
                "bic": float(bic_u),
                "aic": float(aic_u),
                "hyvarinen_score": float(hyvarinen_u),
                "crps": float(crps_u),
                "ks_statistic": float(ks_u),
                "pit_ks_pvalue": float(pit_p_u),
                "ad_pvalue": float(_ad_p_u),
                "ad_pvalue_raw": float(pit_metrics.get("ad_pvalue_raw", _ad_p_u) if isinstance(pit_metrics, dict) else _ad_p_u),
                "ad_correction": pit_metrics.get("ad_correction", {}) if isinstance(pit_metrics, dict) else {},
                "calibration_params": pit_metrics.get("ad_correction", {}).get("calibration_params", {}) if isinstance(pit_metrics, dict) else {},
                "pit_calibration_grade": pit_metrics.get("calibration_grade", "F"),
                "histogram_mad": float(_calibrated_mad_u),
                "berkowitz_pvalue": float(_calibrated_berk_u),
                "berkowitz_lr": float(_calibrated_berk_lr_u),
                "pit_count": int(_calibrated_pit_count_u),
                # Hansen/CST pipeline augmentations (March 2026)
                "hansen_activated": _u_hansen_activated,
                "hansen_lambda": float(_u_hansen_lambda),
                "hansen_diagnostics": _u_hansen_diag,
                "cst_activated": _u_cst_activated,
                "cst_nu_crisis": float(_u_cst_nu_crisis) if _u_cst_nu_crisis is not None else None,
                "cst_epsilon": float(_u_cst_epsilon),
                "cst_diagnostics": _u_cst_diag,
                # Metadata
                "fit_success": True,
                "unified_model": True,
                "degraded": diagnostics.get("degraded", False),
                "hessian_cond": diagnostics.get("hessian_cond", float('inf')),
                "n_params": int(effective_n_params),
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
    # Model 2b: UNIFIED Gaussian (Feb 2026 - nu-free Calibration Pipeline)
    # =========================================================================
    # Includes internal momentum integration (Stage 1.5) and GAS-Q adaptive
    # process noise (Stage 4.5), both with degradation guards.
    # Momentum and GAS-Q are tuned INSIDE optimize_params_unified — not as
    # separate external model variants.
    # =========================================================================
    n_params_gaussian_unified = 8  # base params; momentum/GAS-Q add 1+3 conditionally

    # Pre-compute momentum signal for unified Gaussian (reused for both phi modes)
    _gu_momentum_signal = None
    if MOMENTUM_AUGMENTATION_AVAILABLE and MOMENTUM_AUGMENTATION_ENABLED:
        try:
            _gu_mom_features = compute_momentum_features(returns)
            _gu_momentum_signal = compute_momentum_signal(_gu_mom_features)
        except Exception:
            _gu_momentum_signal = None

    for phi_mode, model_prefix in [(False, "kalman_gaussian_unified"),
                                    (True, "kalman_phi_gaussian_unified")]:
        try:
            g_config, g_diag = GaussianDriftModel.optimize_params_unified(
                returns, vol, phi_mode=phi_mode, train_frac=0.7,
                asset_symbol=asset, momentum_signal=_gu_momentum_signal)
            if not g_diag.get("success", False):
                raise ValueError("Gaussian unified: " + str(g_diag.get("error", "?")))

            # Reconstruct momentum signal for filter_and_calibrate
            _gu_mom_for_calib = _gu_momentum_signal if g_config.momentum_enabled else None

            _, _, mu_pred_gu, S_pred_gu, ll_gu = GaussianDriftModel._filter_phi_with_momentum(
                returns, vol, float(g_config.q), float(g_config.c), float(g_config.phi),
                _gu_mom_for_calib, float(g_config.momentum_weight))
            _pit_gu, _pit_p_gu, _sigma_gu, _crps_gu, _calib_gu = GaussianDriftModel.filter_and_calibrate(
                returns, vol, g_config, train_frac=0.7,
                momentum_signal=_gu_mom_for_calib,
                mu_pred_precomputed=mu_pred_gu,
                S_pred_precomputed=S_pred_gu,
                ll_precomputed=ll_gu)
            pit_p_gu = float(_pit_p_gu) if np.isfinite(_pit_p_gu) else 0.0
            _ad_p_gu = float(_calib_gu.get("ad_pvalue", float('nan')))
            _berk_p_gu = float(_calib_gu.get("berkowitz_pvalue", 0.0))
            _berk_lr_gu = float(_calib_gu.get("berkowitz_lr", 0.0))
            _pit_count_gu = int(_calib_gu.get("pit_count", 0))
            _mad_gu = float(_calib_gu.get("mad", 1.0))

            # Parameter count: base 8 + momentum (1) + GAS-Q (3) when active
            _eff_n_params_gu = n_params_gaussian_unified
            if g_config.momentum_enabled:
                _eff_n_params_gu += 1
            if g_config.gas_q_enabled:
                _eff_n_params_gu += 3

            aic_gu = compute_aic(ll_gu, _eff_n_params_gu)
            bic_gu = compute_bic(ll_gu, _eff_n_params_gu, n_obs)
            n_train_gu = int(n_obs * 0.7)
            returns_test_gu = returns[n_train_gu:]
            mu_eff_gu = _calib_gu.get("mu_effective", mu_pred_gu[n_train_gu:])
            if isinstance(mu_eff_gu, np.ndarray) and len(mu_eff_gu) == len(returns_test_gu):
                forecast_std_gu = np.maximum(_sigma_gu, 1e-10)
            else:
                forecast_std_gu = np.sqrt(np.maximum(S_pred_gu[n_train_gu:], 1e-20))
                mu_eff_gu = mu_pred_gu[n_train_gu:]
            hyvarinen_gu = compute_hyvarinen_score_gaussian(returns_test_gu, mu_eff_gu, forecast_std_gu)
            crps_final_gu = compute_crps_gaussian_inline(returns_test_gu, mu_eff_gu, forecast_std_gu)
            _ks_stat_gu = 1.0
            if hasattr(_pit_gu, "__len__") and len(_pit_gu) > 0:
                _ps_gu = np.sort(np.asarray(_pit_gu))
                _n_ps_gu = len(_ps_gu)
                _dp_gu = np.max(np.arange(1, _n_ps_gu + 1) / _n_ps_gu - _ps_gu)
                _dm_gu = np.max(_ps_gu - np.arange(0, _n_ps_gu) / _n_ps_gu)
                _ks_stat_gu = float(max(_dp_gu, _dm_gu))
            models[model_prefix] = {
                "q": float(g_config.q), "c": float(g_config.c), "phi": float(g_config.phi),
                "variance_inflation": float(g_config.variance_inflation),
                "mu_drift": float(g_config.mu_drift),
                "garch_omega": float(g_config.garch_omega),
                "garch_alpha": float(g_config.garch_alpha),
                "garch_beta": float(g_config.garch_beta),
                "garch_leverage": float(g_config.garch_leverage),
                "garch_unconditional_var": float(g_config.garch_unconditional_var),
                "crps_ewm_lambda": float(g_config.crps_ewm_lambda),
                "crps_sigma_shrinkage": float(g_config.crps_sigma_shrinkage),
                # v7.8 elite MC enhancements
                "leverage_dynamic_decay": float(getattr(g_config, 'leverage_dynamic_decay', 0.0)),
                "liq_stress_coeff": float(getattr(g_config, 'liq_stress_coeff', 0.0)),
                "entropy_sigma_lambda": float(getattr(g_config, 'entropy_sigma_lambda', 0.0)),
                "calibrated_gw": float(g_config.calibrated_gw),
                "calibrated_lambda_rho": float(g_config.calibrated_lambda_rho),
                "calibrated_beta_probit_corr": float(g_config.calibrated_beta_probit_corr),
                # Momentum integration (Stage 1.5)
                "momentum_weight": float(g_config.momentum_weight),
                "momentum_augmented": g_config.momentum_enabled,
                # GAS-Q adaptive process noise (Stage 4.5)
                "gas_q_omega": float(g_config.gas_q_omega),
                "gas_q_alpha": float(g_config.gas_q_alpha),
                "gas_q_beta": float(g_config.gas_q_beta),
                "gas_q_enabled": g_config.gas_q_enabled,
                # Scores
                "log_likelihood": float(ll_gu),
                "mean_log_likelihood": float(ll_gu / max(n_obs, 1)),
                "bic": float(bic_gu), "aic": float(aic_gu),
                "hyvarinen_score": float(hyvarinen_gu),
                "crps": float(crps_final_gu),
                "ks_statistic": _ks_stat_gu,
                "pit_ks_pvalue": float(pit_p_gu),
                "ad_pvalue": float(_ad_p_gu),
                "ad_pvalue_raw": float(_calib_gu.get("ad_pvalue_raw", _ad_p_gu) if isinstance(_calib_gu, dict) else _ad_p_gu),
                "ad_correction": _calib_gu.get("ad_correction", {}) if isinstance(_calib_gu, dict) else {},
                "calibration_params": _calib_gu.get("ad_correction", {}).get("calibration_params", {}) if isinstance(_calib_gu, dict) else {},
                "pit_calibration_grade": "A" if _mad_gu < 0.02 else ("B" if _mad_gu < 0.05 else ("C" if _mad_gu < 0.10 else "F")),
                "histogram_mad": float(_mad_gu),
                "berkowitz_pvalue": float(_berk_p_gu),
                "berkowitz_lr": float(_berk_lr_gu),
                "pit_count": int(_pit_count_gu),
                "fit_success": True, "unified_model": True,
                "gaussian_unified": True,
                "phi_mode": phi_mode,
                "n_params": _eff_n_params_gu,
                "model_type": "gaussian_unified",
            }
        except Exception as e:
            models[model_prefix] = {
                "fit_success": False, "error": str(e),
                "bic": float('inf'), "aic": float('inf'),
                "hyvarinen_score": float('-inf'), "crps": float('inf'),
                "unified_model": True, "gaussian_unified": True,
            }

    # =========================================================================
    # MOMENTUM AUGMENTED STUDENT-T — RETIRED AS SEPARATE MODELS (Feb 2026)
    # =========================================================================
    # Momentum augmentation is now an INTERNAL pipeline step within each
    # phi_student_t_nu_X base model (see Model 1 block above).
    # Activated ONLY if CRPS improves. No separate _momentum model names.
    #
    # Legacy Gaussian momentum models similarly retired — functionality
    # subsumed by unified Gaussian models with internal momentum + GAS-Q.
    # =========================================================================
    
    # =========================================================================
    # LEGACY MODEL GRIDS REMOVED (March 2026)
    # =========================================================================
    # GAS-Q Student-t, MS-q Student-t, VoV, Two-Piece, and Two-Component
    # Mixture grids were all gated by `not UNIFIED_STUDENT_T_ONLY` (never True
    # in production). Functionality is now incorporated internally by unified
    # models via Stages 7.x / U-x pipeline.
    # =========================================================================

    # =========================================================================
    # Phi ACF Lower Bound Enforcement (Story 1.8)
    # =========================================================================
    # Apply the autocorrelation-based phi floor to all fitted models.
    # Store acf_1 and phi_acf_floor in model diagnostics.
    # =========================================================================
    for _mn, _md in models.items():
        if not _md.get("fit_success", False):
            continue
        _md["acf_1"] = _acf_1
        _md["phi_acf_floor"] = _phi_acf_floor
        _phi_val = _md.get("phi")
        if _phi_val is not None and _phi_val < _phi_acf_floor:
            _md["phi_original_pre_acf_floor"] = _phi_val
            _md["phi"] = _phi_acf_floor

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
        # Step 1b: Apply regime-conditional q floor (Story 1.1, April 2026)
        # Story 1.7: Vol-proportional q floor
        # =====================================================================
        _asset_vol_ann = float(np.median(vol_regime)) * np.sqrt(252) if len(vol_regime) > 0 else 0.0
        n_floored, n_total = apply_regime_q_floor(
            models, regime, ret_regime, vol_regime,
            asset_vol=_asset_vol_ann,
        )
        _q_floor_val = compute_vol_proportional_q_floor(regime, _asset_vol_ann) if _asset_vol_ann > 0 else Q_FLOOR_BY_REGIME.get(regime, 0)
        if n_floored > 0:
            _log(f"     Q-floor applied: {n_floored}/{n_total} models "
                 f"(floor={_q_floor_val:.1e} for {regime_name}, vol={_asset_vol_ann:.2%})")

        # =====================================================================
        # Step 2: Extract BIC, Hyvärinen, CRPS, PIT and compute LFO-CV scores
        # =====================================================================
        bic_values = {m: models[m].get("bic", float('inf')) for m in models}
        hyvarinen_scores = {m: models[m].get("hyvarinen_score", float('-inf')) for m in models}
        crps_values = {m: models[m].get("crps", float('inf')) for m in models if models[m].get("crps") is not None}
        # February 2026 - Elite PIT calibration: extract PIT p-values for regime-aware scoring
        pit_pvalues = {m: models[m].get("pit_ks_pvalue") for m in models if models[m].get("pit_ks_pvalue") is not None}
        berk_pvalues = {m: models[m]["berkowitz_pvalue"] for m in models
                        if models[m].get("fit_success", False)
                        and models[m].get("berkowitz_pvalue") is not None}
        mad_values = {m: models[m]["histogram_mad"] for m in models
                      if models[m].get("fit_success", False)
                      and models[m].get("histogram_mad") is not None}
        berk_lr_regime = {m: models[m]["berkowitz_lr"] for m in models
                         if models[m].get("fit_success", False)
                         and models[m].get("berkowitz_lr") is not None}
        pit_count_regime = {m: models[m]["pit_count"] for m in models
                           if models[m].get("fit_success", False)
                           and models[m].get("pit_count") is not None}
        ad_pvalues_regime = {m: models[m]["ad_pvalue"] for m in models
                            if models[m].get("fit_success", False)
                            and models[m].get("ad_pvalue") is not None
                            and np.isfinite(models[m]["ad_pvalue"])}
        
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
        # Model selection uses elite CRPS-dominated 6-component scoring:
        #   Score = w_crps × CRPS_std + w_pit × PIT_dev_std + w_berk × Berk_std
        #         + w_tail × Tail_std + w_mad × MAD_std + w_ad × AD_dev_std
        #
        # All components are robustly standardized via median/MAD (winsorized ±5σ).
        # BIC and Hyvärinen are stored as metadata but NOT part of selection score.
        # Regime-specific weight profiles adjust component emphasis (see diagnostics.py).
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
                berk_pvalues=berk_pvalues,
                ad_pvalues=ad_pvalues_regime,  # AD veto gate
                berkowitz_lr_stats=berk_lr_regime, pit_counts=pit_count_regime,
                mad_values=mad_values,
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
                berk_pvalues=berk_pvalues,
                ad_pvalues=ad_pvalues_regime,  # AD veto gate
                berkowitz_lr_stats=berk_lr_regime, pit_counts=pit_count_regime,
                mad_values=mad_values,
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

    # Apply global q floor (Story 1.1): use regime-count-weighted average floor
    # This ensures global fallback models also have meaningful drift sensitivity
    if regime_labels is not None and len(regime_labels) > 0:
        _regime_counts = np.bincount(regime_labels.astype(int), minlength=5)
        _regime_total = max(_regime_counts.sum(), 1)
        _global_q_floor = sum(
            Q_FLOOR_BY_REGIME.get(r, 0) * _regime_counts[r] / _regime_total
            for r in range(5)
        )
    else:
        # No regime info: use median floor
        _global_q_floor = 5e-5  # median of Q_FLOOR_BY_REGIME values

    _n_fl_global, _n_total_global = 0, 0
    for _gm_name, _gm_info in global_models.items():
        if not _gm_info.get("fit_success", False):
            continue
        _n_total_global += 1
        _gm_q = _gm_info.get("q")
        if _gm_q is not None and _gm_q < _global_q_floor:
            _n_fl_global += 1
            _gm_info["q_mle_original"] = float(_gm_q)
            _gm_info["q"] = float(_global_q_floor)
            _gm_info["q_floor_applied"] = True
            _gm_info["q_floor_value"] = float(_global_q_floor)
        else:
            _gm_info["q_floor_applied"] = False

    if _n_fl_global > 0:
        _log(f"     Global Q-floor applied: {_n_fl_global}/{_n_total_global} models "
             f"(weighted floor={_global_q_floor:.1e})")

    # Compute global model posterior using elite CRPS-dominated scoring
    global_bic = {m: global_models[m].get("bic", float('inf')) for m in global_models if global_models[m].get("fit_success", False)}
    global_hyvarinen = {m: global_models[m].get("hyvarinen_score", float('-inf')) for m in global_models if global_models[m].get("fit_success", False)}
    global_crps = {m: global_models[m]["crps"] for m in global_models
                   if global_models[m].get("fit_success", False) and global_models[m].get("crps") is not None and np.isfinite(global_models[m]["crps"])}
    global_pit = {m: global_models[m]["pit_ks_pvalue"] for m in global_models
                  if global_models[m].get("fit_success", False) and global_models[m].get("pit_ks_pvalue") is not None}
    global_berk = {m: global_models[m]["berkowitz_pvalue"] for m in global_models
                   if global_models[m].get("fit_success", False) and global_models[m].get("berkowitz_pvalue") is not None}
    global_berk_lr = {m: global_models[m]["berkowitz_lr"] for m in global_models
                      if global_models[m].get("fit_success", False) and global_models[m].get("berkowitz_lr") is not None}
    global_pit_counts = {m: global_models[m]["pit_count"] for m in global_models
                         if global_models[m].get("fit_success", False) and global_models[m].get("pit_count") is not None}
    global_mad = {m: global_models[m]["histogram_mad"] for m in global_models
                  if global_models[m].get("fit_success", False) and global_models[m].get("histogram_mad") is not None}
    global_ad = {m: global_models[m]["ad_pvalue"] for m in global_models
                 if global_models[m].get("fit_success", False) and global_models[m].get("ad_pvalue") is not None
                 and np.isfinite(global_models[m]["ad_pvalue"])}
    fallback_weight_metadata = None
    
    if global_crps and CRPS_SCORING_ENABLED:
        global_raw_weights, fallback_weight_metadata = compute_regime_aware_model_weights(
            global_bic, global_hyvarinen, global_crps,
            pit_pvalues=global_pit, berk_pvalues=global_berk,
            ad_pvalues=global_ad,  # AD veto gate
            berkowitz_lr_stats=global_berk_lr, pit_counts=global_pit_counts,
            mad_values=global_mad, regime=None,
            lambda_entropy=DEFAULT_ENTROPY_LAMBDA
        )
    elif model_selection_method == 'bic':
        # Use vectorized BMA weights when available (log-sum-exp stable)
        if VECTORIZED_OPS_AVAILABLE and global_bic:
            model_names = list(global_bic.keys())
            bic_arr = np.array([global_bic[m] for m in model_names])
            w_arr = vectorized_bma_weights(bic_arr)
            global_raw_weights = dict(zip(model_names, w_arr.tolist()))
        else:
            global_raw_weights = compute_bic_model_weights(global_bic)
    elif model_selection_method == 'hyvarinen':
        global_raw_weights = compute_hyvarinen_model_weights(global_hyvarinen)
    else:
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
            crps_std_val = fallback_weight_metadata.get('crps_standardized', {}).get(m)
            global_models[m]['standardized_crps'] = float(crps_std_val) if crps_std_val is not None else None
            scoring_weights = fallback_weight_metadata.get('weights_used', {})
            global_models[m]['scoring_weights'] = dict(scoring_weights)
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
        
        # Adaptive data quality filter (February 2026)
        df, _dq_report = adaptive_data_quality(df, asset=asset, verbose=(not _is_quiet()))
        if _dq_report.get('rows_purged_leading', 0) > 0 or _dq_report.get('window_applied', False):
            _log(f"     🔬  Data quality: {_dq_report['rows_original']} → {_dq_report['rows_final']} rows")
        
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
        _volume_arr = None  # For Volume-based stale filter
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
                    
                    # Extract Volume for stale-price detection (February 2026)
                    _vol_col = cols.get('volume')
                    if _vol_col is not None:
                        _volume_arr = df_aligned[_vol_col].values
                    
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

        # Remove NaN/Inf and stale-price observations (zero-return days)
        # Also filter Volume=0 phantom quotes (February 2026)
        _STALE_RETURN_THRESHOLD = 1e-10
        valid_mask = (np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
                      & (np.abs(returns) > _STALE_RETURN_THRESHOLD))
        # Add Volume >= 100 filter if Volume data available (February 2026)
        # Volume < 100 indicates phantom OTC quotes without genuine price discovery
        # Skip for FX pairs (=X) and indices (^) — Yahoo reports Volume=0
        _MIN_GENUINE_VOLUME = 100
        _skip_vol = (asset.endswith('=X') or asset.startswith('^')) if asset else False
        if _volume_arr is not None and not _skip_vol:
            _vol_aligned = _volume_arr[:min_len]
            _vol_mask = _vol_aligned >= _MIN_GENUINE_VOLUME
            n_zero_vol = int(np.sum(~_vol_mask & valid_mask))
            if n_zero_vol > 0:
                _log(f"     🧹  Filtered {n_zero_vol} additional low-volume phantom rows (Volume<{_MIN_GENUINE_VOLUME})")
            valid_mask = valid_mask & _vol_mask
        n_stale = int(np.sum(np.abs(returns) <= _STALE_RETURN_THRESHOLD))
        if n_stale > 0:
            _log(f"     🧹  Filtered {n_stale}/{len(returns)} stale-price rows ({100*n_stale/len(returns):.1f}%)")
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

        # Assign regime labels (with optional computation cache)
        _log(f"     Assigning regime labels for {len(returns)} observations...")
        if COMPUTATION_CACHE_AVAILABLE and _computation_cache is not None:
            regime_labels = _computation_cache.get_regime(
                asset, returns, lambda r: assign_regime_labels(r, vol)
            )
        else:
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

        # ─────────────────────────────────────────────────────────────
        # CALIBRATION PARAMS PROMOTION (March 2026)
        # Extract BMA-weighted calibration params from per-model results
        # and promote to global level for signals.py consumption.
        #
        # Uses BMA-weighted aggregation across all models that produced
        # calibration params. This ensures the correction reflects the
        # model ensemble, not just a single model.
        # ─────────────────────────────────────────────────────────────
        try:
            _bma_models = bma_result.get("global", {}).get("models", {})
            _bma_posterior = bma_result.get("global", {}).get("model_posterior", {})
            _agg_cal = {}
            _total_w = 0.0
            for _m_name, _m_data in _bma_models.items():
                _m_cal = _m_data.get("calibration_params", {})
                _m_w = _bma_posterior.get(_m_name, 0.0)
                if _m_cal and _m_w > 0:
                    _total_w += _m_w
                    for _k, _v in _m_cal.items():
                        if isinstance(_v, (int, float)):
                            _agg_cal[_k] = _agg_cal.get(_k, 0.0) + _m_w * _v
                        elif isinstance(_v, list) and _k not in _agg_cal:
                            # For isotonic knots, take from highest-weight model
                            _agg_cal[_k] = _v
            # Normalize weighted averages
            if _total_w > 0:
                for _k in _agg_cal:
                    if isinstance(_agg_cal[_k], (int, float)):
                        _agg_cal[_k] = _agg_cal[_k] / _total_w
            if _agg_cal:
                result["global"]["calibration_params"] = _agg_cal
        except Exception:
            pass  # Calibration param promotion is best-effort

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
    # Bayesian regularization parameters
    parser.add_argument('--prior-mean', type=float, default=-6.0,
                       help='Prior mean for log10(q) (default: -6.0)')
    parser.add_argument('--prior-lambda', type=float, default=1.0,
                       help='Regularization strength (default: 1.0, set to 0 to disable)')

    # Hierarchical regime tuning parameters
    parser.add_argument('--lambda-regime', type=float, default=0.05,
                       help='Hierarchical shrinkage toward global (default: 0.05, set to 0 for original behavior)')

    # Story 2.6: Calibration pipeline
    parser.add_argument('--calibrate', action='store_true',
                       help='Run walk-forward calibration pipeline instead of tuning')

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
    print("Models: Gaussian, φ-Gaussian, φ-Student-t (ν ∈ {4, 8, 20})")
    if MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE:
        print("Momentum: ENABLED")
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

    # Story 2.6: Calibration pipeline mode
    if args.calibrate:
        print("\n" + "=" * 60)
        print("CALIBRATION PIPELINE (Story 2.6)")
        print("=" * 60)
        report = run_calibration_pipeline(
            assets, cache, cache_json=args.cache_json,
            start_date=CALIBRATION_DEFAULT_START,
        )
        # Save report
        path = save_calibration_report(report)
        print(f"\nCalibration report saved to: {path}")
        # Save updated cache
        save_cache_json(cache, args.cache_json)
        # Summary
        s = report["summary"]
        print(f"\nSummary: {s['success']}/{s['total_assets']} success, "
              f"{s['failed']} failed, {s['skipped']} skipped, "
              f"{s['n_nontrivial_emos']} non-trivial EMOS")
        return

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
                                'best_model': global_result.get('best_model', global_result.get('noise_model', 'unknown')),
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

    # ==================================================================
    # CROSS-ASSET PHI POOLING (Story 1.3)
    # ==================================================================
    # After all assets are tuned independently, apply hierarchical
    # shrinkage on phi values toward the cross-asset population median.
    # This corrects pathological phi estimates (negative, near-zero)
    # from short regime windows while preserving well-estimated values.
    # ==================================================================
    if cache and len(cache) >= PHI_POOL_MIN_ASSETS:
        print("\nApplying cross-asset phi pooling...")
        cache = apply_cross_asset_phi_pooling(cache)
        _pool_sample = next(iter(cache.values()), {})
        _pool_meta = _pool_sample.get("hierarchical_tuning", {}).get("phi_prior", {})
        if _pool_meta:
            print(f"  Population phi median: {_pool_meta.get('phi_population_median', 'N/A'):.4f}")
            print(f"  Population phi std:    {_pool_meta.get('phi_population_std', 'N/A'):.4f}")
            print(f"  Assets pooled:         {_pool_meta.get('n_assets_pooled', 0)}")
            print(f"  Assets shrunk:         {_pool_meta.get('n_assets_shrunk', 0)}")

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
    print(f"  φ-Student-t:          {student_t_count} (discrete ν ∈ {{4, 8, 20}})")
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
            best_model = data.get('best_model', data.get('noise_model', 'kalman_drift'))

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
            best_bic = mc.get('best_model', mc.get('noise_model', 'unknown'))
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
