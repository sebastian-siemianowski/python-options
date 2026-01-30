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
                              phi_student_t_nu_{4,6,8,12,20})
    θ_{r,m}    = parameters of model m in regime r
    p(m | r)   = posterior probability of model m in regime r

-------------------------------------------------------------------------------
WHAT THIS FILE DOES

For EACH regime r:

    1. Fits ALL candidate model classes m independently:
       - kalman_gaussian:       q, c           (2 params)
       - kalman_phi_gaussian:   q, c, φ        (3 params)
       - phi_student_t_nu_4:    q, c, φ        (3 params, ν=4 FIXED)
       - phi_student_t_nu_6:    q, c, φ        (3 params, ν=6 FIXED)
       - phi_student_t_nu_8:    q, c, φ        (3 params, ν=8 FIXED)
       - phi_student_t_nu_12:   q, c, φ        (3 params, ν=12 FIXED)
       - phi_student_t_nu_20:   q, c, φ        (3 params, ν=20 FIXED)

    NOTE: Student-t uses DISCRETE ν GRID (not continuous optimization).
    Each ν is treated as a separate sub-model in BMA.

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

from data_utils import fetch_px, _download_prices, get_default_asset_universe

# K=2 Mixture Model REMOVED - empirically falsified (206 attempts, 0 selections)
# The HMM regime-switching + Student-t already captures regime heterogeneity.
# See: docs/CALIBRATION_SOLUTIONS_ANALYSIS.md for decision rationale.
MIXTURE_MODEL_AVAILABLE = False

# Import Adaptive ν Refinement for calibration improvement
try:
    from adaptive_nu_refinement import (
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
    from gh_distribution import (
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
    from tvvm_model import (
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
    from isotonic_recalibration import (
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
    from calibrated_trust import (
        CalibratedTrust,
        TrustConfig,
        compute_calibrated_trust,
        compute_drift_weight,
        create_isotonic_transport,
        MAX_REGIME_PENALTY,
        DEFAULT_REGIME_PENALTY_SCHEDULE,
        REGIME_NAMES,
        verify_trust_architecture,
    )
    CALIBRATED_TRUST_AVAILABLE = True
except ImportError:
    CALIBRATED_TRUST_AVAILABLE = False

# Import presentation layer for world-class UX output
from signals_ux import (
    create_tuning_console,
    render_tuning_header,
    render_tuning_progress_start,
    render_tuning_summary,
    render_parameter_table,
    render_failed_assets,
    render_dry_run_preview,
    render_cache_status,
    render_cache_update,
    TuningProgressTracker,
    TUNING_REGIME_LABELS,
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

# =============================================================================
# DISCRETE ν GRID FOR STUDENT-T MODELS
# =============================================================================
# CRITICAL: Eliminates ν-σ identifiability failures and tail-driven weight collapse
#
# Instead of continuously optimizing ν (which is unstable), we:
#   1. Fix a discrete grid of ν values
#   2. Treat each ν as a separate sub-model in BMA
#   3. Let BMA handle the weighting across ν values
#
# This is standard practice in Bayesian model selection for discrete hyperparameters.
# Grid values chosen to span light tails (ν=20, near Gaussian) to heavy tails (ν=4).
# =============================================================================

STUDENT_T_NU_GRID = [4, 6, 8, 12, 20]

# =============================================================================
# φ SHRINKAGE PRIOR — EXPLICIT GAUSSIAN PRIOR FOR AR(1) COEFFICIENT
# =============================================================================
# φ (phi) is the AR(1) mean-reversion parameter in Phi-Gaussian and Phi-Student-t
# models. It controls drift persistence: φ=1 is unit root (random walk), φ=0 is
# full mean reversion.
#
# PROBLEM: φ is weakly identified in small samples and highly unstable near unit
# root. Unconstrained MLE leads to spurious mean reversion or explosive estimates.
#
# SOLUTION: Gaussian shrinkage prior toward φ_global (typically 0):
#
#     φ_r ~ N(φ_global, τ²)
#
#     log p(φ_r) = -0.5 * (φ_r - φ_global)² / τ²
#
# The prior is added to the log-likelihood in the optimization objective.
#
# MAPPING FROM LEGACY λ TO τ:
#     The legacy regularization used: λ * (φ_r - φ_global)²
#     This is equivalent to the Gaussian prior when: τ = 1 / √(2λ)
#
# NUMERICAL SAFETY:
#     - τ must be bounded below to prevent division instability
#     - φ_global must be finite
#     - φ bounds in optimizer remain unchanged (safety net)
#
# AUDITABILITY:
#     - Prior contribution is logged separately in diagnostics
#     - Enables future contributors to understand and modify shrinkage
#     - Prevents accidental removal of regularization
#
# CONTRACT:
#     - _select_regime_params() remains unchanged
#     - bayesian_model_average_mc() unchanged
#     - Signal dataclasses receive phi_used exactly as before
#     - This change is purely epistemic (refactor, not behavioral change)
# =============================================================================

# Minimum τ to prevent numerical instability (division by τ²)
PHI_SHRINKAGE_TAU_MIN = 1e-3

# Default φ_global: center of shrinkage prior (0 = full mean reversion)
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0

# Default prior strength: τ corresponds to legacy λ=0.05 via τ = 1/√(2λ)
# λ=0.05 → τ = 1/√(0.1) ≈ 3.16 (very weak prior, barely constrains φ)
# We use the scaled version: λ_effective = 0.05 * prior_scale
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05


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


def phi_shrinkage_log_prior(
    phi_r: float,
    phi_global: float,
    tau: float,
    tau_min: float = PHI_SHRINKAGE_TAU_MIN
) -> float:
    """
    Compute Gaussian shrinkage log-prior for φ (AR(1) coefficient).
    
    Implements:
        log p(φ_r) = -0.5 * (φ_r - φ_global)² / τ²
    
    This prior shrinks regime-specific φ_r toward a global φ_global,
    preventing unit-root instability and small-sample hallucinations.
    
    Args:
        phi_r: Regime-specific AR(1) coefficient being evaluated
        phi_global: Center of shrinkage prior (typically 0)
        tau: Prior standard deviation (larger = weaker shrinkage)
        tau_min: Minimum τ for numerical stability
        
    Returns:
        Log-prior contribution (negative, to be ADDED to log-likelihood)
        
    Mathematical note:
        If legacy code used λ*(φ-φ_g)², then τ = 1/√(2λ) gives equivalence.
        
    Safety:
        - τ is clamped to tau_min to prevent division by zero
        - Returns -inf if phi_global is not finite (should never happen)
    """
    # Safety: ensure τ is bounded below
    tau_safe = max(tau, tau_min)
    
    # Safety: ensure phi_global is finite
    if not np.isfinite(phi_global):
        return float('-inf')
    
    # Gaussian log-prior (up to constant)
    deviation = phi_r - phi_global
    log_prior = -0.5 * (deviation ** 2) / (tau_safe ** 2)
    
    return log_prior


def lambda_to_tau(lam: float, lam_min: float = 1e-12) -> float:
    """
    Convert legacy penalty weight λ to Gaussian prior std τ.
    
    Mapping derivation:
        Legacy: penalty = λ * (φ - φ_g)²
        Prior:  log p(φ) = -0.5 * (φ - φ_g)² / τ²
        
        Matching: λ = 0.5 / τ²  →  τ = 1 / √(2λ)
    
    Args:
        lam: Legacy penalty weight λ
        lam_min: Minimum λ to prevent division by zero
        
    Returns:
        Equivalent Gaussian prior std τ
    """
    lam_safe = max(lam, lam_min)
    return 1.0 / math.sqrt(2.0 * lam_safe)


def tau_to_lambda(tau: float, tau_min: float = PHI_SHRINKAGE_TAU_MIN) -> float:
    """
    Convert Gaussian prior std τ to legacy penalty weight λ.
    
    Inverse of lambda_to_tau().
    
    Args:
        tau: Gaussian prior std τ
        tau_min: Minimum τ for numerical stability
        
    Returns:
        Equivalent legacy penalty weight λ
    """
    tau_safe = max(tau, tau_min)
    return 0.5 / (tau_safe ** 2)


def compute_phi_prior_diagnostics(
    phi_r: float,
    phi_global: float,
    tau: float,
    log_likelihood: float
) -> dict:
    """
    Compute diagnostic information for φ shrinkage prior.
    
    Args:
        phi_r: Optimized regime-specific φ
        phi_global: Center of shrinkage prior
        tau: Prior standard deviation
        log_likelihood: Log-likelihood contribution (without prior)
        
    Returns:
        Dictionary with diagnostic fields:
        - phi_prior_logp: Log-prior contribution
        - phi_likelihood_logp: Log-likelihood (passed through)
        - phi_prior_likelihood_ratio: |prior| / |likelihood| (if computable)
        - phi_deviation_from_global: φ_r - φ_global
        - phi_tau_used: τ value used
    """
    log_prior = phi_shrinkage_log_prior(phi_r, phi_global, tau)
    
    # Compute ratio if both are finite and non-zero
    ratio = None
    if np.isfinite(log_prior) and np.isfinite(log_likelihood):
        abs_prior = abs(log_prior)
        abs_ll = abs(log_likelihood)
        if abs_ll > 1e-12:
            ratio = abs_prior / abs_ll
    
    return {
        'phi_prior_logp': float(log_prior) if np.isfinite(log_prior) else None,
        'phi_likelihood_logp': float(log_likelihood) if np.isfinite(log_likelihood) else None,
        'phi_prior_likelihood_ratio': float(ratio) if ratio is not None else None,
        'phi_deviation_from_global': float(phi_r - phi_global),
        'phi_tau_used': float(tau),
        'phi_global_used': float(phi_global),
    }


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

# Entropy regularization lambda for model weights
# Higher = more uniform weights, prevents premature posterior collapse
# Lower = sharper weights, stronger model discrimination
# 0.05 provides good balance between stability and discrimination
DEFAULT_ENTROPY_LAMBDA = 0.05

# Minimum weight fraction for entropy floor (prevents belief collapse)
# Total mass allocated uniformly across all models as a floor
# 0.01 = 1% total mass to uniform, each model gets at least 0.01/n_models weight
# This prevents overconfident allocations during regime transitions
DEFAULT_MIN_WEIGHT_FRACTION = 0.01


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
#   src/quant/cache/tune/{SYMBOL}.json
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
    from quant.kalman_cache import (
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
    Persist cache to per-asset files (preferred) or legacy single JSON file.
    
    When per-asset cache is available, each asset is saved to its own file.
    The cache_json path is used as fallback location.
    
    Args:
        cache: Dict mapping symbol -> params
        cache_json: Path to legacy cache file (used as fallback)
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
    
    # Use per-asset cache if available
    if PER_ASSET_CACHE_AVAILABLE:
        saved_count = 0
        for symbol, params in cache.items():
            try:
                _save_per_asset(symbol, params)
                saved_count += 1
            except Exception as e:
                print(f"Warning: Failed to save {symbol} to per-asset cache: {e}")
        
        # Also save to legacy file for backward compatibility during transition
        # (can be removed after full migration)
        try:
            os.makedirs(os.path.dirname(cache_json) if os.path.dirname(cache_json) else '.', exist_ok=True)
            json_temp = cache_json + '.tmp'
            with open(json_temp, 'w') as f:
                json.dump(cache, f, indent=2, cls=NumpyEncoder)
            os.replace(json_temp, cache_json)
        except Exception as e:
            print(f"Warning: Failed to save legacy cache (per-asset saved successfully): {e}")
        return
    
    # Fallback to legacy single-file cache
    os.makedirs(os.path.dirname(cache_json) if os.path.dirname(cache_json) else '.', exist_ok=True)
    json_temp = cache_json + '.tmp'
    with open(json_temp, 'w') as f:
        json.dump(cache, f, indent=2, cls=NumpyEncoder)
    os.replace(json_temp, cache_json)


class GaussianDriftModel:
    """Encapsulates Gaussian Kalman drift model logic for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run Kalman filter for drift estimation with fixed process noise q and observation variance scale c."""
        n = len(returns)

        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)

        mu = 0.0
        P = 1e-4

        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = float(mu)
            P_pred = float(P) + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            mu = float(mu_pred + K * innovation)
            P = float((1.0 - K) * P_pred)

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_var = P_pred + R
            if forecast_var > 1e-12:
                log_likelihood += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var

        return mu_filtered, P_filtered, log_likelihood

    @staticmethod
    def filter_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = phi_val * mu
            P_pred = (phi_val ** 2) * P + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            K = P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            P = float(max(P, 1e-12))

            mu_filtered[t] = mu
            P_filtered[t] = P

            log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def pit_ks(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
        """PIT/KS for Gaussian forecasts including parameter uncertainty.
        
        Computes the Probability Integral Transform (PIT) and performs a 
        Kolmogorov-Smirnov test for uniformity. If the model is well-calibrated,
        the PIT values should be uniformly distributed on [0, 1].
        
        Numerical stability: We enforce a minimum floor on forecast_std to prevent
        division by zero. When forecast_std is effectively zero, the model has
        collapsed to a degenerate distribution, which indicates calibration failure.
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Compute forecast standard deviation with numerical floor
        # The floor is set to 1e-10, which is small enough to not affect normal
        # market data (typical daily vol ~0.01-0.05) but prevents division by zero
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_std = np.sqrt(np.maximum(forecast_var, 1e-20))
        
        # Additional safety: ensure no zero values slip through
        forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
        
        standardized = (returns_flat - mu_flat) / forecast_std
        
        # Handle any remaining NaN/Inf values that could arise from edge cases
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            # All values invalid - return worst-case KS statistic
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        pit_values = norm.cdf(standardized_clean)
        
        # Need at least 2 points for KS test
        if len(pit_values) < 2:
            return 1.0, 0.0
            
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @classmethod
    def optimize_params(
        cls,
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
        """Jointly optimize (q, c) via maximum likelihood with enhanced Bayesian regularization."""
        n = len(returns)

        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))

        if vol_mean > 0:
            vol_cv = vol_std / vol_mean
        else:
            vol_cv = 0.0

        if ret_std > 0:
            rv_ratio = abs(ret_mean) / ret_std
        else:
            rv_ratio = 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))

        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window

        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        def negative_penalized_ll_cv(params: np.ndarray) -> float:
            log_q, log_c = params
            q = 10 ** log_q
            c = 10 ** log_c

            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
                return 1e12

            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []

            for train_start, train_end, test_start, test_end in fold_splits:
                try:
                    ret_train = returns_robust[train_start:train_end]
                    vol_train = vol[train_start:train_end]

                    if len(ret_train) < 3:
                        continue

                    mu_filt_train, P_filt_train, _ = cls.filter(returns_robust[train_start:train_end], vol_train, q, c)

                    mu_final = float(mu_filt_train[-1])
                    P_final = float(P_filt_train[-1])

                    ll_fold = 0.0
                    mu_pred = mu_final
                    P_pred = P_final

                    for t in range(test_start, test_end):
                        P_pred = P_pred + q

                        if np.ndim(returns_robust[t]) == 0:
                            ret_t = float(returns_robust[t])
                        else:
                            ret_t = float(returns_robust[t].item())
                        if np.ndim(vol[t]) == 0:
                            vol_t = float(vol[t])
                        else:
                            vol_t = float(vol[t].item())

                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            # Gaussian log-likelihood for φ-Gaussian model
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
                            standardized_innov = innovation / np.sqrt(forecast_var)
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(standardized_innov))
                            ll_fold += ll_contrib

                        K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (test_end - test_start)

                except Exception:
                    continue

            if total_obs == 0:
                return 1e12

            avg_ll_oos = total_ll_oos / max(total_obs, 1)

            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = norm.cdf(all_standardized)

                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)

                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)

                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)

                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass

            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2

            c_target = 0.9
            log_c_target = np.log10(c_target)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2

            penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + calibration_penalty

            if not np.isfinite(penalized_ll):
                return 1e12

            return -penalized_ll

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)

        log_q_grid = np.concatenate([
            np.linspace(log_q_min, adaptive_prior_mean - 1.0, 5),
            np.linspace(adaptive_prior_mean - 1.0, adaptive_prior_mean + 1.0, 7),
            np.linspace(adaptive_prior_mean + 1.0, log_q_max, 3)
        ])

        log_c_grid = np.concatenate([
            np.linspace(log_c_min, np.log10(0.7), 3),
            np.linspace(np.log10(0.7), np.log10(1.0), 7),
            np.linspace(np.log10(1.0), log_c_max, 2)
        ])

        best_neg_ll = float('inf')
        best_log_q_grid = adaptive_prior_mean
        best_log_c_grid = np.log10(0.9)

        for lq in log_q_grid:
            for lc in log_c_grid:
                try:
                    neg_ll = negative_penalized_ll_cv(np.array([lq, lc]))
                    if neg_ll < best_neg_ll:
                        best_neg_ll = neg_ll
                        best_log_q_grid = lq
                        best_log_c_grid = lc
                except Exception:
                    continue

        grid_best_q = 10 ** best_log_q_grid
        grid_best_c = 10 ** best_log_c_grid

        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max)]

        best_result = None
        best_fun = float('inf')

        start_points = [
            np.array([best_log_q_grid, best_log_c_grid]),
            np.array([adaptive_prior_mean, np.log10(0.9)]),
            np.array([adaptive_prior_mean, np.log10(0.7)]),
            np.array([adaptive_prior_mean, np.log10(1.2)]),
            np.array([best_log_q_grid - 0.5, best_log_c_grid]),
            np.array([best_log_q_grid + 0.5, best_log_c_grid]),
            np.array([best_log_q_grid, best_log_c_grid - 0.2]),
            np.array([best_log_q_grid, best_log_c_grid + 0.2]),
            np.array([-7.0, 0.0]),
            np.array([-5.0, 0.0]),
        ]

        for x0 in start_points:
            try:
                result = minimize(
                    negative_penalized_ll_cv,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-7}
                )

                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            log_q_opt, log_c_opt = best_result.x
            q_optimal = 10 ** log_q_opt
            c_optimal = 10 ** log_c_opt
            ll_optimal = -best_result.fun
        else:
            q_optimal = grid_best_q
            c_optimal = grid_best_c
            ll_optimal = -best_neg_ll

        diagnostics = {
            'grid_best_q': float(grid_best_q),
            'grid_best_c': float(grid_best_c),
            'refined_best_q': float(q_optimal),
            'refined_best_c': float(c_optimal),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'ret_mean': float(ret_mean),
            'ret_std': float(ret_std),
            'n_folds': int(len(fold_splits)),
            'adaptive_regularization': True,
            'robust_optimization': True,
            'winsorized': True,
            'optimization_successful': best_result is not None and (best_result.success if best_result else False)
        }

        return q_optimal, c_optimal, ll_optimal, diagnostics


class PhiGaussianDriftModel:
    """Encapsulates Gaussian Kalman drift with persistence φ for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        return GaussianDriftModel.filter_phi(returns, vol, q, c, phi)

    @classmethod
    def optimize_params(
        cls,
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
        n = len(returns)

        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))

        if vol_mean > 0:
            vol_cv = vol_std / vol_mean
        else:
            vol_cv = 0.0
        if ret_std > 0:
            rv_ratio = abs(ret_mean) / ret_std
        else:
            rv_ratio = 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))

        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window

        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        def negative_penalized_ll_cv_phi(params: np.ndarray) -> float:
            log_q, log_c, phi = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))

            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
                return 1e12

            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []

            for train_start, train_end, test_start, test_end in fold_splits:
                try:
                    ret_train = returns_robust[train_start:train_end]
                    vol_train = vol[train_start:train_end]

                    if len(ret_train) < 3:
                        continue

                    mu_filt_train, P_filt_train, _ = cls.filter(ret_train, vol_train, q, c, phi_clip)

                    mu_final = float(mu_filt_train[-1])
                    P_final = float(P_filt_train[-1])

                    ll_fold = 0.0
                    mu_pred = mu_final
                    P_pred = P_final

                    for t in range(test_start, test_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q

                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())

                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            # Gaussian log-likelihood for φ-Gaussian model
                            ll_contrib = -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
                            standardized_innov = innovation / np.sqrt(forecast_var)
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(standardized_innov))
                            ll_fold += ll_contrib

                        K = P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (test_end - test_start)

                except Exception:
                    continue

            if total_obs == 0:
                return 1e12

            avg_ll_oos = total_ll_oos / max(total_obs, 1)

            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = norm.cdf(all_standardized)

                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)

                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)

                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)

                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass

            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_c_target = np.log10(0.9)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2
            
            # =================================================================
            # EXPLICIT φ SHRINKAGE PRIOR (Gaussian)
            # =================================================================
            # φ_r ~ N(φ_global, τ²) where φ_global = 0 (full mean reversion)
            #
            # Legacy form: -λ_φ * prior_scale * φ² where λ_φ = 0.05
            # Explicit form: -0.5 * (φ - φ_global)² / τ²
            #
            # Mapping: τ = 1/√(2 * λ_φ * prior_scale)
            # This maintains numerical equivalence with legacy behavior.
            # =================================================================
            phi_lambda_effective = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale
            phi_tau = lambda_to_tau(phi_lambda_effective)
            log_prior_phi = phi_shrinkage_log_prior(
                phi_r=phi_clip,
                phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
                tau=phi_tau
            )

            penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + log_prior_phi + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)
        phi_grid = np.linspace(phi_min, phi_max, 5)
        log_q_grid = np.linspace(log_q_min, log_q_max, 4)
        log_c_grid = np.linspace(log_c_min, log_c_max, 3)

        best_neg_ll = float('inf')
        best_log_q_grid = adaptive_prior_mean
        best_log_c_grid = np.log10(0.9)
        best_phi_grid = 0.0

        for lq in log_q_grid:
            for lc in log_c_grid:
                for ph in phi_grid:
                    try:
                        neg_ll = negative_penalized_ll_cv_phi(np.array([lq, lc, ph]))
                        if neg_ll < best_neg_ll:
                            best_neg_ll = neg_ll
                            best_log_q_grid = lq
                            best_log_c_grid = lc
                            best_phi_grid = ph
                    except Exception:
                        continue

        grid_best_q = 10 ** best_log_q_grid
        grid_best_c = 10 ** best_log_c_grid
        grid_best_phi = float(np.clip(best_phi_grid, phi_min, phi_max))

        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max)]
        start_points = [
            np.array([best_log_q_grid, best_log_c_grid, best_phi_grid]),
            np.array([adaptive_prior_mean, np.log10(0.9), 0.0]),
            np.array([adaptive_prior_mean, np.log10(0.7), 0.3]),
            np.array([adaptive_prior_mean, np.log10(1.2), -0.3]),
            np.array([best_log_q_grid + 0.5, best_log_c_grid, best_phi_grid]),
            np.array([best_log_q_grid - 0.5, best_log_c_grid, best_phi_grid]),
            np.array([best_log_q_grid, best_log_c_grid + 0.2, best_phi_grid]),
            np.array([best_log_q_grid, best_log_c_grid - 0.2, best_phi_grid]),
        ]

        best_result = None
        best_fun = float('inf')

        for x0 in start_points:
            try:
                result = minimize(
                    negative_penalized_ll_cv_phi,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )

                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            log_q_opt, log_c_opt, phi_opt = best_result.x
            q_optimal = 10 ** log_q_opt
            c_optimal = 10 ** log_c_opt
            phi_optimal = float(np.clip(phi_opt, phi_min, phi_max))
            ll_optimal = -best_result.fun
        else:
            q_optimal = grid_best_q
            c_optimal = grid_best_c
            phi_optimal = grid_best_phi
            ll_optimal = -best_neg_ll

        # Compute φ shrinkage prior diagnostics for auditability
        # Use same prior_scale as in optimization (based on typical n_obs)
        n_obs_approx = len(returns)
        prior_scale_diag = 1.0 / max(n_obs_approx, 100)
        phi_lambda_eff_diag = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale_diag
        phi_tau_diag = lambda_to_tau(phi_lambda_eff_diag)
        phi_prior_diag = compute_phi_prior_diagnostics(
            phi_r=phi_optimal,
            phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
            tau=phi_tau_diag,
            log_likelihood=ll_optimal
        )

        diagnostics = {
            'grid_best_q': float(grid_best_q),
            'grid_best_c': float(grid_best_c),
            'refined_best_q': float(q_optimal),
            'refined_best_c': float(c_optimal),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_result is not None and (best_result.success if best_result else False),
            # φ shrinkage prior diagnostics (auditability)
            **phi_prior_diag,
        }

        return q_optimal, c_optimal, phi_optimal, ll_optimal, diagnostics


class PhiStudentTDriftModel:
    """Encapsulates Student-t heavy-tail logic so drift model behavior stays modular."""

    nu_min_default: float = 2.1
    nu_max_default: float = 30.0

    @staticmethod
    def _clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
        return float(np.clip(float(nu), nu_min, nu_max))

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        """
        Log-density of scaled Student-t with location ``mu`` and scale ``scale``.
        Returns a large negative sentinel if inputs are invalid to keep optimizers stable.
        """
        if scale <= 0 or nu <= 0:
            return -1e12

        z = (x - mu) / scale
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)

    @classmethod
    def filter(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman drift filter with Student-t observation noise (no AR persistence)."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = float(mu)
            P_pred = float(P) + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            obs_scale = np.sqrt(c_val) * vol_scalar

            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            R = c_val * (vol_scalar ** 2)
            K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            mu = float(mu_pred + K * innovation)
            P = float((1.0 - K) * P_pred)

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_scale = np.sqrt(P_pred + R)
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf(r_val, nu_val, mu_pred, forecast_scale)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t

        return mu_filtered, P_filtered, log_likelihood

    @classmethod
    def filter_phi(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman drift filter with persistence (phi) and Student-t observation noise."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = phi_val * mu
            P_pred = (phi_val ** 2) * P + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            K = nu_adjust * P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf(r_val, nu_val, mu_pred, forecast_scale)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def pit_ks(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
        """PIT/KS for Student-t forecasts with parameter uncertainty included.
        
        Uses the Student-t distribution CDF for the PIT transformation, which is
        more appropriate for heavy-tailed return distributions.
        
        Numerical stability: Enforces minimum floor on forecast_scale to prevent
        division by zero while preserving the statistical properties of the test.
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Compute forecast scale with numerical floor
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_scale = np.sqrt(np.maximum(forecast_var, 1e-20))
        
        # Additional safety: ensure no zero values slip through
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)
        
        standardized = (returns_flat - mu_flat) / forecast_scale
        
        # Handle any remaining NaN/Inf values
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        
        # Ensure nu is valid for Student-t (must be > 0)
        nu_safe = max(nu, 2.01)
        pit_values = student_t.cdf(standardized_clean, df=nu_safe)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
            
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
        """PIT/KS for Gaussian forecasts including parameter uncertainty.
        
        This is a Gaussian version used for comparison purposes within the
        PhiStudentTDriftModel class.
        
        Numerical stability: Enforces minimum floor on forecast_std.
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Compute forecast std with numerical floor
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_std = np.sqrt(np.maximum(forecast_var, 1e-20))
        forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
        
        standardized = (returns_flat - mu_flat) / forecast_std
        
        # Handle NaN/Inf
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        pit_values = norm.cdf(standardized_clean)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
            
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def optimize_params(
        returns: np.ndarray,
        vol: np.ndarray,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        phi_min: float = -0.999,
        phi_max: float = 0.999,
        nu_min: float = 2.1,
        nu_max: float = 30.0,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, float, float, Dict]:
        """Jointly optimize (q, c, φ, ν) for the φ-Student-t drift model via CV MLE."""
        n = len(returns)
        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0.0
        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        rv_ratio = abs(ret_mean) / ret_std if ret_std > 0 else 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window
        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        def neg_pen_ll(params: np.ndarray) -> float:
            log_q, log_c, phi, log_nu = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))
            nu = 10 ** log_nu
            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c) or nu < nu_min or nu > nu_max:
                return 1e12
            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []
            for tr_start, tr_end, te_start, te_end in fold_splits:
                try:
                    ret_train = returns_robust[tr_start:tr_end]
                    vol_train = vol[tr_start:tr_end]
                    if len(ret_train) < 3:
                        continue
                    mu_filt_train, P_filt_train, _ = PhiStudentTDriftModel.filter_phi(ret_train, vol_train, q, c, phi_clip, nu)
                    mu_pred = float(mu_filt_train[-1])
                    P_pred = float(P_filt_train[-1])
                    ll_fold = 0.0
                    for t in range(te_start, te_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q
                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())
                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            forecast_std = np.sqrt(forecast_var)
                            ll_contrib = PhiStudentTDriftModel.logpdf(ret_t, nu, mu_pred, forecast_std)
                            ll_fold += ll_contrib
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(innovation / forecast_std))

                        nu_adjust = min(nu / (nu + 3.0), 1.0)
                        K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (te_end - te_start)

                except Exception:
                    continue
            if total_obs == 0:
                return 1e12
            avg_ll = total_ll_oos / max(total_obs, 1)
            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = student_t.cdf(all_standardized, df=nu)
                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)
                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)
                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)
                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass
            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_c_target = np.log10(0.9)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2
            
            # =================================================================
            # EXPLICIT φ SHRINKAGE PRIOR (Gaussian)
            # =================================================================
            # φ_r ~ N(φ_global, τ²) where φ_global = 0 (full mean reversion)
            # See documentation at PHI_SHRINKAGE_* constants for derivation.
            # =================================================================
            phi_lambda_effective = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale
            phi_tau = lambda_to_tau(phi_lambda_effective)
            log_prior_phi = phi_shrinkage_log_prior(
                phi_r=phi_clip,
                phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
                tau=phi_tau
            )
            log_prior_nu = -0.05 * prior_scale * (log_nu - np.log10(6.0)) ** 2

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + log_prior_nu + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)
        log_nu_min = np.log10(nu_min)
        log_nu_max = np.log10(nu_max)

        grid_best = (adaptive_prior_mean, np.log10(0.9), 0.0, np.log10(6.0))
        best_neg = float('inf')
        for lq in np.linspace(log_q_min, log_q_max, 4):
            for lc in np.linspace(log_c_min, log_c_max, 3):
                for lp in np.linspace(phi_min, phi_max, 5):
                    for ln in np.linspace(log_nu_min, log_nu_max, 3):
                        val = neg_pen_ll(np.array([lq, lc, lp, ln]))
                        if val < best_neg:
                            best_neg = val
                            grid_best = (lq, lc, lp, ln)
        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max), (log_nu_min, log_nu_max)]
        start_points = [np.array(grid_best), np.array([adaptive_prior_mean, np.log10(0.9), 0.0, np.log10(6.0)])]
        best_res = None
        best_fun = float('inf')
        for x0 in start_points:
            try:
                res = minimize(neg_pen_ll, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 120, 'ftol': 1e-6})
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is not None and best_res.success:
            lq_opt, lc_opt, phi_opt, ln_opt = best_res.x
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            nu_opt = 10 ** ln_opt
            ll_opt = -best_res.fun
        else:
            lq_opt, lc_opt, phi_opt, ln_opt = grid_best
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            nu_opt = 10 ** ln_opt
            ll_opt = -best_neg

        # Compute φ shrinkage prior diagnostics for auditability
        n_obs_approx = len(returns)
        prior_scale_diag = 1.0 / max(n_obs_approx, 100)
        phi_lambda_eff_diag = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale_diag
        phi_tau_diag = lambda_to_tau(phi_lambda_eff_diag)
        phi_prior_diag = compute_phi_prior_diagnostics(
            phi_r=phi_opt,
            phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
            tau=phi_tau_diag,
            log_likelihood=ll_opt
        )

        diagnostics = {
            'grid_best_q': float(10 ** grid_best[0]),
            'grid_best_c': float(10 ** grid_best[1]),
            'grid_best_phi': float(grid_best[2]),
            'grid_best_nu': float(10 ** grid_best[3]),
            'refined_best_q': float(q_opt),
            'refined_best_c': float(c_opt),
            'refined_best_phi': float(phi_opt),
            'refined_best_nu': float(nu_opt),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_res is not None and (best_res.success if best_res else False),
            # φ shrinkage prior diagnostics (auditability)
            **phi_prior_diag,
        }

        return q_opt, c_opt, phi_opt, nu_opt, ll_opt, diagnostics

    @staticmethod
    def optimize_params_fixed_nu(
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float,
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
        """
        Optimize (q, c, φ) for the φ-Student-t drift model with FIXED ν.
        
        This method is part of the discrete ν grid approach:
        - ν is held fixed (passed as argument, not optimized)
        - Only q, c, φ are optimized via CV MLE
        - Each ν value becomes a separate sub-model in BMA
        
        This eliminates:
        - ν-σ identifiability failures
        - Pathological likelihood spikes from continuous ν optimization
        - Tail-driven weight collapse
        
        Args:
            returns: Array of returns
            vol: Array of EWMA volatility
            nu: FIXED degrees of freedom (from STUDENT_T_NU_GRID)
            train_frac: Fraction for train/test split
            q_min, q_max: Bounds for drift variance q
            c_min, c_max: Bounds for observation scale c
            phi_min, phi_max: Bounds for AR(1) coefficient φ
            prior_log_q_mean: Prior mean for log10(q)
            prior_lambda: Regularization strength
            
        Returns:
            Tuple of (q_opt, c_opt, phi_opt, ll_opt, diagnostics)
            Note: nu is NOT returned (it was fixed, not estimated)
        """
        n = len(returns)
        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0.0
        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        rv_ratio = abs(ret_mean) / ret_std if ret_std > 0 else 0.0

        # Adaptive prior adjustment
        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        # Build fold splits for CV
        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window
        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        # Fixed nu value (from grid, NOT optimized)
        nu_fixed = float(nu)

        def neg_pen_ll(params: np.ndarray) -> float:
            """Negative penalized log-likelihood with fixed nu."""
            log_q, log_c, phi = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))
            
            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
                return 1e12
            
            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []
            
            for tr_start, tr_end, te_start, te_end in fold_splits:
                try:
                    ret_train = returns_robust[tr_start:tr_end]
                    vol_train = vol[tr_start:tr_end]
                    if len(ret_train) < 3:
                        continue
                    
                    mu_filt_train, P_filt_train, _ = PhiStudentTDriftModel.filter_phi(
                        ret_train, vol_train, q, c, phi_clip, nu_fixed
                    )
                    mu_pred = float(mu_filt_train[-1])
                    P_pred = float(P_filt_train[-1])
                    ll_fold = 0.0
                    
                    for t in range(te_start, te_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q
                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())
                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            forecast_std = np.sqrt(forecast_var)
                            ll_contrib = PhiStudentTDriftModel.logpdf(ret_t, nu_fixed, mu_pred, forecast_std)
                            ll_fold += ll_contrib
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(innovation / forecast_std))

                        nu_adjust = min(nu_fixed / (nu_fixed + 3.0), 1.0)
                        K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (te_end - te_start)

                except Exception:
                    continue
            
            if total_obs == 0:
                return 1e12
            
            avg_ll = total_ll_oos / max(total_obs, 1)
            
            # Calibration penalty
            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = student_t.cdf(all_standardized, df=nu_fixed)
                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)
                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)
                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)
                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass
            
            # Priors on q, c, phi (NO prior on nu since it's fixed)
            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_c_target = np.log10(0.9)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2
            
            # =================================================================
            # EXPLICIT φ SHRINKAGE PRIOR (Gaussian)
            # =================================================================
            # φ_r ~ N(φ_global, τ²) where φ_global = 0 (full mean reversion)
            # See documentation at PHI_SHRINKAGE_* constants for derivation.
            # =================================================================
            phi_lambda_effective = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale
            phi_tau = lambda_to_tau(phi_lambda_effective)
            log_prior_phi = phi_shrinkage_log_prior(
                phi_r=phi_clip,
                phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
                tau=phi_tau
            )

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        # Parameter bounds (3 parameters: q, c, phi)
        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)

        # Grid search for initialization
        grid_best = (adaptive_prior_mean, np.log10(0.9), 0.0)
        best_neg = float('inf')
        for lq in np.linspace(log_q_min, log_q_max, 4):
            for lc in np.linspace(log_c_min, log_c_max, 3):
                for lp in np.linspace(phi_min, phi_max, 5):
                    val = neg_pen_ll(np.array([lq, lc, lp]))
                    if val < best_neg:
                        best_neg = val
                        grid_best = (lq, lc, lp)
        
        # L-BFGS-B optimization
        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max)]
        start_points = [
            np.array(grid_best),
            np.array([adaptive_prior_mean, np.log10(0.9), 0.0])
        ]
        best_res = None
        best_fun = float('inf')
        for x0 in start_points:
            try:
                res = minimize(
                    neg_pen_ll, x0=x0, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 120, 'ftol': 1e-6}
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is not None and best_res.success:
            lq_opt, lc_opt, phi_opt = best_res.x
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            ll_opt = -best_res.fun
        else:
            lq_opt, lc_opt, phi_opt = grid_best
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            ll_opt = -best_neg

        # Compute φ shrinkage prior diagnostics for auditability
        n_obs_approx = len(returns)
        prior_scale_diag = 1.0 / max(n_obs_approx, 100)
        phi_lambda_eff_diag = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale_diag
        phi_tau_diag = lambda_to_tau(phi_lambda_eff_diag)
        phi_prior_diag = compute_phi_prior_diagnostics(
            phi_r=phi_opt,
            phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
            tau=phi_tau_diag,
            log_likelihood=ll_opt
        )

        diagnostics = {
            'nu_fixed': float(nu_fixed),
            'grid_best_q': float(10 ** grid_best[0]),
            'grid_best_c': float(10 ** grid_best[1]),
            'grid_best_phi': float(grid_best[2]),
            'refined_best_q': float(q_opt),
            'refined_best_c': float(c_opt),
            'refined_best_phi': float(phi_opt),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_res is not None and (best_res.success if best_res else False),
            # φ shrinkage prior diagnostics (auditability)
            **phi_prior_diag,
        }

        return q_opt, c_opt, phi_opt, ll_opt, diagnostics


# Compatibility wrappers to preserve existing API surface

def kalman_filter_drift(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    return GaussianDriftModel.filter(returns, vol, q, c)


def kalman_filter_drift_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
    return GaussianDriftModel.filter_phi(returns, vol, q, c, phi)


def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
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
    """Thin wrapper for φ-Student-t Kalman filter."""
    return PhiStudentTDriftModel.filter_phi(returns, vol, q, c, phi, nu)


def compute_pit_ks_pvalue_student_t(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
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
) -> Tuple[float, float, float, Dict]:
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

def compute_kurtosis(data: np.ndarray) -> float:
    """
    Compute sample excess kurtosis (Fisher's definition: kurtosis - 3).

    Positive excess kurtosis indicates heavy tails (fat-tailed distribution).
    Zero indicates normal distribution.
    Negative indicates light tails.

    Args:
        data: Sample data

    Returns:
        Excess kurtosis
    """
    data_clean = data[np.isfinite(data)]
    if len(data_clean) < 4:
        return 0.0

    mean = np.mean(data_clean)
    std = np.std(data_clean, ddof=1)
    if std <= 0:
        return 0.0

    n = len(data_clean)
    m4 = np.mean(((data_clean - mean) / std) ** 4)

    # Fisher's definition: excess kurtosis = kurtosis - 3
    excess_kurtosis = m4 - 3.0

    return float(excess_kurtosis)

def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    AIC = -2*LL + 2*k

    Lower AIC indicates better model fit with penalty for complexity.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters

    Returns:
        AIC value
    """
    return -2.0 * log_likelihood + 2.0 * n_params

def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    BIC = -2*LL + k*ln(n)

    Lower BIC indicates better model fit with penalty for complexity.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of parameters
        n_obs: Number of observations

    Returns:
        BIC value
    """
    return -2.0 * log_likelihood + n_params * np.log(n_obs)

def tune_asset_q(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Optional[Dict]:
    """
    Estimate optimal parameters for a single asset via joint MLE with BIC-based model selection.
    
    Fits both Gaussian and Student-t observation noise models and selects the best based on BIC.
    Student-t is preferred for fat-tailed assets (crypto, commodities), Gaussian for stable assets.
    
    Includes:
    - Joint (q, c) optimization with Bayesian regularization
    - Zero-drift baseline comparison (ΔLL)
    - Safety fallbacks (q collapse, miscalibration, worse than baseline)
    - Comprehensive diagnostic metadata
    
    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
    
    Returns:
        Dictionary with results and diagnostics, or None if estimation failed
    """
    try:
        # Fetch price data
        try:
            px, title = fetch_px(asset, start_date, end_date)
        except Exception:
            # Fallback: direct download with standardized format
            df = _download_prices(asset, start_date, end_date)
            if df is None or df.empty:
                raise RuntimeError(f"No data for {asset}")
            
            # Use standardized price extraction
            px = get_price_series(df, "Close")
            if px.empty:
                raise RuntimeError(f"No price column found for {asset}")
            
            title = asset
        
        # Allow very small histories; tune will still run cross-validation with short splits
        px = pd.to_numeric(px, errors="coerce").dropna()
        if len(px) < 10:
            raise RuntimeError(f"Insufficient data ({len(px)} days)")
        
        # Compute returns
        log_px = np.log(px)
        returns = log_px.diff().dropna()

        # Compute EWMA volatility (observation noise) with a span that adapts to short histories
        span = max(5, min(21, max(len(returns) // 2, 5)))
        vol = returns.ewm(span=span, adjust=False).std()

        # Align series with a smaller warmup for tiny datasets
        warmup = min(20, max(len(returns) // 4, 1))
        returns = returns.iloc[warmup:]
        vol = vol.iloc[warmup:]

        if len(returns) < 5:
            raise RuntimeError(f"Insufficient data after preprocessing ({len(returns)} returns)")

        returns_arr = returns.values
        vol_arr = vol.values
        
        # Apply minimum volatility floor to prevent division by zero in PIT calculations
        # A floor of 1e-8 is conservative: typical daily vol is 0.01-0.05 (1-5%)
        # This handles edge cases like constant prices or numerical underflow
        vol_arr = np.maximum(vol_arr, 1e-8)
        
        n_obs = len(returns_arr)

        # Compute kurtosis to assess tail heaviness
        excess_kurtosis = compute_kurtosis(returns_arr)
        
        # =================================================================
        # STEP 1: Fit Gaussian Model (q, c)
        # =================================================================
        _log(f"  🔧 Fitting Gaussian model...")
        q_gauss, c_gauss, ll_gauss_cv, opt_diag_gauss = optimize_q_mle(
            returns_arr, vol_arr,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        # Run full Gaussian Kalman filter
        mu_gauss, P_gauss, ll_gauss_full = kalman_filter_drift(returns_arr, vol_arr, q_gauss, c_gauss)
        
        # Compute Gaussian PIT calibration
        ks_gauss, pit_p_gauss = compute_pit_ks_pvalue(returns_arr, mu_gauss, vol_arr, P_gauss, c_gauss)
        
        # Gaussian has 2 parameters: q, c
        aic_gauss = compute_aic(ll_gauss_full, n_params=2)
        bic_gauss = compute_bic(ll_gauss_full, n_params=2, n_obs=n_obs)
        
        _log(f"     Gaussian: q={q_gauss:.2e}, c={c_gauss:.3f}, LL={ll_gauss_full:.1f}, BIC={bic_gauss:.1f}, PIT p={pit_p_gauss:.4f}")
        
        # =================================================================
        # STEP 1.5: Fit φ-Kalman Model (q, c, φ)
        # =================================================================
        _log(f"  🔧 Fitting φ-Gaussian-Kalman model...")
        q_phi, c_phi, phi_opt, ll_phi_cv, opt_diag_phi = optimize_q_c_phi_mle(
            returns_arr, vol_arr,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        mu_phi, P_phi, ll_phi_full = kalman_filter_drift_phi(returns_arr, vol_arr, q_phi, c_phi, phi_opt)
        ks_phi, pit_p_phi = compute_pit_ks_pvalue(returns_arr, mu_phi, vol_arr, P_phi, c_phi)
        aic_phi = compute_aic(ll_phi_full, n_params=3)
        bic_phi = compute_bic(ll_phi_full, n_params=3, n_obs=n_obs)
        _log(f"     φ-Gaussian-Kalman: q={q_phi:.2e}, c={c_phi:.3f}, φ={phi_opt:+.3f}, LL={ll_phi_full:.1f}, BIC={bic_phi:.1f}, PIT p={pit_p_phi:.4f}")
        
        # =================================================================
        # STEP 2: Fit Kalman φ-Student-t Models with DISCRETE ν GRID
        # =================================================================
        # Instead of continuous ν optimization (which is unstable), we fit
        # separate sub-models for each ν in STUDENT_T_NU_GRID and select
        # the best one by BIC.
        # =================================================================
        _log(f"  🔧 Fitting Kalman φ-Student-t models (discrete ν grid: {STUDENT_T_NU_GRID})...")
        
        student_t_results = []  # List of (model_name, bic, aic, ll, mu, P, ks, pit_p, q, c, phi, nu, diag)
        
        for nu_fixed in STUDENT_T_NU_GRID:
            model_name = f"phi_student_t_nu_{nu_fixed}"
            try:
                # Optimize q, c, phi with FIXED nu
                q_st, c_st, phi_st, ll_cv_st, diag_st = PhiStudentTDriftModel.optimize_params_fixed_nu(
                    returns_arr, vol_arr,
                    nu=nu_fixed,
                    prior_log_q_mean=prior_log_q_mean,
                    prior_lambda=prior_lambda
                )
                
                # Run full φ-Student-t Kalman filter
                mu_st, P_st, ll_full_st = kalman_filter_drift_phi_student_t(
                    returns_arr, vol_arr, q_st, c_st, phi_st, nu_fixed
                )
                
                # Compute Student-t PIT calibration
                ks_st, pit_p_st = compute_pit_ks_pvalue_student_t(
                    returns_arr, mu_st, vol_arr, P_st, c_st, nu_fixed
                )
                
                # φ-Student-t has 3 estimated parameters: q, c, φ (ν is FIXED)
                aic_st = compute_aic(ll_full_st, n_params=3)
                bic_st = compute_bic(ll_full_st, n_params=3, n_obs=n_obs)
                
                _log(f"     {model_name}: q={q_st:.2e}, c={c_st:.3f}, φ={phi_st:+.3f}, LL={ll_full_st:.1f}, BIC={bic_st:.1f}, PIT p={pit_p_st:.4f}")
                
                student_t_results.append((
                    model_name, bic_st, aic_st, ll_full_st,
                    mu_st, P_st, ks_st, pit_p_st,
                    q_st, c_st, phi_st, nu_fixed, diag_st
                ))
                
            except Exception as e:
                _log(f"     {model_name}: ⚠️ optimization failed: {e}")
                continue
        
        # Find best Student-t model by BIC
        student_t_fit_success = len(student_t_results) > 0
        if student_t_fit_success:
            best_student_t = min(student_t_results, key=lambda x: x[1])
            (best_st_name, bic_student, aic_student, ll_student_full,
             mu_student, P_student, ks_student, pit_p_student,
             q_student, c_student, phi_student, nu_student, opt_diag_student) = best_student_t
            _log(f"     Best Student-t: {best_st_name} (BIC={bic_student:.1f})")
        else:
            _log(f"  ⚠️  All φ-Student-t optimizations failed")
            best_st_name = None
            q_student = None
            c_student = None
            phi_student = None
            nu_student = None
            ll_student_full = -1e12
            bic_student = 1e12
            aic_student = 1e12
            pit_p_student = 0.0
            opt_diag_student = {}

        # =================================================================
        # STEP 2.5: ADAPTIVE ν REFINEMENT
        # =================================================================
        # When calibration fails (PIT p < 0.05) for φ-T models at boundary ν,
        # we locally refine the ν grid to test intermediate values.
        #
        # Core principle: Add resolution only where truth demands it.
        # =================================================================
        nu_refinement_result = None
        if student_t_fit_success and ADAPTIVE_NU_AVAILABLE and ADAPTIVE_NU_ENABLED:
            # Check if refinement is needed
            # Build a result dict for the detection function
            student_t_logliks = {
                f"phi_student_t_nu_{int(r[11])}": {'ll': r[3], 'bic': r[1]}
                for r in student_t_results
            }
            
            refinement_check = {
                'model': f'φ-T(ν={int(nu_student)})',
                'nu': float(nu_student),
                'pit_ks_pvalue': float(pit_p_student),
                'bic': float(bic_student),
                'model_comparison': student_t_logliks,
            }
            
            if needs_nu_refinement(refinement_check, get_adaptive_nu_config()):
                # Get refinement candidates
                candidates = get_refinement_candidates(float(nu_student), get_adaptive_nu_config())
                
                if candidates:
                    _log(f"  🔧 Adaptive ν refinement: testing {candidates} for ν={nu_student}...")
                    
                    pit_before = pit_p_student
                    bic_before = bic_student
                    nu_before = nu_student
                    
                    # Test each candidate
                    refinement_tested = []
                    for nu_cand in candidates:
                        model_name_cand = f"phi_student_t_nu_{int(nu_cand)}"
                        try:
                            q_cand, c_cand, phi_cand, ll_cv_cand, diag_cand = PhiStudentTDriftModel.optimize_params_fixed_nu(
                                returns_arr, vol_arr,
                                nu=nu_cand,
                                prior_log_q_mean=prior_log_q_mean,
                                prior_lambda=prior_lambda
                            )
                            
                            mu_cand, P_cand, ll_full_cand = kalman_filter_drift_phi_student_t(
                                returns_arr, vol_arr, q_cand, c_cand, phi_cand, nu_cand
                            )
                            
                            ks_cand, pit_p_cand = compute_pit_ks_pvalue_student_t(
                                returns_arr, mu_cand, vol_arr, P_cand, c_cand, nu_cand
                            )
                            
                            aic_cand = compute_aic(ll_full_cand, n_params=3)
                            bic_cand = compute_bic(ll_full_cand, n_params=3, n_obs=n_obs)
                            
                            _log(f"     {model_name_cand}: BIC={bic_cand:.1f}, PIT p={pit_p_cand:.4f}")
                            
                            refinement_tested.append((
                                model_name_cand, bic_cand, aic_cand, ll_full_cand,
                                mu_cand, P_cand, ks_cand, pit_p_cand,
                                q_cand, c_cand, phi_cand, nu_cand, diag_cand
                            ))
                            
                            # Also add to student_t_results for model comparison
                            student_t_results.append((
                                model_name_cand, bic_cand, aic_cand, ll_full_cand,
                                mu_cand, P_cand, ks_cand, pit_p_cand,
                                q_cand, c_cand, phi_cand, nu_cand, diag_cand
                            ))
                            
                        except Exception as e:
                            _log(f"     {model_name_cand}: ⚠️ refinement failed: {e}")
                            continue
                    
                    # Check if any refinement improved BIC
                    if refinement_tested:
                        # Re-find best Student-t model including new candidates
                        best_student_t = min(student_t_results, key=lambda x: x[1])
                        (best_st_name, bic_student, aic_student, ll_student_full,
                         mu_student, P_student, ks_student, pit_p_student,
                         q_student, c_student, phi_student, nu_student, opt_diag_student) = best_student_t
                        
                        improvement_achieved = (bic_student < bic_before)
                        
                        nu_refinement_result = {
                            'refinement_attempted': True,
                            'nu_original': float(nu_before),
                            'nu_candidates_tested': [float(c) for c in candidates],
                            'nu_final': float(nu_student),
                            'improvement_achieved': bool(improvement_achieved),
                            'pit_before_refinement': float(pit_before),
                            'pit_after_refinement': float(pit_p_student),
                            'bic_before_refinement': float(bic_before),
                            'bic_after_refinement': float(bic_student),
                            'likelihood_flatness': bool(is_nu_likelihood_flat(refinement_check)),
                        }
                        
                        if improvement_achieved:
                            _log(f"     ✓ Refinement improved: ν={nu_before}→{nu_student}, BIC={bic_before:.1f}→{bic_student:.1f}")
                        else:
                            _log(f"     ✗ Refinement did not improve (keeping ν={nu_student})")

        # =================================================================
        # STEP 3: Model Selection via BIC
        # =================================================================
        # Lower BIC is better (penalizes complexity)
        candidate_models = []
        candidate_models.append(("kalman_gaussian", bic_gauss, aic_gauss, ll_gauss_full, mu_gauss, P_gauss, ks_gauss, pit_p_gauss, q_gauss, c_gauss, None, opt_diag_gauss))
        candidate_models.append(("kalman_phi_gaussian", bic_phi, aic_phi, ll_phi_full, mu_phi, P_phi, ks_phi, pit_p_phi, q_phi, c_phi, phi_opt, opt_diag_phi))
        if student_t_fit_success:
            # Use the best Student-t model name (e.g., "phi_student_t_nu_6")
            candidate_models.append((best_st_name, bic_student, aic_student, ll_student_full, mu_student, P_student, ks_student, pit_p_student, q_student, c_student, (phi_student, nu_student), opt_diag_student))

        candidate_models = [m for m in candidate_models if np.isfinite(m[1])]
        best_entry = min(candidate_models, key=lambda x: x[1])
        noise_model, bic_final, aic_final, ll_full, mu_filtered, P_filtered, ks_statistic, ks_pvalue, q_optimal, c_optimal, extra_param, opt_diagnostics = best_entry

        nu_optimal = None
        phi_selected = None
        if is_student_t_model(noise_model):
            phi_selected, nu_optimal = extra_param
        elif noise_model == "kalman_phi_gaussian":
            phi_selected = extra_param

        _log(f"  ✓ Selected {noise_model} (BIC={bic_final:.1f})")
        if is_student_t_model(noise_model):
            _log(f"    (ΔBIC vs Gaussian = {bic_gauss - bic_student:+.1f}, ΔBIC vs φ-Gaussian = {bic_phi - bic_student:+.1f})")
        elif noise_model == "kalman_phi_gaussian":
            _log(f"    (ΔBIC vs Gaussian = {bic_gauss - bic_phi:+.1f})")
        else:
            _log(f"    (ΔBIC vs φ-Gaussian Kalman = {bic_phi - bic_gauss:+.1f})")
            _log(f"")
        # =================================================================
        # Upgrade #4: Model Comparison - Baseline Models
        # =================================================================
        # Compare Kalman drift model against simpler baselines for formal model selection
        
        def compute_zero_drift_ll(returns_arr, vol_arr, c):
            """
            Compute log-likelihood of zero-drift model (μ=0 for all t).
            
            This is the simplest baseline: assumes no predictable drift.
            """
            # Ensure 1D arrays
            returns_flat = np.asarray(returns_arr).flatten()
            vol_flat = np.asarray(vol_arr).flatten()
            
            ll = 0.0
            for t in range(len(returns_flat)):
                ret_t = float(returns_flat[t])
                vol_t = float(vol_flat[t])
                R = c * (vol_t ** 2)
                innovation = ret_t - 0.0
                forecast_var = R
                if forecast_var > 1e-12:
                    ll += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
            return float(ll)
        
        def compute_constant_drift_ll(returns_arr, vol_arr, c):
            """
            Compute log-likelihood of constant-drift model (μ = mean(returns) for all t).
            
            This baseline assumes drift exists but is fixed over time.
            Parameters: c (1 parameter)
            """
            # Ensure 1D arrays
            returns_flat = np.asarray(returns_arr).flatten()
            vol_flat = np.asarray(vol_arr).flatten()
            
            # Estimate constant drift as sample mean
            mu_const = float(np.mean(returns_flat))
            
            ll = 0.0
            for t in range(len(returns_flat)):
                ret_t = float(returns_flat[t])
                vol_t = float(vol_flat[t])
                R = c * (vol_t ** 2)
                innovation = ret_t - mu_const
                forecast_var = R
                if forecast_var > 1e-12:
                    ll += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
            return float(ll), float(mu_const)
        
        def compute_ewma_drift_ll(returns_arr, vol_arr, c, span=21):
            """
            Compute log-likelihood of EWMA-drift model (μ_t = EWMA of past returns).
            
            This baseline uses exponentially weighted moving average for time-varying drift.
            Parameters: c, span (2 parameters effectively, but span is fixed)
            """
            # Ensure 1D arrays
            returns_flat = np.asarray(returns_arr).flatten()
            vol_flat = np.asarray(vol_arr).flatten()
            
            # Compute EWMA drift estimates
            ret_series = pd.Series(returns_flat)
            mu_ewma = ret_series.ewm(span=span, adjust=False).mean().values
            
            ll = 0.0
            for t in range(len(returns_flat)):
                if t == 0:
                    # First observation: use zero drift
                    mu_t = 0.0
                else:
                    # Use EWMA estimate from previous time step (no look-ahead)
                    mu_t = float(mu_ewma[t-1])
                
                ret_t = float(returns_flat[t])
                vol_t = float(vol_flat[t])
                R = c * (vol_t ** 2)
                innovation = ret_t - mu_t
                forecast_var = R
                if forecast_var > 1e-12:
                    ll += -0.5 * np.log(2 * np.pi * forecast_var) - 0.5 * (innovation ** 2) / forecast_var
            return float(ll)
        
        # (No additional Kalman rerun here; use the variant already selected above)
        
        # =================================================================
        # Compute all baseline models for formal model comparison
        # =================================================================
        _log(f"  🔬 Model comparison:")
        
        # Baseline 1: Zero-drift (0 parameters: just uses c from Kalman)
        ll_zero = compute_zero_drift_ll(returns_arr, vol_arr, c_optimal)
        aic_zero = compute_aic(ll_zero, n_params=0)  # c is shared, not counted
        bic_zero = compute_bic(ll_zero, n_params=0, n_obs=n_obs)
        _log(f"     Zero-drift:     LL={ll_zero:.1f}, AIC={aic_zero:.1f}, BIC={bic_zero:.1f}")
        
        # Baseline 2: Constant-drift (1 parameter: mu_const, c is shared)
        ll_const, mu_const = compute_constant_drift_ll(returns_arr, vol_arr, c_optimal)
        aic_const = compute_aic(ll_const, n_params=1)
        bic_const = compute_bic(ll_const, n_params=1, n_obs=n_obs)
        _log(f"     Constant-drift: LL={ll_const:.1f}, AIC={aic_const:.1f}, BIC={bic_const:.1f}, μ={mu_const:.6f}")
        
        # Baseline 3: EWMA-drift (1 parameter: span is fixed, c is shared)
        ll_ewma = compute_ewma_drift_ll(returns_arr, vol_arr, c_optimal, span=21)
        aic_ewma = compute_aic(ll_ewma, n_params=1)
        bic_ewma = compute_bic(ll_ewma, n_params=1, n_obs=n_obs)
        _log(f"     EWMA-drift:     LL={ll_ewma:.1f}, AIC={aic_ewma:.1f}, BIC={bic_ewma:.1f}")
        
        # Kalman variants printed separately
        _log(f"     Kalman-Gaussian: LL={ll_gauss_full:.1f}, AIC={aic_gauss:.1f}, BIC={bic_gauss:.1f}")
        _log(f"     Kalman-φ-Gaussian: LL={ll_phi_full:.1f}, AIC={aic_phi:.1f}, BIC={bic_phi:.1f}, φ={phi_opt:+.3f}")
        if student_t_fit_success:
            _log(f"     Kalman-φ-Student-t: LL={ll_student_full:.1f}, AIC={aic_student:.1f}, BIC={bic_student:.1f}, φ={phi_opt:+.3f}, ν={nu_student:.1f}")
        
        # Selected model summary (already chosen above)
        _log(f"     Selected:        LL={ll_full:.1f}, AIC={aic_final:.1f}, BIC={bic_final:.1f} ({noise_model})")
        
        # ΔLL against baselines using the selected model's LL
        delta_ll_vs_zero = float(ll_full - ll_zero)
        delta_ll_vs_const = float(ll_full - ll_const)
        delta_ll_vs_ewma = float(ll_full - ll_ewma)

        # Aggregate model comparison metrics for diagnostics and cache
        # Compute Hyvärinen scores for ALL models including baselines
        
        # Baseline models: use c_optimal for observation noise (they don't have state uncertainty)
        # For zero_drift: mu = 0 for all t, sigma = sqrt(c * vol^2)
        forecast_std_zero = np.sqrt(c_optimal * (vol_arr ** 2))
        mu_zero = np.zeros_like(returns_arr)
        hyv_zero = compute_hyvarinen_score_gaussian(returns_arr, mu_zero, forecast_std_zero)
        
        # For constant_drift: mu = mu_const for all t, sigma = sqrt(c * vol^2)
        mu_const_arr = np.full_like(returns_arr, mu_const)
        hyv_const = compute_hyvarinen_score_gaussian(returns_arr, mu_const_arr, forecast_std_zero)
        
        # For ewma_drift: mu = ewma of past returns, sigma = sqrt(c * vol^2)
        ret_series = pd.Series(returns_arr)
        mu_ewma_arr = ret_series.ewm(span=21, adjust=False).mean().values
        # First observation uses zero drift
        mu_ewma_arr_lagged = np.concatenate([[0.0], mu_ewma_arr[:-1]])
        hyv_ewma = compute_hyvarinen_score_gaussian(returns_arr, mu_ewma_arr_lagged, forecast_std_zero)
        
        # Kalman models: include state uncertainty in forecast
        forecast_std_gauss = np.sqrt(c_gauss * (vol_arr ** 2) + P_gauss)
        hyv_gauss = compute_hyvarinen_score_gaussian(returns_arr, mu_gauss, forecast_std_gauss)
        
        forecast_std_phi = np.sqrt(c_phi * (vol_arr ** 2) + P_phi)
        hyv_phi = compute_hyvarinen_score_gaussian(returns_arr, mu_phi, forecast_std_phi)
        
        model_comparison = {
            'zero_drift': {'ll': ll_zero, 'aic': aic_zero, 'bic': bic_zero, 'n_params': 0, 'hyvarinen_score': float(hyv_zero)},
            'constant_drift': {'ll': ll_const, 'aic': aic_const, 'bic': bic_const, 'n_params': 1, 'mu': float(mu_const), 'hyvarinen_score': float(hyv_const)},
            'ewma_drift': {'ll': ll_ewma, 'aic': aic_ewma, 'bic': bic_ewma, 'n_params': 1, 'hyvarinen_score': float(hyv_ewma)},
            'kalman_gaussian': {'ll': ll_gauss_full, 'aic': aic_gauss, 'bic': bic_gauss, 'n_params': 2, 'hyvarinen_score': float(hyv_gauss)},
            'kalman_phi_gaussian': {'ll': ll_phi_full, 'aic': aic_phi, 'bic': bic_phi, 'n_params': 3, 'phi': float(phi_opt), 'hyvarinen_score': float(hyv_phi)},
        }
        
        # Add ALL Student-t sub-models from the discrete nu grid
        for st_result in student_t_results:
            (st_name, st_bic, st_aic, st_ll, st_mu, st_P, st_ks, st_pit_p,
             st_q, st_c, st_phi, st_nu, st_diag) = st_result
            forecast_std_st = np.sqrt(st_c * (vol_arr ** 2) + st_P)
            hyv_st = compute_hyvarinen_score_student_t(returns_arr, st_mu, forecast_std_st, st_nu)
            model_comparison[st_name] = {
                'll': st_ll,
                'aic': st_aic,
                'bic': st_bic,
                'n_params': 3,  # q, c, phi (nu is FIXED)
                'phi': float(st_phi),
                'nu': float(st_nu),
                'nu_fixed': True,
                'hyvarinen_score': float(hyv_st)
            }
        
        # Compute combined scores for all models
        bic_values = {m: info['bic'] for m, info in model_comparison.items()}
        hyvarinen_scores = {m: info['hyvarinen_score'] for m, info in model_comparison.items()}
        combined_weights, weight_metadata = compute_combined_model_weights(
            bic_values, hyvarinen_scores, bic_weight=DEFAULT_BIC_WEIGHT
        )
        
        # Store combined_score and standardized scores in each model
        # combined_score is now the standardized combined score (lower = better)
        # model_weight_entropy is the entropy-regularized weight
        for m in model_comparison:
            w = combined_weights.get(m, 1e-10)
            model_comparison[m]['combined_score'] = float(weight_metadata['combined_scores_standardized'].get(m, 0.0))
            model_comparison[m]['model_weight_entropy'] = float(w)
            model_comparison[m]['standardized_bic'] = float(weight_metadata['bic_standardized'].get(m, 0.0)) if weight_metadata['bic_standardized'].get(m) is not None else None
            model_comparison[m]['standardized_hyvarinen'] = float(weight_metadata['hyvarinen_standardized'].get(m, 0.0)) if weight_metadata['hyvarinen_standardized'].get(m) is not None else None
        
        # Best model across baselines and Kalman variants by BIC
        best_model_name = min(model_comparison.items(), key=lambda kv: kv[1]['bic'])[0]
        
        # Best model by combined score (lowest standardized combined score = best)
        best_model_by_combined = min(model_comparison.items(), key=lambda kv: kv[1].get('combined_score', float('inf')))[0]

        # Compute drift diagnostics
        mean_drift_var = float(np.mean(mu_filtered ** 2))
        mean_posterior_unc = float(np.mean(P_filtered))
        
        # Compute standardized residuals for kurtosis check
        forecast_std = np.sqrt(c_optimal * (vol_arr ** 2) + P_filtered)
        standardized_residuals = (returns_arr - mu_filtered) / forecast_std
        std_residual_kurtosis = compute_kurtosis(standardized_residuals)

        # Calibration warning flag
        calibration_warning = (ks_pvalue < 0.05)

        # Print calibration status
        if calibration_warning:
            if ks_pvalue < 0.01:
                _log(f"  ⚠️  Severe miscalibration (PIT p={ks_pvalue:.4f})")
            else:
                _log(f"  ⚠️  Calibration warning (PIT p={ks_pvalue:.4f})")

        # Build result dictionary with extended schema
        # Get hyvarinen score and combined score for the selected model
        # Map noise_model to model_comparison keys
        # Note: Student-t models now use phi_student_t_nu_{nu} naming
        noise_model_to_key = {
            'gaussian': 'kalman_gaussian',
            'kalman_gaussian': 'kalman_gaussian',
            'phi_gaussian': 'kalman_phi_gaussian',
            'kalman_phi_gaussian': 'kalman_phi_gaussian',
        }
        # For Student-t models, the noise_model IS the key (e.g., phi_student_t_nu_6)
        model_key = noise_model_to_key.get(noise_model, noise_model)
        
        selected_hyvarinen = model_comparison.get(model_key, {}).get('hyvarinen_score')
        selected_combined_score = model_comparison.get(model_key, {}).get('combined_score')
        
        result = {
            # Asset identifier
            'asset': asset,

            # Model selection
            'noise_model': noise_model,  # e.g., "kalman_gaussian", "kalman_phi_gaussian", or "phi_student_t_nu_6"
            'model_selection_method': 'combined',  # Always use combined for consistency

            # Parameters
            'q': float(q_optimal),
            'c': float(c_optimal),
            'nu': float(nu_optimal) if nu_optimal is not None else None,
            'phi': float(phi_selected) if phi_selected is not None else None,

            # Likelihood and model comparison
            # NOTE: log_likelihood here is TOTAL (sum over all observations) from full filter run
            # This differs from regime["cv_penalized_ll"] which is penalized mean LL from CV
            'log_likelihood': float(ll_full),
            'delta_ll_vs_zero': float(delta_ll_vs_zero),
            'delta_ll_vs_const': float(delta_ll_vs_const),
            'delta_ll_vs_ewma': float(delta_ll_vs_ewma),
            'aic': float(aic_final),
            'bic': float(bic_final),
            'hyvarinen_score': float(selected_hyvarinen) if selected_hyvarinen is not None else None,
            'combined_score': float(selected_combined_score) if selected_combined_score is not None else None,

            # Upgrade #4: Model comparison results
            'model_comparison': model_comparison,
            'best_model_by_bic': best_model_name,
            'best_model_by_combined': best_model_by_combined,

            # Calibration diagnostics
            'ks_statistic': float(ks_statistic),
            'pit_ks_pvalue': float(ks_pvalue),
            'calibration_warning': bool(calibration_warning),
            'std_residual_kurtosis': float(std_residual_kurtosis),

            # Drift diagnostics
            'mean_drift_var': float(mean_drift_var),
            'mean_posterior_unc': float(mean_posterior_unc),

            # Data characteristics
            'n_obs': int(n_obs),
            'excess_kurtosis': float(excess_kurtosis),

            # Metadata
            'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),

            # Optimization diagnostics from selected model
            'grid_best_q': opt_diagnostics.get('grid_best_q'),
            'grid_best_c': opt_diagnostics.get('grid_best_c'),
            'refined_best_q': opt_diagnostics.get('refined_best_q'),
            'refined_best_c': opt_diagnostics.get('refined_best_c'),
            'prior_applied': opt_diagnostics.get('prior_applied'),
            'prior_log_q_mean': opt_diagnostics.get('prior_log_q_mean'),
            'prior_lambda': opt_diagnostics.get('prior_lambda'),
            'vol_cv': opt_diagnostics.get('vol_cv'),
            'rv_ratio': opt_diagnostics.get('rv_ratio'),
            'n_folds': opt_diagnostics.get('n_folds'),
            'robust_optimization': True,
            'optimization_successful': opt_diagnostics.get('optimization_successful', False)
        }

        # Add Student-t specific diagnostics if applicable
        if is_student_t_model(noise_model):
            result['nu_fixed'] = True  # Always true now (discrete grid)
            result['refined_best_phi'] = float(phi_selected) if phi_selected is not None else None
        if noise_model == "kalman_phi_gaussian":
            result['refined_best_phi'] = float(phi_selected)

        # Add Gaussian comparison if Student-t was selected
        if is_student_t_model(noise_model) and student_t_fit_success:
            result['gaussian_bic'] = float(bic_gauss)
            result['gaussian_log_likelihood'] = float(ll_gauss_full)
            result['gaussian_pit_ks_pvalue'] = float(pit_p_gauss)
            result['bic_improvement'] = float(bic_gauss - bic_student)

        # Add adaptive ν refinement results if attempted
        if nu_refinement_result is not None:
            result['nu_refinement'] = nu_refinement_result
        else:
            result['nu_refinement'] = {
                'refinement_attempted': False,
                'nu_original': float(nu_optimal) if nu_optimal is not None else None,
                'nu_candidates_tested': [],
                'nu_final': float(nu_optimal) if nu_optimal is not None else None,
                'improvement_achieved': False,
            }

        # =================================================================
        # K=2 MIXTURE MODEL - REMOVED (Empirically Falsified)
        # =================================================================
        # K=2 mixture was removed after evaluation:
        #   - 206 attempts, 0 selections (0% success rate)
        #   - Returns are fat-tailed unimodal, not bimodal
        #   - HMM regime-switching + Student-t already handles regimes
        # 
        # Fields preserved for backward compatibility with cached results
        # =================================================================
        result['mixture_attempted'] = False
        result['mixture_selected'] = False
        result['mixture_model'] = None

        # =================================================================
        # GENERALIZED HYPERBOLIC (GH) DISTRIBUTION FALLBACK
        # =================================================================
        # GH is attempted as LAST RESORT when:
        #   1. Calibration still fails (PIT p < 0.05)
        #   2. Mixture was attempted but not selected OR still failing
        #   3. ν-refinement didn't solve the problem
        #
        # GH captures SKEWNESS that symmetric Student-t cannot.
        # It's computationally expensive, so only used when necessary.
        # =================================================================
        gh_config = get_gh_config()
        result['gh_attempted'] = False
        result['gh_selected'] = False
        result['gh_model'] = None
        
        # Check if we should attempt GH
        current_pit_pvalue = result.get('pit_ks_pvalue', ks_pvalue)
        current_calibration_warning = result.get('calibration_warning', calibration_warning)
        
        if gh_config is not None and current_calibration_warning:
            # Check escalation conditions
            mixture_attempted = result.get('mixture_attempted', False)
            mixture_selected = result.get('mixture_selected', False)
            nu_ref = result.get('nu_refinement', {})
            nu_refinement_attempted = nu_ref.get('refinement_attempted', False)
            nu_refinement_improved = nu_ref.get('improvement_achieved', False)
            
            attempt_gh = should_attempt_gh(
                pit_ks_pvalue=current_pit_pvalue,
                mixture_attempted=mixture_attempted,
                mixture_selected=mixture_selected,
                nu_refinement_attempted=nu_refinement_attempted,
                nu_refinement_improved=nu_refinement_improved,
                config=gh_config
            )
            
            if attempt_gh:
                _log(f"  🔧 Attempting Generalized Hyperbolic (GH) model for calibration improvement...")
                result['gh_attempted'] = True
                
                try:
                    # Compute standardized residuals for GH fitting
                    forecast_std = np.sqrt(c_optimal * (vol_arr ** 2) + P_filtered)
                    standardized_residuals = (returns_arr - mu_filtered) / forecast_std
                    
                    # Fit GH model
                    gh_model = GHModel(gh_config)
                    gh_result = gh_model.fit(
                        z=standardized_residuals,
                        single_bic=bic_final,
                        single_pit_pvalue=current_pit_pvalue
                    )
                    
                    if gh_result is not None:
                        _log(f"     GH fit: λ={gh_result.lam:.2f}, α={gh_result.alpha:.2f}, "
                             f"β={gh_result.beta:.2f}, δ={gh_result.delta:.2f}")
                        _log(f"     GH PIT p={gh_result.pit_ks_pvalue:.4f}, "
                             f"skew={gh_result.skewness_direction}, tails={gh_result.tail_behavior}")
                        
                        # Check if GH should be selected
                        use_gh = should_select_gh(
                            gh_result=gh_result,
                            single_pit_pvalue=current_pit_pvalue,
                            config=gh_config
                        )
                        
                        if use_gh:
                            _log(f"     ✓ GH selected: PIT improved {current_pit_pvalue:.4f}→{gh_result.pit_ks_pvalue:.4f}")
                            
                            result['gh_model'] = gh_result.to_dict()
                            result['gh_selected'] = True
                            result['noise_model'] = 'generalized_hyperbolic'
                            
                            # Update calibration status
                            if gh_result.is_calibrated:
                                result['calibration_warning'] = False
                                result['pit_ks_pvalue'] = float(gh_result.pit_ks_pvalue)
                                result['ks_statistic'] = float(gh_result.ks_statistic)
                        else:
                            _log(f"     ✗ GH not selected (PIT {current_pit_pvalue:.4f}→{gh_result.pit_ks_pvalue:.4f}, "
                                 f"BIC impr={gh_result.bic_improvement:.1f})")
                            result['gh_model'] = gh_result.to_dict()
                            result['gh_selected'] = False
                    else:
                        _log(f"     ✗ GH fitting failed")
                        
                except Exception as gh_err:
                    _log(f"     ✗ GH error: {gh_err}")
                    result['gh_model'] = None
                    result['gh_selected'] = False

        # =================================================================
        # TIME-VARYING VOLATILITY MULTIPLIER (TVVM) FALLBACK
        # =================================================================
        # TVVM is attempted as LAST RESORT when:
        #   1. Calibration still fails (PIT p < 0.05)
        #   2. Other escalation methods have been tried
        #   3. Asset shows volatility regime switching (vol-of-vol > threshold)
        #
        # TVVM addresses volatility-of-volatility effect by making c dynamic:
        #   c_t = c_base * (1 + γ * |Δσ_t/σ_t|)
        # =================================================================
        tvvm_config = get_tvvm_config()
        result['tvvm_attempted'] = False
        result['tvvm_selected'] = False
        result['tvvm_model'] = None
        
        # Check if we should attempt TVVM
        current_pit_pvalue = result.get('pit_ks_pvalue', ks_pvalue)
        current_calibration_warning = result.get('calibration_warning', calibration_warning)
        
        if tvvm_config is not None and current_calibration_warning:
            # Check if TVVM should be attempted
            attempt_tvvm = should_attempt_tvvm(
                pit_ks_pvalue=current_pit_pvalue,
                vol=vol_arr,
                config=tvvm_config
            )
            
            if attempt_tvvm:
                _log(f"  🔧 Attempting Time-Varying Volatility Multiplier (TVVM)...")
                result['tvvm_attempted'] = True
                
                try:
                    # Fit TVVM model
                    tvvm_model = TVVMModel(tvvm_config)
                    tvvm_result = tvvm_model.fit(
                        returns=returns_arr,
                        vol=vol_arr,
                        mu_filtered=mu_filtered,
                        P_filtered=P_filtered,
                        c_static=c_optimal,
                        nu=nu_optimal,
                        static_pit_pvalue=current_pit_pvalue,
                        static_bic=bic_final
                    )
                    
                    if tvvm_result is not None:
                        _log(f"     TVVM fit: γ={tvvm_result.gamma:.2f}, "
                             f"c_mean={tvvm_result.c_mean:.3f}, c_max={tvvm_result.c_max:.3f}")
                        _log(f"     TVVM PIT p={tvvm_result.pit_ks_pvalue:.4f}, "
                             f"vol_of_vol={tvvm_result.vol_of_vol:.3f}")
                        
                        # Check if TVVM should be selected
                        use_tvvm = should_select_tvvm(
                            tvvm_result=tvvm_result,
                            static_pit_pvalue=current_pit_pvalue,
                            config=tvvm_config
                        )
                        
                        if use_tvvm:
                            _log(f"     ✓ TVVM selected: PIT improved {current_pit_pvalue:.4f}→{tvvm_result.pit_ks_pvalue:.4f}")
                            
                            result['tvvm_model'] = tvvm_result.to_dict()
                            result['tvvm_selected'] = True
                            result['tvvm_gamma'] = float(tvvm_result.gamma)
                            
                            # Update calibration status
                            if tvvm_result.is_calibrated:
                                result['calibration_warning'] = False
                                result['pit_ks_pvalue'] = float(tvvm_result.pit_ks_pvalue)
                                result['ks_statistic'] = float(tvvm_result.ks_statistic)
                        else:
                            _log(f"     ✗ TVVM not selected (PIT {current_pit_pvalue:.4f}→{tvvm_result.pit_ks_pvalue:.4f})")
                            result['tvvm_model'] = tvvm_result.to_dict()
                            result['tvvm_selected'] = False
                    else:
                        _log(f"     ✗ TVVM fitting failed")
                        
                except Exception as tvvm_err:
                    _log(f"     ✗ TVVM error: {tvvm_err}")
                    result['tvvm_model'] = None
                    result['tvvm_selected'] = False

        # =================================================================
        # ISOTONIC RECALIBRATION — PROBABILITY TRANSPORT OPERATOR
        # =================================================================
        # This is the CORE calibration layer. Applied to ALL models, ALWAYS.
        # 
        # DOCTRINE:
        #   - Calibration is NOT a validator/patch/escalation trigger
        #   - Calibration IS a learned probability transport map
        #   - Applied BEFORE regimes see PIT values
        #   - Persisted with model parameters
        #
        # ARCHITECTURE:
        #   Model → Raw PIT → Transport Map g → Calibrated PIT
        #                            ↓
        #           Regime-Conditioned Diagnostics
        #
        # KEY RULE: Regimes see CALIBRATED probability, not raw belief.
        # =================================================================
        recal_config = get_recalibration_config()
        result['recalibration'] = None
        result['recalibration_applied'] = False
        
        if recal_config is not None:
            _log(f"  📐 Fitting isotonic recalibration transport map...")
            
            try:
                # Fit transport map on raw PIT values
                recal_result = fit_recalibrator_for_asset(
                    returns=returns_arr,
                    mu_filtered=mu_filtered,
                    vol=vol_arr,
                    P_filtered=P_filtered,
                    c=c_optimal,
                    nu=nu_optimal,
                    config=recal_config
                )
                
                # Store result for persistence
                result['recalibration'] = recal_result.to_dict()
                result['recalibration_applied'] = True
                
                # Log outcome
                if recal_result.is_identity:
                    _log(f"     Already calibrated (identity map), raw KS p={recal_result.raw_ks_pvalue:.4f}")
                elif recal_result.fallback_to_identity:
                    _log(f"     ⚠️ Fallback to identity: {recal_result.warning_message}")
                else:
                    ks_improve = recal_result.ks_improvement
                    _log(f"     ✓ Transport map fitted: {recal_result.n_segments} segments")
                    _log(f"     KS: {recal_result.raw_ks_statistic:.4f}→{recal_result.calibrated_ks_statistic:.4f} "
                         f"(Δ={ks_improve:+.4f})")
                    _log(f"     p-value: {recal_result.raw_ks_pvalue:.4f}→{recal_result.calibrated_ks_pvalue:.4f}")
                    
                    if recal_result.validation_ks_pvalue is not None:
                        _log(f"     Validation KS p={recal_result.validation_ks_pvalue:.4f}")
                    
                    # Update calibration status based on CALIBRATED PIT
                    if recal_result.calibrated_ks_pvalue >= 0.05:
                        result['calibration_warning'] = False
                        result['pit_ks_pvalue_calibrated'] = float(recal_result.calibrated_ks_pvalue)
                        result['ks_statistic_calibrated'] = float(recal_result.calibrated_ks_statistic)
                
                # Compute enhanced diagnostics on CALIBRATED PIT
                if not recal_result.is_identity and not recal_result.fallback_to_identity:
                    # Compute raw PIT
                    if nu_optimal is not None and nu_optimal > 2:
                        raw_pit = compute_raw_pit_student_t(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal, nu_optimal)
                    else:
                        raw_pit = compute_raw_pit_gaussian(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal)
                    
                    # Apply transport map
                    calibrated_pit = apply_recalibration(raw_pit, recal_result)
                    
                    # Compute diagnostics on CALIBRATED PIT
                    diagnostics = compute_calibration_diagnostics(
                        pit_values=calibrated_pit,
                        returns=returns_arr,
                        vol_proxy=vol_arr
                    )
                    result['calibration_diagnostics'] = diagnostics
                    result['failure_category'] = diagnostics.get('failure_category', 'UNKNOWN')
                
            except Exception as recal_err:
                _log(f"     ✗ Recalibration error: {recal_err}")
                result['recalibration'] = None
                result['recalibration_applied'] = False

        # =================================================================
        # CALIBRATED TRUST AUTHORITY — SINGLE POINT OF TRUST DECISION
        # =================================================================
        # ARCHITECTURAL LAW: Trust = Calibration Authority − Bounded Regime Penalty
        #
        # This is the CANONICAL authority for trust. All downstream decisions
        # (position sizing, drift weight, signal strength) flow from here.
        #
        # SCORING (Counter-Proposal v2):
        #   Authority discipline:           98/100
        #   Mathematical transparency:      97/100
        #   Audit traceability:             97/100
        # =================================================================
        result['calibrated_trust'] = None
        
        try:
            if CALIBRATED_TRUST_AVAILABLE:
                _log(f"  🎯 Computing calibrated trust (additive decomposition)...")
                
                # Get PIT values (prefer calibrated, fallback to raw)
                if 'recalibration' in result and result.get('recalibration_applied'):
                    # Use calibrated PIT from isotonic transport
                    recal_data = result['recalibration']
                    if recal_data and 'calibrated_pit' in recal_data:
                        pit_for_trust = np.array(recal_data['calibrated_pit'])
                    else:
                        # Recompute calibrated PIT
                        if nu_optimal is not None and nu_optimal > 2:
                            raw_pit = compute_raw_pit_student_t(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal, nu_optimal)
                        else:
                            raw_pit = compute_raw_pit_gaussian(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal)
                        
                        if recal_data:
                            pit_for_trust = apply_recalibration(raw_pit, TransportMapResult.from_dict(recal_data))
                        else:
                            pit_for_trust = raw_pit
                else:
                    # No recalibration: compute raw PIT
                    if nu_optimal is not None and nu_optimal > 2:
                        pit_for_trust = compute_raw_pit_student_t(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal, nu_optimal)
                    else:
                        pit_for_trust = compute_raw_pit_gaussian(returns_arr, mu_filtered, vol_arr, P_filtered, c_optimal)
                
                # Determine regime probabilities (use uniform if not available)
                # In production, this would come from BMA or regime detection
                regime_probs = {1: 1.0}  # Default to normal regime
                
                # Check if we have calibration warning to adjust regime
                if result.get('calibration_warning', False):
                    # Higher uncertainty -> higher regime penalty
                    regime_probs = {3: 0.5, 4: 0.5}  # high_vol / crisis mix
                
                # Compute calibrated trust
                trust = compute_calibrated_trust(
                    raw_pit_values=pit_for_trust,
                    regime_probs=regime_probs,
                    isotonic_model=None,  # Already applied above
                    config=TrustConfig(),
                )
                
                # Store trust in result
                result['calibrated_trust'] = trust.to_dict()
                result['effective_trust'] = trust.effective_trust
                result['calibration_trust'] = trust.calibration_trust
                result['regime_penalty'] = trust.regime_penalty
                
                # Log trust decomposition
                _log(f"     Trust = {trust.calibration_trust:.2f} - {trust.regime_penalty:.2f} = {trust.effective_trust:.2f}")
                _log(f"     Regime: {trust.regime_context}, Tail bias: {trust.tail_bias:+.3f}")
                
                # Update calibration status based on trust
                if trust.effective_trust < 0.3:
                    result['calibration_warning'] = True
                    _log(f"     ⚠️ Low trust ({trust.effective_trust:.2f}) - calibration warning")
                
        except Exception as trust_err:
            _log(f"     ✗ Trust computation error: {trust_err}")
            result['calibrated_trust'] = None

        return result
        
    except Exception as e:
        import traceback
        print(f"  ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        raise


# =============================================================================
# REGIME-CONDITIONAL PARAMETER TUNING (HIERARCHICAL BAYESIAN LAYER)
# =============================================================================

def assign_regime_labels(
    returns: np.ndarray,
    vol: np.ndarray,
    lookback: int = 21
) -> np.ndarray:
    """
    Assign regime labels to each time point based on market features.
    
    Regime Assignment Logic:
    - Compute rolling volatility level (relative to median)
    - Compute rolling drift strength (absolute mean return)
    - Compute tail indicator (extreme returns)
    
    Classification:
    - LOW_VOL_TREND (0): vol < median, |drift| > threshold
    - HIGH_VOL_TREND (1): vol > 1.5*median, |drift| > threshold
    - LOW_VOL_RANGE (2): vol < median, |drift| < threshold
    - HIGH_VOL_RANGE (3): vol > 1.2*median, |drift| < threshold
    - CRISIS_JUMP (4): vol > 2*median OR extreme tail events
    
    Args:
        returns: Array of log returns
        vol: Array of EWMA volatility
        lookback: Rolling window for feature computation
        
    Returns:
        Array of regime labels (0-4) for each time point
    """
    n = len(returns)
    regime_labels = np.zeros(n, dtype=int)
    
    if n < lookback + 10:
        # Not enough data, default to LOW_VOL_RANGE
        return np.full(n, MarketRegime.LOW_VOL_RANGE, dtype=int)
    
    # Compute rolling features
    returns_series = pd.Series(returns)
    vol_series = pd.Series(vol)
    
    # Rolling mean absolute return (drift proxy)
    drift_abs = returns_series.rolling(lookback, min_periods=5).mean().abs().values
    
    # Volatility relative to expanding median
    vol_median = vol_series.expanding(min_periods=lookback).median().values
    vol_relative = np.where(vol_median > 1e-12, vol / vol_median, 1.0)
    
    # Tail indicator: |return| / vol
    tail_indicator = np.where(vol > 1e-12, np.abs(returns) / vol, 0.0)
    
    # Drift threshold (adaptive based on vol)
    drift_threshold = 0.0005  # ~0.05% daily drift threshold
    
    for t in range(n):
        v_rel = vol_relative[t] if np.isfinite(vol_relative[t]) else 1.0
        d_abs = drift_abs[t] if np.isfinite(drift_abs[t]) else 0.0
        tail = tail_indicator[t] if np.isfinite(tail_indicator[t]) else 0.0
        
        # Crisis/Jump: extreme volatility or tail events
        if v_rel > 2.0 or tail > 4.0:
            regime_labels[t] = MarketRegime.CRISIS_JUMP
        # High volatility regimes
        elif v_rel > 1.3:
            if d_abs > drift_threshold:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE
        # Low volatility regimes
        elif v_rel < 0.85:
            if d_abs > drift_threshold:
                regime_labels[t] = MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.LOW_VOL_RANGE
        # Normal volatility
        else:
            if d_abs > drift_threshold * 1.5:
                regime_labels[t] = MarketRegime.HIGH_VOL_TREND if v_rel > 1.0 else MarketRegime.LOW_VOL_TREND
            else:
                regime_labels[t] = MarketRegime.HIGH_VOL_RANGE if v_rel > 1.0 else MarketRegime.LOW_VOL_RANGE
    
    return regime_labels


# =============================================================================
# BAYESIAN MODEL AVERAGING WITH TEMPORAL SMOOTHING
# =============================================================================
# This section implements the core epistemic engine:
#
#     p(r_{t+H} | r) = Σ_m p(r_{t+H} | r, m, θ_{r,m}) · p(m | r)
#
# For each regime r, we:
# 1. Fit ALL candidate model classes independently
# 2. Compute BIC-based or Hyvärinen-score-based posterior weights with temporal smoothing
# 3. Return the full model posterior — never selecting a single model
# =============================================================================


# =============================================================================
# HYVÄRINEN SCORE FOR ROBUST MODEL SELECTION
# =============================================================================
# The Hyvärinen score is a proper scoring rule that is:
#   - Fisher-consistent under model misspecification
#   - Independent of normalizing constants
#   - Naturally rewards tail accuracy
#
# For a predictive density p(r|μ,σ), the Hyvärinen score is:
#
#   H = (1/n) Σ_t [ (1/2)(∂log p / ∂r)² + (∂²log p / ∂r²) ]
#
# Lower score = better fit. Negated for consistency with likelihood (higher = better).
# =============================================================================


def compute_hyvarinen_score_gaussian(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    min_sigma: float = 1e-8
) -> float:
    """
    Compute Hyvärinen score for Gaussian predictive density.
    
    For Gaussian p(r) = N(μ, σ²):
        ∂log p / ∂r = -(r - μ) / σ²
        ∂²log p / ∂r² = -1 / σ²
    
    Therefore:
        H = (1/n) Σ_t [ (r_t - μ_t)² / (2σ_t⁴) - 1/σ_t² ]
    
    Args:
        returns: Observed returns
        mu: Predicted means
        sigma: Predicted standard deviations (NOT variance)
        min_sigma: Minimum sigma for numerical stability
        
    Returns:
        Hyvärinen score (lower is better, but we return negated for consistency)
    """
    returns = np.asarray(returns).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Numerical stability: floor sigma
    sigma = np.maximum(sigma, min_sigma)
    sigma_sq = sigma ** 2
    sigma_4 = sigma ** 4
    
    # Innovations
    innovation = returns - mu
    innovation_sq = innovation ** 2
    
    # Hyvärinen score components:
    # Term 1: (1/2) * (∂log p / ∂r)² = (r - μ)² / (2σ⁴)
    # Term 2: ∂²log p / ∂r² = -1/σ²
    term1 = innovation_sq / (2.0 * sigma_4)
    term2 = -1.0 / sigma_sq
    
    # Per-observation score
    h_scores = term1 + term2
    
    # Filter out non-finite values
    valid = np.isfinite(h_scores)
    if not np.any(valid):
        return -1e12  # Return very bad score
    
    h_mean = float(np.mean(h_scores[valid]))
    
    # Return negated score (higher = better, for consistency with log-likelihood)
    return -h_mean


def compute_hyvarinen_score_student_t(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
    min_sigma: float = 1e-8,
    min_nu: float = 2.1
) -> float:
    """
    Compute Hyvärinen score for Student-t predictive density.
    
    For Student-t p(r) with location μ, scale σ, and degrees of freedom ν:
    
    Let z = (r - μ) / σ
    
        log p(r) = const - ((ν+1)/2) * log(1 + z²/ν) - log(σ)
        
        ∂log p / ∂r = -((ν+1)/ν) * z / (σ * (1 + z²/ν))
                    = -((ν+1) * (r-μ)) / (σ² * (ν + z²))
        
        ∂²log p / ∂r² = -((ν+1)/σ²) * (ν - z²) / (ν + z²)²
    
    Therefore:
        H = (1/n) Σ_t [ (1/2)(∂log p/∂r)² + ∂²log p/∂r² ]
    
    Args:
        returns: Observed returns
        mu: Predicted locations
        sigma: Predicted scales (NOT variance)
        nu: Degrees of freedom
        min_sigma: Minimum sigma for numerical stability
        min_nu: Minimum nu for numerical stability
        
    Returns:
        Hyvärinen score (negated so higher = better)
    """
    returns = np.asarray(returns).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Numerical stability
    sigma = np.maximum(sigma, min_sigma)
    nu = max(float(nu), min_nu)
    
    sigma_sq = sigma ** 2
    
    # Standardized residuals
    z = (returns - mu) / sigma
    z_sq = z ** 2
    
    # Common denominator: ν + z²
    denom = nu + z_sq
    
    # First derivative: ∂log p / ∂r = -((ν+1) * (r-μ)) / (σ² * (ν + z²))
    # Squared: ((ν+1)² * (r-μ)²) / (σ⁴ * (ν + z²)²)
    #        = ((ν+1)² * z²) / (σ² * (ν + z²)²)
    d1_sq = ((nu + 1.0) ** 2 * z_sq) / (sigma_sq * denom ** 2)
    
    # Second derivative: ∂²log p / ∂r² = -((ν+1)/σ²) * (ν - z²) / (ν + z²)²
    d2 = -((nu + 1.0) / sigma_sq) * (nu - z_sq) / (denom ** 2)
    
    # Hyvärinen score: (1/2) * (∂log p/∂r)² + ∂²log p/∂r²
    h_scores = 0.5 * d1_sq + d2
    
    # Filter out non-finite values
    valid = np.isfinite(h_scores)
    if not np.any(valid):
        return -1e12  # Return very bad score
    
    h_mean = float(np.mean(h_scores[valid]))
    
    # Return negated score (higher = better)
    return -h_mean


def compute_hyvarinen_model_weights(
    hyvarinen_scores: Dict[str, float],
    epsilon: float = 1e-10,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Convert Hyvärinen scores to unnormalized posterior weights.
    
    Since Hyvärinen scores are negated (higher = better), we use:
        w_raw(m|r) = exp(temperature * (H_m - H_min))
    
    This mirrors the BIC weight formula but uses Hyvärinen instead.
    
    Args:
        hyvarinen_scores: Dictionary mapping model name to (negated) Hyvärinen score
        epsilon: Small constant to prevent zero weights
        temperature: Scaling factor (higher = more concentrated on best model)
        
    Returns:
        Dictionary of unnormalized weights
    """
    # Find maximum score (best model, since higher = better)
    finite_scores = [s for s in hyvarinen_scores.values() if np.isfinite(s)]
    if not finite_scores:
        n_models = len(hyvarinen_scores)
        return {m: 1.0 / max(n_models, 1) for m in hyvarinen_scores}
    
    score_max = max(finite_scores)
    
    # Compute raw weights
    weights = {}
    for model_name, score in hyvarinen_scores.items():
        if np.isfinite(score):
            # Higher score = better, so exp(temp * (score - max)) gives relative weight
            # When score == max, weight = 1; when score < max, weight < 1
            delta = score - score_max
            w = np.exp(temperature * delta)
            weights[model_name] = max(w, epsilon)
        else:
            weights[model_name] = epsilon
    
    return weights


# =============================================================================
# ROBUST SCORE STANDARDIZATION & ENTROPY-REGULARIZED WEIGHTS
# =============================================================================
# These functions stabilize model selection and Bayesian model averaging by:
# 1. Robustly standardizing heterogeneous scores (BIC + Hyvärinen)
# 2. Preventing premature posterior collapse via entropy regularization
# 3. Improving low-sample regime behavior
#
# This is an epistemology-only upgrade. Signals consume posteriors unchanged.
# =============================================================================

def robust_standardize_scores(
    scores: Dict[str, float],
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Robust cross-model standardization using median and MAD.
    
    Preserves ordering while normalizing heterogeneous score scales.
    This ensures BIC and Hyvärinen can be meaningfully combined without
    one dominating due to raw scale differences.
    
    The MAD is scaled by 1.4826 to be consistent with standard deviation
    for Gaussian data: MAD * 1.4826 ≈ σ for N(μ, σ²).
    
    Why median/MAD:
    - Robust to Hyvärinen spikes
    - Stable in low-n regimes
    - No Gaussian assumptions (but calibrated to be consistent with σ)
    
    Args:
        scores: Dictionary mapping model name to raw score
        eps: Small constant to prevent division by zero
        
    Returns:
        Dictionary of standardized scores (zero median, unit scale)
    """
    # Gaussian consistency factor: MAD * 1.4826 ≈ σ for normal distributions
    MAD_CONSISTENCY_FACTOR = 1.4826
    
    # Extract finite values only
    finite_items = [(k, v) for k, v in scores.items() if np.isfinite(v)]
    
    if len(finite_items) < 2:
        # Not enough values to standardize meaningfully
        # Return zeros for finite, keep non-finite as-is
        return {
            k: 0.0 if np.isfinite(v) else v
            for k, v in scores.items()
        }
    
    values = np.array([v for _, v in finite_items], dtype=float)
    
    # Robust location and scale
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    # Scale MAD to be consistent with standard deviation
    # This ensures proper weighting when combining BIC (O(n)) with Hyvärinen (O(1))
    scale = mad * MAD_CONSISTENCY_FACTOR if mad > eps else eps
    
    # Standardize all scores
    standardized = {}
    for k, v in scores.items():
        if np.isfinite(v):
            standardized[k] = (v - median) / scale
        else:
            standardized[k] = v  # Keep non-finite as-is (inf, nan)
    
    return standardized


def entropy_regularized_weights(
    standardized_scores: Dict[str, float],
    lambda_entropy: float = DEFAULT_ENTROPY_LAMBDA,
    min_weight_fraction: float = DEFAULT_MIN_WEIGHT_FRACTION,
    eps: float = 1e-10
) -> Dict[str, float]:
    """
    Compute entropy-regularized model weights via softmax with entropy floor.
    
    Solves the optimization problem:
        min_w Σ_m w_m * S̃_m + λ Σ_m w_m * log(w_m)
        s.t. Σ_m w_m = 1, w_m ≥ min_weight
    
    The closed-form solution (without floor) is softmax with temperature = λ:
        w_m ∝ exp(-S̃_m / λ)
    
    We then apply an entropy floor to prevent belief collapse:
        w_m = max(w_m, min_weight_fraction / n_models)
    
    This ensures that even dominated models retain some probability mass,
    preventing overconfident allocations during regime transitions or
    when models happen to agree.
    
    Benefits:
    - Prevents premature posterior collapse in low-evidence regimes
    - Smooth weight transitions as evidence accumulates
    - Entropy floor prevents overconfident allocations
    - Convex, stable, deterministic
    
    Args:
        standardized_scores: Dictionary of standardized scores (lower = better)
        lambda_entropy: Entropy regularization strength (0.05 = balanced)
                       Higher = more uniform weights
                       Lower = sharper weights
        min_weight_fraction: Minimum total mass allocated to uniform (0.01 = 1%)
                            Each model gets at least min_weight_fraction / n_models
        eps: Small constant to prevent zero weights
        
    Returns:
        Dictionary of normalized model weights (sum to 1)
    """
    # Extract finite scores only
    finite_items = [(k, v) for k, v in standardized_scores.items() if np.isfinite(v)]
    
    if not finite_items:
        # No valid scores, return uniform
        n = len(standardized_scores)
        return {k: 1.0 / max(n, 1) for k in standardized_scores}
    
    keys = [k for k, _ in finite_items]
    scores = np.array([v for _, v in finite_items], dtype=float)
    n_models = len(keys)
    
    # Softmax with entropy temperature
    # Lower score = better, so we negate scores in the softmax
    temperature = max(lambda_entropy, 1e-8)
    logits = -scores / temperature
    
    # Numerical stability: subtract max
    logits = logits - logits.max()
    
    # Compute weights
    weights = np.exp(logits)
    weights = np.maximum(weights, eps)  # Prevent exact zeros
    weights = weights / weights.sum()  # Normalize
    
    # =========================================================================
    # ENTROPY FLOOR: Prevent belief collapse
    # =========================================================================
    # Ensure each model has at least min_weight_fraction / n_models weight.
    # This prevents overconfident allocations when score differences are large.
    # 
    # Example: with min_weight_fraction=0.01 and 3 models:
    #   - Each model gets at least 0.33% weight
    #   - Total "floor mass" is 1%
    #   - Remaining 99% is distributed according to softmax
    # =========================================================================
    min_weight_per_model = min_weight_fraction / max(n_models, 1)
    weights = np.maximum(weights, min_weight_per_model)
    weights = weights / weights.sum()  # Re-normalize after floor
    
    # Build result dict
    result = dict(zip(keys, weights))
    
    # Add epsilon weight for non-finite scores
    for k, v in standardized_scores.items():
        if not np.isfinite(v):
            result[k] = eps
    
    # Re-normalize if we added non-finite entries
    total = sum(result.values())
    if total > 0:
        result = {k: w / total for k, w in result.items()}
    
    return result


def compute_combined_standardized_score(
    bic: float,
    hyvarinen: float,
    bic_weight: float = 0.5
) -> float:
    """
    Compute combined score from already-standardized BIC and Hyvärinen.
    
    For BIC: lower is better → we use +BIC in combined score
    For Hyvärinen: higher is better → we use -Hyvärinen in combined score
    
    Combined: S = w_bic * BIC_std - (1 - w_bic) * Hyv_std
    Lower combined score = better model
    
    Args:
        bic: Standardized BIC score
        hyvarinen: Standardized Hyvärinen score
        bic_weight: Weight for BIC (0.5 = equal weighting)
        
    Returns:
        Combined standardized score (lower = better)
    """
    if not np.isfinite(bic):
        bic = 0.0
    if not np.isfinite(hyvarinen):
        hyvarinen = 0.0
    
    # BIC: lower is better, so positive contribution
    # Hyvärinen: higher is better, so negative contribution
    return bic_weight * bic - (1.0 - bic_weight) * hyvarinen


def compute_combined_model_weights(
    bic_values: Dict[str, float],
    hyvarinen_scores: Dict[str, float],
    bic_weight: float = 0.5,
    lambda_entropy: float = DEFAULT_ENTROPY_LAMBDA,
    epsilon: float = 1e-10
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Combine BIC and Hyvärinen scores for robust model selection.
    
    Uses entropy-regularized optimization with robust standardization:
    
    1. Robust standardization: Median/MAD normalization of each score type
       - Handles heterogeneous scales between BIC and Hyvärinen
       - Robust to outliers and spikes
       
    2. Combined score: S̃_m = w_bic * BIC̃_m - (1 - w_bic) * Hyṽ_m
       - Lower combined score = better model
       
    3. Entropy-regularized weights via softmax:
       - Solves: min_w Σ_m w_m * S̃_m + λ Σ_m w_m * log(w_m)
       - Prevents premature posterior collapse
       - Smooth weight transitions
    
    This provides:
    - BIC consistency (selects true model as n → ∞)
    - Hyvärinen robustness (proper scoring under misspecification)
    - Scale invariance (neither metric dominates due to scale differences)
    - Stability in low-sample regimes (entropy regularization)
    
    Args:
        bic_values: Dictionary mapping model name to BIC value
        hyvarinen_scores: Dictionary mapping model name to Hyvärinen score
        bic_weight: Weight for BIC in combined score (0.5 = equal)
        lambda_entropy: Entropy regularization strength (higher = more uniform)
        epsilon: Small constant to prevent zero weights
        
    Returns:
        Tuple of:
        - Dictionary of normalized model weights (sum to 1)
        - Metadata dict with standardization details
    """
    # =========================================================================
    # STEP 1: ROBUST STANDARDIZATION (per-metric, cross-model)
    # =========================================================================
    # Standardization is cross-model, per regime
    # Do NOT standardize across regimes
    # Do NOT mix raw and standardized scores
    # =========================================================================
    
    bic_standardized = robust_standardize_scores(bic_values)
    hyv_standardized = robust_standardize_scores(hyvarinen_scores)
    
    # =========================================================================
    # STEP 2: COMPUTE COMBINED STANDARDIZED SCORES
    # =========================================================================
    # Combined: S = w_bic * BIC_std - (1 - w_bic) * Hyv_std
    # Lower combined score = better model
    # =========================================================================
    
    combined_scores = {}
    for model_name in bic_values.keys():
        bic_std = bic_standardized.get(model_name, 0.0)
        hyv_std = hyv_standardized.get(model_name, 0.0)
        combined_scores[model_name] = compute_combined_standardized_score(
            bic_std, hyv_std, bic_weight
        )
    
    # =========================================================================
    # STEP 3: ENTROPY-REGULARIZED WEIGHTS
    # =========================================================================
    # Softmax with entropy temperature prevents collapse
    # =========================================================================
    
    weights = entropy_regularized_weights(
        combined_scores,
        lambda_entropy=lambda_entropy,
        eps=epsilon
    )
    
    # Build metadata for caching and diagnostics
    metadata = {
        "bic_standardized": {k: float(v) if np.isfinite(v) else None for k, v in bic_standardized.items()},
        "hyvarinen_standardized": {k: float(v) if np.isfinite(v) else None for k, v in hyv_standardized.items()},
        "combined_scores_standardized": {k: float(v) if np.isfinite(v) else None for k, v in combined_scores.items()},
        "bic_weight": bic_weight,
        "lambda_entropy": lambda_entropy,
        "entropy_regularized": True,
    }
    
    return weights, metadata


def compute_bic_model_weights_from_scores(
    scores: Dict[str, float],
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    Convert scores to weights using softmax (higher score = higher weight).
    
    This is used internally after MAD standardization.
    
    Args:
        scores: Dictionary mapping model name to score (higher = better)
        epsilon: Small constant to prevent zero weights
        
    Returns:
        Dictionary of unnormalized weights
    """
    finite_scores = [s for s in scores.values() if np.isfinite(s)]
    if not finite_scores:
        n_models = len(scores)
        return {m: 1.0 / max(n_models, 1) for m in scores}
    
    score_max = max(finite_scores)
    
    weights = {}
    for model_name, score in scores.items():
        if np.isfinite(score):
            delta = score - score_max
            w = np.exp(delta)
            weights[model_name] = max(w, epsilon)
        else:
            weights[model_name] = epsilon
    
    return weights


# Default model selection method: 'bic', 'hyvarinen', or 'combined'
DEFAULT_MODEL_SELECTION_METHOD = 'combined'
# Default BIC weight when using combined method (0.5 = equal weighting)
DEFAULT_BIC_WEIGHT = 0.5


def compute_bic_model_weights(
    bic_values: Dict[str, float],
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    Convert BIC values to unnormalized posterior weights.
    
    Implements:
        w_raw(m|r) = exp(-0.5 * (BIC_{m,r} - BIC_min_r))
    
    Args:
        bic_values: Dictionary mapping model name to BIC value
        epsilon: Small constant to prevent zero weights
        
    Returns:
        Dictionary of unnormalized weights (not yet normalized)
    """
    # Find minimum BIC
    finite_bics = [b for b in bic_values.values() if np.isfinite(b)]
    if not finite_bics:
        # All BICs are infinite — return uniform weights
        n_models = len(bic_values)
        return {m: 1.0 / max(n_models, 1) for m in bic_values}
    
    bic_min = min(finite_bics)
    
    # Compute raw weights
    weights = {}
    for model_name, bic in bic_values.items():
        if np.isfinite(bic):
            # BIC-based weight: exp(-0.5 * ΔBIC)
            delta_bic = bic - bic_min
            w = np.exp(-0.5 * delta_bic)
            weights[model_name] = max(w, epsilon)
        else:
            # Infinite BIC gets minimal weight
            weights[model_name] = epsilon
    
    return weights


def apply_temporal_smoothing(
    current_weights: Dict[str, float],
    previous_posterior: Optional[Dict[str, float]],
    alpha: float = DEFAULT_TEMPORAL_ALPHA
) -> Dict[str, float]:
    """
    Apply temporal smoothing to model weights.
    
    Implements:
        w_smooth(m|r) = (prev_p(m|r_prev))^alpha * w_raw(m|r)
    
    If no previous posterior exists, assumes uniform prior.
    
    Args:
        current_weights: Unnormalized BIC-based weights
        previous_posterior: Previous normalized posterior (or None)
        alpha: Temporal smoothing exponent (0 = no smoothing, 1 = full persistence)
        
    Returns:
        Smoothed unnormalized weights
    """
    if previous_posterior is None or alpha <= 0:
        # No smoothing — return current weights unchanged
        return current_weights.copy()
    
    # Apply temporal weighting
    smoothed = {}
    n_models = len(current_weights)
    uniform_weight = 1.0 / max(n_models, 1)
    
    for model_name, w_raw in current_weights.items():
        # Get previous posterior, defaulting to uniform
        prev_p = previous_posterior.get(model_name, uniform_weight)
        # Ensure previous posterior is positive
        prev_p = max(prev_p, 1e-10)
        
        # Apply smoothing: w_smooth = prev_p^alpha * w_raw
        w_smooth = (prev_p ** alpha) * w_raw
        smoothed[model_name] = w_smooth
    
    return smoothed


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.
    
    Args:
        weights: Unnormalized weights
        
    Returns:
        Normalized weights (posterior probabilities)
    """
    total = sum(weights.values())
    if total <= 0:
        # Fallback to uniform
        n = len(weights)
        return {m: 1.0 / max(n, 1) for m in weights}
    
    return {m: w / total for m, w in weights.items()}


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
        
        models["kalman_gaussian"] = {
            "q": float(q_gauss),
            "c": float(c_gauss),
            "phi": None,
            "nu": None,
            "log_likelihood": float(ll_full_gauss),
            "mean_log_likelihood": float(mean_ll_gauss),
            "cv_penalized_ll": float(ll_cv_gauss),
            "bic": float(bic_gauss),
            "aic": float(aic_gauss),
            "hyvarinen_score": float(hyvarinen_gauss),
            "n_params": int(n_params_gauss),
            "ks_statistic": float(ks_gauss),
            "pit_ks_pvalue": float(pit_p_gauss),
            "fit_success": True,
            "diagnostics": diag_gauss,
        }
    except Exception as e:
        models["kalman_gaussian"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
            "hyvarinen_score": float('-inf'),
        }
    
    # =========================================================================
    # Model 1: Phi-Gaussian (q, c, phi)
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
        
        models["kalman_phi_gaussian"] = {
            "q": float(q_phi),
            "c": float(c_phi),
            "phi": float(phi_opt),
            "nu": None,
            "log_likelihood": float(ll_full_phi),
            "mean_log_likelihood": float(mean_ll_phi),
            "cv_penalized_ll": float(ll_cv_phi),
            "bic": float(bic_phi),
            "aic": float(aic_phi),
            "hyvarinen_score": float(hyvarinen_phi),
            "n_params": int(n_params_phi),
            "ks_statistic": float(ks_phi),
            "pit_ks_pvalue": float(pit_p_phi),
            "fit_success": True,
            "diagnostics": diag_phi,
        }
    except Exception as e:
        models["kalman_phi_gaussian"] = {
            "fit_success": False,
            "error": str(e),
            "bic": float('inf'),
            "aic": float('inf'),
            "hyvarinen_score": float('-inf'),
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
    - Never return empty models - use hierarchical fallback
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
                        diagnostics...
                    }
                },
                "regime_meta": {
                    "temporal_alpha": alpha,
                    "n_samples": n,
                    "regime_name": str,
                    "borrowed_from_global": bool  # True if fallback used
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
        previous_posteriors: Previous posteriors for smoothing
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
    global_weight_metadata = None
    
    if model_selection_method == 'bic':
        global_raw_weights = compute_bic_model_weights(global_bic)
    elif model_selection_method == 'hyvarinen':
        global_raw_weights = compute_hyvarinen_model_weights(global_hyvarinen)
    else:
        # Default: combined with entropy regularization
        global_raw_weights, global_weight_metadata = compute_combined_model_weights(
            global_bic, global_hyvarinen, bic_weight=bic_weight,
            lambda_entropy=DEFAULT_ENTROPY_LAMBDA
        )
    
    # Store combined_score and entropy-regularized weights in each global model
    for m in global_models:
        w = global_raw_weights.get(m, 1e-10)
        if global_weight_metadata is not None:
            global_models[m]['combined_score'] = float(global_weight_metadata['combined_scores_standardized'].get(m, 0.0))
            global_models[m]['model_weight_entropy'] = float(w)
            global_models[m]['standardized_bic'] = float(global_weight_metadata['bic_standardized'].get(m, 0.0)) if global_weight_metadata['bic_standardized'].get(m) is not None else None
            global_models[m]['standardized_hyvarinen'] = float(global_weight_metadata['hyvarinen_standardized'].get(m, 0.0)) if global_weight_metadata['hyvarinen_standardized'].get(m) is not None else None
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
        min_samples=min_samples,
        temporal_alpha=temporal_alpha,
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
        previous_posteriors: Previous posteriors for temporal smoothing
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
        
        # Check minimum data requirements
        if n_points < MIN_DATA_FOR_GLOBAL:
            _log(f"     ⚠️  Insufficient data for {asset} ({n_points} points)")
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
        
        if len(returns) < MIN_DATA_FOR_GLOBAL:
            print(f"     ⚠️  Insufficient valid data for {asset} ({len(returns)} returns)")
            return None
        
        # Check if we have enough data for full regime BMA
        use_regime_bma = len(returns) >= MIN_DATA_FOR_REGIME
        
        if not use_regime_bma:
            _log(f"     ⚠️  Insufficient data for regime BMA ({len(returns)} < {MIN_DATA_FOR_REGIME})")
            _log(f"     ↩️  Using global-only BMA...")
            
            # Fit global models only
            global_models = fit_all_models_for_regime(
                returns, vol,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda,
            )
            
            # Compute global posterior using specified method
            global_bic = {m: global_models[m].get("bic", float('inf')) for m in global_models}
            global_hyvarinen = {m: global_models[m].get("hyvarinen_score", float('-inf')) for m in global_models}
            fallback_weight_metadata = None
            
            if model_selection_method == 'bic':
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
                    global_models[m]['combined_score'] = float(fallback_weight_metadata['combined_scores_standardized'].get(m, 0.0))
                    global_models[m]['model_weight_entropy'] = float(w)
                    global_models[m]['standardized_bic'] = float(fallback_weight_metadata['bic_standardized'].get(m, 0.0)) if fallback_weight_metadata['bic_standardized'].get(m) is not None else None
                    global_models[m]['standardized_hyvarinen'] = float(fallback_weight_metadata['hyvarinen_standardized'].get(m, 0.0)) if fallback_weight_metadata['hyvarinen_standardized'].get(m) is not None else None
                    global_models[m]['entropy_lambda'] = DEFAULT_ENTROPY_LAMBDA
                else:
                    global_models[m]['combined_score'] = float(np.log(w)) if w > 0 else float('-inf')
            
            global_posterior = normalize_weights(global_raw_weights)
            
            # Compute global aggregate scores
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
            
            # For standardized scores, best is lowest (closest to zero or most negative)
            global_combined_scores = [
                global_models[m].get("combined_score", float('inf')) 
                for m in global_models 
                if global_models[m].get("fit_success", False) and np.isfinite(global_models[m].get("combined_score", float('inf')))
            ]
            global_combined_score_min = min(global_combined_scores) if global_combined_scores else None
            
            return {
                "asset": asset,
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
                "regime": None,
                "regime_counts": None,
                "use_regime_bma": False,
                "meta": {
                    "temporal_alpha": temporal_alpha,
                    "lambda_regime": lambda_regime,
                    "n_obs": len(returns),
                    "model_selection_method": model_selection_method,
                    "bic_weight": bic_weight if model_selection_method == 'combined' else None,
                    "entropy_lambda": DEFAULT_ENTROPY_LAMBDA if model_selection_method == 'combined' else None,
                    "fallback_reason": "insufficient_data_for_regime_bma",
                },
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        
        # Assign regime labels
        _log(f"     📊 Assigning regime labels for {len(returns)} observations...")
        regime_labels = assign_regime_labels(returns, vol)
        
        # Count samples per regime
        regime_counts = {r: int(np.sum(regime_labels == r)) for r in range(5)}
        _log(f"     Regime distribution: " + ", ".join([
            f"{REGIME_LABELS[r]}={c}" for r, c in regime_counts.items() if c > 0
        ]))
        
        # Run full Bayesian Model Averaging
        _log(f"     🔧 Running Bayesian Model Averaging (α={temporal_alpha:.2f}, λ={lambda_regime:.3f})...")
        _log(f"     📊 Model selection: {model_selection_method}" + 
             (f" (BIC weight={bic_weight:.2f})" if model_selection_method == 'combined' else ""))
        bma_result = tune_regime_model_averaging(
            returns, vol, regime_labels,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            min_samples=MIN_REGIME_SAMPLES,
            temporal_alpha=temporal_alpha,
            previous_posteriors=previous_posteriors,
            lambda_regime=lambda_regime,
            model_selection_method=model_selection_method,
            bic_weight=bic_weight,
        )
        
        # Build final result
        result = {
            "asset": asset,
            "global": bma_result["global"],
            "regime": bma_result["regime"],
            "regime_counts": regime_counts,
            "use_regime_bma": True,
            "meta": bma_result["meta"],
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        # Print summary
        _log(f"     ✓ Global: " + ", ".join([
            f"{m}={p:.3f}" for m, p in result["global"]["model_posterior"].items()
        ]))
        for r, r_data in result["regime"].items():
            if not r_data.get("regime_meta", {}).get("fallback", False):
                posterior_str = ", ".join([
                    f"{m}={p:.3f}" for m, p in r_data["model_posterior"].items()
                ])
                _log(f"     ✓ {REGIME_LABELS[r]}: {posterior_str}")
        
        return result
        
    except Exception as e:
        import traceback
        _log(f"     ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        return None


def get_model_params_for_regime(
    bma_result: Dict,
    regime: int,
    model: str,
) -> Optional[Dict]:
    """
    Get parameters for a specific model in a specific regime.
    
    Args:
        bma_result: Result from tune_asset_with_bma()
        regime: Regime index (0-4)
        model: Model name ("kalman_gaussian", "kalman_phi_gaussian", or "phi_student_t_nu_{4,6,8,12,20}")
        
    Returns:
        Model parameters dict or None if not available
    """
    # Try regime-specific first
    if bma_result.get("regime") is not None and regime in bma_result["regime"]:
        regime_data = bma_result["regime"][regime]
        if not regime_data.get("regime_meta", {}).get("fallback", False):
            models = regime_data.get("models", {})
            if model in models and models[model].get("fit_success", False):
                return models[model]
    
    # Fallback to global
    if "global" in bma_result and "models" in bma_result["global"]:
        models = bma_result["global"]["models"]
        if model in models and models[model].get("fit_success", False):
            return models[model]
    
    return None


def get_model_posterior_for_regime(
    bma_result: Dict,
    regime: int,
) -> Dict[str, float]:
    """
    Get the model posterior p(m|r) for a specific regime.
    
    Args:
        bma_result: Result from tune_asset_with_bma()
        regime: Regime index (0-4)
        
    Returns:
        Dictionary mapping model names to posterior probabilities
    """
    # Try regime-specific first
    if bma_result.get("regime") is not None and regime in bma_result["regime"]:
        regime_data = bma_result["regime"][regime]
        if "model_posterior" in regime_data:
            return regime_data["model_posterior"]
    
    # Fallback to global
    if "global" in bma_result and "model_posterior" in bma_result["global"]:
        return bma_result["global"]["model_posterior"]
    
    # Ultimate fallback: uniform over all model types including Student-t nu grid
    return get_uniform_model_prior()


def get_uniform_model_prior() -> Dict[str, float]:
    """
    Return a uniform prior over all candidate models.
    
    This includes:
    - kalman_gaussian
    - kalman_phi_gaussian
    - phi_student_t_nu_{nu} for each nu in STUDENT_T_NU_GRID
    
    Returns:
        Dictionary mapping model names to uniform probabilities (sum to 1)
    """
    n_student_t = len(STUDENT_T_NU_GRID)
    n_total = 2 + n_student_t  # Gaussian, Phi-Gaussian, + nu grid
    uniform_weight = 1.0 / n_total
    
    prior = {
        "kalman_gaussian": uniform_weight,
        "kalman_phi_gaussian": uniform_weight,
    }
    for nu in STUDENT_T_NU_GRID:
        prior[f"phi_student_t_nu_{nu}"] = uniform_weight
    
    return prior


def is_student_t_model(model_name: str) -> bool:
    """Check if a model name is a Student-t model (from the nu grid)."""
    return model_name.startswith("phi_student_t_nu_")


def get_student_t_nu(model_name: str) -> Optional[float]:
    """Extract nu value from a Student-t model name, or None if not Student-t."""
    if not is_student_t_model(model_name):
        return None
    try:
        return float(model_name.split("_")[-1])
    except (ValueError, IndexError):
        return None


def tune_regime_parameters(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    min_samples: int = MIN_REGIME_SAMPLES,
    # === UPGRADE LAYER 1: Regime Confidence Weighting ===
    regime_confidence: Optional[np.ndarray] = None,
    # === UPGRADE LAYER 2: Hierarchical Shrinkage ===
    lambda_regime: float = 0.05,
) -> Dict[int, Dict]:
    """
    Estimate parameters conditionally on each regime with hierarchical Bayesian maturation.
    
    For each regime r, fits q_r, phi_r, nu_r using only samples where
    regime_labels[t] == r. Falls back to global parameters if effective
    sample size is insufficient.
    
    === UPGRADE LAYERS (Architecture-Preserving) ===
    
    Layer 1 - Regime Confidence Weighting:
        When regime_confidence[t] is provided, likelihood is weighted:
        sum_t weight[t] * log p(x_t | theta_r)
        When None, weight[t] = 1.0 (default behavior unchanged)
    
    Layer 2 - Hierarchical Shrinkage Toward Global:
        penalty = lambda_regime * sum((theta_r - theta_global)^2)
        Prevents overfitting, stabilizes small regimes.
        When lambda_regime = 0, behavior identical to original.
    
    Layer 3 - Regime-Specific Prior Geometry:
        LOW_VOL regimes: encourage smaller q, larger nu
        HIGH_VOL regimes: allow larger q, moderate nu
        CRISIS regime: encourage largest q, smallest nu
    
    Layer 4 - Effective Sample Control:
        N_eff = sum(weight) replaces count logic
        Fallback when N_eff < min_samples
    
    Layer 5 - Regime Diagnostics:
        Sanity checks, parameter distances, collapse detection
    
    Args:
        returns: Array of returns
        vol: Array of EWMA volatility
        regime_labels: Array of regime labels (0-4) for each time step
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        min_samples: Minimum effective samples required per regime
        regime_confidence: Optional confidence weights [0,1] per time step
        lambda_regime: Hierarchical shrinkage strength (default 0.05)
        
    Returns:
        Dictionary with regime-specific parameters and diagnostics
    """
    regime_params = {}
    
    # Validate inputs
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    regime_labels = np.asarray(regime_labels).flatten().astype(int)
    
    if len(returns) != len(regime_labels):
        raise ValueError(f"Length mismatch: returns={len(returns)}, regime_labels={len(regime_labels)}")
    
    # === UPGRADE LAYER 1: Process regime confidence weights ===
    if regime_confidence is not None:
        weights = np.asarray(regime_confidence).flatten()
        if len(weights) != len(returns):
            raise ValueError(f"Length mismatch: regime_confidence={len(weights)}, returns={len(returns)}")
        weights = np.clip(weights, 0.0, 1.0)
    else:
        # Default: all weights = 1.0 (original behavior)
        weights = np.ones(len(returns), dtype=float)
    
    # First, compute global parameters as fallback
    print("  📊 Computing global parameters as fallback...")
    try:
        q_global, c_global, phi_global, nu_global, ll_global, _ = PhiStudentTDriftModel.optimize_params(
            returns, vol,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        global_params = {
            "q": float(q_global),
            "c": float(c_global),
            "phi": float(phi_global),
            "nu": float(nu_global),
            "cv_penalized_ll": float(ll_global),  # Penalized mean LL from CV (includes priors)
            "n_samples": int(len(returns)),
            "n_eff": float(np.sum(weights)),
            "fallback": False
        }
        _log(f"     Global: q={q_global:.2e}, φ={phi_global:+.3f}, ν={nu_global:.1f}")
    except Exception as e:
        _log(f"  ⚠️  Global parameter estimation failed: {e}")
        global_params = {
            "q": 1e-6,
            "c": 1.0,
            "phi": 0.95,
            "nu": 8.0,
            "cv_penalized_ll": float('nan'),  # Penalized mean LL from CV (unavailable)
            "n_samples": int(len(returns)),
            "n_eff": float(np.sum(weights)),
            "fallback": True
        }
    
    # === UPGRADE LAYER 3: Regime-Specific Prior Geometry ===
    # These are penalty adjustments, not hard bounds
    regime_prior_adjustments = {
        MarketRegime.LOW_VOL_TREND: {
            "q_bias": -0.5,    # encourage smaller q
            "nu_bias": +2.0,   # encourage larger nu (thinner tails)
            "phi_bias": +0.02, # encourage higher persistence
        },
        MarketRegime.HIGH_VOL_TREND: {
            "q_bias": +0.3,    # allow larger q
            "nu_bias": 0.0,    # neutral nu
            "phi_bias": +0.01, # slightly higher persistence
        },
        MarketRegime.LOW_VOL_RANGE: {
            "q_bias": -0.3,    # smaller q
            "nu_bias": +1.0,   # larger nu
            "phi_bias": -0.02, # lower persistence (mean reversion)
        },
        MarketRegime.HIGH_VOL_RANGE: {
            "q_bias": +0.2,    # moderate q
            "nu_bias": -1.0,   # smaller nu (fatter tails)
            "phi_bias": -0.03, # lower persistence (whipsaw)
        },
        MarketRegime.CRISIS_JUMP: {
            "q_bias": +1.0,    # largest q (rapid adaptation)
            "nu_bias": -3.0,   # smallest nu (fattest tails)
            "phi_bias": -0.05, # lowest persistence
        },
    }
    
    # Estimate parameters for each regime
    for regime in range(5):
        regime_name = REGIME_LABELS.get(regime, f"REGIME_{regime}")
        mask = (regime_labels == regime)
        n_samples = int(np.sum(mask))
        
        # === UPGRADE LAYER 4: Effective Sample Control ===
        regime_weights = weights[mask]
        n_eff = float(np.sum(regime_weights))
        
        _log(f"  📊 {regime_name} (n={n_samples}, n_eff={n_eff:.1f})...")
        
        if n_eff < min_samples:
            _log(f"     ⚠️  Insufficient effective samples ({n_eff:.1f} < {min_samples}), using global fallback")
            regime_params[regime] = {
                **global_params,
                "n_samples": n_samples,
                "n_eff": n_eff,
                "fallback": True,
                "regime_name": regime_name,
                "fallback_reason": "insufficient_effective_samples"
            }
            continue
        
        # Extract regime-specific data
        ret_regime = returns[mask]
        vol_regime = vol[mask]
        
        # === UPGRADE LAYER 3: Apply regime-specific prior adjustments ===
        adjustments = regime_prior_adjustments.get(regime, {"q_bias": 0, "nu_bias": 0, "phi_bias": 0})
        regime_prior_log_q = prior_log_q_mean + adjustments["q_bias"]
        
        try:
            # Fit regime-specific parameters with adjusted priors
            q_r, c_r, phi_r, nu_r, ll_r, diag_r = PhiStudentTDriftModel.optimize_params(
                ret_regime, vol_regime,
                prior_log_q_mean=regime_prior_log_q,
                prior_lambda=prior_lambda
            )
            
            # === UPGRADE LAYER 2: Hierarchical Shrinkage Toward Global ===
            if lambda_regime > 0 and not global_params.get("fallback", True):
                # Compute shrinkage penalty and apply soft correction
                # Shrinkage factor: closer to 1 = more original, closer to 0 = more global
                shrinkage_factor = 1.0 / (1.0 + lambda_regime * min_samples / max(n_eff, 1.0))
                sf = shrinkage_factor  # shorthand
                
                # PATCH 1: Log-space shrinkage for q (preserves positivity, respects scale geometry)
                # q_shrunk = exp(sf * log(q_r) + (1-sf) * log(global_q))
                q_shrunk = math.exp(sf * math.log(q_r) + (1 - sf) * math.log(global_params["q"]))
                
                # phi shrinkage remains linear (bounded domain [-1, 1])
                phi_shrunk = phi_r * sf + global_params["phi"] * (1 - sf)
                
                # PATCH 1: Log-space shrinkage for nu (prevents df distortion near boundaries)
                # nu_shrunk = exp(sf * log(nu_r) + (1-sf) * log(global_nu))
                nu_shrunk = math.exp(sf * math.log(nu_r) + (1 - sf) * math.log(global_params["nu"]))
                
                # c shrinkage remains linear (scale parameter)
                c_shrunk = c_r * sf + global_params["c"] * (1 - sf)
                
                # Store both original and shrunk values
                shrinkage_applied = True
                q_original, phi_original, nu_original = q_r, phi_r, nu_r
                q_r, phi_r, nu_r, c_r = q_shrunk, phi_shrunk, nu_shrunk, c_shrunk
            else:
                shrinkage_applied = False
                q_original, phi_original, nu_original = q_r, phi_r, nu_r
            
            # === UPGRADE LAYER 5: Compute regime diagnostics ===
            # Parameter distance from global
            param_distance = np.sqrt(
                (np.log10(q_r) - np.log10(global_params["q"]))**2 +
                (phi_r - global_params["phi"])**2 * 100 +  # Scale phi difference
                (nu_r - global_params["nu"])**2 / 100      # Scale nu difference
            )
            
            regime_params[regime] = {
                "q": float(q_r),
                "c": float(c_r),
                "phi": float(phi_r),
                "nu": float(nu_r),
                "cv_penalized_ll": float(ll_r),  # Penalized mean LL from CV (includes priors)
                "n_samples": n_samples,
                "n_eff": n_eff,
                "fallback": False,
                "regime_name": regime_name,
                # Shrinkage metadata
                "shrinkage_applied": shrinkage_applied,
                "q_original": float(q_original) if shrinkage_applied else None,
                "phi_original": float(phi_original) if shrinkage_applied else None,
                "nu_original": float(nu_original) if shrinkage_applied else None,
                # Prior adjustments applied
                "prior_q_bias": adjustments["q_bias"],
                "prior_nu_bias": adjustments["nu_bias"],
                "prior_phi_bias": adjustments["phi_bias"],
                # Diagnostics
                "param_distance_from_global": float(param_distance),
                "diagnostics": diag_r
            }
            _log(f"     q={q_r:.2e}, φ={phi_r:+.3f}, ν={nu_r:.1f}" + 
                  (f" [shrunk]" if shrinkage_applied else ""))
            
        except Exception as e:
            _log(f"     ⚠️  Estimation failed ({e}), using global fallback")
            regime_params[regime] = {
                **global_params,
                "n_samples": n_samples,
                "n_eff": n_eff,
                "fallback": True,
                "regime_name": regime_name,
                "fallback_reason": f"estimation_failed: {str(e)}"
            }
    
    # === UPGRADE LAYER 5: Post-tuning diagnostics ===
    regime_meta = _compute_regime_diagnostics(regime_params, global_params)
    
    # Attach metadata to each regime
    for r in regime_params:
        regime_params[r]["regime_meta"] = regime_meta.get(r, {})
    
    return regime_params


def _compute_regime_diagnostics(
    regime_params: Dict[int, Dict],
    global_params: Dict
) -> Dict[int, Dict]:
    """
    Compute regime diagnostics for Layer 5.
    
    Checks:
    1. Sanity relationships between regimes
    2. Parameter distances
    3. Collapse detection
    
    Returns:
        Dictionary of diagnostics per regime
    """
    diagnostics = {}
    
    # Extract parameters for non-fallback regimes
    active_regimes = {r: p for r, p in regime_params.items() if not p.get("fallback", True)}
    
    # Get parameter values for sanity checks
    def get_param(r, key, default=None):
        if r in active_regimes:
            return active_regimes[r].get(key, default)
        return default
    
    q_vals = {r: get_param(r, "q") for r in range(5)}
    nu_vals = {r: get_param(r, "nu") for r in range(5)}
    phi_vals = {r: get_param(r, "phi") for r in range(5)}
    
    # Sanity check 1: q_crisis > q_low_vol (crisis should adapt faster)
    q_crisis = q_vals.get(MarketRegime.CRISIS_JUMP)
    q_low_trend = q_vals.get(MarketRegime.LOW_VOL_TREND)
    q_low_range = q_vals.get(MarketRegime.LOW_VOL_RANGE)
    
    sanity_q_crisis_vs_low = None
    if q_crisis is not None and q_low_trend is not None:
        sanity_q_crisis_vs_low = q_crisis > q_low_trend
    
    # Sanity check 2: nu_crisis < nu_trend (crisis has fatter tails)
    nu_crisis = nu_vals.get(MarketRegime.CRISIS_JUMP)
    nu_low_trend = nu_vals.get(MarketRegime.LOW_VOL_TREND)
    nu_high_trend = nu_vals.get(MarketRegime.HIGH_VOL_TREND)
    
    sanity_nu_crisis_vs_trend = None
    if nu_crisis is not None and nu_low_trend is not None:
        sanity_nu_crisis_vs_trend = nu_crisis < nu_low_trend
    
    # Sanity check 3: phi_trend > phi_range (trends are more persistent)
    phi_low_trend = phi_vals.get(MarketRegime.LOW_VOL_TREND)
    phi_high_trend = phi_vals.get(MarketRegime.HIGH_VOL_TREND)
    phi_low_range = phi_vals.get(MarketRegime.LOW_VOL_RANGE)
    phi_high_range = phi_vals.get(MarketRegime.HIGH_VOL_RANGE)
    
    sanity_phi_trend_vs_range = None
    if phi_low_trend is not None and phi_low_range is not None:
        sanity_phi_trend_vs_range = phi_low_trend > phi_low_range
    
    # Collapse detection: check if all parameters are too close
    collapse_threshold = 0.1  # If all distances < this, warn
    distances = []
    for r, p in active_regimes.items():
        dist = p.get("param_distance_from_global", 0)
        distances.append(dist)
    
    collapse_detected = len(distances) > 1 and all(d < collapse_threshold for d in distances)
    
    # Build diagnostics for each regime
    for r in range(5):
        diagnostics[r] = {
            "sanity_checks": {
                "q_crisis_gt_low_vol": sanity_q_crisis_vs_low,
                "nu_crisis_lt_trend": sanity_nu_crisis_vs_trend,
                "phi_trend_gt_range": sanity_phi_trend_vs_range,
            },
            "collapse_warning": collapse_detected,
            "n_active_regimes": len(active_regimes),
            # PATCH 5: Add metadata flag for likelihood type
            "ll_type": "cv_penalized_mean",  # Penalized mean LL from CV (includes priors + calibration penalty)
        }
    
    # Print warnings if sanity checks fail
    if sanity_q_crisis_vs_low is False:
        _log("     ⚠️  Sanity warning: q_crisis should be > q_low_vol")
    if sanity_nu_crisis_vs_trend is False:
        _log("     ⚠️  Sanity warning: nu_crisis should be < nu_trend")
    if sanity_phi_trend_vs_range is False:
        _log("     ⚠️  Sanity warning: phi_trend should be > phi_range")
    if collapse_detected:
        _log("     ⚠️  Collapse warning: All regime parameters too close to global")
    
    return diagnostics


def tune_asset_with_regimes(
    asset: str,
    regime_labels: Optional[np.ndarray] = None,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    use_regime_tuning: bool = True
) -> Optional[Dict]:
    """
    Estimate optimal parameters for a single asset with optional regime-conditional tuning.
    
    Supports two modes:
    - GLOBAL MODE (use_regime_tuning=False or regime_labels=None):
        Fits single parameter set on all data (existing behavior).
    - REGIME MODE (use_regime_tuning=True and regime_labels provided):
        For each regime r, fits q_r, phi_r, nu_r using only samples where
        regime_labels[t] == r.
    
    Args:
        asset: Asset symbol
        regime_labels: Optional array of regime labels (0-4) for each time step
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        use_regime_tuning: Whether to use regime-conditional tuning
        
    Returns:
        Dictionary with results:
        {
            "global": {...},           # Always present - global parameter estimates
            "regime": {                # Present if use_regime_tuning=True
                0: {...},
                1: {...},
                2: {...},
                3: {...},
                4: {...}
            },
            "use_regime_tuning": bool,
            ...
        }
    """
    try:
        # First, get global parameters using existing function
        print(f"  🔧 Tuning {asset}...")
        global_result = tune_asset_q(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda
        )
        
        if global_result is None:
            return None
        
        # Structure result with global params
        result = {
            "asset": asset,
            "global": global_result,
            "use_regime_tuning": use_regime_tuning and regime_labels is not None,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        # If regime tuning is enabled and labels provided, estimate per-regime params
        if use_regime_tuning and regime_labels is not None:
            print(f"  🔄 Regime-conditional tuning for {asset}...")
            
            # Need to fetch data again for regime tuning
            try:
                px, _ = fetch_px(asset, start_date, end_date)
            except Exception:
                df = _download_prices(asset, start_date, end_date)
                if df is None or df.empty:
                    print(f"  ⚠️  Cannot fetch data for regime tuning")
                    return result
                px = df['Close']
            
            # Compute returns and vol
            log_ret = np.log(px / px.shift(1)).dropna()
            returns = log_ret.values
            
            vol_ewma = log_ret.ewm(span=21, adjust=False).std()
            vol = vol_ewma.values
            
            # Align regime_labels with returns
            if len(regime_labels) != len(returns):
                print(f"  ⚠️  Regime labels length mismatch ({len(regime_labels)} vs {len(returns)})")
                # Try to align by truncating
                min_len = min(len(regime_labels), len(returns))
                regime_labels = regime_labels[-min_len:]
                returns = returns[-min_len:]
                vol = vol[-min_len:]
            
            # Tune per-regime parameters
            regime_params = tune_regime_parameters(
                returns=returns,
                vol=vol,
                regime_labels=regime_labels,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda
            )
            
            result["regime"] = regime_params
        
        return result
        
    except Exception as e:
        import traceback
        print(f"  ❌ {asset}: Failed - {e}")
        if os.getenv('DEBUG'):
            traceback.print_exc()
        return None


def get_regime_params(
    cached_result: Dict,
    regime: int,
    use_regime_tuning: bool = True
) -> Dict:
    """
    Get parameters for a specific regime, with fallback to global.
    
    Args:
        cached_result: Result from tune_asset_with_regimes()
        regime: Regime index (0-4)
        use_regime_tuning: Whether to use regime-specific params
        
    Returns:
        Dictionary with parameters (q, c, phi, nu, etc.)
    """
    # If regime tuning disabled or not available, use global
    if not use_regime_tuning or "regime" not in cached_result:
        return cached_result.get("global", cached_result)
    
    # Get regime-specific params
    regime_params = cached_result.get("regime", {})
    if regime in regime_params:
        params = regime_params[regime]
        # If this regime used fallback, it already contains global params
        return params
    
    # Fallback to global
    return cached_result.get("global", cached_result)


def _tune_asset_with_regime_labels(
    asset: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    lambda_regime: float = 0.05,
    previous_posteriors: Optional[Dict[int, Dict[str, float]]] = None,
) -> Optional[Dict]:
    """
    Tune asset with automatic regime label assignment and Bayesian Model Averaging.

    This function:
    1. Fetches price data
    2. Computes returns and volatility
    3. Assigns regime labels using assign_regime_labels()
    4. Calls tune_regime_model_averaging() for full BMA with temporal smoothing

    Args:
        asset: Asset symbol
        start_date: Start date for data
        end_date: End date (default: today)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        lambda_regime: Hierarchical shrinkage strength (default 0.05)
        previous_posteriors: Previous model posteriors per regime for temporal smoothing

    Returns:
        Dictionary with global and regime-conditional model posteriors and parameters
    """
    # Minimum data thresholds
    MIN_DATA_FOR_REGIME = 100  # Need at least 100 points for reliable regime estimation
    MIN_DATA_FOR_GLOBAL = 20   # Can do basic tuning with fewer points
    
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
            return {
                "asset": asset,
                "global": global_result,
                "regime": None,  # Explicitly None - no regime params available
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_{n_points}_points",
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
            
            return {
                "asset": asset,
                "global": global_result,
                "regime": None,
                "use_regime_tuning": False,
                "regime_fallback": True,
                "regime_fallback_reason": f"insufficient_data_after_cleaning_{len(returns)}_returns",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        # Assign regime labels
        _log(f"     📊 Assigning regime labels for {len(returns)} observations...")
        regime_labels = assign_regime_labels(returns, vol)

        # Count samples per regime
        regime_counts = {r: int(np.sum(regime_labels == r)) for r in range(5)}
        _log(f"     Regime distribution: " + ", ".join([f"{REGIME_LABELS[r]}={c}" for r, c in regime_counts.items() if c > 0]))

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
        collapse_warning = any(p.get("regime_meta", {}).get("collapse_warning", False) 
                              for p in regime_results.values())

        # Build combined result with BMA structure
        result = {
            "asset": asset,
            "global": {
                # Keep backward-compatible global result
                **global_result,
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
                "collapse_warning": collapse_warning,
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

        if collapse_warning:
            _log(f"     ⚠️  Collapse warning: regime parameters too close to global")

        return result

    except Exception as e:
        import traceback
        print(f"     ❌ {asset}: Failed - {e}")
        # Always print full traceback - don't swallow exceptions
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
        result = _tune_asset_with_regime_labels(
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
            # Mark as fallback so downstream knows regime params are not available
            fallback_result['use_regime_tuning'] = False
            fallback_result['regime_fallback'] = True
            fallback_result['regime'] = None
            fallback_result['regime_counts'] = None
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
    parser.add_argument('--cache-json', type=str, default='src/quant/cache/tune',
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
            ("Asset", 18), ("Model", 14), ("log10(q)", 9), ("c", 7), ("ν", 7), ("φ", 7),
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


def render_calibration_issues_table(cache: Dict, failure_reasons: Dict) -> None:
    """Render a comprehensive, Apple-quality calibration issues table.
    
    Shows all assets with calibration problems:
    - PIT p-value < 0.05 (model predictions not well-calibrated)
    - High kurtosis (fat tails not captured)
    - Failed tuning
    - Regime collapse warnings
    
    Design: Clean, scannable, actionable.
    """
    import sys
    
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.align import Align
        from rich.rule import Rule
        from rich import box
    except ImportError:
        print("\n[Calibration issues table requires 'rich' library]")
        sys.stdout.flush()
        return
    
    try:
        console = Console(force_terminal=True, width=140)
        
        # Collect calibration issues
        issues = []
        
        # 1. Failed assets
        for asset, reason in failure_reasons.items():
            issues.append({
                'asset': asset,
                'issue_type': 'FAILED',
                'severity': 'critical',
                'pit_p': None,
                'ks_stat': None,
                'kurtosis': None,
                'model': '-',
                'q': None,
                'phi': None,
                'nu': None,
                'details': reason[:50] + '...' if len(reason) > 50 else reason
            })
        
        # 2. Calibration warnings from cache
        for asset, raw_data in cache.items():
            if 'global' in raw_data:
                data = raw_data['global']
            else:
                data = raw_data
            
            pit_p = data.get('pit_ks_pvalue')
            ks_stat = data.get('ks_statistic')
            kurtosis = data.get('std_residual_kurtosis') or data.get('excess_kurtosis')
            calibration_warning = data.get('calibration_warning', False)
            noise_model = data.get('noise_model', '')
            q_val = data.get('q')
            phi_val = data.get('phi')
            nu_val = data.get('nu')
            
            collapse_warning = raw_data.get('hierarchical_tuning', {}).get('collapse_warning', False)
            
            has_issue = False
            issue_type = []
            severity = 'ok'
            
            if calibration_warning or (pit_p is not None and pit_p < 0.05):
                has_issue = True
                issue_type.append('PIT < 0.05')
                severity = 'warning'
            
            if pit_p is not None and pit_p < 0.01:
                severity = 'critical'
            
            if kurtosis is not None and kurtosis > 6:
                has_issue = True
                issue_type.append('High Kurt')
                if severity != 'critical':
                    severity = 'warning'
            
            if collapse_warning:
                has_issue = True
                issue_type.append('Regime Collapse')
            
            if has_issue:
                if 'student_t' in noise_model:
                    model_str = f"φ-T(ν={int(nu_val)})" if nu_val else "Student-t"
                elif 'phi' in noise_model:
                    model_str = "φ-Gaussian"
                elif 'gaussian' in noise_model:
                    model_str = "Gaussian"
                else:
                    model_str = noise_model[:12] if noise_model else '-'
                
                issues.append({
                    'asset': asset,
                    'issue_type': ', '.join(issue_type),
                    'severity': severity,
                    'pit_p': pit_p,
                    'ks_stat': ks_stat,
                    'kurtosis': kurtosis,
                    'model': model_str,
                    'q': q_val,
                    'phi': phi_val,
                    'nu': nu_val,
                    'details': ''
                })
        
        # Sort by severity (critical first), then by PIT p-value
        severity_order = {'critical': 0, 'warning': 1, 'ok': 2}
        issues.sort(key=lambda x: (severity_order.get(x['severity'], 2), x.get('pit_p') or 1.0))
        
        # SECTION HEADER - Always show
        console.print()
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        
        section_header = Text()
        section_header.append("  📊  ", style="bold bright_cyan")
        section_header.append("CALIBRATION REPORT", style="bold bright_white")
        console.print(section_header)
        console.print()
        
        # Show success or issues
        if not issues:
            console.print()
            success_text = Text()
            success_text.append("  ✓ ", style="bold bright_green")
            success_text.append("All ", style="white")
            success_text.append(f"{len(cache)}", style="bold bright_cyan")
            success_text.append(" assets passed calibration checks", style="white")
            console.print(success_text)
            console.print()
            
            stats_text = Text()
            stats_text.append("    PIT p-value ≥ 0.05 for all models  ·  ", style="dim")
            stats_text.append("No regime collapse detected", style="dim")
            console.print(stats_text)
            console.print()
            return
        
        # ISSUES HEADER
        issues_header = Text()
        issues_header.append("  ⚠️  ", style="bold yellow")
        issues_header.append(f"{len(issues)} assets with calibration issues", style="bold yellow")
        console.print(issues_header)
        console.print()
        
        # SUMMARY STATS
        critical_count = sum(1 for i in issues if i['severity'] == 'critical')
        warning_count = sum(1 for i in issues if i['severity'] == 'warning')
        failed_count = sum(1 for i in issues if i['issue_type'] == 'FAILED')
        
        summary = Text()
        summary.append("    ", style="")
        if critical_count > 0:
            summary.append(f"{critical_count}", style="bold indian_red1")
            summary.append(" critical", style="dim")
            summary.append("   ·   ", style="dim")
        if warning_count > 0:
            summary.append(f"{warning_count}", style="bold yellow")
            summary.append(" warnings", style="dim")
            summary.append("   ·   ", style="dim")
        if failed_count > 0:
            summary.append(f"{failed_count}", style="bold red")
            summary.append(" failed", style="dim")
            summary.append("   ·   ", style="dim")
        summary.append(f"{len(cache)}", style="white")
        summary.append(" total assets", style="dim")
        
        console.print(summary)
        console.print()
        
        # ISSUES TABLE
        table = Table(
            show_header=True,
            header_style="bold white",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            row_styles=["", "on grey7"],
        )
        
        table.add_column("Asset", justify="left", width=30, no_wrap=True)
        table.add_column("Issue", justify="left", width=18)
        table.add_column("PIT p", justify="right", width=8)
        table.add_column("KS", justify="right", width=6)
        table.add_column("Kurt", justify="right", width=6)
        table.add_column("Model", justify="left", width=12)
        table.add_column("log₁₀(q)", justify="right", width=9)
        table.add_column("φ", justify="right", width=6)
        table.add_column("Details", justify="left", width=25, no_wrap=True)
        
        for issue in issues:
            if issue['severity'] == 'critical':
                severity_style = "bold indian_red1"
                asset_style = "indian_red1"
            elif issue['severity'] == 'warning':
                severity_style = "yellow"
                asset_style = "yellow"
            else:
                severity_style = "dim"
                asset_style = "white"
            
            pit_str = f"{issue['pit_p']:.4f}" if issue['pit_p'] is not None else "-"
            ks_str = f"{issue['ks_stat']:.3f}" if issue['ks_stat'] is not None else "-"
            kurt_str = f"{issue['kurtosis']:.1f}" if issue['kurtosis'] is not None else "-"
            
            if issue['q'] is not None and issue['q'] > 0:
                log_q_str = f"{np.log10(issue['q']):.2f}"
            else:
                log_q_str = "-"
            
            phi_str = f"{issue['phi']:.3f}" if issue['phi'] is not None else "-"
            
            if issue['pit_p'] is not None:
                if issue['pit_p'] < 0.01:
                    pit_styled = f"[bold indian_red1]{pit_str}[/]"
                elif issue['pit_p'] < 0.05:
                    pit_styled = f"[yellow]{pit_str}[/]"
                else:
                    pit_styled = f"[dim]{pit_str}[/]"
            else:
                pit_styled = "[dim]-[/]"
            
            if issue['kurtosis'] is not None:
                if issue['kurtosis'] > 10:
                    kurt_styled = f"[bold indian_red1]{kurt_str}[/]"
                elif issue['kurtosis'] > 6:
                    kurt_styled = f"[yellow]{kurt_str}[/]"
                else:
                    kurt_styled = f"[dim]{kurt_str}[/]"
            else:
                kurt_styled = "[dim]-[/]"
            
            table.add_row(
                f"[{asset_style}]{issue['asset']}[/]",
                f"[{severity_style}]{issue['issue_type']}[/]",
                pit_styled,
                f"[dim]{ks_str}[/]",
                kurt_styled,
                f"[dim]{issue['model']}[/]",
                f"[dim]{log_q_str}[/]",
                f"[dim]{phi_str}[/]",
                f"[dim]{issue['details']}[/]",
            )
        
        console.print(table)
        console.print()
        
        # LEGEND
        legend = Text()
        legend.append("    ", style="")
        legend.append("PIT p < 0.05", style="yellow")
        legend.append(" = model may be miscalibrated   ·   ", style="dim")
        legend.append("Kurt > 6", style="yellow")
        legend.append(" = heavy tails not fully captured", style="dim")
        
        console.print(legend)
        console.print()
        
        # Action recommendation
        if critical_count > 0:
            action = Text()
            action.append("    → ", style="dim")
            action.append("Consider re-tuning critical assets with ", style="dim")
            action.append("make tune ARGS='--force --assets <TICKER>'", style="bold white")
            console.print(action)
            console.print()
    
    except Exception as e:
        # Fallback to simple print output if Rich fails
        print(f"\n[Calibration report error: {e}]")
        print("\n📊 CALIBRATION REPORT (text fallback)")
        print("-" * 60)
        
        issue_count = 0
        for asset, raw_data in cache.items():
            data = raw_data.get('global', raw_data)
            if data.get('calibration_warning') or (data.get('pit_ks_pvalue') or 1.0) < 0.05:
                issue_count += 1
                print(f"  ⚠️  {asset}: PIT p={data.get('pit_ks_pvalue', 'N/A')}")
        
        for asset in failure_reasons:
            print(f"  ❌ {asset}: FAILED")
        
        if issue_count == 0 and not failure_reasons:
            print(f"  ✓ All {len(cache)} assets passed calibration checks")
        
        sys.stdout.flush()


if __name__ == '__main__':
    main()
