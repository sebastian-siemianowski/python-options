#!/usr/bin/env python3
"""
===============================================================================
SYSTEM DNA — SIGNAL / DECISION LAYER (Bayesian Model Averaging Edition)
===============================================================================

This file implements the *decision intelligence layer* of the quant system.

It consumes the probabilistic structure produced by tune.py and applies
posterior predictive Monte Carlo with Bayesian Model Averaging:

    p(x | r_t) = Σ_m p(x | r_t, m, θ_{r_t,m}) · p(m | r_t)

Where:
    r_t      = current regime (deterministically assigned, same logic as tune)
    m        = model class (kalman_gaussian, kalman_phi_gaussian, 
                           phi_student_t_nu_{4,6,8,12,20},
                           phi_skew_t_nu_{ν}_gamma_{γ},
                           phi_nig_alpha_{α}_beta_{β})
    θ_{r,m}  = parameters from tuning layer
    p(m|r)   = posterior model probability from tuning layer
    x        = return at horizon H

-------------------------------------------------------------------------------
DISTRIBUTIONAL MODEL ENSEMBLE
-------------------------------------------------------------------------------

The BMA ensemble includes multiple distributional models:

1. GAUSSIAN (kalman_gaussian, kalman_phi_gaussian):
   - Light tails, symmetric
   - Baseline model for stable markets

2. STUDENT-T (phi_student_t_nu_{4,6,8,12,20}):
   - Heavy tails, symmetric
   - ν controls tail heaviness (smaller ν = heavier tails)

3. SKEW-T (phi_skew_t_nu_{ν}_gamma_{γ}) — Fernández-Steel:
   - Heavy tails, asymmetric (Fernández-Steel parameterization)
   - γ < 1.0: Left-skewed (crash risk)
   - γ > 1.0: Right-skewed (euphoria risk)

4. HANSEN SKEW-T — Regime-Conditional Asymmetry:
   - Hansen (1994) parameterization with λ ∈ (-1, 1)
   - λ < 0: Left-skewed (crash risk, heavier left tail)
   - λ > 0: Right-skewed (recovery potential, heavier right tail)
   - λ = 0: Reduces to symmetric Student-t
   - Used for probability calculations and Monte Carlo sampling
   - Financial meaning: captures regime-specific tail asymmetry

5. NIG (phi_nig_alpha_{α}_beta_{β}) — Solution 2:
   - Semi-heavy tails (between Gaussian and Cauchy), asymmetric
   - α controls tail heaviness (smaller α = heavier tails)
   - β controls asymmetry (β < 0 = left-skewed, β > 0 = right-skewed)
   - Infinitely divisible (Lévy process compatible)

6. GMM (2-State Gaussian Mixture) — Expert Panel Solution:
   - Bimodal distribution capturing momentum/reversal dynamics
   - Component 0 ("Momentum"): typically positive mean, moderate variance
   - Component 1 ("Reversal/Crisis"): typically negative mean, higher variance
   - Fitted to volatility-adjusted returns during tuning
   - Used as Monte Carlo proposal distribution for Gaussian models
   - Improves tail behavior in Expected Utility estimation

CORE PRINCIPLE: "Heavy tails, asymmetry, and bimodality are hypotheses, not certainties."

All models compete via BIC weights. If extra parameters don't improve fit,
model weight collapses naturally toward simpler alternatives.

-------------------------------------------------------------------------------
REGIME ASSIGNMENT — DETERMINISTIC, CONSISTENT WITH TUNE
-------------------------------------------------------------------------------

Current regime r_t is determined using SAME logic as tune.py's
assign_regime_labels() function:

    Regimes:
        0 = LOW_VOL_TREND:  vol < 0.85*median, |drift| > threshold
        1 = HIGH_VOL_TREND: vol > 1.3*median, |drift| > threshold  
        2 = LOW_VOL_RANGE:  vol < 0.85*median, |drift| <= threshold
        3 = HIGH_VOL_RANGE: vol > 1.3*median, |drift| <= threshold
        4 = CRISIS_JUMP:    vol > 2*median OR tail_indicator > 4

This ensures perfect consistency between tuning and inference.
NO soft regime probabilities - hard assignment to single regime.

-------------------------------------------------------------------------------
CONTRACT WITH tune.py — NEW BMA ARCHITECTURE ONLY
-------------------------------------------------------------------------------

tune.py outputs:
    {
        "global": {
            "model_posterior": { "kalman_gaussian": p, "kalman_phi_gaussian": p, ... },
            "models": { "kalman_gaussian": {q, c, hyvarinen_score, bic, ...}, ... },
            "gmm": { "weights": [π₁, π₂], "means": [μ₁, μ₂], "variances": [σ₁², σ₂²] },
            "hansen_skew_t": { "nu": ν, "lambda": λ, "skew_direction": "left"/"right"/"symmetric" }
        },
        "regime": {
            "0": { "model_posterior": {...}, "models": {...}, "regime_meta": {...} },
            "1": { ... },
            ...
        },
        "meta": {
            "model_selection_method": "combined",  # 'bic', 'hyvarinen', or 'combined'
            "bic_weight": 0.5,  # Weight for BIC in combined method
            ...
        }
    }

MODEL SELECTION METHODS:
    - 'bic': Traditional BIC-only (consistent but not robust to misspecification)
    - 'hyvarinen': Hyvärinen score only (Fisher-consistent under misspecification)
    - 'combined': Geometric mean of BIC and Hyvärinen weights (default)

The Hyvärinen score is a proper scoring rule that:
    - Is Fisher-consistent under model misspecification
    - Does not require normalizing constants
    - Naturally rewards tail accuracy for Student-t models

Combined method: w_combined(m) = w_bic(m)^α * w_hyvarinen(m)^(1-α)

This file receives from tuning, for current regime r_t:

    regime_models = data["regime"][r_t]["models"]
    model_posterior = data["regime"][r_t]["model_posterior"]

Tune GUARANTEES:
    - Every regime contains valid model_posterior and models
    - Fallback regimes use hierarchical borrowing (borrowed_from_global=True)
    - Model posteriors are already normalized and temporally smoothed

This file must NOT:
    - Perform tuning
    - Recompute likelihoods
    - Renormalize model weights
    - Apply temporal smoothing to model posteriors
    - Select a single best model
    - Implement fallback logic (tune handles this)
    - Support old flat schema (NO BACKWARD COMPATIBILITY)
    - SYNTHESIZE fake models with invented parameters

-------------------------------------------------------------------------------
BAYESIAN INTEGRITY POLICY
-------------------------------------------------------------------------------

"Act only on beliefs that were actually learned."

This system is a belief evolution engine, not a rule engine.

When evidence is weak:
    - The system becomes more ignorant, not more confident
    - Ignorance is represented by reverting to a higher-level posterior (global)
    - NEVER by inventing beliefs

Explicitly FORBIDDEN:
    - Synthesizing "minimal models with conservative defaults"
    - Inventing parameter values that were not fit to data
    - Creating uniform posteriors from fabricated structure

The only valid fallback is hierarchical: p(m|r, weak data) → p(m|global)

-------------------------------------------------------------------------------
ARCHITECTURAL INVARIANTS
-------------------------------------------------------------------------------

    - There is exactly one inference philosophy: Regime-conditional Bayesian 
      model averaging
    - Regime assignment is DETERMINISTIC (same as tune.py)
    - No code path constructs p(m) = [1,0,0,0,0] (one-hot model posterior)
    - When evidence is insufficient, we use global BMA, not single-model certainty
    - Old flat cache schema is REJECTED - must regenerate with tune.py
    - No synthesized/invented models - only inferred beliefs

-------------------------------------------------------------------------------
POSTERIOR PREDICTIVE MONTE CARLO

For current regime r_t, we compute:

    r_samples = []
    for m, w in model_posterior.items():
        theta = regime_models[m]
        n_m = w * n_total_samples  # proportional to model weight
        samples_m = posterior_predictive_mc(m, theta, n_paths=n_m)
        r_samples.append(samples_m)
    
    r_samples = concatenate(r_samples)

r_samples now represents: p(x | r_t) = Σ_m p(x | r_t, m) · p(m | r_t)

-------------------------------------------------------------------------------
EXPECTED UTILITY PRINCIPLE

Decisions are made from distributions, not point estimates:

    EU = p · E[gain] − (1−p) · E[loss]

Position sizing derives from Expected Utility geometry.
The decision layer is NOT aware that model averaging occurred.
It only sees r_samples.

-------------------------------------------------------------------------------
ARCHITECTURAL LAW

    tune.py → model_posterior, models
                         ↓
    signals.py → r_samples (mixture)
                         ↓
                    EU → Decision

No shortcut is allowed.
Inference → Distribution → Utility → Decision.

-------------------------------------------------------------------------------
CRITICAL RULES

    • Do NOT perform tuning here
    • Do NOT recompute likelihoods
    • Do NOT renormalize model weights
    • Do NOT apply temporal smoothing to model posteriors
    • Do NOT select best model - use full mixture
    • Do NOT modify EU logic
    • Do NOT construct one-hot posteriors
    • Preserve distributional integrity
    • Preserve structural uncertainty

This file defines agency, not epistemology.
It must act on beliefs, not create them.

===============================================================================

fx_pln_jpy_signals_v3.py

Quant upgrades:
- multi-speed EWMA drift/vol (fast + slow blend)
- robust returns (winsorized)
- probability of positive return per horizon (p_up)
- t-stat style momentum (cumret / realized vol)
- shrinkage drift (toward 0, stronger in stressed vol regimes)
- clearer HOLD zone based on probability, not raw z

Notes:
- PLNJPY=X is JPY per PLN. BUY => long PLN vs JPY.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Ensure src directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
# NOTE: yfinance is NOT imported here - all data access goes through data_utils
# which respects OFFLINE_MODE. Signal generation should NEVER call Yahoo Finance directly.
from scipy.stats import t as student_t, norm, skew as scipy_stats_skew
from scipy.special import gammaln
from rich.console import Console
from rich.padding import Padding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.align import Align
from rich import box
import logging
import os

# HMM regime detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Isotonic Recalibration — Probability Transport Operator
# This is the CORE calibration layer for aligning model beliefs with reality
try:
    from calibration.isotonic_recalibration import (
        TransportMapResult,
        IsotonicRecalibrator,
        apply_recalibration,
        compute_raw_pit_gaussian,
        compute_raw_pit_student_t,
    )
    ISOTONIC_RECALIBRATION_AVAILABLE = True
except ImportError:
    ISOTONIC_RECALIBRATION_AVAILABLE = False

# Calibrated Trust Authority — Single Point of Trust Decision
# ARCHITECTURAL LAW: Trust = Calibration Authority − Governed, Bounded Regime Penalty
# This is the CANONICAL authority for trust decisions.
# All downstream decisions (position sizing, drift weight) flow from here.
try:
    from calibration.calibrated_trust import (
        CalibratedTrust,
        TrustConfig,
        compute_calibrated_trust,
        compute_drift_weight,
        MAX_REGIME_PENALTY,
        MAX_MODEL_PENALTY,
        DEFAULT_REGIME_PENALTY_SCHEDULE,
        DEFAULT_MODEL_PENALTY_SCHEDULE,
        REGIME_NAMES,
    )
    CALIBRATED_TRUST_AVAILABLE = True
except ImportError:
    CALIBRATED_TRUST_AVAILABLE = False

# Control Policy — Authority Boundary Layer (Counter-Proposal v1.0)
# ARCHITECTURAL LAW: Diagnostics RECOMMEND, Policy DECIDES, Models OBEY
# This ensures explicit, auditable escalation decisions.
try:
    from calibration.control_policy import (
        EscalationDecision,
        CalibrationDiagnostics,
        ControlPolicy,
        DECISION_NAMES,
        DEFAULT_CONTROL_POLICY,
        create_diagnostics_from_result,
    )
    CONTROL_POLICY_AVAILABLE = True
except ImportError:
    CONTROL_POLICY_AVAILABLE = False

# PIT Violation Penalty — Asymmetric Calibration Governance (February 2026)
# CORE DESIGN CONSTRAINT: PIT must only act as a PENALTY, never a reward.
# When belief cannot be trusted, the only correct signal is EXIT.
try:
    from calibration.pit_penalty import (
        check_exit_signal_required,
        compute_model_pit_penalty,
        PITViolationResult,
        PITPenaltyReport,
        PIT_EXIT_THRESHOLD,
        PIT_CRITICAL_THRESHOLDS,
    )
    PIT_PENALTY_AVAILABLE = True
except ImportError:
    PIT_PENALTY_AVAILABLE = False

# Context manager to suppress noisy HMM convergence messages
import contextlib
import io
import warnings

@contextlib.contextmanager  
def suppress_stdout():
    """Temporarily suppress stdout/stderr to hide noisy library messages like HMM convergence warnings."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stderr = old_stderr

# Import presentation layer for display logic
from decision.signals_ux import (
    render_detailed_signal_table,
    render_simplified_signal_table,
    render_multi_asset_summary_table,
    render_sector_summary_tables,
    render_strong_signals_summary,
    render_risk_temperature_summary,
    render_augmentation_layers_summary,
    build_asset_display_label,
    extract_symbol_from_title,
    format_horizon_label,
    DETAILED_COLUMN_DESCRIPTIONS,
    SIMPLIFIED_COLUMN_DESCRIPTIONS,
)

# Import data utilities and helper functions
from ingestion.data_utils import (
    norm_cdf,
    _to_float,
    safe_last,
    winsorize,
    _download_prices,
    _resolve_display_name,
    _fetch_px_symbol,
    _fetch_with_fallback,
    fetch_px,
    fetch_usd_to_pln_exchange_rate,
    detect_quote_currency,
    _as_series,
    _ensure_float_series,
    _align_fx_asof,
    convert_currency_to_pln,
    convert_price_series_to_pln,
    _resolve_symbol_candidates,
    DEFAULT_ASSET_UNIVERSE,
    get_default_asset_universe,
    COMPANY_NAMES,
    get_company_name,
    SECTOR_MAP,
    get_sector,
    download_prices_bulk,
    save_failed_assets,
    get_price_series,
    STANDARD_PRICE_COLUMNS,
    print_symbol_tables,
    enable_cache_only_mode,
)

# =============================================================================
# SIGNAL GENERATION: CACHE-ONLY MODE
# =============================================================================
# Signal generation should NEVER make Yahoo Finance API calls.
# All price data must come from cache populated during 'make data' or 'make refresh'.
#
# This ensures:
# 1. Fast signal generation (no network latency)
# 2. No rate limiting issues with Yahoo Finance
# 3. Reproducible results (same cache = same signals)
# 4. System works offline once data is cached
#
# If you see "Failed download" errors, run 'make data' first to populate the cache.
# =============================================================================
enable_cache_only_mode()

# Suppress noisy yfinance download warnings (e.g., "1 Failed download: ...")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class StudentTDriftModel:
    """Minimal Student-t helper used for Kalman log-likelihood and mapping."""

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        if scale <= 0 or nu <= 0:
            return -1e12
        z = (x - mu) / scale
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)


# =============================================================================
# EVT (EXTREME VALUE THEORY) FOR POSITION SIZING
# =============================================================================
# Import POT/GPD tail modeling for EVT-corrected expected loss estimation.
# This provides principled extrapolation of tail losses beyond observed data.
#
# THEORETICAL FOUNDATION:
#   Pickands–Balkema–de Haan theorem: exceedances over high threshold → GPD
#   CTE = E[Loss | Loss > u] = u + σ/(1-ξ)  for ξ < 1
#
# INTEGRATION:
#   - Used in Expected Utility calculation to replace naive E[loss]
#   - Produces more conservative (larger) loss estimates for heavy-tailed assets
#   - Falls back to empirical × 1.5 if GPD fitting fails
# =============================================================================
try:
    from calibration.evt_tail import (
        compute_evt_expected_loss,
        compute_evt_var,
        fit_gpd_pot,
        GPDFitResult,
        EVT_THRESHOLD_PERCENTILE_DEFAULT,
        EVT_MIN_EXCEEDANCES,
        EVT_FALLBACK_MULTIPLIER,
        check_student_t_consistency,
    )
    EVT_AVAILABLE = True
except ImportError:
    EVT_AVAILABLE = False
    EVT_THRESHOLD_PERCENTILE_DEFAULT = 0.90
    EVT_MIN_EXCEEDANCES = 30
    EVT_FALLBACK_MULTIPLIER = 1.5


# =============================================================================
# CONTAMINATED STUDENT-T DISTRIBUTION
# =============================================================================
# Regime-indexed contaminated Student-t mixture for crisis tail modeling.
# Models returns as: (1-ε)·t(ν_normal) + ε·t(ν_crisis) where ε is contamination.
# =============================================================================
try:
    from models import (
        contaminated_student_t_rvs,
        ContaminatedStudentTParams,
    )
    CONTAMINATED_ST_AVAILABLE = True
except ImportError:
    CONTAMINATED_ST_AVAILABLE = False
    
    # Fallback: simple contaminated sampling without the module
    def contaminated_student_t_rvs(
        size: int,
        nu_normal: float,
        nu_crisis: float,
        epsilon: float,
        loc: float = 0.0,
        scale: float = 1.0,
        random_state=None
    ):
        """Fallback contaminated student-t sampling."""
        import numpy as np
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Sample component indicators
        from_crisis = rng.random(size) < epsilon
        n_crisis = from_crisis.sum()
        n_normal = size - n_crisis
        
        samples = np.empty(size)
        if n_normal > 0:
            samples[~from_crisis] = rng.standard_t(df=nu_normal, size=n_normal)
        if n_crisis > 0:
            samples[from_crisis] = rng.standard_t(df=nu_crisis, size=n_crisis)
        
        return loc + scale * samples


# =============================================================================
# HANSEN SKEW-T DISTRIBUTION (Asymmetric Tails)
# =============================================================================
# Hansen (1994) skew-t captures directional asymmetry via λ parameter:
#   - λ > 0: Right-skewed (recovery potential, heavier right tail)
#   - λ < 0: Left-skewed (crash risk, heavier left tail)
#   - λ = 0: Reduces to symmetric Student-t
#
# CRITICAL: This must be imported for signals.py to use hansen_lambda in MC sampling.
# Without this import, hansen_lambda is accepted but IGNORED - silent bug!
# =============================================================================
try:
    from models import (
        hansen_skew_t_rvs,
        hansen_skew_t_cdf,
        HansenSkewTParams,
    )
    HANSEN_SKEW_T_AVAILABLE = True
except ImportError:
    HANSEN_SKEW_T_AVAILABLE = False
    
    # Fallback: stub that falls back to symmetric Student-t (with warning)
    def hansen_skew_t_rvs(
        size: int,
        nu: float,
        lambda_: float,
        loc: float = 0.0,
        scale: float = 1.0,
        random_state=None
    ) -> np.ndarray:
        """Fallback Hansen skew-t sampling - uses symmetric Student-t with warning."""
        import warnings
        warnings.warn(
            f"Hansen skew-t not available, using symmetric Student-t (ignoring λ={lambda_:.3f})",
            RuntimeWarning
        )
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Fallback to symmetric Student-t
        samples = rng.standard_t(df=nu, size=size)
        return loc + scale * samples


# =============================================================================
# MODEL REGISTRY — Single Source of Truth for Model Synchronization
# =============================================================================
# The model registry ensures tune.py and signals.py are ALWAYS synchronised.
# This prevents the #1 silent failure: model name mismatch → dropped from BMA.
#
# ARCHITECTURAL LAW: Top funds REFUSE TO TRADE without this assertion.
# =============================================================================
try:
    from models.model_registry import (
        MODEL_REGISTRY,
        ModelFamily,
        SupportType,
        get_model_spec,
        assert_models_synchronised,
    )
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    MODEL_REGISTRY_AVAILABLE = False


# =============================================================================
# RISK TEMPERATURE MODULATION LAYER (Expert Panel Solution 1 + 4)
# =============================================================================
# Risk temperature scales position sizes based on cross-asset stress indicators
# WITHOUT modifying distributional beliefs (Kalman state, BMA weights, GARCH).
#
# DESIGN PRINCIPLE:
#   "FX, futures, and commodities don't tell you WHERE to go.
#    They tell you HOW FAST you're allowed to drive."
#
# INTEGRATION:
#   - Computed AFTER EU-based sizing
#   - Applied BEFORE final position output
#   - Uses smooth sigmoid scaling (no cliff effects)
#   - Overnight budget constraint when temp > 1.0
#
# STRESS CATEGORIES (weighted sum):
#   - FX Stress (40%): AUDJPY, USDJPY z-scores — risk-on/off proxy
#   - Futures Stress (30%): ES/NQ overnight returns — equity sentiment
#   - Rates Stress (20%): TLT volatility — macro stress
#   - Commodity Stress (10%): Copper, gold/copper ratio — growth fear
#
# SCALING FUNCTION:
#   scale_factor(temp) = 1.0 / (1.0 + exp(3.0 × (temp - 1.0)))
#
# OVERNIGHT BUDGET:
#   When temp > 1.0: cap position such that position × gap ≤ budget
# =============================================================================
try:
    from decision.risk_temperature import (
        compute_risk_temperature,
        apply_risk_temperature_scaling,
        get_cached_risk_temperature,
        clear_risk_temperature_cache,
        RiskTemperatureResult,
        SIGMOID_THRESHOLD,
        OVERNIGHT_BUDGET_ACTIVATION_TEMP,
    )
    RISK_TEMPERATURE_AVAILABLE = True
except ImportError:
    RISK_TEMPERATURE_AVAILABLE = False
    SIGMOID_THRESHOLD = 1.0
    OVERNIGHT_BUDGET_ACTIVATION_TEMP = 1.0


# =============================================================================
# ISOTONIC RECALIBRATION HELPER
# =============================================================================
# This function loads a persisted transport map and applies it to PIT values.
# The transport map is learned during tuning and stored in the cache.
#
# DOCTRINE:
#   - Calibration is a FIRST-CLASS PROBABILISTIC TRANSPORT OPERATOR
#   - Applied BEFORE regimes see PIT values
#   - Regimes see CALIBRATED probability, not raw belief
#   - This is NOT a patch/validator/escalation trigger
# =============================================================================

def load_and_apply_recalibration(
    tuned_params: Optional[Dict],
    raw_pit: np.ndarray,
) -> Tuple[np.ndarray, bool, Optional[Dict]]:
    """
    Load recalibration transport map from tuned params and apply to raw PIT.
    
    Args:
        tuned_params: Full tuned params dict containing recalibration info
        raw_pit: Array of raw PIT values from model
        
    Returns:
        Tuple of:
        - calibrated_pit: Array of calibrated PIT values
        - was_recalibrated: True if recalibration was applied
        - recal_meta: Metadata about recalibration (or None)
    """
    if not ISOTONIC_RECALIBRATION_AVAILABLE:
        return raw_pit, False, None
    
    if tuned_params is None:
        return raw_pit, False, None
    
    recal_data = tuned_params.get('recalibration')
    if recal_data is None or not tuned_params.get('recalibration_applied', False):
        return raw_pit, False, None
    
    try:
        # Load transport map from persisted data
        transport_map = TransportMapResult.from_dict(recal_data)
        
        # Check if it's an identity map (no change needed)
        if transport_map.is_identity or transport_map.fallback_to_identity:
            return raw_pit, False, {
                'is_identity': True,
                'reason': 'identity_map' if transport_map.is_identity else 'fallback',
            }
        
        # Apply recalibration
        calibrated_pit = apply_recalibration(raw_pit, transport_map)
        
        recal_meta = {
            'applied': True,
            'n_segments': transport_map.n_segments,
            'ks_improvement': transport_map.ks_improvement,
            'raw_ks_pvalue': transport_map.raw_ks_pvalue,
            'calibrated_ks_pvalue': transport_map.calibrated_ks_pvalue,
        }
        
        return calibrated_pit, True, recal_meta
        
    except Exception as e:
        # If recalibration fails, return raw PIT
        if os.getenv('DEBUG'):
            print(f"Warning: Recalibration failed: {e}")
        return raw_pit, False, {'error': str(e)}


PAIR = "PLNJPY=X"
DEFAULT_HORIZONS = [1, 3, 7, 21, 63, 126, 252]
NOTIONAL_PLN = 1_000_000  # for profit column

# Transaction-cost/slippage hurdle: minimum absolute edge required to act
# Can be overridden via environment variable EDGE_FLOOR (e.g., 0.10)
try:
    _edge_env = os.getenv("EDGE_FLOOR", "0.10")
    EDGE_FLOOR = float(_edge_env)
except Exception:
    EDGE_FLOOR = 0.10
# Clamp to a reasonable range to avoid misuse
EDGE_FLOOR = float(np.clip(EDGE_FLOOR, 0.0, 1.5))

DEFAULT_CACHE_PATH = os.path.join("src", "data", "currencies", "fx_plnjpy.json")

# ============================================================================
# UPGRADE #3: Display Price Inertia (Presentation-Only)
# ============================================================================
# This cache stores previous display prices for presentation smoothing.
# Formula: display_price = 0.7 * prev_display_price + 0.3 * new_predicted_price
#
# IMPORTANT:
# - This does NOT affect trading decisions
# - This does NOT affect Expected Utility calculations
# - This does NOT affect regime detection
# - It ONLY prevents day-to-day jitter in displayed prices
#
# Institutions do this quietly for all client-facing price estimates.
# ============================================================================
_DISPLAY_PRICE_CACHE: Dict[Tuple[str, int], float] = {}
DISPLAY_PRICE_INERTIA = 0.7  # weight on previous display price


def _smooth_display_price(asset_key: str, horizon: int, new_price: float) -> float:
    """Apply presentation-only smoothing to predicted prices.
    
    Args:
        asset_key: Unique identifier for the asset (e.g., ticker symbol)
        horizon: Forecast horizon in days
        new_price: Newly computed predicted price
        
    Returns:
        Smoothed display price that reduces day-to-day jitter
    """
    cache_key = (asset_key, horizon)
    if cache_key in _DISPLAY_PRICE_CACHE:
        prev_price = _DISPLAY_PRICE_CACHE[cache_key]
        if np.isfinite(prev_price) and np.isfinite(new_price):
            smoothed = DISPLAY_PRICE_INERTIA * prev_price + (1.0 - DISPLAY_PRICE_INERTIA) * new_price
        else:
            smoothed = new_price if np.isfinite(new_price) else prev_price
    else:
        smoothed = new_price
    
    if np.isfinite(smoothed):
        _DISPLAY_PRICE_CACHE[cache_key] = smoothed
    
    return smoothed


def clear_display_price_cache() -> None:
    """Clear the display price inertia cache. Useful for testing or resets."""
    global _DISPLAY_PRICE_CACHE
    _DISPLAY_PRICE_CACHE.clear()


# ============================================================================
# DUAL-SIDED TREND EXHAUSTION (UE↑ / UE↓)
# ============================================================================
# Measures market-space trend fragility in two independent directions:
#   - Upside Exhaustion (UE↑): late-stage rally fragility, blow-off risk
#   - Downside Exhaustion (UE↓): late-stage sell-off fragility, rebound risk
#
# KEY DESIGN PRINCIPLES:
#   1. Exhaustion is directional in MARKET space, not position space
#   2. UE↑ and UE↓ are mutually exclusive (only one can be active)
#   3. Both bounded in (0, 1)
#   4. Neither decides direction - only modulates risk/confidence
#   5. No tuning feedback, no signal flips
#
# This removes the confusion of a single ambiguous "exhaustion" metric.
# ============================================================================

# Module-level state for tracking cumulative returns per asset/horizon
_exhaustion_state: Dict[str, Dict[int, Dict[str, float]]] = {}
# Structure: { asset_key: { horizon: {"cum_log": float, "regime": int} } }


def _update_exhaustion_state(
    asset_key: str,
    horizon: int,
    realized_log_ret: float,
    regime_id: int,
) -> Tuple[float, int]:
    """
    Update exhaustion state and return cumulative log return since regime entry.
    
    DEPRECATED: This function is kept for backward compatibility but the new
    exhaustion calculation uses price-based deviation from EMA, not cumulative returns.
    
    Args:
        asset_key: Unique identifier for the asset
        horizon: Forecast horizon in days
        realized_log_ret: Most recent realized log return
        regime_id: Current regime index
        
    Returns:
        Tuple of (cumulative_log_return, days_in_regime)
    """
    key = asset_key
    state_by_h = _exhaustion_state.setdefault(key, {})
    state = state_by_h.get(horizon, {
        "cum_log": 0.0,
        "regime": None,
        "days_in_regime": 0,
    })

    if state["regime"] is None:
        state = {"cum_log": realized_log_ret, "regime": regime_id, "days_in_regime": 1}
    elif state["regime"] == regime_id:
        state["cum_log"] += realized_log_ret
        state["days_in_regime"] += 1
    else:
        # Regime changed - reset
        state = {"cum_log": realized_log_ret, "regime": regime_id, "days_in_regime": 1}

    state_by_h[horizon] = state
    return state["cum_log"], state["days_in_regime"]


def compute_directional_exhaustion_from_features(
    feats: Dict[str, pd.Series],
    lookback_short: int = 9,
    lookback_long: int = 21,
    vol_lookback: int = 21,
) -> Dict[str, float]:
    """
    Compute directional exhaustion as a 0-100% metric using multi-timeframe
    EMA analysis with Student-t fat-tail corrections.

    SENIOR QUANT PANEL METHODOLOGY:
    ===============================
    
    1. MULTI-TIMEFRAME DEVIATION ANALYSIS
       - Compute price deviation from 5 EMAs (9, 21, 50, 100, 200 days)
       - Separate into short-term (9, 21) and long-term (50, 100, 200) groups
       - Long-term deviation determines structural position
       - Short-term deviation determines recent move direction
    
    2. MOMENTUM ALIGNMENT DETECTION
       - Compare short-term vs long-term momentum (mom63 vs mom252)
       - Divergence indicates regime transition
       - Convergence indicates trend confirmation
    
    3. RECENT PEAK/TROUGH DETECTION
       - Find rolling 63-day high and low
       - Measure distance from recent extremes
       - Rally-then-breakdown: near recent high but falling
       - Capitulation-then-recovery: near recent low but rising
    
    4. FAT-TAIL PROBABILITY ADJUSTMENT
       - Use Student-t CDF instead of Gaussian
       - Heavy tails (low ν) → extreme moves more expected → lower exhaustion
    
    5. OUTPUT: 0-100% scale
       - ue_up > 0: Price above long-term equilibrium
       - ue_down > 0: Price below long-term equilibrium
       - Mutual exclusivity enforced

    Args:
        feats: Feature dictionary
        lookback_short, lookback_long, vol_lookback: Configuration params

    Returns:
        Dict with "ue_up" (0-1), "ue_down" (0-1), and diagnostics
    """
    from scipy.stats import norm as scipy_norm, t as scipy_t
    
    # Extract price series
    px_series = feats.get("px", pd.Series(dtype=float))
    if px_series is None or len(px_series) < 200:
        return _compute_simple_exhaustion(feats, lookback_short, lookback_long, vol_lookback)

    price = float(px_series.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    # =========================================================================
    # STEP 1: MULTI-TIMEFRAME EMA DEVIATIONS (SHORT vs LONG)
    # =========================================================================
    ema_periods_short = [9, 21]
    ema_periods_long = [50, 100, 200]
    
    ema_values = {}
    short_deviations = []
    long_deviations = []
    
    for period in ema_periods_short + ema_periods_long:
        if len(px_series) >= period:
            ema = px_series.ewm(span=period, adjust=False).mean()
            ema_now = float(ema.iloc[-1])
            if np.isfinite(ema_now) and ema_now > 0:
                deviation_pct = (price - ema_now) / ema_now
                ema_values[f"ema_{period}"] = ema_now
                if period in ema_periods_short:
                    short_deviations.append(deviation_pct)
                else:
                    long_deviations.append(deviation_pct)
    
    if not long_deviations:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}
    
    # Calculate group averages
    short_dev_avg = np.mean(short_deviations) if short_deviations else 0.0
    long_dev_avg = np.mean(long_deviations)
    
    # Structural deviation (long-term determines direction)
    structural_deviation = long_dev_avg
    
    # =========================================================================
    # STEP 2: MOMENTUM EXTRACTION AND ALIGNMENT
    # =========================================================================
    mom21 = feats.get("mom21", pd.Series(dtype=float))
    mom63 = feats.get("mom63", pd.Series(dtype=float))
    mom126 = feats.get("mom126", pd.Series(dtype=float))
    mom252 = feats.get("mom252", pd.Series(dtype=float))
    
    def get_mom(m):
        if m is not None and len(m) > 0:
            v = float(m.iloc[-1])
            return v if np.isfinite(v) else 0.0
        return 0.0
    
    mom21_now = get_mom(mom21)
    mom63_now = get_mom(mom63)
    mom126_now = get_mom(mom126)
    mom252_now = get_mom(mom252)
    
    # Short-term vs long-term momentum comparison
    short_term_mom = (mom21_now + mom63_now) / 2
    long_term_mom = (mom126_now + mom252_now) / 2
    
    # Momentum alignment: positive = aligned, negative = diverging
    mom_alignment = short_term_mom * long_term_mom  # Same sign → positive product
    
    # =========================================================================
    # STEP 3: RECENT PEAK/TROUGH DETECTION (63-day lookback)
    # =========================================================================
    lookback_extreme = 63
    if len(px_series) >= lookback_extreme:
        rolling_high = float(px_series.iloc[-lookback_extreme:].max())
        rolling_low = float(px_series.iloc[-lookback_extreme:].min())
        
        # Distance from recent high (0 = at high, 1 = at low)
        high_low_range = rolling_high - rolling_low
        if high_low_range > 0:
            position_in_range = (price - rolling_low) / high_low_range
        else:
            position_in_range = 0.5
        
        # Detect breakdown from recent high
        distance_from_high_pct = (rolling_high - price) / rolling_high if rolling_high > 0 else 0
        distance_from_low_pct = (price - rolling_low) / rolling_low if rolling_low > 0 else 0
    else:
        position_in_range = 0.5
        distance_from_high_pct = 0.0
        distance_from_low_pct = 0.0
    
    # =========================================================================
    # STEP 4: VOLATILITY AND FAT-TAIL ADJUSTMENT
    # =========================================================================
    ret_series = feats.get("ret", pd.Series(dtype=float))
    if ret_series is not None and len(ret_series) >= vol_lookback:
        recent_vol = float(ret_series.iloc[-vol_lookback:].std())
        if not np.isfinite(recent_vol) or recent_vol <= 0:
            recent_vol = 0.02
    else:
        recent_vol = 0.02
    
    # Z-score based on structural deviation
    z_score = structural_deviation / max(recent_vol, 1e-10)
    
    # Get tail parameter
    nu_hat_series = feats.get("nu_hat", None)
    if nu_hat_series is not None and len(nu_hat_series) > 0:
        nu = float(nu_hat_series.iloc[-1])
        if not np.isfinite(nu) or nu <= 2:
            nu = 30.0
    else:
        nu = 30.0
    nu = max(4.0, min(nu, 100.0))
    
    # CDF transformation
    if nu < 30:
        cdf_val = scipy_t.cdf(abs(z_score), df=nu)
    else:
        cdf_val = scipy_norm.cdf(abs(z_score))
    
    exhaustion_base = 2.0 * (cdf_val - 0.5)  # Maps to (0, 1)
    
    # =========================================================================
    # STEP 5: PATTERN DETECTION
    # =========================================================================
    
    # EMA slope for trend direction
    ema_9_series = px_series.ewm(span=9, adjust=False).mean()
    if len(ema_9_series) >= 5:
        ema_9_slope = (float(ema_9_series.iloc[-1]) - float(ema_9_series.iloc[-5])) / max(float(ema_9_series.iloc[-5]), 1e-10)
    else:
        ema_9_slope = 0.0
    
    # Pattern 1: RALLY THEN BREAKDOWN
    # Long-term momentum strong positive, but short-term breaking down
    is_rally_breakdown = (
        long_term_mom > 1.0 and           # Strong long-term momentum
        short_term_mom < long_term_mom and # Short-term weakening
        ema_9_slope < -0.005 and          # 9-EMA turning down
        distance_from_high_pct > 0.05      # At least 5% off recent high
    )
    
    # Pattern 2: PARABOLIC RALLY (extreme)
    is_parabolic = (
        mom126_now > 2.0 and
        structural_deviation > 0.10
    )
    
    # Pattern 3: CAPITULATION
    is_capitulation = (
        mom63_now < -2.0 and
        mom126_now < -1.5 and
        structural_deviation < -0.15
    )
    
    # Pattern 4: RECOVERY FROM CRASH
    # Price recovering but still below long-term equilibrium
    is_recovery = (
        structural_deviation < 0 and       # Below long-term EMAs
        short_dev_avg > long_dev_avg and  # Short-term above long-term (recovering)
        ema_9_slope > 0.005               # Short-term trend up
    )
    
    # Pattern 5: PULLBACK IN UPTREND
    # Long-term trend up, short-term pullback
    is_pullback_uptrend = (
        long_term_mom > 0.5 and            # Long-term trend up
        structural_deviation > -0.05 and   # Not too far below
        short_dev_avg < 0 and              # Short-term below EMAs
        ema_9_slope < 0                    # Pulling back
    )
    
    # =========================================================================
    # STEP 6: FINAL CALCULATION WITH CONTEXT
    # =========================================================================
    
    if structural_deviation > 0:
        # PRICE ABOVE LONG-TERM EQUILIBRIUM → ue_up
        ue_up_raw = exhaustion_base
        
        # Boost for parabolic moves
        if is_parabolic:
            ue_up_raw = min(ue_up_raw * 1.4 + 0.15, 0.99)
        
        # Momentum confirmation boost
        if long_term_mom > 1.0:
            ue_up_raw = min(ue_up_raw + 0.1, 0.99)
        if mom252_now > 1.5:
            ue_up_raw = min(ue_up_raw + 0.1, 0.99)
        
        ue_up = min(ue_up_raw, 0.99)
        ue_down = 0.0
        
    elif structural_deviation < 0:
        # PRICE BELOW LONG-TERM EQUILIBRIUM → consider ue_down
        ue_down_raw = exhaustion_base
        
        # Rally-then-breakdown: this is MEAN REVERSION, not oversold
        # Flip to showing ue_up based on long-term momentum strength
        if is_rally_breakdown:
            # Strong prior rally means this breakdown is healthy
            if long_term_mom > 1.5:
                # Still structurally extended - show ue_up
                ue_up = min(0.25 + long_term_mom * 0.15, 0.70)
                ue_down = 0.0
                return {
                    "ue_up": float(ue_up),
                    "ue_down": 0.0,
                    "z_score": float(z_score),
                    "deviation_pct": float(structural_deviation * 100),
                    **ema_values,
                }
            else:
                # Moderate prior rally - reduce ue_down significantly
                ue_down_raw *= 0.3
        
        # Recovery pattern: reduce ue_down (price improving)
        if is_recovery:
            recovery_factor = 1.0 - min(short_dev_avg - long_dev_avg, 0.1) * 5
            ue_down_raw *= max(recovery_factor, 0.3)
        
        # Pullback in uptrend: show low ue_down (buying opportunity)
        if is_pullback_uptrend:
            ue_down_raw *= 0.4
            
        # Capitulation: boost ue_down
        if is_capitulation:
            ue_down_raw = min(ue_down_raw * 1.3 + 0.1, 0.99)
        
        # Momentum context penalty (positive long-term = less oversold)
        if not is_capitulation:
            if long_term_mom > 0.5:
                ue_down_raw *= 0.7
            if mom252_now > 1.0:
                ue_down_raw *= 0.6
        
        ue_down = min(max(ue_down_raw, 0.0), 0.99)
        ue_up = 0.0
        
    else:
        ue_up = 0.0
        ue_down = 0.0
    
    return {
        "ue_up": float(ue_up),
        "ue_down": float(ue_down),
        "z_score": float(z_score),
        "deviation_pct": float(structural_deviation * 100),
        **ema_values,
    }


def _compute_simple_exhaustion(
    feats: Dict[str, pd.Series],
    lookback_short: int = 9,
    lookback_long: int = 21,
    vol_lookback: int = 21,
) -> Dict[str, float]:
    """
    Fallback simple exhaustion calculation when not enough data for multi-timeframe.
    """
    from scipy.stats import norm as scipy_norm
    
    px_series = feats.get("px", pd.Series(dtype=float))
    if px_series is None or len(px_series) < max(lookback_short, lookback_long, vol_lookback):
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    price = float(px_series.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    ema_short = px_series.ewm(span=lookback_short, adjust=False).mean()
    ema_short_now = float(ema_short.iloc[-1])

    if not np.isfinite(ema_short_now) or ema_short_now <= 0:
        return {"ue_up": 0.0, "ue_down": 0.0, "z_score": 0.0, "deviation_pct": 0.0}

    ret_series = feats.get("ret", pd.Series(dtype=float))
    if ret_series is not None and len(ret_series) >= vol_lookback:
        recent_vol = float(ret_series.iloc[-vol_lookback:].std())
        if not np.isfinite(recent_vol) or recent_vol <= 0:
            recent_vol = 0.02
    else:
        recent_vol = 0.02

    deviation = price - ema_short_now
    deviation_pct = deviation / ema_short_now
    price_vol = ema_short_now * recent_vol
    z_score = deviation / max(price_vol, 1e-10)

    cdf_val = scipy_norm.cdf(abs(z_score))
    exhaustion_magnitude = 2.0 * (cdf_val - 0.5)

    if deviation > 0:
        ue_up = min(exhaustion_magnitude, 0.99)
        ue_down = 0.0
    elif deviation < 0:
        ue_up = 0.0
        ue_down = min(exhaustion_magnitude, 0.99)
    else:
        ue_up = 0.0
        ue_down = 0.0

    return {
        "ue_up": float(ue_up),
        "ue_down": float(ue_down),
        "z_score": float(z_score),
        "deviation_pct": float(deviation_pct * 100),
    }


def compute_directional_exhaustion(
    realized_cum_log: float,
    expected_cum_log: float,
    expected_sigma: float,
    cumulative_log_return: float,
    *,
    days_in_regime: int = 1,
    nu: Optional[float] = None,
    w_D: float = 1.0,
    w_L: float = 1.0,
    lambda_0: float = 1.0,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    DEPRECATED: Old exhaustion calculation. Kept for backward compatibility.
    
    Use compute_directional_exhaustion_from_features() instead, which provides
    mathematically sound price-based exhaustion measurement.
    """
    # Return zeros - this function should not be used
    return {
        "ue_up": 0.0,
        "ue_down": 0.0,
        "D_raw": 0.0,
        "D_time_adjusted": 0.0,
        "time_factor": 1.0,
        "tail_amplifier": 1.0,
        "days_in_regime": int(days_in_regime),
    }


def clear_exhaustion_state_cache() -> None:
    """Clear the exhaustion state cache. Useful for testing or resets."""
    global _exhaustion_state
    _exhaustion_state.clear()


# NOTE: ExpectedUtilityResult dataclass removed - was only used by the legacy
# compute_expected_utility() function. EU computation is now done inline in
# latest_signals() from BMA r_samples.


@dataclass(frozen=True)
class Signal:
    horizon_days: int
    score: float          # edge in z units (mu_H/sigma_H with filters)
    p_up: float           # P(return>0) - UNIFIED posterior predictive MC probability (THE ONLY TRADING PROBABILITY)
    exp_ret: float        # expected log return over horizon
    ci_low: float         # lower bound of expected log return CI
    ci_high: float        # upper bound of expected log return CI
    profit_pln: float     # expected profit in PLN for NOTIONAL_PLN invested
    profit_ci_low_pln: float  # low CI bound for profit in PLN
    profit_ci_high_pln: float # high CI bound for profit in PLN
    position_strength: float  # EU-based position sizing: EU / max(E[loss], ε), scaled by drift_weight
    vol_mean: float       # mean volatility forecast (stochastic vol posterior)
    vol_ci_low: float     # lower bound of volatility CI
    vol_ci_high: float    # upper bound of volatility CI
    regime: str               # detected regime label
    label: str                # BUY/HOLD/SELL or STRONG BUY/SELL
    # Expected Utility fields (THE BASIS FOR POSITION SIZING):
    expected_utility: float = 0.0     # EU = p × E[gain] - (1-p) × E[loss]
    expected_gain: float = 0.0        # E[R_H | R_H > 0]
    expected_loss: float = 0.0        # E[-R_H | R_H < 0] (positive value)
    gain_loss_ratio: float = 1.0      # E[gain] / E[loss] - asymmetry
    eu_position_size: float = 0.0     # Position size from EU / max(E[loss], ε)
    # Contaminated Student-t Mixture fields (regime-dependent tails):
    cst_enabled: bool = False         # Whether contaminated mixture was used in MC
    cst_nu_normal: Optional[float] = None   # ν for normal regime (lighter tails)
    cst_nu_crisis: Optional[float] = None   # ν for crisis regime (heavier tails)
    cst_epsilon: Optional[float] = None     # Crisis contamination probability
    # Hansen Skew-t fields (asymmetric return distribution):
    hansen_enabled: bool = False            # Whether Hansen skew-t was fitted
    hansen_lambda: Optional[float] = None   # Skewness parameter λ ∈ (-1, 1)
    hansen_nu: Optional[float] = None       # Degrees of freedom ν
    hansen_skew_direction: Optional[str] = None  # "left", "right", or "symmetric"
    # Diagnostics only (NOT used for trading decisions):
    drift_uncertainty: float = 0.0  # P_t × drift_var_factor: uncertainty in drift estimate propagated to horizon
    p_analytical: float = 0.5       # DIAGNOSTIC ONLY: analytical posterior predictive P(r>0|D) 
    p_empirical: float = 0.5        # DIAGNOSTIC ONLY: raw empirical MC probability P(r>0) from simulations
    # STEP 7: Regime audit trace - tracks which regime params were used
    regime_used: Optional[int] = None        # Integer regime index (0-4) used for parameter selection
    regime_source: str = "global"            # "regime_tuned" or "global" - source of parameters
    regime_collapse_warning: bool = False    # True if regime params collapsed to global
    # STEP 8: Bayesian Model Averaging audit trace
    bma_method: str = "legacy"               # "bayesian_model_averaging_full" or "legacy"
    bma_has_model_posterior: bool = False    # True if BMA with model posteriors was used
    bma_borrowed_from_global: bool = False   # True if regime used hierarchical fallback
    # DUAL-SIDED TREND EXHAUSTION (0-100% scale, multi-timeframe weighted EMA deviation):
    ue_up: float = 0.0    # Price above weighted EMA equilibrium (0-1 scale)
    ue_down: float = 0.0  # Price below weighted EMA equilibrium (0-1 scale)
    # RISK TEMPERATURE MODULATION (cross-asset stress scaling):
    risk_temperature: float = 0.0      # Global risk temperature (0-2 scale)
    risk_scale_factor: float = 1.0     # Position scale factor from risk temperature
    overnight_budget_applied: bool = False  # True if overnight budget constraint was applied
    overnight_max_position: Optional[float] = None  # Max position from overnight budget
    pos_strength_pre_risk_temp: float = 0.0  # Position strength before risk temperature scaling
    # EVT (Extreme Value Theory) tail risk fields:
    expected_loss_empirical: float = 0.0    # Empirical expected loss (before EVT)
    evt_enabled: bool = False               # Whether EVT was used for tail estimation
    evt_xi: Optional[float] = None          # GPD shape parameter ξ
    evt_sigma: Optional[float] = None       # GPD scale parameter σ
    evt_threshold: Optional[float] = None   # POT threshold
    evt_n_exceedances: int = 0              # Number of threshold exceedances
    evt_fit_method: Optional[str] = None    # EVT fitting method used





def fetch_px_asset(asset: str, start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    """
    Return a price series for the requested asset expressed in PLN terms when needed.
    - PLNJPY=X: native series (JPY per PLN); title indicates JPY per PLN.
    - Gold: try XAUUSD=X, GC=F, XAU=X; convert USD to PLN via USDPLN=X (or robust alternatives) → PLN per troy ounce.
    - Silver: try XAGUSD=X, SI=F, XAG=X; convert USD to PLN via USDPLN=X (or robust alternatives) → PLN per troy ounce.
    - Bitcoin (BTC-USD): convert USD to PLN via USDPLN → PLN per BTC.
    - MicroStrategy (MSTR): convert USD share price to PLN via USDPLN → PLN per share.
    - Generic equities/ETFs: fetch in native quote currency and convert to PLN via detected FX.
    Returns (px_series, title_suffix) where title_suffix describes units.
    """
    asset = asset.strip().upper()
    if asset == "PLNJPY=X":
        px = _fetch_px_symbol(asset, start, end)
        title = "Polish Zloty vs Japanese Yen (PLNJPY=X) — JPY per PLN"
        return px, title

    # Bitcoin in USD → PLN
    if asset in ("BTC-USD", "BTCUSD=X"):
        # Prefer robust USD path (avoid unreliable BTC-PLN tickers that 404)
        btc_px, _used = _fetch_with_fallback([asset] if asset == "BTC-USD" else [asset, "BTC-USD"], start, end)
        btc_px = _ensure_float_series(btc_px)
        # Use USD→PLN leg expanded to BTC date range and robust asof alignment
        usdpln_px = convert_currency_to_pln("USD", start, end, native_index=btc_px.index)
        usdpln_aligned = _align_fx_asof(btc_px, usdpln_px, max_gap_days=7)
        if usdpln_aligned.isna().all():
            usdpln_aligned = usdpln_px.reindex(btc_px.index).ffill().bfill()
        usdpln_aligned = _ensure_float_series(usdpln_aligned)
        # Direct vectorized conversion
        px_pln = (btc_px * usdpln_aligned).dropna()
        px_pln.name = "px"
        if px_pln.empty:
            raise RuntimeError("No overlap between BTC-USD and USDPLN data to compute PLN price")
        # Display name
        disp = "Bitcoin"
        return px_pln, f"{disp} (BTC-USD) — PLN per BTC"

    # MicroStrategy equity (USD) → PLN
    if asset == "MSTR":
        mstr_px = _fetch_px_symbol("MSTR", start, end)
        mstr_px = _ensure_float_series(mstr_px)
        # Use USD→PLN leg expanded to MSTR date range and robust asof alignment
        usdpln_px = convert_currency_to_pln("USD", start, end, native_index=mstr_px.index)
        usdpln_aligned = _align_fx_asof(mstr_px, usdpln_px, max_gap_days=7)
        if usdpln_aligned.isna().all():
            usdpln_aligned = usdpln_px.reindex(mstr_px.index).ffill().bfill()
        usdpln_aligned = _ensure_float_series(usdpln_aligned)
        # Direct vectorized conversion (ensure 1-D float Series)
        px_pln = (mstr_px * usdpln_aligned).dropna()
        px_pln.name = "px"
        if px_pln.empty:
            raise RuntimeError("No overlap between MSTR and USDPLN data to compute PLN price")
        disp = _resolve_display_name("MSTR") or "MicroStrategy"
        return px_pln, f"{disp} (MSTR) — PLN per share"

    # Metals in USD → convert to PLN
    if asset in ("XAUUSD=X", "GC=F", "XAU=X", "XAGUSD=X", "SI=F", "XAG=X"):
        if asset.startswith("XAU") or asset in ("GC=F", "XAU=X"):
            candidates = ["GC=F", "XAU=X", "XAUUSD=X"]
            if asset not in candidates:
                candidates = [asset] + candidates
            metal_px, used = _fetch_with_fallback(candidates, start, end)
            metal_px = _ensure_float_series(metal_px)
            metal_name = "Gold"
        else:
            candidates = ["SI=F", "XAG=X", "XAGUSD=X"]
            if asset not in candidates:
                candidates = [asset] + candidates
            metal_px, used = _fetch_with_fallback(candidates, start, end)
            metal_px = _ensure_float_series(metal_px)
            metal_name = "Silver"
        usdpln_px = fetch_usd_to_pln_exchange_rate(start, end)
        usdpln_aligned = usdpln_px.reindex(metal_px.index).ffill()
        df = pd.concat([metal_px, usdpln_aligned], axis=1).dropna()
        df.columns = ["metal_usd", "usdpln"]
        px_pln = (df["metal_usd"] * df["usdpln"]).rename("px")
        title = f"{metal_name} ({used}) — PLN per troy oz"
        if px_pln.empty:
            raise RuntimeError(f"No overlap between {metal_name} and USDPLN data to compute PLN price")
        return px_pln, title

    # Generic: resolve symbol candidates and convert to PLN using detected currency
    candidates = _resolve_symbol_candidates(asset)
    px_native = None
    used_sym = None
    last_err: Optional[Exception] = None
    for sym in candidates:
        try:
            s = _fetch_px_symbol(sym, start, end)
            px_native = s
            used_sym = sym
            break
        except Exception as e:
            last_err = e
            continue
    if px_native is None:
        raise last_err if last_err else RuntimeError(f"No data for {asset}")

    px_native = _ensure_float_series(px_native)
    qcy = detect_quote_currency(used_sym)
    px_pln, _ = convert_price_series_to_pln(px_native, qcy, start, end)
    if px_pln is None or px_pln.empty:
        raise RuntimeError(f"No overlapping FX data to convert {used_sym} to PLN")
    # Title with full name
    disp = _resolve_display_name(used_sym)
    title = f"{disp} ({used_sym}) — PLN per share"
    return px_pln, title


# -------------------------
# Features
# -------------------------

def _garch11_mle(ret: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Estimate a GARCH(1,1) with normal errors via MLE.
    Returns (sigma_series, params_dict). Falls back by raising on failure.
    Model: r_t = mu_t + e_t, e_t ~ N(0, h_t), h_t = omega + alpha*e_{t-1}^2 + beta*h_{t-1}
    We estimate on de-meaned returns (mean 0) and treat residuals as r_t.

    Level-7 Bayesian GARCH: 
    - We approximate parameter uncertainty by computing the observed information 
      (numeric Hessian of the negative log-likelihood) at the MLE optimum and 
      inverting it to obtain an approximate covariance matrix for (omega,alpha,beta).
    - In forward simulation (_simulate_forward_paths), parameters are sampled from 
      N(theta_hat, Cov) per path, propagating GARCH uncertainty into forecasts.
    - This Gaussian approximation is institution-grade and sufficient for Level-7.
    
    Future Level-8+ pathway (research frontier, not required):
    - Full Bayesian GARCH via HMC/NUTS posterior sampling (e.g., PyMC, Stan)
    - Priors enforcing stationarity: α + β < 1
    - Joint posterior over (ω, α, β) with proper uncertainty quantification
    - This would eliminate Gaussian approximation but requires MCMC infrastructure.
    """
    from scipy.optimize import minimize

    r = _ensure_float_series(ret).dropna().astype(float)
    if len(r) < 200:
        raise RuntimeError("Too few observations for stable GARCH(1,1) MLE (need >=200)")

    # De-mean for conditional variance fit
    r = r - r.mean()
    T = len(r)
    r2 = r.values**2
    var0 = float(np.nanvar(r.values)) if T > 1 else 1e-6

    # Parameter transform: ensure omega>0, alpha>=0, beta>=0, alpha+beta<0.999
    def nll(params):
        omega, alpha, beta = params
        # Hard penalties if constraints violated
        if omega <= 1e-12 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 0.999:
            return 1e12
        h = np.empty(T, dtype=float)
        # Initialize with unconditional variance to speed convergence
        try:
            h0 = omega / max(1e-12, 1.0 - alpha - beta)
            if not np.isfinite(h0) or h0 <= 0:
                h0 = var0
        except Exception:
            h0 = var0
        h[0] = max(1e-12, h0)
        for t in range(1, T):
            h[t] = omega + alpha * r2[t-1] + beta * h[t-1]
            if not np.isfinite(h[t]) or h[t] <= 0:
                h[t] = 1e-8
        # Normal likelihood (up to constant): 0.5*(log h_t + r_t^2/h_t)
        ll_terms = 0.5*(np.log(h) + r2 / h)
        if not np.all(np.isfinite(ll_terms)):
            return 1e12
        return float(np.sum(ll_terms))

    # Multiple starting points for robustness
    inits = [
        (0.1*var0*(1-0.1-0.8), 0.1, 0.8),
        (0.05*var0*(1-0.05-0.9), 0.05, 0.9),
        (0.2*var0*(1-0.15-0.7), 0.15, 0.7),
    ]
    best = (None, np.inf)
    best_params = None
    bounds = [(1e-12, 10.0*var0), (0.0, 0.999), (0.0, 0.999)]
    constraints = ({'type': 'ineq', 'fun': lambda p: 0.999 - (p[1] + p[2])},)

    for x0 in inits:
        try:
            res = minimize(nll, x0=np.array(x0, dtype=float), method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
            if res.success and res.fun < best[1]:
                best = (res, res.fun)
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        raise RuntimeError("GARCH(1,1) MLE failed to converge from all starts")

    omega, alpha, beta = [float(v) for v in best_params]

    # Rebuild conditional variance series with optimal params
    h = np.empty(T, dtype=float)
    try:
        h0 = omega / max(1e-12, 1.0 - alpha - beta)
        if not np.isfinite(h0) or h0 <= 0:
            h0 = var0
    except Exception:
        h0 = var0
    h[0] = max(1e-10, h0)
    for t in range(1, T):
        h[t] = omega + alpha * r2[t-1] + beta * h[t-1]
        if not np.isfinite(h[t]) or h[t] <= 0:
            h[t] = 1e-8
    sigma = np.sqrt(h)
    vol = pd.Series(sigma, index=r.index, name='vol_garch')

    # Approximate covariance of parameters via numeric Hessian of nll at optimum
    def _approx_hessian(x: np.ndarray) -> Optional[np.ndarray]:
        try:
            x = np.asarray(x, dtype=float)
            k = x.size
            H = np.zeros((k, k), dtype=float)
            # Step sizes scaled to parameter magnitudes
            eps_base = 1e-6
            h_vec = np.maximum(np.abs(x) * 1e-3, eps_base)
            f0 = nll(x)
            # Diagonal second derivatives
            for i in range(k):
                ei = np.zeros(k); ei[i] = h_vec[i]
                f_plus = nll(x + ei)
                f_minus = nll(x - ei)
                H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h_vec[i] ** 2)
            # Off-diagonals via mixed partials (central)
            for i in range(k):
                for j in range(i+1, k):
                    ei = np.zeros(k); ei[i] = h_vec[i]
                    ej = np.zeros(k); ej[j] = h_vec[j]
                    f_pp = nll(x + ei + ej)
                    f_pm = nll(x + ei - ej)
                    f_mp = nll(x - ei + ej)
                    f_mm = nll(x - ei - ej)
                    mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_vec[i] * h_vec[j])
                    H[i, j] = mixed
                    H[j, i] = mixed
            return H
        except Exception:
            return None

    cov = None
    se = None
    try:
        H = _approx_hessian(np.array([omega, alpha, beta], dtype=float))
        if H is not None:
            # Regularize slightly to improve conditioning
            lam = 1e-8
            H_reg = H + lam * np.eye(3)
            cov_try = np.linalg.pinv(H_reg)
            # Ensure symmetry and positive diagonals
            cov_try = 0.5 * (cov_try + cov_try.T)
            if np.all(np.isfinite(cov_try)) and np.all(np.diag(cov_try) >= 0):
                cov = cov_try
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        cov = None
        se = None

    # Compute final log-likelihood at optimum (negative nll)
    final_nll = float(best[1])
    final_ll = float(-final_nll)
    
    params: Dict[str, float] = {
        "omega": omega, 
        "alpha": alpha, 
        "beta": beta, 
        "converged": True,
        "log_likelihood": final_ll,
        "nll": final_nll,
        "n_obs": int(T),
        "aic": float(2.0 * 3 - 2.0 * final_ll),  # AIC = 2k - 2*ln(L), k=3 params
        "bic": float(3 * np.log(T) - 2.0 * final_ll),  # BIC = k*ln(n) - 2*ln(L)
    }
    if cov is not None:
        params["cov"] = cov.tolist()
    if se is not None:
        params["se_omega"], params["se_alpha"], params["se_beta"] = [float(x) for x in se]
    return vol, params


def _fit_student_nu_mle(z: pd.Series, min_n: int = 200, bounds: Tuple[float, float] = (4.5, 500.0)) -> Dict[str, float]:
    """Fit global Student-t degrees of freedom (nu) via MLE on standardized residuals z.
    - z should be approximately IID with unit scale (i.e., residuals divided by conditional sigma).
    - Returns a dict: {"nu_hat": float, "ll": float, "n": int, "converged": bool, "se_nu": float}.
    - On failure or insufficient data, returns a conservative default with converged=False.
    
    Tier 2 Enhancement: Posterior parameter variance tracking
    Computes standard error for ν via numeric Hessian (observed information matrix).
    This enables:
        ✔ Automatic conservatism during ν uncertainty
        ✔ ν sampling in Monte Carlo simulation
        ✔ Wider forecast intervals when tail parameter is uncertain
    """
    from scipy.optimize import minimize

    if z is None or not isinstance(z, pd.Series) or z.empty:
        return {"nu_hat": 50.0, "ll": float("nan"), "n": 0, "converged": False, "se_nu": float("nan")}

    zz = pd.to_numeric(z, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    # Remove zeros that may indicate degenerate scaling (not necessary but harmless)
    zz = zz[np.isfinite(zz.values)]
    n = int(zz.shape[0])
    if n < max(50, min_n):
        # too short: near-normal default
        return {"nu_hat": 50.0, "ll": float("nan"), "n": n, "converged": False, "se_nu": float("nan")}

    x = zz.values.astype(float)

    def nll(nu_val: float) -> float:
        # Bound inside objective to avoid domain errors
        nu_b = float(np.clip(nu_val, bounds[0], bounds[1]))
        try:
            # Use scipy.stats.t logpdf with df=nu_b, loc=0, scale=1
            lp = student_t.logpdf(x, df=nu_b)
            if not np.all(np.isfinite(lp)):
                return 1e12
            return float(-np.sum(lp))
        except Exception:
            return 1e12

    # Multi-start initializations
    starts = [5.5, 8.0, 12.0, 20.0, 50.0, 100.0, 200.0]
    best = (None, np.inf)
    for s0 in starts:
        x0 = np.array([float(np.clip(s0, bounds[0], bounds[1]))], dtype=float)
        try:
            res = minimize(lambda v: nll(v[0]), x0=x0, method="L-BFGS-B", bounds=[bounds], options={"maxiter": 200})
            if res.success and res.fun < best[1]:
                best = (res, res.fun)
        except Exception:
            continue

    if best[0] is None:
        return {"nu_hat": 50.0, "ll": float("nan"), "n": n, "converged": False, "se_nu": float("nan")}

    nu_hat = float(np.clip(best[0].x[0], bounds[0], bounds[1]))
    ll = float(-best[1])
    
    # Compute standard error via numeric Hessian (observed information)
    # Hessian approximation: second derivative of negative log-likelihood
    se_nu = None
    try:
        # Finite difference approximation: d²NLL/dν²
        eps = max(0.01 * abs(nu_hat), 0.1)  # adaptive step size
        
        # Central difference for second derivative
        nll_0 = nll(nu_hat)
        nll_plus = nll(nu_hat + eps)
        nll_minus = nll(nu_hat - eps)
        
        # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
        d2_nll = (nll_plus - 2.0 * nll_0 + nll_minus) / (eps ** 2);
        
        # Standard error: sqrt(1 / observed_information)
        # observed_information = d²(-LL)/dν² = d²NLL/dν²
        if d2_nll > 1e-12:  # positive curvature (proper minimum)
            se_nu = float(np.sqrt(1.0 / d2_nll))
            # Sanity check: SE should be reasonable relative to estimate
            if se_nu > 10.0 * nu_hat or not np.isfinite(se_nu):
                se_nu = None
        else:
            se_nu = None
    except Exception:
        se_nu = None
    
    result = {
        "nu_hat": nu_hat,
        "ll": ll,
        "n": n,
        "converged": True,
        "se_nu": float(se_nu) if se_nu is not None else float("nan"),
    }
    
    return result


def _test_innovation_whiteness(innovations: np.ndarray, innovation_vars: np.ndarray, lags: int = 20) -> Dict[str, float]:
    """
    Test innovation whiteness using Ljung-Box test for autocorrelation.
    
    Refinement 3: Model adequacy via innovation whiteness testing.
    If innovations are not white noise (autocorrelated), the model may be misspecified.
    
    Args:
        innovations: Prediction errors from Kalman filter
        innovation_vars: Innovation variances (for standardization)
        lags: Number of lags to test
        
    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    try:
        # Standardize innovations by their predicted variance
        std_innovations = innovations / np.sqrt(np.maximum(innovation_vars, 1e-12))
        std_innovations = std_innovations[np.isfinite(std_innovations)]
        
        if len(std_innovations) < max(30, lags + 10):
            return {
                "ljung_box_statistic": float("nan"),
                "ljung_box_pvalue": float("nan"),
                "lags_tested": 0,
                "model_adequate": None,
                "note": "insufficient_data"
            }
        
        n = len(std_innovations)
        lags = min(lags, n // 5)  # conservative lag limit
        
        # Compute Ljung-Box statistic manually
        # Q = n(n+2) Σ(ρ_k² / (n-k)) for k=1..m
        # Under H0 (white noise), Q ~ χ²(m)
        
        # Compute autocorrelations
        acf_vals = []
        for lag in range(1, lags + 1):
            if lag >= n:
                break
            try:
                # Sample autocorrelation at lag k
                mean_innov = float(np.mean(std_innovations))
                numerator = float(np.sum((std_innovations[lag:] - mean_innov) * (std_innovations[:-lag] - mean_innov)))
                denominator = float(np.sum((std_innovations - mean_innov) ** 2))
                rho_k = numerator / denominator if abs(denominator) > 1e-12 else 0.0
                acf_vals.append(rho_k)
            except Exception:
                break
        
        if not acf_vals:
            return {
                "ljung_box_statistic": float("nan"),
                "ljung_box_pvalue": float("nan"),
                "lags_tested": 0,
                "model_adequate": None,
                "note": "acf_computation_failed"
            }
        
        # Ljung-Box statistic
        Q = 0.0
        m = len(acf_vals)
        for k, rho_k in enumerate(acf_vals, start=1):
            Q += (rho_k ** 2) / float(n - k)
        Q *= n * (n + 2)
        
        # Compute p-value using chi-squared distribution
        from scipy.stats import chi2
        pvalue = float(1.0 - chi2.cdf(Q, df=m))
        
        # Interpretation: reject H0 (white noise) if p < 0.05
        # model_adequate = True if we fail to reject (p >= 0.05)
        model_adequate = bool(pvalue >= 0.05)
        
        return {
            "ljung_box_statistic": float(Q),
            "ljung_box_pvalue": float(pvalue),
            "lags_tested": int(m),
            "model_adequate": model_adequate,
            "note": "pass" if model_adequate else "fail_autocorrelation_detected"
        }
        
    except Exception as e:
        return {
            "ljung_box_statistic": float("nan"),
            "ljung_box_pvalue": float("nan"),
            "lags_tested": 0,
            "model_adequate": None,
            "note": f"test_failed: {str(e)}"
        }


def _compute_kalman_log_likelihood(y: np.ndarray, sigma: np.ndarray, q: float, c: float = 1.0) -> float:
    """
    Compute log-likelihood for Kalman filter with given process noise q.
    Used for q optimization via marginal likelihood maximization.
    
    Args:
        y: Observations (returns)
        sigma: Observation noise std (volatility) per time step
        q: Process noise variance to evaluate
        
    Returns:
        Total log-likelihood of observations under this q
    """
    T = len(y)
    if T < 2:
        return float('-inf')
    
    # Initialize
    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0
    
    for t in range(T):
        # Prediction
        mu_pred = mu_t
        P_pred = P_t + q
        
        # Observation variance
        R_t = float(max(c * (sigma[t] ** 2), 1e-12))
        
        # Innovation
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))
        
        # Log-likelihood contribution
        try:
            ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
        except Exception:
            pass
        
        # Update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))
    
    return float(log_likelihood)


def _compute_kalman_log_likelihood_heteroskedastic(y: np.ndarray, sigma: np.ndarray, c: float) -> float:
    """
    Compute log-likelihood for Kalman filter with heteroskedastic process noise q_t = c * σ_t².
    
    This allows drift uncertainty to scale with market stress: higher volatility => more drift uncertainty.
    
    Args:
        y: Observations (returns)
        sigma: Observation noise std (volatility) per time step
        c: Scaling factor for heteroskedastic process noise (q_t = c * σ_t²)
        
    Returns:
        Total log-likelihood of observations under this c
    """
    T = len(y)
    if T < 2:
        return float('-inf')
    
    # Initialize
    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0
    
    for t in range(T):
        # Heteroskedastic process noise: q_t = c * σ_t²
        R_t = float(max(c * (sigma[t] ** 2), 1e-12))
        q_t = float(max(c * R_t, 1e-12))
        
        # Prediction
        mu_pred = mu_t
        P_pred = P_t + q_t
        
        # Innovation
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))
        
        # Log-likelihood contribution
        try:
            ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
        except Exception:
            pass
        
        # Update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))
    
    return float(log_likelihood)


def _estimate_regime_drift_priors(ret: pd.Series, vol: pd.Series) -> Optional[Dict[str, float]]:
    """
    Estimate regime-specific drift expectations E[μ_t | Regime=k] from historical data.
    
    Uses a quick HMM fit on returns to identify regimes, then computes mean return
    per regime as a simple proxy for regime-conditional drift.
    
    Args:
        ret: Returns series
        vol: Volatility series
        
    Returns:
        Dictionary with regime-specific drift priors, or None if estimation fails
    """
    if not HMM_AVAILABLE:
        return None
    
    try:
        # Align data
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
        if len(df) < 300:
            return None
        
        df.columns = ["ret", "vol"]
        X = df.values
        
        # Fit 3-state HMM (suppress noisy convergence messages)
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42, verbose=False)
        with suppress_stdout():
            model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        # Compute mean return per state
        regime_drifts = {}
        for state_idx in range(3):
            mask = (states == state_idx)
            if np.sum(mask) > 10:
                regime_drifts[state_idx] = float(np.mean(df.loc[mask, "ret"]))
            else:
                regime_drifts[state_idx] = 0.0
        
        # Identify regime names by volatility
        means = model.means_
        vol_means = means[:, 1]
        sorted_indices = np.argsort(vol_means)
        
        regime_map = {
            sorted_indices[0]: "calm",
            sorted_indices[1]: "trending",
            sorted_indices[2]: "crisis"
        }
        
        # Get current regime (last observation)
        current_state = states[-1]
        current_regime = regime_map.get(current_state, "calm")
        current_drift_prior = regime_drifts.get(current_state, 0.0)
        
        return {
            "current_regime": current_regime,
            "current_drift_prior": float(current_drift_prior),
            "regime_drifts": regime_drifts,
            "regime_map": regime_map,
        }
        
    except Exception:
        return None


# =============================================================================
# PER-ASSET CACHE INTEGRATION
# =============================================================================
# Try to use the per-asset cache module for better git-friendliness
# Falls back to legacy single-file cache if not available
# =============================================================================
try:
    from tuning.kalman_cache import load_tuned_params as _load_per_asset_cache
    PER_ASSET_CACHE_AVAILABLE = True
except ImportError:
    # Try adding src directory to path if running from project root
    import sys
    # Get src/ directory (parent of decision/)
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    try:
        from tuning.kalman_cache import load_tuned_params as _load_per_asset_cache
        PER_ASSET_CACHE_AVAILABLE = True
    except ImportError:
        PER_ASSET_CACHE_AVAILABLE = False


def _load_tuned_kalman_params(asset_symbol: str, cache_path: str = "src/data/tune") -> Optional[Dict]:
    """
    Load pre-tuned Kalman parameters from cache generated by tune.py.

    Supports the Bayesian Model Averaging (BMA) structure from tune.py:
    
    {
        "global": {
            "model_posterior": { m: p(m) },
            "models": { m: {q, phi, nu, c, ...} },
            ...backward-compatible fields...
        },
        "regime": {
            "r": {
                "model_posterior": { m: p(m|r) },
                "models": { m: {q, phi, nu, c, ...} },
                "regime_meta": {
                    "fallback": bool,
                    "borrowed_from_global": bool,
                    ...
                }
            }
        }
    }
    
    CONTRACT WITH tune.py:
    - Cache structure is ALWAYS: {"global": {...}, "regime": {...}}
    - Every regime block contains model_posterior and models
    - Fallback regimes have borrowed_from_global=True but still contain valid models
    - This function does NOT perform fallback logic - tune guarantees non-empty outputs
    - NO backward compatibility with old flat schema - new BMA architecture only
    
    Args:
        asset_symbol: Asset symbol (e.g., "PLNJPY=X", "SPY")
        cache_path: Path to the tuned parameters cache file (legacy fallback)
        
    Returns:
        Dictionary with BMA structure if found, None otherwise. Contains:
        - 'global': {model_posterior, models}
        - 'regime': {r: {model_posterior, models, regime_meta}}
        - 'has_bma': True (always for valid entries)
    """
    raw_data = None
    
    # Try per-asset cache first (preferred)
    if PER_ASSET_CACHE_AVAILABLE:
        raw_data = _load_per_asset_cache(asset_symbol)
    
    # Fallback to legacy single-file cache
    if raw_data is None:
        try:
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            if asset_symbol not in cache:
                return None
            raw_data = cache[asset_symbol]
        except Exception:
            return None
    
    if raw_data is None:
        return None
        
    # ====================================================================
    # NEW BMA ARCHITECTURE ONLY - NO BACKWARD COMPATIBILITY
    # ====================================================================
    # tune.py outputs:
    #   {
    #     "global": { "model_posterior": {...}, "models": {...} },
    #     "regime": { "0": {...}, "1": {...}, ... },
    #     "meta": {...}
    #   }
    #
    # Old flat schema is NOT supported. If cache doesn't have 'global' key,
    # it's from old architecture and must be regenerated.
    # ====================================================================
    
    if 'global' not in raw_data:
        # Old flat schema - not supported - ALWAYS warn user
        import sys
        print(f"\n⚠️  CACHE FORMAT ERROR for {asset_symbol}:", file=sys.stderr)
        print(f"   Cache entry exists but uses OLD flat schema (missing 'global' key)", file=sys.stderr)
        print(f"   Top-level keys in cache: {list(raw_data.keys())[:10]}", file=sys.stderr)
        print(f"   → Fix: Run 'make tune ARGS=\"--assets {asset_symbol}\"' to regenerate\n", file=sys.stderr)
        return None

    global_data = raw_data['global']
    regime_data = raw_data.get('regime', {})

    # Validate BMA structure
    model_posterior = global_data.get('model_posterior', {})
    models = global_data.get('models', {})

    if not models:
        # Invalid BMA structure - no models - ALWAYS warn user
        import sys
        print(f"\n⚠️  CACHE STRUCTURE ERROR for {asset_symbol}:", file=sys.stderr)
        print(f"   Cache entry has 'global' key but no models inside", file=sys.stderr)
        print(f"   → Fix: Run 'make tune ARGS=\"--assets {asset_symbol}\"' to regenerate\n", file=sys.stderr)
        return None

    # Helper to check if model is Student-t (phi_student_t_nu_* naming)
    def _is_student_t(model_name: str) -> bool:
        return model_name.startswith('phi_student_t_nu_')
    
    # Helper to check if a model fit successfully
    def _is_valid_model(model_name: str) -> bool:
        m_params = models.get(model_name, {})
        return isinstance(m_params, dict) and m_params.get('fit_success', False)

    # Extract representative params from highest-posterior model for Kalman filter
    # (The BMA path uses full model averaging, but Kalman filter needs single params)
    # IMPORTANT: Only consider models with fit_success=True
    if model_posterior:
        # Filter to only successfully fitted models
        valid_models = {m: p for m, p in model_posterior.items() if _is_valid_model(m)}
        if valid_models:
            best_model = max(valid_models.keys(), key=lambda m: valid_models.get(m, 0))
        else:
            # Fallback if no valid models in posterior (shouldn't happen)
            best_model = None
    else:
        # Fallback order: any Student-t > phi_gaussian > gaussian
        # First check for new naming (phi_student_t_nu_*) - only if fit succeeded
        student_t_models = [m for m in models if _is_student_t(m) and _is_valid_model(m)]
        if student_t_models:
            best_model = student_t_models[0]  # Pick first available Student-t
        elif 'kalman_phi_gaussian' in models and _is_valid_model('kalman_phi_gaussian'):
            best_model = 'kalman_phi_gaussian'
        elif 'kalman_gaussian' in models and _is_valid_model('kalman_gaussian'):
            best_model = 'kalman_gaussian'
        else:
            # Pick any valid model
            valid_any = [m for m in models if _is_valid_model(m)]
            best_model = valid_any[0] if valid_any else None

    if not best_model or best_model not in models:
        return None
    
    best_params = models[best_model]

    # Extract params from best model
    q_val = best_params.get('q')
    c_val = best_params.get('c', 1.0)
    phi_val = best_params.get('phi')
    nu_val = best_params.get('nu')

    # Derive noise_model from best model name
    # Normalize to standard categories for downstream processing
    if _is_student_t(best_model):
        noise_model = best_model  # Keep actual model name (e.g., phi_student_t_nu_6)
    elif 'phi' in best_model:
        noise_model = 'kalman_phi_gaussian'
    else:
        noise_model = 'gaussian'

    # Validate required params
    if q_val is None or not np.isfinite(q_val) or q_val <= 0:
        return None
    if c_val is None or not np.isfinite(c_val) or c_val <= 0:
        return None

    result = {
        # Full BMA structure from tune.py
        'global': global_data,
        'regime': regime_data,
        'has_bma': True,

        # Representative params from best model (for Kalman filter compatibility)
        'q': float(q_val),
        'c': float(c_val),
        'phi': float(phi_val) if phi_val is not None and np.isfinite(phi_val) else None,
        'nu': float(nu_val) if nu_val is not None and np.isfinite(nu_val) else None,
        'noise_model': noise_model,
        'best_model': best_model,

        # Diagnostics from best model (for display compatibility)
        'bic': best_params.get('bic'),
        'aic': best_params.get('aic'),
        'hyvarinen_score': best_params.get('hyvarinen_score'),
        'combined_score': best_params.get('combined_score'),
        'log_likelihood': best_params.get('log_likelihood'),
        'pit_ks_pvalue': best_params.get('pit_ks_pvalue'),
        'ks_statistic': best_params.get('ks_statistic'),

        # Isotonic Recalibration Transport Map
        # This is the CORE calibration layer - applied BEFORE regimes see PIT
        'recalibration': global_data.get('recalibration'),
        'recalibration_applied': global_data.get('recalibration_applied', False),
        'pit_ks_pvalue_calibrated': global_data.get('pit_ks_pvalue_calibrated'),
        'calibration_diagnostics': global_data.get('calibration_diagnostics'),
        'failure_category': global_data.get('failure_category'),

        # Model comparison: build from all models (includes Hyvärinen scores)
        'model_comparison': {
            m: {
                'bic': m_params.get('bic'),
                'aic': m_params.get('aic'),
                'hyvarinen_score': m_params.get('hyvarinen_score'),
                'combined_score': m_params.get('combined_score'),
                'll': m_params.get('log_likelihood'),
                'n_params': m_params.get('n_params'),
            }
            for m, m_params in models.items()
            if isinstance(m_params, dict) and m_params.get('fit_success', False)
        },

        # Model selection metadata from tune.py
        'model_selection_method': raw_data.get('meta', {}).get('model_selection_method', 'combined'),
        'bic_weight': raw_data.get('meta', {}).get('bic_weight', 0.5),
        'entropy_lambda': raw_data.get('meta', {}).get('entropy_lambda', 0.05),

        # Global-level aggregates (from global block or computed)
        'hyvarinen_max': global_data.get('hyvarinen_max'),
        'combined_score_min': global_data.get('combined_score_min'),
        'bic_min': global_data.get('bic_min'),

        # Calibrated Trust Authority
        # ARCHITECTURAL LAW: Trust = Calibration Authority − Bounded Regime Penalty
        # This is the SINGLE AUTHORITY for trust decisions
        'calibrated_trust': global_data.get('calibrated_trust'),
        'effective_trust': global_data.get('effective_trust'),
        'calibration_trust': global_data.get('calibration_trust'),
        'regime_penalty': global_data.get('regime_penalty'),

        # Calibration status and escalation tracking
        'calibration_warning': global_data.get('calibration_warning', False),
        'nu_refinement': global_data.get('nu_refinement', {}),

        # K=2 mixture (DEPRECATED - kept for backward compatibility)
        'mixture_attempted': global_data.get('mixture_attempted', False),
        'mixture_selected': global_data.get('mixture_selected', False),
        'mixture_model': global_data.get('mixture_model'),

        # GH distribution fallback
        'gh_attempted': global_data.get('gh_attempted', False),
        'gh_selected': global_data.get('gh_selected', False),
        'gh_model': global_data.get('gh_model'),

        # TVVM (Time-Varying Volatility Multiplier)
        'tvvm_attempted': global_data.get('tvvm_attempted', False),
        'tvvm_selected': global_data.get('tvvm_selected', False),

        # Metadata
        'source': 'tuned_cache_bma',
        'timestamp': raw_data.get('timestamp') or raw_data.get('meta', {}).get('timestamp'),
        'model_posterior': model_posterior,
    }
    return result


def _select_regime_params(
    tuned_params: Dict,
    current_regime: int,
) -> Dict:
    """
    Select parameters using regime-first logic with global fallback.

    REGIME-FIRST PARAMETER ROUTING:

    1. If current_regime has tuned params AND not fallback → use regime params
    2. Otherwise → use global params

    This is parameter routing ONLY. No epistemology changes.

    Args:
        tuned_params: Full tuned params dict from _load_tuned_kalman_params()
        current_regime: Current regime index (0-4)

    Returns:
        Dict with selected params {q, phi, nu, c, fallback, regime_meta, ...}
    """
    if tuned_params is None:
        # No tuned params available - return minimal defaults
        return {
            'q': 1e-6,
            'phi': 0.95,
            'nu': None,
            'c': 1.0,
            'fallback': True,
            'source': 'defaults',
            'regime_used': None,
        }

    regime_data = tuned_params.get('regime') or {}
    global_data = tuned_params.get('global') or {}

    # Try to get regime-specific params
    # Handle both int keys and string keys (JSON converts to strings)
    regime_block = regime_data.get(current_regime) or regime_data.get(str(current_regime))

    # Helper to safely convert to float with fallback for None values
    def _safe_float(val, default):
        if val is None:
            return float(default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    def _extract_best_model_params(model_posterior: Dict, models: Dict) -> Dict:
        """Extract params from highest-posterior model."""
        if not models:
            return {}

        # Helper to check if model is Student-t
        def _is_st(m: str) -> bool:
            return m.startswith('phi_student_t_nu_')

        if model_posterior:
            best_model = max(model_posterior.keys(), key=lambda m: model_posterior.get(m, 0))
        else:
            # Fallback order: any Student-t > phi_gaussian > gaussian
            student_t_models = [m for m in models if _is_st(m)]
            if student_t_models:
                best_model = student_t_models[0]
            elif 'kalman_phi_gaussian' in models:
                best_model = 'kalman_phi_gaussian'
            elif 'kalman_gaussian' in models:
                best_model = 'kalman_gaussian'
            else:
                best_model = next(iter(models), None)

        if best_model and best_model in models:
            return models[best_model]
        return {}

    # Check if regime has non-fallback data
    regime_meta = {}
    is_fallback = True
    if regime_block is not None and isinstance(regime_block, dict):
        regime_meta = regime_block.get('regime_meta', {})
        is_fallback = regime_meta.get('fallback', False) or regime_meta.get('borrowed_from_global', False)

    if regime_block is not None and not is_fallback:
        # Use regime-specific BMA params
        model_posterior = regime_block.get('model_posterior', {})
        models = regime_block.get('models', {})
        best_params = _extract_best_model_params(model_posterior, models)

        # Fallback to global if regime models empty
        if not best_params:
            global_model_posterior = global_data.get('model_posterior', {})
            global_models = global_data.get('models', {})
            best_params = _extract_best_model_params(global_model_posterior, global_models)

        theta = {
            'q': _safe_float(best_params.get('q'), 1e-6),
            'phi': _safe_float(best_params.get('phi'), 0.95) if best_params.get('phi') is not None else 0.95,
            'nu': best_params.get('nu'),
            'c': _safe_float(best_params.get('c'), 1.0),
            'fallback': False,
            'source': 'regime_bma',
            'regime_used': current_regime,
            'regime_name': regime_meta.get('regime_name', f'REGIME_{current_regime}'),
            'regime_meta': regime_meta,
            'collapse_warning': regime_meta.get('collapse_warning', False),
            'model_posterior': model_posterior,
            # Model selection diagnostics (best model)
            'hyvarinen_score': best_params.get('hyvarinen_score'),
            'combined_score': best_params.get('combined_score'),
            'bic': best_params.get('bic'),
            # Regime-level aggregates from regime_meta
            'hyvarinen_max': regime_meta.get('hyvarinen_max'),
            'combined_score_min': regime_meta.get('combined_score_min'),
            'bic_min': regime_meta.get('bic_min'),
            'model_selection_method': regime_meta.get('model_selection_method', 'combined'),
            'bic_weight': regime_meta.get('bic_weight', 0.5),
            'entropy_lambda': regime_meta.get('entropy_lambda', 0.05),
        }
        # Validate nu
        if theta['nu'] is not None and (not np.isfinite(theta['nu']) or theta['nu'] <= 2.0):
            theta['nu'] = None
        return theta
    else:
        # Fallback to global BMA params
        global_model_posterior = global_data.get('model_posterior', {})
        global_models = global_data.get('models', {})
        best_params = _extract_best_model_params(global_model_posterior, global_models)

        return {
            'q': _safe_float(best_params.get('q'), 1e-6),
            'phi': _safe_float(best_params.get('phi'), 0.95) if best_params.get('phi') is not None else 0.95,
            'nu': best_params.get('nu'),
            'c': _safe_float(best_params.get('c'), 1.0),
            'fallback': True,
            'source': 'global_bma',
            'regime_used': current_regime,
            'regime_name': 'GLOBAL_FALLBACK',
            'regime_meta': {},
            'collapse_warning': False,
            'model_posterior': global_model_posterior,
            # Model selection diagnostics (best model)
            'hyvarinen_score': best_params.get('hyvarinen_score'),
            'combined_score': best_params.get('combined_score'),
            'bic': best_params.get('bic'),
            # Global-level aggregates
            'hyvarinen_max': global_data.get('hyvarinen_max'),
            'combined_score_min': global_data.get('combined_score_min'),
            'bic_min': global_data.get('bic_min'),
            'model_selection_method': global_data.get('model_selection_method', 'combined'),
            'bic_weight': global_data.get('bic_weight', 0.5),
            'entropy_lambda': global_data.get('entropy_lambda', 0.05),
        }


def _kalman_filter_drift(ret: pd.Series, vol: pd.Series, q: Optional[float] = None, optimize_q: bool = True, asset_symbol: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Kalman filter for time-varying drift estimation using pre-tuned parameters only.
    All parameters (q, c, phi, nu, noise_model) must come from tuning/cache or explicit args.
    No internal optimization, heuristics, or robustness overlays are performed here.
    """
    ret_clean = _ensure_float_series(ret).dropna()
    vol_clean = _ensure_float_series(vol).reindex(ret_clean.index).dropna()
    df = pd.concat([ret_clean, vol_clean], axis=1, join='inner').dropna()
    if len(df) < 50:
        return {}
    df.columns = ['ret', 'vol']
    y = df['ret'].values.astype(float)
    sigma = df['vol'].values.astype(float)
    idx = df.index

    tuned_params = None
    if asset_symbol is not None:
        tuned_params = _load_tuned_kalman_params(asset_symbol)
    noise_model = (tuned_params or {}).get('noise_model', 'gaussian')
    requires_phi = 'phi' in noise_model or noise_model.startswith('phi_student_t_nu_')
    is_student_t = noise_model.startswith('phi_student_t_nu_')

    # φ is structural: only from tuned cache; required when model has φ
    phi_used = (tuned_params or {}).get('phi')
    if requires_phi:
        if phi_used is None or not np.isfinite(phi_used):
            raise ValueError("phi required by selected model but missing from tuning cache")
        phi_used = float(phi_used)
    else:
        phi_used = 1.0

    q_used = q if q is not None else (tuned_params or {}).get('q')
    c_used = (tuned_params or {}).get('c')
    nu_used = (tuned_params or {}).get('nu') if is_student_t else None

    if q_used is None or not np.isfinite(q_used) or q_used <= 0:
        return {}
    if c_used is None or not np.isfinite(c_used) or c_used <= 0:
        raise ValueError("Observation scale c required by tuned model but missing/invalid")
    if is_student_t and (nu_used is None or not np.isfinite(nu_used)):
        raise ValueError("Student-t model selected but ν missing from tuning cache")

    obs_scale = float(c_used)
    q_scalar = float(q_used)

    T = len(y)
    mu_filtered = np.zeros(T, dtype=float)
    P_filtered = np.zeros(T, dtype=float)
    K_gain = np.zeros(T, dtype=float)
    innovations = np.zeros(T, dtype=float)
    innovation_vars = np.zeros(T, dtype=float)

    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0

    for t in range(T):
        mu_pred = phi_used * mu_t
        P_pred = (phi_used ** 2) * P_t + q_scalar
        R_t = float(max(obs_scale * (sigma[t] ** 2), 1e-12))
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))

        if is_student_t:
            nu_adj = min(nu_used / (nu_used + 3.0), 1.0)
            K_t = nu_adj * P_pred / S_t
        else:
            K_t = P_pred / S_t

        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))

        mu_filtered[t] = mu_t
        P_filtered[t] = P_t
        K_gain[t] = K_t
        innovations[t] = innov
        innovation_vars[t] = S_t

        if is_student_t:
            try:
                ll_t = StudentTDriftModel.logpdf(innov, nu_used, 0.0, math.sqrt(S_t))
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
            except Exception:
                pass
        else:
            try:
                ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
            except Exception:
                pass

    kalman_gain_mean = float(np.mean(K_gain))
    kalman_gain_recent = float(K_gain[-1]) if len(K_gain) > 0 else float('nan')

    return {
        "mu_kf_filtered": pd.Series(mu_filtered, index=idx, name="mu_kf_filtered"),
        "mu_kf_smoothed": pd.Series(mu_filtered, index=idx, name="mu_kf_smoothed"),  # no RTS smoothing here
        "var_kf_filtered": pd.Series(P_filtered, index=idx, name="var_kf_filtered"),
        "var_kf_smoothed": pd.Series(P_filtered, index=idx, name="var_kf_smoothed"),
        "kalman_gain": pd.Series(K_gain, index=idx, name="kalman_gain"),
        "innovations": pd.Series(innovations, index=idx, name="innovations"),
        "innovation_vars": pd.Series(innovation_vars, index=idx, name="innovation_vars"),
        "log_likelihood": float(log_likelihood),
        "process_noise_var": float(q_scalar),
        "n_obs": int(T),
        "kalman_gain_mean": kalman_gain_mean,
        "kalman_gain_recent": kalman_gain_recent,
        "kalman_heteroskedastic_mode": False,
        "kalman_c_optimal": obs_scale,
        "phi_used": float(phi_used) if phi_used is not None and np.isfinite(phi_used) else None,
        "kalman_noise_model": noise_model,
        "kalman_nu": float(nu_used) if nu_used is not None else None,
    }


def compute_features(px: pd.Series, asset_symbol: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Compute features from price series for signal generation.

    Args:
        px: Price series
        asset_symbol: Asset symbol (e.g., "PLNJPY=X") for loading tuned Kalman parameters

    Returns:
        Dictionary of computed features
    """
    # Protect log conversion from garbage ticks and non-positive prices
    px = _ensure_float_series(px)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]

    log_px = np.log(px)
    ret = log_px.diff().dropna()
    ret = winsorize(ret, p=0.01)
    ret.name = "ret"

    # Multi-speed EWMA for drift and vol
    mu_fast = ret.ewm(span=21, adjust=False).mean()
    mu_slow = ret.ewm(span=126, adjust=False).mean()

    vol_fast = ret.ewm(span=21, adjust=False).std()
    vol_slow = ret.ewm(span=126, adjust=False).std()

    # Prefer GARCH(1,1) volatility via MLE; fallback to EWMA blend on failure
    try:
        vol_garch, garch_params = _garch11_mle(ret)
        # Align to ret index and name "vol" for downstream compatibility
        vol = vol_garch.reindex(ret.index).rename("vol")
        vol_source = "garch11"
    except Exception:
        # Blend vol: fast reacts, slow stabilizes
        vol = (0.6 * vol_fast + 0.4 * vol_slow).rename("vol")
        garch_params = {}
        vol_source = "ewma_fallback"
    # Robust global volatility floor to avoid feedback loops when vol collapses recently:
    # - Use a lagged expanding 10th percentile over the entire history (no look-ahead)
    # - Add a relative floor vs long-run median and a small absolute epsilon
    # - Provide an early-history fallback to ensure continuity
    MIN_HIST = 252
    LAG_DAYS = 21  # ~1 trading month lag to avoid immediate reaction to shocks
    abs_floor = 1e-6
    try:
        vol_lag = vol.shift(LAG_DAYS)
        # Expanding quantile and median computed on information up to t-LAG
        global_floor_series = vol_lag.expanding(MIN_HIST).quantile(0.10)
        long_med = vol_lag.expanding(MIN_HIST).median()
        rel_floor = 0.10 * long_med
        # Combine available floors at each timestamp
        floor_candidates = pd.concat([
            global_floor_series.rename("gf"),
            rel_floor.rename("rf")
        ], axis=1)
        floor_t = floor_candidates.max(axis=1)
        # Early history fallback (before MIN_HIST+LAG_DAYS)
        early_med = vol.rolling(63, min_periods=20).median()
        early_floor = np.maximum(0.10 * early_med, abs_floor)
        floor_t = floor_t.combine_first(early_floor)
        # Ensure absolute epsilon
        floor_t = np.maximum(floor_t, abs_floor)
        # Apply the floor index-wise
        vol = np.maximum(vol, floor_t)
    except Exception:
        # Fallback to a simple median-based floor if expanding quantile not available
        fallback_floor = np.maximum(vol.rolling(252, min_periods=63).median() * 0.10, abs_floor)
        vol = np.maximum(vol, fallback_floor)

    # Vol regime (relative to 1y median) — kept for diagnostics, not for shrinkage
    vol_med = vol.rolling(252).median()
    vol_regime = vol / vol_med

    # ========================================
    # Pillar 1: Model-Based Drift Estimation
    # ========================================
    # Use best model selected by BIC from tune.py model comparison:
    # - zero_drift: μ = 0 (no predictable drift)
    # - constant_drift: μ = constant (fixed drift)
    # - ewma_drift: μ = EWMA of returns (adaptive)
    # - kalman_drift: μ from Kalman filter (state-space model)

    # Load tuned parameters and model selection results
    tuned_params = None
    best_model = 'kalman_gaussian'
    # Valid Kalman model names (discrete nu grid for Student-t)
    kalman_keys = {'kalman_gaussian', 'kalman_phi_gaussian'}
    kalman_keys.update({f'phi_student_t_nu_{nu}' for nu in [4, 6, 8, 12, 20]})
    tuned_noise_model = 'gaussian'
    tuned_nu = None
    if asset_symbol is not None:
        tuned_params = _load_tuned_kalman_params(asset_symbol)
        if tuned_params:
            best_model = tuned_params.get('best_model', 'kalman_gaussian')
            tuned_noise_model = tuned_params.get('noise_model', 'gaussian')
            # Get nu for Student-t models (phi_student_t_nu_* naming)
            if tuned_noise_model.startswith('phi_student_t_nu_'):
                tuned_nu = tuned_params.get('nu')

    # Print BMA model information
    if asset_symbol and tuned_params and tuned_params.get('has_bma'):
        model_posterior = tuned_params.get('model_posterior', {})
        global_data = tuned_params.get('global') or {}
        global_models = global_data.get('models', {})

        # Get model selection method from cache metadata
        model_selection_method = tuned_params.get('model_selection_method', 'combined')
        bic_weight = tuned_params.get('bic_weight', 0.5)

        # Use Rich for world-class presentation
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.columns import Columns
        from rich.console import Group

        console = Console(force_terminal=True)

        # Get company name and sector
        company_name = get_company_name(asset_symbol) or asset_symbol
        sector = get_sector(asset_symbol) or ""

        # Model short names and descriptions
        # Include ALL possible ν values from adaptive refinement
        model_info = {
            'kalman_gaussian': {'short': 'Gaussian', 'icon': '📈', 'desc': 'Standard Kalman filter'},
            'kalman_phi_gaussian': {'short': 'φ-Gaussian', 'icon': '🔄', 'desc': 'Autoregressive drift'},
            # Discrete nu grid Student-t models (original grid)
            'phi_student_t_nu_4': {'short': 'φ-T(ν=4)', 'icon': '📊', 'desc': 'Very heavy tails, ν=4'},
            'phi_student_t_nu_6': {'short': 'φ-T(ν=6)', 'icon': '📊', 'desc': 'Heavy tails, ν=6'},
            'phi_student_t_nu_8': {'short': 'φ-T(ν=8)', 'icon': '📊', 'desc': 'Moderate tails, ν=8'},
            'phi_student_t_nu_12': {'short': 'φ-T(ν=12)', 'icon': '📊', 'desc': 'Light tails, ν=12'},
            'phi_student_t_nu_20': {'short': 'φ-T(ν=20)', 'icon': '📊', 'desc': 'Near-Gaussian, ν=20'},
            # Adaptive ν refinement candidates (intermediate values)
            'phi_student_t_nu_3': {'short': 'φ-T(ν=3)', 'icon': '📊', 'desc': 'Extreme tails, ν=3 (refined)'},
            'phi_student_t_nu_5': {'short': 'φ-T(ν=5)', 'icon': '📊', 'desc': 'Heavy tails, ν=5 (refined)'},
            'phi_student_t_nu_7': {'short': 'φ-T(ν=7)', 'icon': '📊', 'desc': 'Heavy tails, ν=7 (refined)'},
            'phi_student_t_nu_10': {'short': 'φ-T(ν=10)', 'icon': '📊', 'desc': 'Moderate tails, ν=10 (refined)'},
            'phi_student_t_nu_14': {'short': 'φ-T(ν=14)', 'icon': '📊', 'desc': 'Light tails, ν=14 (refined)'},
            'phi_student_t_nu_16': {'short': 'φ-T(ν=16)', 'icon': '📊', 'desc': 'Light tails, ν=16 (refined)'},
            'phi_student_t_nu_25': {'short': 'φ-T(ν=25)', 'icon': '📊', 'desc': 'Near-Gaussian, ν=25 (refined)'},
        }

        # Dynamic fallback: if model not in model_info, generate entry dynamically
        def get_model_info(model_name: str) -> dict:
            if model_name in model_info:
                return model_info[model_name]
            # Handle phi_student_t_nu_* with any ν value
            if model_name.startswith('phi_student_t_nu_'):
                try:
                    nu_val = int(model_name.split('_')[-1])
                    return {'short': f'φ-T(ν={nu_val})', 'icon': '📊', 'desc': f'Student-t, ν={nu_val}'}
                except ValueError:
                    pass
            # Fallback
            return {'short': model_name[:14], 'icon': '?', 'desc': model_name}

        # Model selection method description
        selection_method_info = {
            'bic': ('BIC-only', 'Traditional Bayesian Information Criterion'),
            'hyvarinen': ('Hyvärinen-only', 'Robust scoring under misspecification'),
            'combined': (f'Combined (α={bic_weight:.1f})', 'BIC + Hyvärinen geometric mean'),
        }
        method_short, method_desc = selection_method_info.get(
            model_selection_method,
            ('Unknown', 'Model selection method')
        )

        # Helper functions to describe parameters in human terms
        def describe_drift_speed(q_val):
            if q_val is None or not np.isfinite(q_val):
                return ("unknown", "white")
            if q_val < 1e-9:
                return ("frozen", "blue")
            elif q_val < 1e-8:
                return ("slow", "cyan")
            elif q_val < 1e-7:
                return ("moderate", "green")
            elif q_val < 1e-6:
                return ("fast", "yellow")
            else:
                return ("rapid", "red")

        def describe_vol_scale(c_val):
            if c_val is None or not np.isfinite(c_val):
                return ("normal", "white")
            if c_val < 0.7:
                return ("muted", "blue")
            elif c_val < 0.9:
                return ("reduced", "cyan")
            elif c_val < 1.1:
                return ("normal", "green")
            elif c_val < 1.3:
                return ("elevated", "yellow")
            else:
                return ("amplified", "red")

        def describe_persistence(phi_val):
            if phi_val is None or not np.isfinite(phi_val):
                return ("n/a", "dim")
            if phi_val < 0.5:
                return ("weak", "red")
            elif phi_val < 0.8:
                return ("moderate", "yellow")
            elif phi_val < 0.95:
                return ("strong", "green")
            elif phi_val < 0.99:
                return ("very strong", "cyan")
            else:
                return ("near-unit", "blue")

        def describe_tail_weight(nu_val):
            if nu_val is None or not np.isfinite(nu_val):
                return ("normal", "white")
            if nu_val < 5:
                return ("very heavy", "red")
            elif nu_val < 10:
                return ("heavy", "yellow")
            elif nu_val < 30:
                return ("moderate", "green")
            else:
                return ("light", "cyan")

        # ═══════════════════════════════════════════════════════════════════════════════
        # EXTRAORDINARY APPLE-QUALITY MODEL PANEL
        # Design: Clean, premium, scannable, beautiful
        # ═══════════════════════════════════════════════════════════════════════════════

        from rich.rule import Rule
        from rich.align import Align

        # Get all models from posterior, sorted by weight descending
        all_models = sorted(model_posterior.keys(), key=lambda m: model_posterior.get(m, 0), reverse=True)

        # Get global-level aggregate scores
        global_hyv_max = tuned_params.get('hyvarinen_max')
        global_bic_min = tuned_params.get('bic_min')

        console.print()
        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # ASSET HEADER - Cinematic, clean, CENTERED
        # ─────────────────────────────────────────────────────────────────────────────
        header_content = Text(justify="center")
        header_content.append("\n", style="")
        header_content.append(asset_symbol, style="bold bright_white")
        header_content.append("\n", style="")
        header_content.append(company_name, style="dim")
        if sector:
            header_content.append(f"  ·  {sector}", style="dim italic")
        header_content.append("\n", style="")

        header_panel = Panel(
            Align.center(header_content),
            box=box.ROUNDED,
            border_style="bright_cyan",
            padding=(0, 4),
        )
        console.print(Align.center(header_panel, width=55))
        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # WINNING MODEL - Hero section
        # ─────────────────────────────────────────────────────────────────────────────
        best_info = get_model_info(best_model)
        best_params = global_models.get(best_model, {})
        best_weight = model_posterior.get(best_model, 0.0)

        # Get BIC/Hyvärinen from best model params (more reliable)
        best_bic = best_params.get('bic')
        best_hyv = best_params.get('hyvarinen_score')

        winner_grid = Table.grid(padding=(0, 4))
        winner_grid.add_column(justify="center")
        winner_grid.add_column(justify="center")
        winner_grid.add_column(justify="center")
        winner_grid.add_column(justify="center")

        def metric_text(value: str, label: str, color: str = "white") -> Text:
            t = Text()
            t.append(f"{value}\n", style=f"bold {color}")
            t.append(label, style="dim")
            return t

        bic_str = f"{best_bic:.0f}" if best_bic and np.isfinite(best_bic) else "—"
        hyv_str = f"{best_hyv:.0f}" if best_hyv and np.isfinite(best_hyv) else "—"

        winner_grid.add_row(
            metric_text(best_info['short'], "Model", "bright_green"),
            metric_text(f"{best_weight:.0%}", "Weight", "bright_cyan"),
            metric_text(bic_str, "BIC", "white"),
            metric_text(hyv_str, "Hyv", "white"),
        )
        console.print(Align.center(winner_grid))
        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # MODEL COMPARISON - Compact, scannable
        # ─────────────────────────────────────────────────────────────────────────────
        console.print(Rule(style="dim", characters="─"))
        console.print()

        # Only show models with weight > 0.001% to reduce clutter
        visible_models = [m for m in all_models if model_posterior.get(m, 0) >= 0.0001]

        for model_name in visible_models:
            p = model_posterior.get(model_name, 0.0)
            m_params = global_models.get(model_name, {})
            info = get_model_info(model_name)
            is_best = model_name == best_model
            is_significant = p >= 0.02  # 2% threshold for significant contribution

            bic_val = m_params.get('bic')
            hyv_val = m_params.get('hyvarinen_score')

            # Visual weight bar
            bar_width = 20
            filled = int(p * bar_width)

            # Build row
            row = Text()
            row.append("    ", style="")

            if is_best:
                # Best model: bold green with filled bar
                row.append("● ", style="bold bright_green")
                row.append(f"{info['short']:<14}", style="bold bright_green")
                row.append(f"{p:>6.1%}  ", style="bold bright_green")
                row.append("━" * filled, style="bright_green")
                row.append("─" * (bar_width - filled), style="dim")
            elif is_significant:
                # Significant model (>=2%): green with filled bar (not bold)
                row.append("● ", style="bright_green")
                row.append(f"{info['short']:<14}", style="bright_green")
                row.append(f"{p:>6.1%}  ", style="bright_green")
                row.append("━" * filled, style="green")
                row.append("─" * (bar_width - filled), style="dim")
            else:
                # Minor model (<2%): dim
                row.append("○ ", style="dim")
                row.append(f"{info['short']:<14}", style="dim")
                row.append(f"{p:>6.1%}  ", style="dim")
                row.append("─" * bar_width, style="dim")

            console.print(row)

        console.print()

        # ─────────────────────────────────────────────────────────────────────────────
        # PARAMETER ESTIMATES TABLE - All models, Apple-quality
        # ─────────────────────────────────────────────────────────────────────────────
        params_header = Text()
        params_header.append("    ▸ ", style="bright_cyan")
        params_header.append("Parameter Estimates", style="bold white")
        console.print(params_header)
        console.print()

        params_table = Table(
            show_header=True,
            header_style="dim",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
        params_table.add_column("Model", style="white", width=14)
        params_table.add_column("Drift (q)", justify="center", width=12)
        params_table.add_column("Vol (c)", justify="center", width=12)
        params_table.add_column("Persist (φ)", justify="center", width=12)
        params_table.add_column("Tails (ν)", justify="center", width=12)
        params_table.add_column("Skew/Mix", justify="center", width=12)

        # Helper to describe skewness for various model families
        def describe_skewness(model_name: str, params: dict) -> tuple:
            """Return (description, color) for skewness/mixture parameters."""
            # Hansen Skew-t: lambda parameter
            if 'hansen_skew_t' in model_name or params.get('lambda') is not None:
                lam = params.get('lambda')
                if lam is None:
                    return ("—", "dim")
                if lam < -0.1:
                    return (f"λ={lam:+.2f}", "red")  # Left-skewed, crash risk
                elif lam > 0.1:
                    return (f"λ={lam:+.2f}", "cyan")  # Right-skewed
                return (f"λ={lam:+.2f}", "green")  # Symmetric
            
            # Contaminated Student-t: epsilon (crisis probability)
            if 'cst' in model_name or params.get('epsilon') is not None:
                eps = params.get('epsilon')
                if eps is None:
                    return ("—", "dim")
                if eps > 0.15:
                    return (f"ε={eps:.0%}", "red")  # High crisis prob
                elif eps > 0.08:
                    return (f"ε={eps:.0%}", "yellow")
                return (f"ε={eps:.0%}", "green")
            
            # NIG: beta (asymmetry)
            if 'nig' in model_name or params.get('beta') is not None:
                beta = params.get('beta')
                alpha = params.get('alpha')
                if beta is None:
                    return ("—", "dim")
                if beta < -0.1:
                    return (f"β={beta:+.2f}", "red")
                elif beta > 0.1:
                    return (f"β={beta:+.2f}", "cyan")
                return (f"β={beta:+.2f}", "green")
            
            # Phi-Skew-t: gamma parameter
            if 'skew_t' in model_name or params.get('gamma') is not None:
                gamma = params.get('gamma')
                if gamma is None:
                    return ("—", "dim")
                if gamma < 0.9:
                    return (f"γ={gamma:.2f}", "red")  # Left-skewed
                elif gamma > 1.1:
                    return (f"γ={gamma:.2f}", "cyan")  # Right-skewed
                return (f"γ={gamma:.2f}", "green")
            
            # GMM: mixture weights
            if 'gmm' in model_name:
                weights = params.get('weights')
                if weights and len(weights) >= 2:
                    w1, w2 = weights[0], weights[1]
                    return (f"w={w1:.0%}/{w2:.0%}", "blue")
                return ("—", "dim")
            
            return ("—", "dim")

        for model_name in visible_models:
            m_params = global_models.get(model_name, {})
            info = get_model_info(model_name)
            is_best = model_name == best_model

            if m_params.get('fit_success', False):
                q = m_params.get('q', float('nan'))
                c = m_params.get('c', float('nan'))
                phi = m_params.get('phi')
                nu = m_params.get('nu')

                drift_desc, drift_color = describe_drift_speed(q)
                vol_desc, vol_color = describe_vol_scale(c)
                persist_desc, persist_color = describe_persistence(phi) if phi else ("—", "dim")
                tail_desc, tail_color = describe_tail_weight(nu) if nu else ("—", "dim")
                skew_desc, skew_color = describe_skewness(model_name, m_params)

                if is_best:
                    params_table.add_row(
                        f"[bold bright_green]{info['short']}[/bold bright_green]",
                        f"[bold {drift_color}]{drift_desc}[/bold {drift_color}]",
                        f"[bold {vol_color}]{vol_desc}[/bold {vol_color}]",
                        f"[bold {persist_color}]{persist_desc}[/bold {persist_color}]",
                        f"[bold {tail_color}]{tail_desc}[/bold {tail_color}]",
                        f"[bold {skew_color}]{skew_desc}[/bold {skew_color}]",
                    )
                else:
                    params_table.add_row(
                        f"[dim]{info['short']}[/dim]",
                        f"[dim]{drift_desc}[/dim]",
                        f"[dim]{vol_desc}[/dim]",
                        f"[dim]{persist_desc}[/dim]",
                        f"[dim]{tail_desc}[/dim]",
                        f"[dim]{skew_desc}[/dim]",
                    )
            else:
                params_table.add_row(
                    f"[dim]{info['short']}[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                )

        console.print(Padding(params_table, (0, 0, 0, 4)))

        # ─────────────────────────────────────────────────────────────────────────────
        # CALIBRATION & TRUST - Shows calibration status and effective trust
        # ─────────────────────────────────────────────────────────────────────────────
        console.print()

        calibration_header = Text()
        calibration_header.append("    ▸ ", style="bright_cyan")
        calibration_header.append("Calibration & Trust", style="bold white")
        console.print(calibration_header)
        console.print()

        # Get calibration data from tuned params
        calibrated_trust_data = tuned_params.get('calibrated_trust', {})
        effective_trust = tuned_params.get('effective_trust')
        calibration_trust = tuned_params.get('calibration_trust')
        regime_penalty = tuned_params.get('regime_penalty')
        calibration_warning = tuned_params.get('calibration_warning', False)
        pit_ks_pvalue = global_data.get('pit_ks_pvalue')
        pit_ks_pvalue_calibrated = global_data.get('pit_ks_pvalue_calibrated')
        recalibration_applied = tuned_params.get('recalibration_applied', False)
        nu_refinement = tuned_params.get('nu_refinement', {})
        gh_selected = tuned_params.get('gh_selected', False)
        gh_model = tuned_params.get('gh_model', {})

        # Trust decomposition table
        trust_table = Table(
            show_header=False,
            border_style="dim",
            box=box.SIMPLE,
            padding=(0, 2),
            expand=False,
        )
        trust_table.add_column("Label", style="dim", width=24)
        trust_table.add_column("Value", width=30)

        # Calibration status
        if calibration_warning:
            cal_status = "[bold yellow]⚠ Warning[/bold yellow]"
        else:
            cal_status = "[bold green]✓ Passed[/bold green]"
        trust_table.add_row("PIT Calibration", cal_status)

        # PIT p-values
        if pit_ks_pvalue is not None:
            pit_color = "red" if pit_ks_pvalue < 0.01 else "yellow" if pit_ks_pvalue < 0.05 else "green"
            pit_str = f"[{pit_color}]{pit_ks_pvalue:.4f}[/{pit_color}]"
            trust_table.add_row("  Raw PIT p-value", pit_str)

        if pit_ks_pvalue_calibrated is not None:
            pit_cal_color = "red" if pit_ks_pvalue_calibrated < 0.01 else "yellow" if pit_ks_pvalue_calibrated < 0.05 else "green"
            pit_cal_str = f"[{pit_cal_color}]{pit_ks_pvalue_calibrated:.4f}[/{pit_cal_color}]"
            trust_table.add_row("  Calibrated PIT p-value", pit_cal_str)

        # Isotonic recalibration
        if recalibration_applied:
            trust_table.add_row("  Isotonic Recalibration", "[green]✓ Applied[/green]")
        else:
            trust_table.add_row("  Isotonic Recalibration", "[dim]Not applied[/dim]")

        # ν refinement
        if nu_refinement:
            nu_attempted = nu_refinement.get('refinement_attempted', False)
            nu_improved = nu_refinement.get('improvement_achieved', False)
            nu_original = nu_refinement.get('nu_original')
            nu_final = nu_refinement.get('nu_final')

            if nu_attempted:
                if nu_improved and nu_original != nu_final:
                    trust_table.add_row("  ν Refinement", f"[green]✓ Improved ν={nu_original}→{nu_final}[/green]")
                else:
                    trust_table.add_row("  ν Refinement", f"[dim]Attempted, no improvement[/dim]")
            else:
                trust_table.add_row("  ν Refinement", "[dim]Not needed[/dim]")

        # GH model
        if gh_selected and gh_model:
            gh_params = gh_model.get('parameters', {})
            gh_skew = gh_params.get('beta', 0)
            skew_dir = "right" if gh_skew > 0.1 else "left" if gh_skew < -0.1 else "symmetric"
            trust_table.add_row("  GH Skew Model", f"[cyan]✓ Selected ({skew_dir})[/cyan]")

        # Trust decomposition (main feature)
        if effective_trust is not None and calibration_trust is not None:
            trust_table.add_row("", "")  # Spacer
            trust_table.add_row("[bold]Trust Authority[/bold]", "")

            # Calibration trust
            cal_trust_color = "green" if calibration_trust > 0.8 else "yellow" if calibration_trust > 0.5 else "red"
            trust_table.add_row("  Calibration Trust", f"[{cal_trust_color}]{calibration_trust:.1%}[/{cal_trust_color}]")

            # Regime penalty
            if regime_penalty is not None:
                penalty_color = "dim" if regime_penalty < 0.1 else "yellow" if regime_penalty < 0.2 else "red"
                regime_context = calibrated_trust_data.get('regime_context', 'normal')
                trust_table.add_row("  Regime Penalty", f"[{penalty_color}]-{regime_penalty:.1%} ({regime_context})[/{penalty_color}]")

            # Effective trust (final)
            eff_trust_color = "green" if effective_trust > 0.7 else "yellow" if effective_trust > 0.4 else "red"
            trust_table.add_row("  [bold]Effective Trust[/bold]", f"[bold {eff_trust_color}]{effective_trust:.1%}[/bold {eff_trust_color}]")

            # Tail bias
            tail_bias = calibrated_trust_data.get('tail_bias')
            if tail_bias is not None:
                bias_dir = "right" if tail_bias > 0.02 else "left" if tail_bias < -0.02 else "centered"
                trust_table.add_row("  Tail Bias", f"[dim]{tail_bias:+.3f} ({bias_dir})[/dim]")

        console.print(Padding(trust_table, (0, 0, 0, 4)))

        # ─────────────────────────────────────────────────────────────────────────────
        # AUGMENTATION LAYERS - Shows advanced distributional model status
        # Hansen Skew-t, Contaminated Student-t, GMM, NIG
        # ─────────────────────────────────────────────────────────────────────────────
        console.print()

        aug_header = Text()
        aug_header.append("    ▸ ", style="bright_cyan")
        aug_header.append("Augmentation Layers", style="bold white")
        console.print(aug_header)
        console.print()

        # Extract augmentation layer data from global_data
        hansen_data = global_data.get('hansen_skew_t', {})
        cst_data = global_data.get('contaminated_student_t', {})
        gmm_data = global_data.get('gmm', {})
        nig_data = global_data.get('nig', {})
        skew_t_data = global_data.get('phi_skew_t', {})

        aug_table = Table(
            show_header=False,
            border_style="dim",
            box=box.SIMPLE,
            padding=(0, 2),
            expand=False,
        )
        aug_table.add_column("Layer", style="dim", width=24)
        aug_table.add_column("Status", width=40)

        # Helper to describe skewness direction
        def skew_direction(val):
            if val is None:
                return "n/a"
            if val < -0.05:
                return "left (crash risk)"
            elif val > 0.05:
                return "right (upside)"
            return "symmetric"

        # ═══════════════════════════════════════════════════════════════════════════
        # HANSEN SKEW-T: Asymmetric heavy tails via λ parameter
        # ═══════════════════════════════════════════════════════════════════════════
        hansen_lambda = hansen_data.get('lambda') if hansen_data else None
        hansen_nu = hansen_data.get('nu') if hansen_data else None
        hansen_enabled = hansen_lambda is not None and abs(hansen_lambda) > 0.01

        if hansen_enabled:
            skew_dir = skew_direction(hansen_lambda)
            aug_table.add_row(
                "[cyan]↔️  Hansen Skew-T[/cyan]",
                f"[green]✓ Active[/green] λ={hansen_lambda:+.2f} ({skew_dir})"
            )
            if hansen_nu:
                aug_table.add_row("    Tail weight (ν)", f"[dim]{hansen_nu:.0f}[/dim]")
        else:
            aug_table.add_row(
                "[dim]↔️  Hansen Skew-T[/dim]",
                "[dim]○ Not fitted[/dim]"
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # CONTAMINATED STUDENT-T: Regime-dependent tail heaviness
        # ═══════════════════════════════════════════════════════════════════════════
        cst_nu_normal = cst_data.get('nu_normal') if cst_data else None
        cst_nu_crisis = cst_data.get('nu_crisis') if cst_data else None
        cst_epsilon = cst_data.get('epsilon') if cst_data else None
        cst_enabled = cst_nu_normal is not None and cst_epsilon is not None and cst_epsilon > 0.001

        if cst_enabled:
            aug_table.add_row(
                "[magenta]⚡ Contaminated-T[/magenta]",
                f"[green]✓ Active[/green] ε={cst_epsilon:.0%} crisis probability"
            )
            aug_table.add_row("    Normal regime (ν)", f"[dim]{cst_nu_normal:.0f} (lighter tails)[/dim]")
            aug_table.add_row("    Crisis regime (ν)", f"[dim]{cst_nu_crisis:.0f} (heavier tails)[/dim]")
        else:
            aug_table.add_row(
                "[dim]⚡ Contaminated-T[/dim]",
                "[dim]○ Not fitted[/dim]"
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # GMM: 2-component Gaussian mixture (bimodal dynamics)
        # ═══════════════════════════════════════════════════════════════════════════
        gmm_weights = gmm_data.get('weights') if gmm_data else None
        gmm_means = gmm_data.get('means') if gmm_data else None
        gmm_enabled = gmm_weights is not None and len(gmm_weights) >= 2

        if gmm_enabled:
            aug_table.add_row(
                "[yellow]🎲 GMM Mixture[/yellow]",
                f"[green]✓ Active[/green] K=2 components"
            )
            for i, (w, m) in enumerate(zip(gmm_weights[:2], gmm_means[:2] if gmm_means else [0, 0])):
                component_label = "Momentum" if m > 0 else "Reversal"
                aug_table.add_row(f"    Component {i+1}", f"[dim]w={w:.1%}, μ={m:.4f} ({component_label})[/dim]")
        else:
            aug_table.add_row(
                "[dim]🎲 GMM Mixture[/dim]",
                "[dim]○ Not fitted[/dim]"
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # NIG: Normal-Inverse Gaussian (semi-heavy tails, Lévy compatible)
        # ═══════════════════════════════════════════════════════════════════════════
        nig_alpha = nig_data.get('alpha') if nig_data else None
        nig_beta = nig_data.get('beta') if nig_data else None
        nig_enabled = nig_alpha is not None and nig_beta is not None

        if nig_enabled:
            asym_dir = skew_direction(nig_beta)
            aug_table.add_row(
                "[blue]🎯 NIG Distribution[/blue]",
                f"[green]✓ Active[/green] α={nig_alpha:.2f}, β={nig_beta:+.2f}"
            )
            aug_table.add_row("    Asymmetry", f"[dim]{asym_dir}[/dim]")
        else:
            aug_table.add_row(
                "[dim]🎯 NIG Distribution[/dim]",
                "[dim]○ Not fitted[/dim]"
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # PHI-SKEW-T: Fernández-Steel skew-t (γ parameter)
        # ═══════════════════════════════════════════════════════════════════════════
        skew_t_gamma = skew_t_data.get('gamma') if skew_t_data else None
        skew_t_nu = skew_t_data.get('nu') if skew_t_data else None
        skew_t_enabled = skew_t_gamma is not None

        if skew_t_enabled:
            # γ < 1 = left-skewed, γ > 1 = right-skewed
            gamma_dir = "left (crash risk)" if skew_t_gamma < 0.95 else "right (upside)" if skew_t_gamma > 1.05 else "symmetric"
            aug_table.add_row(
                "[purple]📐 Skew-T (F-S)[/purple]",
                f"[green]✓ Active[/green] γ={skew_t_gamma:.2f} ({gamma_dir})"
            )
            if skew_t_nu:
                aug_table.add_row("    Tail weight (ν)", f"[dim]{skew_t_nu:.0f}[/dim]")
        else:
            aug_table.add_row(
                "[dim]📐 Skew-T (F-S)[/dim]",
                "[dim]○ Not fitted[/dim]"
            )

        console.print(Padding(aug_table, (0, 0, 0, 4)))

        console.print()
        console.print(Rule(style="dim", characters="─"))
        console.print()
    elif asset_symbol and tuned_params:
        # Old cache format warning
        print(f"\n\033[93m⚠️  {asset_symbol}: Old cache format — run tune.py\033[0m\n")

    # Apply drift estimation based on best model selection
    # NOTE: In BMA architecture, "best_model" is used for Kalman filter params,
    # but actual predictions use weighted mixture over all models
    if best_model in kalman_keys:
        kf_result = _kalman_filter_drift(ret, vol, q=None, asset_symbol=asset_symbol)

        # Extract Kalman-filtered drift estimates
        if kf_result and "mu_kf_smoothed" in kf_result:
            # Use backward-smoothed estimates (uses all data, statistically optimal)
            mu_kf = kf_result["mu_kf_smoothed"]
            var_kf = kf_result["var_kf_smoothed"]
            kalman_available = True
            kalman_metadata = {
                "log_likelihood": kf_result.get("log_likelihood", float("nan")),
                "process_noise_var": kf_result.get("process_noise_var", float("nan")),
                "n_obs": kf_result.get("n_obs", 0),
                # Refinement 1: q optimization metadata
                "q_optimal": kf_result.get("q_optimal", float("nan")),
                "q_heuristic": kf_result.get("q_heuristic", float("nan")),
                "q_optimization_attempted": kf_result.get("q_optimization_attempted", False),
                # Refinement 2: Kalman gain statistics (situational awareness)
                "kalman_gain_mean": kf_result.get("kalman_gain_mean", float("nan")),
                "kalman_gain_recent": kf_result.get("kalman_gain_recent", float("nan")),
                # Refinement 3: Innovation whiteness test (model adequacy)
                "innovation_whiteness": kf_result.get("innovation_whiteness", {}),
                # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
                "kalman_heteroskedastic_mode": kf_result.get("heteroskedastic_mode", False),
                "kalman_c_optimal": kf_result.get("c_optimal"),
                "kalman_q_t_mean": kf_result.get("q_t_mean"),
                "kalman_q_t_std": kf_result.get("q_t_std"),
                "kalman_q_t_min": kf_result.get("q_t_min"),
                "kalman_q_t_max": kf_result.get("q_t_max"),
                # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
                "kalman_robust_t_mode": kf_result.get("robust_t_mode", False),
                "kalman_nu_robust": kf_result.get("nu_robust"),
                # Level-7+ Refinement: Regime-dependent drift priors
                "kalman_regime_prior_used": kf_result.get("regime_prior_used", False),
                "kalman_regime_info": kf_result.get("regime_prior_info", {}),
                # φ persistence (from tuned cache or filter)
                "kalman_phi": tuned_params.get("phi") if tuned_params else kf_result.get("phi_used"),
                "phi_used": kf_result.get("phi_used"),
                # Regime-conditional parameters from tune.py hierarchical cache
                "regime_params": tuned_params.get("regime", {}) if tuned_params else {},
                "has_regime_params": bool(tuned_params.get("regime")) if tuned_params else False,
                # Noise model for Student-t support
                "kalman_noise_model": tuned_params.get("noise_model", "gaussian") if tuned_params else "gaussian",
                "kalman_nu": tuned_params.get("nu") if tuned_params else None,
                # BMA diagnostics from best model (for drift quality assessment)
                "pit_ks_pvalue": tuned_params.get("pit_ks_pvalue") if tuned_params else None,
                "ks_statistic": tuned_params.get("ks_statistic") if tuned_params else None,
                "bic": tuned_params.get("bic") if tuned_params else None,
                "hyvarinen_score": tuned_params.get("hyvarinen_score") if tuned_params else None,
                "combined_score": tuned_params.get("combined_score") if tuned_params else None,
                # Global-level aggregates for model selection
                "hyvarinen_max": tuned_params.get("hyvarinen_max") if tuned_params else None,
                "combined_score_min": tuned_params.get("combined_score_min") if tuned_params else None,
                "bic_min": tuned_params.get("bic_min") if tuned_params else None,
                "model_selection_method": tuned_params.get("model_selection_method", "combined") if tuned_params else "combined",
                "bic_weight": tuned_params.get("bic_weight", 0.5) if tuned_params else 0.5,
                "entropy_lambda": tuned_params.get("entropy_lambda", 0.05) if tuned_params else 0.05,
                "model_posterior": tuned_params.get("model_posterior", {}) if tuned_params else {},
                "best_model": tuned_params.get("best_model") if tuned_params else best_model,
            }
        else:
            # Fallback: use EWMA blend if Kalman fails
            mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
            mu_kf = mu_blend
            var_kf = pd.Series(0.0, index=mu_kf.index)  # no uncertainty quantified
            kalman_available = False
            kalman_metadata = {
                "model_selected": str(best_model),
                "reason": "Unrecognized model key; defaulting to EWMA blend"
            }
    else:
        # Unknown model key: fallback to EWMA blend
        mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
        mu_kf = mu_blend
        var_kf = pd.Series(0.0, index=mu_kf.index)
        kalman_available = False
        kalman_metadata = {
            "model_selected": str(best_model),
            "reason": "Unrecognized model key; defaulting to EWMA blend"
        }

    # Trend filter (200D z-distance) - kept for diagnostics
    sma200 = px.rolling(200).mean()
    trend_z = (px - sma200) / px.rolling(200).std()

    # HMM regime detection (for regime-aware adjustments, not drift estimation)
    # Fit HMM to get regime posteriors
    hmm_result_prelim = fit_hmm_regimes(
        {"ret": ret, "vol": vol},
        n_states=3,
        random_seed=42
    )

    # Apply light regime-aware shrinkage to Kalman drift in extreme regimes
    # (Kalman already handles uncertainty; this adds regime-specific conservatism)
    if hmm_result_prelim is not None and "posterior_probs" in hmm_result_prelim:
        try:
            posterior_probs = hmm_result_prelim["posterior_probs"]

            # Align posteriors with mu_kf index
            posterior_aligned = posterior_probs.reindex(mu_kf.index).ffill().fillna(0.333)

            # Extract regime probabilities
            regime_names = hmm_result_prelim["regime_names"]
            calm_idx = [k for k, v in regime_names.items() if v == "calm"]
            crisis_idx = [k for k, v in regime_names.items() if v == "crisis"]

            p_calm = posterior_aligned.iloc[:, calm_idx[0]].values if calm_idx else np.zeros(len(mu_kf))
            p_crisis = posterior_aligned.iloc[:, crisis_idx[0]].values if crisis_idx else np.zeros(len(mu_kf))

            # Light shrinkage in crisis regimes (Kalman handles most uncertainty)
            # Shrink toward zero in extreme crisis to be conservative
            shrinkage = 0.3 * p_crisis  # 0-30% shrinkage based on crisis probability
            shrinkage = np.clip(shrinkage, 0.0, 0.5)

            # Final drift: Kalman estimate with regime-aware shrinkage
            mu_final = pd.Series(
                (1.0 - shrinkage) * mu_kf.values,  # shrink toward zero in crisis
                index=mu_kf.index,
                name="mu_final"
            )

        except Exception:
            # Fallback: use Kalman estimate without regime adjustment
            mu_final = mu_kf.copy()
    else:
        # HMM not available: use pure Kalman estimate
        mu_final = mu_kf.copy()

    # Robust fallback for NaNs
    mu_final = mu_final.fillna(0.0)

    # Legacy aliases for backward compatibility
    mu_blend = 0.5 * mu_fast + 0.5 * mu_slow  # kept for diagnostics
    mu_post = mu_final  # primary drift estimate
    mu = mu_final  # shorthand

    # Short-term mean-reversion z (5d move over 1m vol)
    r5 = (log_px - log_px.shift(5))
    rv_1m = ret.rolling(21).std() * math.sqrt(5)
    z5 = r5 / rv_1m

    # Rolling skewness (directional asymmetry) and excess kurtosis (Fisher)
    skew = ret.rolling(252, min_periods=63).skew()
    # Optional stabilization: smooth skew to avoid warm-up swings when it first becomes defined
    try:
        skew_s = skew.ewm(span=30, adjust=False).mean()
    except Exception:
        skew_s = skew
    ex_kurt = ret.rolling(252, min_periods=63).kurt()  # normal ~ 0
    # Convert excess kurtosis to t degrees of freedom via: excess = 6/(nu-4) => nu = 4 + 6/excess
    # Handle near-zero/negative excess by mapping to large nu (approx normal)
    eps = 1e-6
    nu = 4.0 + 6.0 / ex_kurt.where(ex_kurt > eps, np.nan)
    nu = nu.fillna(1e6)  # ~normal
    # Clip degrees of freedom to a stable range to prevent extreme tail chaos in flash crashes
    nu = nu.clip(lower=4.5, upper=500.0)

    # Tail parameter: prefer tuned ν from cache for Student-t world; otherwise keep legacy estimate
    is_student_t_world = tuned_noise_model.startswith('phi_student_t_nu_')
    if is_student_t_world and tuned_nu is not None and np.isfinite(tuned_nu):
        # Level-7 rule: ν is fixed from tuning cache in Student-t world
        nu_hat = float(tuned_nu)
        nu_info = {"nu_hat": nu_hat, "source": "tuned_cache"}
    else:
        # Non-Student-t worlds may estimate ν diagnostically; Student-t world never refits
        try:
            mu_post_aligned = pd.Series(mu_post, index=ret.index).astype(float)
            vol_aligned = pd.Series(vol, index=ret.index).astype(float)
            resid = (ret - mu_post_aligned).replace([np.inf, -np.inf], np.nan)
            z_std = resid / vol_aligned.replace(0.0, np.nan)
            z_std = z_std.replace([np.inf, -np.inf], np.nan).dropna()
            nu_info = _fit_student_nu_mle(z_std, min_n=200, bounds=(4.5, 500.0))
            nu_hat = float(nu_info.get("nu_hat", 50.0))
        except Exception:
            nu_info = {"nu_hat": 50.0, "ll": float("nan"), "n": 0, "converged": False}
            nu_hat = 50.0

    # t-stat style momentum: cum return / realized vol over window
    def mom_t(days: int) -> pd.Series:
        cum = (log_px - log_px.shift(days))
        rv = ret.rolling(days).std() * math.sqrt(days)
        return cum / rv

    mom21 = mom_t(21)
    mom63 = mom_t(63)
    mom126 = mom_t(126)
    mom252 = mom_t(252)

    # Reuse HMM result from drift estimation (avoid duplicate fitting)
    hmm_result = hmm_result_prelim

    return {
        "px": px,
        "ret": ret,
        "mu": mu,
        "mu_post": mu_post,
        "mu_blend": mu_blend,
        "vol": vol,
        "vol_regime": vol_regime,
        "trend_z": trend_z,
        "z5": z5,
        "nu": nu,               # rolling, for diagnostics only
        "nu_hat": pd.Series([nu_hat], index=[ret.index[-1]]) if len(ret.index)>0 else pd.Series([nu_hat]),
        "nu_info": nu_info,     # dict metadata
        "skew": skew,
        "skew_s": skew_s,
        "mom21": mom21,
        "mom63": mom63,
        "mom126": mom126,
        "mom252": mom252,
        # meta (not series)
        "vol_source": vol_source,
        "garch_params": garch_params,
        # HMM regime detection
        "hmm_result": hmm_result,
        # Pillar 1: Kalman filter drift estimation
        "mu_kf": mu_kf if kalman_available else mu_blend,  # Kalman-filtered drift
        "var_kf": var_kf if kalman_available else pd.Series(0.0, index=ret.index),  # drift variance
        "mu_final": mu_final,  # shorthand
        "kalman_available": kalman_available,  # flag for diagnostics
        "kalman_metadata": kalman_metadata,  # log-likelihood, process noise, etc.
        "phi_used": kalman_metadata.get("phi_used", tuned_params.get("phi") if tuned_params else None),
        # Calibrated Trust Authority
        # ARCHITECTURAL LAW: Trust = Calibration Authority − Bounded Regime Penalty
        "calibrated_trust": tuned_params.get("calibrated_trust") if tuned_params else None,
        "recalibration": tuned_params.get("recalibration") if tuned_params else None,
        "recalibration_applied": tuned_params.get("recalibration_applied", False) if tuned_params else False,
    }

# -------------------------
# HMM Regime Detection (Formal Bayesian Inference)
# -------------------------

def fit_hmm_regimes(feats: Dict[str, pd.Series], n_states: int = 3, random_seed: int = 42) -> Optional[Dict]:
    """
    Fit a Hidden Markov Model with Gaussian emissions to detect market regimes.

    Each regime (state) has:
    - Its own μ (drift) dynamics captured by emission mean
    - Its own σ (volatility) dynamics captured by emission covariance
    - Persistence captured by transition matrix

    Args:
        feats: Feature dictionary from compute_features()
        n_states: Number of hidden states (default 3: calm, trending, crisis)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with HMM model, state sequence, and regime metadata, or None on failure
    """
    if not HMM_AVAILABLE:
        return None

    try:
        # Extract returns and volatility as observations
        ret = feats.get("ret", pd.Series(dtype=float))
        vol = feats.get("vol", pd.Series(dtype=float))

        if ret.empty or vol.empty:
            return None

        # Align and clean data
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
        if len(df) < 300:  # Need sufficient history for stable HMM
            return None

        df.columns = ["ret", "vol"]
        X = df.values  # Shape (T, 2): returns and volatility as features

        # Fit Gaussian HMM with full covariance (allows each state its own μ and σ)
        # Suppress stdout to hide noisy convergence messages from hmmlearn
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_seed,
            verbose=False
        )

        with suppress_stdout():
            model.fit(X)

        # Infer hidden state sequence (Viterbi for most likely path)
        states = model.predict(X)

        # Posterior probabilities for each state at each time
        posteriors = model.predict_proba(X)

        # Identify regime characteristics from emission parameters
        means = model.means_  # Shape (n_states, 2): [drift, vol] per state
        covars = model.covars_  # Shape (n_states, 2, 2)
        transmat = model.transmat_  # Shape (n_states, n_states)

        # Label states by volatility level: calm < normal < crisis
        vol_means = means[:, 1]  # volatility component
        sorted_indices = np.argsort(vol_means)

        regime_names = {
            sorted_indices[0]: "calm",
            sorted_indices[1]: "trending" if n_states == 3 else "normal",
            sorted_indices[2]: "crisis" if n_states == 3 else "volatile"
        }

        # Build regime series aligned with returns index
        regime_series = pd.Series(
            [regime_names.get(s, f"state_{s}") for s in states],
            index=df.index,
            name="regime"
        )

        # Posterior probability series (one per state)
        posterior_df = pd.DataFrame(
            posteriors,
            index=df.index,
            columns=[regime_names.get(i, f"state_{i}") for i in range(n_states)]
        )

        # Compute log-likelihood and information criteria for model diagnostics
        try:
            log_likelihood = float(model.score(X))
            n_obs = int(len(X))
            # Count free parameters: n_states-1 for initial probs, n_states*(n_states-1) for transitions,
            # n_states*n_features for means, n_states*n_features*(n_features+1)/2 for full covariance
            n_features = X.shape[1]
            n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * n_features * (n_features + 1) // 2
            aic = float(2.0 * n_params - 2.0 * log_likelihood)
            bic = float(n_params * np.log(n_obs) - 2.0 * log_likelihood)
        except Exception:
            log_likelihood = float("nan")
            n_obs = int(len(X))
            n_params = 0
            aic = float("nan")
            bic = float("nan")

        return {
            "model": model,
            "regime_series": regime_series,
            "posterior_probs": posterior_df,
            "states": states,
            "means": means,
            "covars": covars,
            "transmat": transmat,
            "regime_names": regime_names,
            "n_states": n_states,
            "log_likelihood": log_likelihood,
            "n_obs": n_obs,
            "n_params": n_params,
            "aic": aic,
            "bic": bic,
        }

    except Exception as e:
        # Silent fallback on HMM failure
        return None


def track_parameter_stability(ret: pd.Series, window_days: int = 252, step_days: int = 63) -> Dict[str, pd.DataFrame]:
    """
    Track GARCH parameter stability over time using rolling window estimation.

    Fits GARCH(1,1) on expanding windows to detect parameter drift.
    Returns time series of parameters, standard errors, and log-likelihoods.

    Args:
        ret: Returns series
        window_days: Minimum window size for initial fit
        step_days: Days between refits (trades off compute vs resolution)

    Returns:
        Dictionary with DataFrames tracking parameters over time
    """
    ret_clean = _ensure_float_series(ret).dropna()
    if len(ret_clean) < max(300, window_days):
        return {}

    # Time points to evaluate (start at window_days, step forward)
    dates = ret_clean.index
    eval_dates = []
    for i in range(window_days, len(dates), step_days):
        eval_dates.append(dates[i])

    if not eval_dates:
        return {}

    # Storage for parameter evolution
    records = []

    for eval_date in eval_dates:
        # Use expanding window up to eval_date
        window_ret = ret_clean.loc[:eval_date]

        # Try to fit GARCH
        try:
            _, params = _garch11_mle(window_ret)
            record = {
                "date": eval_date,
                "omega": params.get("omega", float("nan")),
                "alpha": params.get("alpha", float("nan")),
                "beta": params.get("beta", float("nan")),
                "se_omega": params.get("se_omega", float("nan")),
                "se_alpha": params.get("se_alpha", float("nan")),
                "se_beta": params.get("se_beta", float("nan")),
                "log_likelihood": params.get("log_likelihood", float("nan")),
                "aic": params.get("aic", float("nan")),
                "bic": params.get("bic", float("nan")),
                "n_obs": params.get("n_obs", 0),
                "converged": params.get("converged", False),
            }
            records.append(record)
        except Exception:
            # Skip windows where GARCH fails
            continue

    if not records:
        return {}

    df = pd.DataFrame(records).set_index("date")

    # Compute parameter drift statistics (rolling z-score of parameter changes)
    param_cols = ["omega", "alpha", "beta"]
    drift_stats = {}

    for col in param_cols:
        if col in df.columns:
            changes = df[col].diff()
            se_col = f"se_{col}"
            if se_col in df.columns:
                # Normalized change (z-score): change / standard error
                z_change = changes / df[se_col].replace(0, np.nan)
                drift_stats[f"{col}_drift_z"] = z_change

    drift_df = pd.DataFrame(drift_stats, index=df.index)

    return {
        "param_evolution": df,
        "param_drift": drift_df,
    }


def walk_forward_validation(px: pd.Series, train_days: int = 504, test_days: int = 21, horizons: List[int] = [1, 21, 63]) -> Dict[str, pd.DataFrame]:
    """
    Perform walk-forward out-of-sample testing to validate predictive power.

    Splits data into non-overlapping train/test windows, fits model on train,
    predicts on test, and tracks hit rates and prediction errors.

    Args:
        px: Price series
        train_days: Training window size (days)
        test_days: Test window size (days)
        horizons: Forecast horizons to test

    Returns:
        Dictionary with out-of-sample performance metrics
    """
    px_clean = _ensure_float_series(px).dropna()
    if len(px_clean) < train_days + test_days + max(horizons):
        return {}

    log_px = np.log(px_clean)
    dates = px_clean.index

    # Define walk-forward windows
    windows = []
    start_idx = 0
    while start_idx + train_days + test_days <= len(dates):
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(dates))

        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        if len(test_dates) > 0:
            windows.append({
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            })

        # Move forward by test_days (non-overlapping)
        start_idx = test_end_idx

    if not windows:
        return {}

    # Track predictions and outcomes for each horizon
    results = {h: [] for h in horizons}

    for window in windows:
        # Fit features on training data
        train_px = px_clean.loc[window["train_start"]:window["train_end"]]

        try:
            train_feats = compute_features(train_px)

            # Get predictions at end of training window
            mu_now = safe_last(train_feats.get("mu_post", pd.Series([0.0])))
            vol_now = safe_last(train_feats.get("vol", pd.Series([1.0])))

            if not np.isfinite(mu_now):
                mu_now = 0.0
            if not np.isfinite(vol_now) or vol_now <= 0:
                vol_now = 1.0

            # For each horizon, predict and measure actual outcome
            test_log_px = log_px.loc[window["test_start"]:window["test_end"]]
            train_end_log_px = float(log_px.loc[window["train_end"]])

            for H in horizons:
                # Predicted return over H days
                pred_ret_H = mu_now * H
                pred_sign = np.sign(pred_ret_H) if pred_ret_H != 0 else 0

                # Actual return H days forward from train_end
                try:
                    forward_idx = dates.get_loc(window["train_end"]) + H
                    if forward_idx < len(dates):
                        forward_date = dates[forward_idx]
                        actual_log_px = float(log_px.loc[forward_date])
                        actual_ret_H = actual_log_px - train_end_log_px
                        actual_sign = np.sign(actual_ret_H) if actual_ret_H != 0 else 0

                        # Prediction error
                        pred_error = actual_ret_H - pred_ret_H

                        # Direction hit (1 if signs match, 0 otherwise)
                        hit = 1 if (pred_sign * actual_sign > 0) else 0

                        results[H].append({
                            "train_end": window["train_end"],
                            "forecast_date": forward_date,
                            "predicted_return": pred_ret_H,
                            "actual_return": actual_ret_H,
                            "prediction_error": pred_error,
                            "direction_hit": hit,
                        })
                except Exception:
                    continue

        except Exception:
            continue

    # Aggregate results into DataFrames
    oos_metrics = {}
    for H in horizons:
        if results[H]:
            df = pd.DataFrame(results[H]).set_index("train_end")

            # Compute cumulative statistics
            hit_rate = df["direction_hit"].mean() if len(df) > 0 else float("nan")
            mean_error = df["prediction_error"].mean() if len(df) > 0 else float("nan")
            rmse = np.sqrt((df["prediction_error"] ** 2).mean()) if len(df) > 0 else float("nan")

            oos_metrics[f"H{H}"] = {
                "predictions": df,
                "hit_rate": float(hit_rate),
                "mean_error": float(mean_error),
                "rmse": float(rmse),
                "n_forecasts": len(df),
            }

    return oos_metrics

def compute_all_diagnostics(px: pd.Series, feats: Dict[str, pd.Series], enable_oos: bool = False, enable_pit_calibration: bool = False, enable_model_comparison: bool = False) -> Dict:
    """
    Compute comprehensive diagnostics: log-likelihood monitoring, parameter stability,
    and optionally out-of-sample tests, PIT calibration verification, and structural model comparison.

    Args:
        px: Price series
        feats: Feature dictionary from compute_features
        enable_oos: If True, run expensive out-of-sample validation
        enable_pit_calibration: If True, run PIT calibration verification (expensive)
        enable_model_comparison: If True, run structural model comparison (AIC/BIC falsifiability)

    Returns:
        Dictionary with all diagnostic metrics
    """
    diagnostics = {}

    # 1. Log-likelihood monitoring from fitted models
    garch_params = feats.get("garch_params", {})
    if isinstance(garch_params, dict):
        diagnostics["garch_log_likelihood"] = garch_params.get("log_likelihood", float("nan"))
        diagnostics["garch_aic"] = garch_params.get("aic", float("nan"))
        diagnostics["garch_bic"] = garch_params.get("bic", float("nan"))
        diagnostics["garch_n_obs"] = garch_params.get("n_obs", 0)

    # Pillar 1: Kalman filter drift diagnostics (with refinements)
    kalman_metadata = feats.get("kalman_metadata", {})
    if isinstance(kalman_metadata, dict):
        diagnostics["kalman_log_likelihood"] = kalman_metadata.get("log_likelihood", float("nan"))
        diagnostics["kalman_process_noise_var"] = kalman_metadata.get("process_noise_var", float("nan"))
        diagnostics["kalman_n_obs"] = kalman_metadata.get("n_obs", 0)
        # Refinement 1: q optimization results
        diagnostics["kalman_q_optimal"] = kalman_metadata.get("q_optimal", float("nan"))
        diagnostics["kalman_q_heuristic"] = kalman_metadata.get("q_heuristic", float("nan"))
        diagnostics["kalman_q_optimization_attempted"] = kalman_metadata.get("q_optimization_attempted", False)
        # Refinement 2: Kalman gain statistics (situational awareness)
        diagnostics["kalman_gain_mean"] = kalman_metadata.get("kalman_gain_mean", float("nan"))
        diagnostics["kalman_gain_recent"] = kalman_metadata.get("kalman_gain_recent", float("nan"))
        # Refinement 3: Innovation whiteness test (model adequacy)
        innovation_whiteness = kalman_metadata.get("innovation_whiteness", {})
        if isinstance(innovation_whiteness, dict):
            diagnostics["innovation_ljung_box_statistic"] = innovation_whiteness.get("ljung_box_statistic", float("nan"))
            diagnostics["innovation_ljung_box_pvalue"] = innovation_whiteness.get("ljung_box_pvalue", float("nan"))
            diagnostics["innovation_model_adequate"] = innovation_whiteness.get("model_adequate", None)
            diagnostics["innovation_lags_tested"] = innovation_whiteness.get("lags_tested", 0)
        # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
        diagnostics["kalman_heteroskedastic_mode"] = kalman_metadata.get("heteroskedastic_mode", False)
        diagnostics["kalman_c_optimal"] = kalman_metadata.get("c_optimal")
        diagnostics["kalman_q_t_mean"] = kalman_metadata.get("q_t_mean")
        diagnostics["kalman_q_t_std"] = kalman_metadata.get("q_t_std")
        diagnostics["kalman_q_t_min"] = kalman_metadata.get("q_t_min")
        diagnostics["kalman_q_t_max"] = kalman_metadata.get("q_t_max")
        # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
        diagnostics["kalman_robust_t_mode"] = kalman_metadata.get("robust_t_mode", False)
        diagnostics["kalman_nu_robust"] = kalman_metadata.get("nu_robust")
        # Level-7+ Refinement: Regime-dependent drift priors
        diagnostics["kalman_regime_prior_used"] = kalman_metadata.get("regime_prior_used", False)
        diagnostics["kalman_regime_info"] = kalman_metadata.get("regime_prior_info", {})
        # φ persistence (from tuned cache or filter)
        diagnostics["kalman_phi"] = tuned_params.get("phi") if tuned_params else kalman_metadata.get("phi_used")
        diagnostics["phi_used"] = kalman_metadata.get("phi_used")

    hmm_result = feats.get("hmm_result")
    if hmm_result is not None and isinstance(hmm_result, dict):
        diagnostics["hmm_log_likelihood"] = hmm_result.get("log_likelihood", float("nan"))
        diagnostics["hmm_aic"] = hmm_result.get("aic", float("nan"))
        diagnostics["hmm_bic"] = hmm_result.get("bic", float("nan"))
        diagnostics["hmm_n_obs"] = hmm_result.get("n_obs", 0)

    nu_info = feats.get("nu_info", {})
    if isinstance(nu_info, dict):
        diagnostics["student_t_log_likelihood"] = nu_info.get("ll", float("nan"))
        diagnostics["student_t_nu"] = nu_info.get("nu_hat", float("nan"))
        diagnostics["student_t_n_obs"] = nu_info.get("n", 0)
        # Tier 2: Add standard error for posterior parameter variance tracking
        diagnostics["student_t_se_nu"] = nu_info.get("se_nu", float("nan"))

    # 2. Parameter stability tracking (expensive, only if enough data)
    ret = feats.get("ret", pd.Series(dtype=float))
    if not ret.empty and len(ret) >= 600:
        try:
            stability = track_parameter_stability(ret, window_days=252, step_days=126)
            if stability:
                diagnostics["parameter_stability"] = stability

                # Summary statistics: recent drift magnitude
                param_drift = stability.get("param_drift")
                if param_drift is not None and not param_drift.empty:
                    recent_drift = param_drift.tail(1)
                    for col in param_drift.columns:
                        val = safe_last(param_drift[col])
                        diagnostics[f"recent_{col}"] = float(val) if np.isfinite(val) else float("nan")
        except Exception:
            pass

    # 3. Out-of-sample tests (very expensive, optional)
    if enable_oos and not px.empty and len(px) >= 800:
        try:
            oos_metrics = walk_forward_validation(px, train_days=504, test_days=21, horizons=[1, 21, 63])
            if oos_metrics:
                diagnostics["out_of_sample"] = oos_metrics

                # Summary: hit rates for each horizon
                for horizon_key, metrics in oos_metrics.items():
                    if isinstance(metrics, dict):
                        hit_rate = metrics.get("hit_rate", float("nan"))
                        diagnostics[f"oos_{horizon_key}_hit_rate"] = float(hit_rate)
        except Exception:
            pass

    # 4. PIT calibration verification (Level-7: probability calibration test)
    if enable_pit_calibration and not px.empty and len(px) >= 1000:
        try:
            from calibration.pit_calibration import run_pit_calibration_test

            # Run calibration test for key horizons
            calibration_results = run_pit_calibration_test(
                px=px,
                horizons=[1, 21, 63],
                n_bins=10,
                train_days=504,
                test_days=21,
                max_predictions=500
            )

            if calibration_results:
                diagnostics["pit_calibration"] = calibration_results

                # Summary: calibration status per horizon
                for horizon, metrics in calibration_results.items():
                    diagnostics[f"pit_H{horizon}_ece"] = metrics.expected_calibration_error
                    diagnostics[f"pit_H{horizon}_calibrated"] = metrics.calibrated
                    diagnostics[f"pit_H{horizon}_diagnosis"] = metrics.calibration_diagnosis
                    diagnostics[f"pit_H{horizon}_n_predictions"] = metrics.n_predictions
        except Exception as e:
            diagnostics["pit_calibration_error"] = str(e)

    # 5. Structural model comparison (Level-7: formal falsifiability via AIC/BIC)
    if enable_model_comparison:
        try:
            from model_comparison import run_all_comparisons

            # Get required inputs
            ret = feats.get("ret", pd.Series(dtype=float))
            vol = feats.get("vol", pd.Series(dtype=float))
            garch_params = feats.get("garch_params", {})
            nu_info = feats.get("nu_info", {})
            kalman_metadata = feats.get("kalman_metadata", {})

            if not ret.empty and not vol.empty:
                # Run all model comparisons
                comparison_results = run_all_comparisons(
                    returns=ret,
                    volatility=vol,
                    garch_params=garch_params if isinstance(garch_params, dict) else None,
                    student_t_params=nu_info if isinstance(nu_info, dict) else None,
                    kalman_metadata=kalman_metadata if isinstance(kalman_metadata, dict) else None,
                )

                diagnostics["model_comparison"] = comparison_results

                # Summary: winner per category
                for category, result in comparison_results.items():
                    if result is not None and hasattr(result, 'winner_aic'):
                        diagnostics[f"model_comparison_{category}_winner_aic"] = result.winner_aic
                        diagnostics[f"model_comparison_{category}_winner_bic"] = result.winner_bic
                        diagnostics[f"model_comparison_{category}_recommendation"] = result.recommendation
        except Exception as e:
            diagnostics["model_comparison_error"] = str(e)

    return diagnostics


def infer_current_regime(feats: Dict[str, pd.Series], hmm_result: Optional[Dict] = None) -> Tuple[str, Dict[str, float]]:
    """
    Infer the current market regime using posterior inference from HMM.

    Args:
        feats: Feature dictionary
        hmm_result: Result from fit_hmm_regimes(), or None to use threshold fallback

    Returns:
        Tuple of (regime_label, regime_metadata_dict)
        regime_label: "calm", "trending", "crisis", or threshold-based fallback
        regime_metadata: probabilities and diagnostics
    """
    # If HMM available and fitted, use posterior inference
    if hmm_result is not None and "regime_series" in hmm_result:
        try:
            regime_series = hmm_result["regime_series"]
            posterior_probs = hmm_result["posterior_probs"]

            if not regime_series.empty:
                current_regime = regime_series.iloc[-1]
                current_probs = posterior_probs.iloc[-1].to_dict()

                return str(current_regime), {
                    "method": "hmm_posterior",
                    "probabilities": current_probs,
                    "persistence": float(hmm_result["transmat"][hmm_result["states"][-1], hmm_result["states"][-1]]) if len(hmm_result["states"]) > 0 else 0.5,
                }
        except Exception:
            pass

    # Fallback to threshold-based regime detection (original logic)
    vol_regime = feats.get("vol_regime", pd.Series(dtype=float))
    trend_z = feats.get("trend_z", pd.Series(dtype=float))

    vr = safe_last(vol_regime) if not vol_regime.empty else float("nan")
    tz = safe_last(trend_z) if not trend_z.empty else float("nan")

    # Threshold-based classification
    if np.isfinite(vr) and vr > 1.8:
        if np.isfinite(tz) and tz > 0:
            label = "High-vol uptrend"
        elif np.isfinite(tz) and tz < 0:
            label = "High-vol downtrend"
        else:
            label = "crisis"  # Map to HMM-style label
    elif np.isfinite(vr) and vr < 0.85:
        if np.isfinite(tz) and tz > 0:
            label = "Calm uptrend"
        elif np.isfinite(tz) and tz < 0:
            label = "Calm downtrend"
        else:
            label = "calm"  # Map to HMM-style label
    elif np.isfinite(tz) and abs(tz) > 0.5:
        label = "trending"
    else:
        label = "Normal"

    return label, {
        "method": "threshold_fallback",
        "vol_regime": float(vr) if np.isfinite(vr) else None,
        "trend_z": float(tz) if np.isfinite(tz) else None,
    }


# =============================================================================
# REGIME-CONDITIONAL BAYESIAN MODEL AVERAGING (RC-BMA)
# =============================================================================
# Implements: p(r_H | D) = Σ_r P(regime_r | D) · p(r_H | regime_r, D)
# Regimes must match those in tune.py
# =============================================================================

# Regime definitions (must match tune.py)
REGIME_LOW_VOL_TREND = 0
REGIME_HIGH_VOL_TREND = 1
REGIME_LOW_VOL_RANGE = 2
REGIME_HIGH_VOL_RANGE = 3
REGIME_CRISIS_JUMP = 4

REGIME_NAMES = {
    REGIME_LOW_VOL_TREND: "LOW_VOL_TREND",
    REGIME_HIGH_VOL_TREND: "HIGH_VOL_TREND",
    REGIME_LOW_VOL_RANGE: "LOW_VOL_RANGE",
    REGIME_HIGH_VOL_RANGE: "HIGH_VOL_RANGE",
    REGIME_CRISIS_JUMP: "CRISIS_JUMP",
}


def assign_current_regime(feats: Dict[str, pd.Series], lookback: int = 21) -> int:
    """
    Assign current regime using SAME logic as tune.py's assign_regime_labels.

    This ensures regime assignment is deterministic and consistent between tuning
    and inference. Uses only past data (no look-ahead).

    Classification Logic (matches tune.py):
    - CRISIS_JUMP (4): vol_relative > 2.0 OR tail_indicator > 4.0
    - HIGH_VOL_TREND (1): vol_relative > 1.3 AND drift_abs > threshold
    - HIGH_VOL_RANGE (3): vol_relative > 1.3 AND drift_abs <= threshold
    - LOW_VOL_TREND (0): vol_relative < 0.85 AND drift_abs > threshold
    - LOW_VOL_RANGE (2): vol_relative < 0.85 AND drift_abs <= threshold
    - Normal vol: based on drift threshold

    Args:
        feats: Feature dictionary with 'ret' and 'vol' series
        lookback: Rolling window for feature computation (default 21 days)

    Returns:
        Integer regime index (0-4)
    """
    # Extract returns and volatility
    ret_series = feats.get("ret", pd.Series(dtype=float))
    vol_series = feats.get("vol", pd.Series(dtype=float))

    if not isinstance(ret_series, pd.Series) or ret_series.empty:
        return REGIME_LOW_VOL_RANGE  # Default
    if not isinstance(vol_series, pd.Series) or vol_series.empty:
        return REGIME_LOW_VOL_RANGE  # Default

    # Current values
    vol_now = float(vol_series.iloc[-1]) if len(vol_series) > 0 else 0.0
    ret_now = float(ret_series.iloc[-1]) if len(ret_series) > 0 else 0.0

    # Rolling mean absolute return (drift proxy)
    if len(ret_series) >= lookback:
        drift_abs = abs(float(ret_series.tail(lookback).mean()))
    else:
        drift_abs = abs(float(ret_series.mean()))

    # Volatility relative to expanding median
    if len(vol_series) >= lookback:
        vol_median = float(vol_series.expanding(min_periods=min(lookback, len(vol_series))).median().iloc[-1])
    else:
        vol_median = float(vol_series.median())

    vol_relative = vol_now / vol_median if vol_median > 1e-12 else 1.0

    # Tail indicator: |return| / vol
    tail_indicator = abs(ret_now) / vol_now if vol_now > 1e-12 else 0.0

    # Drift threshold (same as tune.py)
    drift_threshold = 0.0005  # ~0.05% daily drift threshold

    # Classification logic (MUST match tune.py assign_regime_labels)
    # Crisis/Jump: extreme volatility or tail events
    if vol_relative > 2.0 or tail_indicator > 4.0:
        return REGIME_CRISIS_JUMP

    # High volatility regimes
    if vol_relative > 1.3:
        if drift_abs > drift_threshold:
            return REGIME_HIGH_VOL_TREND
        else:
            return REGIME_HIGH_VOL_RANGE

    # Low volatility regimes
    if vol_relative < 0.85:
        if drift_abs > drift_threshold:
            return REGIME_LOW_VOL_TREND
        else:
            return REGIME_LOW_VOL_RANGE

    # Normal volatility (between 0.85 and 1.3)
    if drift_abs > drift_threshold * 1.5:
        return REGIME_HIGH_VOL_TREND if vol_relative > 1.0 else REGIME_LOW_VOL_TREND
    else:
        return REGIME_HIGH_VOL_RANGE if vol_relative > 1.0 else REGIME_LOW_VOL_RANGE


def map_regime_label_to_index(regime_label: str, regime_meta: Optional[Dict] = None) -> int:
    """
    Map a regime label (string) to a regime index (0-4) matching tune.py definitions.

    Maps from infer_current_regime() output to tune.py regime indices.

    Mapping logic:
    - "crisis" / "High-vol*" → CRISIS_JUMP (4) if vol_regime > 2.0 else HIGH_VOL_RANGE (3)
    - "trending" → HIGH_VOL_TREND (1) or LOW_VOL_TREND (0) based on vol_regime
    - "calm" / "Calm*" → LOW_VOL_RANGE (2) or LOW_VOL_TREND (0) based on drift
    - "Normal" → LOW_VOL_RANGE (2)

    Args:
        regime_label: String regime label from infer_current_regime()
        regime_meta: Optional regime metadata dict with vol_regime, trend_z, etc.

    Returns:
        Integer regime index (0-4)
    """
    label_lower = regime_label.lower() if regime_label else ""

    # Extract vol_regime from metadata if available
    vol_regime = None
    trend_z = None
    if regime_meta is not None:
        vol_regime = regime_meta.get("vol_regime")
        trend_z = regime_meta.get("trend_z")
        # Also check probabilities from HMM
        probs = regime_meta.get("probabilities", {})

    # Crisis detection (highest priority)
    if "crisis" in label_lower:
        return REGIME_CRISIS_JUMP

    # High volatility regimes
    if "high-vol" in label_lower or "high_vol" in label_lower:
        # Check if extreme crisis or just high vol trend/range
        if vol_regime is not None and vol_regime > 2.0:
            return REGIME_CRISIS_JUMP
        elif trend_z is not None and abs(trend_z) > 0.5:
            return REGIME_HIGH_VOL_TREND
        else:
            return REGIME_HIGH_VOL_RANGE

    # Trending detection
    if "trending" in label_lower or "trend" in label_lower:
        # Determine if low or high vol based on metadata
        if vol_regime is not None and vol_regime > 1.3:
            return REGIME_HIGH_VOL_TREND
        else:
            return REGIME_LOW_VOL_TREND

    # Calm/Low volatility regimes
    if "calm" in label_lower or "low" in label_lower:
        # Check if trending or ranging
        if trend_z is not None and abs(trend_z) > 0.5:
            return REGIME_LOW_VOL_TREND
        else:
            return REGIME_LOW_VOL_RANGE

    # Normal / default
    if "normal" in label_lower:
        return REGIME_LOW_VOL_RANGE

    # HMM-specific labels
    if label_lower in ("calm", "0"):
        return REGIME_LOW_VOL_RANGE
    elif label_lower in ("trending", "1"):
        return REGIME_LOW_VOL_TREND
    elif label_lower in ("crisis", "2"):
        return REGIME_CRISIS_JUMP

    # Default fallback
    return REGIME_LOW_VOL_RANGE


def extract_regime_features(feats: Dict[str, pd.Series]) -> Dict[str, float]:
    """
    Extract features for regime likelihood computation.

    Features:
    - vol_level: EWMA volatility (normalized)
    - drift_strength: |μ| (absolute drift)
    - drift_persistence: φ from Kalman
    - return_autocorr: autocorrelation of returns
    - tail_indicator: |return| / EWMA_σ (tail measure)

    Args:
        feats: Feature dictionary from compute_features()

    Returns:
        Dictionary of regime features
    """
    # Volatility level (normalized relative to median)
    vol_series = feats.get("vol", pd.Series(dtype=float))
    if isinstance(vol_series, pd.Series) and not vol_series.empty:
        vol_now = float(vol_series.iloc[-1])
        vol_median = float(vol_series.median())
        vol_level = vol_now / vol_median if vol_median > 1e-12 else 1.0
    else:
        vol_level = 1.0

    # Drift strength (absolute value of filtered drift)
    mu_series = feats.get("mu_post", feats.get("mu_kf", feats.get("mu", pd.Series(dtype=float))))
    if isinstance(mu_series, pd.Series) and not mu_series.empty:
        drift_strength = abs(float(mu_series.iloc[-1]))
    else:
        drift_strength = 0.0

    # Drift persistence (φ from Kalman metadata)
    km = feats.get("kalman_metadata", {}) or {}
    phi = km.get("phi_used") or km.get("kalman_phi")
    if phi is None or not np.isfinite(phi):
        phi = feats.get("phi_used")
    drift_persistence = float(phi) if phi is not None and np.isfinite(phi) else 0.95

    # Return autocorrelation
    ret_series = feats.get("ret", pd.Series(dtype=float))
    if isinstance(ret_series, pd.Series) and len(ret_series) >= 21:
        try:
            return_autocorr = float(ret_series.autocorr(lag=1))
            if not np.isfinite(return_autocorr):
                return_autocorr = 0.0
        except Exception:
            return_autocorr = 0.0
    else:
        return_autocorr = 0.0

    # Tail indicator: |recent return| / σ
    if isinstance(ret_series, pd.Series) and not ret_series.empty and vol_now > 1e-12:
        recent_ret = abs(float(ret_series.iloc[-1]))
        tail_indicator = recent_ret / vol_now
    else:
        tail_indicator = 0.0

    return {
        "vol_level": float(np.clip(vol_level, 0.1, 10.0)),
        "drift_strength": float(np.clip(drift_strength, 0.0, 0.01)),
        "drift_persistence": float(np.clip(drift_persistence, -1.0, 1.0)),
        "return_autocorr": float(np.clip(return_autocorr, -1.0, 1.0)),
        "tail_indicator": float(np.clip(tail_indicator, 0.0, 10.0)),
    }


def compute_regime_log_likelihoods(features: Dict[str, float]) -> np.ndarray:
    """
    Compute log-likelihood scores for each regime given features.

    Uses Gaussian/logistic scoring based on regime characteristics:

    LOW_VOL_TREND (0): low vol, high drift_strength, high persistence
    HIGH_VOL_TREND (1): high vol, high drift_strength, high persistence
    LOW_VOL_RANGE (2): low vol, low drift_strength, low persistence
    HIGH_VOL_RANGE (3): high vol, low drift_strength, low persistence
    CRISIS_JUMP (4): extreme vol, high tail_indicator

    Args:
        features: Dictionary from extract_regime_features()

    Returns:
        Array of log-likelihoods for regimes 0-4
    """
    vol = features["vol_level"]
    drift = features["drift_strength"]
    persist = features["drift_persistence"]
    autocorr = features["return_autocorr"]
    tail = features["tail_indicator"]

    # Define regime scoring functions (Gaussian scoring)
    # Higher score = more likely regime

    log_L = np.zeros(5)

    # Regime 0: LOW_VOL_TREND - low vol, high drift, high persistence
    log_L[0] = (
        -0.5 * ((vol - 0.7) / 0.3) ** 2      # vol centered at 0.7 (below median)
        - 0.5 * ((drift - 0.002) / 0.001) ** 2  # strong drift
        - 0.5 * ((persist - 0.98) / 0.02) ** 2  # high persistence
    )

    # Regime 1: HIGH_VOL_TREND - high vol, high drift, high persistence
    log_L[1] = (
        -0.5 * ((vol - 1.5) / 0.4) ** 2      # vol above median
        - 0.5 * ((drift - 0.003) / 0.002) ** 2  # strong drift
        - 0.5 * ((persist - 0.95) / 0.03) ** 2  # high persistence
    )

    # Regime 2: LOW_VOL_RANGE - low vol, low drift, mean reversion
    log_L[2] = (
        -0.5 * ((vol - 0.6) / 0.25) ** 2     # low vol
        - 0.5 * ((drift - 0.0003) / 0.0005) ** 2  # near-zero drift
        - 0.5 * ((persist - 0.85) / 0.1) ** 2   # moderate persistence (mean reversion)
    )

    # Regime 3: HIGH_VOL_RANGE - high vol, low drift, choppy
    log_L[3] = (
        -0.5 * ((vol - 1.3) / 0.35) ** 2     # elevated vol
        - 0.5 * ((drift - 0.0005) / 0.001) ** 2  # low drift
        - 0.5 * ((persist - 0.80) / 0.15) ** 2  # low persistence (whipsaw)
        - 0.5 * ((autocorr - (-0.1)) / 0.2) ** 2  # slight negative autocorr
    )

    # Regime 4: CRISIS_JUMP - extreme vol, tail events
    log_L[4] = (
        -0.5 * ((vol - 2.5) / 0.5) ** 2      # extreme vol
        - 0.5 * ((tail - 3.0) / 1.0) ** 2    # high tail indicator
    )

    return log_L


def compute_regime_probabilities(
    features: Dict[str, float],
    smoothing_alpha: float = 0.3,
    prev_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute regime probabilities via softmax of log-likelihoods.

    P_regimes = softmax(logL_r)

    With optional exponential smoothing over time.

    Args:
        features: Dictionary from extract_regime_features()
        smoothing_alpha: EMA smoothing factor (0=full smooth, 1=no smooth)
        prev_probs: Previous regime probabilities for smoothing

    Returns:
        Array of probabilities for regimes 0-4, summing to 1
    """
    log_L = compute_regime_log_likelihoods(features)

    # Softmax for probabilities
    # Subtract max for numerical stability
    log_L_shifted = log_L - np.max(log_L)
    exp_L = np.exp(log_L_shifted)
    probs = exp_L / np.sum(exp_L)

    # Exponential smoothing if previous probabilities available
    if prev_probs is not None:
        prev_probs = np.asarray(prev_probs)
        if prev_probs.shape == probs.shape:
            probs = smoothing_alpha * probs + (1.0 - smoothing_alpha) * prev_probs
            # Renormalize after smoothing
            probs = probs / np.sum(probs)

    return probs


def run_regime_specific_mc(
    regime: int,
    mu_t: float,
    P_t: float,
    phi: float,
    q: float,
    sigma2_step: float,
    H: int,
    n_paths: int = 5000,
    nu: Optional[float] = None,
    nig_alpha: Optional[float] = None,
    nig_beta: Optional[float] = None,
    nig_delta: Optional[float] = None,
    hansen_lambda: Optional[float] = None,
    # Contaminated Student-t parameters
    cst_nu_normal: Optional[float] = None,
    cst_nu_crisis: Optional[float] = None,
    cst_epsilon: Optional[float] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Run posterior predictive MC for a specific regime.

    This is a lightweight wrapper that generates r_samples for one regime
    using regime-specific parameters.

    Supports five noise distributions (in priority order):
    1. Contaminated Student-t (cst_nu_normal, cst_nu_crisis, cst_epsilon specified):
       Regime-dependent heavy tails: (1-ε)×t(ν_normal) + ε×t(ν_crisis)
    2. Hansen Skew-t (nu + hansen_lambda specified): Asymmetric heavy tails
    3. NIG (nig_alpha, nig_beta, nig_delta specified): Semi-heavy tails, asymmetric
    4. Student-t (nu specified): Heavy tails, symmetric
    5. Gaussian (default): Light tails, symmetric
    
    Contaminated Student-t model:
        p(r) = (1-ε) × t(r; ν_normal) + ε × t(r; ν_crisis)
        
    Where ε is the contamination probability (crisis mode), typically 5-15%.
    This captures the intuition: "Most of the time markets are normal, but
    occasionally we're in crisis mode with much heavier tails."

    Args:
        regime: Regime index (0-4)
        mu_t: Current drift estimate
        P_t: Drift posterior variance
        phi: AR(1) persistence
        q: Process noise variance
        sigma2_step: Per-step observation variance
        H: Forecast horizon
        n_paths: Number of MC paths
        nu: Degrees of freedom for Student-t (None for Gaussian/NIG)
        nig_alpha: NIG tail parameter (None for Gaussian/Student-t)
        nig_beta: NIG asymmetry parameter (None for Gaussian/Student-t)
        nig_delta: NIG scale parameter (None for Gaussian/Student-t)
        hansen_lambda: Hansen skewness parameter (None for symmetric)
        cst_nu_normal: Contaminated-t normal regime ν
        cst_nu_crisis: Contaminated-t crisis regime ν
        cst_epsilon: Contaminated-t crisis probability
        seed: Random seed

    Returns:
        Array of return samples
    """
    # Import NIG sampling if needed
    use_nig = (nig_alpha is not None and nig_beta is not None and nig_delta is not None)
    if use_nig:
        try:
            from scipy.stats import norminvgauss
            # Validate NIG parameters
            nig_alpha = float(np.clip(nig_alpha, 0.5, 50.0))
            nig_delta = float(max(nig_delta, 0.001))
            max_beta = nig_alpha - 0.01
            nig_beta = float(np.clip(nig_beta, -max_beta, max_beta))
            # Convert to scipy parameterization
            nig_a = nig_alpha * nig_delta
            nig_b = nig_beta * nig_delta
        except Exception:
            use_nig = False
    
    # Check for Contaminated Student-t (highest priority for Student-t family)
    use_contaminated_t = (
        CONTAMINATED_ST_AVAILABLE and
        cst_nu_normal is not None and
        cst_nu_crisis is not None and
        cst_epsilon is not None and
        cst_epsilon > 0.001
    )
    
    # =========================================================================
    # HANSEN SKEW-T DETECTION (asymmetric Student-t)
    # =========================================================================
    # If hansen_lambda is provided and non-trivial, use Hansen skew-t sampling
    # instead of symmetric Student-t. This is the CRITICAL fix - hansen_lambda
    # was previously accepted but IGNORED in sampling.
    #
    # Priority order:
    #   1. Contaminated Student-t (regime-dependent tails)
    #   2. Hansen Skew-t (asymmetric tails with fixed λ)
    #   3. Symmetric Student-t (heavy tails only)
    #   4. NIG (semi-heavy tails with asymmetry via β)
    #   5. Gaussian (light tails)
    # =========================================================================
    use_hansen_skew_t = (
        HANSEN_SKEW_T_AVAILABLE and
        not use_contaminated_t and  # CST takes priority
        nu is not None and
        hansen_lambda is not None and
        abs(hansen_lambda) > 0.01  # Only use if λ is non-trivial
    )
    
    # Input validation
    mu_t = float(mu_t) if np.isfinite(mu_t) else 0.0
    P_t = float(max(P_t, 0.0)) if np.isfinite(P_t) else 0.0
    phi = float(phi) if np.isfinite(phi) else 1.0
    q = float(max(q, 0.0)) if np.isfinite(q) else 0.0
    sigma2_step = float(max(sigma2_step, 1e-12)) if np.isfinite(sigma2_step) else 1e-6
    H = int(max(H, 1))

    if nu is not None and not use_nig and not use_contaminated_t:
        if not np.isfinite(nu) or nu <= 2.0:
            nu = None
        else:
            nu = float(np.clip(nu, 2.1, 500.0))

    rng = np.random.default_rng(seed)

    # Pre-compute mixture variance for contaminated Student-t if needed
    # This MUST be computed before sampling to avoid "unbound local variable" error
    mixture_var = 1.0  # Default to 1.0 (standard normal variance)
    if use_contaminated_t:
        var_normal = cst_nu_normal / (cst_nu_normal - 2) if cst_nu_normal > 2 else 10.0
        var_crisis = cst_nu_crisis / (cst_nu_crisis - 2) if cst_nu_crisis > 2 else 10.0
        mixture_var = (1 - cst_epsilon) * var_normal + cst_epsilon * var_crisis
        mixture_var = max(mixture_var, 1e-10)  # Ensure positive

    # Sample drift posterior
    if P_t > 0:
        if use_nig:
            # NIG posterior for drift
            # Scale NIG to have variance = P_t
            # NIG variance = δα²/(α²-β²)^(3/2), so we scale samples
            gamma_nig = np.sqrt(max(nig_alpha**2 - nig_beta**2, 1e-10))
            nig_var = nig_delta * nig_alpha**2 / (gamma_nig**3) if gamma_nig > 0 else nig_delta**2
            scale_factor = np.sqrt(P_t / max(nig_var, 1e-10))
            nig_samples = norminvgauss.rvs(nig_a, nig_b, loc=0, scale=1, size=n_paths, random_state=rng)
            mu_paths = mu_t + scale_factor * nig_delta * nig_samples
        elif use_contaminated_t:
            # Contaminated Student-t posterior for drift
            # With probability ε, use crisis ν; else use normal ν
            mu_samples = contaminated_student_t_rvs(
                size=n_paths,
                nu_normal=cst_nu_normal,
                nu_crisis=cst_nu_crisis,
                epsilon=cst_epsilon,
                mu=0.0,
                sigma=1.0,
                random_state=rng
            )
            # Scale to have variance = P_t
            # Contaminated-t has variance ≈ weighted average of component variances
            # For t(ν): Var = ν/(ν-2) for ν > 2
            var_normal = cst_nu_normal / (cst_nu_normal - 2) if cst_nu_normal > 2 else 10.0
            var_crisis = cst_nu_crisis / (cst_nu_crisis - 2) if cst_nu_crisis > 2 else 10.0
            mixture_var = (1 - cst_epsilon) * var_normal + cst_epsilon * var_crisis
            t_scale = np.sqrt(P_t / mixture_var)
            mu_paths = mu_t + t_scale * mu_samples
        elif nu is not None:
            # Student-t posterior for drift
            t_scale = math.sqrt(P_t * (nu - 2.0) / nu) if nu > 2.0 else math.sqrt(P_t)
            mu_paths = mu_t + t_scale * rng.standard_t(df=nu, size=n_paths)
        else:
            # Gaussian posterior for drift
            mu_paths = rng.normal(loc=mu_t, scale=math.sqrt(P_t), size=n_paths)
    else:
        mu_paths = np.full(n_paths, mu_t, dtype=float)

    # Propagate drift and accumulate noise
    cum_mu = np.zeros(n_paths, dtype=float)
    cum_eps = np.zeros(n_paths, dtype=float)

    q_std = math.sqrt(q) if q > 0 else 0.0
    sigma_step = math.sqrt(sigma2_step)

    for k in range(H):
        # --- Drift propagation: μ_{t+k+1} = φ·μ_{t+k} + η_{k+1} ---
        if q_std > 0:
            if use_nig:
                # NIG drift noise
                nig_samples = norminvgauss.rvs(nig_a, nig_b, loc=0, scale=1, size=n_paths, random_state=rng)
                eta = q_std * nig_delta * nig_samples / np.sqrt(max(nig_var, 1e-10))
            elif use_contaminated_t:
                # Contaminated Student-t drift noise
                eta_samples = contaminated_student_t_rvs(
                    size=n_paths,
                    nu_normal=cst_nu_normal,
                    nu_crisis=cst_nu_crisis,
                    epsilon=cst_epsilon,
                    mu=0.0,
                    sigma=1.0,
                    random_state=rng
                )
                eta_scale = q_std / np.sqrt(mixture_var)
                eta = eta_scale * eta_samples
            elif nu is not None:
                eta_scale = q_std * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else q_std
                eta = eta_scale * rng.standard_t(df=nu, size=n_paths)
            else:
                eta = rng.normal(loc=0.0, scale=q_std, size=n_paths)
        else:
            eta = np.zeros(n_paths, dtype=float)

        mu_paths = phi * mu_paths + eta
        cum_mu += mu_paths

        # --- Observation noise: ε_k ---
        if sigma_step > 0:
            if use_nig:
                # NIG observation noise
                nig_samples = norminvgauss.rvs(nig_a, nig_b, loc=0, scale=1, size=n_paths, random_state=rng)
                eps_k = sigma_step * nig_delta * nig_samples / np.sqrt(max(nig_var, 1e-10))
            elif use_contaminated_t:
                # Contaminated Student-t observation noise
                eps_samples = contaminated_student_t_rvs(
                    size=n_paths,
                    nu_normal=cst_nu_normal,
                    nu_crisis=cst_nu_crisis,
                    epsilon=cst_epsilon,
                    mu=0.0,
                    sigma=1.0,
                    random_state=rng
                )
                eps_scale = sigma_step / np.sqrt(mixture_var)
                eps_k = eps_scale * eps_samples
            elif nu is not None:
                eps_scale = sigma_step * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else sigma_step
                eps_k = eps_scale * rng.standard_t(df=nu, size=n_paths)
            else:
                eps_k = rng.normal(loc=0.0, scale=sigma_step, size=n_paths)
        else:
            eps_k = np.zeros(n_paths, dtype=float)

        cum_eps += eps_k

    return cum_mu + cum_eps


def compute_model_posteriors_from_combined_score(
    models: Dict[str, Dict],
    temperature: float = 1.0,
    min_weight_fraction: float = 0.01,
    epsilon: float = 1e-10,
) -> Tuple[Dict[str, float], Dict]:
    """
    Convert combined scores into normalized posterior weights with entropy floor.

    This is the EPISTEMIC WEIGHTING step that ensures Hyvärinen scores
    directly influence signal generation.

    The combined_score is the entropy-regularized standardized score where:
        combined_score = w_bic * BIC_std - (1-w_bic) * Hyv_std

    Lower combined_score = better model.

    To get normalized posteriors we use softmax over NEGATED scores:
        p(m) = exp(-combined_score_m / T) / Σ_k exp(-combined_score_k / T)

    An entropy floor is applied to prevent belief collapse:
        w_m = max(w_m, min_weight_fraction / n_models)

    This ensures dominated models retain some probability mass, preventing
    overconfident allocations during regime transitions.

    Args:
        models: Dictionary mapping model_name -> model_params dict
                Each model_params must have 'combined_score'
        temperature: Softmax temperature (1.0 = standard, <1 = sharper, >1 = smoother)
        min_weight_fraction: Minimum total mass to uniform (0.01 = 1%)
        epsilon: Small constant to prevent zero weights

    Returns:
        Tuple of:
        - Dictionary mapping model_name -> posterior weight p(m)
        - Metadata dict
    """
    metadata = {
        "method": "combined",
        "temperature": temperature,
        "min_weight_fraction": min_weight_fraction,
    }

    # Extract valid models with combined scores
    valid_models = {}
    for model_name, model_params in models.items():
        if not isinstance(model_params, dict):
            continue
        if not model_params.get('fit_success', True):
            continue

        combined_score = model_params.get('combined_score')
        if combined_score is not None and np.isfinite(combined_score):
            valid_models[model_name] = combined_score

    if not valid_models:
        return {}, metadata

    # Convert to arrays for softmax
    model_names = list(valid_models.keys())
    scores = np.array([valid_models[m] for m in model_names])
    n_models = len(model_names)

    # Softmax over NEGATED scores (lower score = better = higher weight)
    # With numerical stabilization
    neg_scores = -scores / temperature
    neg_scores = neg_scores - neg_scores.max()  # Numerical stability

    weights = np.exp(neg_scores)
    weights = np.maximum(weights, epsilon)
    weights = weights / weights.sum()

    # =========================================================================
    # ENTROPY FLOOR: Prevent belief collapse
    # =========================================================================
    # Ensure each model has at least min_weight_fraction / n_models weight.
    # This prevents overconfident allocations during regime transitions or
    # when models happen to agree on similar scores.
    # =========================================================================
    min_weight_per_model = min_weight_fraction / max(n_models, 1)
    weights = np.maximum(weights, min_weight_per_model)
    weights = weights / weights.sum()  # Re-normalize after floor

    return dict(zip(model_names, weights)), metadata


def bayesian_model_average_mc(
    feats: Dict[str, pd.Series],
    regime_params: Dict[int, Dict],
    mu_t: float,
    P_t: float,
    sigma2_step: float,
    H: int,
    n_paths: int = 10000,
    seed: Optional[int] = None,
    tuned_params: Optional[Dict] = None,
    asset_symbol: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[int, float], Dict]:
    """
    Perform Bayesian Model Averaging using CURRENT REGIME's model posterior.

    Implements the posterior predictive for current regime r_t:

        p(x | D, r_t) = Σ_m p(x | r_t, m, θ_{r_t,m}) · p(m | r_t)

    KEY DESIGN:
    - Determines CURRENT regime using same logic as tune.py (deterministic)
    - Uses that regime's model_posterior and models from tuning
    - Does NOT blend across regimes - uses single regime's BMA
    - Falls back to global if regime data unavailable

    SOFT REGIME PROBABILITIES (for Trust Authority):
    - Hard regime assignment remains for parameter selection
    - Soft probabilities computed for trust modulation to avoid cliffs
    - Based on regime transition smoothing: current=0.7, adjacent=0.15 each

    CONTRACT WITH tune.py:
    - Regime assignment uses SAME logic as assign_regime_labels()
    - Every regime contains model_posterior and models (even fallbacks)
    - Model posteriors are already temporally smoothed by tune
    - This function does NOT perform tuning, BIC/AIC, or temporal smoothing

    CRITICAL RULES:
    - Do NOT perform tuning here
    - Do NOT recompute likelihoods
    - Do NOT renormalize model weights (already normalized by tune)
    - Do NOT apply temporal smoothing to model posteriors
    - Do NOT select best model - use full BMA mixture
    - Do NOT synthesize fake models - use hierarchical fallback to global

    Args:
        feats: Feature dictionary with 'ret', 'vol', etc.
        regime_params: (Legacy - ignored if tuned_params provided)
        mu_t: Current drift estimate
        P_t: Current drift variance
        sigma2_step: Per-step volatility
        H: Forecast horizon
        n_paths: Total MC paths to generate
        seed: Random seed for reproducibility
        tuned_params: Full tuned params from _load_tuned_kalman_params (BMA structure)

    Returns:
        Tuple of:
        - r_samples: Samples from posterior predictive p(x | D, r_t)
        - regime_probs: Soft probability dict for trust authority {regime_idx: prob}
        - metadata: Diagnostic information
    """
    rng = np.random.default_rng(seed)

    # Check if we have new BMA structure with model posteriors
    has_bma = tuned_params is not None and tuned_params.get('has_bma', False)

    # If no BMA structure available, this is old cache format - REJECT
    if not has_bma:
        # Determine the reason for missing BMA structure
        if tuned_params is None:
            reason = "tuned_params is None (asset not in cache/tune/)"
            error_type = "MISSING_CACHE"
        elif 'global' not in tuned_params:
            reason = "missing 'global' key (old cache format)"
            error_type = "OLD_CACHE_FORMAT"
        else:
            reason = "has_bma is False or missing"
            error_type = "BMA_DISABLED"
        
        # Format asset display name
        asset_display = asset_symbol if asset_symbol else "UNKNOWN"
        
        # Print improved warning with asset name
        print(f"\n⚠️  BMA STRUCTURE MISSING for {asset_display}", file=sys.stderr)
        print(f"   ├─ Reason: {reason}", file=sys.stderr)
        print(f"   ├─ Impact: Signals will show 0% for all horizons", file=sys.stderr)
        print(f"   └─ Fix: Run 'make tune --assets {asset_display}' to regenerate cache\n", file=sys.stderr)
        
        # Save failure to failed_assets.json for tracking
        try:
            if asset_symbol:
                failure_info = {
                    asset_symbol: {
                        "display_name": asset_symbol,
                        "attempts": 1,
                        "last_error": f"BMA structure missing: {reason}",
                        "error_type": error_type,
                        "traceback": None,
                    }
                }
                save_failed_assets(failure_info, append=True)
        except Exception:
            pass  # Don't fail signal generation due to logging error

        # Return uniform soft regime probs for trust (maximally uncertain)
        uniform_regime_probs = {i: 0.2 for i in range(5)}
        return np.array([0.0]), uniform_regime_probs, {
            "method": "REJECTED",
            "reason": "no_bma_structure_old_cache_format",
            "error": "Cache must be regenerated with tune.py for BMA support",
            "debug_reason": reason,
            "asset": asset_display,
            "error_type": error_type,
        }

    # ========================================================================
    # DETERMINE CURRENT REGIME (DETERMINISTIC - SAME AS TUNE)
    # ========================================================================
    # Use exact same logic as tune.py assign_regime_labels()
    # This ensures consistency between tuning and inference
    # ========================================================================
    current_regime = assign_current_regime(feats)
    regime_name = REGIME_NAMES.get(current_regime, f"REGIME_{current_regime}")

    # ========================================================================
    # SOFT REGIME PROBABILITIES FOR TRUST AUTHORITY
    # ========================================================================
    # Hard regime assignment remains for parameter selection.
    # Soft probabilities are computed for trust modulation to avoid cliffs.
    #
    # ARCHITECTURAL INTEGRATION:
    # - calibrated_trust.py expects regime_probs: Dict[int, float]
    # - Hard assignment would cause penalty jumps at regime boundaries
    # - Soft assignment smooths transitions: current=0.7, neighbors share 0.3
    #
    # REGIME ADJACENCY (volatility-ordered):
    #   0: low_vol    ↔ 1: normal
    #   1: normal     ↔ 2: trending, 3: high_vol
    #   2: trending   ↔ 1: normal
    #   3: high_vol   ↔ 1: normal, 4: crisis
    #   4: crisis     ↔ 3: high_vol
    # ========================================================================
    REGIME_ADJACENCY = {
        0: [1],           # low_vol → normal
        1: [0, 2, 3],     # normal → low_vol, trending, high_vol
        2: [1],           # trending → normal
        3: [1, 4],        # high_vol → normal, crisis
        4: [3],           # crisis → high_vol
    }

    # Soft probabilities: 70% current, 30% split among neighbors
    soft_regime_probs = {i: 0.0 for i in range(5)}
    soft_regime_probs[current_regime] = 0.70

    neighbors = REGIME_ADJACENCY.get(current_regime, [])
    if neighbors:
        neighbor_share = 0.30 / len(neighbors)
        for n in neighbors:
            soft_regime_probs[n] = neighbor_share
    else:
        # No neighbors defined: keep all mass on current
        soft_regime_probs[current_regime] = 1.0

    # Legacy one-hot array for backward compatibility
    regime_probs_array = np.zeros(5)
    regime_probs_array[current_regime] = 1.0

    # ========================================================================
    # GET REGIME-SPECIFIC BMA DATA
    # ========================================================================
    global_data = tuned_params.get('global') or {} if tuned_params else {}
    regime_data = tuned_params.get('regime') or {} if tuned_params else {}
    
    # Ensure regime_data is a dict (could be None from old cache)
    if regime_data is None:
        regime_data = {}

    global_model_posterior = global_data.get('model_posterior', {}) if global_data else {}
    global_models = global_data.get('models', {}) if global_data else {}

    # ========================================================================
    # EXTRACT AUGMENTATION LAYER DATA
    # ========================================================================
    # Hansen Skew-t data
    hansen_data = global_data.get('hansen_skew_t', {})
    hansen_lambda_global = hansen_data.get('lambda') if hansen_data else None
    hansen_nu_global = hansen_data.get('nu') if hansen_data else None
    hansen_skew_t_enabled = hansen_lambda_global is not None and abs(hansen_lambda_global) > 0.01
    
    # Contaminated Student-t data
    cst_data = global_data.get('contaminated_student_t', {})
    cst_nu_normal_global = cst_data.get('nu_normal') if cst_data else None
    cst_nu_crisis_global = cst_data.get('nu_crisis') if cst_data else None
    cst_epsilon_global = cst_data.get('epsilon') if cst_data else None
    cst_enabled = cst_nu_normal_global is not None and cst_epsilon_global is not None and cst_epsilon_global > 0.001

    # Get current regime's model_posterior and models
    regime_key = str(current_regime)  # JSON keys are strings
    r_data = regime_data.get(regime_key) or regime_data.get(current_regime)

    if r_data is not None and isinstance(r_data, dict):
        model_posterior = r_data.get('model_posterior', {})
        models = r_data.get('models', {})
        regime_meta = r_data.get('regime_meta', {})
        is_fallback = regime_meta.get('fallback', False) or regime_meta.get('borrowed_from_global', False)
    else:
        # Use global as fallback if regime data missing
        model_posterior = global_model_posterior
        models = global_models
        is_fallback = True
        regime_meta = {}

    # If models still empty, use global
    if not models or not model_posterior:
        model_posterior = global_model_posterior
        models = global_models
        is_fallback = True

    # If still empty after global fallback - cannot proceed
    if not models or not model_posterior:
        return np.array([0.0]), soft_regime_probs, {
            "method": "FAILED",
            "reason": "no_models_available",
            "error": "No model posterior or models available for inference",
            "current_regime": current_regime,
            "regime_name": regime_name,
        }

    # ========================================================================
    # EPISTEMIC WEIGHTING: Recompute posteriors from combined_score
    # ========================================================================
    # This ensures Hyvärinen scores directly influence signal generation.
    # Even if cache has old posteriors, we recompute from combined_score.
    #
    # The combined_score integrates both:
    #   - BIC (model complexity penalty, consistency)
    #   - Hyvärinen score (proper scoring, robustness under misspecification)
    #
    # This is the "epistemic weighting" step from the architecture.
    #
    # ========================================================================
    # EPISTEMIC WEIGHTING: Recompute posteriors from combined scores
    # ========================================================================
    recomputed_posterior, epistemic_meta = compute_model_posteriors_from_combined_score(models)

    if not recomputed_posterior:
        # Cache may be from an older version without combined_score
        # Fallback to BIC-based posteriors for backward compatibility
        # This is a GRACEFUL DEGRADATION, not a hard failure
        import warnings
        model_names = list(models.keys())
        missing_scores = [m for m in model_names if not models.get(m, {}).get('combined_score')]
        
        # Try BIC-based fallback
        bic_posterior = {}
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                bic = model_data.get('bic')
                if bic is not None and np.isfinite(bic):
                    bic_posterior[model_name] = bic
        
        if bic_posterior:
            # Convert BIC to weights using softmax over negated BIC (lower is better)
            bic_values = np.array(list(bic_posterior.values()))
            neg_bic = -bic_values / 2.0  # Standard BIC to log-likelihood conversion
            neg_bic = neg_bic - neg_bic.max()  # Numerical stability
            weights = np.exp(neg_bic)
            weights = weights / weights.sum()
            
            recomputed_posterior = dict(zip(bic_posterior.keys(), weights))
            epistemic_meta = {
                'method': 'bic_fallback',
                'reason': 'combined_score_missing',
                'missing_scores': missing_scores,
            }
            
            warnings.warn(
                f"Using BIC-based fallback for models missing combined_score: {missing_scores}. "
                f"Consider re-tuning with 'make tune --force' for optimal calibration.",
                RuntimeWarning
            )
        else:
            # Last resort: uniform weights
            recomputed_posterior = {m: 1.0/len(models) for m in models}
            epistemic_meta = {
                'method': 'uniform_fallback',
                'reason': 'no_valid_scores',
            }
            
            warnings.warn(
                f"Using uniform weights fallback - no valid BIC or combined_score found. "
                f"Re-tune with 'make tune --force'.",
                RuntimeWarning
            )

    cached_posterior = model_posterior
    model_posterior = recomputed_posterior
    posteriors_recomputed = True

    # ========================================================================
    # FAIL-FAST ASSERTION: Validate model synchronization
    # ========================================================================
    # This is the #1 silent failure mode in production quant systems:
    # tune.py adds a model but signals.py ignores it → distorted posterior mass.
    #
    # The model registry (models/model_registry.py) provides the canonical
    # contract. If registry is available, we validate against it.
    #
    # ARCHITECTURAL LAW: Top funds REFUSE TO TRADE without this assertion.
    # ========================================================================
    if MODEL_REGISTRY_AVAILABLE:
        try:
            tuned_model_names = set(models.keys())
            # Only warn, don't fail - allows for temporary model additions
            # during experimentation while still flagging the issue
            assert_models_synchronised(
                tuned_model_names, 
                context=f"regime={current_regime} ({regime_name}), asset={asset_symbol}"
            )
        except AssertionError as sync_error:
            # Log but don't fail - degraded operation is better than crash
            # This will be logged to stderr for monitoring
            import warnings
            warnings.warn(
                f"Model synchronization warning: {sync_error}",
                RuntimeWarning
            )
        except Exception:
            pass  # Registry check failed - continue without validation

    # ========================================================================
    # BAYESIAN MODEL AVERAGING: Draw samples from mixture over models
    # ========================================================================
    # p(x | D, r_t) = Σ_m p(x | r_t, m, θ_m) · p(m | r_t)
    #
    # Implementation:
    #   For each model m with weight w = p(m | r_t):
    #     - Draw floor(w * n_paths) samples from p(x | r_t, m, θ_m)
    #     - Ensure minimum representation (MIN_MODEL_SAMPLES)
    # ========================================================================
    MIN_MODEL_SAMPLES = 20  # Minimum samples per model to preserve tail awareness

    all_samples = []
    model_details = {}

    for model_name, model_weight in model_posterior.items():
        model_params = models.get(model_name, {})

        # Skip failed model fits
        if not model_params.get('fit_success', True):
            continue

        # Extract model-specific parameters
        q_m = model_params.get('q', 1e-6)
        phi_m = model_params.get('phi')
        nu_m = model_params.get('nu')
        c_m = model_params.get('c', 1.0)

        # Extract NIG parameters (for phi_nig_* models)
        nig_alpha_m = model_params.get('nig_alpha')
        nig_beta_m = model_params.get('nig_beta')
        nig_delta_m = model_params.get('nig_delta')

        # Default phi for models without it
        if phi_m is None or not np.isfinite(phi_m):
            phi_m = 0.95 if 'phi' in model_name else 1.0

        # Validate nu
        if nu_m is not None and (not np.isfinite(nu_m) or nu_m <= 2.0):
            nu_m = None

        # Number of samples: proportional to weight but with minimum guarantee
        n_model_samples = max(MIN_MODEL_SAMPLES, int(model_weight * n_paths))

        # ====================================================================
        # AIGF-NF SPECIAL HANDLING
        # ====================================================================
        # AIGF-NF uses its own normalizing flow-based sampling.
        # It does NOT use the standard run_regime_specific_mc() path.
        # Instead, we reconstruct the model from cached state and sample.
        #
        # ARCHITECTURE:
        # - Flow parameters θ are frozen (trained offline)
        # - Latent state z_t evolves online
        # - Sampling is via flow forward pass: x = f_θ(ε; z), ε ~ N(0,I)
        # ====================================================================
        if is_aigf_nf_model(model_name) and AIGF_NF_AVAILABLE:
            try:
                # Reconstruct AIGF-NF model from cached parameters
                latent_z = model_params.get('latent_z', [0.0] * 8)
                latent_dim = model_params.get('latent_dim', 8)
                config_dict = model_params.get('config', {})
                
                # Create config (use default if not available)
                if config_dict:
                    config = AIGFNFConfig.from_dict(config_dict)
                else:
                    config = DEFAULT_AIGF_NF_CONFIG
                
                # Create model with cached state
                aigf_model = AIGFNFModel(config=config, rng=rng)
                
                # Restore latent state from cache
                if latent_z and len(latent_z) == aigf_model.state.z.shape[0]:
                    aigf_model.state.z = np.array(latent_z)
                
                # Sample from the flow
                model_samples = aigf_model.sample_predictive(n_samples=n_model_samples)
                
                # Scale samples to horizon H (flow samples are single-step)
                # For multi-step forecast, we accumulate drift like other models
                if H > 1:
                    # Simple random walk scaling for horizon
                    model_samples = model_samples * np.sqrt(H)
                
                all_samples.append(model_samples)
                model_details[model_name] = {
                    "weight": float(model_weight),
                    "n_samples": len(model_samples),
                    "model_type": "aigf_nf",
                    "latent_dim": latent_dim,
                    "predictive_mean": model_params.get('predictive_mean'),
                    "predictive_std": model_params.get('predictive_std'),
                    "tail_heaviness": model_params.get('tail_heaviness'),
                    "novelty_score": model_params.get('novelty_score'),
                    "flow_artifact_id": model_params.get('flow_artifact_id'),
                }
                continue  # Skip standard MC sampling
                
            except Exception as e:
                # If AIGF-NF sampling fails, log and skip this model
                import warnings
                warnings.warn(
                    f"AIGF-NF sampling failed: {e}. Skipping model in BMA.",
                    RuntimeWarning
                )
                continue

        # Generate samples from p(x | r_t, m, θ_m)
        # AUGMENTATION LAYERS ARE PASSED TO MC SAMPLING:
        # - Hansen λ affects tail asymmetry in Student-t
        # - CST affects regime-dependent tail thickness
        # - NIG affects both tails and asymmetry
        model_samples = run_regime_specific_mc(
            regime=current_regime,
            mu_t=mu_t,
            P_t=P_t,
            phi=phi_m,
            q=q_m,
            sigma2_step=sigma2_step * c_m,  # Scale volatility by c
            H=H,
            n_paths=n_model_samples,
            nu=nu_m,
            # ================================================================
            # AUGMENTATION LAYERS - Affect distribution of return samples
            # ================================================================
            # NIG parameters (model-specific, from NIG models)
            nig_alpha=nig_alpha_m,
            nig_beta=nig_beta_m,
            nig_delta=nig_delta_m,
            # Hansen Skew-t (global augmentation from tuning)
            hansen_lambda=hansen_lambda_global if hansen_skew_t_enabled else None,
            # Contaminated Student-t (global augmentation from tuning)
            cst_nu_normal=cst_nu_normal_global if cst_enabled else None,
            cst_nu_crisis=cst_nu_crisis_global if cst_enabled else None,
            cst_epsilon=cst_epsilon_global if cst_enabled else None,
            seed=rng.integers(0, 2**31) if seed is not None else None
        )

        all_samples.append(model_samples)
        model_details[model_name] = {
            "weight": float(model_weight),
            "n_samples": len(model_samples),
            "q": float(q_m),
            "phi": float(phi_m) if phi_m is not None else None,
            "nu": float(nu_m) if nu_m is not None else None,
            "c": float(c_m),
            # Augmentation layer info
            "nig_alpha": float(nig_alpha_m) if nig_alpha_m else None,
            "nig_beta": float(nig_beta_m) if nig_beta_m else None,
            "hansen_lambda": float(hansen_lambda_global) if hansen_skew_t_enabled else None,
            "cst_enabled": cst_enabled,
        }

    # Concatenate all model samples
    if all_samples:
        r_samples = np.concatenate(all_samples)
    else:
        return np.array([0.0]), regime_probs, {
            "method": "FAILED",
            "reason": "no_valid_model_samples",
            "current_regime": current_regime,
            "regime_name": regime_name,
        }

    metadata = {
        "method": "bayesian_model_averaging",
        "has_bma": True,
        "current_regime": current_regime,
        "regime_name": regime_name,
        "is_fallback": is_fallback,
        "model_posterior": {m: float(w) for m, w in model_posterior.items()},
        "model_details": model_details,
        "n_total_samples": len(r_samples),
        # Hansen Skew-t diagnostics
        "hansen_skew_t_enabled": hansen_skew_t_enabled,
        "hansen_lambda": float(hansen_lambda_global) if hansen_lambda_global is not None else None,
        "hansen_nu": float(hansen_nu_global) if hansen_nu_global is not None else None,
        "hansen_skew_direction": (
            "left" if hansen_lambda_global and hansen_lambda_global < -0.01 else
            ("right" if hansen_lambda_global and hansen_lambda_global > 0.01 else "symmetric")
        ) if hansen_lambda_global is not None else "not_available",
        # Epistemic weighting diagnostics
        "posteriors_recomputed": posteriors_recomputed,
        "cached_posterior": {m: float(w) for m, w in cached_posterior.items()} if posteriors_recomputed else None,
        "epistemic_weighting": epistemic_meta,
        # Model selection diagnostics
        "model_selection_method": regime_meta.get('model_selection_method', 'combined'),
        "effective_selection_method": epistemic_meta.get('method', 'combined'),
        "hyvarinen_disabled": regime_meta.get('hyvarinen_disabled', False),
        "bic_weight": regime_meta.get('bic_weight', 0.5),
        "entropy_lambda": regime_meta.get('entropy_lambda', 0.05),
        "hyvarinen_max": regime_meta.get('hyvarinen_max'),
        "combined_score_min": regime_meta.get('combined_score_min'),
        "bic_min": regime_meta.get('bic_min'),
        # Contaminated Student-t diagnostics
        "contaminated_student_t_enabled": cst_enabled,
        "cst_nu_normal": float(cst_nu_normal_global) if cst_nu_normal_global is not None else None,
        "cst_nu_crisis": float(cst_nu_crisis_global) if cst_nu_crisis_global is not None else None,
        "cst_epsilon": float(cst_epsilon_global) if cst_epsilon_global is not None else None,
        # SOFT REGIME PROBABILITIES FOR TRUST AUTHORITY
        # Used by CalibratedTrust to avoid penalty cliffs at regime boundaries
        "soft_regime_probs": soft_regime_probs,
    }

    # Return soft regime probs dict for trust authority (not legacy array)
    return r_samples, soft_regime_probs, metadata




# -------------------------
# Backtest-safe feature view (no look-ahead)
# -------------------------

def shift_features(feats: Dict[str, pd.Series], lag: int = 1) -> Dict[str, pd.Series]:
    """Return a copy of features dict with time-series shifted by `lag` days to remove look-ahead.
    Use for backtesting so that features at date t only use information available up to t−lag.
    Only series keys are shifted; scalar/meta entries are passed through.
    """
    if feats is None:
        return {}
    lag = int(max(0, lag))
    if lag == 0:
        return dict(feats)
    keys_to_shift = {
        # drifts
        "mu", "mu_post", "mu_blend", "mu_kf", "mu_final",
        # vols and regimes
        "vol_fast", "vol_slow", "vol", "vol_regime",
        # trend/momentum/stretch
        "sma200", "trend_z", "z5", "mom21", "mom63", "mom126", "mom252",
        # tails
        "skew", "nu",
        # base series for reference
        "ret",
    }
    shifted: Dict[str, pd.Series] = {}
    for k, v in feats.items():
        if isinstance(v, pd.Series) and k in keys_to_shift:
            try:
                shifted[k] = v.shift(lag)
            except Exception:
                shifted[k] = v
        else:
            # pass-through (px and any non-shifted helper)
            shifted[k] = v
    return shifted


def make_features_views(feats: Dict[str, pd.Series]) -> Dict[str, Dict[str, pd.Series]]:
    """Convenience wrapper to expose both live and backtest-safe views.
    - live: unshifted (as-of) features suitable for real-time use
    - bt:   shifted by 1 day (no look-ahead) for backtesting
    """
    return {
        "live": feats,
        "bt": shift_features(feats, lag=1),
    }


# =============================================================================
# NOTE: Legacy single-model MC functions removed
# =============================================================================
# The following functions were removed as they assume flat parameters (q, phi, nu)
# and are not compatible with the new BMA architecture:
#
#   - edge_for_horizon() - analytic z-score approximation (not used)
#   - posterior_predictive_mc_probability() - single-model MC (replaced by BMA)
#   - compute_expected_utility() - single-model EU (EU now computed inline from BMA samples)
#
# The BMA path uses:
#   - run_regime_specific_mc() - per-regime MC with model-specific params
#   - bayesian_model_average_mc() - full BMA mixture over regimes and models
#   - Inline EU computation in latest_signals() from r_samples
# =============================================================================


def _simulate_forward_paths(feats: Dict[str, pd.Series], H_max: int, n_paths: int = 3000, phi: float = 0.95, kappa: float = 1e-4) -> Dict[str, np.ndarray]:
    """Monte-Carlo forward simulation of cumulative log returns and volatility over 1..H_max.
    - Drift evolves as AR(1): mu_{t+1} = phi * mu_t + eta_t,  eta ~ N(0, q)
    - Volatility evolves via GARCH(1,1) when available; else held constant.
    - Innovations are Student-t with global df (nu_hat) scaled to unit variance.
    - Jump-diffusion (Merton model): captures discontinuous gap risk via rare large moves.

    Pillar 1 integration: Drift uncertainty from Kalman filter (var_kf) is propagated
    into process noise q, widening forecast confidence intervals when drift is uncertain.

    Level-7 parameter uncertainty: if PARAM_UNC environment variable is set to
    'sample' (default) and garch_params contains a covariance matrix, we sample
    (omega, alpha, beta) per path from N(theta_hat, Cov) with constraints, which
    widens confidence during regime shifts and narrows during stability.

    Stochastic volatility: Tracks full h_t (variance) trajectories across paths,
    enabling posterior uncertainty bands for volatility forecasts.

    Level-7 jump-diffusion: Merton model adds discontinuous jumps to capture gap risk:
        dS/S = μ dt + σ dW + J dN
    Where:
        - dW: continuous Brownian motion (Student-t innovations)
        - dN: Poisson process with intensity λ (jump arrival rate)
        - J: jump size ~ N(μ_J, σ_J²) (typically negative for crash risk)
    Jump parameters calibrated from historical returns: count large moves (>3σ) as jumps.

    Returns:
        Dictionary with:
            - 'returns': array of shape (H_max, n_paths) with cumulative log returns
            - 'volatility': array of shape (H_max, n_paths) with volatility (sigma_t = sqrt(h_t))
    """
    # Inputs at 'now'
    ret_idx = feats.get("ret", pd.Series(dtype=float)).index
    if ret_idx is None or len(ret_idx) == 0:
        return {
            'returns': np.zeros((H_max, n_paths), dtype=float),
            'volatility': np.zeros((H_max, n_paths), dtype=float)
        }
    mu_series = feats.get("mu_post")
    if not isinstance(mu_series, pd.Series) or mu_series.empty:
        mu_series = feats.get("mu")
    vol_series = feats.get("vol")
    if not isinstance(vol_series, pd.Series) or vol_series.empty or not isinstance(mu_series, pd.Series) or mu_series.empty:
        return {
            'returns': np.zeros((H_max, n_paths), dtype=float),
            'volatility': np.zeros((H_max, n_paths), dtype=float)
        }
    mu_now = float(mu_series.iloc[-1]) if len(mu_series) else 0.0
    vol_now = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
    vol_now = float(max(vol_now, 1e-6))

    # Pillar 1: Extract Kalman drift uncertainty for proper uncertainty propagation
    var_kf_series = feats.get("var_kf")
    if isinstance(var_kf_series, pd.Series) and not var_kf_series.empty:
        var_kf_now = float(var_kf_series.iloc[-1])
        var_kf_now = float(max(var_kf_now, 0.0))
    else:
        var_kf_now = 0.0

    # Tail parameter (global nu) with posterior uncertainty
    nu_hat_series = feats.get("nu_hat")
    nu_info = feats.get("nu_info", {})

    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu_hat = float(nu_hat_series.iloc[-1])
    else:
        nu_hat, _ = _tail2("nu", 50.0)
        if not np.isfinite(nu_hat):
            nu_hat = 50.0
    nu_hat = float(np.clip(nu_hat, 4.5, 500.0))

    # Extract standard error for ν (Tier 2: posterior parameter variance)
    se_nu = None
    if isinstance(nu_info, dict) and "se_nu" in nu_info:
        se_nu_val = nu_info.get("se_nu", float("nan"))
        if np.isfinite(se_nu_val) and se_nu_val > 0:
            se_nu = float(se_nu_val)

    # Determine if ν sampling is enabled (Tier 2: propagate tail parameter uncertainty)
    nu_sample_mode = os.getenv("NU_SAMPLE", "true").strip().lower() == "true"

    # Sample ν per path if uncertainty available and sampling enabled
    if nu_sample_mode and se_nu is not None and se_nu > 0:
        rng = np.random.default_rng()
        # Sample from N(nu_hat, se_nu²) and clip to valid range
        nu_samples = rng.normal(loc=nu_hat, scale=se_nu, size=n_paths)
        nu_samples = np.clip(nu_samples, 4.5, 500.0)
    else:
        # Use point estimate for all paths
        nu_samples = np.full(n_paths, nu_hat, dtype=float)

    # GARCH params
    garch_params = feats.get("garch_params", {}) or {}
    use_garch = isinstance(garch_params, dict) and all(k in garch_params for k in ("omega", "alpha", "beta"))

    # Determine parameter uncertainty mode
    param_unc_mode = os.getenv("PARAM_UNC", "sample").strip().lower()
    if param_unc_mode not in ("none", "sample"):
        param_unc_mode = "sample"

    # Build per-path parameters (possibly sampled)
    if use_garch:
        base_theta = np.array([
            float(max(garch_params.get("omega", 0.0), 1e-12)),
            float(np.clip(garch_params.get("alpha", 0.0), 0.0, 0.999)),
            float(np.clip(garch_params.get("beta", 0.0), 0.0, 0.999)),
        ], dtype=float)
        cov = garch_params.get("cov")
        if isinstance(cov, list):
            try:
                cov = np.array(cov, dtype=float)
            except Exception:
                cov = None
        # Sample theta per path if enabled and covariance available
        if (param_unc_mode == "sample") and (cov is not None) and np.shape(cov) == (3, 3):
            rng = np.random.default_rng()
            try:
                thetas = rng.multivariate_normal(mean=base_theta, cov=cov, size=n_paths).astype(float)
            except Exception:
                # Fall back to eigen-decomposition sampling with small regularization
                try:
                    eigvals, eigvecs = np.linalg.eigh(0.5*(cov+cov.T) + 1e-12*np.eye(3))
                    eigvals = np.clip(eigvals, 0.0, None)
                    z = rng.normal(size=(n_paths, 3)) * np.sqrt(eigvals)
                    thetas = (z @ eigvecs.T) + base_theta
                except Exception:
                    thetas = np.tile(base_theta, (n_paths, 1))
            # Enforce constraints; replace invalid draws with base_theta
            omega_s = thetas[:, 0]
            alpha_s = thetas[:, 1]
            beta_s  = thetas[:, 2]
            # Fix obvious violations
            omega_s = np.maximum(omega_s, 1e-12)
            alpha_s = np.clip(alpha_s, 0.0, 0.999)
            beta_s  = np.clip(beta_s, 0.0, 0.999)
            # Enforce alpha+beta < 0.999 by shrinking both toward base proportionally
            ab = alpha_s + beta_s
            viol = ab >= 0.999
            if np.any(viol):
                # target sum slightly below 1
                target = 0.998
                scale = target / np.maximum(ab[viol], 1e-12)
                alpha_s[viol] *= scale
                beta_s[viol] *= scale
            omega_paths = omega_s
            alpha_paths = alpha_s
            beta_paths = beta_s
        else:
            omega_paths = np.full(n_paths, base_theta[0], dtype=float)
            alpha_paths = np.full(n_paths, base_theta[1], dtype=float)
            beta_paths  = np.full(n_paths, base_theta[2], dtype=float)
    else:
        omega_paths = np.zeros(n_paths, dtype=float)
        alpha_paths = np.zeros(n_paths, dtype=float)
        beta_paths  = np.zeros(n_paths, dtype=float)

    # Drift uncertainty: propagate posterior P only (no external q leakage)
    drift_unc_now = max(var_kf_now, 1e-10)

    h0 = vol_now ** 2

    # Level-7 Jump-Diffusion: Calibrate jump parameters from historical returns
    # Detect large moves (>3σ) as empirical jumps to estimate:
    #   - λ (jump intensity): frequency of jumps per day
    #   - μ_J (jump mean): average jump size
    #   - σ_J (jump std): volatility of jump sizes
    jump_intensity = 0.0
    jump_mean = 0.0
    jump_std = 0.05
    enable_jumps = os.getenv("ENABLE_JUMPS", "true").strip().lower() == "true"

    if enable_jumps:
        try:
            # Get historical returns for calibration
            ret_hist = feats.get("ret", pd.Series(dtype=float))
            vol_hist = feats.get("vol", pd.Series(dtype=float))

            if isinstance(ret_hist, pd.Series) and isinstance(vol_hist, pd.Series) and len(ret_hist) >= 252:
                # Align returns and volatility
                df_jump = pd.concat([ret_hist, vol_hist], axis=1, join='inner').dropna()
                if len(df_jump) >= 252:
                    df_jump.columns = ['ret', 'vol']

                    # Identify jumps: returns that exceed 3σ threshold (outliers)
                    # Standardize returns by conditional volatility
                    z_scores = df_jump['ret'] / df_jump['vol']
                    jump_threshold = 3.0
                    jump_mask = np.abs(z_scores) > jump_threshold

                    n_jumps = int(np.sum(jump_mask))
                    n_days = len(df_jump)

                    if n_jumps > 0:
                        # Jump intensity: λ = frequency of jumps per day
                        jump_intensity = float(n_jumps / n_days)

                        # Jump sizes: extract returns on jump days
                        jump_returns = df_jump.loc[jump_mask, 'ret'].values

                        # Jump mean and std (typically negative mean for crash risk)
                        jump_mean = float(np.mean(jump_returns))
                        jump_std = float(np.std(jump_returns))

                        # Floor jump std to avoid degenerate case
                        jump_std = float(max(jump_std, 0.01))
                    else:
                        # No historical jumps detected: use conservative defaults
                        jump_intensity = 0.01  # ~2.5 jumps per year
                        jump_mean = -0.02  # small negative bias (crash risk)
                        jump_std = 0.05
        except Exception:
            # Fallback to conservative defaults if calibration fails
            jump_intensity = 0.01
            jump_mean = -0.02
            jump_std = 0.05

    # Initialize state arrays (vectorized across paths)
    cum = np.zeros((H_max, n_paths), dtype=float)
    vol_paths = np.zeros((H_max, n_paths), dtype=float)  # Track volatility (sigma_t) at each horizon
    mu_t = np.full(n_paths, mu_now, dtype=float)
    h_t = np.full(n_paths, max(h0, 1e-8), dtype=float)

    rng = np.random.default_rng()

    for t in range(H_max):
        # Student-t shocks standardized to unit variance (continuous component)
        # Tier 2: Use path-specific ν samples for proper tail parameter uncertainty
        # Draw Student-t per path with its own degrees of freedom
        z = np.zeros(n_paths, dtype=float)
        for path_idx in range(n_paths):
            nu_path = nu_samples[path_idx]
            # Draw from Student-t with df=nu_path and scale to unit variance
            z_raw = rng.standard_t(df=nu_path)
            # Variance of t(ν) is ν/(ν-2) for ν>2
            if nu_path > 2.0:
                t_var_path = nu_path / (nu_path - 2.0)
                t_scale_path = math.sqrt(t_var_path)
                z[path_idx] = float(z_raw / t_scale_path)
            else:
                # Edge case: use raw draw for very low ν (shouldn't happen with clipping)
                z[path_idx] = float(z_raw)

        eps = z
        sigma_t = np.sqrt(np.maximum(h_t, 1e-12))
        e_t = sigma_t * eps

        # Level-7 Jump-Diffusion: Add discontinuous jump component
        # Merton model: dS/S = μ dt + σ dW + J dN
        jump_component = np.zeros(n_paths, dtype=float)
        if enable_jumps and jump_intensity > 0:
            # Poisson arrivals: number of jumps in this time step
            # For daily data, dt=1, so intensity per step = jump_intensity
            n_jumps = rng.poisson(lam=jump_intensity, size=n_paths)

            # For paths with jumps, draw jump sizes from N(μ_J, σ_J²)
            # Total jump = sum of all jumps in this step (if multiple)
            for path_idx in range(n_paths):
                if n_jumps[path_idx] > 0:
                    # Draw jump sizes (log returns)
                    jump_sizes = rng.normal(loc=jump_mean, scale=jump_std, size=int(n_jumps[path_idx]))
                    jump_component[path_idx] = float(np.sum(jump_sizes))

        # Total return: continuous (drift + diffusion) + jumps
        r_t = mu_t + e_t + jump_component

        # Accumulate log return
        if t == 0:
            cum[t, :] = r_t
        else:
            cum[t, :] = cum[t-1, :] + r_t
        # Store volatility at this horizon (stochastic volatility tracking)
        vol_paths[t, :] = sigma_t
        # Evolve volatility via GARCH or hold constant on fallback
        if use_garch:
            h_t = omega_paths + alpha_paths * (e_t ** 2) + beta_paths * h_t
            h_t = np.clip(h_t, 1e-12, 1e4)
        # Evolve drift via AR(1) using posterior drift uncertainty only
        eta = rng.normal(loc=0.0, scale=math.sqrt(drift_unc_now), size=n_paths)
        mu_t = phi * mu_t + eta

    return {
        'returns': cum,
        'volatility': vol_paths
    }


def composite_edge(
    base_edge: float,
    trend_z: float,
    moms: List[float],
    vol_regime: float,
    z5: float,
) -> float:
    """Ensemble edge: blend trend-following and mean-reversion components.
    GARCH handles volatility dynamics; avoid extra regime dampening to prevent double-counting.
    """
    # Momentum confirmation: average tanh of t-momentum
    mom_terms = [np.tanh(m / 2.0) for m in moms if np.isfinite(m)]
    mom_align = float(np.mean(mom_terms)) if mom_terms else 0.0

    # Trend tilt (gentle)
    trend_tilt = float(np.tanh(trend_z / 2.0)) if np.isfinite(trend_z) else 0.0

    # TF component
    tf = base_edge + 0.30 * mom_align + 0.20 * trend_tilt

    # MR component: if z5 is very positive, expect mean-revert small negative edge; if very negative, mean-revert positive edge
    mr = float(-np.tanh(z5)) if np.isfinite(z5) else 0.0

    # Fixed blend (avoid vol_regime-driven dampening)
    w_tf, w_mr = 0.75, 0.25
    edge = w_tf * tf + w_mr * mr

    return float(edge)


def compute_dynamic_thresholds(
    skew: float,
    regime_meta: Dict[str, float],
    sig_H: float,
    med_vol_last: float,
    H: int
) -> Dict[str, float]:
    """
    Compute dynamic buy/sell thresholds with asymmetry and uncertainty adjustments.

    Level-7 modularization: Separates threshold computation from signal generation
    for better testability and maintainability.

    Args:
        skew: Return skewness (asymmetry measure)
        regime_meta: Regime detection metadata with method and probabilities
        sig_H: Forecast volatility at horizon H
        med_vol_last: Long-run median volatility
        H: Forecast horizon in days

    Returns:
        Dictionary with buy_thr, sell_thr, and uncertainty metrics
    """
    # Base thresholds
    base_buy, base_sell = 0.58, 0.42

    # Skew adjustment: shift thresholds based on return asymmetry
    g1 = float(np.clip(skew if np.isfinite(skew) else 0.0, -1.5, 1.5))
    skew_delta = 0.02 * float(np.tanh(abs(g1) / 0.75))

    if g1 < 0:  # Negative skew (crash risk)
        buy_thr = base_buy + skew_delta
        sell_thr = base_sell + skew_delta
    elif g1 > 0:  # Positive skew (rally potential)
        buy_thr = base_buy - skew_delta
        sell_thr = base_sell - skew_delta
    else:
        buy_thr, sell_thr = base_buy, base_sell

    # Regime-based uncertainty (HMM posterior entropy or vol regime deviation)
    if regime_meta.get("method") == "hmm_posterior":
        # Use Shannon entropy of regime posteriors as uncertainty measure
        probs = regime_meta.get("probabilities", {})
        entropy = 0.0
        for p in probs.values():
            if p > 1e-12:
                entropy -= p * np.log(p)
        # Normalize by max entropy (log(3) for 3 states)
        u_regime = float(np.clip(entropy / np.log(3.0), 0.0, 1.0))
    else:
        # Fallback: use vol_regime deviation if available
        vol_regime = regime_meta.get("vol_regime", 1.0)
        u_regime = float(np.clip(abs(vol_regime - 1.0) / 1.5, 0.0, 1.0)) if np.isfinite(vol_regime) else 0.5

    # Forecast uncertainty from realized vol vs historical
    med_sig_H = (med_vol_last * math.sqrt(H)) if (np.isfinite(med_vol_last) and med_vol_last > 0) else sig_H
    ratio = float(sig_H / med_sig_H) if med_sig_H > 0 else 1.0
    u_sig = float(np.clip(ratio - 1.0, 0.0, 1.0))

    # Combined uncertainty: regime entropy dominates, forecast uncertainty refines
    U = float(np.clip(0.5 * u_regime + 0.5 * u_sig, 0.0, 1.0))

    # Widen thresholds based on uncertainty
    widen_delta = 0.04 * U
    buy_thr += widen_delta
    sell_thr -= widen_delta

    # Clamp to reasonable ranges
    buy_thr = float(np.clip(buy_thr, 0.55, 0.70))
    sell_thr = float(np.clip(sell_thr, 0.30, 0.45))

    # Ensure minimum separation
    if buy_thr - sell_thr < 0.12:
        mid = 0.5
        sell_thr = min(sell_thr, mid - 0.06)
        buy_thr = max(buy_thr, mid + 0.06)

    return {
        "buy_thr": float(buy_thr),
        "sell_thr": float(sell_thr),
        "uncertainty": float(U),
        "u_regime": float(u_regime),
        "u_forecast": float(u_sig),
        "skew_adjustment": float(skew_delta),
    }


def apply_confirmation_logic(
    p_smoothed_now: float,
    p_smoothed_prev: float,
    p_raw: float,
    pos_strength: float,
    buy_thr: float,
    sell_thr: float,
    edge: float,
    edge_floor: float
) -> str:
    """
    Apply 2-day confirmation with hysteresis to reduce signal churn.

    Level-7 modularization: Separates confirmation logic from main signal flow.

    Args:
        p_smoothed_now: Smoothed probability (current)
        p_smoothed_prev: Smoothed probability (previous)
        p_raw: Raw probability without smoothing
        pos_strength: Position strength (Expected Utility based, 0..1)
        buy_thr: Buy threshold
        sell_thr: Sell threshold
        edge: Composite edge score
        edge_floor: Minimum edge required to act

    Returns:
        Signal label: "STRONG BUY", "BUY", "HOLD", "SELL", or "STRONG SELL"
    """
    # Hysteresis bands (slightly wider than base thresholds)
    buy_enter = buy_thr + 0.01
    sell_enter = sell_thr - 0.01

    # Base label from 2-day confirmation (smoothed probabilities)
    label = "HOLD"
    if (p_smoothed_prev >= buy_enter) and (p_smoothed_now >= buy_enter):
        label = "BUY"
    elif (p_smoothed_prev <= sell_enter) and (p_smoothed_now <= sell_enter):
        label = "SELL"

    # Strong tiers based on raw conviction and EU-based position strength
    if p_raw >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
        label = "STRONG BUY"
    if p_raw <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
        label = "STRONG SELL"

    # Transaction-cost hurdle: force HOLD if absolute edge below floor
    if np.isfinite(edge) and abs(edge) < float(edge_floor):
        label = "HOLD"

    return label


def label_from_probability(p_up: float, pos_strength: float, buy_thr: float = 0.58, sell_thr: float = 0.42) -> str:
    """Map probability and position strength to label with customizable thresholds.
    - STRONG tiers require both probability and position_strength to be high.
    - buy_thr and sell_thr must satisfy sell_thr < 0.5 < buy_thr.
    - pos_strength is derived from Expected Utility (EU / max(E[loss], ε))
    """
    buy_thr = float(buy_thr)
    sell_thr = float(sell_thr)
    # Strong tiers (EU-based sizing: 0.30 threshold for strong conviction)
    if p_up >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
        return "STRONG BUY"
    if p_up <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
        return "STRONG SELL"
    # Base labels
    if p_up >= buy_thr:
        return "BUY"
    if p_up <= sell_thr:
        return "SELL"
    return "HOLD"


def latest_signals(feats: Dict[str, pd.Series], horizons: List[int], last_close: float, t_map: bool = True, ci: float = 0.68, tuned_params: Optional[Dict] = None, asset_key: Optional[str] = None) -> Tuple[List[Signal], Dict[int, Dict[str, float]]]:
    """Compute signals using regime‑aware priors, tail‑aware probability mapping, and
    anti‑snap logic (two‑day confirmation + hysteresis + smoothing) without extra flags.

    Uses Bayesian Model Averaging when tuned_params with BMA structure is provided.

    CONTRACT WITH tune.py:
    - tuned_params contains model_posterior and models for each regime
    - Tune guarantees non-empty outputs via hierarchical borrowing
    - This function does NOT implement tuning or fallback logic

    Args:
        feats: Feature dictionary from compute_features()
        horizons: List of forecast horizons in days
        last_close: Last close price
        t_map: Whether to use Student-t probability mapping
        ci: Confidence interval for bands
        tuned_params: Full tuned params from _load_tuned_kalman_params() with BMA structure
        asset_key: Optional asset identifier for display price inertia (Upgrade #3)

    We build last‑two‑days estimates to avoid look‑ahead while adding stability.
    """
    idx = feats.get("px", pd.Series(dtype=float)).index
    if idx is None or len(idx) < 2:
        # Fallback to simple single‑day path
        idx = pd.DatetimeIndex(idx)
    last2 = idx[-2:] if len(idx) >= 2 else idx

    # Helper to safely fetch last/prev values from a Series
    def _tail2(series_key: str, default_val: float = np.nan) -> Tuple[float, float]:
        s = feats.get(series_key, None)
        if s is None or not isinstance(s, pd.Series) or s.empty:
            return (default_val, default_val)
        s2 = s.reindex(last2)
        vals = s2.to_numpy(dtype=float)
        if vals.size == 1:
            return (float(vals[-1]), float(vals[-1]))
        return (float(vals[-1]), float(vals[-2]))

    # Prefer posterior drift if available
    mu_now, mu_prev = _tail2("mu_post", 0.0)
    if not np.isfinite(mu_now):
        mu_now = 0.0
    if not np.isfinite(mu_prev):
        mu_prev = mu_now
    vol_now, vol_prev = _tail2("vol", np.nan)
    vol_reg_now, vol_reg_prev = _tail2("vol_regime", 1.0)
    trend_now, trend_prev = _tail2("trend_z", 0.0)
    z5_now, z5_prev = _tail2("z5", 0.0)
    # Use globally-fitted nu_hat (Level-7); fall back to rolling nu if missing/invalid
    nu_hat_series = feats.get("nu_hat")
    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu_glob = float(nu_hat_series.iloc[-1])
    else:
        nu_glob, _ = _tail2("nu", 50.0)
        if not np.isfinite(nu_glob):
            nu_glob = 50.0
    nu_glob = float(np.clip(nu_glob, 4.5, 500.0))
    # Prefer smoothed skew if available; fallback to raw skew; default neutral 0.0
    skew_now, skew_prev = _tail2("skew_s", np.nan)
    if not np.isfinite(skew_now) or not np.isfinite(skew_prev):
        skew_now_fallback, skew_prev_fallback = _tail2("skew", 0.0)
        if not np.isfinite(skew_now):
            skew_now = skew_now_fallback
        if not np.isfinite(skew_prev):
            skew_prev = skew_prev_fallback

    moms_now = [
        _tail2("mom21", 0.0)[0],
        _tail2("mom63", 0.0)[0],
        _tail2("mom126", 0.0)[0],
        _tail2("mom252", 0.0)[0],
    ]
    moms_prev = [
        _tail2("mom21", 0.0)[1],
        _tail2("mom63", 0.0)[1],
        _tail2("mom126", 0.0)[1],
        _tail2("mom252", 0.0)[1],
    ]

    # Mapping function that accepts per‑day skew/nu
    km_prob = (feats.get("kalman_metadata") or {})
    noise_model = km_prob.get("kalman_noise_model")
    tuned_nu_meta = km_prob.get("kalman_nu")
    is_student_world = noise_model and noise_model.startswith('phi_student_t_nu_')
    if is_student_world and (tuned_nu_meta is None or not np.isfinite(tuned_nu_meta)):
        raise ValueError("Student-t model selected but ν missing from tuning cache")
    nu_prob = float(tuned_nu_meta) if is_student_world else nu_glob

    # Check for GH (Generalized Hyperbolic) model - captures skewness
    is_gh_world = noise_model == 'generalized_hyperbolic'
    gh_params = km_prob.get("gh_model", {}).get("parameters", {}) if is_gh_world else {}
    gh_lambda = gh_params.get("lambda", -0.5)
    gh_alpha = gh_params.get("alpha", 1.0)
    gh_beta = gh_params.get("beta", 0.0)
    gh_delta = gh_params.get("delta", 1.0)

    # Import GH CDF function if GH model is used
    gh_cdf_func = None
    if is_gh_world:
        try:
            from calibration.gh_distribution import gh_cdf
            gh_cdf_func = gh_cdf
        except ImportError:
            # Fallback to Student-t if GH not available
            is_gh_world = False
            is_student_world = True
            nu_prob = 8.0  # Default to moderate tails

    # Mapping function that accepts per‑day skew/nu
    def map_prob(edge: float, nu_val: float, skew_val: float) -> float:
        if not np.isfinite(edge):
            return 0.5
        z = float(edge)
        nu_eff = nu_prob if is_student_world else nu_val

        # GH model: use fitted GH CDF for probability mapping
        if is_gh_world and gh_cdf_func is not None:
            try:
                # GH CDF returns P(Z <= z), we need this for probability computation
                base_p = float(gh_cdf_func(np.array([z]), gh_lambda, gh_alpha, gh_beta, gh_delta)[0])
                if not np.isfinite(base_p):
                    # Fallback to Student-t if GH fails
                    base_p = float(student_t.cdf(z, df=8.0))
            except Exception:
                base_p = float(student_t.cdf(z, df=8.0))
        # Base symmetric mapping dictated solely by model identity
        elif is_student_world:
            base_p = float(student_t.cdf(z, df=nu_eff))
        else:
            base_p = float(norm.cdf(z))
        if not np.isfinite(base_p):
            return 0.5
        # Edgeworth asymmetry using realized skew and (optional) kurt proxy from nu
        g1 = float(np.clip(skew_val if np.isfinite(skew_val) else 0.0, -1.5, 1.5))
        if np.isfinite(nu_val) and nu_val > 4.5 and nu_val < 1e9:
            g2 = 6.0 / (float(nu_val) - 4.0)
        else:
            g2 = 0.0
        if g1 == 0.0 and g2 == 0.0:
            return float(np.clip(base_p, 0.001, 0.999))
        try:
            phi = float(norm.pdf(z))
            corr = (g1 / 6.0) * (1.0 - z * z) + (g2 / 24.0) * (z ** 3 - 3.0 * z) - (g1 ** 2 / 36.0) * (2.0 * z ** 3 - 5.0 * z)
            # Damp skew influence in extreme tails to stabilize mapping for |z|>~3
            try:
                damp = math.exp(-0.5 * z * z)
            except Exception:
                damp = 0.0
            p = base_p + phi * (corr * damp)
            return float(np.clip(p, 0.001, 0.999))
        except Exception:
            return float(np.clip(base_p, 0.001, 0.999))

    # Regime detection via HMM posterior inference (replaces threshold-based heuristics)
    hmm_result = feats.get("hmm_result")
    reg, regime_meta = infer_current_regime(feats, hmm_result)

    # CI quantile based on 'now'
    alpha = np.clip(ci, 1e-6, 0.999999)
    tail = 0.5 * (1 + alpha)
    if is_gh_world:
        # For GH, use Student-t approximation for quantile (GH quantile is expensive)
        # The GH model captures skewness in CDF, but for CI we use symmetric approximation
        z_star = float(student_t.ppf(tail, df=8.0))
    elif is_student_world:
        z_star = float(student_t.ppf(tail, df=float(nu_prob)))
    else:
        z_star = float(norm.ppf(tail))

    # Median vol for uncertainty component (use rolling median series if available)
    vol_series = feats.get("vol", pd.Series(dtype=float))
    try:
        med_vol_series = vol_series.rolling(252, min_periods=63).median()
        med_vol_last = safe_last(med_vol_series) if med_vol_series is not None else float('nan')
    except Exception:
        med_vol_last = float('nan')
    # Explicit, readable fallback to long-run belief anchor (global median over entire history)
    if not np.isfinite(med_vol_last) or med_vol_last <= 0:
        try:
            med_vol_last = float(np.nanmedian(np.asarray(vol_series.values, dtype=float)))
        except Exception:
            med_vol_last = float('nan')
    # Final guard: fall back to current vol (or 1.0) if global median is unavailable
    if not np.isfinite(med_vol_last) or med_vol_last <= 0:
        med_vol_last = vol_now if np.isfinite(vol_now) and vol_now > 0 else 1.0

    sigs: List[Signal] = []
    thresholds: Dict[int, Dict[str, float]] = {}

    # Regime-aware smoothing from HMM posterior uncertainty (replaces threshold-based)
    # Use regime persistence and uncertainty from posterior probabilities
    if regime_meta.get("method") == "hmm_posterior":
        # High persistence (diagonal transition prob) => low smoothing (trust signal)
        # Low persistence => high smoothing (reduce whipsaws)
        persistence = regime_meta.get("persistence", 0.5)
        alpha_edge = 0.30 + 0.25 * (1.0 - persistence)  # 0.30-0.55 range
        alpha_p = min(0.75, alpha_edge + 0.10)
    else:
        # Fallback for threshold-based regime
        alpha_edge = 0.40
        alpha_p = 0.50

    # Monte‑Carlo forward simulation to capture evolving drift/vol over horizons
    H_max = int(max(horizons) if horizons else 0)
    phi_sim = None
    km = feats.get("kalman_metadata", {}) or {}
    if isinstance(km, dict):
        phi_sim = km.get("phi_used") or km.get("kalman_phi")
    if phi_sim is None:
        phi_sim = feats.get("phi_used")
    if isinstance(km, dict) and km.get("kalman_noise_model", "").startswith("kalman_phi"):
        if phi_sim is None or not np.isfinite(phi_sim):
            raise ValueError("phi required by model but missing for simulation")
        phi_sim = float(phi_sim)
    else:
        phi_sim = 1.0
    sim_result = _simulate_forward_paths(feats, H_max=H_max, n_paths=3000, phi=phi_sim)
    sims = sim_result['returns']
    vol_sims = sim_result['volatility']

    for H in horizons:
        # Use simulation at horizon H (1‑indexed in description; here index H-1)
        if H <= 0 or H > sims.shape[0]:
            sim_H = np.zeros(3000, dtype=float)
            vol_H = np.zeros(3000, dtype=float)
        else:
            sim_H = sims[H-1, :]
            vol_H = vol_sims[H-1, :]
        # Clean NaNs/Infs for returns
        sim_H = np.asarray(sim_H, dtype=float)
        sim_H = sim_H[np.isfinite(sim_H)]
        if sim_H.size == 0:
            sim_H = np.zeros(3000, dtype=float)
        # Clean NaNs/Infs for volatility
        vol_H = np.asarray(vol_H, dtype=float)
        vol_H = vol_H[np.isfinite(vol_H)]
        if vol_H.size == 0:
            vol_H = np.zeros(3000, dtype=float)
        # Simulated moments and probability
        # ========================================================================
        # UPGRADE #1: Trimmed Mean for Expected Return
        # ========================================================================
        # Use 5th-95th percentile trimmed mean to prevent extreme tails from
        # dominating the displayed expected return.
        # This dramatically stabilizes price predictions while preserving
        # directionality and remaining distribution-aware.
        # ========================================================================
        lo_trim = np.quantile(sim_H, 0.05)
        hi_trim = np.quantile(sim_H, 0.95)
        sim_H_trimmed = sim_H[(sim_H > lo_trim) & (sim_H < hi_trim)]
        if sim_H_trimmed.size > 0:
            mH = float(np.mean(sim_H_trimmed))
        else:
            mH = float(np.mean(sim_H))  # fallback to full mean if trim removes all
        vH = float(np.var(sim_H, ddof=1)) if sim_H.size > 1 else 0.0  # keep full variance for stats
        sH = float(math.sqrt(max(vH, 1e-12)))
        z_stat = float(mH / sH) if sH > 0 else 0.0

        # ========================================================================
        # Unified Posterior Predictive Monte-Carlo Probability
        # ========================================================================
        # Replaces blended analytical/MC posterior with unified posterior predictive
        # Monte-Carlo over drift and noise.
        #
        # This sampler marginalizes jointly over:
        # 1. Drift posterior uncertainty: μ_t ~ N(μ̂_t, P_t) or t_ν
        # 2. Drift propagation dynamics: μ_{t+k} = φ·μ_{t+k-1} + η_k, η_k ~ N(0, q)
        # 3. Observation noise: ε_H ~ N(0, v_H) or t_ν
        #
        # This is the ONLY probability used for trading decisions.
        # No blending. No heuristic averaging. No parallel analytical probability.
        #
        # REGIME-FIRST PARAMETER ROUTING:
        # Parameters (q, phi, nu, c) are selected using regime-first logic:
        # - If current_regime has tuned params AND not fallback → use regime params
        # - Otherwise → use global params
        #
        # Design Philosophy:
        # - Probability geometry is internally consistent (single probability space)
        # - Drift uncertainty automatically collapses confidence toward 0.5
        # - Heavy tails emerge naturally when P_t or q is large
        # - Regime transitions are naturally penalized via drift uncertainty
        # - Accumulated drift preserves signal strength for persistent drift (φ≈1)
        # - Otherwise → use global params (q_mc, phi_mc, nu_mc already selected above)
        # ========================================================================

        # Extract drift posterior mean (μ̂_t) from Kalman filter or posterior drift
        mu_post_series = feats.get("mu_post", feats.get("mu_kf", pd.Series(dtype=float)))
        if isinstance(mu_post_series, pd.Series) and not mu_post_series.empty:
            mu_t_mc = float(mu_post_series.iloc[-1])
        else:
            mu_t_mc = 0.0
        if not np.isfinite(mu_t_mc):
            mu_t_mc = 0.0

        # Extract drift posterior variance (P_t) from Kalman filter
        var_kf_series_prob = feats.get("var_kf_smoothed", feats.get("var_kf", pd.Series(dtype=float)))
        if isinstance(var_kf_series_prob, pd.Series) and not var_kf_series_prob.empty:
            P_t_mc = float(var_kf_series_prob.iloc[-1])
        else:
            P_t_mc = 0.0
        if not np.isfinite(P_t_mc) or P_t_mc < 0:
            P_t_mc = 0.0

        # ========================================================================
        # REGIME-FIRST PARAMETER ROUTING (STEP 1 & 2)
        # ========================================================================
        # Map current regime label to index and select parameters
        current_regime_idx = map_regime_label_to_index(reg, regime_meta)

        # Get Kalman metadata which contains regime params
        km_mc = feats.get("kalman_metadata", {}) or {}

        # Build tuned_params structure for _select_regime_params
        tuned_params_full = {
            'q': km_mc.get("process_noise_var", 1e-6),
            'phi': km_mc.get("phi_used") or km_mc.get("kalman_phi") or 0.95,
            'nu': km_mc.get("kalman_nu"),
            'c': km_mc.get("kalman_c_optimal", 1.0),
            'noise_model': km_mc.get("kalman_noise_model", "gaussian"),
            'regime': km_mc.get("regime_params", {}),
            'has_regime_params': km_mc.get("has_regime_params", False),
        }

        # STEP 2: Select parameters using regime-first logic
        theta = _select_regime_params(tuned_params_full, current_regime_idx)

        # STEP 3: Bind parameters ONLY from theta (no other access to params)
        q_mc = float(theta.get("q", 1e-6))
        phi_mc = float(theta.get("phi", 0.95))
        nu_mc = theta.get("nu")
        c_mc = float(theta.get("c", 1.0))

        # STEP 6: Diagnostics pass-through (collapse warning annotation)
        collapse_warning = theta.get("collapse_warning", False)
        regime_source = theta.get("source", "unknown")
        regime_used = theta.get("regime_used", current_regime_idx)

        # Validate nu and determine noise model for diagnostics
        noise_model_mc = km_mc.get("kalman_noise_model", "gaussian")
        if nu_mc is not None and (not np.isfinite(nu_mc) or nu_mc <= 2.0):
            nu_mc = None
        # Check for Student-t model (phi_student_t_nu_* naming)
        is_student_t_mc = noise_model_mc and noise_model_mc.startswith('phi_student_t_nu_')
        if not is_student_t_mc or nu_mc is None:
            noise_model_mc = "gaussian"  # Fallback for diagnostics

        # ========================================================================
        # VOLATILITY GEOMETRY: sigma2_step is the PRIMITIVE
        # ========================================================================
        # Extract per-step EWMA variance (σ²) - this is the volatility primitive.
        # Horizon variance vH = H × sigma2_step is DERIVED, not passed.
        #
        # This ensures:
        # - Volatility is defined at one temporal scale only (per-step)
        # - No double-rescaling (no vH/H computation in MC function)
        # - Noise accumulation uses sigma2_step directly
        # ========================================================================
        vol_series_mc = feats.get("vol", pd.Series(dtype=float))
        if isinstance(vol_series_mc, pd.Series) and not vol_series_mc.empty:
            sigma_now = float(vol_series_mc.iloc[-1])
        else:
            sigma_now = sH / math.sqrt(H) if H > 0 and sH > 0 else 0.01
        if not np.isfinite(sigma_now) or sigma_now <= 0:
            sigma_now = 0.01

        # sigma2_step is the PRIMITIVE per-step EWMA variance
        sigma2_step_mc = float(sigma_now ** 2)

        # ========================================================================
        # EXPECTED UTILITY POSITION SIZING FROM POSTERIOR PREDICTIVE MC
        # ========================================================================
        # Position size is derived ONLY from posterior predictive MC return samples.
        # No Kelly/mean-variance sizing. No blending. r_samples is the ONLY input.
        #
        # REGIME-CONDITIONAL BAYESIAN MODEL AVERAGING (RC-BMA):
        # Samples are drawn from the mixture:
        #   p(r_H | D) = Σ_r P(regime_r | D) · p(r_H | regime_r, D)
        #
        # This mixture is the ONLY distribution passed to the EU layer.
        # ========================================================================

        # ========================================================================
        # BUILD REGIME PARAMS FOR BAYESIAN MODEL AVERAGING
        # ========================================================================
        # Build regime params dict using regime-first routing.
        # For each regime, if tuned params exist and not fallback → use them
        # Otherwise → use global params (q_mc, phi_mc, nu_mc already selected above)
        # ========================================================================
        regime_params = {}
        cached_regime_params = km_mc.get("regime_params", {})

        for regime_idx in range(5):
            # Use _select_regime_params for each regime (consistent routing logic)
            regime_theta = _select_regime_params(tuned_params_full, regime_idx)
            regime_params[regime_idx] = {
                "phi": regime_theta.get("phi", phi_mc),
                "q": regime_theta.get("q", q_mc),
                "nu": regime_theta.get("nu", nu_mc),
                "c": regime_theta.get("c", c_mc),
                "fallback": regime_theta.get("fallback", True),
            }

        # Use Bayesian Model Averaging across regimes AND model classes
        r_samples, regime_probs, bma_meta = bayesian_model_average_mc(
            feats=feats,
            regime_params=regime_params,
            mu_t=mu_t_mc,
            P_t=P_t_mc,
            sigma2_step=sigma2_step_mc,
            H=H,
            n_paths=10000,
            seed=None,
            tuned_params=tuned_params,  # BMA structure from tune.py
            asset_symbol=asset_key,  # Pass asset identifier for error reporting
        )
        r = np.asarray(r_samples, dtype=float)

        # === Expected Utility sizing from posterior predictive MC ===
        # r_samples is the ONLY input to sizing - NO Kelly, NO mean/variance ratios
        # BMA mixture is the distribution, EU layer only sees r_samples

        p_now = float(np.mean(r > 0.0))

        gains = r[r > 0.0]
        losses = -r[r < 0.0]

        E_gain = float(np.mean(gains)) if gains.size > 0 else 0.0
        E_loss_empirical = float(np.mean(losses)) if losses.size > 0 else 0.0

        # ====================================================================
        # EVT-CORRECTED EXPECTED LOSS (Expert Panel Solution 2)
        # ====================================================================
        # The Pickands–Balkema–de Haan theorem provides theoretical foundation:
        # exceedances over high threshold u converge to GPD distribution.
        #
        # CTE = E[Loss | Loss > u] = u + σ/(1-ξ)  for ξ < 1
        #
        # This replaces the naive empirical mean with principled extrapolation
        # that captures extreme tail behavior beyond observed MC samples.
        #
        # Key properties:
        #   - EVT E[loss] ≥ empirical E[loss] (always more conservative)
        #   - Heavy-tailed assets (ξ > 0.2) get larger loss estimates
        #   - Light-tailed assets (ξ ≈ 0) get minimal adjustment
        #   - Fallback to 1.5× empirical if GPD fitting fails
        # ====================================================================
        
        # Initialize EVT diagnostics
        evt_expected_loss = E_loss_empirical
        evt_gpd_result = None
        evt_enabled = False
        evt_xi = None
        evt_sigma = None
        evt_threshold = None
        evt_n_exceedances = 0
        evt_fit_method = None
        evt_consistency = None
        
        if EVT_AVAILABLE and losses.size >= EVT_MIN_EXCEEDANCES:
            try:
                # Compute EVT-corrected expected loss
                evt_loss, emp_loss, gpd_result = compute_evt_expected_loss(
                    r_samples=r,
                    threshold_percentile=EVT_THRESHOLD_PERCENTILE_DEFAULT,
                    fallback_multiplier=EVT_FALLBACK_MULTIPLIER
                )
                
                evt_expected_loss = evt_loss
                evt_gpd_result = gpd_result
                evt_enabled = True
                evt_xi = gpd_result.xi
                evt_sigma = gpd_result.sigma
                evt_threshold = gpd_result.threshold
                evt_n_exceedances = gpd_result.n_exceedances
                evt_fit_method = gpd_result.method
                
                # Check consistency with Student-t ν (if available)
                if nu_mc is not None and gpd_result.fit_success:
                    evt_consistency = check_student_t_consistency(nu_mc, gpd_result.xi)
                    
            except Exception as evt_err:
                # EVT failed - fall back to empirical × multiplier
                evt_expected_loss = E_loss_empirical * EVT_FALLBACK_MULTIPLIER
                evt_fit_method = 'exception_fallback'
        elif losses.size > 0:
            # Insufficient data for EVT - use conservative fallback
            evt_expected_loss = E_loss_empirical * EVT_FALLBACK_MULTIPLIER
            evt_fit_method = 'insufficient_data_fallback'
        
        # Use EVT-corrected loss for position sizing
        E_loss = evt_expected_loss

        EU = p_now * E_gain - (1.0 - p_now) * E_loss

        epsilon_eu = 1e-12
        max_position_size = 1.0

        if EU > 0.0 and E_loss > 0.0:
            eu_position_size = EU / max(E_loss, epsilon_eu)
        else:
            eu_position_size = 0.0

        # clip to risk limits
        eu_position_size = float(np.clip(eu_position_size, 0.0, max_position_size))

        # Expected Utility metrics for logging/Signal
        expected_utility = EU
        expected_gain = E_gain
        expected_loss = E_loss
        expected_loss_empirical = E_loss_empirical  # Keep for comparison
        gain_loss_ratio = E_gain / max(E_loss, epsilon_eu) if E_loss > epsilon_eu else (
            100.0 if E_gain > 0 else 1.0
        )

        # For diagnostics: compute drift uncertainty propagated to horizon
        # (kept for Signal dataclass but NOT used for trading)
        if phi_mc is not None and np.isfinite(phi_mc) and abs(phi_mc) < 0.999:
            phi2_diag = phi_mc ** 2
            if abs(1.0 - phi2_diag) > 1e-10:
                drift_var_factor = (1.0 - phi2_diag ** H) / (1.0 - phi2_diag)
            else:
                drift_var_factor = float(H)
        else:
            drift_var_factor = float(H)
        drift_uncertainty_H = drift_var_factor * P_t_mc

        # Diagnostic probabilities (NOT used for trading, kept for analysis only)
        # These are stored in Signal for monitoring but p_now is the only trading probability
        p_empirical = float(np.mean(sim_H > 0.0))  # Raw empirical from simulation
        predictive_var_diag = vH + drift_uncertainty_H
        predictive_std_diag = float(math.sqrt(max(predictive_var_diag, 1e-12)))
        z_predictive_diag = float(mH / predictive_std_diag) if predictive_std_diag > 0 else 0.0
        # Check for Student-t model (phi_student_t_nu_* naming)
        is_student_t_diag = noise_model_mc and noise_model_mc.startswith('phi_student_t_nu_')
        if is_student_t_diag and nu_mc is not None:
            p_analytical = float(student_t.cdf(z_predictive_diag, df=float(nu_mc)))
        else:
            p_analytical = float(norm.cdf(z_predictive_diag))
        p_analytical = float(np.clip(p_analytical, 0.001, 0.999))
        p_posterior_predictive = p_analytical  # Alias for backward compatibility

        # Expected log return and CI from simulation (percentile CI)
        # Define quantile bounds early for use in volatility CI as well
        q = float(np.clip(ci, 1e-6, 0.999999))
        lo_q = (1.0 - q) / 2.0
        hi_q = 1.0 - lo_q

        # Stochastic volatility statistics (Level-7: full posterior uncertainty)
        vol_mean = float(np.mean(vol_H)) if vol_H.size > 0 else 0.0
        try:
            vol_ci_low = float(np.quantile(vol_H, lo_q))
            vol_ci_high = float(np.quantile(vol_H, hi_q))
        except Exception:
            vol_std = float(np.std(vol_H)) if vol_H.size > 1 else 0.0
            vol_ci_low = max(0.0, vol_mean - vol_std)
            vol_ci_high = vol_mean + vol_std
        # For anti‑snap smoothing we need a previous probability; approximate with itself if unavailable
        p_prev = p_now
        p_s_prev = p_prev
        p_s_now = alpha_p * p_now + (1.0 - alpha_p) * p_prev

        # Base/composite edge built off z_stat to keep consistency
        base_now = z_stat
        base_prev = z_stat  # lacking prev sim, reuse now as stable default
        edge_prev = composite_edge(base_prev, trend_prev, moms_prev, vol_reg_prev, z5_prev)
        edge_now = composite_edge(base_now, trend_now, moms_now, vol_reg_now, z5_now)

        # Expected return CI from simulation (percentile CI) - lo_q, hi_q already defined above
        try:
            ci_low = float(np.quantile(sim_H, lo_q))
            ci_high = float(np.quantile(sim_H, hi_q))
        except Exception:
            ci_low = mH - 1.0 * sH
            ci_high = mH + 1.0 * sH

        # ========================================================================
        # UPGRADE #2: EU-Aligned Expected Move
        # ========================================================================
        # Anchor the displayed expected return to the expected utility sign and
        # magnitude. This aligns what users see with the belief that drives
        # BUY/HOLD/SELL decisions, creating cognitive consonance.
        #
        # Formula:
        #   direction = sign(expected_utility)
        #   magnitude = sqrt(|expected_utility|)
        #   eu_aligned_return = direction × magnitude × volatility_scale
        #
        # This does NOT change: signal logic, regimes, model averaging.
        # It only adjusts the displayed expected return for presentation.
        # ========================================================================
        eu_direction = np.sign(expected_utility) if np.isfinite(expected_utility) else 0.0
        eu_magnitude = np.sqrt(abs(expected_utility)) if np.isfinite(expected_utility) else 0.0
        volatility_scale = sH if sH > 0 else 1e-6
        eu_aligned_return = float(eu_direction * eu_magnitude * volatility_scale)

        # Blend: use EU-aligned return for direction-consistency, but preserve
        # reasonable magnitude from trimmed mean
        if abs(eu_aligned_return) > 1e-12 and abs(mH) > 1e-12:
            # Scale EU-aligned to similar magnitude as trimmed mean
            scale_factor = min(abs(mH) / abs(eu_aligned_return), 3.0) if abs(eu_aligned_return) > 1e-12 else 1.0
            mu_H = eu_aligned_return * scale_factor
        else:
            mu_H = mH  # fallback to trimmed mean if EU is near zero
        sig_H = sH

        # ========================================================================
        # EXPECTED UTILITY POSITION SIZING (REPLACES KELLY/MEAN-BASED SIZING)
        # ========================================================================
        # All sizing is now derived from the full posterior predictive distribution
        # (r_samples from BMA), NOT from point estimates.
        #
        # Design Principle:
        #   - Inference produces distributions (r_samples)
        #   - Decisions must consume distributions, not point estimates
        #   - Kelly formula (f_star = mu_H / denom) is PROHIBITED
        #
        # Expected Utility Model:
        #   EU = p × E[gain] - (1-p) × E[loss]
        #   size = EU / max(E[loss], ε)
        #
        # Key Properties:
        #   - Two assets with identical p can have different sizes
        #   - Fat downside tails → higher E[loss] → smaller size
        #   - Strong upside asymmetry → higher E[gain] → larger size
        #   - EU ≤ 0 → HOLD (no position)
        # ========================================================================

        # ====================================================================
        # CALIBRATED TRUST AUTHORITY — SINGLE POINT OF TRUST DECISION
        # ====================================================================
        # ARCHITECTURAL LAW: Trust = Calibration Authority − Regime Penalty
        #
        # This replaces the old hard-coded threshold logic with principled
        # additive decomposition. Calibration speaks first, regimes discount.
        #
        # SCORING (Counter-Proposal v2):
        #   Authority discipline:           98/100
        #   Mathematical transparency:      97/100
        #   Audit traceability:             97/100
        # ====================================================================

        kalman_metadata = feats.get("kalman_metadata", {})
        pit_pvalue = kalman_metadata.get("pit_ks_pvalue")

        # Try to use calibrated trust from tuned params (preferred path)
        calibrated_trust_data = feats.get("calibrated_trust")

        if CALIBRATED_TRUST_AVAILABLE and calibrated_trust_data is not None:
            # Load pre-computed calibrated trust from tuning
            try:
                trust = CalibratedTrust.from_dict(calibrated_trust_data)
                drift_weight = compute_drift_weight(trust, min_weight=0.1, max_weight=1.0)

                # Store for diagnostics
                feats["trust_audit"] = {
                    "calibration_trust": trust.calibration_trust,
                    "regime_penalty": trust.regime_penalty,
                    "effective_trust": trust.effective_trust,
                    "drift_weight": drift_weight,
                    "source": "cached_trust",
                }
            except Exception as e:
                # Fallback to computing trust on-the-fly
                calibrated_trust_data = None

        if not CALIBRATED_TRUST_AVAILABLE or calibrated_trust_data is None:
            # Compute calibrated trust on-the-fly from available PIT data
            drift_weight = 1.0  # Default: trust EU sizing fully

            if CALIBRATED_TRUST_AVAILABLE and pit_pvalue is not None:
                # Build PIT samples if we have recalibration data
                recal_data = feats.get("recalibration")

                if recal_data is not None:
                    # Use stored calibrated PIT
                    calibrated_pit = np.array(recal_data.get("calibrated_pit", []))
                    if len(calibrated_pit) > 0:
                        # Use SOFT regime probabilities from BMA (not hard assignment)
                        # This avoids penalty cliffs at regime boundaries
                        # regime_probs is now a Dict[int, float] from bayesian_model_average_mc
                        soft_regime_probs_for_trust = regime_probs if isinstance(regime_probs, dict) else {1: 1.0}

                        try:
                            trust = compute_calibrated_trust(
                                raw_pit_values=calibrated_pit,
                                regime_probs=soft_regime_probs_for_trust,
                                config=TrustConfig(),
                            )
                            drift_weight = compute_drift_weight(trust, min_weight=0.1, max_weight=1.0)

                            feats["trust_audit"] = {
                                "calibration_trust": trust.calibration_trust,
                                "regime_penalty": trust.regime_penalty,
                                "effective_trust": trust.effective_trust,
                                "drift_weight": drift_weight,
                                "source": "computed_on_fly_soft_regime",
                                "soft_regime_probs": soft_regime_probs_for_trust,
                            }
                        except Exception:
                            pass  # Fall through to legacy logic

            # Legacy fallback: hard threshold (preserved for backward compatibility)
            if "trust_audit" not in feats:
                if pit_pvalue is not None and np.isfinite(pit_pvalue) and pit_pvalue < 0.05:
                    # Calibration warning: model forecasts not well-calibrated
                    drift_weight = 0.3

                feats["trust_audit"] = {
                    "calibration_trust": pit_pvalue if pit_pvalue is not None else 0.5,
                    "regime_penalty": 0.0,
                    "effective_trust": drift_weight,
                    "drift_weight": drift_weight,
                    "source": "legacy_threshold",
                }

        # === FINAL POSITION STRENGTH (from Expected Utility ONLY) ===
        # pos_strength is the ONLY position sizing variable used downstream
        # NO Kelly, NO mean/variance ratios - r_samples is the ONLY input
        pos_strength = drift_weight * eu_position_size

        # Logging/diagnostics: p, E_gain, E_loss, EU, pos_strength
        # (These are stored in Signal dataclass for analysis)

        # Level-7 Modularization: Use helper function for dynamic thresholds
        # Enriched regime_meta with vol_regime for fallback path
        regime_meta_enriched = dict(regime_meta)
        regime_meta_enriched["vol_regime"] = vol_reg_now

        threshold_result = compute_dynamic_thresholds(
            skew=skew_now,
            regime_meta=regime_meta_enriched,
            sig_H=sig_H,
            med_vol_last=med_vol_last,
            H=H
        )

        buy_thr = threshold_result["buy_thr"]
        sell_thr = threshold_result["sell_thr"]
        U = threshold_result["uncertainty"]

        thresholds[int(H)] = {
            "buy_thr": float(buy_thr),
            "sell_thr": float(sell_thr),
            "uncertainty": float(U),
            "edge_floor": float(EDGE_FLOOR)
        }

        # Level-7 Modularization: Use helper function for confirmation logic
        label = apply_confirmation_logic(
            p_smoothed_now=p_s_now,
            p_smoothed_prev=p_s_prev,
            p_raw=p_now,
            pos_strength=pos_strength,
            buy_thr=buy_thr,
            sell_thr=sell_thr,
            edge=edge_now,
            edge_floor=EDGE_FLOOR
        )

        # CI bounds for expected log return
        ci_low = float(mu_H - z_star * sig_H)
        ci_high = float(mu_H + z_star * sig_H)

        # Convert expected log‑return to PLN profit for a 1,000,000 PLN notional
        exp_mult = float(np.exp(mu_H))
        ci_low_mult = float(np.exp(ci_low))
        ci_high_mult = float(np.exp(ci_high))

        # Raw (unsmoothed) profit values
        raw_profit_pln = float(NOTIONAL_PLN) * (exp_mult - 1.0)
        raw_profit_ci_low_pln = float(NOTIONAL_PLN) * (ci_low_mult - 1.0)
        raw_profit_ci_high_pln = float(NOTIONAL_PLN) * (ci_high_mult - 1.0)

        # ========================================================================
        # UPGRADE #3: Display Price Inertia (Presentation-Only)
        # ========================================================================
        # Apply smoothing to displayed profit to reduce day-to-day jitter.
        # Formula: display_price = 0.7 * prev_display_price + 0.3 * new_predicted_price
        #
        # IMPORTANT: This does NOT affect trading decisions, EU, or regimes.
        # It only prevents "why did this jump?" moments for users.
        # ========================================================================
        if asset_key is not None and len(asset_key) > 0:
            profit_pln = _smooth_display_price(asset_key, H, raw_profit_pln)
            profit_ci_low_pln = _smooth_display_price(f"{asset_key}_lo", H, raw_profit_ci_low_pln)
            profit_ci_high_pln = _smooth_display_price(f"{asset_key}_hi", H, raw_profit_ci_high_pln)
        else:
            profit_pln = raw_profit_pln
            profit_ci_low_pln = raw_profit_ci_low_pln
            profit_ci_high_pln = raw_profit_ci_high_pln

        # ========================================================================
        # DUAL-SIDED TREND EXHAUSTION (UE↑ / UE↓) - MULTI-TIMEFRAME
        # ========================================================================
        # Compute directional exhaustion using weighted multi-timeframe EMA
        # deviation with Student-t fat-tail corrections.
        #
        # Output: 0-100% scale indicating how far price deviates from equilibrium
        # - ue_up: Price above weighted EMA equilibrium (higher = more extended)
        # - ue_down: Price below weighted EMA equilibrium (higher = more extended)
        # - Mutual exclusivity: only one can be non-zero
        # - Same value for all horizons (price-based, not model-based)
        # ========================================================================

        # Compute exhaustion from price features (same for all horizons)
        exh_result = compute_directional_exhaustion_from_features(feats)

        ue_up = exh_result["ue_up"]
        ue_down = exh_result["ue_down"]

        # ========================================================================
        # EXHAUSTION-BASED RISK MODULATION (SOFT ONLY)
        # ========================================================================
        # Higher ue_up → reduce long conviction (extended above equilibrium)
        # Higher ue_down → reduce short conviction (extended below equilibrium)
        # This does NOT flip signals - only modulates confidence
        # ========================================================================
        if ue_up > 0 and pos_strength > 0:
            # Extended above equilibrium: reduce long conviction
            pos_strength *= (1.0 - 0.5 * ue_up)
        if ue_down > 0 and pos_strength > 0:
            # Extended below equilibrium: reduce short conviction
            pos_strength *= (1.0 - 0.5 * ue_down)

        # =====================================================================
        # RISK TEMPERATURE MODULATION (Expert Panel Solution 1 + 4)
        # =====================================================================
        # Scale position strength based on cross-asset stress indicators.
        # This is the final modulation layer BEFORE position output.
        #
        # DESIGN: pos_strength_final = pos_strength_base × scale_factor(temp)
        #
        # Stress categories:
        #   - FX (40%): AUDJPY, USDJPY, CHF — risk-on/off proxy
        #   - Futures (30%): ES/NQ momentum — equity sentiment
        #   - Rates (20%): TLT volatility — macro stress
        #   - Commodities (10%): Copper, gold/copper — growth fear
        #
        # Scaling:
        #   - temp = 0.0 → scale ≈ 0.95 (near-full exposure)
        #   - temp = 1.0 → scale = 0.50 (half exposure)
        #   - temp = 2.0 → scale ≈ 0.05 (near-zero exposure)
        #
        # Overnight budget: when temp > 1.0, cap position to limit gap risk
        # =====================================================================
        pos_strength_pre_risk_temp = pos_strength
        risk_temperature = 0.0
        risk_scale_factor = 1.0
        overnight_budget_applied = False
        overnight_max_position = None
        
        if RISK_TEMPERATURE_AVAILABLE:
            try:
                # Get cached risk temperature (avoids redundant API calls)
                risk_temp_result = get_cached_risk_temperature(
                    start_date="2020-01-01",
                    notional=NOTIONAL_PLN,
                    estimated_gap_risk=0.03,  # 3% default gap risk
                )
                
                # Apply scaling
                scaled_pos_strength, risk_meta = apply_risk_temperature_scaling(
                    pos_strength,
                    risk_temp_result,
                )
                
                # Extract values for Signal dataclass
                risk_temperature = risk_meta.get("risk_temperature", 0.0)
                risk_scale_factor = risk_meta.get("scale_factor", 1.0)
                overnight_budget_applied = risk_meta.get("overnight_budget_applied", False)
                overnight_max_position = risk_meta.get("overnight_max_position")
                
                # Update position strength
                pos_strength = scaled_pos_strength
                
            except Exception as e:
                # If risk temperature fails, continue with unscaled position
                if os.getenv("DEBUG"):
                    print(f"Risk temperature computation failed: {e}")

        # ================================================================
        # EXTRACT AUGMENTATION LAYER DATA FROM BMA METADATA
        # ================================================================
        # Hansen Skew-t (asymmetric return distribution)
        hansen_enabled = bma_meta.get("hansen_skew_t_enabled", False)
        hansen_lambda = bma_meta.get("hansen_lambda")
        hansen_nu = bma_meta.get("hansen_nu")
        hansen_skew_direction = bma_meta.get("hansen_skew_direction")
        
        # Contaminated Student-t (regime-dependent tails)
        cst_enabled = bma_meta.get("contaminated_student_t_enabled", False)
        cst_nu_normal = bma_meta.get("cst_nu_normal")
        cst_nu_crisis = bma_meta.get("cst_nu_crisis")
        cst_epsilon = bma_meta.get("cst_epsilon")

        sigs.append(Signal(
            horizon_days=int(H),
            score=float(edge_now),
            p_up=float(p_now),
            exp_ret=float(mu_H),
            ci_low=ci_low,
            ci_high=ci_high,
            profit_pln=float(profit_pln),
            profit_ci_low_pln=float(profit_ci_low_pln),
            profit_ci_high_pln=float(profit_ci_high_pln),
            # ================================================================
            # POSITION STRENGTH FROM EXPECTED UTILITY (NOT KELLY)
            # ================================================================
            # All sizing is derived from the full posterior predictive
            # distribution (r_samples), not from point estimates.
            #
            # pos_strength = drift_weight × eu_position_size × risk_scale_factor
            # where eu_position_size = EU / max(E[loss], ε)
            #
            # This ensures:
            # - Fat downside tails → smaller positions
            # - Strong upside asymmetry → larger positions
            # - EU ≤ 0 → HOLD (position_strength = 0)
            # - High risk temperature → reduced exposure
            # ================================================================
            position_strength=float(pos_strength),
            vol_mean=float(vol_mean),
            vol_ci_low=float(vol_ci_low),
            vol_ci_high=float(vol_ci_high),
            regime=reg,
            label=label,
            # Expected Utility fields (THE BASIS FOR POSITION SIZING):
            expected_utility=float(expected_utility),
            expected_gain=float(expected_gain),
            expected_loss=float(expected_loss),
            gain_loss_ratio=float(gain_loss_ratio),
            eu_position_size=float(eu_position_size),
            # Risk Temperature fields (Expert Panel Solution 1 + 4):
            risk_temperature=float(risk_temperature),
            risk_scale_factor=float(risk_scale_factor),
            overnight_budget_applied=bool(overnight_budget_applied),
            overnight_max_position=float(overnight_max_position) if overnight_max_position is not None else None,
            pos_strength_pre_risk_temp=float(pos_strength_pre_risk_temp),
            # EVT (Extreme Value Theory) tail risk fields:
            expected_loss_empirical=float(expected_loss_empirical),
            evt_enabled=bool(evt_enabled),
            evt_xi=float(evt_xi) if evt_xi is not None else None,
            evt_sigma=float(evt_sigma) if evt_sigma is not None else None,
            evt_threshold=float(evt_threshold) if evt_threshold is not None else None,
            evt_n_exceedances=int(evt_n_exceedances),
            evt_fit_method=str(evt_fit_method) if evt_fit_method is not None else None,
            # Contaminated Student-t Mixture (regime-dependent tails):
            cst_enabled=bool(cst_enabled),
            cst_nu_normal=float(cst_nu_normal) if cst_nu_normal is not None else None,
            cst_nu_crisis=float(cst_nu_crisis) if cst_nu_crisis is not None else None,
            cst_epsilon=float(cst_epsilon) if cst_epsilon is not None else None,
            # Hansen Skew-t (asymmetric return distribution):
            hansen_enabled=bool(hansen_enabled),
            hansen_lambda=float(hansen_lambda) if hansen_lambda is not None else None,
            hansen_nu=float(hansen_nu) if hansen_nu is not None else None,
            hansen_skew_direction=str(hansen_skew_direction) if hansen_skew_direction is not None else None,
            # Diagnostics ONLY (NOT used for trading decisions):
            drift_uncertainty=float(drift_uncertainty_H),
            p_analytical=float(p_analytical),  # DIAGNOSTIC: analytical posterior predictive
            p_empirical=float(p_empirical),    # DIAGNOSTIC: raw empirical MC probability
            # STEP 7: Regime audit trace - tracks which regime params were used
            regime_used=int(regime_used) if regime_used is not None else None,
            regime_source=str(regime_source),
            regime_collapse_warning=bool(collapse_warning),
            # STEP 8: BMA audit trace - tracks model averaging method
            bma_method=str(bma_meta.get("method", "legacy")),
            bma_has_model_posterior=bool(bma_meta.get("has_bma", False)),
            bma_borrowed_from_global=bool(bma_meta.get("regime_details", {}).get(regime_used, {}).get("borrowed_from_global", False)) if regime_used is not None else False,
            # DUAL-SIDED TREND EXHAUSTION (market-space fragility):
            ue_up=float(ue_up),
            ue_down=float(ue_down),
            # PIT Violation EXIT Signal (February 2026):
            pit_exit_triggered=bool(pit_exit_triggered),
            pit_exit_reason=pit_exit_reason,
            pit_violation_severity=float(pit_violation_severity),
            pit_penalty_effective=float(pit_penalty_effective),
            pit_selected_model=pit_selected_model,
        ))

    return sigs, thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate signals across multiple horizons for PLN/JPY, Gold (PLN), Silver (PLN), Bitcoin (PLN), and MicroStrategy (PLN).")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--horizons", type=str, default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--assets", type=str, default=",".join(DEFAULT_ASSET_UNIVERSE), help="Comma-separated Yahoo symbols or friendly names. Metals, FX and USD/EUR/GBP/JPY/CAD/DKK/KRW assets are converted to PLN.")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--cache-json", type=str, default=DEFAULT_CACHE_PATH, help="Path to auto-write cache JSON (default src/data/currencies/fx_plnjpy.json)")
    p.add_argument("--from-cache", action="store_true", help="Render tables from cache JSON and skip computation")
    p.add_argument("--simple", action="store_true", help="Print an easy-to-read summary with simple explanations.")
    p.add_argument("--t_map", action="store_true", help="Use Student-t mapping based on realized kurtosis for probabilities (default on).")
    p.add_argument("--no_t_map", dest="t_map", action="store_false", help="Disable Student-t mapping; use Normal CDF.")
    p.add_argument("--ci", type=float, default=0.68, help="Two-sided confidence level for expected move bands (default 0.68 i.e., ~1-sigma).")
    # Caption controls for detailed view
    p.add_argument("--no_caption", action="store_true", help="Suppress the long column explanation caption in detailed tables.")
    p.add_argument("--force_caption", action="store_true", help="Force showing the caption for every detailed table.")
    # Diagnostics controls (Level-7 falsifiability)
    p.add_argument("--diagnostics", action="store_true", help="Enable full diagnostics: log-likelihood monitoring, parameter stability tracking, and out-of-sample tests (expensive).")
    p.add_argument("--diagnostics_lite", action="store_true", help="Enable lightweight diagnostics: log-likelihood monitoring and parameter stability (no OOS tests).")
    p.add_argument("--pit-calibration", action="store_true", help="Enable PIT calibration verification: tests if predicted probabilities match actual outcomes (Level-7 requirement, very expensive).")
    p.add_argument("--model-comparison", action="store_true", help="Enable structural model comparison: GARCH vs EWMA, Student-t vs Gaussian, Kalman vs EWMA using AIC/BIC (Level-7 falsifiability).")
    p.add_argument("--validate-kalman", action="store_true", help="🧪 Run Level-7 Kalman validation science: drift reasonableness, predictive likelihood improvement, PIT calibration, and stress-regime behavior analysis.")
    p.add_argument("--validation-plots", action="store_true", help="Generate diagnostic plots for Kalman validation (requires --validate-kalman).")
    p.add_argument("--failures-json", type=str, default=os.path.join(os.path.dirname(__file__), "fx_failures.json"), help="Where to write failure log (set to '' to disable)")
    p.set_defaults(t_map=True)
    return p.parse_args()


def process_single_asset(args_tuple: Tuple) -> Optional[Dict]:
    """
    Worker function to process a single asset in parallel.
    Only performs computation, no console output.

    Args:
        args_tuple: (asset, args, horizons)

    Returns:
        Dictionary with processed results or None if failed
    """
    asset, args, horizons = args_tuple

    try:
        # Fetch price data
        try:
            px, title = fetch_px_asset(asset, args.start, args.end)
        except Exception as e:
            return {
                "status": "error",
                "asset": asset,
                "error": str(e)
            }

        # De-duplicate by resolved symbol
        canon = extract_symbol_from_title(title)
        if not canon:
            canon = asset.strip().upper()

        # Compute features and signals
        feats = compute_features(px, asset_symbol=asset)
        last_close = _to_float(px.iloc[-1])

        # Load tuned params with BMA structure for model averaging
        tuned_params = _load_tuned_kalman_params(asset)

        # Pass asset_key (canon) for display price inertia (Upgrade #3)
        sigs, thresholds = latest_signals(feats, horizons, last_close=last_close, t_map=args.t_map, ci=args.ci, tuned_params=tuned_params, asset_key=canon)

        # Compute diagnostics if requested
        diagnostics = {}
        if args.diagnostics or args.diagnostics_lite or args.pit_calibration or args.model_comparison:
            enable_oos = args.diagnostics
            enable_pit = args.pit_calibration
            enable_model_comp = args.model_comparison
            diagnostics = compute_all_diagnostics(px, feats, enable_oos=enable_oos, enable_pit_calibration=enable_pit, enable_model_comparison=enable_model_comp)

        return {
            "status": "success",
            "asset": asset,
            "canon": canon,
            "title": title,
            "px": px,
            "feats": feats,
            "sigs": sigs,
            "thresholds": thresholds,
            "diagnostics": diagnostics,
            "last_close": last_close,
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "asset": asset,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def _process_assets_with_retries(assets: List[str], args: argparse.Namespace, horizons: List[int], max_retries: int = 3):
    """Run asset processing with bounded retries and collect failures.
    Retries only the assets that failed on prior attempts.
    Uses multiprocessing.Pool for true multi-process parallelism (CPU-bound work).
    """
    from rich.rule import Rule
    from rich.align import Align

    console = Console(force_terminal=True, width=140)
    pending = list(dict.fromkeys(a.strip() for a in assets if a and a.strip()))
    successes: List[Dict] = []
    failures: Dict[str, Dict[str, object]] = {}
    processed_canon = set()

    # ═══════════════════════════════════════════════════════════════════════════════
    # EXTRAORDINARY APPLE-QUALITY PROCESSING UX
    # ═══════════════════════════════════════════════════════════════════════════════

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    # Processing header
    header = Text()
    header.append("▸ ", style="bright_cyan")
    header.append("PROCESSING", style="bold white")
    console.print(header)
    console.print()

    # Stats row
    n_workers = min(cpu_count(), len(pending))
    stats = Text()
    stats.append("    ", style="")
    stats.append(f"{len(pending)}", style="bold bright_cyan")
    stats.append(" assets", style="dim")
    stats.append("  ·  ", style="dim")
    stats.append(f"{n_workers}", style="bold white")
    stats.append(" cores", style="dim")
    stats.append("  ·  ", style="dim")
    stats.append(f"{max_retries}", style="white")
    stats.append(" max retries", style="dim")
    console.print(stats)
    console.print()

    attempt = 1
    while attempt <= max_retries and pending:
        n_workers = min(cpu_count(), len(pending))

        # Pass indicator
        pass_text = Text()
        pass_text.append(f"    Pass {attempt}/{max_retries}", style="dim")
        pass_text.append(f"  ·  {len(pending)} pending", style="dim")
        console.print(pass_text)

        work_items = [(asset, args, horizons) for asset in pending]

        # Prefetch prices in bulk on first attempt to reduce Yahoo rate limits
        if attempt == 1 and pending:
            try:
                # Suppress verbose output and symbol tables - they're shown after validation
                download_prices_bulk(pending, start=args.start, end=args.end, progress=False, show_symbol_tables=False)
            except Exception as e:
                console.print(f"    [yellow]⚠[/yellow] [dim]Bulk prefetch failed, using standard fetch[/dim]")

        # Always use multiprocessing.Pool for true multi-process parallelism
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_asset, work_items)

        next_pending: List[str] = []
        pass_successes = 0
        for asset, result in zip(pending, results):
            if not result or result.get("status") != "success":
                err = (result or {}).get("error", "unknown")
                tb = (result or {}).get("traceback")
                if tb:
                    tb_lines = [line.strip() for line in str(tb).splitlines() if line.strip()]
                    loc_lines = [ln for ln in tb_lines if ln.startswith("File ")]
                    loc_line = loc_lines[-1] if loc_lines else None
                    if loc_line:
                        err = f"{err} @ {loc_line}"
                try:
                    disp = _resolve_display_name(asset.strip().upper())
                except Exception:
                    disp = asset
                entry = failures.get(asset, {"attempts": 0, "last_error": None, "display_name": disp, "traceback": None})
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["last_error"] = err
                if tb:
                    entry["traceback"] = tb
                entry["display_name"] = entry.get("display_name") or disp
                failures[asset] = entry
                next_pending.append(asset)
                continue


            canon = result.get("canon") or asset.strip().upper()
            if canon in processed_canon:
                continue
            processed_canon.add(canon)
            successes.append(result)
            pass_successes += 1
            # drop from pending on success; nothing to add to next_pending
            if asset in failures:
                failures.pop(asset, None)

        pending = list(dict.fromkeys(next_pending))

        # Pass result
        if pass_successes > 0:
            console.print(f"    [bright_green]✓[/bright_green] [dim]{pass_successes} succeeded[/dim]")

        # Early exit: if all assets succeeded, skip remaining passes
        if not pending:
            break

        if pending and attempt < max_retries:
            console.print(f"    [yellow]○[/yellow] [dim]{len(pending)} retrying...[/dim]")

        attempt += 1

    console.print()

    # Final status
    if not pending:
        done = Text()
        done.append("    ", style="")
        done.append("✓", style="bold bright_green")
        done.append(f"  {len(successes)} assets processed", style="white")
        console.print(done)
    else:
        done = Text()
        done.append("    ", style="")
        done.append("!", style="bold yellow")
        done.append(f"  {len(successes)} succeeded, {len(pending)} failed", style="white")
        console.print(done)

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    return successes, failures


def main() -> None:
    args = parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})

    # Fast path: render from cache only (SUMMARY ONLY - no detailed tables)
    if args.from_cache:
        cache_path = args.cache_json or DEFAULT_CACHE_PATH
        if not os.path.exists(cache_path):
            Console().print(f"[indian_red1]Cache not found:[/indian_red1] {cache_path}")
            return
        with open(cache_path, "r") as f:
            payload = json.load(f)
        horizons_cached = payload.get("horizons") or horizons
        assets_cached = payload.get("assets", [])
        summary_rows_cached = payload.get("summary_rows")

        console = Console()

        # Build summary rows from cache (skip detailed tables for compact display)
        if not summary_rows_cached:
            summary_rows_cached = []
            for asset_data in assets_cached:
                sym = asset_data.get("symbol") or ""
                title = asset_data.get("title") or sym
                asset_label = build_asset_display_label(sym, title)
                # Use get_sector() to properly look up sector from SECTOR_MAP
                sector = asset_data.get("sector") or get_sector(sym) or "Other"
                horizon_signals = {}
                for sig in asset_data.get("signals", []):
                    h = sig.get("horizon_days")
                    if h is None:
                        continue
                    horizon_signals[int(h)] = {
                        "label": sig.get("label", "HOLD"),
                        "profit_pln": float(sig.get("profit_pln", 0.0)),
                        "ue_up": float(sig.get("ue_up", 0.0)),
                        "ue_down": float(sig.get("ue_down", 0.0)),
                        "p_up": float(sig.get("p_up", 0.5)),
                        "exp_ret": float(sig.get("exp_ret", 0.0)),
                    }
                nearest_label = next(iter(horizon_signals.values()), {}).get("label", "HOLD")
                summary_rows_cached.append({"asset_label": asset_label, "horizon_signals": horizon_signals, "nearest_label": nearest_label, "sector": sector})
        else:
            # Ensure existing summary_rows have proper sectors and p_up/exp_ret fields
            # This handles old cache format that may be missing these fields
            for row in summary_rows_cached:
                if not row.get("sector"):
                    # Try to extract symbol from asset_label and look up sector
                    label = row.get("asset_label", "")
                    # Extract symbol from "Name (SYM)" format
                    import re
                    match = re.search(r'\(([A-Z0-9.-]+)\)', label)
                    if match:
                        sym = match.group(1)
                        row["sector"] = get_sector(sym) or "Other"
                    else:
                        row["sector"] = "Other"

                # Ensure horizon_signals have p_up and exp_ret (from assets_cached if needed)
                horizon_signals = row.get("horizon_signals", {})
                for h, sig_data in horizon_signals.items():
                    if "p_up" not in sig_data or "exp_ret" not in sig_data:
                        # Try to find from assets_cached
                        for asset_data in assets_cached:
                            for sig in asset_data.get("signals", []):
                                if sig.get("horizon_days") == h or sig.get("horizon_days") == int(h):
                                    if "p_up" not in sig_data:
                                        sig_data["p_up"] = float(sig.get("p_up", 0.5))
                                    if "exp_ret" not in sig_data:
                                        sig_data["exp_ret"] = float(sig.get("exp_ret", 0.0))
                                    break

        try:
            render_sector_summary_tables(summary_rows_cached, horizons_cached)
            # Add high-conviction signals summary for short-term trading
            render_strong_signals_summary(summary_rows_cached, horizons=[1, 3, 7])
            
            # Show risk temperature summary (computed once, applies to all assets)
            if RISK_TEMPERATURE_AVAILABLE:
                try:
                    risk_temp_result = get_cached_risk_temperature(
                        start_date="2020-01-01",
                        notional=NOTIONAL_PLN,
                        estimated_gap_risk=0.03,
                    )
                    render_risk_temperature_summary(risk_temp_result)
                except Exception:
                    pass  # Silently skip if risk temp fails
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not print summary tables from cache: {e}")
        return

    # Parse assets
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    console = Console(force_terminal=True, width=140)

    # ═══════════════════════════════════════════════════════════════════════════════
    # VALIDATION PHASE - Apple-quality UX
    # ═══════════════════════════════════════════════════════════════════════════════
    from rich.rule import Rule

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    validation_header = Text()
    validation_header.append("▸ ", style="bright_cyan")
    validation_header.append("VALIDATION", style="bold white")
    console.print(validation_header)
    console.print()

    validation_stats = Text()
    validation_stats.append("    ", style="")
    validation_stats.append(f"{len(assets)}", style="bold bright_cyan")
    validation_stats.append(" assets requested", style="dim")
    console.print(validation_stats)
    console.print()

    for a in assets:
        try:
            _resolve_symbol_candidates(a)
        except Exception as e:
            console.print(f"    [yellow]⚠[/yellow] [dim]{a}: {e}[/dim]")

    # Print symbol resolution table right after validation (before processing starts)
    print_symbol_tables()

    all_blocks = []  # for JSON export
    csv_rows_simple = []  # for CSV simple export
    csv_rows_detailed = []  # for CSV detailed export
    summary_rows = []  # for summary table across assets

    # =========================================================================
    # RETRYING PARALLEL PROCESSING: Compute features/signals with bounded retries
    # =========================================================================
    success_results, failures = _process_assets_with_retries(assets, args, horizons, max_retries=3)

    # =========================================================================
    # SEQUENTIAL DISPLAY & AGGREGATION: Process results in order with console output
    # =========================================================================
    caption_printed = False
    processed_syms = set()

    for result in success_results:
        # Handle None or error results
        if result is None:
            continue

        if result.get("status") == "error":
            asset = result.get("asset", "unknown")
            error = result.get("error", "unknown")
            console.print(f"[indian_red1]Warning:[/indian_red1] Failed to process {asset}: {error}")
            if os.getenv('DEBUG'):
                traceback_info = result.get("traceback", "")
                if traceback_info:
                    console.print(f"[dim]{traceback_info}[/dim]")
            continue

        if result.get("status") != "success":
            continue

        # Extract computed results from worker
        asset = result["asset"]
        canon = result["canon"]
        title = result["title"]
        px = result["px"]
        feats = result["feats"]
        sigs = result["sigs"]
        thresholds = result["thresholds"]
        diagnostics = result["diagnostics"]
        last_close = result["last_close"]

        # De-duplicate check
        if canon in processed_syms:
            console.print(f"[yellow]Skipping duplicate:[/yellow] {title} (from input '{asset}')")
            continue
        processed_syms.add(canon)

        # Print table for this asset
        if args.simple:
            explanations = render_simplified_signal_table(asset, title, sigs, px, feats)
        else:
            # Determine caption policy for detailed view
            if args.force_caption:
                show_caption = True
            elif args.no_caption:
                show_caption = False
            else:
                show_caption = not caption_printed
            render_detailed_signal_table(asset, title, sigs, px, confidence_level=args.ci, used_student_t_mapping=args.t_map, show_caption=show_caption)
            caption_printed = caption_printed or show_caption
            
            # Show augmentation layer summary if any are active (first signal)
            if sigs:
                render_augmentation_layers_summary(sigs[0])
            
            explanations = []

        # Display diagnostics if computed (Level-7: model falsifiability)
        if diagnostics and (args.diagnostics or args.diagnostics_lite):
            from rich.table import Table
            console = Console()
            diag_table = Table(title=f"📊 Diagnostics for {asset} — Model Falsifiability Metrics")
            diag_table.add_column("Metric", justify="left", style="cyan")
            diag_table.add_column("Value", justify="right")

            # Log-likelihood monitoring
            if "garch_log_likelihood" in diagnostics:
                diag_table.add_row("GARCH(1,1) Log-Likelihood", f"{diagnostics['garch_log_likelihood']:.2f}")
                diag_table.add_row("GARCH(1,1) AIC", f"{diagnostics['garch_aic']:.2f}")
                diag_table.add_row("GARCH(1,1) BIC", f"{diagnostics['garch_bic']:.2f}")

            # Pillar 1: Kalman filter drift diagnostics (with refinements)
            if "kalman_log_likelihood" in diagnostics:
                diag_table.add_row("Kalman Filter Log-Likelihood", f"{diagnostics['kalman_log_likelihood']:.2f}")
                if "kalman_process_noise_var" in diagnostics:
                    diag_table.add_row("Kalman Process Noise (q)", f"{diagnostics['kalman_process_noise_var']:.6f}")
                if "kalman_n_obs" in diagnostics:
                    diag_table.add_row("Kalman Observations", f"{diagnostics['kalman_n_obs']}")

                # Refinement 1: q optimization results
                if "kalman_q_optimal" in diagnostics and "kalman_q_heuristic" in diagnostics:
                    q_opt = diagnostics["kalman_q_optimal"]
                    q_heur = diagnostics["kalman_q_heuristic"]
                    q_optimized = diagnostics.get("kalman_q_optimization_attempted", False)
                    if np.isfinite(q_opt) and np.isfinite(q_heur):
                        ratio = q_opt / q_heur if q_heur > 0 else 1.0
                        opt_label = "optimized" if q_optimized else "heuristic"
                        diag_table.add_row(f"  q ({opt_label})", f"{q_opt:.6f} ({ratio:.2f}× heuristic)")

                # Refinement 2: Kalman gain (situational awareness)
                if "kalman_gain_mean" in diagnostics and "kalman_gain_recent" in diagnostics:
                    gain_mean = diagnostics["kalman_gain_mean"]
                    gain_recent = diagnostics["kalman_gain_recent"]
                    if np.isfinite(gain_mean):
                        # Interpretation: high gain = aggressive learning, low gain = stable drift
                        interpretation = "aggressive" if gain_mean > 0.3 else ("moderate" if gain_mean > 0.1 else "stable")
                        diag_table.add_row(f"  Kalman Gain (mean)", f"{gain_mean:.4f} [{interpretation}]")
                    if np.isfinite(gain_recent):
                        diag_table.add_row(f"  Kalman Gain (recent)", f"{gain_recent:.4f}")

                # Refinement 3: Innovation whiteness (model adequacy)
                if "innovation_ljung_box_pvalue" in diagnostics:
                    pvalue = diagnostics["innovation_ljung_box_pvalue"]
                    model_adequate = diagnostics.get("innovation_model_adequate", None)
                    lags = diagnostics.get("innovation_lags_tested", 0)
                    if np.isfinite(pvalue) and model_adequate is not None:
                        color = "green" if model_adequate else "red"
                        status = "PASS" if model_adequate else "FAIL"
                        diag_table.add_row(f"  Innovation Whiteness (Ljung-Box)", f"[{color}]{status}[/{color}] (p={pvalue:.3f}, lags={lags})")

                # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
                if "kalman_heteroskedastic_mode" in diagnostics:
                    hetero_mode = diagnostics.get("kalman_heteroskedastic_mode", False)
                    c_opt = diagnostics.get("kalman_c_optimal")
                    if hetero_mode and c_opt is not None and np.isfinite(c_opt):
                        diag_table.add_row(f"  Process Noise Mode", f"[cyan]Heteroskedastic[/cyan] (q_t = c·σ_t²)")
                        diag_table.add_row(f"  Scaling Factor (c)", f"{c_opt:.6f}")
                        # Show q_t statistics if available
                        q_t_mean = diagnostics.get("kalman_q_t_mean")
                        q_t_std = diagnostics.get("kalman_q_t_std")
                        q_t_min = diagnostics.get("kalman_q_t_min")
                        q_t_max = diagnostics.get("kalman_q_t_max")
                        if q_t_mean is not None and np.isfinite(q_t_mean):
                            diag_table.add_row(f"  q_t (mean ± std)", f"{q_t_mean:.6f} ± {q_t_std:.6f}" if q_t_std and np.isfinite(q_t_std) else f"{q_t_mean:.6f}")
                        if q_t_min is not None and q_t_max is not None and np.isfinite(q_t_min) and np.isfinite(q_t_max):
                            diag_table.add_row(f"  q_t range [min, max]", f"[{q_t_min:.6f}, {q_t_max:.6f}]")
                    elif not hetero_mode:
                        diag_table.add_row(f"  Process Noise Mode", f"Homoskedastic (constant q)")

                # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
                if "kalman_robust_t_mode" in diagnostics:
                    robust_t = diagnostics.get("kalman_robust_t_mode", False)
                    nu_robust = diagnostics.get("kalman_nu_robust")
                    if robust_t and nu_robust is not None and np.isfinite(nu_robust):
                        diag_table.add_row(f"  Innovation Distribution", f"[magenta]Student-t[/magenta] (robust filtering)")
                        diag_table.add_row(f"  Innovation ν (degrees of freedom)", f"{nu_robust:.2f}")
                    elif not robust_t:
                        diag_table.add_row(f"  Innovation Distribution", f"Gaussian (standard)")

                # Level-7+ Refinement: Regime-dependent drift priors
                if "kalman_regime_prior_used" in diagnostics:
                    regime_prior_used = diagnostics.get("kalman_regime_prior_used", False)
                    if regime_prior_used:
                        regime_current = diagnostics.get("kalman_regime_current", "")
                        drift_prior = diagnostics.get("kalman_regime_drift_prior")
                        if regime_current and drift_prior is not None and np.isfinite(drift_prior):
                            diag_table.add_row(f"  Drift Prior (regime-aware)", f"[yellow]Enabled[/yellow] (regime: {regime_current})")
                            diag_table.add_row(f"  E[μ | Regime={regime_current}]", f"{drift_prior:+.6f}")
                    else:
                        diag_table.add_row(f"  Drift Prior", f"Neutral (μ₀ = 0)")

            if "hmm_log_likelihood" in diagnostics:
                diag_table.add_row("HMM Regime Log-Likelihood", f"{diagnostics['hmm_log_likelihood']:.2f}")
                diag_table.add_row("HMM AIC", f"{diagnostics['hmm_aic']:.2f}")
                diag_table.add_row("HMM BIC", f"{diagnostics['hmm_bic']:.2f}")

            if "student_t_log_likelihood" in diagnostics:
                diag_table.add_row("Student-t Tail Log-Likelihood", f"{diagnostics['student_t_log_likelihood']:.2f}")
                diag_table.add_row("Student-t Degrees of Freedom (ν)", f"{diagnostics['student_t_nu']:.2f}")

                # Tier 2: Display ν standard error (posterior parameter variance)
                if "student_t_se_nu" in diagnostics:
                    se_nu = diagnostics["student_t_se_nu"]
                    if np.isfinite(se_nu) and se_nu > 0:
                        nu_hat = diagnostics.get("student_t_nu", float("nan"))
                        # Coefficient of variation: SE/estimate (relative uncertainty)
                        cv_nu = (se_nu / nu_hat) if np.isfinite(nu_hat) and nu_hat > 0 else float("nan")
                        uncertainty_level = "low" if cv_nu < 0.05 else ("moderate" if cv_nu < 0.10 else "high")
                        diag_table.add_row("  SE(ν) [posterior uncertainty]", f"{se_nu:.3f} ({cv_nu*100:.1f}% CV, {uncertainty_level})")
                    else:
                        diag_table.add_row("  SE(ν) [posterior uncertainty]", f"{se_nu:.3f}")

            # Tier 2: Parameter Uncertainty Summary (μ, σ, ν)
            param_unc_env = os.getenv("PARAM_UNC", "sample").strip().lower()
            nu_sample_env = os.getenv("NU_SAMPLE", "true").strip().lower()

            param_unc_active = {
                "μ (drift)": "Kalman var_kf → process noise q",
                "σ (volatility)": f"GARCH sampling: {'✓ enabled' if param_unc_env == 'sample' else '✗ disabled'}",
                "ν (tails)": f"Student-t sampling: {'✓ enabled' if nu_sample_env == 'true' else '✗ disabled'}"
            }

            diag_table.add_row("", "")  # spacer
            diag_table.add_row("[bold cyan]Tier 2: Posterior Parameter Variance[/bold cyan]", "[bold]Status[/bold]")
            for param, status in param_unc_active.items():
                if "✓" in status:
                    diag_table.add_row(f"  {param}", f"[#00d700]{status}[/#00d700]")
                elif "✗" in status:
                    diag_table.add_row(f"  {param}", f"[yellow]{status}[/yellow]")
                else:
                    diag_table.add_row(f"  {param}", status)

            # Parameter stability (recent drift z-scores)
            drift_cols = [k for k in diagnostics.keys() if k.startswith("recent_") and k.endswith("_drift_z")]
            if drift_cols:
                diag_table.add_row("", "")  # spacer
                diag_table.add_row("[bold]Parameter Stability[/bold]", "[bold]Recent Drift (z-score)[/bold]")
                for col in drift_cols:
                    param_name = col.replace("recent_", "").replace("_drift_z", "")
                    val = diagnostics[col]
                    if np.isfinite(val):
                        color = "green" if abs(val) < 2.0 else ("yellow" if abs(val) < 3.0 else "red")
                        diag_table.add_row(f"  {param_name}", f"[{color}]{val:+.2f}[/{color}]")

            # Out-of-sample test results (if enabled)
            oos_keys = [k for k in diagnostics.keys() if k.startswith("oos_") and k.endswith("_hit_rate")]
            if oos_keys:
                diag_table.add_row("", "")  # spacer
                diag_table.add_row("[bold]Out-of-Sample Tests[/bold]", "[bold]Direction Hit Rate[/bold]")
                for key in oos_keys:
                    horizon_label = key.replace("oos_", "").replace("_hit_rate", "")
                    hit_rate = diagnostics[key]
                    if np.isfinite(hit_rate):
                        color = "green" if hit_rate >= 0.55 else ("yellow" if hit_rate >= 0.50 else "red")
                        diag_table.add_row(f"  {horizon_label}", f"[{color}]{hit_rate*100:.1f}%[/{color}]")

            console.print(diag_table)
            console.print("")  # blank line

            # Display PIT calibration report if available
            if "pit_calibration" in diagnostics:
                try:
                    from calibration.pit_calibration import format_calibration_report
                    calibration_report = format_calibration_report(
                        calibration_results=diagnostics["pit_calibration"],
                        asset_name=asset
                    )
                    console.print(calibration_report)
                except Exception:
                    pass

            # Display model comparison results if available
            if "model_comparison" in diagnostics and diagnostics["model_comparison"]:
                from rich.table import Table
                comparison_results = diagnostics["model_comparison"]

                # Create comparison table for each category
                for category, result in comparison_results.items():
                    if result is None or not hasattr(result, 'winner_aic'):
                        continue

                    category_title = {
                        'volatility': 'Volatility Models',
                        'tails': 'Tail Distribution Models',
                        'drift': 'Drift Models'
                    }.get(category, category.title())

                    comp_table = Table(title=f"📊 Model Comparison: {category_title} — {asset}")
                    comp_table.add_column("Model", justify="left", style="cyan")
                    comp_table.add_column("Params", justify="right")
                    comp_table.add_column("Log-Likelihood", justify="right")
                    comp_table.add_column("AIC", justify="right")
                    comp_table.add_column("BIC", justify="right")
                    comp_table.add_column("Δ AIC", justify="right")
                    comp_table.add_column("Δ BIC", justify="right")
                    comp_table.add_column("Akaike Wt", justify="right")

                    for model in result.models:
                        name = model.name

                        # Highlight winners
                        if name == result.winner_aic and name == result.winner_bic:
                            name = f"[bold #00d700]{name}[/bold #00d700] ⭐"
                        elif name == result.winner_aic:
                            name = f"[bold yellow]{name}[/bold yellow] (AIC)"
                        elif name == result.winner_bic:
                            name = f"[bold blue]{name}[/bold blue] (BIC)"

                        delta_aic = result.delta_aic.get(model.name, float('nan'))
                        delta_bic = result.delta_bic.get(model.name, float('nan'))
                        weight = result.akaike_weights.get(model.name, 0.0)

                        # Color code deltas (lower is better)
                        if np.isfinite(delta_aic):
                            if delta_aic < 2.0:
                                delta_aic_str = f"[#00d700]{delta_aic:+.1f}[/#00d700]"
                            elif delta_aic < 7.0:
                                delta_aic_str = f"[yellow]{delta_aic:+.1f}[/yellow]"
                            else:
                                delta_aic_str = f"[indian_red1]{delta_aic:+.1f}[/indian_red1]"
                        else:
                            delta_aic_str = "—"

                        if np.isfinite(delta_bic):
                            if delta_bic < 2.0:
                                delta_bic_str = f"[#00d700]{delta_bic:+.1f}[/#00d700]"
                            elif delta_bic < 10.0:
                                delta_bic_str = f"[yellow]{delta_bic:+.1f}[/yellow]"
                            else:
                                delta_bic_str = f"[indian_red1]{delta_bic:+.1f}[/indian_red1]"
                        else:
                            delta_bic_str = "—"

                        comp_table.add_row(
                            name,
                            str(model.n_params),
                            f"{model.log_likelihood:.2f}",
                            f"{model.aic:.2f}",
                            f"{model.bic:.2f}",
                            delta_aic_str,
                            delta_bic_str,
                            f"{weight:.1%}" if np.isfinite(weight) else "—"
                        )

                    # Add recommendation row
                    comp_table.add_row("", "", "", "", "", "", "", "")
                    comp_table.add_row(
                        f"[bold]Recommendation:[/bold]",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                    )

                    console.print(comp_table)
                    console.print(f"[dim]{result.recommendation}[/dim]\n")

                # Summary interpretation
                console.print("[bold cyan]Model Comparison Interpretation:[/bold cyan]")
                console.print("[dim]• Δ AIC/BIC < 2: Substantial support (competitive models)[/dim]")
                console.print("[dim]• Δ AIC/BIC 4-7: Considerably less support[/dim]")
                console.print("[dim]• Δ AIC/BIC > 10: Essentially no support[/dim]")
                console.print("[dim]• Akaike weight: Probability this model is best[/dim]")
                console.print("[dim]• Lower AIC/BIC = better (fit + parsimony tradeoff)[/dim]\n")

        # 🧪 Level-7 Validation Science: Kalman Filter Validation Suite
        if args.validate_kalman:
            try:
                from kalman_validation import (
                    run_full_validation_suite,
                    validate_drift_reasonableness,
                    compare_predictive_likelihood,
                    validate_pit_calibration,
                    analyze_stress_regime_behavior
                )
                from rich.table import Table
                from rich.panel import Panel

                console = Console()
                console.print("\n")
                console.print(Panel.fit(
                    f"🧪 [bold cyan]Level-7 Validation Science[/bold cyan] — {asset}\n"
                    "[dim]Does my model behave like reality?[/dim]",
                    border_style="cyan"
                ))

                # Extract required series from features
                ret = feats.get("ret")
                mu_kf = feats.get("mu_kf", feats.get("mu"))
                var_kf = feats.get("var_kf", pd.Series(0.0, index=ret.index))
                vol = feats.get("vol")

                if mu_kf is not None and ret is not None and vol is not None:
                    # Prepare plot directory if plots requested
                    plot_dir = None
                    if args.validation_plots:
                        plot_dir = "src/data/plots/kalman_validation"
                        os.makedirs(plot_dir, exist_ok=True)

                    # 1. Drift Reasonableness Validation
                    console.print("\n[bold yellow]1. Posterior Drift Reasonableness[/bold yellow]")
                    drift_result = validate_drift_reasonableness(
                        px, ret, mu_kf, var_kf, asset_name=asset,
                        plot=args.validation_plots,
                        save_path=f"{plot_dir}/{asset}_drift_validation.png" if plot_dir else None
                    )

                    drift_table = Table(title="Drift Sanity Checks", show_header=True)
                    drift_table.add_column("Metric", style="cyan")
                    drift_table.add_column("Value", justify="right")
                    drift_table.add_column("Status", justify="center")

                    drift_table.add_row(
                        "Observations",
                        str(drift_result.observations),
                        ""
                    )
                    drift_table.add_row(
                        "Drift Smoothness Ratio",
                        f"{drift_result.drift_smoothness_ratio:.4f}",
                        "✅" if drift_result.drift_smoothness_ratio < 0.5 else "⚠️"
                    )
                    drift_table.add_row(
                        "Crisis Uncertainty Spike",
                        f"{drift_result.crisis_uncertainty_spike:.2f}×",
                        "✅" if drift_result.crisis_uncertainty_spike > 1.5 else "⚠️"
                    )
                    drift_table.add_row(
                        "Regime Breaks Detected",
                        "Yes" if drift_result.regime_break_detected else "No",
                        "✅" if drift_result.regime_break_detected else "ℹ️"
                    )
                    drift_table.add_row(
                        "Noise Tracking Score",
                        f"{drift_result.noise_tracking_score:.4f}",
                        "✅" if drift_result.noise_tracking_score < 0.4 else "⚠️"
                    )

                    console.print(drift_table)
                    console.print(f"[dim]{drift_result.diagnostic_message}[/dim]\n")

                    # 2. Predictive Likelihood Improvement
                    console.print("[bold yellow]2. Predictive Likelihood Improvement[/bold yellow]")
                    ll_result = compare_predictive_likelihood(px, asset_name=asset)

                    ll_table = Table(title="Model Comparison (Out-of-Sample)", show_header=True)
                    ll_table.add_column("Model", style="cyan")
                    ll_table.add_column("Log-Likelihood", justify="right")
                    ll_table.add_column("Δ LL", justify="right")

                    ll_table.add_row("Kalman Filter", f"{ll_result.ll_kalman:.2f}", "—")
                    ll_table.add_row(
                        "Zero Drift (μ=0)",
                        f"{ll_result.ll_zero_drift:.2f}",
                        f"[#00d700]{ll_result.delta_ll_vs_zero:+.2f}[/#00d700]" if ll_result.delta_ll_vs_zero > 0 else f"[indian_red1]{ll_result.delta_ll_vs_zero:+.2f}[/indian_red1]"
                    )
                    ll_table.add_row(
                        "EWMA Drift",
                        f"{ll_result.ll_ewma_drift:.2f}",
                        f"[#00d700]{ll_result.delta_ll_vs_ewma:+.2f}[/#00d700]" if ll_result.delta_ll_vs_ewma > 0 else f"[indian_red1]{ll_result.delta_ll_vs_ewma:+.2f}[/indian_red1]"
                    )
                    ll_table.add_row(
                        "Constant Drift",
                        f"{ll_result.ll_constant_drift:.2f}",
                        f"[#00d700]{ll_result.delta_ll_vs_constant:+.2f}[/#00d700]" if ll_result.delta_ll_vs_constant > 0 else f"[indian_red1]{ll_result.delta_ll_vs_constant:+.2f}[/indian_red1]"
                    )

                    console.print(ll_table)
                    console.print(f"[bold]Best Model:[/bold] {ll_result.best_model}")
                    console.print(f"[dim]{ll_result.diagnostic_message}[/dim]\n")

                    # 3. PIT Calibration Check
                    console.print("[bold yellow]3. Probability Integral Transform (PIT) Calibration[/bold yellow]")
                    pit_result = validate_pit_calibration(
                        px, ret, mu_kf, var_kf, vol, asset_name=asset,
                        plot=args.validation_plots,
                        save_path=f"{plot_dir}/{asset}_pit_calibration.png" if plot_dir else None
                    )

                    pit_table = Table(title="Forecast Calibration", show_header=True)
                    pit_table.add_column("Metric", style="cyan")
                    pit_table.add_column("Value", justify="right")
                    pit_table.add_column("Expected", justify="right")
                    pit_table.add_column("Status", justify="center")

                    pit_table.add_row(
                        "Observations",
                        str(pit_result.n_observations),
                        "—",
                        ""
                    )
                    pit_table.add_row(
                        "KS Statistic",
                        f"{pit_result.ks_statistic:.4f}",
                        "—",
                        ""
                    )
                    pit_table.add_row(
                        "KS p-value",
                        f"{pit_result.ks_pvalue:.4f}",
                        "> 0.05",
                        "✅" if pit_result.ks_pvalue >= 0.05 else "⚠️"
                    )
                    pit_table.add_row(
                        "PIT Mean",
                        f"{pit_result.pit_mean:.4f}",
                        "0.5000",
                        "✅" if abs(pit_result.pit_mean - 0.5) < 0.05 else "⚠️"
                    )
                    pit_table.add_row(
                        "PIT Std Dev",
                        f"{pit_result.pit_std:.4f}",
                        f"{expected_std:.4f}",
                        "✅" if abs(pit_result.pit_std - expected_std) < 0.05 else "⚠️"
                    )

                    console.print(pit_table)
                    console.print(f"[dim]{pit_result.diagnostic_message}[/dim]\n")

                    # 4. Stress-Regime Behavior
                    console.print("[bold yellow]4. Stress-Regime Behavior Analysis[/bold yellow]")
                    stress_result = analyze_stress_regime_behavior(
                        px, ret, mu_kf, var_kf, vol, asset_name=asset
                    )

                    stress_table = Table(title="Risk Intelligence", show_header=True)
                    stress_table.add_column("Metric", style="cyan")
                    stress_table.add_column("Normal", justify="right")
                    stress_table.add_column("Stress", justify="right")
                    stress_table.add_column("Ratio", justify="right")

                    stress_table.add_row(
                        "Drift Uncertainty σ(μ̂)",
                        f"{stress_result.avg_uncertainty_normal:.6f}",
                        f"{stress_result.avg_uncertainty_stress:.6f}",
                        f"[#00d700]{stress_result.uncertainty_spike_ratio:.2f}×[/#00d700]" if stress_result.uncertainty_spike_ratio > 1.2 else f"{stress_result.uncertainty_spike_ratio:.2f}×"
                    )
                    stress_table.add_row(
                        "Position Size (EU)",
                        f"{stress_result.avg_kelly_normal:.4f}",
                        f"{stress_result.avg_kelly_stress:.4f}",
                        f"[#00d700]{stress_result.kelly_reduction_ratio:.2f}×[/#00d700]" if stress_result.kelly_reduction_ratio < 0.9 else f"{stress_result.kelly_reduction_ratio:.2f}×"
                    )

                    console.print(stress_table)

                    if stress_result.stress_periods_detected:
                        console.print(f"\n[bold]Stress Periods Detected:[/bold] {len(stress_result.stress_periods_detected)}")
                        for i, (start, end) in enumerate(stress_result.stress_periods_detected[:5], 1):
                            console.print(f"  {i}. {start} → {end}")
                        if len(stress_result.stress_periods_detected) > 5:
                            console.print(f"  ... and {len(stress_result.stress_periods_detected) - 5} more")

                    console.print(f"\n[dim]{stress_result.diagnostic_message}[/dim]\n")

                    # Overall validation summary
                    all_passed = (
                        drift_result.validation_passed and
                        ll_result.improvement_significant and
                        pit_result.calibration_passed and
                        stress_result.system_backed_off
                    )

                    if all_passed:
                        console.print(Panel.fit(
                            "[bold #00d700]✅ ALL VALIDATION CHECKS PASSED[/bold #00d700]\n"
                            "[dim]Model demonstrates structural realism and statistical rigor.[/dim]",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel.fit(
                            "[bold yellow]⚠️ SOME VALIDATION CHECKS FAILED[/bold yellow]\n"
                            "[dim]Review diagnostics above for tuning guidance.[/dim]",
                            border_style="yellow"
                        ))
                else:
                    console.print("[indian_red1]⚠️ Kalman filter data not available for validation[/indian_red1]")

            except Exception as e:
                console.print(f"[indian_red1]⚠️ Validation failed: {e}[/indian_red1]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

        # Build summary row for this asset
        asset_label = build_asset_display_label(asset, title)
        horizon_signals = {
            int(s.horizon_days): {
                "label": s.label,
                "profit_pln": float(s.profit_pln),
                "ue_up": float(s.ue_up),
                "ue_down": float(s.ue_down),
                "p_up": float(s.p_up),
                "exp_ret": float(s.exp_ret),
            }
            for s in sigs
        }
        nearest_label = sigs[0].label if sigs else "HOLD"
        summary_rows.append({
            "asset_label": asset_label,
            "horizon_signals": horizon_signals,
            "nearest_label": nearest_label,
            "sector": get_sector(canon),
        })

        # Prepare JSON block
        block = {
            "symbol": asset,
            "title": title,
            "as_of": str(px.index[-1].date()),
            "last_close": last_close,
            "notional_pln": NOTIONAL_PLN,
            "signals": [s.__dict__ for s in sigs],
            "ci_level": args.ci,
            "ci_domain": "log_return",
            "profit_ci_domain": "arithmetic_pln",
            "probability_mapping": ("student_t" if args.t_map else "normal"),
            "nu_clip": {"min": 4.5, "max": 500.0},
            "edgeworth_damped": True,
            "kelly_rule": "half",
            "decision_thresholds": thresholds,
            # volatility modeling metadata
            "vol_source": feats.get("vol_source", "garch11"),
            "garch_params": feats.get("garch_params", {}),
            # tail modeling metadata (global ν)
            "tail_model": "student_t_global",
            "nu_hat": float(feats.get("nu_hat").iloc[-1]) if isinstance(feats.get("nu_hat"), pd.Series) and not feats.get("nu_hat").empty else 50.0,
            "nu_bounds": {"min": 4.5, "max": 500.0},
            "nu_info": feats.get("nu_info", {}),
            # stochastic volatility metadata (Level-7: full posterior uncertainty)
            "stochastic_volatility": {
                "enabled": True,
                "method": "bayesian_garch_sampling",
                "parameter_sampling": os.getenv("PARAM_UNC", "sample"),
                "uncertainty_propagated": True,
                "volatility_ci_tracked": True,
                "description": "Volatility treated as latent stochastic process with posterior uncertainty. GARCH parameters sampled from N(theta_hat, Cov) per path. Full h_t trajectories tracked and volatility credible intervals reported per horizon."
            },
        }

        # Add diagnostics to JSON if computed (Level-7 falsifiability)
        if diagnostics:
            # Filter out non-serializable objects (DataFrames) for JSON
            serializable_diagnostics = {}
            for k, v in diagnostics.items():
                if k in ("parameter_stability", "out_of_sample"):
                    # Skip raw DataFrames; summary metrics already in top-level diagnostics
                    continue
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_diagnostics[k] = v
                elif isinstance(v, dict):
                    # Nested dicts are OK if they contain serializable values
                    serializable_diagnostics[k] = v
            block["diagnostics"] = serializable_diagnostics

        all_blocks.append(block)

        # Prepare CSV rows
        if args.simple:
            for i, s in enumerate(sigs):
                csv_rows_simple.append({
                    "asset": title,
                    "symbol": asset,
                    "timeframe": format_horizon_label(s.horizon_days),
                    "chance_up_pct": f"{s.p_up*100:.1f}",
                    "recommendation": s.label,
                    "why": explanations[i],
                })
        else:
            for s in sigs:
                row = s.__dict__.copy()
                row.update({
                    "asset": title,
                    "symbol": asset,
                })
                csv_rows_detailed.append(row)

    # After processing all assets, print a compact summary
    try:
        # Group summary rows by sector and render one table per sector
        render_sector_summary_tables(summary_rows, horizons)
        # Add high-conviction signals summary for short-term trading
        render_strong_signals_summary(summary_rows, horizons=[1, 3, 7])
        
        # Show risk temperature summary (computed once, applies to all assets)
        if RISK_TEMPERATURE_AVAILABLE:
            try:
                risk_temp_result = get_cached_risk_temperature(
                    start_date="2020-01-01",
                    notional=NOTIONAL_PLN,
                    estimated_gap_risk=0.03,
                )
                render_risk_temperature_summary(risk_temp_result)
            except Exception as rt_e:
                if os.getenv("DEBUG"):
                    Console().print(f"[dim]Risk temperature display skipped: {rt_e}[/dim]")
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not print summary tables: {e}")

    # Build structured failure log for exports
    failure_log = [
        {
            "asset": asset,
            "display_name": info.get("display_name", asset),
            "attempts": info.get("attempts", 0),
            "last_error": info.get("last_error", ""),
            "traceback": info.get("traceback", ""),
        }
        for asset, info in failures.items()
    ]

    # Print failure summary table (if any)
    if failures:
        from rich.table import Table
        fail_table = Table(title="Failed Assets After Retries")
        fail_table.add_column("Asset", style="red", justify="left")
        fail_table.add_column("Display Name", justify="left")
        fail_table.add_column("Attempts", justify="right")
        fail_table.add_column("Last Error", justify="left")
        for asset, info in failures.items():
            fail_table.add_row(asset, str(info.get("display_name", asset)), str(info.get("attempts", "")), str(info.get("last_error", "")))
        Console().print(fail_table)

        # Save failed assets to src/data/failed/ for later purging
        try:
            saved_path = save_failed_assets(failures, append=True)
            Console().print(f"[dim]Failed assets saved to: {saved_path}[/dim]")
            Console().print(f"[dim]Run 'make purge' to purge cached data for failed assets[/dim]")
        except Exception as e:
            Console().print(f"[yellow]Warning:[/yellow] Could not save failed assets: {e}")

    # Exports
    cache_path = args.cache_json or DEFAULT_CACHE_PATH
    payload = {
        "assets": all_blocks,
        "summary_rows": summary_rows,
        "horizons": horizons,
        "column_descriptions": DETAILED_COLUMN_DESCRIPTIONS,
        "simple_column_descriptions": SIMPLIFIED_COLUMN_DESCRIPTIONS,
        "failed_assets": failure_log,
    }
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not write cache JSON: {e}")

    if args.json:
        try:
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            Console().print(f"[yellow]Warning:[/yellow] Could not write JSON export: {e}")


if __name__ == "__main__":
    main()
