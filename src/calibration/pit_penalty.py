"""
===============================================================================
PIT VIOLATION PENALTY MODULE
===============================================================================

Implements Asymmetric PIT Violation Penalty (February 2026)

CORE DESIGN CONSTRAINT (Non-Negotiable):
    "PIT must only act as a penalty, never as a reward."

This module provides:
    1. Regime-conditional PIT critical thresholds
    2. One-sided violation severity computation
    3. Exponential penalty multiplier
    4. Sparse data safeguards
    5. EXIT signal trigger logic

BEHAVIORAL PHILOSOPHY:
    - Good PIT → neutral (P = 1.0, no effect)
    - Borderline PIT → neutral (P = 1.0, no effect)
    - Bad PIT → penalized (P < 1.0, model demoted)

PIT does NOT boost models. It only speaks when something is wrong.

MATHEMATICAL MODEL:

    Step 1: Violation Severity (One-Sided)
        V_{m,r} = max(0, (p_crit - p_{m,r}^PIT) / p_crit)
    
    Step 2: Penalty Multiplier
        P_{m,r}^PIT = exp(-λ_r × V_{m,r})
    
    Step 3: Adjusted Model Score
        S_{m,r}^adjusted = S_{m,r}^fit × P_{m,r}^PIT

    Properties:
        - p >= p_crit  →  V = 0  →  P = 1 (no penalty)
        - p → 0        →  V → 1  →  P → exp(-λ) (maximum penalty)

SPARSE DATA SAFEGUARD:

    P_{m,r}^eff = ω_r × P_{m,r}^PIT + (1 - ω_r) × 1.0
    
    Where ω_r = min(1, N_r / N_min)
    
    If observations are sparse, penalty fades toward neutral.

EXIT SIGNAL TRIGGER:

    When the SELECTED model has critical PIT violation:
        - Signal MUST be EXIT
        - No directional forecast
        - Position closure required

REFERENCES:
    Chinese Staff Professor Panel (February 2026)
    Institutional-Grade Calibration Governance

===============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME DEFINITIONS (Must match tune.py MarketRegime)
# =============================================================================

class MarketRegime(IntEnum):
    """Market regime labels - must match tune.py exactly."""
    LOW_VOL_TREND = 0
    HIGH_VOL_TREND = 1
    LOW_VOL_RANGE = 2
    HIGH_VOL_RANGE = 3
    CRISIS_JUMP = 4


REGIME_NAMES = {
    0: "LOW_VOL_TREND",
    1: "HIGH_VOL_TREND",
    2: "LOW_VOL_RANGE",
    3: "HIGH_VOL_RANGE",
    4: "CRISIS_JUMP",
}


# =============================================================================
# PIT VIOLATION CONSTANTS
# =============================================================================

# Critical PIT p-value thresholds (per regime)
# Lower threshold = stricter (only severe violations trigger penalty)
# Calm regimes: strict (p_crit = 0.01) - we expect good calibration
# Crisis regimes: relaxed (p_crit = 0.05) - PIT is less reliable
PIT_CRITICAL_THRESHOLDS = {
    MarketRegime.LOW_VOL_TREND: 0.01,   # Strict - calm trending market
    MarketRegime.HIGH_VOL_TREND: 0.02,  # Slightly relaxed - volatile trending
    MarketRegime.LOW_VOL_RANGE: 0.01,   # Strict - calm ranging market
    MarketRegime.HIGH_VOL_RANGE: 0.03,  # Relaxed - volatile ranging
    MarketRegime.CRISIS_JUMP: 0.05,     # Most relaxed - crisis conditions
}

# Default threshold for global/unknown regime
PIT_CRITICAL_THRESHOLD_DEFAULT = 0.01

# Penalty severity (λ) per regime
# Higher λ = faster trust collapse for PIT violations
# Crisis regimes have LOWER λ because PIT is less reliable there
PIT_PENALTY_LAMBDAS = {
    MarketRegime.LOW_VOL_TREND: 5.0,    # Strong penalty in calm markets
    MarketRegime.HIGH_VOL_TREND: 4.0,   # Moderate penalty
    MarketRegime.LOW_VOL_RANGE: 5.0,    # Strong penalty in calm markets
    MarketRegime.HIGH_VOL_RANGE: 3.0,   # Reduced penalty
    MarketRegime.CRISIS_JUMP: 2.0,      # Weak penalty - PIT unreliable in crisis
}

# Default lambda for global/unknown regime
PIT_PENALTY_LAMBDA_DEFAULT = 4.0

# Minimum observations for reliable PIT diagnostics
PIT_MIN_RELIABLE_SAMPLES = 50

# EXIT signal threshold - if effective penalty drops below this, trigger EXIT
# This means the model's calibration is so poor we cannot trust any forecast
PIT_EXIT_THRESHOLD = 0.3  # P < 0.3 triggers EXIT

# Logging threshold - log when PIT penalty significantly affects selection
PIT_SELECTION_DIVERGENCE_LOG_THRESHOLD = 0.1  # Log if weight changes by >10%


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PITViolationResult:
    """Result of PIT violation computation for a single model."""
    model_name: str
    regime: int
    regime_name: str
    
    # Input values
    pit_pvalue: float
    p_critical: float
    n_samples: int
    
    # Computed values
    violation_severity: float       # V ∈ [0, 1], 0 = no violation
    raw_penalty: float              # P = exp(-λV) ∈ (0, 1]
    effective_penalty: float        # P_eff after sparse data adjustment
    
    # Flags
    is_violated: bool               # True if p < p_crit
    sparse_data_adjustment: float   # ω ∈ [0, 1]
    triggers_exit: bool             # True if effective_penalty < EXIT_THRESHOLD
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "pit_pvalue": float(self.pit_pvalue),
            "p_critical": float(self.p_critical),
            "n_samples": self.n_samples,
            "violation_severity": float(self.violation_severity),
            "raw_penalty": float(self.raw_penalty),
            "effective_penalty": float(self.effective_penalty),
            "is_violated": self.is_violated,
            "sparse_data_adjustment": float(self.sparse_data_adjustment),
            "triggers_exit": self.triggers_exit,
        }


@dataclass
class PITPenaltyReport:
    """Complete PIT penalty report for all models in a regime."""
    regime: int
    regime_name: str
    n_samples: int
    
    # Per-model results
    model_penalties: Dict[str, PITViolationResult]
    
    # Aggregate statistics
    n_violated: int
    n_exit_triggered: int
    max_penalty_model: Optional[str]
    max_penalty_value: float
    
    # Selection impact
    best_model_by_fit: Optional[str]           # argmax S^fit
    best_model_after_penalty: Optional[str]    # argmax (S^fit × P^PIT)
    selection_diverged: bool                   # True if penalty changed selection
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "regime_name": self.regime_name,
            "n_samples": self.n_samples,
            "model_penalties": {k: v.to_dict() for k, v in self.model_penalties.items()},
            "n_violated": self.n_violated,
            "n_exit_triggered": self.n_exit_triggered,
            "max_penalty_model": self.max_penalty_model,
            "max_penalty_value": float(self.max_penalty_value),
            "best_model_by_fit": self.best_model_by_fit,
            "best_model_after_penalty": self.best_model_after_penalty,
            "selection_diverged": self.selection_diverged,
        }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_pit_critical_threshold(regime: int) -> float:
    """
    Get the critical PIT p-value threshold for a regime.
    
    Below this threshold, a model is considered to have a PIT violation.
    
    Args:
        regime: Regime index (0-4) or -1 for global
        
    Returns:
        Critical threshold p_crit
    """
    if regime in PIT_CRITICAL_THRESHOLDS:
        return PIT_CRITICAL_THRESHOLDS[regime]
    return PIT_CRITICAL_THRESHOLD_DEFAULT


def get_pit_penalty_lambda(regime: int) -> float:
    """
    Get the penalty severity λ for a regime.
    
    Higher λ = faster trust collapse when PIT is violated.
    
    Args:
        regime: Regime index (0-4) or -1 for global
        
    Returns:
        Penalty severity λ
    """
    if regime in PIT_PENALTY_LAMBDAS:
        return PIT_PENALTY_LAMBDAS[regime]
    return PIT_PENALTY_LAMBDA_DEFAULT


def compute_pit_violation_severity(
    pit_pvalue: float,
    p_critical: float
) -> float:
    """
    Compute one-sided PIT violation severity.
    
    V = max(0, (p_crit - p) / p_crit)
    
    Properties:
        - p >= p_crit → V = 0 (no violation)
        - p = 0 → V = 1 (maximum violation)
        - Smooth, bounded ∈ [0, 1]
    
    Args:
        pit_pvalue: PIT KS test p-value
        p_critical: Critical threshold
        
    Returns:
        Violation severity V ∈ [0, 1]
    """
    if pit_pvalue is None or not np.isfinite(pit_pvalue):
        # Missing PIT data - assume no violation (conservative)
        return 0.0
    
    if p_critical <= 0:
        return 0.0
    
    # One-sided: only penalize if below threshold
    violation = (p_critical - pit_pvalue) / p_critical
    return max(0.0, min(1.0, violation))


def compute_pit_penalty(
    violation_severity: float,
    lambda_penalty: float
) -> float:
    """
    Convert violation severity to penalty multiplier.
    
    P = exp(-λ × V)
    
    Properties:
        - V = 0 → P = 1 (no penalty, model unaffected)
        - V = 1 → P = exp(-λ) (maximum penalty)
        - Smooth, bounded ∈ (0, 1]
    
    Args:
        violation_severity: V ∈ [0, 1]
        lambda_penalty: Severity parameter λ > 0
        
    Returns:
        Penalty multiplier P ∈ (0, 1]
    """
    if violation_severity <= 0:
        return 1.0  # No violation → no penalty
    
    return math.exp(-lambda_penalty * violation_severity)


def compute_effective_penalty(
    raw_penalty: float,
    n_samples: int,
    min_samples: int = PIT_MIN_RELIABLE_SAMPLES
) -> Tuple[float, float]:
    """
    Apply sparse data adjustment to penalty.
    
    P_eff = ω × P + (1 - ω) × 1.0
    
    Where ω = min(1, N / N_min)
    
    If we have insufficient data to trust PIT diagnostics,
    the penalty fades toward neutral (1.0).
    
    Args:
        raw_penalty: Raw penalty P from compute_pit_penalty
        n_samples: Number of observations
        min_samples: Minimum for reliable PIT
        
    Returns:
        Tuple of (effective_penalty, omega)
    """
    if n_samples >= min_samples:
        omega = 1.0
    else:
        omega = n_samples / min_samples
    
    effective = omega * raw_penalty + (1.0 - omega) * 1.0
    return effective, omega


def compute_model_pit_penalty(
    model_name: str,
    pit_pvalue: float,
    regime: int,
    n_samples: int,
    min_samples: int = PIT_MIN_RELIABLE_SAMPLES
) -> PITViolationResult:
    """
    Compute complete PIT violation penalty for a single model.
    
    This is the main entry point for per-model PIT penalty computation.
    
    Args:
        model_name: Name of the model
        pit_pvalue: PIT KS test p-value
        regime: Regime index (0-4) or -1 for global
        n_samples: Number of observations in regime
        min_samples: Minimum samples for reliable PIT
        
    Returns:
        PITViolationResult with all computed values
    """
    # Get regime-specific parameters
    p_critical = get_pit_critical_threshold(regime)
    lambda_penalty = get_pit_penalty_lambda(regime)
    regime_name = REGIME_NAMES.get(regime, "GLOBAL")
    
    # Step 1: Compute violation severity
    violation = compute_pit_violation_severity(pit_pvalue, p_critical)
    
    # Step 2: Compute raw penalty
    raw_penalty = compute_pit_penalty(violation, lambda_penalty)
    
    # Step 3: Apply sparse data adjustment
    effective_penalty, omega = compute_effective_penalty(
        raw_penalty, n_samples, min_samples
    )
    
    # Determine flags
    is_violated = pit_pvalue is not None and pit_pvalue < p_critical
    triggers_exit = effective_penalty < PIT_EXIT_THRESHOLD
    
    return PITViolationResult(
        model_name=model_name,
        regime=regime,
        regime_name=regime_name,
        pit_pvalue=pit_pvalue if pit_pvalue is not None else float('nan'),
        p_critical=p_critical,
        n_samples=n_samples,
        violation_severity=violation,
        raw_penalty=raw_penalty,
        effective_penalty=effective_penalty,
        is_violated=is_violated,
        sparse_data_adjustment=omega,
        triggers_exit=triggers_exit,
    )


def apply_pit_penalties_to_weights(
    raw_weights: Dict[str, float],
    model_pit_pvalues: Dict[str, float],
    regime: int,
    n_samples: int,
    fit_scores: Optional[Dict[str, float]] = None,
    min_samples: int = PIT_MIN_RELIABLE_SAMPLES
) -> Tuple[Dict[str, float], PITPenaltyReport]:
    """
    Apply PIT violation penalties to model weights.
    
    This modifies raw_weights by multiplying each by its PIT penalty:
        w_adjusted(m) = w_raw(m) × P_{m,r}^PIT
    
    Then re-normalizes to ensure weights sum to 1.
    
    CRITICAL: This only REDUCES weights, never increases them.
    Good PIT → P = 1 → weight unchanged.
    
    Args:
        raw_weights: Dictionary of model name → raw BMA weight
        model_pit_pvalues: Dictionary of model name → PIT KS p-value
        regime: Regime index (0-4) or -1 for global
        n_samples: Number of observations in regime
        fit_scores: Optional dict of model → fit score (for selection divergence check)
        min_samples: Minimum samples for reliable PIT
        
    Returns:
        Tuple of (adjusted_weights, PITPenaltyReport)
    """
    regime_name = REGIME_NAMES.get(regime, "GLOBAL")
    
    # Compute penalties for each model
    model_penalties = {}
    adjusted_weights = {}
    
    for model_name, raw_weight in raw_weights.items():
        pit_p = model_pit_pvalues.get(model_name)
        
        result = compute_model_pit_penalty(
            model_name=model_name,
            pit_pvalue=pit_p,
            regime=regime,
            n_samples=n_samples,
            min_samples=min_samples
        )
        
        model_penalties[model_name] = result
        
        # Apply penalty to weight
        adjusted_weights[model_name] = raw_weight * result.effective_penalty
    
    # Normalize adjusted weights
    total = sum(adjusted_weights.values())
    if total > 0:
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
    
    # Compute aggregate statistics
    n_violated = sum(1 for r in model_penalties.values() if r.is_violated)
    n_exit_triggered = sum(1 for r in model_penalties.values() if r.triggers_exit)
    
    # Find most penalized model
    max_penalty_model = None
    max_penalty_value = 1.0
    for model_name, result in model_penalties.items():
        if result.effective_penalty < max_penalty_value:
            max_penalty_value = result.effective_penalty
            max_penalty_model = model_name
    
    # Check if selection diverged due to penalties
    best_by_fit = max(raw_weights.items(), key=lambda x: x[1])[0] if raw_weights else None
    best_after_penalty = max(adjusted_weights.items(), key=lambda x: x[1])[0] if adjusted_weights else None
    selection_diverged = best_by_fit != best_after_penalty
    
    # Log if selection diverged
    if selection_diverged:
        logger.info(
            f"PIT penalty changed model selection in regime {regime_name}: "
            f"{best_by_fit} → {best_after_penalty}"
        )
    
    report = PITPenaltyReport(
        regime=regime,
        regime_name=regime_name,
        n_samples=n_samples,
        model_penalties=model_penalties,
        n_violated=n_violated,
        n_exit_triggered=n_exit_triggered,
        max_penalty_model=max_penalty_model,
        max_penalty_value=max_penalty_value,
        best_model_by_fit=best_by_fit,
        best_model_after_penalty=best_after_penalty,
        selection_diverged=selection_diverged,
    )
    
    return adjusted_weights, report


def check_exit_signal_required(
    selected_model: str,
    pit_penalty_report: PITPenaltyReport
) -> Tuple[bool, Optional[str]]:
    """
    Check if an EXIT signal is required due to critical PIT violation.
    
    EXIT is triggered when the SELECTED model has a severe PIT violation
    that drops its effective penalty below PIT_EXIT_THRESHOLD.
    
    Args:
        selected_model: Name of the selected model
        pit_penalty_report: PITPenaltyReport from apply_pit_penalties_to_weights
        
    Returns:
        Tuple of (exit_required, exit_reason)
    """
    if selected_model not in pit_penalty_report.model_penalties:
        return False, None
    
    result = pit_penalty_report.model_penalties[selected_model]
    
    if result.triggers_exit:
        reason = (
            f"Critical PIT violation for {selected_model} in regime {result.regime_name}: "
            f"p={result.pit_pvalue:.4f} < p_crit={result.p_critical:.4f}, "
            f"effective_penalty={result.effective_penalty:.3f} < {PIT_EXIT_THRESHOLD:.2f}"
        )
        return True, reason
    
    return False, None


def get_pit_critical_stocks(
    cache: Dict[str, Dict],
    threshold: float = 0.05
) -> List[str]:
    """
    Identify stocks with critical PIT calibration issues.
    
    A stock is PIT-critical if pit_ks_pvalue < threshold for the SELECTED model.
    
    IMPORTANT (February 2026): This checks only the selected/best model's PIT,
    not all models. The unified Student-t model may have good calibration even
    if other models failed.
    
    Args:
        cache: The tuning cache dictionary
        threshold: PIT p-value threshold for critical status
        
    Returns:
        List of asset symbols with critical PIT issues
    """
    critical_stocks = []
    
    for asset, data in cache.items():
        # Handle regime-conditional structure
        if 'global' in data:
            global_data = data['global']
        else:
            global_data = data
        
        # First, check the top-level pit_ks_pvalue (this is for the SELECTED model)
        pit_p = global_data.get('pit_ks_pvalue')
        
        # If no top-level pit_p, check model_comparison for selected model
        if pit_p is None:
            model_comparison = global_data.get('model_comparison', {})
            selected_model = model_comparison.get('selected_model')
            
            if selected_model and 'models' in global_data:
                model_data = global_data['models'].get(selected_model, {})
                pit_p = model_data.get('pit_ks_pvalue')
        
        
        # Only mark as critical if we have a p-value AND it's below threshold
        if pit_p is not None and pit_p < threshold:
            critical_stocks.append(asset)
    
    return sorted(set(critical_stocks))


def is_pit_critical(
    asset: str,
    cache: Dict[str, Dict],
    threshold: float = 0.05
) -> bool:
    """
    Check if a specific asset has critical PIT calibration issues.
    
    IMPORTANT (February 2026): This checks only the SELECTED model's PIT p-value,
    not all models. The model selection is done during tuning and the selected
    model's pit_ks_pvalue is stored at the global/top level.
    
    Args:
        asset: Asset symbol
        cache: The tuning cache dictionary
        threshold: PIT p-value threshold
        
    Returns:
        True if the asset has pit_ks_pvalue < threshold for selected model
    """
    if asset not in cache:
        return False
    
    data = cache[asset]
    if 'global' in data:
        global_data = data['global']
    else:
        global_data = data
    
    # The top-level pit_ks_pvalue is for the SELECTED model
    pit_p = global_data.get('pit_ks_pvalue')
    
    # If no top-level pit_p, try to find it from the selected model
    if pit_p is None:
        model_comparison = global_data.get('model_comparison', {})
        selected_model = model_comparison.get('selected_model')
        
        # Also check noise_model which stores the selected model name
        if selected_model is None:
            noise_model = global_data.get('noise_model', '')
            if noise_model:
                selected_model = noise_model
        
        if selected_model and 'models' in global_data:
            model_data = global_data['models'].get(selected_model, {})
            pit_p = model_data.get('pit_ks_pvalue')
    
    # Only return True if we have a p-value AND it's below threshold
    if pit_p is not None and pit_p < threshold:
        return True
    
    return False


# =============================================================================
# REGIME PROBABILITY MIXING (for uncertain regimes)
# =============================================================================

def compute_mixed_penalty(
    regime_penalties: Dict[int, float],
    regime_probabilities: Dict[int, float]
) -> float:
    """
    Compute mixed penalty under regime uncertainty.
    
    P = Σ_r π_r × P_r^eff
    
    This prevents regime-detection lag from causing abrupt behavior.
    
    Args:
        regime_penalties: Dict of regime → effective penalty
        regime_probabilities: Dict of regime → probability
        
    Returns:
        Mixed penalty P
    """
    total = 0.0
    prob_sum = 0.0
    
    for regime, penalty in regime_penalties.items():
        prob = regime_probabilities.get(regime, 0.0)
        total += prob * penalty
        prob_sum += prob
    
    if prob_sum > 0:
        return total / prob_sum
    return 1.0  # Default to no penalty


# =============================================================================
# DIAGNOSTIC LOGGING
# =============================================================================

def log_pit_penalty_divergence(
    asset: str,
    regime: int,
    best_by_fit: str,
    best_after_penalty: str,
    fit_weight_before: float,
    fit_weight_after: float,
    penalty_applied: float
) -> None:
    """
    Log when PIT penalty changes model selection.
    
    These events are diagnostic signals - often precede regime shifts.
    
    Args:
        asset: Asset symbol
        regime: Regime index
        best_by_fit: Best model by fit score
        best_after_penalty: Best model after penalty
        fit_weight_before: Weight of best_by_fit before penalty
        fit_weight_after: Weight of best_after_penalty after penalty
        penalty_applied: Penalty that caused the change
    """
    regime_name = REGIME_NAMES.get(regime, "GLOBAL")
    logger.warning(
        f"PIT PENALTY DIVERGENCE [{asset}] Regime={regime_name}: "
        f"{best_by_fit}(w={fit_weight_before:.3f}) → "
        f"{best_after_penalty}(w={fit_weight_after:.3f}) "
        f"[penalty={penalty_applied:.3f}]"
    )


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_pit_penalty_architecture() -> bool:
    """
    Verify that PIT penalty architecture is correctly configured.
    
    Returns:
        True if all checks pass
    """
    # Check all regimes have thresholds and lambdas
    for regime in MarketRegime:
        assert regime in PIT_CRITICAL_THRESHOLDS, f"Missing threshold for {regime}"
        assert regime in PIT_PENALTY_LAMBDAS, f"Missing lambda for {regime}"
        assert 0 < PIT_CRITICAL_THRESHOLDS[regime] <= 0.1, f"Invalid threshold for {regime}"
        assert PIT_PENALTY_LAMBDAS[regime] > 0, f"Invalid lambda for {regime}"
    
    # Check EXIT threshold is reasonable
    assert 0 < PIT_EXIT_THRESHOLD < 1.0, "Invalid EXIT threshold"
    
    # Check min samples is positive
    assert PIT_MIN_RELIABLE_SAMPLES > 0, "Invalid min samples"
    
    # Test penalty computation
    penalty = compute_pit_penalty(0.5, 4.0)
    assert 0 < penalty < 1, "Penalty computation failed"
    
    # Test no penalty for non-violation
    v = compute_pit_violation_severity(0.5, 0.01)
    assert v == 0.0, "Non-violation should have V=0"
    
    # Test penalty for violation
    v = compute_pit_violation_severity(0.001, 0.01)
    assert v > 0, "Violation should have V>0"
    
    logger.info("PIT penalty architecture verification PASSED")
    return True


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Verify architecture on import
try:
    verify_pit_penalty_architecture()
except AssertionError as e:
    logger.error(f"PIT penalty architecture verification FAILED: {e}")
