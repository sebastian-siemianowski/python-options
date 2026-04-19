"""
Story 11.3: Regime-Conditional Confidence Adjustment
=====================================================

Adjusts directional confidence scores based on regime-specific historical
predictability. Different regimes have different inherent predictability:
  - Trend regimes: more predictable -> boost confidence
  - Crisis regimes: less predictable -> reduce confidence

The key scaling formula:
    conf_adjusted = confidence * (hit_rate_regime / hit_rate_global)

This ensures that stated confidence reflects actual achievable accuracy
within each regime.

Regime definitions (from AGENTS.md):
    LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE, HIGH_VOL_RANGE, CRISIS_JUMP
"""
import os
import sys
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ===================================================================
# Constants
# ===================================================================

REGIME_NAMES = [
    "LOW_VOL_TREND",
    "HIGH_VOL_TREND",
    "LOW_VOL_RANGE",
    "HIGH_VOL_RANGE",
    "CRISIS_JUMP",
]

TRAILING_WINDOW = 252               # 1 year trailing for hit rate estimation
MIN_REGIME_SAMPLES = 20             # Minimum samples to estimate regime hit rate
DEFAULT_GLOBAL_HIT_RATE = 0.52      # Default global hit rate when insufficient data
CONFIDENCE_FLOOR = 0.0              # Minimum adjusted confidence
CONFIDENCE_CAP = 1.0                # Maximum adjusted confidence
TREND_BOOST_THRESHOLD = 0.55        # Boost confidence when hit rate > 55%
TREND_BOOST_FACTOR = 1.05           # 5% boost for trend regimes above threshold
CRISIS_SCALE_FLOOR = 0.5            # Don't reduce crisis confidence below 50% of original


# ===================================================================
# Result Dataclass
# ===================================================================

@dataclass
class RegimeConfidenceResult:
    """Result of regime-conditional confidence adjustment."""
    adjusted_confidence: float       # Adjusted confidence score
    regime: str                      # Regime name
    regime_hit_rate: float           # Historical hit rate for this regime
    global_hit_rate: float           # Global historical hit rate
    scale_factor: float              # Applied scaling factor
    n_regime_samples: int            # Number of samples used for regime hit rate
    used_default: bool               # Whether default rates were used


@dataclass
class RegimeHitRates:
    """Historical hit rates per regime."""
    rates: Dict[str, float]          # Regime name -> hit rate
    counts: Dict[str, int]           # Regime name -> sample count
    global_rate: float               # Overall hit rate
    total_count: int                 # Total samples


# ===================================================================
# Core Functions
# ===================================================================

def compute_regime_hit_rates(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    regimes: np.ndarray,
    trailing_window: int = TRAILING_WINDOW,
) -> RegimeHitRates:
    """Compute historical hit rates per regime from trailing data.

    Args:
        predictions: Predicted directions (+1 or -1 or P(r>0)), shape (n,)
        outcomes: Actual returns (positive = up), shape (n,)
        regimes: Regime labels at each time step, shape (n,)
        trailing_window: Use last N observations

    Returns:
        RegimeHitRates
    """
    n = len(predictions)
    start = max(0, n - trailing_window)

    preds = predictions[start:]
    outs = outcomes[start:]
    regs = regimes[start:]

    # Convert probabilities to binary predictions if needed
    if np.all((preds >= 0) & (preds <= 1)):
        pred_binary = (preds > 0.5).astype(float)
    else:
        pred_binary = (preds > 0).astype(float)

    out_binary = (outs > 0).astype(float)

    # Global hit rate
    correct = (pred_binary == out_binary)
    global_rate = float(correct.mean()) if len(correct) > 0 else DEFAULT_GLOBAL_HIT_RATE

    # Per-regime hit rates
    rates = {}
    counts = {}
    for regime in REGIME_NAMES:
        mask = (regs == regime)
        n_regime = mask.sum()
        counts[regime] = int(n_regime)

        if n_regime >= MIN_REGIME_SAMPLES:
            rates[regime] = float(correct[mask].mean())
        else:
            rates[regime] = global_rate  # Fallback to global

    return RegimeHitRates(
        rates=rates,
        counts=counts,
        global_rate=global_rate,
        total_count=len(preds),
    )


def regime_confidence_scale(
    confidence: float,
    regime: str,
    historical_hit_rates: RegimeHitRates,
) -> RegimeConfidenceResult:
    """Adjust confidence based on regime-specific historical predictability.

    Scaling formula:
        scale = hit_rate_regime / hit_rate_global

    For trend regimes with hit_rate > 55%: additional 5% boost.
    For crisis regimes: floor at 50% of original confidence.

    Args:
        confidence: Raw directional confidence in [0, 1]
        regime: Current regime name
        historical_hit_rates: Pre-computed regime hit rates

    Returns:
        RegimeConfidenceResult
    """
    hit_rates = historical_hit_rates
    global_rate = hit_rates.global_rate

    # Get regime-specific hit rate
    if regime in hit_rates.rates:
        regime_rate = hit_rates.rates[regime]
        n_samples = hit_rates.counts.get(regime, 0)
        used_default = n_samples < MIN_REGIME_SAMPLES
    else:
        regime_rate = global_rate
        n_samples = 0
        used_default = True

    # Compute scale factor
    if global_rate > 0:
        scale = regime_rate / global_rate
    else:
        scale = 1.0

    # Trend regime boost
    is_trend = regime in ("LOW_VOL_TREND", "HIGH_VOL_TREND")
    if is_trend and regime_rate > TREND_BOOST_THRESHOLD:
        scale *= TREND_BOOST_FACTOR

    # Crisis regime floor
    is_crisis = regime == "CRISIS_JUMP"
    if is_crisis:
        scale = max(scale, CRISIS_SCALE_FLOOR)

    # Apply scaling
    adjusted = confidence * scale
    adjusted = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CAP, adjusted))

    return RegimeConfidenceResult(
        adjusted_confidence=float(adjusted),
        regime=regime,
        regime_hit_rate=float(regime_rate),
        global_hit_rate=float(global_rate),
        scale_factor=float(scale),
        n_regime_samples=n_samples,
        used_default=used_default,
    )


def adjust_confidence_timeseries(
    confidences: np.ndarray,
    regimes: np.ndarray,
    predictions: np.ndarray,
    outcomes: np.ndarray,
    trailing_window: int = TRAILING_WINDOW,
) -> np.ndarray:
    """Walk-forward regime-conditional confidence adjustment.

    For each time step t, computes regime hit rates from [max(0, t-window), t-1]
    and adjusts confidence at t.

    Args:
        confidences: Raw confidence scores, shape (n,)
        regimes: Regime labels, shape (n,)
        predictions: Predicted directions, shape (n,)
        outcomes: Actual returns, shape (n,)
        trailing_window: Trailing window for hit rate estimation

    Returns:
        Adjusted confidence scores, shape (n,)
    """
    n = len(confidences)
    adjusted = confidences.copy().astype(float)

    for t in range(MIN_REGIME_SAMPLES, n):
        start = max(0, t - trailing_window)
        hit_rates = compute_regime_hit_rates(
            predictions[start:t],
            outcomes[start:t],
            regimes[start:t],
            trailing_window=trailing_window,
        )

        result = regime_confidence_scale(
            confidences[t], regimes[t], hit_rates
        )
        adjusted[t] = result.adjusted_confidence

    return adjusted


def compute_regime_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute ECE per regime.

    Args:
        probs: Predicted probabilities, shape (n,)
        labels: Binary outcomes, shape (n,)
        regimes: Regime labels, shape (n,)
        n_bins: Number of ECE bins

    Returns:
        Dict mapping regime name to ECE value
    """
    results = {}
    for regime in REGIME_NAMES:
        mask = (regimes == regime)
        n_regime = mask.sum()
        if n_regime < n_bins:
            results[regime] = float('nan')
            continue

        p = probs[mask]
        l = labels[mask]

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bmask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
            if i == n_bins - 1:
                bmask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
            nb = bmask.sum()
            if nb == 0:
                continue
            acc = l[bmask].mean()
            conf = p[bmask].mean()
            ece += (nb / n_regime) * abs(acc - conf)

        results[regime] = float(ece)

    return results
