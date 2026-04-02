"""
Story 8.3: Conviction-Weighted Signal Ranking.

4-factor conviction score for ranking signals:
  1. Model agreement (BMA weight concentration)
  2. Regime confidence (how well data fits current regime)
  3. Historical accuracy (rolling hit rate)
  4. Forecast stability (consistency across horizons)

Usage:
    from decision.conviction_scoring import compute_conviction, ConvictionScore
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# Factor weights (sum to 1)
W_MODEL_AGREEMENT = 0.30
W_REGIME_CONFIDENCE = 0.25
W_HISTORICAL_ACCURACY = 0.25
W_FORECAST_STABILITY = 0.20

# Classification thresholds
HIGH_CONVICTION_THRESHOLD = 0.70
MEDIUM_CONVICTION_THRESHOLD = 0.40


@dataclass
class ConvictionScore:
    """Conviction assessment for a signal."""
    symbol: str
    composite: float             # Overall conviction [0, 1]
    model_agreement: float       # BMA concentration [0, 1]
    regime_confidence: float     # Regime fit [0, 1]
    historical_accuracy: float   # Rolling accuracy [0, 1]
    forecast_stability: float    # Cross-horizon consistency [0, 1]
    category: str                # "HIGH", "MEDIUM", "LOW"


def compute_model_agreement(bma_weights: np.ndarray) -> float:
    """
    Model agreement from BMA weight concentration.
    
    Uses normalized Herfindahl index:
    HHI = sum(w_i^2)
    Normalized: (HHI - 1/N) / (1 - 1/N)
    
    High concentration = models agree on one dominant model.
    
    Args:
        bma_weights: Array of BMA weights (sum to 1).
    
    Returns:
        Agreement score [0, 1]. 1 = perfect agreement.
    """
    if len(bma_weights) == 0:
        return 0.0
    
    weights = np.asarray(bma_weights, dtype=float)
    n = len(weights)
    
    if n == 1:
        return 1.0
    
    hhi = float(np.sum(weights ** 2))
    min_hhi = 1.0 / n
    
    if abs(1.0 - min_hhi) < 1e-10:
        return 0.0
    
    return (hhi - min_hhi) / (1.0 - min_hhi)


def compute_regime_confidence(regime_fit: float) -> float:
    """
    Regime confidence from regime fit score.
    
    Args:
        regime_fit: Regime fit metric from classification [0, 1].
    
    Returns:
        Regime confidence [0, 1].
    """
    return max(0.0, min(1.0, regime_fit))


def compute_historical_accuracy(
    recent_hits: np.ndarray,
    window: int = 20,
) -> float:
    """
    Historical accuracy from recent directional hit rate.
    
    Args:
        recent_hits: Boolean array (True = correct direction).
        window: Lookback window.
    
    Returns:
        Rolling hit rate [0, 1].
    """
    if len(recent_hits) == 0:
        return 0.5  # prior
    
    hits = np.asarray(recent_hits, dtype=float)
    if len(hits) > window:
        hits = hits[-window:]
    
    return float(np.mean(hits))


def compute_forecast_stability(
    forecasts_by_horizon: Dict[int, float],
) -> float:
    """
    Forecast stability: consistency of sign and magnitude across horizons.
    
    If all horizons agree on direction AND magnitude is monotonic,
    stability is high.
    
    Args:
        forecasts_by_horizon: {horizon: forecast_pct}.
    
    Returns:
        Stability score [0, 1].
    """
    if len(forecasts_by_horizon) < 2:
        return 0.5  # neutral
    
    values = np.array(list(forecasts_by_horizon.values()))
    signs = np.sign(values)
    
    # Sign consistency
    if np.all(signs == signs[0]) and signs[0] != 0:
        sign_score = 1.0
    elif np.sum(signs == signs[0]) >= len(signs) * 0.7:
        sign_score = 0.7
    else:
        sign_score = 0.3
    
    # Magnitude consistency (low CV = stable)
    abs_values = np.abs(values)
    mean_abs = np.mean(abs_values)
    if mean_abs > 1e-10:
        cv = float(np.std(abs_values) / mean_abs)
        magnitude_score = max(0.0, 1.0 - cv)
    else:
        magnitude_score = 0.5
    
    return 0.6 * sign_score + 0.4 * magnitude_score


def compute_conviction(
    symbol: str,
    bma_weights: np.ndarray,
    regime_fit: float,
    recent_hits: np.ndarray,
    forecasts_by_horizon: Dict[int, float],
) -> ConvictionScore:
    """
    Compute composite 4-factor conviction score.
    
    Args:
        symbol: Asset symbol.
        bma_weights: BMA model weights.
        regime_fit: Regime classification fit [0, 1].
        recent_hits: Recent directional accuracy (boolean array).
        forecasts_by_horizon: {horizon: forecast_pct}.
    
    Returns:
        ConvictionScore with composite and per-factor scores.
    """
    ma = compute_model_agreement(bma_weights)
    rc = compute_regime_confidence(regime_fit)
    ha = compute_historical_accuracy(recent_hits)
    fs = compute_forecast_stability(forecasts_by_horizon)
    
    composite = (
        W_MODEL_AGREEMENT * ma +
        W_REGIME_CONFIDENCE * rc +
        W_HISTORICAL_ACCURACY * ha +
        W_FORECAST_STABILITY * fs
    )
    
    if composite >= HIGH_CONVICTION_THRESHOLD:
        category = "HIGH"
    elif composite >= MEDIUM_CONVICTION_THRESHOLD:
        category = "MEDIUM"
    else:
        category = "LOW"
    
    return ConvictionScore(
        symbol=symbol,
        composite=composite,
        model_agreement=ma,
        regime_confidence=rc,
        historical_accuracy=ha,
        forecast_stability=fs,
        category=category,
    )


def rank_by_conviction(scores: List[ConvictionScore]) -> List[ConvictionScore]:
    """Rank signals by conviction score (descending)."""
    return sorted(scores, key=lambda s: s.composite, reverse=True)
