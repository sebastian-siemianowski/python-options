"""
===============================================================================
ADVANCED SCORING METRICS — CSS, FEC, DIG
===============================================================================

Three orthogonal metrics for elite model governance:

1. Calibration Stability Under Stress (CSS)
   - Measures calibration stability during volatility spikes/regime transitions
   - Penalizes models whose calibration collapses when needed most

2. Forecast Entropy Consistency (FEC)
   - Measures whether model's confidence is coherent with market uncertainty
   - Checks if the model "knows when it doesn't know"

3. Directional Information Gain (DIG)
   - Measures information about sign of future returns beyond naive baseline
   - Distinguishes "well-calibrated noise" from "actionable signal"

Reference: Lumen Mode Elite Quant Governance
Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import kstest, entropy as scipy_entropy
from enum import Enum


class StressRegime(Enum):
    """Market stress regime classification."""
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class CSSResult:
    """
    Calibration Stability Under Stress result.
    
    Attributes:
        css_score: Overall CSS score (0-1, higher = more stable)
        calibration_by_regime: PIT deviation per regime
        worst_regime: Regime with worst calibration
        stability_ratio: Ratio of crisis to stable calibration
        regime_counts: Number of observations per regime
    """
    css_score: float
    calibration_by_regime: Dict[str, float]
    worst_regime: str
    stability_ratio: float
    regime_counts: Dict[str, int]
    
    def to_dict(self) -> Dict:
        return {
            "css_score": self.css_score,
            "calibration_by_regime": self.calibration_by_regime,
            "worst_regime": self.worst_regime,
            "stability_ratio": self.stability_ratio,
            "regime_counts": self.regime_counts,
        }


@dataclass
class FECResult:
    """
    Forecast Entropy Consistency result.
    
    Attributes:
        fec_score: Overall FEC score (0-1, higher = more consistent)
        entropy_vol_correlation: Correlation between entropy and volatility
        overconfidence_count: Times model was overconfident in high vol
        paralysis_count: Times model had excessive uncertainty in low vol
        entropy_stats: Entropy statistics (mean, std, min, max)
    """
    fec_score: float
    entropy_vol_correlation: float
    overconfidence_count: int
    paralysis_count: int
    entropy_stats: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "fec_score": self.fec_score,
            "entropy_vol_correlation": self.entropy_vol_correlation,
            "overconfidence_count": self.overconfidence_count,
            "paralysis_count": self.paralysis_count,
            "entropy_stats": self.entropy_stats,
        }


@dataclass
class DIGResult:
    """
    Directional Information Gain result.
    
    Attributes:
        dig_score: Overall DIG score (0-1, higher = more informative)
        kl_divergence: KL divergence from naive baseline
        hit_rate: Simple directional accuracy (for reference only)
        information_ratio: Bits of information per prediction
        vol_weighted_dig: Volatility-weighted directional information
    """
    dig_score: float
    kl_divergence: float
    hit_rate: float
    information_ratio: float
    vol_weighted_dig: float
    
    def to_dict(self) -> Dict:
        return {
            "dig_score": self.dig_score,
            "kl_divergence": self.kl_divergence,
            "hit_rate": self.hit_rate,
            "information_ratio": self.information_ratio,
            "vol_weighted_dig": self.vol_weighted_dig,
        }


def classify_stress_regime(
    vol: np.ndarray,
    lookback: int = 252,
) -> np.ndarray:
    """
    Classify each time point into a stress regime.
    
    Args:
        vol: Volatility series
        lookback: Lookback window for percentile computation
        
    Returns:
        Array of StressRegime enum values
    """
    n = len(vol)
    regimes = np.empty(n, dtype=object)
    
    for t in range(n):
        start = max(0, t - lookback)
        historical_vol = vol[start:t] if t > start else vol[:max(1, t)]
        
        if len(historical_vol) < 10:
            regimes[t] = StressRegime.NORMAL
            continue
        
        p25, p50, p75, p95 = np.percentile(historical_vol, [25, 50, 75, 95])
        current_vol = vol[t]
        
        if current_vol > p95:
            regimes[t] = StressRegime.CRISIS
        elif current_vol > p75:
            regimes[t] = StressRegime.HIGH_VOL
        elif current_vol < p25:
            regimes[t] = StressRegime.LOW_VOL
        else:
            regimes[t] = StressRegime.NORMAL
    
    return regimes


def compute_css(
    pit_values: np.ndarray,
    vol: np.ndarray,
    warmup: int = 60,
) -> CSSResult:
    """
    Compute Calibration Stability Under Stress (CSS).
    
    CSS measures how stable probabilistic calibration remains during
    volatility spikes, drawdowns, and regime transitions.
    
    Args:
        pit_values: PIT values from model predictions
        vol: Volatility series
        warmup: Warmup period to skip
        
    Returns:
        CSSResult with calibration stability metrics
    """
    pit_clean = pit_values[warmup:]
    vol_clean = vol[warmup:]
    
    if len(pit_clean) < 100:
        return CSSResult(
            css_score=0.5,
            calibration_by_regime={},
            worst_regime="unknown",
            stability_ratio=1.0,
            regime_counts={},
        )
    
    regimes = classify_stress_regime(vol_clean)
    calibration_by_regime = {}
    regime_counts = {}
    
    for regime in StressRegime:
        mask = regimes == regime
        regime_pit = pit_clean[mask]
        regime_counts[regime.value] = int(np.sum(mask))
        
        if len(regime_pit) < 20:
            calibration_by_regime[regime.value] = 0.5
            continue
        
        regime_pit_valid = regime_pit[(regime_pit > 0.001) & (regime_pit < 0.999)]
        if len(regime_pit_valid) < 10:
            calibration_by_regime[regime.value] = 0.5
            continue
        
        try:
            _, pvalue = kstest(regime_pit_valid, 'uniform')
            calibration_by_regime[regime.value] = float(pvalue)
        except:
            calibration_by_regime[regime.value] = 0.5
    
    worst_regime = min(calibration_by_regime, key=calibration_by_regime.get) if calibration_by_regime else "unknown"
    
    stable_cal = calibration_by_regime.get(StressRegime.LOW_VOL.value, 0.5) + calibration_by_regime.get(StressRegime.NORMAL.value, 0.5)
    stress_cal = calibration_by_regime.get(StressRegime.HIGH_VOL.value, 0.5) + calibration_by_regime.get(StressRegime.CRISIS.value, 0.5)
    
    if stable_cal > 0.001:
        stability_ratio = stress_cal / stable_cal
    else:
        stability_ratio = 0.0
    
    css_components = []
    for regime, pvalue in calibration_by_regime.items():
        if regime in [StressRegime.HIGH_VOL.value, StressRegime.CRISIS.value]:
            weight = 2.0
        else:
            weight = 1.0
        css_components.append(min(1.0, pvalue / 0.05) * weight)
    
    total_weight = len([r for r in calibration_by_regime if r in [StressRegime.HIGH_VOL.value, StressRegime.CRISIS.value]]) * 2 + \
                   len([r for r in calibration_by_regime if r not in [StressRegime.HIGH_VOL.value, StressRegime.CRISIS.value]])
    
    if total_weight > 0 and css_components:
        css_score = sum(css_components) / total_weight
    else:
        css_score = 0.5
    
    penalty = max(0, 0.5 - stability_ratio) * 0.3
    css_score = max(0, css_score - penalty)
    
    return CSSResult(
        css_score=float(np.clip(css_score, 0, 1)),
        calibration_by_regime=calibration_by_regime,
        worst_regime=worst_regime,
        stability_ratio=float(stability_ratio),
        regime_counts=regime_counts,
    )


def compute_fec(
    sigma_pred: np.ndarray,
    vol: np.ndarray,
    warmup: int = 60,
) -> FECResult:
    """
    Compute Forecast Entropy Consistency (FEC).
    
    FEC measures whether model's confidence (entropy of predictive distribution)
    behaves rationally relative to market uncertainty:
    - High vol -> entropy should rise
    - Low vol -> entropy should contract
    
    Args:
        sigma_pred: Predicted standard deviations from model
        vol: Market volatility series
        warmup: Warmup period to skip
        
    Returns:
        FECResult with entropy consistency metrics
    """
    sigma_clean = sigma_pred[warmup:]
    vol_clean = vol[warmup:]
    
    if len(sigma_clean) < 100:
        return FECResult(
            fec_score=0.5,
            entropy_vol_correlation=0.0,
            overconfidence_count=0,
            paralysis_count=0,
            entropy_stats={},
        )
    
    sigma_safe = np.maximum(sigma_clean, 1e-8)
    forecast_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma_safe**2)
    
    vol_safe = np.maximum(vol_clean, 1e-8)
    market_entropy = 0.5 * np.log(2 * np.pi * np.e * vol_safe**2)
    
    valid_mask = np.isfinite(forecast_entropy) & np.isfinite(market_entropy)
    forecast_entropy = forecast_entropy[valid_mask]
    market_entropy = market_entropy[valid_mask]
    vol_clean = vol_clean[valid_mask]
    
    if len(forecast_entropy) < 50:
        return FECResult(
            fec_score=0.5,
            entropy_vol_correlation=0.0,
            overconfidence_count=0,
            paralysis_count=0,
            entropy_stats={},
        )
    
    try:
        correlation = np.corrcoef(forecast_entropy, market_entropy)[0, 1]
        if not np.isfinite(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    vol_high_threshold = np.percentile(vol_clean, 75)
    vol_low_threshold = np.percentile(vol_clean, 25)
    entropy_median = np.median(forecast_entropy)
    
    high_vol_mask = vol_clean > vol_high_threshold
    low_vol_mask = vol_clean < vol_low_threshold
    
    overconfidence_count = int(np.sum(high_vol_mask & (forecast_entropy < entropy_median)))
    paralysis_count = int(np.sum(low_vol_mask & (forecast_entropy > entropy_median * 1.5)))
    
    entropy_stats = {
        "mean": float(np.mean(forecast_entropy)),
        "std": float(np.std(forecast_entropy)),
        "min": float(np.min(forecast_entropy)),
        "max": float(np.max(forecast_entropy)),
    }
    
    correlation_score = (correlation + 1) / 2
    
    n_high_vol = max(1, np.sum(high_vol_mask))
    n_low_vol = max(1, np.sum(low_vol_mask))
    overconfidence_rate = overconfidence_count / n_high_vol
    paralysis_rate = paralysis_count / n_low_vol
    
    behavior_score = 1.0 - 0.5 * overconfidence_rate - 0.5 * paralysis_rate
    
    fec_score = 0.6 * correlation_score + 0.4 * behavior_score
    
    return FECResult(
        fec_score=float(np.clip(fec_score, 0, 1)),
        entropy_vol_correlation=float(correlation),
        overconfidence_count=overconfidence_count,
        paralysis_count=paralysis_count,
        entropy_stats=entropy_stats,
    )


def compute_dig(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    returns: np.ndarray,
    vol: np.ndarray,
    warmup: int = 60,
) -> DIGResult:
    """
    Compute Directional Information Gain (DIG).
    
    DIG measures how much information about the sign of future returns
    the model provides beyond a naive baseline (symmetric zero-mean distribution).
    
    This is NOT hit rate or Sharpe - it's information gain (bits).
    
    Args:
        mu_pred: Predicted means from model
        sigma_pred: Predicted standard deviations
        returns: Actual realized returns
        vol: Market volatility series
        warmup: Warmup period to skip
        
    Returns:
        DIGResult with directional information metrics
    """
    mu_clean = mu_pred[warmup:]
    sigma_clean = sigma_pred[warmup:]
    returns_clean = returns[warmup:]
    vol_clean = vol[warmup:]
    
    if len(mu_clean) < 100:
        return DIGResult(
            dig_score=0.5,
            kl_divergence=0.0,
            hit_rate=0.5,
            information_ratio=0.0,
            vol_weighted_dig=0.0,
        )
    
    from scipy.stats import norm
    
    sigma_safe = np.maximum(sigma_clean, 1e-8)
    prob_positive = norm.cdf(0, loc=-mu_clean, scale=sigma_safe)
    prob_positive = np.clip(prob_positive, 0.001, 0.999)
    
    actual_positive = (returns_clean > 0).astype(float)
    
    model_probs = np.column_stack([1 - prob_positive, prob_positive])
    baseline_probs = np.array([[0.5, 0.5]] * len(prob_positive))
    
    kl_divs = []
    for i in range(len(prob_positive)):
        p = model_probs[i]
        q = baseline_probs[i]
        p_safe = np.clip(p, 1e-10, 1 - 1e-10)
        q_safe = np.clip(q, 1e-10, 1 - 1e-10)
        kl = np.sum(p_safe * np.log(p_safe / q_safe))
        if np.isfinite(kl):
            kl_divs.append(kl)
    
    kl_divergence = float(np.mean(kl_divs)) if kl_divs else 0.0
    
    predicted_direction = (prob_positive > 0.5).astype(float)
    hit_rate = float(np.mean(predicted_direction == actual_positive))
    
    bits_per_prediction = kl_divergence / np.log(2) if kl_divergence > 0 else 0.0
    
    vol_weights = vol_clean / np.mean(vol_clean)
    vol_weighted_kl = []
    for i, kl in enumerate(kl_divs[:len(vol_weights)]):
        vol_weighted_kl.append(kl * vol_weights[i])
    vol_weighted_dig = float(np.mean(vol_weighted_kl)) if vol_weighted_kl else 0.0
    
    kl_score = min(1.0, kl_divergence / 0.1)
    
    hit_rate_excess = max(0, hit_rate - 0.5)
    hit_score = min(1.0, hit_rate_excess / 0.1)
    
    dig_score = 0.7 * kl_score + 0.3 * hit_score
    
    return DIGResult(
        dig_score=float(np.clip(dig_score, 0, 1)),
        kl_divergence=float(kl_divergence),
        hit_rate=float(hit_rate),
        information_ratio=float(bits_per_prediction),
        vol_weighted_dig=float(vol_weighted_dig),
    )


@dataclass
class AdvancedScoringResult:
    """
    Combined result of all advanced scoring metrics.
    
    Attributes:
        css: Calibration Stability Under Stress
        fec: Forecast Entropy Consistency
        dig: Directional Information Gain
        advanced_score: Combined advanced score
    """
    css: CSSResult
    fec: FECResult
    dig: DIGResult
    advanced_score: float
    
    def to_dict(self) -> Dict:
        return {
            "css": self.css.to_dict(),
            "fec": self.fec.to_dict(),
            "dig": self.dig.to_dict(),
            "advanced_score": self.advanced_score,
        }


def compute_advanced_scores(
    pit_values: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    returns: np.ndarray,
    vol: np.ndarray,
    warmup: int = 60,
    css_weight: float = 0.4,
    fec_weight: float = 0.35,
    dig_weight: float = 0.25,
) -> AdvancedScoringResult:
    """
    Compute all advanced scoring metrics.
    
    Args:
        pit_values: PIT values from model
        mu_pred: Predicted means
        sigma_pred: Predicted standard deviations
        returns: Actual returns
        vol: Market volatility
        warmup: Warmup period
        css_weight: Weight for CSS in combined score
        fec_weight: Weight for FEC in combined score
        dig_weight: Weight for DIG in combined score
        
    Returns:
        AdvancedScoringResult with all metrics
    """
    css = compute_css(pit_values, vol, warmup)
    fec = compute_fec(sigma_pred, vol, warmup)
    dig = compute_dig(mu_pred, sigma_pred, returns, vol, warmup)
    
    advanced_score = (
        css_weight * css.css_score +
        fec_weight * fec.fec_score +
        dig_weight * dig.dig_score
    )
    
    return AdvancedScoringResult(
        css=css,
        fec=fec,
        dig=dig,
        advanced_score=float(advanced_score),
    )
