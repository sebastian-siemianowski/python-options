"""
===============================================================================
COMBINED SCORING — Unified Score with Adaptive Weights
===============================================================================

Combines multiple scoring metrics into a single unified score:
- BIC (Bayesian Information Criterion) — model complexity penalty
- CRPS (Continuous Ranked Probability Score) — calibration + sharpness
- Hyvärinen Score — robustness to misspecification
- PIT Calibration — distributional correctness

The combined score can use:
1. Fixed weights (simple weighted average)
2. Adaptive weights (regime-dependent)
3. Pareto optimization (multi-objective frontier)

Reference: Gneiting & Raftery (2007), Wei Chen Panel Recommendation

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ScoringMethod(Enum):
    """Method for combining multiple scores."""
    FIXED_WEIGHTS = "fixed_weights"
    ADAPTIVE_WEIGHTS = "adaptive_weights"
    PARETO_FRONTIER = "pareto_frontier"
    CRPS_ONLY = "crps_only"


@dataclass
class ScoringConfig:
    """
    Configuration for combined scoring.
    
    Attributes:
        method: How to combine scores
        bic_weight: Weight for BIC (negative, so lower BIC = higher score)
        crps_weight: Weight for CRPS (negative, lower is better)
        hyvarinen_weight: Weight for Hyvärinen score
        pit_weight: Weight for PIT calibration (1 = calibrated, 0 = not)
        pit_hard_constraint: If True, uncalibrated models get score 0
        pit_threshold: PIT p-value threshold for calibration
        regime_adaptation: Enable regime-dependent weight adjustment
    """
    method: ScoringMethod = ScoringMethod.FIXED_WEIGHTS
    bic_weight: float = 0.25
    crps_weight: float = 0.35
    hyvarinen_weight: float = 0.15
    pit_weight: float = 0.25
    pit_hard_constraint: bool = True
    pit_threshold: float = 0.05
    regime_adaptation: bool = False
    
    def validate(self) -> bool:
        """Validate configuration."""
        total = self.bic_weight + self.crps_weight + self.hyvarinen_weight + self.pit_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return True


# Default configuration
DEFAULT_SCORING_CONFIG = ScoringConfig()

# Regime-specific weight adjustments
# Crisis: weight PIT heavily (calibration critical)
# Stable: weight BIC (efficiency matters)
REGIME_WEIGHT_ADJUSTMENTS = {
    "crisis": {
        "bic_weight": 0.15,
        "crps_weight": 0.30,
        "hyvarinen_weight": 0.10,
        "pit_weight": 0.45,
    },
    "high_vol": {
        "bic_weight": 0.20,
        "crps_weight": 0.35,
        "hyvarinen_weight": 0.15,
        "pit_weight": 0.30,
    },
    "stable": {
        "bic_weight": 0.35,
        "crps_weight": 0.30,
        "hyvarinen_weight": 0.15,
        "pit_weight": 0.20,
    },
}


@dataclass
class CombinedScoreResult:
    """
    Result of combined scoring computation.
    
    Attributes:
        combined_score: Final combined score (higher is better)
        bic_component: BIC contribution to score
        crps_component: CRPS contribution to score
        hyvarinen_component: Hyvärinen contribution to score
        pit_component: PIT contribution to score
        pit_calibrated: Whether model passed PIT calibration
        raw_scores: Dictionary of raw score values
        weights_used: Weights used for combination
    """
    combined_score: float
    bic_component: float
    crps_component: float
    hyvarinen_component: float
    pit_component: float
    pit_calibrated: bool
    raw_scores: Dict[str, float]
    weights_used: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "combined_score": self.combined_score,
            "bic_component": self.bic_component,
            "crps_component": self.crps_component,
            "hyvarinen_component": self.hyvarinen_component,
            "pit_component": self.pit_component,
            "pit_calibrated": self.pit_calibrated,
            "raw_scores": self.raw_scores,
            "weights_used": self.weights_used,
        }


def compute_combined_score(
    bic: float,
    crps: float,
    hyvarinen: float,
    pit_pvalue: float,
    config: Optional[ScoringConfig] = None,
    regime: Optional[str] = None,
    reference_scores: Optional[Dict[str, Tuple[float, float]]] = None,
) -> CombinedScoreResult:
    """
    Compute combined score from individual metrics.
    
    The combined score transforms each metric to [0, 1] range and
    computes a weighted average. Higher combined score is better.
    
    Args:
        bic: Bayesian Information Criterion (lower is better)
        crps: Continuous Ranked Probability Score (lower is better)
        hyvarinen: Hyvärinen score (interpretation depends on model)
        pit_pvalue: PIT uniformity test p-value (higher is better)
        config: Scoring configuration
        regime: Market regime for adaptive weights
        reference_scores: Dict of (min, max) for each metric for normalization
        
    Returns:
        CombinedScoreResult with combined score and breakdown
    """
    config = config or DEFAULT_SCORING_CONFIG
    
    # Get weights (possibly adjusted for regime)
    if config.regime_adaptation and regime in REGIME_WEIGHT_ADJUSTMENTS:
        weights = REGIME_WEIGHT_ADJUSTMENTS[regime]
    else:
        weights = {
            "bic_weight": config.bic_weight,
            "crps_weight": config.crps_weight,
            "hyvarinen_weight": config.hyvarinen_weight,
            "pit_weight": config.pit_weight,
        }
    
    # Check PIT calibration
    pit_calibrated = pit_pvalue >= config.pit_threshold
    
    # If PIT hard constraint and not calibrated, return zero score
    if config.pit_hard_constraint and not pit_calibrated:
        return CombinedScoreResult(
            combined_score=0.0,
            bic_component=0.0,
            crps_component=0.0,
            hyvarinen_component=0.0,
            pit_component=0.0,
            pit_calibrated=False,
            raw_scores={"bic": bic, "crps": crps, "hyvarinen": hyvarinen, "pit_pvalue": pit_pvalue},
            weights_used=weights,
        )
    
    # Normalize scores to [0, 1] where higher is better
    # BIC: lower is better, so negate and normalize
    if reference_scores and "bic" in reference_scores:
        bic_min, bic_max = reference_scores["bic"]
        bic_range = bic_max - bic_min if bic_max > bic_min else 1.0
        bic_normalized = 1.0 - (bic - bic_min) / bic_range
    else:
        # Use sigmoid transformation for unbounded BIC
        bic_normalized = 1.0 / (1.0 + np.exp(bic / 1000))
    
    # CRPS: lower is better, normalize
    if reference_scores and "crps" in reference_scores:
        crps_min, crps_max = reference_scores["crps"]
        crps_range = crps_max - crps_min if crps_max > crps_min else 1.0
        crps_normalized = 1.0 - (crps - crps_min) / crps_range
    else:
        # Use exponential decay for CRPS
        crps_normalized = np.exp(-crps / 0.1)  # Typical CRPS scale
    
    # Hyvärinen: interpretation varies, use sigmoid
    if reference_scores and "hyvarinen" in reference_scores:
        hyv_min, hyv_max = reference_scores["hyvarinen"]
        hyv_range = hyv_max - hyv_min if hyv_max > hyv_min else 1.0
        hyv_normalized = (hyvarinen - hyv_min) / hyv_range
    else:
        # Sigmoid transformation
        hyv_normalized = 1.0 / (1.0 + np.exp(-hyvarinen))
    
    # PIT: p-value directly usable, but transform for better discrimination
    # Higher p-value = better calibration
    pit_normalized = min(1.0, pit_pvalue / config.pit_threshold) if pit_pvalue < config.pit_threshold else 1.0
    
    # Clamp all normalized values to [0, 1]
    bic_normalized = np.clip(bic_normalized, 0.0, 1.0)
    crps_normalized = np.clip(crps_normalized, 0.0, 1.0)
    hyv_normalized = np.clip(hyv_normalized, 0.0, 1.0)
    pit_normalized = np.clip(pit_normalized, 0.0, 1.0)
    
    # Compute weighted components
    bic_component = weights["bic_weight"] * bic_normalized
    crps_component = weights["crps_weight"] * crps_normalized
    hyv_component = weights["hyvarinen_weight"] * hyv_normalized
    pit_component = weights["pit_weight"] * pit_normalized
    
    # Combined score
    combined = bic_component + crps_component + hyv_component + pit_component
    
    return CombinedScoreResult(
        combined_score=float(combined),
        bic_component=float(bic_component),
        crps_component=float(crps_component),
        hyvarinen_component=float(hyv_component),
        pit_component=float(pit_component),
        pit_calibrated=pit_calibrated,
        raw_scores={"bic": bic, "crps": crps, "hyvarinen": hyvarinen, "pit_pvalue": pit_pvalue},
        weights_used=weights,
    )


def compute_pareto_frontier(
    models: Dict[str, Dict[str, float]],
    objectives: List[str] = None,
) -> List[str]:
    """
    Compute Pareto frontier of models across multiple objectives.
    
    A model is Pareto-optimal if no other model dominates it across
    all objectives.
    
    Args:
        models: Dict mapping model name to dict of objective values
        objectives: List of objective names (default: all objectives)
        
    Returns:
        List of model names on the Pareto frontier
    """
    if not models:
        return []
    
    model_names = list(models.keys())
    if objectives is None:
        objectives = list(next(iter(models.values())).keys())
    
    # Convert to matrix (rows = models, cols = objectives)
    # Assume higher is better for all objectives
    n_models = len(model_names)
    n_obj = len(objectives)
    
    values = np.zeros((n_models, n_obj))
    for i, name in enumerate(model_names):
        for j, obj in enumerate(objectives):
            values[i, j] = models[name].get(obj, 0.0)
    
    # Find Pareto frontier
    pareto_mask = np.ones(n_models, dtype=bool)
    
    for i in range(n_models):
        if not pareto_mask[i]:
            continue
        for j in range(n_models):
            if i == j or not pareto_mask[j]:
                continue
            # Check if j dominates i
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                pareto_mask[i] = False
                break
    
    return [name for name, on_frontier in zip(model_names, pareto_mask) if on_frontier]


def rank_models_by_combined_score(
    model_scores: Dict[str, CombinedScoreResult],
) -> List[Tuple[str, float]]:
    """
    Rank models by combined score.
    
    Args:
        model_scores: Dict mapping model name to CombinedScoreResult
        
    Returns:
        List of (model_name, score) tuples, sorted descending by score
    """
    ranked = [(name, result.combined_score) for name, result in model_scores.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
