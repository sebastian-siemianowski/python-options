"""
===============================================================================
SIGNAL GEOMETRY — Epistemic Signal Fields & Action Mapping
===============================================================================

This module introduces the correct abstraction layer between models and execution.

CORE INSIGHT:
    Models are distributional geometry estimators, NOT forecasters.
    They answer: "What is the curvature / uncertainty / constraint geometry?"
    The old system heard: "Buy or sell?"

NEW CONTRACT:
    model.evaluate() → SignalFields (epistemic state)
    signal_geometry → Action (what is ALLOWED, not what to DO)
    backtest_engine → Execution (when allowed)

This layer ensures:
    - DT-CWT fills asymmetry + momentum
    - Fisher-Rao fills confidence + stability  
    - Free probability fills tail asymmetry
    - Log barrier fills stability
    - No model needs to lie about what it knows

Author: Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import numpy as np


# =============================================================================
# SIGNAL FIELDS — What models actually know
# =============================================================================

@dataclass
class SignalFields:
    """
    Epistemic signal fields that models can honestly populate.
    
    Each field is in [-1, +1] range where:
        -1 = strong negative signal
         0 = neutral / no information
        +1 = strong positive signal
    
    Models fill ONLY the fields they can honestly estimate.
    Unfilled fields default to 0 (no information).
    """
    
    # Directional signals (traditional)
    direction: float = 0.0          # Raw directional bias [-1=short, +1=long]
    direction_confidence: float = 0.0  # How confident in direction [0=none, 1=certain]
    
    # Distributional geometry signals (what elite models actually compute)
    confidence: float = 0.0         # Overall epistemic confidence [-1=confused, +1=clear]
    stability: float = 0.0          # Regime/distribution stability [-1=unstable, +1=stable]
    asymmetry: float = 0.0          # Tail/distribution asymmetry [-1=left, +1=right]
    belief_momentum: float = 0.0    # Rate of belief change [-1=collapsing, +1=strengthening]
    regime_fit: float = 0.0         # How well current regime fits model [-1=poor, +1=good]
    
    # Risk geometry signals
    tail_risk: float = 0.0          # Estimated tail risk [-1=left tail, +1=right tail]
    uncertainty: float = 0.0        # Model uncertainty [0=certain, 1=maximum uncertainty]
    curvature: float = 0.0          # Information geometry curvature [-1=concave, +1=convex]
    
    # Constraint signals
    barrier_proximity: float = 0.0  # Distance to constraints [0=far, 1=at barrier]
    liquidity_stress: float = 0.0   # Liquidity/market stress [0=normal, 1=stressed]
    
    # Metadata
    model_name: str = ""
    timestamp: Optional[str] = None
    raw_outputs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Clip all fields to valid ranges."""
        for field_name in ['direction', 'direction_confidence', 'confidence', 
                          'stability', 'asymmetry', 'belief_momentum', 'regime_fit',
                          'tail_risk', 'uncertainty', 'curvature', 
                          'barrier_proximity', 'liquidity_stress']:
            val = getattr(self, field_name)
            setattr(self, field_name, float(np.clip(val, -1.0, 1.0)))
    
    @property
    def has_directional_signal(self) -> bool:
        """Does this have meaningful directional information?"""
        return abs(self.direction) > 0.1 and self.direction_confidence > 0.3
    
    @property
    def has_geometric_signal(self) -> bool:
        """Does this have meaningful geometric/distributional information?"""
        return (abs(self.confidence) > 0.1 or 
                abs(self.stability) > 0.1 or 
                abs(self.asymmetry) > 0.1)
    
    @property
    def epistemic_quality(self) -> float:
        """Overall quality of epistemic state [0, 1]."""
        # High confidence + high stability + low uncertainty = high quality
        quality = (
            (1 + self.confidence) / 2 * 0.3 +      # confidence contributes 30%
            (1 + self.stability) / 2 * 0.3 +       # stability contributes 30%
            (1 - self.uncertainty) * 0.2 +          # low uncertainty contributes 20%
            (1 + self.regime_fit) / 2 * 0.2        # regime fit contributes 20%
        )
        return float(np.clip(quality, 0, 1))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'direction_confidence': self.direction_confidence,
            'confidence': self.confidence,
            'stability': self.stability,
            'asymmetry': self.asymmetry,
            'belief_momentum': self.belief_momentum,
            'regime_fit': self.regime_fit,
            'tail_risk': self.tail_risk,
            'uncertainty': self.uncertainty,
            'curvature': self.curvature,
            'barrier_proximity': self.barrier_proximity,
            'liquidity_stress': self.liquidity_stress,
            'model_name': self.model_name,
            'epistemic_quality': self.epistemic_quality,
        }


# =============================================================================
# ACTIONS — What the signal geometry layer decides
# =============================================================================

class SignalAction(Enum):
    """Actions that signal geometry can permit or deny."""
    
    # Entry permissions
    ALLOW_LONG = auto()         # Long entry is permitted
    ALLOW_SHORT = auto()        # Short entry is permitted
    DENY_ENTRY = auto()         # No entry permitted (protect capital)
    
    # Position sizing permissions
    ALLOW_FULL_SIZE = auto()    # Full position size permitted
    ALLOW_REDUCED_SIZE = auto() # Only reduced size permitted
    DENY_SCALING = auto()       # No scaling up permitted
    
    # Exit signals
    SUGGEST_EXIT = auto()       # Consider exiting (soft)
    FORCE_EXIT = auto()         # Must exit (hard)
    HOLD = auto()               # Maintain current position
    
    # Risk signals
    HEIGHTEN_RISK = auto()      # Increase risk monitoring
    NORMAL_RISK = auto()        # Normal risk state
    REDUCE_EXPOSURE = auto()    # Reduce overall exposure


@dataclass
class ActionDecision:
    """Decision from signal geometry layer."""
    
    # Primary action
    entry_action: SignalAction = SignalAction.DENY_ENTRY
    sizing_action: SignalAction = SignalAction.DENY_SCALING
    exit_action: SignalAction = SignalAction.HOLD
    risk_action: SignalAction = SignalAction.NORMAL_RISK
    
    # Sizing multiplier [0, 1]
    size_multiplier: float = 0.0
    
    # Directional bias (only meaningful if entry allowed)
    directional_bias: float = 0.0  # [-1, +1]
    
    # Confidence in decision
    decision_confidence: float = 0.0
    
    # Rationale
    rationale: List[str] = field(default_factory=list)
    
    @property
    def allows_trade(self) -> bool:
        """Is any trade action permitted?"""
        return self.entry_action in [SignalAction.ALLOW_LONG, SignalAction.ALLOW_SHORT]
    
    @property
    def position_signal(self) -> float:
        """
        Convert to position signal for backtest engine.
        
        Returns:
            -1 to +1 signal, or 0 if no trade allowed
        """
        if not self.allows_trade:
            return 0.0
        
        if self.entry_action == SignalAction.ALLOW_LONG:
            return self.directional_bias * self.size_multiplier
        elif self.entry_action == SignalAction.ALLOW_SHORT:
            return self.directional_bias * self.size_multiplier
        
        return 0.0


# =============================================================================
# SIGNAL GEOMETRY ENGINE — Maps fields to actions
# =============================================================================

@dataclass
class SignalGeometryConfig:
    """Configuration for signal geometry mapping."""
    
    # Entry thresholds (premium signals only - capital preservation paramount)
    min_confidence_for_entry: float = 0.40     # Require premium confidence
    min_stability_for_entry: float = 0.30      # Require premium stability
    max_uncertainty_for_entry: float = 0.40    # Block most uncertainty
    min_regime_fit_for_entry: float = 0.15     # Require good regime fit
    
    # Directional thresholds (require premium signals)
    min_direction_strength: float = 0.32       # Premium direction required
    min_direction_confidence: float = 0.45     # Premium confidence in direction
    
    # Asymmetry thresholds (for directional inference)
    asymmetry_direction_weight: float = 0.08   # Minimal asymmetry contribution
    momentum_direction_weight: float = 0.05    # Minimal momentum contribution
    
    # Sizing thresholds
    full_size_confidence: float = 0.85         # Ultra high bar for full size
    full_size_stability: float = 0.7           # Require excellent stability
    min_size_multiplier: float = 0.45          # Good minimum size when trading
    
    # Exit thresholds
    exit_confidence_threshold: float = 0.1     # Exit when confidence weakens
    exit_stability_threshold: float = 0.0      # Exit when stability drops
    
    # Risk thresholds
    high_risk_uncertainty: float = 0.35        # Very early risk warning
    high_risk_barrier_proximity: float = 0.55  # Barrier proximity threshold


class SignalGeometryEngine:
    """
    Maps SignalFields to ActionDecisions.
    
    This is the thin layer that translates epistemic state into allowed actions.
    It does NOT decide trades — it decides what trades are PERMITTED.
    """
    
    def __init__(self, config: SignalGeometryConfig = None):
        self.config = config or SignalGeometryConfig()
    
    def evaluate(self, fields: SignalFields) -> ActionDecision:
        """
        Evaluate signal fields and produce action decision.
        
        IMPROVED VERSION:
        - More permissive entry gates (trade more often)
        - Better direction signal combination
        - Size scaling based on confidence (trade smaller when uncertain)
        - Preserve asymmetric opportunities
        """
        decision = ActionDecision()
        rationale = []
        
        # =================================================================
        # STEP 1: Compute composite entry score (not binary gate)
        # =================================================================
        # Instead of hard gates, compute an entry quality score
        
        entry_score = 1.0  # Start with full permission
        
        # Confidence penalty (but allow negative confidence with penalty)
        if fields.confidence < self.config.min_confidence_for_entry:
            penalty = (self.config.min_confidence_for_entry - fields.confidence) * 0.5
            entry_score -= penalty
            rationale.append(f"Confidence penalty: {penalty:.2f}")
        
        # Stability penalty
        if fields.stability < self.config.min_stability_for_entry:
            penalty = (self.config.min_stability_for_entry - fields.stability) * 0.3
            entry_score -= penalty
            rationale.append(f"Stability penalty: {penalty:.2f}")
        
        # Uncertainty penalty
        if fields.uncertainty > self.config.max_uncertainty_for_entry:
            penalty = (fields.uncertainty - self.config.max_uncertainty_for_entry) * 0.5
            entry_score -= penalty
            rationale.append(f"Uncertainty penalty: {penalty:.2f}")
        
        # Regime fit penalty
        if fields.regime_fit < self.config.min_regime_fit_for_entry:
            penalty = (self.config.min_regime_fit_for_entry - fields.regime_fit) * 0.2
            entry_score -= penalty
            rationale.append(f"Regime fit penalty: {penalty:.2f}")
        
        entry_score = max(0, entry_score)
        
        # Only deny entry if entry_score is very low
        if entry_score < 0.3:
            decision.entry_action = SignalAction.DENY_ENTRY
            decision.rationale = rationale + [f"Entry score too low: {entry_score:.2f}"]
            return decision
        
        # =================================================================
        # STEP 2: Determine directional bias (improved combination)
        # =================================================================
        
        directional_bias = 0.0
        direction_sources = []
        
        # Primary: Raw direction signal (weighted by confidence)
        if abs(fields.direction) > 0.01:
            dir_contribution = fields.direction * (0.5 + 0.5 * fields.direction_confidence)
            directional_bias += dir_contribution * 0.5  # 50% weight
            direction_sources.append(f"direction:{dir_contribution:.3f}")
        
        # Secondary: Asymmetry contribution
        if abs(fields.asymmetry) > 0.1:
            asym_contribution = fields.asymmetry * self.config.asymmetry_direction_weight
            directional_bias += asym_contribution
            direction_sources.append(f"asymmetry:{asym_contribution:.3f}")
        
        # Tertiary: Belief momentum (trend following)
        if abs(fields.belief_momentum) > 0.1:
            momentum_contribution = fields.belief_momentum * self.config.momentum_direction_weight
            directional_bias += momentum_contribution
            direction_sources.append(f"momentum:{momentum_contribution:.3f}")
        
        # Quaternary: Tail risk asymmetry
        if abs(fields.tail_risk) > 0.2:
            tail_contribution = fields.tail_risk * 0.1
            directional_bias += tail_contribution
            direction_sources.append(f"tail:{tail_contribution:.3f}")
        
        if direction_sources:
            rationale.append(f"Direction: {' + '.join(direction_sources)} = {directional_bias:.3f}")
        
        # Clip to valid range
        directional_bias = float(np.clip(directional_bias, -1, 1))
        decision.directional_bias = directional_bias
        
        # =================================================================
        # STEP 3: Determine entry action
        # =================================================================
        
        if abs(directional_bias) < self.config.min_direction_strength:
            decision.entry_action = SignalAction.DENY_ENTRY
            rationale.append(f"Insufficient direction: {abs(directional_bias):.3f} < {self.config.min_direction_strength}")
        elif directional_bias > 0:
            decision.entry_action = SignalAction.ALLOW_LONG
            rationale.append(f"Long permitted")
        else:
            decision.entry_action = SignalAction.ALLOW_SHORT
            rationale.append(f"Short permitted")
        
        # =================================================================
        # STEP 4: Determine position sizing (improved)
        # =================================================================
        
        if decision.allows_trade:
            # Base size on entry score and epistemic quality
            base_size = entry_score * fields.epistemic_quality
            
            # Boost size if direction confidence is high
            if fields.direction_confidence > 0.5:
                base_size *= 1.0 + 0.3 * (fields.direction_confidence - 0.5)
            
            # Reduce size if stability is poor but still allow trading
            if fields.stability < self.config.full_size_stability:
                stability_factor = 0.5 + 0.5 * max(0, (fields.stability + 1) / (self.config.full_size_stability + 1))
                base_size *= stability_factor
            
            # Ensure minimum size when we do trade
            base_size = max(self.config.min_size_multiplier, base_size)
            
            decision.size_multiplier = float(np.clip(base_size, 0, 1))
            
            if decision.size_multiplier > 0.7:
                decision.sizing_action = SignalAction.ALLOW_FULL_SIZE
            elif decision.size_multiplier > 0.3:
                decision.sizing_action = SignalAction.ALLOW_REDUCED_SIZE
            else:
                decision.sizing_action = SignalAction.DENY_SCALING
            
            rationale.append(f"Size: {decision.size_multiplier:.2f}")
        
        # =================================================================
        # STEP 5: Determine exit signals
        # =================================================================
        
        if fields.confidence < self.config.exit_confidence_threshold:
            decision.exit_action = SignalAction.SUGGEST_EXIT
            rationale.append(f"Exit suggested: low confidence")
        
        if fields.stability < self.config.exit_stability_threshold:
            decision.exit_action = SignalAction.FORCE_EXIT
            rationale.append(f"Exit forced: regime collapse")
        
        # =================================================================
        # STEP 6: Determine risk state
        # =================================================================
        
        if (fields.uncertainty > self.config.high_risk_uncertainty or 
            fields.barrier_proximity > self.config.high_risk_barrier_proximity or
            fields.liquidity_stress > 0.5):
            decision.risk_action = SignalAction.HEIGHTEN_RISK
            rationale.append("Heightened risk")
        
        if fields.stability < -0.4 or fields.confidence < -0.4:
            decision.risk_action = SignalAction.REDUCE_EXPOSURE
            rationale.append("Reduce exposure")
        
        # =================================================================
        # STEP 7: Compute decision confidence
        # =================================================================
        
        decision.decision_confidence = entry_score * (1 - fields.uncertainty * 0.5)
        decision.rationale = rationale
        
        return decision


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def signal_fields_to_position(fields: SignalFields, config: SignalGeometryConfig = None) -> float:
    """
    Convenience function: Convert SignalFields directly to position signal.
    
    This is what the backtest engine will call.
    
    Returns:
        Position signal in [-1, +1] range, or 0 if no trade permitted.
    """
    engine = SignalGeometryEngine(config)
    decision = engine.evaluate(fields)
    return decision.position_signal


def create_signal_fields_from_kalman(
    mu: np.ndarray, 
    sigma: np.ndarray, 
    returns: np.ndarray,
    model_name: str = ""
) -> SignalFields:
    """
    Create SignalFields from Kalman filter outputs.
    
    This is the adapter for DT-CWT and similar models that output
    mu (filtered state) and sigma (uncertainty).
    """
    # Use latest values
    mu_last = mu[-1] if len(mu) > 0 else 0
    sigma_last = sigma[-1] if len(sigma) > 0 else 1
    
    # Compute rolling statistics for context
    if len(mu) >= 20:
        mu_mean = np.mean(mu[-20:])
        mu_std = np.std(mu[-20:]) + 1e-8
        sigma_mean = np.mean(sigma[-20:])
    else:
        mu_mean = np.mean(mu) if len(mu) > 0 else 0
        mu_std = np.std(mu) + 1e-8 if len(mu) > 1 else 1
        sigma_mean = np.mean(sigma) if len(sigma) > 0 else 1
    
    # Direction: normalized mu
    direction = float(np.tanh(mu_last / (mu_std * 2)))
    
    # Direction confidence: inverse of normalized sigma
    direction_confidence = float(1 - np.clip(sigma_last / (sigma_mean * 2), 0, 1))
    
    # Confidence: based on signal-to-noise
    snr = abs(mu_last) / (sigma_last + 1e-8)
    confidence = float(np.tanh(snr - 0.5))  # Centered around SNR=0.5
    
    # Stability: based on sigma consistency
    if len(sigma) >= 10:
        sigma_stability = 1 - np.std(sigma[-10:]) / (np.mean(sigma[-10:]) + 1e-8)
        stability = float(np.clip(sigma_stability * 2 - 1, -1, 1))
    else:
        stability = 0.0
    
    # Asymmetry: from return distribution around filtered mean
    if len(returns) >= 20:
        residuals = returns[-20:] - mu[-20:] if len(mu) >= 20 else returns[-20:]
        skew = np.mean(residuals**3) / (np.std(residuals)**3 + 1e-8)
        asymmetry = float(np.clip(skew / 2, -1, 1))
    else:
        asymmetry = 0.0
    
    # Belief momentum: change in mu direction
    if len(mu) >= 5:
        mu_delta = mu[-1] - mu[-5]
        belief_momentum = float(np.tanh(mu_delta / (mu_std + 1e-8)))
    else:
        belief_momentum = 0.0
    
    # Regime fit: how well recent observations fit the model
    if len(returns) >= 10 and len(mu) >= 10:
        prediction_errors = returns[-10:] - mu[-10:]
        mae = np.mean(np.abs(prediction_errors))
        expected_mae = sigma_mean
        regime_fit = float(1 - np.clip(mae / (expected_mae + 1e-8), 0, 2))
    else:
        regime_fit = 0.0
    
    # Uncertainty: normalized sigma
    uncertainty = float(np.clip(sigma_last / (sigma_mean * 2), 0, 1))
    
    return SignalFields(
        direction=direction,
        direction_confidence=direction_confidence,
        confidence=confidence,
        stability=stability,
        asymmetry=asymmetry,
        belief_momentum=belief_momentum,
        regime_fit=regime_fit,
        uncertainty=uncertainty,
        model_name=model_name,
        raw_outputs={'mu_last': float(mu_last), 'sigma_last': float(sigma_last)}
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SignalFields',
    'SignalAction', 
    'ActionDecision',
    'SignalGeometryConfig',
    'SignalGeometryEngine',
    'signal_fields_to_position',
    'create_signal_fields_from_kalman',
]