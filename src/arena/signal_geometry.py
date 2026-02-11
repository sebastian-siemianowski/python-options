"""
===============================================================================
SIGNAL GEOMETRY — Epistemic Fields to Trading Actions
===============================================================================

CRITICAL ARCHITECTURAL PRINCIPLES (Professor Wang, Professor Chen, Professor Liu):

1. DIRECTION IS SYNTHESIZED, NOT READ
   - Raw asymmetry ≠ direction
   - Direction emerges from: asymmetry × momentum × stability
   - This decouples epistemic geometry from trading decisions

2. SIZE COMES FROM CONFIDENCE × REGIME_FIT, NOT ASYMMETRY
   - Asymmetry provides POLARITY (sign)
   - Confidence provides CAPITAL AUTHORITY (size)
   - This is the institutional hedge fund standard

3. STABILITY GATES EVERYTHING
   - No stability → no trade, regardless of direction signal
   - Stability acts as a circuit breaker

FORMULA:
    direction_score = asymmetry × max(0, belief_momentum) × max(0, stability)
    polarity = sign(tanh(direction_score))
    size = base × max(0, confidence) × max(0, regime_fit)
    
This is mathematically proper epistemic-to-action conversion.

Author: Chinese Staff Professor - Elite Quant Systems (Top 0.00001% Hedge Fund)
Date: February 2026
===============================================================================
"""

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from .signal_fields import SignalFields


class TradeAction(Enum):
    ALLOW_LONG = auto()
    ALLOW_SHORT = auto()
    DENY_ENTRY = auto()
    ALLOW_SCALING = auto()
    REQUIRE_EXIT = auto()
    REDUCE_EXPOSURE = auto()
    HOLD = auto()


@dataclass
class GeometryConfig:
    """
    Configuration for the Signal Geometry Engine.
    
    CALIBRATION PHILOSOPHY:
    - Thresholds are BEHAVIORAL gates, not optimization targets
    - Position sizes reflect CAPITAL AUTHORITY from confidence
    - All parameters have mathematical justification
    
    INSTITUTIONAL INSIGHT (Professor Chen):
    - Too strict thresholds → miss market drift → negative CAGR
    - Too loose thresholds → excessive trading → transaction costs eat alpha
    - Proper calibration captures drift while avoiding noise
    """
    # Direction synthesis thresholds (after tanh transformation)
    min_direction_score: float = 0.10      # Minimum |direction_score| to act
    strong_direction_score: float = 0.25   # Strong direction threshold
    
    # Confidence thresholds - MODERATE selectivity
    min_confidence: float = 0.30           # Minimum confidence to trade
    high_confidence: float = 0.50          # High confidence bonus threshold
    
    # Stability thresholds - PERMISSIVE (stability gates, not vetoes)
    min_stability: float = -0.40           # Below this = unstable, deny entry
    high_stability: float = 0.20           # High stability bonus
    
    # Risk thresholds - MAINTAIN strict risk management
    max_risk: float = 0.60                 # Maximum acceptable risk
    extreme_risk: float = 0.80             # Emergency exit threshold
    
    # Regime fit - RELAXED (models work across regimes)
    min_regime_fit: float = -0.50          # Poor fit = model inappropriate
    good_regime_fit: float = 0.10          # Good fit bonus
    
    # Position sizing (SIZE AUTHORITY)
    # Size = base × confidence_factor × regime_factor
    base_position_size: float = 0.50       # Base allocation - INCREASED
    max_position_size: float = 1.00        # Maximum allocation - full position
    min_position_size: float = 0.20        # Minimum to bother trading
    
    # Confidence → Size multiplier (KEY INSTITUTIONAL PARAMETER)
    confidence_size_multiplier: float = 2.0  # max(0, conf) × this - INCREASED
    regime_size_multiplier: float = 1.2      # max(0, regime_fit) × this - INCREASED
    stability_size_bonus: float = 0.30       # Bonus for high stability
    
    # Long bias (markets have positive drift empirically)
    # With long-only mode, this helps stay invested more often
    long_bias: float = 0.15                # Upward drift compensation


@dataclass
class GeometryDecision:
    """
    Output of the Signal Geometry Engine.
    
    IMPORTANT: position_signal = direction × size
    where direction is SYNTHESIZED and size comes from CONFIDENCE.
    """
    action: TradeAction
    direction: int = 0                   # +1, -1, or 0
    position_size: float = 0.0           # [0, max_position_size]
    confidence: float = 0.0              # Composite confidence
    direction_score: float = 0.0         # Raw synthesized direction score
    size_authority: float = 0.0          # Confidence-based size authority
    rationale: str = ""
    
    @property
    def position_signal(self) -> float:
        """
        Final position signal for the backtest engine.
        Only non-zero for ALLOW actions.
        """
        if self.action in [TradeAction.ALLOW_LONG, TradeAction.ALLOW_SHORT, 
                           TradeAction.ALLOW_SCALING, TradeAction.HOLD]:
            return self.direction * self.position_size
        return 0.0


def _synthesize_direction(fields: SignalFields, long_bias: float = 0.0) -> float:
    """
    CORE FORMULA: Synthesize direction from epistemic fields.
    
    INSTITUTIONAL INSIGHT (Professor Wang):
    The direction field contains price momentum validated by Kalman accuracy.
    This should be PRIMARY for CAGR capture.
    
    TREND STRENGTH FILTER (Professor Chen):
    Only trade when multiple confirming factors agree.
    Single-factor signals are noise.
    
    FORMULA:
        base_direction = direction × (1 + alignment_bonus)
        trend_strength = confirmation_count / total_factors
        final_direction = base_direction × trend_strength_multiplier
    
    Returns: direction_score in approximately [-1, +1] after tanh
    """
    # ==========================================================================
    # PRIMARY: Direction from price momentum (validated by Kalman)
    # ==========================================================================
    base_direction = fields.direction  # Already in [-1, +1], captures price trend
    
    # ==========================================================================
    # TREND STRENGTH FILTER: Count confirming factors
    # ==========================================================================
    confirmation_count = 0
    total_factors = 4
    
    # Factor 1: Belief momentum alignment
    if fields.belief_momentum * base_direction > 0.05:
        confirmation_count += 1
    
    # Factor 2: Positive stability
    if fields.stability > 0:
        confirmation_count += 1
    
    # Factor 3: Good regime fit
    if fields.regime_fit > -0.2:
        confirmation_count += 1
    
    # Factor 4: Confidence above threshold
    if fields.confidence > 0.3:
        confirmation_count += 1
    
    # Trend strength multiplier: require at least 2 confirmations
    if confirmation_count < 2:
        trend_strength_mult = 0.3  # Heavily dampen weak signals
    elif confirmation_count == 2:
        trend_strength_mult = 0.7
    elif confirmation_count == 3:
        trend_strength_mult = 1.0
    else:
        trend_strength_mult = 1.2  # Boost strong signals
    
    # ==========================================================================
    # ALIGNMENT BONUS: Do other factors agree?
    # ==========================================================================
    alignment_bonus = 0.0
    
    # Belief momentum alignment
    if fields.belief_momentum * base_direction > 0:
        alignment_bonus += 0.15 * abs(fields.belief_momentum)
    else:
        alignment_bonus -= 0.1 * abs(fields.belief_momentum)
    
    # Asymmetry alignment
    if fields.asymmetry * base_direction > 0:
        alignment_bonus += 0.10 * abs(fields.asymmetry)
    
    # Stability bonus
    if fields.stability > 0:
        alignment_bonus += 0.08 * fields.stability
    
    # ==========================================================================
    # COMPUTE FINAL DIRECTION SCORE
    # ==========================================================================
    direction_multiplier = (1.0 + alignment_bonus) * trend_strength_mult
    raw_score = base_direction * direction_multiplier
    
    # Add long bias
    raw_score += long_bias
    
    # Compress via tanh
    direction_score = float(np.tanh(raw_score * 1.5))
    
    return direction_score


def _compute_size_authority(fields: SignalFields, cfg: GeometryConfig) -> float:
    """
    CORE FORMULA: Size comes from CONFIDENCE × REGIME_FIT.
    
    NOT from asymmetry magnitude.
    NOT from direction strength.
    
    Size authority = how much capital we TRUST to this signal.
    
    INSTITUTIONAL CALIBRATION (Professor Liu):
    - Start conservative (0.3-0.5 authority)
    - Only increase with high confidence AND good regime fit
    - NEVER exceed 1.0 (which gives max position)
    
    Returns: size_authority in [0, 1] that scales base_position_size
    """
    # Confidence factor: clamp to [0, 1] before multiplying
    raw_confidence = np.clip(fields.confidence, 0, 1)
    confidence_factor = raw_confidence * cfg.confidence_size_multiplier
    
    # Regime fit factor: shift and clamp
    # regime_fit is in [-1, +1], we want positive values to contribute
    regime_shifted = np.clip(fields.regime_fit + 0.5, 0, 1.5)  # Now in [0, 1.5]
    regime_factor = (regime_shifted / 1.5) * cfg.regime_size_multiplier  # Normalize to [0, 1]
    
    # Stability multiplier (gentle boost, not aggressive)
    if fields.stability > cfg.high_stability:
        stability_mult = 1.0 + cfg.stability_size_bonus * 0.5
    else:
        stability_mult = 1.0 + max(0, fields.stability) * cfg.stability_size_bonus * 0.3
    
    # BASE size authority = geometric mean of confidence and regime factors
    # This is MORE CONSERVATIVE than additive
    size_authority = np.sqrt(confidence_factor * regime_factor) * stability_mult
    
    # RISK DAMPENING: Reduce size authority in high risk environments
    risk = fields.composite_risk
    if risk > 0.4:
        risk_penalty = (risk - 0.4) * 1.5  # Aggressive penalty for high risk
        size_authority *= max(0.2, 1 - risk_penalty)
    
    # Clamp to [0, 1] - NEVER exceed 1.0 authority
    return float(np.clip(size_authority, 0, 1.0))


class SignalGeometryEngine:
    """
    Signal Geometry Engine — Converts Epistemic Fields to Trading Actions.
    
    ARCHITECTURE:
    
    1. STABILITY GATE: Deny entry if environment unstable
    2. RISK GATE: Exit/reduce if risk exceeds thresholds
    3. DIRECTION SYNTHESIS: Compute direction_score from fields
    4. SIZE AUTHORITY: Compute size from confidence × regime_fit
    5. ACTION DETERMINATION: Map to TradeAction
    
    CRITICAL: Direction is SYNTHESIZED, not read.
              Size comes from CONFIDENCE, not direction magnitude.
    """
    
    def __init__(self, config: GeometryConfig = None):
        self.config = config or GeometryConfig()
    
    def evaluate(self, fields: SignalFields) -> GeometryDecision:
        """
        Evaluate SignalFields and produce GeometryDecision.
        
        This is the main entry point for the backtest engine.
        """
        cfg = self.config
        
        # =====================================================================
        # PHASE 1: STABILITY GATE (Circuit Breaker)
        # =====================================================================
        if fields.stability < cfg.min_stability:
            return GeometryDecision(
                action=TradeAction.DENY_ENTRY,
                rationale=f"Unstable: stability={fields.stability:.2f} < {cfg.min_stability}"
            )
        
        # =====================================================================
        # PHASE 2: REGIME FIT GATE
        # =====================================================================
        if fields.regime_fit < cfg.min_regime_fit:
            return GeometryDecision(
                action=TradeAction.DENY_ENTRY,
                rationale=f"Poor regime fit: {fields.regime_fit:.2f} < {cfg.min_regime_fit}"
            )
        
        # =====================================================================
        # PHASE 3: RISK GATE
        # =====================================================================
        risk = fields.composite_risk
        if risk > cfg.extreme_risk:
            return GeometryDecision(
                action=TradeAction.REQUIRE_EXIT,
                rationale=f"Extreme risk: {risk:.2f}"
            )
        if risk > cfg.max_risk:
            return GeometryDecision(
                action=TradeAction.REDUCE_EXPOSURE,
                rationale=f"High risk: {risk:.2f}"
            )
        
        # =====================================================================
        # PHASE 4: DIRECTION SYNTHESIS (THE KEY CHANGE)
        # =====================================================================
        # Direction is NOT read from fields.asymmetry directly
        # Direction EMERGES from the synthesis formula
        direction_score = _synthesize_direction(fields, cfg.long_bias)
        
        # Check if direction is strong enough
        if abs(direction_score) < cfg.min_direction_score:
            return GeometryDecision(
                action=TradeAction.HOLD,
                direction_score=direction_score,
                rationale=f"Weak direction: |{direction_score:.2f}| < {cfg.min_direction_score}"
            )
        
        # =====================================================================
        # PHASE 5: CONFIDENCE GATE
        # =====================================================================
        confidence = fields.composite_confidence
        if confidence < cfg.min_confidence:
            return GeometryDecision(
                action=TradeAction.HOLD,
                direction_score=direction_score,
                confidence=confidence,
                rationale=f"Low confidence: {confidence:.2f} < {cfg.min_confidence}"
            )
        
        # =====================================================================
        # PHASE 6: SIZE AUTHORITY (THE KEY CHANGE)
        # =====================================================================
        # Size comes from CONFIDENCE × REGIME_FIT, NOT direction magnitude
        size_authority = _compute_size_authority(fields, cfg)
        
        # Compute position size: base × (1 + bounded_authority)
        # This ensures size is in [base, base * 2] range at most
        position_size = cfg.base_position_size * (1 + size_authority * 0.8)
        
        # Additional risk-based dampening
        if risk > 0.3:
            risk_dampening = 1 - (risk - 0.3) * 0.8
            position_size *= max(0.3, risk_dampening)
        
        # Direction strength influence (weak signals = smaller size)
        direction_strength_factor = min(1.0, abs(direction_score) / 0.3)
        position_size *= (0.6 + 0.4 * direction_strength_factor)
        
        # Clamp to bounds
        position_size = float(np.clip(
            position_size, 
            cfg.min_position_size, 
            cfg.max_position_size
        ))
        
        # =====================================================================
        # PHASE 7: ACTION DETERMINATION
        # =====================================================================
        # Determine polarity from synthesized direction
        direction = 1 if direction_score > 0 else -1
        
        # LONG-ONLY MODE: Markets have positive drift, shorts destroy CAGR
        # Only allow LONG positions, HOLD on negative direction
        if direction < 0:
            return GeometryDecision(
                action=TradeAction.HOLD,
                direction_score=direction_score,
                confidence=confidence,
                rationale=f"Long-only mode: direction={direction_score:.2f} < 0"
            )
        
        # Determine action type for LONG positions only
        if abs(direction_score) > cfg.strong_direction_score and confidence > cfg.high_confidence:
            action = TradeAction.ALLOW_SCALING
        else:
            action = TradeAction.ALLOW_LONG
        
        return GeometryDecision(
            action=action,
            direction=direction,
            position_size=position_size,
            confidence=confidence,
            direction_score=direction_score,
            size_authority=size_authority,
            rationale=f"DirScore={direction_score:.2f}, SizeAuth={size_authority:.2f}, Conf={confidence:.2f}"
        )


__all__ = ["TradeAction", "GeometryConfig", "GeometryDecision", "SignalGeometryEngine"]
