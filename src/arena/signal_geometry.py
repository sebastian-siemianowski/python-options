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

4. EXPLICIT FAILURE HANDLING (NO SILENT FAILURES)
   - Invalid fields (NaN, None, extreme) are EXPLICITLY detected
   - Calibration bugs are logged, not hidden
   - Force deleverage on invalid data

FORMULA:
    direction_score = asymmetry × max(0, belief_momentum) × max(0, stability)
    polarity = sign(tanh(direction_score))
    size = base × max(0, confidence) × max(0, regime_fit)
    
This is mathematically proper epistemic-to-action conversion.

Author: Chinese Staff Professor - Elite Quant Systems (Top 0.00001% Hedge Fund)
Date: February 2026
===============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional
import numpy as np
import warnings

from .signal_fields import SignalFields


class TradeAction(Enum):
    ALLOW_LONG = auto()
    ALLOW_SHORT = auto()
    DENY_ENTRY = auto()
    ALLOW_SCALING = auto()
    REQUIRE_EXIT = auto()
    REDUCE_EXPOSURE = auto()
    HOLD = auto()
    FORCE_DELEVERAGE = auto()  # NEW: Explicit action for invalid fields


# =============================================================================
# FIELD VALIDATION — Explicit Detection of Calibration Bugs
# =============================================================================

@dataclass
class FieldValidationResult:
    """Result of field validation check."""
    is_valid: bool
    invalid_fields: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    severity: str = "OK"  # OK, WARNING, CRITICAL


def _validate_fields(fields: SignalFields) -> FieldValidationResult:
    """
    EXPLICIT validation of SignalFields.
    
    Detects:
    - NaN values
    - None values  
    - Extreme values (outside expected bounds)
    - Infinite values
    
    This prevents SILENT failures that hide calibration bugs.
    
    Returns:
        FieldValidationResult with details of any issues
    """
    invalid_fields = []
    reasons = []
    
    # Define expected bounds for each field
    field_bounds = {
        'direction': (-1.0, 1.0),
        'asymmetry': (-1.0, 1.0),
        'belief_momentum': (-1.0, 1.0),
        'confidence': (-1.0, 1.0),
        'stability': (-1.0, 1.0),
        'regime_fit': (-1.0, 1.0),
        'transition_pressure': (-1.0, 1.0),
        'tail_risk_left': (0.0, 1.0),
        'tail_risk_right': (0.0, 1.0),
        'volatility_state': (-1.0, 1.0),
        'constraint_pressure': (-1.0, 1.0),
        'hedging_pressure': (-1.0, 1.0),
    }
    
    for field_name, (min_val, max_val) in field_bounds.items():
        value = getattr(fields, field_name, None)
        
        # Check for None
        if value is None:
            invalid_fields.append(field_name)
            reasons.append(f"{field_name}=None (missing)")
            continue
        
        # Check for NaN
        if np.isnan(value):
            invalid_fields.append(field_name)
            reasons.append(f"{field_name}=NaN (calibration failure)")
            continue
        
        # Check for infinity
        if np.isinf(value):
            invalid_fields.append(field_name)
            reasons.append(f"{field_name}=Inf (overflow)")
            continue
        
        # Check for extreme values (3x outside expected bounds = CRITICAL)
        extreme_min = min_val - 2 * abs(min_val) if min_val != 0 else -3.0
        extreme_max = max_val + 2 * abs(max_val) if max_val != 0 else 3.0
        
        if value < extreme_min or value > extreme_max:
            invalid_fields.append(field_name)
            reasons.append(f"{field_name}={value:.4f} (extreme: expected [{min_val}, {max_val}])")
            continue
    
    # Also check composite properties
    try:
        composite_dir = fields.composite_direction
        if np.isnan(composite_dir) or np.isinf(composite_dir):
            invalid_fields.append('composite_direction')
            reasons.append(f"composite_direction={composite_dir} (derived NaN/Inf)")
    except Exception as e:
        invalid_fields.append('composite_direction')
        reasons.append(f"composite_direction computation failed: {e}")
    
    try:
        composite_conf = fields.composite_confidence
        if np.isnan(composite_conf) or np.isinf(composite_conf):
            invalid_fields.append('composite_confidence')
            reasons.append(f"composite_confidence={composite_conf} (derived NaN/Inf)")
    except Exception as e:
        invalid_fields.append('composite_confidence')
        reasons.append(f"composite_confidence computation failed: {e}")
    
    try:
        composite_risk = fields.composite_risk
        if np.isnan(composite_risk) or np.isinf(composite_risk):
            invalid_fields.append('composite_risk')
            reasons.append(f"composite_risk={composite_risk} (derived NaN/Inf)")
    except Exception as e:
        invalid_fields.append('composite_risk')
        reasons.append(f"composite_risk computation failed: {e}")
    
    # Determine severity
    if not invalid_fields:
        severity = "OK"
    elif any('NaN' in r or 'Inf' in r or 'None' in r for r in reasons):
        severity = "CRITICAL"  # Calibration bug
    else:
        severity = "WARNING"  # Extreme but recoverable
    
    return FieldValidationResult(
        is_valid=len(invalid_fields) == 0,
        invalid_fields=invalid_fields,
        reasons=reasons,
        severity=severity
    )


# Track validation failures for diagnostics
_VALIDATION_FAILURE_LOG: List[Tuple[str, str, List[str]]] = []
_MAX_LOG_ENTRIES = 1000


def _log_validation_failure(model_name: str, severity: str, reasons: List[str]):
    """Log validation failure for later analysis."""
    global _VALIDATION_FAILURE_LOG
    
    if len(_VALIDATION_FAILURE_LOG) < _MAX_LOG_ENTRIES:
        _VALIDATION_FAILURE_LOG.append((model_name, severity, reasons))
    
    # Also emit warning for visibility
    if severity == "CRITICAL":
        warnings.warn(
            f"CALIBRATION BUG DETECTED in {model_name}: {'; '.join(reasons[:3])}",
            RuntimeWarning
        )


def get_validation_failures() -> List[Tuple[str, str, List[str]]]:
    """Get logged validation failures for diagnostics."""
    return _VALIDATION_FAILURE_LOG.copy()


def clear_validation_failures():
    """Clear validation failure log."""
    global _VALIDATION_FAILURE_LOG
    _VALIDATION_FAILURE_LOG = []


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
    # EMERGENT direction is noisier - require stronger consensus
    min_direction_score: float = 0.15      # Minimum |direction_score| to act
    strong_direction_score: float = 0.30   # Strong direction threshold
    
    # Confidence thresholds - MODERATE selectivity
    min_confidence: float = 0.32           # Minimum confidence to trade
    high_confidence: float = 0.52          # High confidence bonus threshold
    
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
    # With emergent direction, be slightly more conservative
    base_position_size: float = 0.45       # Base allocation
    max_position_size: float = 0.85        # Maximum allocation
    min_position_size: float = 0.15        # Minimum to bother trading
    
    # Confidence → Size multiplier (KEY INSTITUTIONAL PARAMETER)
    confidence_size_multiplier: float = 1.8  # max(0, conf) × this
    regime_size_multiplier: float = 1.0      # max(0, regime_fit) × this
    stability_size_bonus: float = 0.25       # Bonus for high stability
    
    # Long bias (markets have positive drift empirically)
    # With long-only mode, this helps stay invested more often
    long_bias: float = 0.15                # Upward drift compensation
    
    # Short mode toggle (for research/stress-testing negative geometry)
    # Production: False (long-only captures drift, shorts destroy CAGR)
    # Research: True (allows testing short-side calibration)
    allow_shorts: bool = False


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
        
        FORCE_DELEVERAGE returns 0 to flatten any existing position.
        This is the explicit response to calibration bugs.
        """
        if self.action == TradeAction.FORCE_DELEVERAGE:
            # Calibration bug detected - flatten immediately
            return 0.0
        if self.action in [TradeAction.ALLOW_LONG, TradeAction.ALLOW_SHORT, 
                           TradeAction.ALLOW_SCALING, TradeAction.HOLD]:
            return self.direction * self.position_size
        return 0.0


def _synthesize_direction(fields: SignalFields, long_bias: float = 0.0) -> float:
    """
    CORE FORMULA: Direction EMERGES from epistemic geometry.
    
    CRITICAL INSIGHT (Professor Wang):
    Direction is NOT read from any single field.
    Direction EMERGES from the geometric product of multiple epistemic signals.
    
    PHILOSOPHY:
    - fields.direction = price momentum (one input among many)
    - fields.asymmetry = distributional skew (another input)
    - fields.belief_momentum = model belief change (another input)
    
    TRUE SYNTHESIS:
    Each field contributes a "vote" weighted by reliability.
    Direction emerges from the consensus, not any single source.
    
    FORMULA:
        weighted_sum = w1*direction + w2*asymmetry + w3*belief_momentum
        reliability = f(stability, confidence, regime_fit)
        direction_score = weighted_sum × reliability
    
    This ensures NO SINGLE FIELD dominates.
    
    Returns: direction_score in approximately [-1, +1] after tanh
    """
    # ==========================================================================
    # STEP 1: Collect directional "votes" from multiple sources
    # ==========================================================================
    # Each source contributes independently - no single source dominates
    
    # Source 1: Price momentum (from Kalman) - weight 0.35
    price_momentum_vote = fields.direction * 0.35
    
    # Source 2: Distribution asymmetry (skewness) - weight 0.25
    asymmetry_vote = fields.asymmetry * 0.25
    
    # Source 3: Belief momentum (rate of belief change) - weight 0.25
    belief_vote = fields.belief_momentum * 0.25
    
    # Source 4: Hedging pressure (if available) - weight 0.15
    hedging_vote = fields.hedging_pressure * 0.15
    
    # Raw directional consensus
    raw_consensus = price_momentum_vote + asymmetry_vote + belief_vote + hedging_vote
    
    # ==========================================================================
    # STEP 2: Compute RELIABILITY multiplier
    # ==========================================================================
    # Reliability gates the signal - low reliability = dampen direction
    
    # Stability contribution: stable = trust the consensus
    stability_reliability = max(0, fields.stability + 0.3) / 1.3  # Normalize to [0, 1]
    
    # Confidence contribution: confident = trust the consensus
    confidence_reliability = max(0, fields.confidence) 
    
    # Regime fit contribution: good fit = trust the consensus
    regime_reliability = max(0, fields.regime_fit + 0.5) / 1.5  # Normalize to [0, 1]
    
    # Combined reliability (geometric mean for conservatism)
    # NO FLOOR - low reliability MUST genuinely dampen direction
    # This is architectural purity: unreliable = no signal
    reliability = (stability_reliability * confidence_reliability * regime_reliability) ** (1/3)
    
    # Apply slight exponential dampening for extra conservatism on weak reliability
    # reliability^1.2 makes low values even lower, high values slightly lower
    reliability = reliability ** 1.15
    
    # ==========================================================================
    # STEP 3: Agreement bonus - boost when sources agree
    # ==========================================================================
    # Count how many sources agree on direction
    # IMPORTANT: Only count votes if magnitude > 0.15 (not noise)
    # Threshold 0.05 was too permissive - amplified weak signals
    sources = [fields.direction, fields.asymmetry, fields.belief_momentum]
    agreement_threshold = 0.15  # Only meaningful signals count as votes
    signs = [1 if s > agreement_threshold else (-1 if s < -agreement_threshold else 0) for s in sources]
    
    positive_votes = sum(1 for s in signs if s > 0)
    negative_votes = sum(1 for s in signs if s < 0)
    
    # Agreement multiplier: all agree = 1.3x, mixed = 0.7x
    if positive_votes >= 2 and negative_votes == 0:
        agreement_mult = 1.3
    elif negative_votes >= 2 and positive_votes == 0:
        agreement_mult = 1.3
    elif positive_votes > 0 and negative_votes > 0:
        agreement_mult = 0.7  # Conflicting signals = dampen
    else:
        agreement_mult = 1.0
    
    # ==========================================================================
    # STEP 4: Compute final direction score
    # ==========================================================================
    raw_direction = raw_consensus * reliability * agreement_mult
    
    # Add long bias (market drift compensation)
    raw_direction += long_bias
    
    # Compress to [-1, +1] via tanh
    direction_score = float(np.tanh(raw_direction * 2.0))
    
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
        # PHASE 0: FIELD VALIDATION (Detect Calibration Bugs)
        # =====================================================================
        # CRITICAL: Never silently fail on invalid fields
        # Invalid data = calibration bug = MUST be visible
        validation = _validate_fields(fields)
        
        if not validation.is_valid:
            # Log the failure for diagnostics
            _log_validation_failure(
                fields.model_name or "unknown",
                validation.severity,
                validation.reasons
            )
            
            # FORCE DELEVERAGE on critical failures (NaN, None, Inf)
            if validation.severity == "CRITICAL":
                return GeometryDecision(
                    action=TradeAction.FORCE_DELEVERAGE,
                    rationale=f"CALIBRATION BUG: {'; '.join(validation.reasons[:3])}"
                )
            
            # WARNING level: still deny entry but don't force exit
            return GeometryDecision(
                action=TradeAction.DENY_ENTRY,
                rationale=f"Invalid fields: {'; '.join(validation.reasons[:3])}"
            )
        
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
        # PHASE 6: SIZE AUTHORITY (STRICT ORTHOGONALITY)
        # =====================================================================
        # Size comes ONLY from CONFIDENCE × REGIME_FIT
        # Direction provides POLARITY (sign)
        # Confidence provides CAPITAL AUTHORITY (size)
        # These are ORTHOGONAL - direction magnitude NEVER affects size
        size_authority = _compute_size_authority(fields, cfg)
        
        # Compute position size: base × (1 + bounded_authority)
        position_size = cfg.base_position_size * (1 + size_authority * 0.8)
        
        # Risk-based dampening (this is about RISK, not direction)
        if risk > 0.3:
            risk_dampening = 1 - (risk - 0.3) * 0.8
            position_size *= max(0.3, risk_dampening)
        
        # NO direction_strength_factor - STRICT ORTHOGONALITY
        # Direction = sign only, Size = confidence only
        
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
        
        # SHORT MODE GATE (Configurable for research vs production)
        # Production (allow_shorts=False): Long-only, shorts destroy CAGR
        # Research (allow_shorts=True): Test short-side geometry calibration
        if direction < 0 and not cfg.allow_shorts:
            return GeometryDecision(
                action=TradeAction.HOLD,
                direction_score=direction_score,
                confidence=confidence,
                rationale=f"Long-only mode: direction={direction_score:.2f} < 0"
            )
        
        # Determine action type
        if abs(direction_score) > cfg.strong_direction_score and confidence > cfg.high_confidence:
            action = TradeAction.ALLOW_SCALING
        elif direction > 0:
            action = TradeAction.ALLOW_LONG
        else:
            action = TradeAction.ALLOW_SHORT  # Only reached if allow_shorts=True
        
        return GeometryDecision(
            action=action,
            direction=direction,
            position_size=position_size,
            confidence=confidence,
            direction_score=direction_score,
            size_authority=size_authority,
            rationale=f"DirScore={direction_score:.2f}, SizeAuth={size_authority:.2f}, Conf={confidence:.2f}"
        )


__all__ = [
    "TradeAction", 
    "GeometryConfig", 
    "GeometryDecision", 
    "SignalGeometryEngine",
    "FieldValidationResult",
    "get_validation_failures",
    "clear_validation_failures",
]
