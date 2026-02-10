"""Signal Geometry - Epistemic Fields to Trading Actions."""

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
    min_direction_strength: float = 0.15
    strong_direction: float = 0.40
    min_confidence: float = 0.10
    high_confidence: float = 0.40
    min_stability: float = -0.30
    max_risk: float = 0.60
    extreme_risk: float = 0.80
    min_regime_fit: float = -0.40
    base_position_size: float = 0.20
    max_position_size: float = 0.35
    min_position_size: float = 0.10


@dataclass
class GeometryDecision:
    action: TradeAction
    direction: int = 0
    position_size: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    
    @property
    def position_signal(self):
        if self.action in [TradeAction.ALLOW_LONG, TradeAction.ALLOW_SHORT, 
                           TradeAction.ALLOW_SCALING, TradeAction.HOLD]:
            return self.direction * self.position_size
        return 0.0


class SignalGeometryEngine:
    def __init__(self, config=None):
        self.config = config or GeometryConfig()
    
    def evaluate(self, fields):
        cfg = self.config
        
        if fields.regime_fit < cfg.min_regime_fit:
            return GeometryDecision(action=TradeAction.DENY_ENTRY, rationale="Poor regime fit")
        
        risk = fields.composite_risk
        if risk > cfg.extreme_risk:
            return GeometryDecision(action=TradeAction.REQUIRE_EXIT, rationale="Extreme risk")
        if risk > cfg.max_risk:
            return GeometryDecision(action=TradeAction.DENY_ENTRY, rationale="High risk")
        
        direction_signal = fields.composite_direction
        confidence = fields.composite_confidence
        
        if abs(direction_signal) < cfg.min_direction_strength:
            return GeometryDecision(action=TradeAction.HOLD, rationale="Direction too weak")
        if confidence < cfg.min_confidence:
            return GeometryDecision(action=TradeAction.HOLD, rationale="Low confidence")
        
        direction = 1 if direction_signal > 0 else -1
        
        if abs(direction_signal) > cfg.strong_direction and confidence > cfg.high_confidence:
            action = TradeAction.ALLOW_SCALING
        else:
            action = TradeAction.ALLOW_LONG if direction > 0 else TradeAction.ALLOW_SHORT
        
        size = cfg.base_position_size
        if confidence > cfg.high_confidence:
            size += (cfg.max_position_size - cfg.base_position_size) * 0.5
        if abs(direction_signal) > cfg.strong_direction:
            size += (cfg.max_position_size - cfg.base_position_size) * 0.3
        
        risk_penalty = max(0, risk - 0.3) * 0.5
        size -= risk_penalty
        
        if fields.stability < 0:
            size *= (1 + fields.stability * 0.3)
        
        size = np.clip(size, cfg.min_position_size, cfg.max_position_size)
        
        return GeometryDecision(
            action=action,
            direction=direction,
            position_size=float(size),
            confidence=float(confidence),
            rationale=f"Dir={direction_signal:.2f}, Conf={confidence:.2f}"
        )


__all__ = ["TradeAction", "GeometryConfig", "GeometryDecision", "SignalGeometryEngine"]
