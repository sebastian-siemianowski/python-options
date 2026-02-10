"""
===============================================================================
SIGNAL GEOMETRY ELITE V2 — Proper Alpha Extraction Framework
===============================================================================

CRITICAL INSIGHT:
The Kalman filter's mu IS the predicted return. The previous implementation
was destroying this signal by over-normalizing it.

PROPER ALPHA EXTRACTION:
1. mu > 0 → predicted positive return → LONG
2. mu < 0 → predicted negative return → SHORT
3. |mu| / sigma → signal-to-noise ratio → confidence
4. Position size = f(confidence, risk_budget)

This module implements the correct translation from distributional
estimates to trading signals, as practiced by elite quant funds.

THEORETICAL FOUNDATION:
- Kelly Criterion: f* = μ/σ² for optimal growth
- Information Ratio: IR = μ/σ measures signal quality
- Sharpe-optimal sizing: position ∝ μ/σ²

Author: Chinese Staff Professor - Elite Quant Systems
Specialization: Top 0.000001% Hedge Fund Quantitative Research
Date: February 2026
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass 
class EliteConfigV2:
    """
    Elite signal geometry configuration - V2.
    
    Key principle: Let the model's predictions flow through with minimal
    distortion. Risk management via position sizing, not signal filtering.
    """
    
    # SIGNAL EXTRACTION
    # The model's mu is in return space (e.g., 0.001 = 0.1% expected return)
    # We scale by annualized volatility to get a standardized signal
    signal_scaling: float = 1.0           # Direct pass-through
    min_signal_threshold: float = 0.0001  # 1bp minimum signal (very permissive)
    
    # POSITION SIZING (Kelly-based)
    kelly_fraction: float = 0.25          # Quarter Kelly for safety
    max_position: float = 0.40            # Maximum 40% of capital
    min_position: float = 0.05            # Minimum 5% when trading
    
    # CONFIDENCE CALIBRATION
    confidence_floor: float = 0.30        # Minimum confidence to trade
    snr_scale: float = 10.0               # SNR scaling factor
    
    # VOLATILITY SCALING
    target_vol: float = 0.15              # Target 15% annualized vol
    vol_lookback: int = 20                # Lookback for vol estimation
    
    # REGIME AWARENESS
    regime_sensitivity: float = 0.3       # How much regime affects sizing
    trend_following_weight: float = 0.2   # Weight on trend persistence
    
    # RISK CONTROLS
    max_drawdown_scale: float = 0.5       # Reduce size as drawdown increases
    correlation_penalty: float = 0.2      # Penalty for high correlation


# =============================================================================
# SIGNAL FIELDS V2 - SIMPLIFIED FOR CLARITY
# =============================================================================

@dataclass
class SignalFieldsV2:
    """
    Simplified signal fields focused on what matters:
    - Direction (the predicted return sign)
    - Magnitude (the predicted return size)
    - Confidence (how sure we are)
    - Risk (how to size the position)
    """
    
    # Core signals
    predicted_return: float = 0.0      # Raw mu from Kalman
    prediction_std: float = 0.0        # Sigma from Kalman
    
    # Derived signals
    direction: float = 0.0             # Sign of predicted return [-1, +1]
    magnitude: float = 0.0             # Absolute predicted return
    signal_to_noise: float = 0.0       # |mu| / sigma
    
    # Confidence metrics
    confidence: float = 0.0            # Overall trading confidence [0, 1]
    persistence: float = 0.0           # Signal persistence [0, 1]
    
    # Risk metrics
    realized_vol: float = 0.0          # Recent realized volatility
    vol_regime: float = 0.5            # Volatility percentile [0, 1]
    drawdown: float = 0.0              # Current drawdown
    
    # Regime metrics
    trend_strength: float = 0.0        # Trend following signal [-1, +1]
    regime_stability: float = 0.0      # Regime stability [-1, +1]
    
    # Metadata
    model_name: str = ""
    
    def __post_init__(self):
        """Derive secondary fields from primary."""
        if self.prediction_std > 0:
            self.signal_to_noise = abs(self.predicted_return) / self.prediction_std
        if self.predicted_return != 0:
            self.direction = np.sign(self.predicted_return)
            self.magnitude = abs(self.predicted_return)


# =============================================================================
# DECISION OUTPUT
# =============================================================================

class ActionV2(Enum):
    """Trading actions."""
    STRONG_LONG = auto()
    LONG = auto()
    HOLD = auto()
    SHORT = auto()
    STRONG_SHORT = auto()


@dataclass
class DecisionV2:
    """Trading decision with full context."""
    
    action: ActionV2 = ActionV2.HOLD
    position_size: float = 0.0          # [0, 1] recommended size
    direction: int = 0                  # -1, 0, or +1
    
    # Kelly-optimal metrics
    kelly_fraction: float = 0.0         # Optimal Kelly fraction
    expected_return: float = 0.0        # Expected return if executed
    expected_risk: float = 0.0          # Expected risk (std)
    
    # Confidence
    confidence: float = 0.0             # Decision confidence
    
    # Rationale
    rationale: List[str] = field(default_factory=list)
    
    @property
    def position_signal(self) -> float:
        """Convert to position signal [-1, +1]."""
        return self.direction * self.position_size


# =============================================================================
# ELITE SIGNAL ENGINE V2
# =============================================================================

class EliteSignalEngineV2:
    """
    Elite signal geometry engine - V2.
    
    PHILOSOPHY:
    The model has done the hard work of predicting returns.
    Our job is to:
    1. Extract the signal cleanly
    2. Assess confidence
    3. Size positions optimally
    4. Manage risk
    
    We do NOT:
    - Second-guess the model's direction
    - Over-filter signals
    - Destroy edge through excessive caution
    """
    
    def __init__(self, config: EliteConfigV2 = None):
        self.config = config or EliteConfigV2()
    
    def extract_signals(
        self, 
        mu: np.ndarray, 
        sigma: np.ndarray, 
        returns: np.ndarray,
        model_name: str = ""
    ) -> SignalFieldsV2:
        """
        Extract trading signals from Kalman filter outputs.
        
        KEY INSIGHT: mu IS the predicted return. Don't destroy it.
        """
        n = len(mu)
        if n < 30:
            return SignalFieldsV2(model_name=model_name)
        
        # Current predictions
        mu_last = mu[-1]
        sigma_last = sigma[-1] if sigma[-1] > 0 else 0.001
        
        # Historical context
        lookback = min(100, n)
        mu_hist = mu[-lookback:]
        sigma_hist = sigma[-lookback:]
        returns_hist = returns[-lookback:] if len(returns) >= lookback else returns
        
        # =================================================================
        # CORE SIGNAL: Predicted return
        # =================================================================
        predicted_return = float(mu_last)
        prediction_std = float(sigma_last)
        
        # Signal-to-noise ratio
        snr = abs(mu_last) / (sigma_last + 1e-10)
        
        # =================================================================
        # CONFIDENCE: Based on signal quality
        # =================================================================
        
        # SNR-based confidence
        snr_confidence = np.tanh(snr * self.config.snr_scale)
        
        # Persistence: Is the signal consistent?
        if n >= 5:
            recent_signs = np.sign(mu[-5:])
            current_sign = np.sign(mu_last)
            persistence = np.mean(recent_signs == current_sign)
        else:
            persistence = 0.5
        
        # Prediction accuracy (how well did past predictions do?)
        if n >= 20:
            # Compare predicted direction with actual direction
            predicted_dirs = np.sign(mu[-21:-1])
            actual_dirs = np.sign(returns[-20:])
            accuracy = np.mean(predicted_dirs == actual_dirs)
        else:
            accuracy = 0.5
        
        # Combined confidence
        confidence = (
            0.4 * snr_confidence +
            0.3 * persistence +
            0.3 * accuracy
        )
        confidence = float(np.clip(confidence, 0, 1))
        
        # =================================================================
        # RISK METRICS
        # =================================================================
        
        # Realized volatility
        realized_vol = float(np.std(returns_hist) * np.sqrt(252))
        
        # Volatility regime (percentile)
        sigma_mean = np.mean(sigma_hist)
        if sigma_last > sigma_mean * 1.5:
            vol_regime = 0.9  # High vol
        elif sigma_last < sigma_mean * 0.5:
            vol_regime = 0.1  # Low vol
        else:
            vol_regime = 0.5  # Normal
        
        # Drawdown
        if len(returns_hist) >= 20:
            cumret = np.cumprod(1 + returns_hist) - 1
            peak = np.maximum.accumulate(cumret + 1)
            drawdown = float(1 - (cumret[-1] + 1) / peak[-1])
        else:
            drawdown = 0.0
        
        # =================================================================
        # REGIME METRICS
        # =================================================================
        
        # Trend strength (momentum in mu)
        if n >= 10:
            mu_momentum = (mu[-1] - mu[-10]) / (np.std(mu[-10:]) + 1e-10)
            trend_strength = float(np.tanh(mu_momentum * 0.5))
        else:
            trend_strength = 0.0
        
        # Regime stability (consistency of sigma)
        sigma_cv = np.std(sigma_hist) / (np.mean(sigma_hist) + 1e-10)
        regime_stability = float(1 - np.clip(sigma_cv * 2, 0, 1)) * 2 - 1
        
        return SignalFieldsV2(
            predicted_return=predicted_return,
            prediction_std=prediction_std,
            signal_to_noise=snr,
            confidence=confidence,
            persistence=persistence,
            realized_vol=realized_vol,
            vol_regime=vol_regime,
            drawdown=drawdown,
            trend_strength=trend_strength,
            regime_stability=regime_stability,
            model_name=model_name
        )
    
    def decide(self, fields: SignalFieldsV2) -> DecisionV2:
        """
        Make trading decision based on signal fields.
        
        CORE LOGIC:
        1. If signal is strong enough and confidence is high → trade
        2. Direction = sign of predicted return
        3. Size = Kelly-optimal fraction adjusted for risk
        """
        decision = DecisionV2()
        rationale = []
        
        # =================================================================
        # STEP 1: Check if signal is tradeable
        # =================================================================
        
        if abs(fields.predicted_return) < self.config.min_signal_threshold:
            decision.action = ActionV2.HOLD
            decision.rationale = ["Signal below threshold"]
            return decision
        
        if fields.confidence < self.config.confidence_floor:
            decision.action = ActionV2.HOLD
            decision.rationale = [f"Low confidence: {fields.confidence:.2f}"]
            return decision
        
        # =================================================================
        # STEP 2: Determine direction
        # =================================================================
        
        direction = int(np.sign(fields.predicted_return))
        decision.direction = direction
        rationale.append(f"Direction: {'LONG' if direction > 0 else 'SHORT'}")
        
        # =================================================================
        # STEP 3: Calculate Kelly-optimal position size
        # =================================================================
        
        # Kelly formula: f* = μ/σ² (for continuous case)
        # We use: f* = (predicted_return / prediction_std²) * kelly_fraction
        
        if fields.prediction_std > 0:
            # Raw Kelly fraction
            kelly_raw = abs(fields.predicted_return) / (fields.prediction_std ** 2)
            
            # Scale by our Kelly fraction (quarter Kelly for safety)
            kelly_scaled = kelly_raw * self.config.kelly_fraction
            
            # Apply confidence scaling
            kelly_confident = kelly_scaled * fields.confidence
            
            decision.kelly_fraction = float(kelly_raw)
            rationale.append(f"Raw Kelly: {kelly_raw:.3f}")
        else:
            kelly_confident = 0.1
        
        # =================================================================
        # STEP 4: Risk adjustments
        # =================================================================
        
        # Volatility targeting
        if fields.realized_vol > 0:
            vol_scalar = self.config.target_vol / fields.realized_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)
            kelly_confident *= vol_scalar
            rationale.append(f"Vol scale: {vol_scalar:.2f}")
        
        # Regime adjustment
        if fields.regime_stability < -0.3:
            kelly_confident *= (1 - self.config.regime_sensitivity)
            rationale.append("Regime unstable")
        elif fields.regime_stability > 0.3:
            kelly_confident *= (1 + self.config.regime_sensitivity * 0.5)
        
        # Trend following boost
        if direction * fields.trend_strength > 0.3:
            kelly_confident *= (1 + self.config.trend_following_weight)
            rationale.append("With trend")
        
        # Drawdown reduction
        if fields.drawdown > 0.1:
            dd_penalty = 1 - fields.drawdown * self.config.max_drawdown_scale
            kelly_confident *= max(0.3, dd_penalty)
            rationale.append(f"DD penalty: {dd_penalty:.2f}")
        
        # =================================================================
        # STEP 5: Apply position limits
        # =================================================================
        
        position_size = float(np.clip(
            kelly_confident,
            self.config.min_position if kelly_confident > 0.01 else 0,
            self.config.max_position
        ))
        
        decision.position_size = position_size
        decision.confidence = fields.confidence
        decision.expected_return = fields.predicted_return * position_size
        decision.expected_risk = fields.prediction_std * position_size
        
        rationale.append(f"Position: {position_size:.2%}")
        
        # =================================================================
        # STEP 6: Classify action
        # =================================================================
        
        signal = direction * position_size
        
        if signal > 0.25:
            decision.action = ActionV2.STRONG_LONG
        elif signal > 0.05:
            decision.action = ActionV2.LONG
        elif signal < -0.25:
            decision.action = ActionV2.STRONG_SHORT
        elif signal < -0.05:
            decision.action = ActionV2.SHORT
        else:
            decision.action = ActionV2.HOLD
        
        decision.rationale = rationale
        
        return decision
    
    def get_position_signal(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        returns: np.ndarray,
        model_name: str = ""
    ) -> float:
        """
        Convenience method: Get position signal directly from Kalman outputs.
        
        Returns: Position signal in [-1, +1] range.
        """
        fields = self.extract_signals(mu, sigma, returns, model_name)
        decision = self.decide(fields)
        return decision.position_signal


# =============================================================================
# ADAPTER FOR BACKTEST ENGINE
# =============================================================================

def elite_signal_v2(
    mu: np.ndarray,
    sigma: np.ndarray,
    returns: np.ndarray,
    model_name: str = "",
    config: EliteConfigV2 = None
) -> float:
    """
    Convert Kalman outputs to position signal using Elite V2 engine.
    
    This is the main entry point for the backtest engine.
    """
    engine = EliteSignalEngineV2(config)
    return engine.get_position_signal(mu, sigma, returns, model_name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'EliteConfigV2',
    'SignalFieldsV2',
    'ActionV2',
    'DecisionV2',
    'EliteSignalEngineV2',
    'elite_signal_v2',
]
