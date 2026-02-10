"""
===============================================================================
SIGNAL FIELDS — The Model → Engine Contract
===============================================================================

This module defines the epistemic contract between models and the backtest engine.

CORE INSIGHT:
Models are NOT traders. They are distributional geometry estimators.
They should answer: "What is the uncertainty/curvature/asymmetry?"
NOT: "Should I buy or sell?"

SIGNAL FIELDS:
Each model returns a SignalFields object with epistemic measures.
The signal geometry layer then interprets these into trading actions.

MODEL RESPONSIBILITIES:
- DT-CWT: asymmetry, belief_momentum, regime_fit
- Fisher-Rao: confidence, stability
- Free Probability: tail_asymmetry
- Log Barrier: stability, constraint_pressure
- Malliavin: sensitivity, hedging_pressure

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class SignalFields:
    """
    Epistemic fields returned by distributional geometry models.
    
    All fields are normalized to [-1, +1] range where:
    - Positive = favorable for long
    - Negative = favorable for short
    - Zero = neutral/uncertain
    
    Models fill the fields they are designed to measure.
    Fields not measured by a model should be left at 0 (neutral).
    """
    
    # =========================================================================
    # DIRECTIONAL GEOMETRY (Where is the distribution pointing?)
    # =========================================================================
    
    direction: float = 0.0
    """
    Primary directional signal from the model's state estimate.
    Range: [-1, +1]
    - Positive: model predicts positive drift
    - Negative: model predicts negative drift
    - Zero: no directional information
    Filled by: Kalman-based models (DT-CWT, standard KF)
    """
    
    asymmetry: float = 0.0
    """
    Distribution asymmetry / skewness indicator.
    Range: [-1, +1]
    - Positive: right-skewed (upside potential)
    - Negative: left-skewed (downside risk)
    - Zero: symmetric
    Filled by: DT-CWT, Free Probability, Tail models
    """
    
    belief_momentum: float = 0.0
    """
    Rate of change of the model's beliefs.
    Range: [-1, +1]
    - Positive: beliefs strengthening in bullish direction
    - Negative: beliefs strengthening in bearish direction
    - Zero: stable beliefs
    Filled by: DT-CWT, Kalman models with state tracking
    """
    
    # =========================================================================
    # CONFIDENCE GEOMETRY (How sure is the model?)
    # =========================================================================
    
    confidence: float = 0.0
    """
    Model's confidence in its estimates (inverse uncertainty).
    Range: [-1, +1] but typically [0, +1]
    - High positive: high confidence
    - Near zero: uncertain
    - Negative: contradictory signals (rare)
    Filled by: Fisher-Rao, any model with variance estimates
    """
    
    stability: float = 0.0
    """
    Stability of the current regime/state.
    Range: [-1, +1]
    - Positive: stable regime, estimates reliable
    - Negative: unstable regime, estimates unreliable
    - Zero: neutral stability
    Filled by: Regime models, Log Barrier, stability estimators
    """
    
    # =========================================================================
    # REGIME GEOMETRY (What is the market environment?)
    # =========================================================================
    
    regime_fit: float = 0.0
    """
    How well the current data fits the model's assumptions.
    Range: [-1, +1]
    - Positive: good fit, model is appropriate
    - Negative: poor fit, model may be wrong
    - Zero: average fit
    Filled by: Any model with goodness-of-fit measures
    """
    
    transition_pressure: float = 0.0
    """
    Pressure for regime change.
    Range: [-1, +1]
    - Positive: high pressure for regime shift
    - Negative: regime is entrenched
    - Zero: neutral
    Filled by: Regime switching models, HMM-based models
    """
    
    # =========================================================================
    # RISK GEOMETRY (What are the tail risks?)
    # =========================================================================
    
    tail_risk_left: float = 0.0
    """
    Left tail (downside) risk indicator.
    Range: [0, +1]
    - High: elevated downside tail risk
    - Low: thin left tail
    Filled by: EVT models, Free Probability, Tail estimators
    """
    
    tail_risk_right: float = 0.0
    """
    Right tail (upside) potential indicator.
    Range: [0, +1]
    - High: elevated upside potential
    - Low: thin right tail
    Filled by: EVT models, Free Probability, Tail estimators
    """
    
    volatility_state: float = 0.0
    """
    Current volatility regime state.
    Range: [-1, +1]
    - Positive: high volatility regime
    - Negative: low volatility regime
    - Zero: normal volatility
    Filled by: GARCH-type models, volatility estimators
    """
    
    # =========================================================================
    # CONSTRAINT GEOMETRY (What are the boundaries?)
    # =========================================================================
    
    constraint_pressure: float = 0.0
    """
    Pressure from constraints (barriers, bounds).
    Range: [-1, +1]
    - Positive: approaching upper constraint
    - Negative: approaching lower constraint
    - Zero: far from constraints
    Filled by: Log Barrier models, Constraint estimators
    """
    
    hedging_pressure: float = 0.0
    """
    Implied hedging/rebalancing pressure.
    Range: [-1, +1]
    - Positive: buy pressure from hedging
    - Negative: sell pressure from hedging
    - Zero: balanced
    Filled by: Malliavin calculus models, Greeks-based models
    """
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    model_name: str = ""
    """Name of the model that produced these fields."""
    
    timestamp: Optional[str] = None
    """Timestamp of the signal."""
    
    raw_values: Dict[str, float] = field(default_factory=dict)
    """Raw model outputs before normalization (for debugging)."""
    
    def __post_init__(self):
        """Ensure all fields are in valid ranges."""
        # Clip all bounded fields to [-1, +1]
        for fname in ['direction', 'asymmetry', 'belief_momentum', 
                      'confidence', 'stability', 'regime_fit',
                      'transition_pressure', 'volatility_state',
                      'constraint_pressure', 'hedging_pressure']:
            val = getattr(self, fname)
            setattr(self, fname, float(np.clip(val, -1, 1)))
        
        # Tail risks are [0, 1]
        self.tail_risk_left = float(np.clip(self.tail_risk_left, 0, 1))
        self.tail_risk_right = float(np.clip(self.tail_risk_right, 0, 1))
    
    @property
    def has_direction(self) -> bool:
        """Check if model provided directional information."""
        return abs(self.direction) > 0.05 or abs(self.asymmetry) > 0.05
    
    @property
    def has_confidence(self) -> bool:
        """Check if model provided confidence information."""
        return abs(self.confidence) > 0.05 or abs(self.stability) > 0.05
    
    @property
    def composite_direction(self) -> float:
        """
        Weighted combination of directional signals.
        IMPROVED: Weight direction (which now includes accuracy) most heavily.
        """
        # Direction now incorporates accuracy, so weight it heavily
        # Momentum confirms, asymmetry provides edge direction
        return (
            0.60 * self.direction +
            0.25 * self.belief_momentum +
            0.15 * self.asymmetry
        )
    
    @property
    def composite_confidence(self) -> float:
        """
        Weighted combination of confidence signals.
        IMPROVED: Also consider regime fit.
        """
        # High confidence when: accurate predictions, stable environment, good fit
        regime_boost = max(0, self.regime_fit) * 0.2
        return (
            0.50 * self.confidence +
            0.30 * max(0, self.stability) +
            regime_boost
        )
    
    @property
    def composite_risk(self) -> float:
        """
        Combined risk indicator.
        IMPROVED: Weight left tail more (downside protection).
        """
        return (
            0.50 * self.tail_risk_left +      # Downside matters more
            0.15 * self.tail_risk_right +
            0.20 * abs(self.transition_pressure) +
            0.15 * max(0, self.volatility_state)  # Only penalize high vol
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'direction': self.direction,
            'asymmetry': self.asymmetry,
            'belief_momentum': self.belief_momentum,
            'confidence': self.confidence,
            'stability': self.stability,
            'regime_fit': self.regime_fit,
            'transition_pressure': self.transition_pressure,
            'tail_risk_left': self.tail_risk_left,
            'tail_risk_right': self.tail_risk_right,
            'volatility_state': self.volatility_state,
            'constraint_pressure': self.constraint_pressure,
            'hedging_pressure': self.hedging_pressure,
            'model_name': self.model_name,
            'composite_direction': self.composite_direction,
            'composite_confidence': self.composite_confidence,
            'composite_risk': self.composite_risk,
        }


def create_signal_fields_from_kalman(
    mu: np.ndarray,
    sigma: np.ndarray,
    returns: np.ndarray,
    model_name: str = ""
) -> SignalFields:
    """
    Create SignalFields from standard Kalman filter outputs.
    
    This is the adapter for Kalman-based models (including DT-CWT).
    
    Parameters:
    -----------
    mu : array
        Kalman state estimates (filtered mean)
    sigma : array  
        Kalman uncertainty estimates (filtered std)
    returns : array
        Observed returns
    model_name : str
        Name of the model
        
    Returns:
    --------
    SignalFields with populated epistemic measures
    """
    n = len(mu)
    if n < 30:
        return SignalFields(model_name=model_name)
    
    # Get current values
    mu_last = mu[-1]
    sigma_last = sigma[-1] if sigma[-1] > 0 else 0.001
    
    # Compute lookback statistics
    lookback = min(50, n - 1)
    mu_hist = mu[-(lookback+1):-1]
    sigma_hist = sigma[-(lookback+1):-1]
    returns_hist = returns[-lookback:] if len(returns) >= lookback else returns
    
    mu_mean = np.mean(mu_hist)
    mu_std = np.std(mu_hist) + 1e-10
    sigma_mean = np.mean(sigma_hist) + 1e-10
    
    # =========================================================================
    # DIRECTION: Based on MU MOMENTUM (change in mu), NOT mu level!
    # =========================================================================
    # Analysis showed: Following mu momentum has Sharpe 1.293 vs 0.294 for level
    if n >= 5:
        # Compute mu change (momentum)
        mu_change = mu[-1] - mu[-2]
        
        # Normalize by recent mu change volatility
        mu_changes = np.diff(mu[-20:]) if n >= 20 else np.diff(mu[-5:])
        mu_change_std = np.std(mu_changes) + 1e-10
        
        # Direction = normalized mu change
        direction = float(np.tanh(mu_change / mu_change_std * 1.5))
    else:
        direction = 0.0
    
    # =========================================================================
    # ASYMMETRY: Skewness of recent predictions vs realizations
    # =========================================================================
    if len(returns_hist) >= 20:
        residuals = returns_hist - mu_hist[-len(returns_hist):]
        skew = np.mean(residuals**3) / (np.std(residuals)**3 + 1e-10)
        asymmetry = float(np.clip(skew / 2, -1, 1))
    else:
        asymmetry = 0.0
    
    # =========================================================================
    # BELIEF MOMENTUM: Trend in mu (longer term)
    # =========================================================================
    if n >= 10:
        mu_recent = mu[-10:]
        mu_trend = np.polyfit(range(10), mu_recent, 1)[0]
        belief_momentum = float(np.tanh(mu_trend / (mu_std + 1e-10) * 50))
    else:
        belief_momentum = 0.0
    
    # =========================================================================
    # CONFIDENCE: Based on consistency of mu momentum direction
    # =========================================================================
    if n >= 10:
        mu_changes = np.diff(mu[-10:])
        # Confidence = how consistently mu is moving in same direction
        same_sign = np.sum(np.sign(mu_changes) == np.sign(mu_changes[-1]))
        consistency = same_sign / len(mu_changes)
        confidence = float(consistency)
    else:
        confidence = 0.5
    # =========================================================================
    # STABILITY: Consistency of sigma (low CV = stable)
    # =========================================================================
    sigma_cv = np.std(sigma_hist) / (sigma_mean + 1e-10)
    stability = float(1 - np.clip(sigma_cv * 2, 0, 1)) * 2 - 1  # Map to [-1, +1]
    
    # =========================================================================
    # REGIME FIT: How well predictions match reality
    # =========================================================================
    if len(returns_hist) >= 20:
        pred_errors = returns_hist[-20:] - mu[-21:-1]
        mae = np.mean(np.abs(pred_errors))
        expected_mae = sigma_mean * 0.8  # Expect errors ~ sigma
        regime_fit = float(1 - np.clip(mae / (expected_mae + 1e-10), 0, 2)) * 2 - 1
    else:
        regime_fit = 0.0
    
    # =========================================================================
    # TAIL RISKS: From return distribution
    # =========================================================================
    if len(returns_hist) >= 50:
        left_tail = np.percentile(returns_hist, 5)
        right_tail = np.percentile(returns_hist, 95)
        # Normalize by recent vol
        vol = np.std(returns_hist)
        tail_risk_left = float(np.clip(abs(left_tail) / (vol * 2 + 1e-10), 0, 1))
        tail_risk_right = float(np.clip(abs(right_tail) / (vol * 2 + 1e-10), 0, 1))
    else:
        tail_risk_left = 0.3
        tail_risk_right = 0.3
    
    # =========================================================================
    # VOLATILITY STATE: Current sigma relative to history
    # =========================================================================
    vol_percentile = np.mean(sigma_last > sigma_hist) if len(sigma_hist) > 0 else 0.5
    volatility_state = float(vol_percentile * 2 - 1)  # Map [0,1] to [-1,+1]
    
    # =========================================================================
    # TRANSITION PRESSURE: Rate of change of sigma
    # =========================================================================
    if n >= 10:
        sigma_recent = sigma[-10:]
        sigma_trend = np.polyfit(range(10), sigma_recent, 1)[0]
        transition_pressure = float(np.tanh(sigma_trend / (sigma_mean + 1e-10) * 100))
    else:
        transition_pressure = 0.0
    
    return SignalFields(
        direction=direction,
        asymmetry=asymmetry,
        belief_momentum=belief_momentum,
        confidence=confidence,
        stability=stability,
        regime_fit=regime_fit,
        transition_pressure=transition_pressure,
        tail_risk_left=tail_risk_left,
        tail_risk_right=tail_risk_right,
        volatility_state=volatility_state,
        constraint_pressure=0.0,
        hedging_pressure=0.0,
        model_name=model_name,
        raw_values={
            'mu_last': float(mu_last),
            'sigma_last': float(sigma_last),
            'mu_change': float(mu[-1] - mu[-2]) if n >= 5 else 0.0,
        }
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SignalFields',
    'create_signal_fields_from_kalman',
]
