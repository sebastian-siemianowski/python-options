"""
===============================================================================
REGIME GOVERNANCE MODULE
===============================================================================

Implements the Copilot Story QUANT-2025-MRT-001 requirements for
institutional-grade risk temperature governance.

CHINESE QUANTITATIVE SYSTEMS PROFESSOR SOLUTION — Control Theory Design:

    1. HYSTERESIS BANDS: Asymmetric thresholds for regime transitions
       - Prevents oscillation at regime boundaries
       - Implements regime memory/persistence
       
    2. CONSERVATIVE IMPUTATION: Missing data handling
       - 75th percentile imputation for unavailable indicators
       - Defensive degradation when data quality deteriorates
       
    3. AUDIT TRAIL: Complete attribution and reconstruction capability
       - Raw values, z-scores, contributions all preserved
       - JSON-serializable for storage and retrieval
       
    4. DYNAMIC GAP RISK: Adaptive overnight budget constraint
       - 95th percentile of trailing 60-day gaps
       - Floor and ceiling bounds for stability
       
    5. RATE LIMITING (Governor): Maximum temperature change per day
       - Prevents whipsawing from single-day market movements

REGIME STATE MACHINE:
    
    States: Calm → Elevated → Stressed → Extreme
    
    Upward Thresholds:
        Calm → Elevated:     temp > 0.5
        Elevated → Stressed: temp > 1.0
        Stressed → Extreme:  temp > 1.5
        
    Downward Thresholds (Hysteresis Gap):
        Extreme → Stressed:  temp < 1.2
        Stressed → Elevated: temp < 0.7
        Elevated → Calm:     temp < 0.3

REFERENCES:
    Institutional Design Review — Phase III Arbitration
    Expert Panel Evaluation (February 2026)
    
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME STATE ENUM
# =============================================================================

class RegimeState(Enum):
    """
    Discrete regime states for risk temperature governance.
    
    Implements control-theoretic regime persistence to prevent
    oscillatory behavior at regime boundaries.
    """
    CALM = "Calm"
    ELEVATED = "Elevated"
    STRESSED = "Stressed"
    EXTREME = "Extreme"
    
    @classmethod
    def from_string(cls, s: str) -> "RegimeState":
        """Convert string representation to RegimeState."""
        mapping = {
            "calm": cls.CALM,
            "elevated": cls.ELEVATED,
            "stressed": cls.STRESSED,
            "extreme": cls.EXTREME,
        }
        return mapping.get(s.lower(), cls.CALM)


# =============================================================================
# HYSTERESIS CONSTANTS
# =============================================================================

# Upward transition thresholds (temperature must exceed to transition up)
THRESHOLD_CALM_TO_ELEVATED = 0.5
THRESHOLD_ELEVATED_TO_STRESSED = 1.0
THRESHOLD_STRESSED_TO_EXTREME = 1.5

# Downward transition thresholds (temperature must fall below to transition down)
# These are lower than upward thresholds to create hysteresis gap
THRESHOLD_EXTREME_TO_STRESSED = 1.2    # Gap of 0.3 from upward threshold
THRESHOLD_STRESSED_TO_ELEVATED = 0.7   # Gap of 0.3 from upward threshold
THRESHOLD_ELEVATED_TO_CALM = 0.3       # Gap of 0.2 from upward threshold

# Minimum holding period for elevated states (trading days)
MIN_HOLDING_PERIOD_DAYS = 5

# Maximum temperature change per day (rate limiting / governor)
MAX_TEMP_CHANGE_PER_DAY = 0.3


# =============================================================================
# CONSERVATIVE IMPUTATION CONSTANTS
# =============================================================================

# Historical lookback for imputation statistics (trading days)
IMPUTATION_HISTORY_DAYS = 252

# Percentile for conservative imputation (higher = more conservative)
IMPUTATION_PERCENTILE = 75

# Threshold for warning on imputed indicators
IMPUTATION_WARNING_THRESHOLD = 0.4  # 40% of indicators imputed

# Temperature floor when too many indicators are imputed
IMPUTATION_TEMPERATURE_FLOOR = 0.5


# =============================================================================
# DYNAMIC GAP RISK CONSTANTS
# =============================================================================

# Lookback for gap risk estimation (trading days)
GAP_RISK_LOOKBACK_DAYS = 60

# Percentile for gap risk estimation
GAP_RISK_PERCENTILE = 95

# Floor and ceiling for gap risk estimate
GAP_RISK_FLOOR = 0.015  # 1.5%
GAP_RISK_CEILING = 0.06  # 6.0%

# Exponential smoothing factor for gap risk (0 = no smoothing, 1 = full weight to new)
GAP_RISK_SMOOTHING_ALPHA = 0.3


# =============================================================================
# SCALE FACTOR MAPPING BY REGIME STATE
# =============================================================================

# Scale factors based on discrete regime state (not continuous temperature)
# This provides more stable position sizing
REGIME_SCALE_FACTORS = {
    RegimeState.CALM: 1.0,
    RegimeState.ELEVATED: 0.75,
    RegimeState.STRESSED: 0.45,
    RegimeState.EXTREME: 0.20,
}


# =============================================================================
# AUDIT TRAIL DATA CLASSES
# =============================================================================

@dataclass
class IndicatorAuditRecord:
    """
    Complete audit record for a single stress indicator.
    
    Contains all information needed to reconstruct the computation.
    """
    name: str
    
    # Raw data
    raw_value: Optional[float]
    data_available: bool
    imputed: bool
    imputed_value: Optional[float] = None
    
    # Lookback window
    lookback_start_date: Optional[str] = None
    lookback_end_date: Optional[str] = None
    
    # Z-score computation
    lookback_mean: Optional[float] = None
    lookback_std: Optional[float] = None
    zscore: Optional[float] = None
    
    # Contribution
    weight: float = 0.0
    stress_contribution: float = 0.0
    
    # Metadata
    interpretation: str = ""
    computation_notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "data_available": self.data_available,
            "imputed": self.imputed,
            "imputed_value": self.imputed_value,
            "lookback_start_date": self.lookback_start_date,
            "lookback_end_date": self.lookback_end_date,
            "lookback_mean": self.lookback_mean,
            "lookback_std": self.lookback_std,
            "zscore": self.zscore,
            "weight": self.weight,
            "stress_contribution": self.stress_contribution,
            "interpretation": self.interpretation,
            "computation_notes": self.computation_notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "IndicatorAuditRecord":
        """Create from dictionary."""
        return cls(
            name=d.get("name", ""),
            raw_value=d.get("raw_value"),
            data_available=d.get("data_available", False),
            imputed=d.get("imputed", False),
            imputed_value=d.get("imputed_value"),
            lookback_start_date=d.get("lookback_start_date"),
            lookback_end_date=d.get("lookback_end_date"),
            lookback_mean=d.get("lookback_mean"),
            lookback_std=d.get("lookback_std"),
            zscore=d.get("zscore"),
            weight=d.get("weight", 0.0),
            stress_contribution=d.get("stress_contribution", 0.0),
            interpretation=d.get("interpretation", ""),
            computation_notes=d.get("computation_notes", ""),
        )


@dataclass
class CategoryAuditRecord:
    """
    Audit record for a stress category (FX, Futures, Rates, etc.)
    """
    name: str
    weight: float
    indicators: List[IndicatorAuditRecord]
    stress_level: float
    weighted_contribution: float
    aggregation_method: str = "max"  # 'max', 'mean', or 'weighted_sum'
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "stress_level": self.stress_level,
            "weighted_contribution": self.weighted_contribution,
            "aggregation_method": self.aggregation_method,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CategoryAuditRecord":
        indicators = [IndicatorAuditRecord.from_dict(i) for i in d.get("indicators", [])]
        return cls(
            name=d.get("name", ""),
            weight=d.get("weight", 0.0),
            indicators=indicators,
            stress_level=d.get("stress_level", 0.0),
            weighted_contribution=d.get("weighted_contribution", 0.0),
            aggregation_method=d.get("aggregation_method", "max"),
        )


@dataclass
class RegimeTransitionRecord:
    """
    Record of a regime state transition event.
    """
    timestamp: str
    previous_state: str
    new_state: str
    raw_temperature: float
    transition_threshold: float
    transition_direction: str  # 'up' or 'down'
    holding_period_met: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "RegimeTransitionRecord":
        return cls(**d)


@dataclass
class GapRiskAuditRecord:
    """
    Audit record for dynamic gap risk estimation.
    """
    computed_at: str
    lookback_start: str
    lookback_end: str
    raw_estimate: float
    smoothed_estimate: float
    applied_estimate: float  # After floor/ceiling
    floor_applied: bool
    ceiling_applied: bool
    smoothing_alpha: float
    previous_estimate: Optional[float]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "GapRiskAuditRecord":
        return cls(**d)


@dataclass
class GovernedRiskTemperatureAudit:
    """
    Complete audit trail for a governed risk temperature computation.
    
    This is the master audit record that enables full reconstruction
    of any historical temperature computation.
    """
    # Computation metadata
    computed_at: str
    version: str = "2.0.0"
    
    # Raw temperature (before governance)
    raw_temperature: float = 0.0
    
    # Rate-limited temperature (after governor)
    rate_limited_temperature: float = 0.0
    previous_temperature: Optional[float] = None
    rate_limit_applied: bool = False
    
    # Regime state
    regime_state: str = "Calm"
    previous_regime_state: Optional[str] = None
    regime_transition_occurred: bool = False
    regime_transition: Optional[RegimeTransitionRecord] = None
    holding_period_start: Optional[str] = None
    holding_period_days: int = 0
    
    # Scale factor
    scale_factor: float = 1.0
    scale_factor_method: str = "regime_based"  # 'regime_based' or 'sigmoid'
    
    # Data quality
    total_indicators: int = 0
    available_indicators: int = 0
    imputed_indicators: int = 0
    data_quality: float = 1.0
    imputation_warning: bool = False
    temperature_floor_applied: bool = False
    
    # Gap risk
    gap_risk_estimate: float = 0.03
    gap_risk_audit: Optional[GapRiskAuditRecord] = None
    
    # Overnight budget
    overnight_budget_active: bool = False
    overnight_max_position: Optional[float] = None
    
    # Category breakdown
    categories: Dict[str, CategoryAuditRecord] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "computed_at": self.computed_at,
            "version": self.version,
            "raw_temperature": self.raw_temperature,
            "rate_limited_temperature": self.rate_limited_temperature,
            "previous_temperature": self.previous_temperature,
            "rate_limit_applied": self.rate_limit_applied,
            "regime_state": self.regime_state,
            "previous_regime_state": self.previous_regime_state,
            "regime_transition_occurred": self.regime_transition_occurred,
            "regime_transition": self.regime_transition.to_dict() if self.regime_transition else None,
            "holding_period_start": self.holding_period_start,
            "holding_period_days": self.holding_period_days,
            "scale_factor": self.scale_factor,
            "scale_factor_method": self.scale_factor_method,
            "total_indicators": self.total_indicators,
            "available_indicators": self.available_indicators,
            "imputed_indicators": self.imputed_indicators,
            "data_quality": self.data_quality,
            "imputation_warning": self.imputation_warning,
            "temperature_floor_applied": self.temperature_floor_applied,
            "gap_risk_estimate": self.gap_risk_estimate,
            "gap_risk_audit": self.gap_risk_audit.to_dict() if self.gap_risk_audit else None,
            "overnight_budget_active": self.overnight_budget_active,
            "overnight_max_position": self.overnight_max_position,
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
        }
        return result
    
    @classmethod
    def from_dict(cls, d: Dict) -> "GovernedRiskTemperatureAudit":
        """Create from dictionary."""
        categories = {k: CategoryAuditRecord.from_dict(v) for k, v in d.get("categories", {}).items()}
        
        regime_transition = None
        if d.get("regime_transition"):
            regime_transition = RegimeTransitionRecord.from_dict(d["regime_transition"])
            
        gap_risk_audit = None
        if d.get("gap_risk_audit"):
            gap_risk_audit = GapRiskAuditRecord.from_dict(d["gap_risk_audit"])
        
        return cls(
            computed_at=d.get("computed_at", datetime.now().isoformat()),
            version=d.get("version", "2.0.0"),
            raw_temperature=d.get("raw_temperature", 0.0),
            rate_limited_temperature=d.get("rate_limited_temperature", 0.0),
            previous_temperature=d.get("previous_temperature"),
            rate_limit_applied=d.get("rate_limit_applied", False),
            regime_state=d.get("regime_state", "Calm"),
            previous_regime_state=d.get("previous_regime_state"),
            regime_transition_occurred=d.get("regime_transition_occurred", False),
            regime_transition=regime_transition,
            holding_period_start=d.get("holding_period_start"),
            holding_period_days=d.get("holding_period_days", 0),
            scale_factor=d.get("scale_factor", 1.0),
            scale_factor_method=d.get("scale_factor_method", "regime_based"),
            total_indicators=d.get("total_indicators", 0),
            available_indicators=d.get("available_indicators", 0),
            imputed_indicators=d.get("imputed_indicators", 0),
            data_quality=d.get("data_quality", 1.0),
            imputation_warning=d.get("imputation_warning", False),
            temperature_floor_applied=d.get("temperature_floor_applied", False),
            gap_risk_estimate=d.get("gap_risk_estimate", 0.03),
            gap_risk_audit=gap_risk_audit,
            overnight_budget_active=d.get("overnight_budget_active", False),
            overnight_max_position=d.get("overnight_max_position"),
            categories=categories,
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "GovernedRiskTemperatureAudit":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def render_human_readable(self) -> str:
        """Render audit trail in human-readable format."""
        lines = []
        lines.append("=" * 70)
        lines.append("RISK TEMPERATURE AUDIT TRAIL")
        lines.append("=" * 70)
        lines.append(f"Computed At:     {self.computed_at}")
        lines.append(f"Version:         {self.version}")
        lines.append("")
        
        lines.append("TEMPERATURE COMPUTATION")
        lines.append("-" * 40)
        lines.append(f"Raw Temperature:          {self.raw_temperature:.4f}")
        if self.rate_limit_applied and self.previous_temperature is not None:
            lines.append(f"Previous Temperature:     {self.previous_temperature:.4f}")
            lines.append(f"Rate Limited Temperature: {self.rate_limited_temperature:.4f}")
            lines.append(f"Rate Limit Applied:       Yes (max Δ = {MAX_TEMP_CHANGE_PER_DAY})")
        lines.append("")
        
        lines.append("REGIME STATE")
        lines.append("-" * 40)
        lines.append(f"Current State:    {self.regime_state}")
        lines.append(f"Previous State:   {self.previous_regime_state or 'N/A'}")
        if self.regime_transition_occurred and self.regime_transition:
            lines.append(f"Transition:       {self.regime_transition.transition_direction.upper()}")
            lines.append(f"Threshold:        {self.regime_transition.transition_threshold:.2f}")
        lines.append(f"Holding Period:   {self.holding_period_days} days")
        lines.append("")
        
        lines.append("POSITION SCALING")
        lines.append("-" * 40)
        lines.append(f"Scale Factor:     {self.scale_factor:.2%}")
        lines.append(f"Method:           {self.scale_factor_method}")
        lines.append("")
        
        lines.append("DATA QUALITY")
        lines.append("-" * 40)
        lines.append(f"Total Indicators:    {self.total_indicators}")
        lines.append(f"Available:           {self.available_indicators}")
        lines.append(f"Imputed:             {self.imputed_indicators}")
        lines.append(f"Quality Score:       {self.data_quality:.2%}")
        if self.imputation_warning:
            lines.append("⚠️  IMPUTATION WARNING: >40% indicators imputed")
        if self.temperature_floor_applied:
            lines.append("⚠️  TEMPERATURE FLOOR APPLIED: {:.2f}".format(IMPUTATION_TEMPERATURE_FLOOR))
        lines.append("")
        
        lines.append("GAP RISK & OVERNIGHT BUDGET")
        lines.append("-" * 40)
        lines.append(f"Gap Risk Estimate:       {self.gap_risk_estimate:.2%}")
        lines.append(f"Overnight Budget Active: {'Yes' if self.overnight_budget_active else 'No'}")
        if self.overnight_max_position:
            lines.append(f"Max Overnight Position:  {self.overnight_max_position:.2%}")
        lines.append("")
        
        lines.append("CATEGORY BREAKDOWN")
        lines.append("-" * 40)
        for cat_name, cat in self.categories.items():
            lines.append(f"\n  {cat.name} (weight={cat.weight:.2f})")
            lines.append(f"    Stress Level:   {cat.stress_level:.4f}")
            lines.append(f"    Contribution:   {cat.weighted_contribution:.4f}")
            for ind in cat.indicators:
                status = "✓" if ind.data_available else ("⟳" if ind.imputed else "✗")
                lines.append(f"      {status} {ind.name}: z={ind.zscore or 0:.2f}, contrib={ind.stress_contribution:.4f}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# REGIME STATE MANAGER (Hysteresis Implementation)
# =============================================================================

@dataclass
class RegimeStateManager:
    """
    Manages regime state transitions with hysteresis bands.
    
    Implements the Chinese Professor's Solution 1: asymmetric thresholds
    that prevent oscillation at regime boundaries.
    """
    current_state: RegimeState = RegimeState.CALM
    holding_period_start: Optional[datetime] = None
    last_temperature: Optional[float] = None
    last_update: Optional[datetime] = None
    
    # Persistence path for state continuity across sessions
    persistence_path: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize and optionally load persisted state."""
        if self.persistence_path:
            self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self.persistence_path and self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                self.current_state = RegimeState.from_string(data.get("current_state", "calm"))
                if data.get("holding_period_start"):
                    self.holding_period_start = datetime.fromisoformat(data["holding_period_start"])
                self.last_temperature = data.get("last_temperature")
                if data.get("last_update"):
                    self.last_update = datetime.fromisoformat(data["last_update"])
                logger.info(f"Loaded regime state: {self.current_state.value}")
            except Exception as e:
                logger.warning(f"Failed to load regime state: {e}")
    
    def _save_state(self) -> None:
        """Persist state to disk."""
        if self.persistence_path:
            try:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "current_state": self.current_state.value,
                    "holding_period_start": self.holding_period_start.isoformat() if self.holding_period_start else None,
                    "last_temperature": self.last_temperature,
                    "last_update": self.last_update.isoformat() if self.last_update else None,
                }
                with open(self.persistence_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save regime state: {e}")
    
    def get_holding_period_days(self) -> int:
        """Get number of days in current holding period."""
        if self.holding_period_start is None:
            return 0
        delta = datetime.now() - self.holding_period_start
        return delta.days
    
    def _check_holding_period_met(self) -> bool:
        """Check if minimum holding period has been met for downward transitions."""
        return self.get_holding_period_days() >= MIN_HOLDING_PERIOD_DAYS
    
    def update(
        self,
        raw_temperature: float,
        force_override: bool = False,
    ) -> Tuple[RegimeState, Optional[RegimeTransitionRecord]]:
        """
        Update regime state based on temperature with hysteresis.
        
        Args:
            raw_temperature: The raw computed temperature (0-2 scale)
            force_override: If True, bypass holding period check (risk officer override)
            
        Returns:
            Tuple of (new_state, transition_record if transition occurred else None)
        """
        now = datetime.now()
        previous_state = self.current_state
        new_state = self.current_state
        transition_record = None
        
        # Check for upward transitions (always allowed)
        if self.current_state == RegimeState.CALM and raw_temperature > THRESHOLD_CALM_TO_ELEVATED:
            new_state = RegimeState.ELEVATED
            threshold = THRESHOLD_CALM_TO_ELEVATED
            direction = "up"
            
        elif self.current_state == RegimeState.ELEVATED and raw_temperature > THRESHOLD_ELEVATED_TO_STRESSED:
            new_state = RegimeState.STRESSED
            threshold = THRESHOLD_ELEVATED_TO_STRESSED
            direction = "up"
            
        elif self.current_state == RegimeState.STRESSED and raw_temperature > THRESHOLD_STRESSED_TO_EXTREME:
            new_state = RegimeState.EXTREME
            threshold = THRESHOLD_STRESSED_TO_EXTREME
            direction = "up"
            
        # Check for downward transitions (require holding period or override)
        elif self.current_state == RegimeState.EXTREME and raw_temperature < THRESHOLD_EXTREME_TO_STRESSED:
            holding_met = self._check_holding_period_met() or force_override
            if holding_met:
                new_state = RegimeState.STRESSED
                threshold = THRESHOLD_EXTREME_TO_STRESSED
                direction = "down"
                
        elif self.current_state == RegimeState.STRESSED and raw_temperature < THRESHOLD_STRESSED_TO_ELEVATED:
            holding_met = self._check_holding_period_met() or force_override
            if holding_met:
                new_state = RegimeState.ELEVATED
                threshold = THRESHOLD_STRESSED_TO_ELEVATED
                direction = "down"
                
        elif self.current_state == RegimeState.ELEVATED and raw_temperature < THRESHOLD_ELEVATED_TO_CALM:
            holding_met = self._check_holding_period_met() or force_override
            if holding_met:
                new_state = RegimeState.CALM
                threshold = THRESHOLD_ELEVATED_TO_CALM
                direction = "down"
        
        # Record transition if state changed
        if new_state != previous_state:
            self.current_state = new_state
            self.holding_period_start = now
            
            transition_record = RegimeTransitionRecord(
                timestamp=now.isoformat(),
                previous_state=previous_state.value,
                new_state=new_state.value,
                raw_temperature=raw_temperature,
                transition_threshold=threshold,
                transition_direction=direction,
                holding_period_met=self._check_holding_period_met() or force_override,
            )
            
            logger.info(
                f"Regime transition: {previous_state.value} → {new_state.value} "
                f"(temp={raw_temperature:.2f}, threshold={threshold:.2f})"
            )
        
        # Update tracking
        self.last_temperature = raw_temperature
        self.last_update = now
        self._save_state()
        
        return new_state, transition_record
    
    def get_scale_factor(self) -> float:
        """Get scale factor based on current regime state."""
        return REGIME_SCALE_FACTORS[self.current_state]
    
    def reset(self) -> None:
        """Reset to initial state (for testing or manual override)."""
        self.current_state = RegimeState.CALM
        self.holding_period_start = None
        self.last_temperature = None
        self.last_update = None
        self._save_state()


# =============================================================================
# CONSERVATIVE IMPUTATION MANAGER
# =============================================================================

@dataclass
class ImputationManager:
    """
    Manages conservative imputation for missing stress indicators.
    
    Implements the European Professor's Solution 5: when indicators become
    unavailable, impute at the 75th percentile of historical contribution.
    """
    # Historical stress contributions per indicator
    history: Dict[str, List[float]] = field(default_factory=dict)
    history_dates: Dict[str, List[str]] = field(default_factory=dict)
    
    # Persistence path
    persistence_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.persistence_path:
            self._load_history()
    
    def _load_history(self) -> None:
        """Load historical imputation data."""
        if self.persistence_path and self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                self.history = data.get("history", {})
                self.history_dates = data.get("history_dates", {})
            except Exception as e:
                logger.warning(f"Failed to load imputation history: {e}")
    
    def _save_history(self) -> None:
        """Persist imputation history."""
        if self.persistence_path:
            try:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "history": self.history,
                    "history_dates": self.history_dates,
                }
                with open(self.persistence_path, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save imputation history: {e}")
    
    def record_observation(
        self,
        indicator_name: str,
        stress_contribution: float,
        date: Optional[str] = None,
    ) -> None:
        """
        Record a successful stress contribution observation.
        
        This builds up the historical distribution for imputation.
        """
        if indicator_name not in self.history:
            self.history[indicator_name] = []
            self.history_dates[indicator_name] = []
        
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        
        self.history[indicator_name].append(stress_contribution)
        self.history_dates[indicator_name].append(date_str)
        
        # Trim to lookback window
        if len(self.history[indicator_name]) > IMPUTATION_HISTORY_DAYS:
            self.history[indicator_name] = self.history[indicator_name][-IMPUTATION_HISTORY_DAYS:]
            self.history_dates[indicator_name] = self.history_dates[indicator_name][-IMPUTATION_HISTORY_DAYS:]
        
        self._save_history()
    
    def get_imputed_value(self, indicator_name: str) -> Tuple[float, str]:
        """
        Get conservative imputed value for a missing indicator.
        
        Returns:
            Tuple of (imputed_value, computation_notes)
        """
        if indicator_name not in self.history or len(self.history[indicator_name]) < 10:
            # Insufficient history — use default conservative value
            return 0.5, "Default imputation (insufficient history)"
        
        values = np.array(self.history[indicator_name])
        imputed = float(np.percentile(values, IMPUTATION_PERCENTILE))
        
        notes = (
            f"Imputed at {IMPUTATION_PERCENTILE}th percentile of "
            f"{len(values)} historical observations (range: "
            f"{values.min():.3f} to {values.max():.3f})"
        )
        
        return imputed, notes
    
    def get_history_summary(self, indicator_name: str) -> Dict:
        """Get summary statistics for an indicator's history."""
        if indicator_name not in self.history or not self.history[indicator_name]:
            return {"available": False}
        
        values = np.array(self.history[indicator_name])
        return {
            "available": True,
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        }


# =============================================================================
# DYNAMIC GAP RISK ESTIMATOR
# =============================================================================

class DynamicGapRiskEstimator:
    """
    Estimates overnight gap risk dynamically from recent market data.
    
    Implements the US Engineer's Solution 4: 95th percentile of trailing
    60-day overnight gaps with floor/ceiling bounds.
    """
    
    def __init__(
        self,
        persistence_path: Optional[Path] = None,
    ):
        self.persistence_path = persistence_path
        self.previous_estimate: Optional[float] = None
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state."""
        if self.persistence_path and self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                self.previous_estimate = data.get("previous_estimate")
            except Exception as e:
                logger.warning(f"Failed to load gap risk state: {e}")
    
    def _save_state(self) -> None:
        """Persist state."""
        if self.persistence_path:
            try:
                self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.persistence_path, 'w') as f:
                    json.dump({"previous_estimate": self.previous_estimate}, f)
            except Exception as e:
                logger.warning(f"Failed to save gap risk state: {e}")
    
    def estimate_from_prices(
        self,
        prices: pd.Series,
        lookback_days: int = GAP_RISK_LOOKBACK_DAYS,
    ) -> Tuple[float, GapRiskAuditRecord]:
        """
        Estimate gap risk from price series.
        
        Args:
            prices: Daily price series (e.g., SPY)
            lookback_days: Number of days to look back
            
        Returns:
            Tuple of (final_estimate, audit_record)
        """
        now = datetime.now()
        
        if prices is None or len(prices) < lookback_days:
            # Insufficient data — return previous or default
            default = self.previous_estimate or 0.03
            audit = GapRiskAuditRecord(
                computed_at=now.isoformat(),
                lookback_start="N/A",
                lookback_end="N/A",
                raw_estimate=default,
                smoothed_estimate=default,
                applied_estimate=default,
                floor_applied=False,
                ceiling_applied=False,
                smoothing_alpha=0.0,
                previous_estimate=self.previous_estimate,
            )
            return default, audit
        
        # Compute overnight returns (close-to-close as proxy)
        returns = prices.pct_change().dropna()
        recent_returns = returns.iloc[-lookback_days:]
        
        # Compute 95th percentile of absolute returns as gap risk
        raw_estimate = float(np.percentile(np.abs(recent_returns), GAP_RISK_PERCENTILE))
        
        # Apply exponential smoothing if we have previous estimate
        if self.previous_estimate is not None:
            smoothed = (
                GAP_RISK_SMOOTHING_ALPHA * raw_estimate +
                (1 - GAP_RISK_SMOOTHING_ALPHA) * self.previous_estimate
            )
        else:
            smoothed = raw_estimate
        
        # Apply floor and ceiling
        floor_applied = smoothed < GAP_RISK_FLOOR
        ceiling_applied = smoothed > GAP_RISK_CEILING
        applied = max(GAP_RISK_FLOOR, min(GAP_RISK_CEILING, smoothed))
        
        # Create audit record
        audit = GapRiskAuditRecord(
            computed_at=now.isoformat(),
            lookback_start=str(recent_returns.index[0]) if len(recent_returns) > 0 else "N/A",
            lookback_end=str(recent_returns.index[-1]) if len(recent_returns) > 0 else "N/A",
            raw_estimate=raw_estimate,
            smoothed_estimate=smoothed,
            applied_estimate=applied,
            floor_applied=floor_applied,
            ceiling_applied=ceiling_applied,
            smoothing_alpha=GAP_RISK_SMOOTHING_ALPHA,
            previous_estimate=self.previous_estimate,
        )
        
        # Update state
        self.previous_estimate = applied
        self._save_state()
        
        return applied, audit


# =============================================================================
# RATE LIMITER (Governor)
# =============================================================================

def apply_rate_limit(
    raw_temperature: float,
    previous_temperature: Optional[float],
    max_change: float = MAX_TEMP_CHANGE_PER_DAY,
) -> Tuple[float, bool]:
    """
    Apply rate limiting (governor) to temperature changes.
    
    Prevents temperature from moving more than max_change per day.
    
    Args:
        raw_temperature: The newly computed temperature
        previous_temperature: The previous temperature (or None if first computation)
        max_change: Maximum allowed change per day
        
    Returns:
        Tuple of (rate_limited_temperature, was_limited)
    """
    if previous_temperature is None:
        return raw_temperature, False
    
    delta = raw_temperature - previous_temperature
    
    if abs(delta) <= max_change:
        return raw_temperature, False
    
    # Limit the change
    if delta > 0:
        limited = previous_temperature + max_change
    else:
        limited = previous_temperature - max_change
    
    logger.info(
        f"Rate limit applied: {previous_temperature:.3f} → {raw_temperature:.3f} "
        f"limited to {limited:.3f}"
    )
    
    return limited, True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_default_persistence_dir() -> Path:
    """Get the default directory for governance persistence files."""
    return Path(__file__).parent.parent / "data" / "governance"


def create_governance_managers(
    persistence_dir: Optional[Path] = None,
) -> Tuple[RegimeStateManager, ImputationManager, DynamicGapRiskEstimator]:
    """
    Create and initialize all governance managers.
    
    Args:
        persistence_dir: Directory for persistence files (default: src/data/governance)
        
    Returns:
        Tuple of (RegimeStateManager, ImputationManager, DynamicGapRiskEstimator)
    """
    if persistence_dir is None:
        persistence_dir = get_default_persistence_dir()
    
    persistence_dir.mkdir(parents=True, exist_ok=True)
    
    regime_manager = RegimeStateManager(
        persistence_path=persistence_dir / "regime_state.json"
    )
    
    imputation_manager = ImputationManager(
        persistence_path=persistence_dir / "imputation_history.json"
    )
    
    gap_risk_estimator = DynamicGapRiskEstimator(
        persistence_path=persistence_dir / "gap_risk_state.json"
    )
    
    return regime_manager, imputation_manager, gap_risk_estimator
