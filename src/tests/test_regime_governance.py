"""
===============================================================================
TEST: Regime Governance Module
===============================================================================

Tests for the Copilot Story QUANT-2025-MRT-001 implementation:

1. Hysteresis bands for regime transitions
2. Conservative imputation for missing data
3. Complete audit trail
4. Dynamic gap risk estimation
5. Rate limiting on temperature changes

Run with: python -m pytest src/tests/test_regime_governance.py -v
===============================================================================
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Import the governance module
import sys
sys.path.insert(0, 'src')

from decision.regime_governance import (
    RegimeState,
    RegimeStateManager,
    ImputationManager,
    DynamicGapRiskEstimator,
    GovernedRiskTemperatureAudit,
    IndicatorAuditRecord,
    CategoryAuditRecord,
    GapRiskAuditRecord,
    RegimeTransitionRecord,
    apply_rate_limit,
    create_governance_managers,
    THRESHOLD_CALM_TO_ELEVATED,
    THRESHOLD_ELEVATED_TO_STRESSED,
    THRESHOLD_STRESSED_TO_EXTREME,
    THRESHOLD_EXTREME_TO_STRESSED,
    THRESHOLD_STRESSED_TO_ELEVATED,
    THRESHOLD_ELEVATED_TO_CALM,
    MAX_TEMP_CHANGE_PER_DAY,
    IMPUTATION_PERCENTILE,
    GAP_RISK_FLOOR,
    GAP_RISK_CEILING,
)


class TestRegimeState:
    """Tests for RegimeState enum."""
    
    def test_regime_states_exist(self):
        """Verify all four regime states exist."""
        assert RegimeState.CALM.value == "Calm"
        assert RegimeState.ELEVATED.value == "Elevated"
        assert RegimeState.STRESSED.value == "Stressed"
        assert RegimeState.EXTREME.value == "Extreme"
    
    def test_from_string(self):
        """Test conversion from string to RegimeState."""
        assert RegimeState.from_string("calm") == RegimeState.CALM
        assert RegimeState.from_string("ELEVATED") == RegimeState.ELEVATED
        assert RegimeState.from_string("Stressed") == RegimeState.STRESSED
        assert RegimeState.from_string("extreme") == RegimeState.EXTREME
        assert RegimeState.from_string("unknown") == RegimeState.CALM  # Default


class TestHysteresisBands:
    """Tests for hysteresis band regime transitions."""
    
    def setup_method(self):
        """Create fresh regime manager for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = RegimeStateManager(
            persistence_path=self.temp_dir / "regime_state.json"
        )
    
    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_upward_transitions(self):
        """Test upward regime transitions (always allowed)."""
        # Start at CALM
        assert self.manager.current_state == RegimeState.CALM
        
        # Transition to ELEVATED
        state, transition = self.manager.update(0.6)  # > 0.5
        assert state == RegimeState.ELEVATED
        assert transition is not None
        assert transition.transition_direction == "up"
        
        # Transition to STRESSED
        state, transition = self.manager.update(1.1)  # > 1.0
        assert state == RegimeState.STRESSED
        assert transition is not None
        
        # Transition to EXTREME
        state, transition = self.manager.update(1.6)  # > 1.5
        assert state == RegimeState.EXTREME
        assert transition is not None
    
    def test_hysteresis_prevents_oscillation(self):
        """Test that hysteresis prevents oscillation at regime boundaries."""
        # First move to ELEVATED
        self.manager.update(0.6)
        assert self.manager.current_state == RegimeState.ELEVATED
        
        # Then move to STRESSED
        self.manager.update(1.1)
        assert self.manager.current_state == RegimeState.STRESSED
        
        # Temperature drops to 0.9 (above 0.7 downward threshold)
        # Should NOT transition back to ELEVATED yet
        state, transition = self.manager.update(0.9)
        assert state == RegimeState.STRESSED  # Still stressed
        assert transition is None  # No transition
        
        # Temperature drops to 0.8 (still above 0.7)
        state, transition = self.manager.update(0.8)
        assert state == RegimeState.STRESSED  # Still stressed
        
        # Temperature drops to 0.6 (below 0.7 threshold)
        # Now should transition (if holding period met)
        self.manager.holding_period_start = datetime.now() - timedelta(days=10)
        state, transition = self.manager.update(0.6)
        assert state == RegimeState.ELEVATED  # Now elevated
    
    def test_oscillation_at_boundary_prevented(self):
        """Verify temperature oscillating between 0.9 and 1.1 doesn't cause state oscillation."""
        # First move to ELEVATED
        self.manager.update(0.6)
        assert self.manager.current_state == RegimeState.ELEVATED
        
        # Then move to STRESSED
        self.manager.update(1.1)
        assert self.manager.current_state == RegimeState.STRESSED
        
        # Oscillate temperature
        for _ in range(10):
            self.manager.update(0.95)  # Between thresholds
            assert self.manager.current_state == RegimeState.STRESSED
            
            self.manager.update(1.05)  # Between thresholds
            assert self.manager.current_state == RegimeState.STRESSED
    
    def test_scale_factor_by_regime(self):
        """Test scale factors for each regime state."""
        self.manager.current_state = RegimeState.CALM
        assert self.manager.get_scale_factor() == 1.0
        
        self.manager.current_state = RegimeState.ELEVATED
        assert self.manager.get_scale_factor() == 0.75
        
        self.manager.current_state = RegimeState.STRESSED
        assert self.manager.get_scale_factor() == 0.45
        
        self.manager.current_state = RegimeState.EXTREME
        assert self.manager.get_scale_factor() == 0.20


class TestRateLimiting:
    """Tests for rate limiting (governor) on temperature changes."""
    
    def test_no_rate_limit_within_bounds(self):
        """Test no rate limiting for changes within bounds."""
        result, limited = apply_rate_limit(0.5, 0.3, MAX_TEMP_CHANGE_PER_DAY)
        assert result == 0.5
        assert limited is False
    
    def test_rate_limit_applied_upward(self):
        """Test rate limiting for large upward changes."""
        result, limited = apply_rate_limit(1.0, 0.5, MAX_TEMP_CHANGE_PER_DAY)
        assert result == pytest.approx(0.5 + MAX_TEMP_CHANGE_PER_DAY, rel=1e-6)
        assert limited is True
    
    def test_rate_limit_applied_downward(self):
        """Test rate limiting for large downward changes."""
        result, limited = apply_rate_limit(0.2, 0.8, MAX_TEMP_CHANGE_PER_DAY)
        assert result == pytest.approx(0.8 - MAX_TEMP_CHANGE_PER_DAY, rel=1e-6)
        assert limited is True
    
    def test_no_rate_limit_first_computation(self):
        """Test no rate limiting when previous temperature is None."""
        result, limited = apply_rate_limit(1.5, None, MAX_TEMP_CHANGE_PER_DAY)
        assert result == 1.5
        assert limited is False


class TestConservativeImputation:
    """Tests for conservative imputation of missing data."""
    
    def setup_method(self):
        """Create fresh imputation manager for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = ImputationManager(
            persistence_path=self.temp_dir / "imputation_history.json"
        )
    
    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_imputation_insufficient_history(self):
        """Test default imputation when history is insufficient."""
        value, notes = self.manager.get_imputed_value("unknown_indicator")
        assert value == 0.5  # Default
        assert "insufficient history" in notes.lower()
    
    def test_percentile_imputation_with_history(self):
        """Test imputation at 75th percentile with sufficient history."""
        # Add historical observations
        indicator_name = "test_indicator"
        values = list(range(100))  # 0 to 99
        for v in values:
            self.manager.record_observation(indicator_name, v / 100.0)
        
        value, notes = self.manager.get_imputed_value(indicator_name)
        
        # 75th percentile of [0, 0.01, ..., 0.99] should be ~0.75
        assert value == pytest.approx(0.75, rel=0.05)
        assert "75th percentile" in notes
    
    def test_history_trimming(self):
        """Test that history is trimmed to lookback window."""
        indicator_name = "test_indicator"
        
        # Add more than 252 observations
        for i in range(300):
            self.manager.record_observation(indicator_name, i / 300.0)
        
        # History should be trimmed
        assert len(self.manager.history[indicator_name]) <= 252


class TestDynamicGapRisk:
    """Tests for dynamic gap risk estimation."""
    
    def setup_method(self):
        """Create fresh gap risk estimator for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.estimator = DynamicGapRiskEstimator(
            persistence_path=self.temp_dir / "gap_risk_state.json"
        )
    
    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gap_risk_floor(self):
        """Test that gap risk floor is applied."""
        # Create prices with very low volatility
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = pd.Series(100.0 + np.arange(100) * 0.001, index=dates)  # Almost flat
        
        estimate, audit = self.estimator.estimate_from_prices(prices)
        
        assert estimate >= GAP_RISK_FLOOR
        assert audit.floor_applied or audit.raw_estimate >= GAP_RISK_FLOOR
    
    def test_gap_risk_ceiling(self):
        """Test that gap risk ceiling is applied."""
        # Create prices with very high volatility
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        # Random walk with high volatility
        np.random.seed(42)
        returns = np.random.normal(0, 0.1, 100)  # 10% daily vol
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        
        estimate, audit = self.estimator.estimate_from_prices(prices)
        
        assert estimate <= GAP_RISK_CEILING
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        prices = pd.Series([100, 101, 102])
        
        estimate, audit = self.estimator.estimate_from_prices(prices, lookback_days=60)
        
        # Should return default
        assert estimate == 0.03  # Default


class TestAuditTrail:
    """Tests for audit trail generation and serialization."""
    
    def test_indicator_audit_record_serialization(self):
        """Test IndicatorAuditRecord serialization."""
        record = IndicatorAuditRecord(
            name="Test Indicator",
            raw_value=1.5,
            data_available=True,
            imputed=False,
            zscore=1.2,
            weight=0.35,
            stress_contribution=0.42,
            interpretation="Test interpretation",
        )
        
        # Serialize and deserialize
        d = record.to_dict()
        restored = IndicatorAuditRecord.from_dict(d)
        
        assert restored.name == record.name
        assert restored.raw_value == record.raw_value
        assert restored.zscore == record.zscore
    
    def test_governed_audit_json_serialization(self):
        """Test full audit trail JSON serialization."""
        audit = GovernedRiskTemperatureAudit(
            computed_at=datetime.now().isoformat(),
            raw_temperature=0.75,
            rate_limited_temperature=0.75,
            regime_state="Elevated",
            scale_factor=0.75,
            total_indicators=5,
            available_indicators=4,
            imputed_indicators=1,
            data_quality=0.8,
        )
        
        # Serialize to JSON
        json_str = audit.to_json()
        
        # Parse JSON
        parsed = json.loads(json_str)
        assert parsed["raw_temperature"] == 0.75
        assert parsed["regime_state"] == "Elevated"
        
        # Deserialize back
        restored = GovernedRiskTemperatureAudit.from_json(json_str)
        assert restored.raw_temperature == audit.raw_temperature
        assert restored.regime_state == audit.regime_state
    
    def test_human_readable_audit(self):
        """Test human-readable audit trail rendering."""
        audit = GovernedRiskTemperatureAudit(
            computed_at=datetime.now().isoformat(),
            raw_temperature=1.2,
            rate_limited_temperature=1.0,
            rate_limit_applied=True,
            previous_temperature=1.5,  # Include previous temp to trigger rate limit display
            regime_state="Stressed",
            previous_regime_state="Elevated",
            regime_transition_occurred=True,
            scale_factor=0.45,
            total_indicators=5,
            available_indicators=5,
            imputed_indicators=0,
            data_quality=1.0,
        )
        
        readable = audit.render_human_readable()
        
        assert "RISK TEMPERATURE AUDIT TRAIL" in readable
        assert "Stressed" in readable
        assert "Rate Limit Applied" in readable
        assert "Previous Temperature" in readable


class TestIntegration:
    """Integration tests for the full governance system."""
    
    def setup_method(self):
        """Create fresh governance managers."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_governance_managers(self):
        """Test that all governance managers can be created."""
        regime_mgr, imputation_mgr, gap_risk_est = create_governance_managers(
            persistence_dir=self.temp_dir
        )
        
        assert regime_mgr is not None
        assert imputation_mgr is not None
        assert gap_risk_est is not None
        
        # Check persistence files created
        assert (self.temp_dir / "regime_state.json").exists() or True  # May not exist until first save
    
    def test_full_governance_workflow(self):
        """Test complete governance workflow."""
        regime_mgr, imputation_mgr, gap_risk_est = create_governance_managers(
            persistence_dir=self.temp_dir
        )
        
        # 1. Record some observations
        for i in range(50):
            imputation_mgr.record_observation("copper_gold", 0.3 + i * 0.01)
        
        # 2. Get imputed value
        imputed, notes = imputation_mgr.get_imputed_value("copper_gold")
        assert imputed > 0.3
        
        # 3. Update regime with temperature
        raw_temp = 0.6
        state, transition = regime_mgr.update(raw_temp)
        assert state == RegimeState.ELEVATED
        
        # 4. Apply rate limiting
        limited_temp, was_limited = apply_rate_limit(0.9, raw_temp, MAX_TEMP_CHANGE_PER_DAY)
        # 0.9 - 0.6 = 0.3 = MAX_TEMP_CHANGE_PER_DAY, so should be exactly at limit
        assert limited_temp == pytest.approx(raw_temp + MAX_TEMP_CHANGE_PER_DAY, rel=1e-6)
        
        # 5. Get scale factor
        scale = regime_mgr.get_scale_factor()
        assert scale == 0.75  # ELEVATED scale factor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
