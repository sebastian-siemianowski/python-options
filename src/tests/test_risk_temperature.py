#!/usr/bin/env python3
"""
Test Risk Temperature Modulation Layer

Tests the Expert Panel's recommended Solution 1 + Solution 4:
Risk Temperature Layer with Overnight Budget Constraint.

Key properties verified:
1. Sigmoid scaling produces correct values at key points
2. Temperature aggregation follows weighted sum formula
3. Overnight budget activates when temperature > 1.0
4. No feedback into inference layer (just modulates output)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import math


class TestScalingFunction:
    """Test the sigmoid scaling function."""
    
    def test_scale_factor_at_zero(self):
        """Temperature = 0 should give scale ≈ 0.95."""
        from decision.risk_temperature import compute_scale_factor
        
        scale = compute_scale_factor(0.0)
        assert 0.94 < scale < 0.96, f"Expected ~0.95, got {scale}"
    
    def test_scale_factor_at_threshold(self):
        """Temperature = 1.0 (threshold) should give scale = 0.5."""
        from decision.risk_temperature import compute_scale_factor
        
        scale = compute_scale_factor(1.0)
        assert 0.49 < scale < 0.51, f"Expected 0.50, got {scale}"
    
    def test_scale_factor_at_max(self):
        """Temperature = 2.0 should give scale ≈ 0.05."""
        from decision.risk_temperature import compute_scale_factor
        
        scale = compute_scale_factor(2.0)
        assert 0.04 < scale < 0.06, f"Expected ~0.05, got {scale}"
    
    def test_scale_factor_monotonic(self):
        """Higher temperature should give lower scale."""
        from decision.risk_temperature import compute_scale_factor
        
        temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        scales = [compute_scale_factor(t) for t in temps]
        
        for i in range(len(scales) - 1):
            assert scales[i] > scales[i + 1], \
                f"Scale should decrease: {scales[i]} > {scales[i+1]} at temps {temps[i]}, {temps[i+1]}"
    
    def test_scale_factor_bounds(self):
        """Scale should always be in (0, 1)."""
        from decision.risk_temperature import compute_scale_factor
        
        for temp in np.linspace(-1, 5, 100):
            scale = compute_scale_factor(temp)
            assert 0 < scale < 1, f"Scale {scale} out of bounds for temp {temp}"


class TestOvernightBudget:
    """Test overnight budget constraint."""
    
    def test_budget_inactive_below_threshold(self):
        """Budget should not apply when temp <= 1.0."""
        from decision.risk_temperature import (
            compute_overnight_budget,
            OVERNIGHT_BUDGET_ACTIVATION_TEMP,
        )
        
        for temp in [0.0, 0.5, 0.9, 1.0]:
            budget = compute_overnight_budget(
                temperature=temp,
                notional=1_000_000,
                estimated_gap_risk=0.03,
            )
            assert budget is None, f"Budget should be None for temp={temp}, got {budget}"
    
    def test_budget_active_above_threshold(self):
        """Budget should apply when temp > 1.0."""
        from decision.risk_temperature import compute_overnight_budget
        
        for temp in [1.1, 1.5, 2.0]:
            budget = compute_overnight_budget(
                temperature=temp,
                notional=1_000_000,
                estimated_gap_risk=0.03,
            )
            assert budget is not None, f"Budget should be set for temp={temp}"
            assert budget > 0, f"Budget should be positive, got {budget}"
    
    def test_budget_decreases_with_temperature(self):
        """Higher temperature should give lower budget."""
        from decision.risk_temperature import compute_overnight_budget
        
        budgets = []
        for temp in [1.1, 1.3, 1.5, 1.8, 2.0]:
            budget = compute_overnight_budget(
                temperature=temp,
                notional=1_000_000,
                estimated_gap_risk=0.03,
            )
            budgets.append(budget)
        
        for i in range(len(budgets) - 1):
            assert budgets[i] > budgets[i + 1], \
                f"Budget should decrease: {budgets[i]} > {budgets[i+1]}"


class TestStressIndicator:
    """Test individual stress indicator."""
    
    def test_indicator_dataclass(self):
        """Test StressIndicator creation and serialization."""
        from decision.risk_temperature import StressIndicator
        
        indicator = StressIndicator(
            name="AUDJPY_5d_return",
            value=-0.02,
            zscore=-1.5,
            contribution=2.25,
            data_available=True,
        )
        
        assert indicator.name == "AUDJPY_5d_return"
        assert indicator.data_available
        
        d = indicator.to_dict()
        assert d["zscore"] == -1.5
        assert d["contribution"] == 2.25


class TestStressCategory:
    """Test stress category aggregation."""
    
    def test_category_dataclass(self):
        """Test StressCategory creation."""
        from decision.risk_temperature import StressCategory, StressIndicator
        
        indicators = [
            StressIndicator("ind1", 0.0, 1.5, 1.5, True),
            StressIndicator("ind2", 0.0, 0.5, 0.5, True),
        ]
        
        category = StressCategory(
            name="FX_Stress",
            weight=0.4,
            indicators=indicators,
            stress_level=1.5,  # Max contribution
            weighted_contribution=0.6,  # 0.4 × 1.5
        )
        
        assert category.name == "FX_Stress"
        assert category.weight == 0.4
        assert category.stress_level == 1.5
        assert category.weighted_contribution == 0.6


class TestRiskTemperatureResult:
    """Test complete risk temperature result."""
    
    def test_result_properties(self):
        """Test result state classification properties."""
        from decision.risk_temperature import (
            RiskTemperatureResult,
            StressCategory,
        )
        
        # Create minimal result
        result = RiskTemperatureResult(
            temperature=1.2,
            scale_factor=0.35,
            categories={},
            overnight_budget_active=True,
            overnight_max_position=0.5,
            computed_at="2026-01-31T12:00:00",
            data_quality=0.8,
        )
        
        assert result.is_elevated  # temp > 0.5
        assert result.is_stressed  # temp > 1.0
        assert not result.is_crisis  # temp <= 1.5
    
    def test_crisis_classification(self):
        """Test crisis temperature classification."""
        from decision.risk_temperature import RiskTemperatureResult
        
        result = RiskTemperatureResult(
            temperature=1.7,
            scale_factor=0.12,
            categories={},
            overnight_budget_active=True,
            overnight_max_position=0.2,
            computed_at="2026-01-31T12:00:00",
            data_quality=0.8,
        )
        
        assert result.is_crisis  # temp > 1.5


class TestApplyScaling:
    """Test the apply_risk_temperature_scaling function."""
    
    def test_scaling_reduces_position(self):
        """Scaling should reduce position when temp > 0."""
        from decision.risk_temperature import (
            apply_risk_temperature_scaling,
            RiskTemperatureResult,
        )
        
        result = RiskTemperatureResult(
            temperature=1.0,
            scale_factor=0.5,
            categories={},
            overnight_budget_active=False,
            overnight_max_position=None,
            computed_at="2026-01-31T12:00:00",
            data_quality=0.8,
        )
        
        original_pos = 1.0
        scaled_pos, meta = apply_risk_temperature_scaling(original_pos, result)
        
        assert scaled_pos == 0.5
        assert meta["original_pos_strength"] == 1.0
        assert meta["scaled_pos_strength"] == 0.5
        assert not meta["overnight_budget_applied"]
    
    def test_overnight_budget_constraint(self):
        """Overnight budget should cap position when active and binding."""
        from decision.risk_temperature import (
            apply_risk_temperature_scaling,
            RiskTemperatureResult,
        )
        
        result = RiskTemperatureResult(
            temperature=1.5,
            scale_factor=0.18,
            categories={},
            overnight_budget_active=True,
            overnight_max_position=0.1,  # Cap at 0.1
            computed_at="2026-01-31T12:00:00",
            data_quality=0.8,
        )
        
        original_pos = 1.0
        scaled_pos, meta = apply_risk_temperature_scaling(original_pos, result)
        
        # Position would be 1.0 × 0.18 = 0.18, but capped at 0.1
        assert scaled_pos == 0.1
        assert meta["overnight_budget_applied"]


class TestZScore:
    """Test z-score computation."""
    
    def test_zscore_basic(self):
        """Test basic z-score calculation."""
        from decision.risk_temperature import _compute_zscore
        import pandas as pd
        
        # Create series with known mean=0, std=1
        np.random.seed(42)
        values = pd.Series(np.random.randn(100))
        
        # Last value is the one being z-scored
        zscore = _compute_zscore(values)
        
        # Should be within reasonable range
        assert -4 < zscore < 4
    
    def test_zscore_insufficient_data(self):
        """Z-score should return 0 for insufficient data."""
        from decision.risk_temperature import _compute_zscore
        import pandas as pd
        
        values = pd.Series([1, 2, 3])  # Too short
        zscore = _compute_zscore(values, lookback=60)
        
        assert zscore == 0.0


class TestCaching:
    """Test risk temperature caching."""
    
    def test_cache_clear(self):
        """Test cache can be cleared."""
        from decision.risk_temperature import (
            _risk_temp_cache,
            clear_risk_temperature_cache,
        )
        
        # Clear should not raise
        clear_risk_temperature_cache()
        
        # Cache should be empty
        from decision.risk_temperature import _risk_temp_cache
        assert len(_risk_temp_cache) == 0


class TestNoFeedback:
    """Test that risk temperature doesn't affect inference layer."""
    
    def test_independent_of_kalman(self):
        """Risk temperature should not modify Kalman state."""
        # This is a conceptual test - risk_temperature.py has no imports from
        # tuning or Kalman modules, ensuring no feedback loops
        from decision.risk_temperature import compute_risk_temperature
        
        # Verify no tuning imports
        import decision.risk_temperature as rt
        module_contents = dir(rt)
        
        # Should not have Kalman-related attributes
        forbidden_terms = ['kalman', 'tune', 'bma', 'garch', 'posterior']
        for term in forbidden_terms:
            matching = [m for m in module_contents if term.lower() in m.lower()]
            assert len(matching) == 0, \
                f"Found forbidden term '{term}' in risk_temperature: {matching}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
