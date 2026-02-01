"""
===============================================================================
UNIT TESTS: Metals Risk Temperature Module
===============================================================================

Tests for the metals_risk_temperature.py module implementing the
Chinese Professors' Panel Hybrid Solution.

Test Coverage:
    1. Stress indicator calculations (ratio-based)
    2. Temperature normalization
    3. Scale factor computation
    4. Individual metal stress metrics
    5. Edge cases and data availability
    6. Caching behavior

===============================================================================
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class TestMetalsStressIndicators(unittest.TestCase):
    """Test ratio-based stress indicator calculations."""
    
    def setUp(self):
        """Set up test data."""
        import sys
        sys.path.insert(0, 'src')
        
        # Create synthetic price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.metals_data = {
            'GOLD': pd.Series(2000 + np.cumsum(np.random.randn(100) * 10), index=dates),
            'SILVER': pd.Series(25 + np.cumsum(np.random.randn(100) * 0.5), index=dates),
            'COPPER': pd.Series(4 + np.cumsum(np.random.randn(100) * 0.1), index=dates),
            'PLATINUM': pd.Series(1000 + np.cumsum(np.random.randn(100) * 20), index=dates),
            'PALLADIUM': pd.Series(1500 + np.cumsum(np.random.randn(100) * 30), index=dates),
        }
    
    def test_copper_gold_stress_calculation(self):
        """Test Copper/Gold ratio stress indicator."""
        from decision.metals_risk_temperature import compute_copper_gold_stress
        
        result = compute_copper_gold_stress(self.metals_data)
        
        self.assertTrue(result.data_available)
        self.assertEqual(result.name, "Copper/Gold")
        self.assertIsInstance(result.value, float)
        self.assertIsInstance(result.zscore, float)
        self.assertGreaterEqual(result.contribution, 0)
        self.assertIn(result.interpretation, [
            "Sharp risk-off (flight to gold)",
            "Mild risk-off",
            "Strong risk-on (industrial demand)",
            "Mild risk-on",
            "Neutral"
        ])
    
    def test_silver_gold_stress_calculation(self):
        """Test Silver/Gold ratio stress indicator."""
        from decision.metals_risk_temperature import compute_silver_gold_stress
        
        result = compute_silver_gold_stress(self.metals_data)
        
        self.assertTrue(result.data_available)
        self.assertEqual(result.name, "Silver/Gold")
        self.assertIsInstance(result.zscore, float)
        self.assertGreaterEqual(result.contribution, 0)
    
    def test_gold_volatility_stress_calculation(self):
        """Test gold volatility stress indicator."""
        from decision.metals_risk_temperature import compute_gold_volatility_stress
        
        result = compute_gold_volatility_stress(self.metals_data)
        
        self.assertTrue(result.data_available)
        self.assertEqual(result.name, "Gold Vol")
        self.assertGreater(result.value, 0)  # Volatility should be positive
    
    def test_precious_industrial_stress_calculation(self):
        """Test precious vs industrial spread indicator."""
        from decision.metals_risk_temperature import compute_precious_industrial_stress
        
        result = compute_precious_industrial_stress(self.metals_data)
        
        self.assertTrue(result.data_available)
        self.assertEqual(result.name, "Precious/Industrial")
    
    def test_platinum_gold_stress_calculation(self):
        """Test Platinum/Gold ratio indicator."""
        from decision.metals_risk_temperature import compute_platinum_gold_stress
        
        result = compute_platinum_gold_stress(self.metals_data)
        
        self.assertTrue(result.data_available)
        self.assertEqual(result.name, "Platinum/Gold")
    
    def test_missing_data_handling(self):
        """Test graceful handling of missing metals data."""
        from decision.metals_risk_temperature import compute_copper_gold_stress
        
        # Empty data
        result = compute_copper_gold_stress({})
        self.assertFalse(result.data_available)
        self.assertEqual(result.contribution, 0.0)
        
        # Partial data (only gold)
        partial_data = {'GOLD': self.metals_data['GOLD']}
        result = compute_copper_gold_stress(partial_data)
        self.assertFalse(result.data_available)


class TestMetalsTemperature(unittest.TestCase):
    """Test overall metals temperature computation."""
    
    def test_temperature_bounds(self):
        """Test that temperature is bounded [0, 2]."""
        from decision.metals_risk_temperature import compute_metals_risk_temperature
        
        # Mock the data fetching to return controlled data
        with patch('decision.metals_risk_temperature._fetch_metals_data') as mock_fetch:
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            mock_fetch.return_value = {
                'GOLD': pd.Series(2000 + np.random.randn(100) * 10, index=dates),
                'SILVER': pd.Series(25 + np.random.randn(100) * 0.5, index=dates),
                'COPPER': pd.Series(4 + np.random.randn(100) * 0.1, index=dates),
                'PLATINUM': pd.Series(1000 + np.random.randn(100) * 20, index=dates),
                'PALLADIUM': pd.Series(1500 + np.random.randn(100) * 30, index=dates),
            }
            
            result = compute_metals_risk_temperature()
            
            self.assertGreaterEqual(result.temperature, 0.0)
            self.assertLessEqual(result.temperature, 2.0)
    
    def test_status_labels(self):
        """Test status labels match temperature ranges."""
        from decision.metals_risk_temperature import MetalsRiskTemperatureResult
        
        # Create results with different temperatures
        for temp, expected_status in [
            (0.1, "Calm"),
            (0.5, "Elevated"),
            (1.0, "Stressed"),
            (1.8, "Extreme"),
        ]:
            result = MetalsRiskTemperatureResult(
                temperature=temp,
                scale_factor=0.5,
                indicators=[],
                metals={},
                computed_at=datetime.now().isoformat(),
                data_quality=1.0,
                status=expected_status if temp < 0.3 else (
                    "Calm" if temp < 0.3 else
                    "Elevated" if temp < 0.7 else
                    "Stressed" if temp < 1.2 else "Extreme"
                ),
                action_text="Test"
            )
            
            # Verify is_elevated, is_stressed, is_extreme properties
            if temp > 0.5:
                self.assertTrue(result.is_elevated)
            if temp > 1.0:
                self.assertTrue(result.is_stressed)
            if temp > 1.5:
                self.assertTrue(result.is_extreme)


class TestScaleFactor(unittest.TestCase):
    """Test position scale factor computation."""
    
    def test_scale_factor_range(self):
        """Test scale factor is in (0, 1)."""
        from decision.metals_risk_temperature import compute_scale_factor
        
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            scale = compute_scale_factor(temp)
            self.assertGreater(scale, 0.0)
            self.assertLess(scale, 1.0)
    
    def test_scale_factor_decreasing(self):
        """Test scale factor decreases with temperature."""
        from decision.metals_risk_temperature import compute_scale_factor
        
        prev_scale = 1.0
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            scale = compute_scale_factor(temp)
            self.assertLess(scale, prev_scale)
            prev_scale = scale
    
    def test_scale_factor_at_threshold(self):
        """Test scale factor is approximately 0.5 at threshold."""
        from decision.metals_risk_temperature import compute_scale_factor, SIGMOID_THRESHOLD
        
        scale = compute_scale_factor(SIGMOID_THRESHOLD)
        self.assertAlmostEqual(scale, 0.5, places=5)


class TestIndividualMetalStress(unittest.TestCase):
    """Test individual metal stress calculations."""
    
    def setUp(self):
        """Set up test data."""
        import sys
        sys.path.insert(0, 'src')
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.metals_data = {
            'GOLD': pd.Series(2000 + np.cumsum(np.random.randn(100) * 10), index=dates),
            'SILVER': pd.Series(25 + np.cumsum(np.random.randn(100) * 0.5), index=dates),
            'COPPER': pd.Series(4 + np.cumsum(np.random.randn(100) * 0.1), index=dates),
            'PLATINUM': pd.Series(1000 + np.cumsum(np.random.randn(100) * 20), index=dates),
            'PALLADIUM': pd.Series(1500 + np.cumsum(np.random.randn(100) * 30), index=dates),
        }
    
    def test_individual_metal_metrics(self):
        """Test individual metal stress categories."""
        from decision.metals_risk_temperature import compute_individual_metal_stress
        
        metals = compute_individual_metal_stress(self.metals_data)
        
        # Check all metals are present
        for metal in ['gold', 'silver', 'copper', 'platinum', 'palladium']:
            self.assertIn(metal, metals)
            metal_data = metals[metal]
            self.assertTrue(metal_data.data_available)
            self.assertIsNotNone(metal_data.price)
            self.assertIsInstance(metal_data.return_5d, float)
            self.assertIsInstance(metal_data.volatility, float)
            self.assertGreaterEqual(metal_data.stress_level, 0.0)
            self.assertLessEqual(metal_data.stress_level, 2.0)
    
    def test_missing_metal_handling(self):
        """Test handling of missing individual metals."""
        from decision.metals_risk_temperature import compute_individual_metal_stress
        
        # Only provide gold data
        partial_data = {'GOLD': self.metals_data['GOLD']}
        metals = compute_individual_metal_stress(partial_data)
        
        self.assertTrue(metals['gold'].data_available)
        self.assertFalse(metals['silver'].data_available)
        self.assertFalse(metals['copper'].data_available)


class TestCaching(unittest.TestCase):
    """Test caching behavior."""
    
    def test_cache_returns_same_result(self):
        """Test that cached result is returned within TTL."""
        from decision.metals_risk_temperature import (
            get_cached_metals_risk_temperature,
            clear_metals_risk_temperature_cache
        )
        
        # Clear cache first
        clear_metals_risk_temperature_cache()
        
        # First call
        result1 = get_cached_metals_risk_temperature()
        
        # Second call should return cached result
        result2 = get_cached_metals_risk_temperature()
        
        self.assertEqual(result1.computed_at, result2.computed_at)
    
    def test_cache_clear(self):
        """Test cache clearing."""
        from decision.metals_risk_temperature import (
            get_cached_metals_risk_temperature,
            clear_metals_risk_temperature_cache,
            _metals_temp_cache
        )
        
        # Populate cache
        get_cached_metals_risk_temperature()
        
        # Clear cache
        clear_metals_risk_temperature_cache()
        
        # Verify cache is empty
        from decision.metals_risk_temperature import _metals_temp_cache as cache
        self.assertEqual(len(cache), 0)


class TestDataClasses(unittest.TestCase):
    """Test data class methods."""
    
    def test_metal_stress_indicator_to_dict(self):
        """Test MetalStressIndicator.to_dict()."""
        from decision.metals_risk_temperature import MetalStressIndicator
        
        indicator = MetalStressIndicator(
            name="Test",
            value=1.5,
            zscore=0.5,
            contribution=0.3,
            data_available=True,
            interpretation="Test interpretation"
        )
        
        d = indicator.to_dict()
        
        self.assertEqual(d['name'], "Test")
        self.assertEqual(d['value'], 1.5)
        self.assertEqual(d['zscore'], 0.5)
        self.assertEqual(d['contribution'], 0.3)
        self.assertTrue(d['data_available'])
        self.assertEqual(d['interpretation'], "Test interpretation")
    
    def test_metals_result_to_dict(self):
        """Test MetalsRiskTemperatureResult.to_dict()."""
        from decision.metals_risk_temperature import (
            MetalsRiskTemperatureResult,
            MetalStressIndicator,
            MetalStressCategory
        )
        
        result = MetalsRiskTemperatureResult(
            temperature=0.8,
            scale_factor=0.6,
            indicators=[
                MetalStressIndicator("Test", 1.0, 0.5, 0.2, True, "Test")
            ],
            metals={
                'gold': MetalStressCategory("Gold", 2000.0, 0.01, 0.15, 0.5, True)
            },
            computed_at="2026-01-31T12:00:00",
            data_quality=1.0,
            status="Elevated",
            action_text="Monitor"
        )
        
        d = result.to_dict()
        
        self.assertEqual(d['temperature'], 0.8)
        self.assertEqual(d['scale_factor'], 0.6)
        self.assertEqual(d['status'], "Elevated")
        self.assertEqual(len(d['indicators']), 1)
        self.assertIn('gold', d['metals'])


class TestZScoreComputation(unittest.TestCase):
    """Test z-score computation helper."""
    
    def test_zscore_normal_data(self):
        """Test z-score with normal data."""
        from decision.metals_risk_temperature import _compute_zscore
        
        # Create data with known properties
        np.random.seed(42)
        values = pd.Series(np.random.randn(100))  # Standard normal
        
        zscore = _compute_zscore(values)
        
        # Z-score of last value should be finite
        self.assertTrue(np.isfinite(zscore))
        # Should be clipped to [-5, 5]
        self.assertGreaterEqual(zscore, -5.0)
        self.assertLessEqual(zscore, 5.0)
    
    def test_zscore_insufficient_data(self):
        """Test z-score with insufficient data."""
        from decision.metals_risk_temperature import _compute_zscore
        
        values = pd.Series([1.0, 2.0, 3.0])  # Too few points
        zscore = _compute_zscore(values, lookback=60)
        
        self.assertEqual(zscore, 0.0)
    
    def test_zscore_none_input(self):
        """Test z-score with None input."""
        from decision.metals_risk_temperature import _compute_zscore
        
        zscore = _compute_zscore(None)
        self.assertEqual(zscore, 0.0)


class TestVolatilityPercentile(unittest.TestCase):
    """Test volatility percentile computation."""
    
    def test_volatility_percentile_range(self):
        """Test volatility percentile is in [0, 1]."""
        from decision.metals_risk_temperature import _compute_volatility_percentile
        
        dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.02)), index=dates)
        
        percentile = _compute_volatility_percentile(prices)
        
        self.assertGreaterEqual(percentile, 0.0)
        self.assertLessEqual(percentile, 1.0)
    
    def test_volatility_percentile_insufficient_data(self):
        """Test volatility percentile with insufficient data."""
        from decision.metals_risk_temperature import _compute_volatility_percentile
        
        prices = pd.Series([100, 101, 102])  # Too few points
        percentile = _compute_volatility_percentile(prices)
        
        self.assertEqual(percentile, 0.5)  # Default neutral


class TestIntegrationWithMainRiskTemp(unittest.TestCase):
    """Test integration with main risk_temperature.py."""
    
    def test_metals_in_main_categories(self):
        """Test that metals appears in main risk temperature categories."""
        from decision.risk_temperature import compute_risk_temperature
        
        result = compute_risk_temperature()
        
        self.assertIn('metals', result.categories)
        metals_cat = result.categories['metals']
        self.assertEqual(metals_cat.name, "Metals_Stress")
    
    def test_metals_weight_in_aggregation(self):
        """Test that metals weight contributes to total temperature."""
        from decision.risk_temperature import METALS_STRESS_WEIGHT
        
        self.assertEqual(METALS_STRESS_WEIGHT, 0.15)
        
        # Verify all weights sum to 1.0
        from decision.risk_temperature import (
            FX_STRESS_WEIGHT,
            FUTURES_STRESS_WEIGHT,
            RATES_STRESS_WEIGHT,
            COMMODITY_STRESS_WEIGHT,
            METALS_STRESS_WEIGHT
        )
        
        total_weight = (
            FX_STRESS_WEIGHT +
            FUTURES_STRESS_WEIGHT +
            RATES_STRESS_WEIGHT +
            COMMODITY_STRESS_WEIGHT +
            METALS_STRESS_WEIGHT
        )
        
        self.assertAlmostEqual(total_weight, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
