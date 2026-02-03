#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
TEST: COPULA-BASED CORRELATION STRESS
═══════════════════════════════════════════════════════════════════════════════

Unit tests for the copula correlation module.

Tests:
1. Clayton copula parameter estimation
2. Gumbel copula parameter estimation
3. Tail dependence coefficient computation
4. Smooth scale factor with hysteresis
5. Unified risk context computation
6. Integration with market temperature

═══════════════════════════════════════════════════════════════════════════════
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to path
SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from calibration.copula_correlation import (
    _to_pseudo_observations,
    _clayton_log_likelihood,
    _gumbel_log_likelihood,
    _estimate_clayton_theta,
    _estimate_gumbel_theta,
    _clayton_tail_dependence,
    _gumbel_tail_dependence,
    estimate_copula_pair,
    compute_copula_correlation_stress,
    compute_smooth_scale_factor,
    compute_unified_risk_context,
    reset_scale_factor_state,
    CopulaEstimate,
    CopulaCorrelationStress,
    UnifiedRiskContext,
    COPULA_MIN_OBSERVATIONS,
    TAIL_DEPENDENCE_HIGH,
    SYSTEMIC_RISK_THRESHOLD,
)


class TestPseudoObservations(unittest.TestCase):
    """Test pseudo-observation transformation."""
    
    def test_basic_transform(self):
        """Test basic rank transformation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = _to_pseudo_observations(x)
        
        # Should be in (0, 1)
        self.assertTrue(np.all(u > 0))
        self.assertTrue(np.all(u < 1))
        
        # Should be monotonically increasing (same order as original)
        self.assertTrue(np.all(np.diff(u) > 0))
    
    def test_handles_ties(self):
        """Test handling of tied values."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        u = _to_pseudo_observations(x)
        
        # Should still be in valid range
        self.assertTrue(np.all(u > 0))
        self.assertTrue(np.all(u < 1))


class TestClaytonCopula(unittest.TestCase):
    """Test Clayton copula estimation."""
    
    def test_positive_dependence(self):
        """Test estimation with positively dependent data."""
        np.random.seed(42)
        
        # Generate correlated data (simulated positive dependence)
        n = 200
        z = np.random.randn(n)
        x = z + np.random.randn(n) * 0.3
        y = z + np.random.randn(n) * 0.3
        
        u = _to_pseudo_observations(x)
        v = _to_pseudo_observations(y)
        
        theta, converged = _estimate_clayton_theta(u, v)
        
        # Should estimate positive dependence
        self.assertGreater(theta, 0.1)
        
        # Tail dependence should be meaningful
        lambda_L = _clayton_tail_dependence(theta)
        self.assertGreater(lambda_L, 0.0)
        self.assertLess(lambda_L, 1.0)
    
    def test_tail_dependence_limits(self):
        """Test tail dependence coefficient limits."""
        # Very low theta -> low dependence
        self.assertLess(_clayton_tail_dependence(0.01), 0.05)
        
        # High theta -> high dependence
        self.assertGreater(_clayton_tail_dependence(10.0), 0.9)
        
        # Zero/negative theta -> zero dependence
        self.assertEqual(_clayton_tail_dependence(0), 0.0)
        self.assertEqual(_clayton_tail_dependence(-1), 0.0)


class TestGumbelCopula(unittest.TestCase):
    """Test Gumbel copula estimation."""
    
    def test_upper_tail_dependence(self):
        """Test upper-tail dependence estimation."""
        # Gumbel theta = 1 -> independence
        self.assertEqual(_gumbel_tail_dependence(1.0), 0.0)
        
        # Higher theta -> more upper-tail dependence
        self.assertGreater(_gumbel_tail_dependence(2.0), 0.0)
        self.assertLess(_gumbel_tail_dependence(2.0), 1.0)
    
    def test_invalid_theta(self):
        """Test handling of invalid theta."""
        self.assertEqual(_gumbel_tail_dependence(0.5), 0.0)


class TestCopulaPairEstimation(unittest.TestCase):
    """Test full copula pair estimation."""
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        r1 = np.random.randn(30)  # Less than COPULA_MIN_OBSERVATIONS
        r2 = np.random.randn(30)
        
        result = estimate_copula_pair(r1, r2)
        
        self.assertIsInstance(result, CopulaEstimate)
        self.assertLess(result.estimation_quality, 1.0)
    
    def test_sufficient_data(self):
        """Test estimation with sufficient data."""
        np.random.seed(42)
        
        n = 100
        z = np.random.randn(n)
        r1 = z + np.random.randn(n) * 0.2
        r2 = z + np.random.randn(n) * 0.2
        
        result = estimate_copula_pair(r1, r2, "AAPL", "MSFT")
        
        self.assertIsInstance(result, CopulaEstimate)
        self.assertEqual(result.asset1, "AAPL")
        self.assertEqual(result.asset2, "MSFT")
        self.assertGreater(result.lower_tail_dependence, 0.0)
        self.assertGreater(result.estimation_quality, 0.5)


class TestCorrelationStress(unittest.TestCase):
    """Test full correlation stress computation."""
    
    def test_insufficient_stocks(self):
        """Test handling of too few stocks."""
        stock_data = {
            "AAPL": pd.Series(np.random.randn(100).cumsum()),
            "MSFT": pd.Series(np.random.randn(100).cumsum()),
        }
        
        result = compute_copula_correlation_stress(stock_data)
        
        self.assertIsInstance(result, CopulaCorrelationStress)
        # Should handle gracefully
    
    def test_correlated_stocks(self):
        """Test with correlated stocks."""
        np.random.seed(42)
        
        n = 120
        z = np.random.randn(n)
        
        stock_data = {
            "AAPL": pd.Series(1000 * np.exp((z + np.random.randn(n) * 0.1).cumsum() * 0.01)),
            "MSFT": pd.Series(300 * np.exp((z + np.random.randn(n) * 0.1).cumsum() * 0.01)),
            "GOOG": pd.Series(150 * np.exp((z + np.random.randn(n) * 0.1).cumsum() * 0.01)),
            "AMZN": pd.Series(180 * np.exp((z + np.random.randn(n) * 0.1).cumsum() * 0.01)),
            "META": pd.Series(400 * np.exp((z + np.random.randn(n) * 0.1).cumsum() * 0.01)),
        }
        
        result = compute_copula_correlation_stress(stock_data)
        
        self.assertIsInstance(result, CopulaCorrelationStress)
        self.assertGreater(result.avg_correlation, 0.5)  # Should be correlated
        self.assertGreater(result.lower_tail_dependence_avg, 0.0)
        self.assertEqual(result.method, "copula")


class TestSmoothScaleFactor(unittest.TestCase):
    """Test smooth exponential scale factor."""
    
    def setUp(self):
        reset_scale_factor_state()
    
    def test_below_threshold(self):
        """Test scale factor below threshold."""
        scale = compute_smooth_scale_factor(0.5, threshold=1.0)
        self.assertEqual(scale, 1.0)  # Should be 1.0 below threshold
    
    def test_above_threshold(self):
        """Test scale factor above threshold."""
        scale = compute_smooth_scale_factor(1.5, threshold=1.0)
        self.assertLess(scale, 1.0)
        self.assertGreater(scale, 0.05)  # Should be reduced but not zero
    
    def test_smooth_transition(self):
        """Test smooth transition around threshold."""
        temps = np.linspace(0.5, 2.0, 20)
        scales = [compute_smooth_scale_factor(t, state_key=f"test_{i}") for i, t in enumerate(temps)]
        
        # Should be monotonically decreasing
        for i in range(1, len(scales)):
            self.assertLessEqual(scales[i], scales[i-1] + 0.01)  # Allow small tolerance
    
    def test_minimum_scale(self):
        """Test minimum scale factor."""
        scale = compute_smooth_scale_factor(5.0, threshold=1.0)
        self.assertGreaterEqual(scale, 0.05)  # Should never go below 5%


class TestUnifiedRiskContext(unittest.TestCase):
    """Test unified risk context computation."""
    
    def test_empty_context(self):
        """Test with no inputs."""
        context = compute_unified_risk_context()
        
        self.assertIsInstance(context, UnifiedRiskContext)
        self.assertEqual(context.combined_temperature, 0.0)
        self.assertEqual(context.combined_scale_factor, 1.0)
    
    def test_combined_temperature(self):
        """Test combined temperature weighting."""
        # Create mock results
        class MockResult:
            def __init__(self, temp, scale):
                self.temperature = temp
                self.scale_factor = scale
                self.data_quality = 1.0
        
        risk = MockResult(1.0, 0.5)
        metals = MockResult(0.5, 0.8)
        market = MockResult(0.8, 0.6)
        
        context = compute_unified_risk_context(
            risk_temp_result=risk,
            metals_temp_result=metals,
            market_temp_result=market,
        )
        
        # Combined temperature should be weighted average
        expected = 0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 0.8
        self.assertAlmostEqual(context.combined_temperature, expected, places=2)
        
        # Combined scale should be minimum
        self.assertEqual(context.combined_scale_factor, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
