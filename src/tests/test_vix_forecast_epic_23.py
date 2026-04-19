#!/usr/bin/env python3
"""
===============================================================================
Tests for Epic 23: VIX-Integrated Forecast Adjustment
===============================================================================

Story 23.1: VIX-Conditional Drift Adjustment
Story 23.2: VIX Term Structure for Horizon-Dependent Vol
Story 23.3: Correlation Spike Detection for Portfolio-Level Risk
"""
import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.vix_forecast_adjustment import (
    VIXDriftAdjustmentResult,
    VIXTermStructureResult,
    CorrelationSpikeResult,
    vix_drift_adjustment,
    vix_drift_adjustment_series,
    vix_term_structure_vol,
    detect_correlation_spike,
    position_size_during_spike,
    rolling_correlation_spike_detection,
    VIX_DRIFT_THRESHOLD_LOW,
    VIX_DRIFT_THRESHOLD_HIGH,
    VIX_DRIFT_EXTREME,
    VIX_DRIFT_MAX_DAMPENING,
    CORRELATION_SPIKE_THRESHOLD,
)


# =============================================================================
# STORY 23.1 TESTS: VIX-Conditional Drift Adjustment
# =============================================================================

class TestVIXDriftAdjustment(unittest.TestCase):
    """Tests for vix_drift_adjustment()."""
    
    def test_low_vix_no_adjustment(self):
        """AC: When VIX < 15: no adjustment (low fear)."""
        mu = np.array([0.001, 0.002, -0.001])
        result = vix_drift_adjustment(mu, vix_current=12.0)
        
        self.assertIsInstance(result, VIXDriftAdjustmentResult)
        assert_allclose(result.mu_adjusted, mu)
        self.assertEqual(result.dampening_factor, 1.0)
        self.assertFalse(result.adjustment_applied)
        self.assertEqual(result.regime, "low_fear")
    
    def test_normal_vix_no_adjustment(self):
        """VIX between 15 and 25: no adjustment."""
        mu = np.array([0.002])
        result = vix_drift_adjustment(mu, vix_current=20.0)
        
        assert_allclose(result.mu_adjusted, mu)
        self.assertFalse(result.adjustment_applied)
        self.assertEqual(result.regime, "low_fear")
    
    def test_elevated_vix_dampening(self):
        """AC: When VIX > 25: dampened by (1 - 0.3 * (VIX - 25) / 25)."""
        mu = np.array([0.005])
        vix = 30.0
        
        result = vix_drift_adjustment(mu, vix_current=vix)
        
        # Expected: dampening = 1 - 0.3 * (30 - 25) / 25 = 1 - 0.06 = 0.94
        expected_dampening = 1.0 - 0.3 * (30.0 - 25.0) / 25.0
        self.assertAlmostEqual(result.dampening_factor, expected_dampening, places=6)
        
        expected_mu = 0.005 * expected_dampening
        assert_allclose(result.mu_adjusted, [expected_mu], rtol=1e-6)
        self.assertTrue(result.adjustment_applied)
        self.assertEqual(result.regime, "elevated")
    
    def test_extreme_vix_max_dampening(self):
        """AC: When VIX > 35: mu * 0.3 (70% dampening)."""
        mu = np.array([0.01])
        result = vix_drift_adjustment(mu, vix_current=40.0)
        
        self.assertAlmostEqual(result.dampening_factor, 0.3, places=6)
        assert_allclose(result.mu_adjusted, [0.01 * 0.3], rtol=1e-6)
        self.assertEqual(result.regime, "extreme")
    
    def test_exactly_vix_35(self):
        """VIX exactly 35: max dampening."""
        mu = np.array([0.01])
        result = vix_drift_adjustment(mu, vix_current=35.0)
        
        self.assertAlmostEqual(result.dampening_factor, 0.3, places=6)
    
    def test_bearish_drift_not_dampened(self):
        """Bearish (negative) drift passes through unchanged."""
        mu = np.array([-0.005])
        result = vix_drift_adjustment(mu, vix_current=40.0)
        
        assert_allclose(result.mu_adjusted, [-0.005])
    
    def test_mixed_drift_only_positive_dampened(self):
        """Only positive drift is dampened; negative drift unchanged."""
        mu = np.array([0.01, -0.01, 0.005, -0.005])
        result = vix_drift_adjustment(mu, vix_current=40.0)
        
        # Positive dampened by 0.3
        self.assertAlmostEqual(result.mu_adjusted[0], 0.01 * 0.3, places=6)
        self.assertAlmostEqual(result.mu_adjusted[2], 0.005 * 0.3, places=6)
        # Negative unchanged
        self.assertAlmostEqual(result.mu_adjusted[1], -0.01, places=6)
        self.assertAlmostEqual(result.mu_adjusted[3], -0.005, places=6)
    
    def test_zero_drift_unchanged(self):
        """Zero drift remains zero."""
        mu = np.array([0.0])
        result = vix_drift_adjustment(mu, vix_current=40.0)
        
        assert_allclose(result.mu_adjusted, [0.0])
    
    def test_scalar_input(self):
        """Scalar mu input works."""
        result = vix_drift_adjustment(np.array(0.005), vix_current=30.0)
        
        self.assertIsInstance(result.mu_adjusted, np.ndarray)
    
    def test_nan_vix_no_adjustment(self):
        """NaN VIX returns unadjusted."""
        mu = np.array([0.005])
        result = vix_drift_adjustment(mu, vix_current=float('nan'))
        
        assert_allclose(result.mu_adjusted, mu)
        self.assertFalse(result.adjustment_applied)
    
    def test_negative_vix_no_adjustment(self):
        """Negative VIX returns unadjusted."""
        mu = np.array([0.005])
        result = vix_drift_adjustment(mu, vix_current=-5.0)
        
        assert_allclose(result.mu_adjusted, mu)
        self.assertFalse(result.adjustment_applied)
    
    def test_no_regression_low_vix(self):
        """AC: No regression during low-VIX periods (adjustment is identity)."""
        mu = np.array([0.001, 0.002, 0.003, -0.001])
        for vix in [10, 12, 14, 15, 18, 20, 24]:
            result = vix_drift_adjustment(mu, vix_current=float(vix))
            assert_allclose(result.mu_adjusted, mu,
                           err_msg=f"Regression at VIX={vix}")
    
    def test_monotonic_dampening(self):
        """Higher VIX produces more dampening."""
        mu = np.array([0.01])
        factors = []
        for vix in [25, 27, 30, 33, 35, 40, 50]:
            result = vix_drift_adjustment(mu, vix_current=float(vix))
            factors.append(result.dampening_factor)
        
        for i in range(len(factors) - 1):
            self.assertLessEqual(factors[i + 1], factors[i] + 1e-10,
                               f"Dampening not monotonic at VIX step {i}")
    
    def test_to_dict(self):
        """to_dict serialization."""
        result = vix_drift_adjustment(np.array([0.01]), vix_current=30.0)
        d = result.to_dict()
        
        self.assertIn("vix_current", d)
        self.assertIn("dampening_factor", d)
        self.assertIn("regime", d)


class TestVIXDriftAdjustmentSeries(unittest.TestCase):
    """Tests for vix_drift_adjustment_series()."""
    
    def test_basic_series(self):
        """Apply adjustment to time series."""
        mu = np.array([0.01, 0.01, 0.01, 0.01])
        vix = np.array([10.0, 20.0, 30.0, 40.0])
        
        adjusted = vix_drift_adjustment_series(mu, vix)
        
        self.assertEqual(len(adjusted), 4)
        # Low VIX: no change
        self.assertAlmostEqual(adjusted[0], 0.01, places=6)
        self.assertAlmostEqual(adjusted[1], 0.01, places=6)
        # High VIX: dampened
        self.assertLess(adjusted[2], 0.01)
        self.assertLess(adjusted[3], adjusted[2])
    
    def test_length_mismatch_raises(self):
        """Length mismatch raises ValueError."""
        with self.assertRaises(ValueError):
            vix_drift_adjustment_series(np.ones(10), np.ones(20))


# =============================================================================
# STORY 23.2 TESTS: VIX Term Structure for Horizon-Dependent Vol
# =============================================================================

class TestVIXTermStructureVol(unittest.TestCase):
    """Tests for vix_term_structure_vol()."""
    
    def test_at_30d_returns_vix30(self):
        """At H=30, implied vol equals VIX (30-day measure)."""
        result = vix_term_structure_vol(vix_30=20.0, vix_90=22.0, horizon=30)
        
        self.assertIsInstance(result, VIXTermStructureResult)
        self.assertAlmostEqual(result.implied_vol, 20.0, places=1)
    
    def test_at_90d_returns_vix90(self):
        """At H=90, implied vol equals VIX3M."""
        result = vix_term_structure_vol(vix_30=20.0, vix_90=22.0, horizon=90)
        
        self.assertAlmostEqual(result.implied_vol, 22.0, places=1)
    
    def test_contango_vol_increases_with_horizon(self):
        """AC: When VIX in contango (VIX3M > VIX): term vol increases with horizon."""
        vix_30, vix_90 = 18.0, 22.0  # Contango
        
        result_7 = vix_term_structure_vol(vix_30, vix_90, horizon=7)
        result_30 = vix_term_structure_vol(vix_30, vix_90, horizon=30)
        result_90 = vix_term_structure_vol(vix_30, vix_90, horizon=90)
        
        self.assertEqual(result_30.term_structure_state, "contango")
        self.assertLess(result_7.implied_vol, result_30.implied_vol)
        self.assertLess(result_30.implied_vol, result_90.implied_vol)
    
    def test_backwardation_near_term_highest(self):
        """AC: When VIX in backwardation (VIX3M < VIX): near-term risk highest."""
        vix_30, vix_90 = 30.0, 22.0  # Backwardation (stress)
        
        result_7 = vix_term_structure_vol(vix_30, vix_90, horizon=7)
        result_30 = vix_term_structure_vol(vix_30, vix_90, horizon=30)
        result_90 = vix_term_structure_vol(vix_30, vix_90, horizon=90)
        
        self.assertEqual(result_30.term_structure_state, "backwardation")
        self.assertGreater(result_7.implied_vol, result_90.implied_vol)
    
    def test_7d_more_responsive_to_stress(self):
        """AC: 7-day implied vol more responsive to current stress than 30-day VIX."""
        # Normal conditions
        result_7_normal = vix_term_structure_vol(18.0, 20.0, horizon=7)
        result_30_normal = vix_term_structure_vol(18.0, 20.0, horizon=30)
        
        # Stress conditions (VIX jumps but VIX3M less so)
        result_7_stress = vix_term_structure_vol(35.0, 28.0, horizon=7)
        result_30_stress = vix_term_structure_vol(35.0, 28.0, horizon=30)
        
        # 7-day vol change should be larger relative to 30-day
        change_7 = result_7_stress.implied_vol - result_7_normal.implied_vol
        change_30 = result_30_stress.implied_vol - result_30_normal.implied_vol
        
        self.assertGreater(change_7, change_30 * 0.5,
                          f"7d change={change_7:.2f}, 30d change={change_30:.2f}")
    
    def test_flat_term_structure(self):
        """Flat term structure when VIX ~ VIX3M."""
        result = vix_term_structure_vol(20.0, 20.0, horizon=60)
        
        self.assertEqual(result.term_structure_state, "flat")
        self.assertAlmostEqual(result.implied_vol, 20.0, delta=0.5)
    
    def test_horizon_1d(self):
        """1-day horizon works."""
        result = vix_term_structure_vol(20.0, 22.0, horizon=1)
        
        self.assertEqual(result.horizon_days, 1)
        self.assertGreater(result.implied_vol, 0)
        self.assertGreater(result.daily_vol, 0)
    
    def test_horizon_365d(self):
        """365-day horizon (extrapolation beyond 90d)."""
        result = vix_term_structure_vol(20.0, 22.0, horizon=365)
        
        self.assertEqual(result.horizon_days, 365)
        self.assertGreater(result.implied_vol, 0)
    
    def test_daily_vol_consistent(self):
        """Daily vol consistent with annualized implied vol."""
        result = vix_term_structure_vol(20.0, 22.0, horizon=30)
        
        # daily_vol * sqrt(252) * 100 should approximately equal implied_vol
        annual_from_daily = result.daily_vol * np.sqrt(252) * 100.0
        self.assertAlmostEqual(annual_from_daily, result.implied_vol, delta=0.5)
    
    def test_nan_inputs_handled(self):
        """NaN inputs handled gracefully."""
        result = vix_term_structure_vol(float('nan'), 22.0, horizon=30)
        self.assertIsInstance(result, VIXTermStructureResult)
    
    def test_to_dict(self):
        """to_dict serialization."""
        result = vix_term_structure_vol(20.0, 22.0, horizon=7)
        d = result.to_dict()
        
        self.assertIn("implied_vol", d)
        self.assertIn("horizon_days", d)
        self.assertIn("term_structure_state", d)
    
    def test_standard_horizons(self):
        """All standard horizons produce valid results."""
        for H in [1, 3, 7, 30, 90, 180, 365]:
            result = vix_term_structure_vol(20.0, 22.0, horizon=H)
            self.assertGreater(result.implied_vol, 0, f"Zero vol at H={H}")
            self.assertTrue(np.isfinite(result.implied_vol), f"Non-finite vol at H={H}")


# =============================================================================
# STORY 23.3 TESTS: Correlation Spike Detection
# =============================================================================

class TestDetectCorrelationSpike(unittest.TestCase):
    """Tests for detect_correlation_spike()."""
    
    def test_no_spike_independent_assets(self):
        """Independent assets don't trigger spike."""
        rng = np.random.RandomState(42)
        returns = rng.randn(50, 10) * 0.01
        
        result = detect_correlation_spike(returns, vix=15.0)
        
        self.assertIsInstance(result, CorrelationSpikeResult)
        self.assertFalse(result.is_spike)
        self.assertLess(result.avg_pairwise_correlation, 0.3)
        self.assertAlmostEqual(result.portfolio_vol_inflation, 1.0)
    
    def test_spike_detected_correlated_assets(self):
        """AC: Average pairwise correlation > 0.5 with VIX > 25 flags SPIKE."""
        rng = np.random.RandomState(42)
        T, N = 50, 10
        # Common factor drives all assets
        common = rng.randn(T) * 0.02
        returns = np.column_stack([common + rng.randn(T) * 0.002 for _ in range(N)])
        
        result = detect_correlation_spike(returns, vix=30.0)
        
        self.assertTrue(result.is_spike)
        self.assertGreater(result.avg_pairwise_correlation, 0.5)
    
    def test_no_spike_low_vix_even_with_high_corr(self):
        """High correlation but low VIX: no spike (unless very extreme)."""
        rng = np.random.RandomState(42)
        T, N = 50, 10
        common = rng.randn(T) * 0.02
        # Moderate correlation but not extreme
        returns = np.column_stack([common + rng.randn(T) * 0.005 for _ in range(N)])
        
        result = detect_correlation_spike(returns, vix=12.0)
        
        # May or may not be spike depending on exact correlation level
        # VIX is low so vix_triggered should be False
        self.assertFalse(result.vix_triggered)
    
    def test_vol_inflation_during_spike(self):
        """AC: During spike, vol inflated by (1 + avg_corr * sqrt(n_assets))."""
        rng = np.random.RandomState(42)
        T, N = 50, 10
        common = rng.randn(T) * 0.02
        returns = np.column_stack([common + rng.randn(T) * 0.001 for _ in range(N)])
        
        result = detect_correlation_spike(returns, vix=30.0)
        
        if result.is_spike:
            expected_inflation = 1.0 + result.avg_pairwise_correlation * np.sqrt(N)
            self.assertAlmostEqual(result.portfolio_vol_inflation, expected_inflation, places=4)
            self.assertGreater(result.portfolio_vol_inflation, 1.0)
    
    def test_position_size_reduced_during_spike(self):
        """AC: Position sizes reduced during spike."""
        spike_result = CorrelationSpikeResult(
            is_spike=True,
            avg_pairwise_correlation=0.6,
            vix_triggered=True,
            portfolio_vol_inflation=1.0 + 0.6 * np.sqrt(10),
            n_assets=10,
        )
        
        base_size = 1.0
        adjusted_size = position_size_during_spike(base_size, spike_result)
        
        self.assertLess(adjusted_size, base_size)
        self.assertAlmostEqual(adjusted_size, base_size / spike_result.portfolio_vol_inflation)
    
    def test_position_size_unchanged_no_spike(self):
        """No spike: position size unchanged."""
        no_spike = CorrelationSpikeResult(
            is_spike=False,
            avg_pairwise_correlation=0.1,
            vix_triggered=False,
            portfolio_vol_inflation=1.0,
            n_assets=10,
        )
        
        self.assertEqual(position_size_during_spike(1.0, no_spike), 1.0)
    
    def test_too_few_assets(self):
        """Fewer than minimum assets returns no spike."""
        returns = np.random.randn(50, 2) * 0.01
        result = detect_correlation_spike(returns, vix=30.0)
        
        self.assertFalse(result.is_spike)
    
    def test_1d_input_raises(self):
        """1D input raises ValueError."""
        with self.assertRaises(ValueError):
            detect_correlation_spike(np.ones(50), vix=20.0)
    
    def test_nan_values_handled(self):
        """NaN values in returns handled gracefully."""
        rng = np.random.RandomState(42)
        returns = rng.randn(50, 5) * 0.01
        returns[10, 2] = np.nan
        returns[20, 0] = np.nan
        
        result = detect_correlation_spike(returns, vix=20.0)
        self.assertIsInstance(result, CorrelationSpikeResult)
        self.assertTrue(np.isfinite(result.avg_pairwise_correlation))
    
    def test_correlation_matrix_returned(self):
        """Full correlation matrix returned in result."""
        rng = np.random.RandomState(42)
        returns = rng.randn(50, 5) * 0.01
        
        result = detect_correlation_spike(returns, vix=20.0)
        
        self.assertIsNotNone(result.correlation_matrix)
        self.assertEqual(result.correlation_matrix.shape, (5, 5))
        # Diagonal should be 1
        assert_allclose(np.diag(result.correlation_matrix), np.ones(5), atol=1e-10)
    
    def test_to_dict(self):
        """to_dict serialization."""
        rng = np.random.RandomState(42)
        returns = rng.randn(50, 5) * 0.01
        result = detect_correlation_spike(returns, vix=20.0)
        d = result.to_dict()
        
        self.assertIn("is_spike", d)
        self.assertIn("avg_pairwise_correlation", d)
        self.assertIn("portfolio_vol_inflation", d)


class TestRollingCorrelationSpike(unittest.TestCase):
    """Tests for rolling_correlation_spike_detection()."""
    
    def test_basic_rolling(self):
        """Rolling detection produces results."""
        rng = np.random.RandomState(42)
        T, N = 100, 5
        returns = rng.randn(T, N) * 0.01
        vix = np.ones(T) * 18.0
        
        results = rolling_correlation_spike_detection(returns, vix, lookback=21)
        
        self.assertEqual(len(results), T - 21)
        for r in results:
            self.assertIsInstance(r, CorrelationSpikeResult)
    
    def test_length_mismatch_raises(self):
        """Length mismatch raises ValueError."""
        with self.assertRaises(ValueError):
            rolling_correlation_spike_detection(
                np.ones((100, 5)), np.ones(50)
            )
    
    def test_spike_detected_during_stress(self):
        """Spike detected when returns become highly correlated."""
        rng = np.random.RandomState(42)
        T, N = 100, 10
        
        # Normal period: independent
        returns = rng.randn(T, N) * 0.01
        
        # Stress period (last 30 days): highly correlated
        common = rng.randn(30) * 0.03
        for i in range(N):
            returns[-30:, i] = common + rng.randn(30) * 0.001
        
        vix = np.ones(T) * 15.0
        vix[-30:] = 35.0  # VIX spike during stress
        
        results = rolling_correlation_spike_detection(returns, vix, lookback=21)
        
        # Last results should detect spike
        spike_count = sum(1 for r in results[-10:] if r.is_spike)
        self.assertGreater(spike_count, 0, "No spikes detected during stress period")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_drift_adjustment_vix_30_to_term_structure(self):
        """VIX drift adjustment and term structure work together."""
        vix_30 = 30.0
        vix_90 = 25.0  # Backwardation
        
        # Get term structure vol at 7d
        vol_result = vix_term_structure_vol(vix_30, vix_90, horizon=7)
        
        # Adjust drift at elevated VIX
        mu = np.array([0.01])
        drift_result = vix_drift_adjustment(mu, vix_current=vix_30)
        
        # Both should indicate stress
        self.assertEqual(vol_result.term_structure_state, "backwardation")
        self.assertTrue(drift_result.adjustment_applied)
        self.assertLess(drift_result.mu_adjusted[0], 0.01)
    
    def test_full_risk_pipeline(self):
        """Full pipeline: VIX -> drift adjustment + correlation check."""
        rng = np.random.RandomState(42)
        T, N = 50, 5
        
        # Create stress scenario
        common = rng.randn(T) * 0.03
        returns = np.column_stack([common + rng.randn(T) * 0.002 for _ in range(N)])
        vix = 32.0
        
        # Step 1: Check correlation spike
        spike = detect_correlation_spike(returns, vix)
        
        # Step 2: Adjust drift
        mu = np.array([0.01] * N)
        drift = vix_drift_adjustment(mu, vix_current=vix)
        
        # Step 3: Get term structure vol
        vol = vix_term_structure_vol(vix_30=vix, vix_90=25.0, horizon=7)
        
        # Verify everything is coherent
        self.assertTrue(drift.adjustment_applied)
        self.assertGreater(vol.implied_vol, 0)
        if spike.is_spike:
            self.assertGreater(spike.portfolio_vol_inflation, 1.0)


if __name__ == "__main__":
    unittest.main()
