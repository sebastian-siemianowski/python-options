#!/usr/bin/env python3
"""
test_kalman_validation.py

Tests for Kalman validation science module (Level-7 diagnostics).

Tests cover all 5 validation components:
1. Drift Reasonableness Validation
2. Predictive Likelihood Improvement
3. PIT Calibration Check
4. Stress-Regime Behavior Analysis
5. Full Validation Suite Integration
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from kalman_validation import (
    validate_drift_reasonableness,
    compare_predictive_likelihood,
    validate_pit_calibration,
    analyze_stress_regime_behavior,
    run_full_validation_suite,
    DriftReasonablenessResult,
    LikelihoodComparisonResult,
    PITCalibrationResult,
    StressRegimeResult
)


class TestKalmanValidation(unittest.TestCase):
    """Test suite for Kalman validation science."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic data for testing."""
        np.random.seed(42)
        
        # Generate synthetic price series (5 years daily data)
        n_days = 252 * 5
        dates = pd.date_range(start='2018-01-01', periods=n_days, freq='D')
        
        # Synthetic returns with drift and volatility
        drift = 0.0003  # ~7.5% annual
        vol = 0.015  # ~24% annual vol
        returns = np.random.normal(drift, vol, n_days)
        
        # Add some regime changes and stress periods
        # Crisis period: days 500-600 (high vol, negative drift)
        returns[500:600] = np.random.normal(-0.002, 0.04, 100)
        
        # Bull period: days 1000-1200 (positive drift)
        returns[1000:1200] = np.random.normal(0.001, 0.012, 200)
        
        log_px = 100 + np.cumsum(returns)
        px = np.exp(log_px)
        
        cls.px = pd.Series(px, index=dates, name='price')
        cls.ret = pd.Series(returns, index=dates, name='returns')
        
        # Synthetic Kalman filter outputs
        # mu_kf: smoothed drift (EWMA of returns)
        cls.mu_kf = cls.ret.ewm(span=63, adjust=False).mean()
        
        # var_kf: drift variance (higher during stress)
        base_var = 0.00001
        var_kf = np.full(n_days, base_var)
        var_kf[500:600] = base_var * 5.0  # Higher uncertainty during crisis
        cls.var_kf = pd.Series(var_kf, index=dates, name='drift_variance')
        
        # vol: volatility series (EWMA of squared returns)
        cls.vol = cls.ret.ewm(span=21, adjust=False).std()
    
    def test_drift_reasonableness_basic(self):
        """Test drift reasonableness validation with synthetic data."""
        result = validate_drift_reasonableness(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            asset_name="TestAsset",
            plot=False
        )
        
        # Check result type
        self.assertIsInstance(result, DriftReasonablenessResult)
        
        # Check basic fields
        self.assertEqual(result.asset_name, "TestAsset")
        self.assertGreater(result.observations, 1000)
        
        # Check smoothness ratio (drift should be smoother than returns)
        self.assertGreater(result.drift_smoothness_ratio, 0.0)
        self.assertLess(result.drift_smoothness_ratio, 1.0)
        
        # Check crisis uncertainty spike
        self.assertGreater(result.crisis_uncertainty_spike, 1.0)
        
        # Check noise tracking score
        self.assertGreaterEqual(result.noise_tracking_score, 0.0)
        self.assertLessEqual(result.noise_tracking_score, 1.0)
        
        # Check validation passed flag
        self.assertIsInstance(result.validation_passed, bool)
        
        # Check diagnostic message exists
        self.assertIsInstance(result.diagnostic_message, str)
        self.assertGreater(len(result.diagnostic_message), 0)
    
    def test_drift_reasonableness_insufficient_data(self):
        """Test drift reasonableness with insufficient data."""
        small_px = self.px.iloc[:50]
        small_ret = self.ret.iloc[:50]
        small_mu = self.mu_kf.iloc[:50]
        small_var = self.var_kf.iloc[:50]
        
        result = validate_drift_reasonableness(
            px=small_px,
            ret=small_ret,
            mu_kf=small_mu,
            var_kf=small_var,
            asset_name="SmallData"
        )
        
        self.assertFalse(result.validation_passed)
        self.assertIn("Insufficient data", result.diagnostic_message)
    
    def test_predictive_likelihood_comparison(self):
        """Test predictive likelihood improvement testing."""
        result = compare_predictive_likelihood(
            px=self.px,
            asset_name="TestAsset",
            train_days=252,
            test_days=63,
            max_windows=5
        )
        
        # Check result type
        self.assertIsInstance(result, LikelihoodComparisonResult)
        
        # Check basic fields
        self.assertEqual(result.asset_name, "TestAsset")
        self.assertGreater(result.n_test_windows, 0)
        
        # Check log-likelihood values are finite
        self.assertTrue(np.isfinite(result.ll_kalman))
        self.assertTrue(np.isfinite(result.ll_zero_drift))
        self.assertTrue(np.isfinite(result.ll_ewma_drift))
        self.assertTrue(np.isfinite(result.ll_constant_drift))
        
        # Check deltas are computed
        self.assertTrue(np.isfinite(result.delta_ll_vs_zero))
        self.assertTrue(np.isfinite(result.delta_ll_vs_ewma))
        self.assertTrue(np.isfinite(result.delta_ll_vs_constant))
        
        # Check best model is identified
        self.assertIn(result.best_model, ['Kalman', 'Zero', 'EWMA', 'Constant'])
        
        # Check improvement flag
        self.assertIsInstance(result.improvement_significant, bool)
        
        # Check diagnostic message
        self.assertIsInstance(result.diagnostic_message, str)
        self.assertGreater(len(result.diagnostic_message), 0)
    
    def test_predictive_likelihood_insufficient_data(self):
        """Test likelihood comparison with insufficient data."""
        small_px = self.px.iloc[:100]
        
        result = compare_predictive_likelihood(
            px=small_px,
            asset_name="SmallData",
            train_days=252,
            test_days=63
        )
        
        self.assertEqual(result.n_test_windows, 0)
        self.assertIn("Insufficient data", result.diagnostic_message)
    
    def test_pit_calibration_basic(self):
        """Test PIT calibration validation."""
        result = validate_pit_calibration(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            vol=self.vol,
            asset_name="TestAsset",
            plot=False
        )
        
        # Check result type
        self.assertIsInstance(result, PITCalibrationResult)
        
        # Check basic fields
        self.assertEqual(result.asset_name, "TestAsset")
        self.assertGreater(result.n_observations, 1000)
        
        # Check KS test statistics
        self.assertGreaterEqual(result.ks_statistic, 0.0)
        self.assertLessEqual(result.ks_statistic, 1.0)
        self.assertGreaterEqual(result.ks_pvalue, 0.0)
        self.assertLessEqual(result.ks_pvalue, 1.0)
        
        # Check PIT statistics
        self.assertGreater(result.pit_mean, 0.0)
        self.assertLess(result.pit_mean, 1.0)
        self.assertGreater(result.pit_std, 0.0)
        
        # Check bias flags
        self.assertIsInstance(result.underestimation_bias, bool)
        self.assertIsInstance(result.overestimation_bias, bool)
        
        # Check uniform hypothesis flag
        self.assertIsInstance(result.uniform_hypothesis_rejected, bool)
        
        # Check calibration flag
        self.assertIsInstance(result.calibration_passed, bool)
        
        # Check diagnostic message
        self.assertIsInstance(result.diagnostic_message, str)
        self.assertGreater(len(result.diagnostic_message), 0)
    
    def test_pit_calibration_insufficient_data(self):
        """Test PIT calibration with insufficient data."""
        small_px = self.px.iloc[:50]
        small_ret = self.ret.iloc[:50]
        small_mu = self.mu_kf.iloc[:50]
        small_var = self.var_kf.iloc[:50]
        small_vol = self.vol.iloc[:50]
        
        result = validate_pit_calibration(
            px=small_px,
            ret=small_ret,
            mu_kf=small_mu,
            var_kf=small_var,
            vol=small_vol,
            asset_name="SmallData"
        )
        
        self.assertFalse(result.calibration_passed)
        self.assertIn("Insufficient data", result.diagnostic_message)
    
    def test_stress_regime_behavior(self):
        """Test stress-regime behavior analysis."""
        result = analyze_stress_regime_behavior(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            vol=self.vol,
            asset_name="TestAsset",
            stress_vol_threshold=2.0
        )
        
        # Check result type
        self.assertIsInstance(result, StressRegimeResult)
        
        # Check basic fields
        self.assertEqual(result.asset_name, "TestAsset")
        self.assertGreater(result.n_periods, 1000)
        
        # Check stress periods detected
        self.assertIsInstance(result.stress_periods_detected, list)
        # Should detect at least the crisis period we injected
        self.assertGreater(len(result.stress_periods_detected), 0)
        
        # Check uncertainty metrics
        self.assertTrue(np.isfinite(result.avg_uncertainty_normal))
        self.assertTrue(np.isfinite(result.avg_uncertainty_stress))
        self.assertGreater(result.uncertainty_spike_ratio, 1.0)
        
        # Check Kelly metrics
        self.assertTrue(np.isfinite(result.avg_kelly_normal))
        self.assertTrue(np.isfinite(result.avg_kelly_stress))
        self.assertGreater(result.kelly_reduction_ratio, 0.0)
        
        # Check system backed off flag
        self.assertIsInstance(result.system_backed_off, bool)
        
        # Check diagnostic message
        self.assertIsInstance(result.diagnostic_message, str)
        self.assertGreater(len(result.diagnostic_message), 0)
    
    def test_stress_regime_insufficient_data(self):
        """Test stress regime analysis with insufficient data."""
        small_px = self.px.iloc[:200]
        small_ret = self.ret.iloc[:200]
        small_mu = self.mu_kf.iloc[:200]
        small_var = self.var_kf.iloc[:200]
        small_vol = self.vol.iloc[:200]
        
        result = analyze_stress_regime_behavior(
            px=small_px,
            ret=small_ret,
            mu_kf=small_mu,
            var_kf=small_var,
            vol=small_vol,
            asset_name="SmallData"
        )
        
        self.assertFalse(result.system_backed_off)
        self.assertIn("Insufficient data", result.diagnostic_message)
    
    def test_full_validation_suite(self):
        """Test complete validation suite integration."""
        results = run_full_validation_suite(
            px=self.px,
            asset_name="TestAsset",
            plot=False
        )
        
        # Check results is a dictionary
        self.assertIsInstance(results, dict)
        
        # Check all components are present
        self.assertIn("drift_reasonableness", results)
        self.assertIn("likelihood_comparison", results)
        self.assertIn("pit_calibration", results)
        self.assertIn("stress_regime", results)
        self.assertIn("overall_passed", results)
        self.assertIn("asset_name", results)
        
        # Check component types
        self.assertIsInstance(results["drift_reasonableness"], DriftReasonablenessResult)
        self.assertIsInstance(results["likelihood_comparison"], LikelihoodComparisonResult)
        self.assertIsInstance(results["pit_calibration"], PITCalibrationResult)
        self.assertIsInstance(results["stress_regime"], StressRegimeResult)
        
        # Check overall passed is boolean
        self.assertIsInstance(results["overall_passed"], bool)
        
        # Check asset name matches
        self.assertEqual(results["asset_name"], "TestAsset")
    
    def test_drift_smoothness_metric(self):
        """Test drift smoothness ratio calculation."""
        result = validate_drift_reasonableness(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            asset_name="SmoothTest"
        )
        
        # Drift should be much smoother than raw returns
        # (EWMA with span=63 reduces variance significantly)
        self.assertLess(result.drift_smoothness_ratio, 0.5,
                       "Drift should be at least 2x smoother than returns")
    
    def test_crisis_uncertainty_spike(self):
        """Test that uncertainty spikes during crisis periods."""
        result = validate_drift_reasonableness(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            asset_name="CrisisTest"
        )
        
        # We injected 5x uncertainty during crisis (days 500-600)
        # Should detect significant spike
        self.assertGreater(result.crisis_uncertainty_spike, 2.0,
                          "Uncertainty should spike significantly during crisis")
    
    def test_regime_break_detection(self):
        """Test regime break detection in drift."""
        result = validate_drift_reasonableness(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            asset_name="RegimeTest"
        )
        
        # Should detect regime breaks (crisis at 500-600, bull at 1000-1200)
        self.assertTrue(result.regime_break_detected,
                       "Should detect regime transitions in drift")
    
    def test_stress_period_identification(self):
        """Test identification of specific stress periods."""
        result = analyze_stress_regime_behavior(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            vol=self.vol,
            asset_name="StressTest"
        )
        
        # Should detect at least one stress period
        self.assertGreater(len(result.stress_periods_detected), 0,
                          "Should identify at least one stress period")
        
        # Each stress period should have start and end dates
        for start, end in result.stress_periods_detected:
            self.assertIsInstance(start, str)
            self.assertIsInstance(end, str)
            # Verify date format (YYYY-MM-DD)
            self.assertEqual(len(start), 10)
            self.assertEqual(len(end), 10)
    
    def test_kelly_reduction_in_stress(self):
        """Test Kelly fraction reduction during stress periods."""
        result = analyze_stress_regime_behavior(
            px=self.px,
            ret=self.ret,
            mu_kf=self.mu_kf,
            var_kf=self.var_kf,
            vol=self.vol,
            asset_name="KellyTest"
        )
        
        # Kelly should be lower in stress than normal times
        # (due to higher uncertainty and volatility)
        self.assertLess(result.avg_kelly_stress, result.avg_kelly_normal,
                       "Kelly sizing should be lower during stress periods")
    
    def test_validation_suite_comprehensive(self):
        """Test that full suite runs all components without errors."""
        try:
            results = run_full_validation_suite(
                px=self.px,
                asset_name="ComprehensiveTest",
                plot=False
            )
            
            # Verify all results have diagnostic messages
            self.assertTrue(len(results["drift_reasonableness"].diagnostic_message) > 0)
            self.assertTrue(len(results["likelihood_comparison"].diagnostic_message) > 0)
            self.assertTrue(len(results["pit_calibration"].diagnostic_message) > 0)
            self.assertTrue(len(results["stress_regime"].diagnostic_message) > 0)
            
        except Exception as e:
            self.fail(f"Full validation suite raised exception: {e}")


class TestKalmanValidationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_series(self):
        """Test handling of empty series."""
        empty_px = pd.Series([], dtype=float)
        empty_ret = pd.Series([], dtype=float)
        empty_mu = pd.Series([], dtype=float)
        empty_var = pd.Series([], dtype=float)
        empty_vol = pd.Series([], dtype=float)
        
        # Should handle empty data gracefully
        result = validate_drift_reasonableness(
            px=empty_px,
            ret=empty_ret,
            mu_kf=empty_mu,
            var_kf=empty_var,
            asset_name="Empty"
        )
        
        self.assertFalse(result.validation_passed)
        self.assertEqual(result.observations, 0)
    
    def test_constant_price_series(self):
        """Test handling of constant (no movement) price series."""
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        const_px = pd.Series(100.0, index=dates)
        const_ret = pd.Series(0.0, index=dates)
        const_mu = pd.Series(0.0, index=dates)
        const_var = pd.Series(1e-10, index=dates)
        const_vol = pd.Series(1e-10, index=dates)
        
        result = validate_drift_reasonableness(
            px=const_px,
            ret=const_ret,
            mu_kf=const_mu,
            var_kf=const_var,
            asset_name="Constant"
        )
        
        # Should complete without errors
        self.assertEqual(result.observations, 500)
        self.assertTrue(np.isfinite(result.drift_smoothness_ratio))
    
    def test_nan_handling(self):
        """Test handling of NaN values in series."""
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        px_with_nan = pd.Series(100 + np.random.randn(500).cumsum(), index=dates)
        px_with_nan.iloc[100:110] = np.nan  # Inject NaNs
        
        ret_with_nan = px_with_nan.pct_change()
        mu_with_nan = ret_with_nan.ewm(span=21).mean()
        var_with_nan = pd.Series(1e-5, index=dates)
        vol_with_nan = ret_with_nan.ewm(span=21).std()
        
        # Should drop NaNs and process remaining data
        result = validate_drift_reasonableness(
            px=px_with_nan,
            ret=ret_with_nan,
            mu_kf=mu_with_nan,
            var_kf=var_with_nan,
            asset_name="WithNaN"
        )
        
        # Should have fewer observations after dropping NaNs
        self.assertLess(result.observations, 500)
        self.assertGreater(result.observations, 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
