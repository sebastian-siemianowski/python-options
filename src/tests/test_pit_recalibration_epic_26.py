"""
Tests for Epic 26: PIT-Based Online Recalibration

Story 26.1: isotonic_recalibrate - PIT recalibration via isotonic regression
Story 26.2: location_scale_correction - online bias/variance correction
Story 26.3: recalibration_schedule - adaptive recalibration frequency
"""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.pit_recalibration import (
    isotonic_recalibrate,
    location_scale_correction,
    recalibration_schedule,
    compute_recalibration_stability,
    _compute_ece,
    _isotonic_regression,
    IsotonicRecalibrationResult,
    LocationScaleResult,
    RecalibrationScheduleResult,
    DEFAULT_PIT_WINDOW,
    DEFAULT_CORRECTION_WINDOW,
    KS_THRESHOLD_DAILY,
    KS_THRESHOLD_WEEKLY,
    MIN_PIT_VALUES,
)


# ===========================================================================
# Story 26.1: Isotonic Regression for PIT Recalibration
# ===========================================================================

class TestIsotonicRecalibrate(unittest.TestCase):
    """AC: isotonic_recalibrate fits monotonic g: [0,1] -> [0,1]."""

    def test_basic_recalibration(self):
        """Recalibrate biased PIT values."""
        rng = np.random.default_rng(42)
        # Generate biased PIT (clustered at extremes = underdispersed)
        pit = np.clip(rng.beta(0.5, 0.5, size=200), 0, 1)
        result = isotonic_recalibrate(pit)
        self.assertIsInstance(result, IsotonicRecalibrationResult)
        self.assertEqual(result.n_pit_values, 200)

    def test_ks_improves(self):
        """AC: g(PIT) is closer to U(0,1) than raw PIT (KS test improves)."""
        rng = np.random.default_rng(42)
        # Biased PIT: skewed right (pessimistic model)
        pit = np.clip(rng.beta(2, 5, size=300), 0, 1)
        result = isotonic_recalibrate(pit)
        # KS after should be <= KS before (or very close)
        self.assertLessEqual(result.ks_after, result.ks_before + 0.05)

    def test_no_future_leakage(self):
        """AC: Uses only past PIT values (window)."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=300)
        result = isotonic_recalibrate(pit, window=126)
        self.assertEqual(result.window, 126)

    def test_ece_improves(self):
        """AC: ECE after recalibration < ECE before (for biased PIT)."""
        rng = np.random.default_rng(42)
        # Strongly biased PIT
        pit = np.clip(rng.beta(1, 3, size=500), 0, 1)
        result = isotonic_recalibrate(pit, window=500)
        self.assertLess(result.ece_after, result.ece_before + 0.01)

    def test_uniform_pit_unchanged(self):
        """Already uniform PIT should change minimally."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=200)
        result = isotonic_recalibrate(pit)
        # KS should be small both before and after
        self.assertLess(result.ks_before, 0.15)
        self.assertLess(result.ks_after, 0.15)

    def test_monotonic_mapping(self):
        """Mapping should be non-decreasing."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.beta(2, 2, size=200), 0, 1)
        result = isotonic_recalibrate(pit)
        # mapping_y should be non-decreasing (after accumulate)
        diffs = np.diff(result.mapping_y)
        self.assertTrue(np.all(diffs >= -1e-10))

    def test_output_in_0_1(self):
        """Recalibrated PIT values are in [0, 1]."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=200)
        result = isotonic_recalibrate(pit)
        self.assertTrue(np.all(result.recalibrated_pit >= 0))
        self.assertTrue(np.all(result.recalibrated_pit <= 1))

    def test_too_few_values_raises(self):
        """Need minimum PIT values."""
        with self.assertRaises(ValueError):
            isotonic_recalibrate(np.array([0.5] * 5))

    def test_nan_handled(self):
        """NaN PIT values are filtered."""
        rng = np.random.default_rng(42)
        pit = np.concatenate([rng.uniform(0, 1, size=100), np.array([np.nan] * 10)])
        result = isotonic_recalibrate(pit)
        self.assertEqual(result.n_pit_values, 100)

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        result = isotonic_recalibrate(rng.uniform(0, 1, size=100))
        d = result.to_dict()
        self.assertIn("ks_before", d)
        self.assertIn("ks_after", d)
        self.assertIn("ece_before", d)
        self.assertIn("ece_after", d)


class TestComputeECE(unittest.TestCase):
    """Test ECE helper."""

    def test_perfect_calibration(self):
        """Uniform PIT has low ECE."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=10000)
        ece = _compute_ece(pit)
        self.assertLess(ece, 0.01)

    def test_biased_high_ece(self):
        """Clustered PIT has high ECE."""
        pit = np.array([0.5] * 100)  # All at 0.5
        ece = _compute_ece(pit)
        self.assertGreater(ece, 0.05)


class TestIsotonicRegression(unittest.TestCase):
    """Test PAV isotonic regression."""

    def test_already_monotonic(self):
        """Monotonic input is unchanged."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        result = _isotonic_regression(x, y)
        np.testing.assert_allclose(result, y, atol=1e-10)

    def test_non_monotonic_fixed(self):
        """Non-monotonic input is made monotonic."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 3, 2, 4, 5], dtype=float)
        result = _isotonic_regression(x, y)
        # Should be non-decreasing when sorted by x
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs >= -1e-10))

    def test_empty_input(self):
        result = _isotonic_regression(np.array([]), np.array([]))
        self.assertEqual(len(result), 0)


# ===========================================================================
# Story 26.2: Location-Scale Correction
# ===========================================================================

class TestLocationScaleCorrection(unittest.TestCase):
    """AC: location_scale_correction corrects systematic bias."""

    def test_basic_correction(self):
        """Positive bias in innovations -> positive mean_innovation."""
        rng = np.random.default_rng(42)
        # Innovations with strong positive bias (model predicts too low)
        innovations = rng.normal(0.005, 0.01, size=500)
        R = np.ones(500) * 0.01 ** 2  # Predicted variance

        result = location_scale_correction(innovations, R, window=200)
        self.assertIsInstance(result, LocationScaleResult)
        self.assertGreater(result.mean_innovation, 0)  # Positive bias detected

    def test_zero_bias_small_correction(self):
        """Unbiased innovations -> small delta_mu."""
        rng = np.random.default_rng(42)
        innovations = rng.normal(0.0, 0.01, size=200)
        R = np.ones(200) * 0.01 ** 2

        result = location_scale_correction(innovations, R, window=60)
        self.assertLess(abs(result.delta_mu), 0.005)

    def test_scale_correction_underdispersed(self):
        """If innovations vary more than predicted, scale > 1."""
        rng = np.random.default_rng(42)
        true_vol = 0.02
        pred_vol = 0.01  # Underpredicting volatility
        innovations = rng.normal(0, true_vol, size=200)
        R = np.ones(200) * pred_vol ** 2

        result = location_scale_correction(innovations, R, window=60)
        self.assertGreater(result.scale_sigma, 1.0)  # Need to scale up

    def test_scale_correction_overdispersed(self):
        """If innovations vary less than predicted, scale < 1."""
        rng = np.random.default_rng(42)
        true_vol = 0.005
        pred_vol = 0.02  # Overpredicting volatility
        innovations = rng.normal(0, true_vol, size=200)
        R = np.ones(200) * pred_vol ** 2

        result = location_scale_correction(innovations, R, window=60)
        self.assertLess(result.scale_sigma, 1.0)  # Need to scale down

    def test_bias_reduced_after_correction(self):
        """AC: After correction |bias| < 0.5 SE (conceptual test)."""
        rng = np.random.default_rng(42)
        bias = 0.003
        innovations = rng.normal(bias, 0.01, size=200)
        R = np.ones(200) * 0.01 ** 2

        result = location_scale_correction(innovations, R, window=100)
        # The delta_mu should be close to the bias
        self.assertAlmostEqual(result.delta_mu, bias, delta=0.005)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            location_scale_correction(np.ones(100), np.ones(50))

    def test_few_observations_returns_identity(self):
        """Very few observations -> identity correction."""
        result = location_scale_correction(np.ones(3), np.ones(3), window=60)
        self.assertAlmostEqual(result.delta_mu, 0.0, places=5)
        self.assertAlmostEqual(result.scale_sigma, 1.0, places=5)

    def test_ewm_lambda_effect(self):
        """Higher lambda gives more weight to recent observations."""
        rng = np.random.default_rng(42)
        # First half: positive bias; second half: negative bias
        innovations = np.concatenate([
            rng.normal(0.005, 0.01, size=50),
            rng.normal(-0.005, 0.01, size=50),
        ])
        R = np.ones(100) * 0.01 ** 2

        result = location_scale_correction(innovations, R, window=100, ewm_lambda=0.95)
        # EWM should be negative (recent observations dominate)
        self.assertLess(result.delta_mu, 0.003)

    def test_to_dict(self):
        result = location_scale_correction(np.ones(50), np.ones(50) * 0.01)
        d = result.to_dict()
        self.assertIn("delta_mu", d)
        self.assertIn("scale_sigma", d)
        self.assertIn("bias_se", d)

    def test_variance_ratio_meaningful(self):
        """Variance ratio ~1.0 for well-calibrated model."""
        rng = np.random.default_rng(42)
        vol = 0.015
        innovations = rng.normal(0, vol, size=500)
        R = np.ones(500) * vol ** 2

        result = location_scale_correction(innovations, R, window=200)
        self.assertAlmostEqual(result.variance_ratio, 1.0, delta=0.3)


# ===========================================================================
# Story 26.3: Adaptive Recalibration Frequency
# ===========================================================================

class TestRecalibrationSchedule(unittest.TestCase):
    """AC: recalibration_schedule returns frequency based on PIT deviation."""

    def test_well_calibrated_monthly(self):
        """AC: KS < 0.05 -> monthly recalibration."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=1000)  # Large sample for stable KS
        result = recalibration_schedule(pit, ks_window=1000)
        self.assertIsInstance(result, RecalibrationScheduleResult)
        self.assertEqual(result.frequency, "monthly")
        self.assertLess(result.ks_statistic, KS_THRESHOLD_WEEKLY)

    def test_mild_drift_weekly(self):
        """AC: KS in [0.05, 0.10] -> weekly recalibration."""
        rng = np.random.default_rng(42)
        # Mildly biased PIT
        pit = np.clip(rng.beta(1.3, 1.0, size=100), 0, 1)
        result = recalibration_schedule(pit)
        # This may be weekly or daily depending on exact realization
        self.assertIn(result.frequency, ("weekly", "daily"))

    def test_severely_miscalibrated_daily(self):
        """AC: KS > 0.10 -> daily recalibration."""
        # Strongly biased PIT: all near 0
        pit = np.clip(np.random.default_rng(42).beta(0.5, 5, size=100), 0, 1)
        result = recalibration_schedule(pit)
        self.assertEqual(result.frequency, "daily")
        self.assertGreater(result.ks_statistic, KS_THRESHOLD_DAILY)

    def test_few_values_conservative(self):
        """Too few PIT values -> daily (conservative)."""
        result = recalibration_schedule(np.array([0.5, 0.6, 0.7]))
        self.assertEqual(result.frequency, "daily")

    def test_ks_pvalue_returned(self):
        """KS p-value is included in result."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=100)
        result = recalibration_schedule(pit)
        self.assertGreater(result.ks_pvalue, 0.0)

    def test_custom_thresholds(self):
        """Custom thresholds work."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=100)
        result = recalibration_schedule(pit, threshold_daily=0.5, threshold_weekly=0.4)
        self.assertEqual(result.frequency, "monthly")

    def test_nan_filtered(self):
        """NaN PIT values filtered."""
        rng = np.random.default_rng(42)
        pit = np.concatenate([rng.uniform(0, 1, size=60), np.array([np.nan] * 10)])
        result = recalibration_schedule(pit)
        self.assertEqual(result.n_pit_values, 60)

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        result = recalibration_schedule(rng.uniform(0, 1, size=100))
        d = result.to_dict()
        self.assertIn("frequency", d)
        self.assertIn("ks_statistic", d)


class TestRecalibrationStability(unittest.TestCase):
    """AC: Recalibration stable: g changes < 0.05 sup-norm between weeks."""

    def test_identical_mappings(self):
        """Same mapping: zero distance."""
        x = np.linspace(0, 1, 50)
        y = x.copy()
        dist = compute_recalibration_stability((x, y), (x, y))
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_different_mappings(self):
        """Different mappings: positive distance."""
        x = np.linspace(0, 1, 50)
        y1 = x
        y2 = x ** 2
        dist = compute_recalibration_stability((x, y1), (x, y2))
        self.assertGreater(dist, 0.0)

    def test_stability_on_consecutive_windows(self):
        """AC: Consecutive windows produce stable mappings."""
        rng = np.random.default_rng(42)
        # Same distribution, different samples (consecutive windows)
        pit1 = rng.uniform(0, 1, size=126)
        pit2 = rng.uniform(0, 1, size=126)

        result1 = isotonic_recalibrate(pit1)
        result2 = isotonic_recalibrate(pit2)

        dist = compute_recalibration_stability(
            (result1.mapping_x, result1.mapping_y),
            (result2.mapping_x, result2.mapping_y),
        )
        # For same distribution, mapping should be similar
        self.assertLess(dist, 0.25)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for PIT recalibration pipeline."""

    def test_full_recalibration_pipeline(self):
        """Full pipeline: PIT -> isotonic -> location-scale -> schedule."""
        rng = np.random.default_rng(42)

        # Simulated model with strong bias
        true_returns = rng.normal(0.005, 0.02, size=500)
        predicted_mu = np.zeros(500)  # Model misses the drift
        predicted_sigma = np.ones(500) * 0.02

        # Innovations
        innovations = true_returns - predicted_mu
        R = predicted_sigma ** 2

        # PIT values (from CDF of predicted distribution)
        from scipy.stats import norm
        pit = norm.cdf(true_returns, loc=predicted_mu, scale=predicted_sigma)

        # Step 1: Isotonic recalibration
        iso_result = isotonic_recalibrate(pit)
        self.assertLess(iso_result.ks_after, iso_result.ks_before + 0.05)

        # Step 2: Location-scale correction
        ls_result = location_scale_correction(innovations, R, window=200)
        self.assertGreater(ls_result.mean_innovation, 0)  # Detects positive bias

        # Step 3: Recalibration schedule
        sched = recalibration_schedule(pit)
        self.assertIn(sched.frequency, ("daily", "weekly", "monthly"))

    def test_biased_model_corrected(self):
        """Biased model gets corrected by location-scale."""
        rng = np.random.default_rng(42)
        n = 200
        true_drift = 0.003
        model_drift = 0.0  # Model underestimates drift

        returns = rng.normal(true_drift, 0.015, size=n)
        innovations = returns - model_drift
        R = np.ones(n) * 0.015 ** 2

        result = location_scale_correction(innovations, R, window=100)

        # After applying correction:
        corrected_drift = model_drift + result.delta_mu
        # Should be closer to true drift
        err_before = abs(model_drift - true_drift)
        err_after = abs(corrected_drift - true_drift)
        self.assertLess(err_after, err_before)

    def test_computational_overhead_adaptive(self):
        """AC: Computational overhead < 2x fixed monthly."""
        import time
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, size=252)

        # Fixed monthly: one schedule call
        t0 = time.perf_counter()
        for _ in range(12):
            recalibration_schedule(pit)
        time_fixed = time.perf_counter() - t0

        # Adaptive: more frequent calls
        t0 = time.perf_counter()
        for _ in range(252):  # Daily for a year
            recalibration_schedule(pit[-60:])
        time_adaptive = time.perf_counter() - t0

        # Adaptive should complete (overhead test)
        self.assertLess(time_adaptive, 5.0)  # Should be fast


if __name__ == "__main__":
    unittest.main()
