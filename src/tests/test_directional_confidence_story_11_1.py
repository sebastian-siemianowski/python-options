"""
Test Suite for Story 11.1: Calibrated Directional Confidence via Platt Scaling
===============================================================================

Tests Platt scaling calibration that transforms raw sign probabilities
into calibrated directional confidence scores.
"""
import os
import sys
import unittest
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.directional_confidence import (
    PlattCalibrationResult,
    WalkForwardCalibrationResult,
    _safe_logit,
    _sigmoid,
    _fit_platt,
    platt_calibrate,
    platt_calibrate_walkforward,
    compute_ece,
    compute_reliability_diagram,
    directional_confidence,
    apply_platt_single,
    PLATT_TRAIN_WINDOW,
    PLATT_MIN_TRAIN,
    PLATT_CLIP_LOGIT,
    PLATT_CLIP_PROB,
    ECE_NUM_BINS,
    RELIABILITY_TOLERANCE,
)


def _generate_calibrated_data(n=2000, seed=42):
    """Generate synthetic data where raw probs are miscalibrated (overconfident)."""
    rng = np.random.default_rng(seed)

    # True probabilities from uniform
    true_p = rng.uniform(0.3, 0.7, n)

    # Raw probs are overconfident: pushed away from 0.5
    raw_probs = 0.5 + 1.5 * (true_p - 0.5)
    raw_probs = np.clip(raw_probs, 0.01, 0.99)

    # Outcomes sampled from true probabilities
    outcomes = (rng.random(n) < true_p).astype(float)

    return raw_probs, outcomes, true_p


def _generate_well_calibrated_data(n=2000, seed=42):
    """Generate data where raw probs are already well-calibrated."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.3, 0.7, n)
    raw_probs = true_p + rng.normal(0, 0.02, n)
    raw_probs = np.clip(raw_probs, 0.01, 0.99)
    outcomes = (rng.random(n) < true_p).astype(float)
    return raw_probs, outcomes


# ===================================================================
# TestSafeLogit
# ===================================================================

class TestSafeLogit(unittest.TestCase):
    """Test logit transform with safety clipping."""

    def test_logit_half_is_zero(self):
        result = _safe_logit(np.array([0.5]))
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_logit_monotone(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        logits = _safe_logit(probs)
        for i in range(len(logits) - 1):
            self.assertLess(logits[i], logits[i + 1])

    def test_logit_clipped_at_boundary(self):
        logits = _safe_logit(np.array([0.0, 1.0]))
        self.assertGreaterEqual(logits[0], -PLATT_CLIP_LOGIT)
        self.assertLessEqual(logits[1], PLATT_CLIP_LOGIT)

    def test_logit_symmetric(self):
        logits = _safe_logit(np.array([0.3, 0.7]))
        self.assertAlmostEqual(logits[0], -logits[1], places=5)

    def test_logit_finite(self):
        probs = np.array([1e-10, 0.5, 1 - 1e-10])
        logits = _safe_logit(probs)
        self.assertTrue(np.all(np.isfinite(logits)))


# ===================================================================
# TestSigmoid
# ===================================================================

class TestSigmoid(unittest.TestCase):
    """Test numerically stable sigmoid."""

    def test_sigmoid_zero_is_half(self):
        self.assertAlmostEqual(float(_sigmoid(np.array([0.0]))[0]), 0.5)

    def test_sigmoid_range(self):
        x = np.array([-100, -10, 0, 10, 100])
        s = _sigmoid(x)
        self.assertTrue(np.all(s >= 0))
        self.assertTrue(np.all(s <= 1))

    def test_sigmoid_monotone(self):
        x = np.linspace(-5, 5, 100)
        s = _sigmoid(x)
        for i in range(len(s) - 1):
            self.assertLessEqual(s[i], s[i + 1])

    def test_sigmoid_inverse_of_logit(self):
        probs = np.array([0.2, 0.5, 0.8])
        roundtrip = _sigmoid(_safe_logit(probs))
        np.testing.assert_allclose(roundtrip, probs, atol=1e-5)


# ===================================================================
# TestFitPlatt
# ===================================================================

class TestFitPlatt(unittest.TestCase):
    """Test Platt parameter fitting."""

    def test_identity_when_calibrated(self):
        """If probs are already calibrated, a ~1, b ~0."""
        rng = np.random.default_rng(42)
        n = 1000
        true_p = rng.uniform(0.2, 0.8, n)
        logits = _safe_logit(true_p)
        labels = (rng.random(n) < true_p).astype(float)
        a, b = _fit_platt(logits, labels)
        # Should be close to identity (a=1, b=0) but with noise
        self.assertAlmostEqual(a, 1.0, delta=0.5)
        self.assertAlmostEqual(b, 0.0, delta=0.3)

    def test_overconfident_correction(self):
        """Overconfident probs should get a < 1 (compression)."""
        rng = np.random.default_rng(42)
        n = 2000
        true_p = rng.uniform(0.35, 0.65, n)
        # Overconfident: push away from 0.5
        raw_p = 0.5 + 2.0 * (true_p - 0.5)
        raw_p = np.clip(raw_p, 0.01, 0.99)
        logits = _safe_logit(raw_p)
        labels = (rng.random(n) < true_p).astype(float)
        a, b = _fit_platt(logits, labels)
        # a should be < 1 to compress overconfidence
        self.assertLess(a, 1.0)

    def test_returns_float(self):
        a, b = _fit_platt(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertIsInstance(a, float)
        self.assertIsInstance(b, float)

    def test_small_sample(self):
        a, b = _fit_platt(np.array([0.5]), np.array([1.0]))
        self.assertEqual(a, 1.0)
        self.assertEqual(b, 0.0)


# ===================================================================
# TestPlattCalibrate
# ===================================================================

class TestPlattCalibrate(unittest.TestCase):
    """Test platt_calibrate() simple version."""

    def test_returns_result(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=1000)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertIsInstance(result, PlattCalibrationResult)

    def test_calibrated_probs_in_range(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=1000)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertTrue(np.all(result.calibrated_probs >= 0))
        self.assertTrue(np.all(result.calibrated_probs <= 1))

    def test_ece_improves(self):
        """ECE should improve (decrease) after calibration."""
        raw_probs, outcomes, _ = _generate_calibrated_data(n=2000)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertLess(result.ece_after, result.ece_before)

    def test_output_length_matches(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=500)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertEqual(len(result.calibrated_probs), 500)

    def test_reliability_bins_10(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=2000)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertEqual(len(result.reliability_bins), 10)
        self.assertEqual(len(result.reliability_counts), 10)

    def test_n_calibrated_positive(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=500)
        result = platt_calibrate(raw_probs, outcomes, validation_frac=0.2)
        self.assertGreater(result.n_calibrated, 0)


# ===================================================================
# TestWalkForwardCalibration
# ===================================================================

class TestWalkForwardCalibration(unittest.TestCase):
    """Test walk-forward Platt calibration (no leakage)."""

    def test_returns_result(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=1000)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=100)
        self.assertIsInstance(result, WalkForwardCalibrationResult)

    def test_no_future_leakage(self):
        """First min_train entries should be uncalibrated (equal to raw)."""
        raw_probs, outcomes, _ = _generate_calibrated_data(n=500)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=100)
        # First 100 should be unchanged
        np.testing.assert_array_equal(
            result.calibrated_probs[:100], raw_probs[:100]
        )

    def test_n_calibrated_correct(self):
        n = 500
        min_train = 100
        raw_probs, outcomes, _ = _generate_calibrated_data(n=n)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=min_train)
        self.assertEqual(result.n_calibrated, n - min_train)

    def test_ece_improves_walkforward(self):
        """Walk-forward ECE should improve on miscalibrated data."""
        raw_probs, outcomes, _ = _generate_calibrated_data(n=2000, seed=123)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=200)
        self.assertLess(result.ece_after, result.ece_before)

    def test_calibrated_probs_in_range(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=500)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=100)
        self.assertTrue(np.all(result.calibrated_probs >= 0))
        self.assertTrue(np.all(result.calibrated_probs <= 1))

    def test_output_length_matches(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=500)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=100)
        self.assertEqual(len(result.calibrated_probs), 500)

    def test_hit_rates_computed(self):
        raw_probs, outcomes, _ = _generate_calibrated_data(n=2000)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=200)
        self.assertGreaterEqual(result.hit_rate_raw, 0.0)
        self.assertLessEqual(result.hit_rate_raw, 1.0)
        self.assertGreaterEqual(result.hit_rate_calibrated, 0.0)
        self.assertLessEqual(result.hit_rate_calibrated, 1.0)


# ===================================================================
# TestComputeECE
# ===================================================================

class TestComputeECE(unittest.TestCase):
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration_zero_ece(self):
        """If probs exactly match frequencies, ECE should be ~0."""
        # All probs = 0.5, outcomes = 50% 0s and 50% 1s
        probs = np.full(1000, 0.5)
        labels = np.array([0.0] * 500 + [1.0] * 500)
        ece = compute_ece(probs, labels)
        self.assertAlmostEqual(ece, 0.0, places=2)

    def test_worst_calibration(self):
        """All predict 1.0 but all outcomes are 0 -> ECE = 1.0."""
        probs = np.full(100, 0.95)
        labels = np.zeros(100)
        ece = compute_ece(probs, labels)
        self.assertGreater(ece, 0.5)

    def test_ece_range(self):
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, 1000)
        labels = rng.integers(0, 2, 1000).astype(float)
        ece = compute_ece(probs, labels)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_empty_returns_zero(self):
        ece = compute_ece(np.array([]), np.array([]))
        self.assertEqual(ece, 0.0)


# ===================================================================
# TestReliabilityDiagram
# ===================================================================

class TestReliabilityDiagram(unittest.TestCase):
    """Test reliability diagram computation."""

    def test_returns_10_bins(self):
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, 500)
        labels = rng.integers(0, 2, 500).astype(float)
        bins, counts = compute_reliability_diagram(probs, labels)
        self.assertEqual(len(bins), 10)
        self.assertEqual(len(counts), 10)

    def test_empty_bins_nan(self):
        # All probs in [0.4, 0.6] -> bins 0-3 and 6-9 should be empty
        probs = np.full(100, 0.45)
        labels = np.ones(100)
        bins, counts = compute_reliability_diagram(probs, labels)
        # Bin 4 (0.4-0.5) should have data
        self.assertEqual(counts[4], 100)
        # Other bins should be empty
        self.assertTrue(np.isnan(bins[0]))

    def test_perfect_calibration_diagonal(self):
        """If calibrated, bins should be near diagonal."""
        rng = np.random.default_rng(42)
        n_per_bin = 1000
        probs = []
        labels = []
        for i in range(10):
            p = (i + 0.5) / 10.0  # bin center
            probs.extend([p] * n_per_bin)
            labels.extend((rng.random(n_per_bin) < p).tolist())

        probs = np.array(probs)
        labels = np.array(labels)
        bins, counts = compute_reliability_diagram(probs, labels)

        for i in range(10):
            if counts[i] > 0:
                expected = (i + 0.5) / 10.0
                self.assertAlmostEqual(bins[i], expected, delta=0.05)

    def test_counts_sum_to_n(self):
        rng = np.random.default_rng(42)
        n = 500
        probs = rng.uniform(0, 1, n)
        labels = rng.integers(0, 2, n).astype(float)
        _, counts = compute_reliability_diagram(probs, labels)
        self.assertEqual(counts.sum(), n)


# ===================================================================
# TestDirectionalConfidence
# ===================================================================

class TestDirectionalConfidence(unittest.TestCase):
    """Test directional confidence conversion."""

    def test_half_gives_zero(self):
        self.assertAlmostEqual(directional_confidence(0.5), 0.0)

    def test_one_gives_one(self):
        self.assertAlmostEqual(directional_confidence(1.0), 1.0)

    def test_zero_gives_one(self):
        self.assertAlmostEqual(directional_confidence(0.0), 1.0)

    def test_symmetric(self):
        self.assertAlmostEqual(
            directional_confidence(0.3),
            directional_confidence(0.7),
        )

    def test_monotone_above_half(self):
        for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if p < 0.9:
                self.assertLess(
                    directional_confidence(p),
                    directional_confidence(p + 0.1),
                )


# ===================================================================
# TestApplyPlattSingle
# ===================================================================

class TestApplyPlattSingle(unittest.TestCase):
    """Test single-point Platt application."""

    def test_identity(self):
        result = apply_platt_single(0.5, a=1.0, b=0.0)
        self.assertAlmostEqual(result, 0.5, places=3)

    def test_in_range(self):
        result = apply_platt_single(0.8, a=0.7, b=-0.1)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_extreme_compression(self):
        """Very small a should compress toward 0.5."""
        result = apply_platt_single(0.9, a=0.1, b=0.0)
        self.assertLess(result, 0.6)

    def test_boundary_safe(self):
        result_0 = apply_platt_single(0.0, a=1.0, b=0.0)
        result_1 = apply_platt_single(1.0, a=1.0, b=0.0)
        self.assertTrue(0.0 < result_0 < 1.0)
        self.assertTrue(0.0 < result_1 < 1.0)


# ===================================================================
# TestConstants
# ===================================================================

class TestConstants(unittest.TestCase):
    """Test that constants are sensible."""

    def test_train_window(self):
        self.assertEqual(PLATT_TRAIN_WINDOW, 500)

    def test_min_train(self):
        self.assertEqual(PLATT_MIN_TRAIN, 100)

    def test_clip_logit_positive(self):
        self.assertGreater(PLATT_CLIP_LOGIT, 0)

    def test_clip_prob_tiny(self):
        self.assertGreater(PLATT_CLIP_PROB, 0)
        self.assertLess(PLATT_CLIP_PROB, 1e-3)

    def test_ece_num_bins(self):
        self.assertEqual(ECE_NUM_BINS, 10)


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_all_same_prob(self):
        raw_probs = np.full(500, 0.6)
        outcomes = np.ones(500)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertTrue(np.all(np.isfinite(result.calibrated_probs)))

    def test_all_same_outcome(self):
        rng = np.random.default_rng(42)
        raw_probs = rng.uniform(0.3, 0.7, 500)
        outcomes = np.ones(500)
        result = platt_calibrate(raw_probs, outcomes)
        self.assertTrue(np.all(np.isfinite(result.calibrated_probs)))

    def test_small_n(self):
        raw_probs = np.array([0.4, 0.6, 0.5, 0.7, 0.3])
        outcomes = np.array([0, 1, 0, 1, 0])
        result = platt_calibrate(raw_probs, outcomes, validation_frac=0.4)
        self.assertEqual(len(result.calibrated_probs), 5)

    def test_walkforward_small_n(self):
        raw_probs = np.random.default_rng(42).uniform(0.3, 0.7, 200)
        outcomes = (np.random.default_rng(42).random(200) > 0.5).astype(float)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=50)
        self.assertEqual(result.n_calibrated, 150)


# ===================================================================
# TestIntegration
# ===================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests with synthetic data."""

    def test_overconfident_probs_get_compressed(self):
        """Overconfident raw probs should be compressed toward 0.5."""
        raw_probs, outcomes, _ = _generate_calibrated_data(n=2000)
        result = platt_calibrate(raw_probs, outcomes)

        # Calibrated should be closer to 0.5 than raw (on average)
        raw_dev = np.mean(np.abs(raw_probs - 0.5))
        cal_dev = np.mean(np.abs(result.calibrated_probs - 0.5))
        self.assertLess(cal_dev, raw_dev)

    def test_well_calibrated_unchanged(self):
        """Already calibrated probs should not change much."""
        raw_probs, outcomes = _generate_well_calibrated_data(n=2000)
        result = platt_calibrate(raw_probs, outcomes)

        # Should be similar
        diff = np.mean(np.abs(result.calibrated_probs - raw_probs))
        self.assertLess(diff, 0.1)

    def test_walkforward_ece_below_threshold(self):
        """Walk-forward ECE after calibration should be reasonable."""
        raw_probs, outcomes, _ = _generate_calibrated_data(n=2000, seed=99)
        result = platt_calibrate_walkforward(raw_probs, outcomes, min_train=200)
        # ECE after should be meaningfully improved
        self.assertLess(result.ece_after, result.ece_before)


if __name__ == "__main__":
    unittest.main()
