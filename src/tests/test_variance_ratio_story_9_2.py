"""
Tests for Story 9.2: Innovation Variance Ratio Test.

Tests that innovation_variance_ratio() properly detects c miscalibration
and produces correct VR^0.5 correction factors.
"""
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.innovation_diagnostics import (
    VR_UPPER_THRESHOLD,
    VR_LOWER_THRESHOLD,
    VR_DEFAULT_WINDOW,
    VR_DAMPENING_POWER,
    VarianceRatioResult,
    innovation_variance_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_well_calibrated(n=300, seed=42):
    """Innovations matching R: VR should be near 1.0."""
    rng = np.random.default_rng(seed)
    R = np.ones(n) * 0.01
    innovations = rng.normal(0, np.sqrt(R))
    return innovations, R


def _make_c_too_small(n=300, scale=3.0, seed=42):
    """Innovations much larger than R: VR >> 1.0 (c too small)."""
    rng = np.random.default_rng(seed)
    R = np.ones(n) * 0.01
    innovations = rng.normal(0, np.sqrt(R) * scale)
    return innovations, R


def _make_c_too_large(n=300, scale=0.3, seed=42):
    """Innovations much smaller than R: VR << 1.0 (c too large)."""
    rng = np.random.default_rng(seed)
    R = np.ones(n) * 0.01
    innovations = rng.normal(0, np.sqrt(R) * scale)
    return innovations, R


class TestVarianceRatioBasic(unittest.TestCase):
    """Basic API and contract tests."""

    def test_returns_result_dataclass(self):
        innovations, R = _make_well_calibrated()
        result = innovation_variance_ratio(innovations, R)
        self.assertIsInstance(result, VarianceRatioResult)

    def test_rolling_vr_length(self):
        innovations, R = _make_well_calibrated(n=200)
        result = innovation_variance_ratio(innovations, R)
        self.assertEqual(len(result.rolling_vr), 200)

    def test_n_obs_correct(self):
        innovations, R = _make_well_calibrated(n=150)
        result = innovation_variance_ratio(innovations, R)
        self.assertEqual(result.n_obs, 150)

    def test_window_stored(self):
        innovations, R = _make_well_calibrated()
        result = innovation_variance_ratio(innovations, R, window=30)
        self.assertEqual(result.window, 30)

    def test_current_vr_is_float(self):
        innovations, R = _make_well_calibrated()
        result = innovation_variance_ratio(innovations, R)
        self.assertIsInstance(result.current_vr, float)

    def test_c_correction_is_float(self):
        innovations, R = _make_well_calibrated()
        result = innovation_variance_ratio(innovations, R)
        self.assertIsInstance(result.c_correction, float)


class TestVarianceRatioWellCalibrated(unittest.TestCase):
    """Tests on well-calibrated innovations (VR near 1.0)."""

    def test_vr_near_1(self):
        """Well-calibrated: VR should be near 1.0."""
        innovations, R = _make_well_calibrated(n=500, seed=100)
        result = innovation_variance_ratio(innovations, R)
        self.assertAlmostEqual(result.current_vr, 1.0, delta=0.5)

    def test_no_correction_needed(self):
        """Well-calibrated: no correction needed."""
        innovations, R = _make_well_calibrated(n=500, seed=200)
        result = innovation_variance_ratio(innovations, R)
        self.assertFalse(result.needs_correction)

    def test_c_correction_is_one(self):
        """Well-calibrated: c_correction = 1.0."""
        innovations, R = _make_well_calibrated(n=500, seed=300)
        result = innovation_variance_ratio(innovations, R)
        self.assertEqual(result.c_correction, 1.0)

    def test_not_flagged_too_small(self):
        innovations, R = _make_well_calibrated(n=500, seed=400)
        result = innovation_variance_ratio(innovations, R)
        self.assertFalse(result.is_c_too_small)

    def test_not_flagged_too_large(self):
        innovations, R = _make_well_calibrated(n=500, seed=400)
        result = innovation_variance_ratio(innovations, R)
        self.assertFalse(result.is_c_too_large)


class TestVarianceRatioCTooSmall(unittest.TestCase):
    """Tests when c is too small (VR > 1.5)."""

    def test_vr_above_threshold(self):
        """c too small: VR should exceed 1.5."""
        innovations, R = _make_c_too_small(n=500, scale=3.0, seed=500)
        result = innovation_variance_ratio(innovations, R)
        self.assertGreater(result.current_vr, VR_UPPER_THRESHOLD)

    def test_flagged_too_small(self):
        """c too small: is_c_too_small should be True."""
        innovations, R = _make_c_too_small(n=500, scale=3.0, seed=510)
        result = innovation_variance_ratio(innovations, R)
        self.assertTrue(result.is_c_too_small)

    def test_needs_correction(self):
        """c too small: needs_correction should be True."""
        innovations, R = _make_c_too_small(n=500, scale=3.0, seed=520)
        result = innovation_variance_ratio(innovations, R)
        self.assertTrue(result.needs_correction)

    def test_c_correction_inflates(self):
        """c too small: c_correction > 1.0 (inflate c)."""
        innovations, R = _make_c_too_small(n=500, scale=3.0, seed=530)
        result = innovation_variance_ratio(innovations, R)
        self.assertGreater(result.c_correction, 1.0)

    def test_c_correction_is_sqrt_vr(self):
        """c_correction = VR^0.5 (square-root dampening)."""
        innovations, R = _make_c_too_small(n=500, scale=3.0, seed=540)
        result = innovation_variance_ratio(innovations, R)
        expected = result.current_vr ** VR_DAMPENING_POWER
        self.assertAlmostEqual(result.c_correction, expected, places=6)

    def test_correction_closes_gap(self):
        """After correction, c_new * c_correction should be closer to true c."""
        innovations, R = _make_c_too_small(n=500, scale=3.0, seed=550)
        result = innovation_variance_ratio(innovations, R)
        # VR ~ scale^2 = 9, so correction ~ 3.0
        # c_new = c_old * 3.0, bringing it closer to truth
        self.assertGreater(result.c_correction, 1.5)


class TestVarianceRatioCTooLarge(unittest.TestCase):
    """Tests when c is too large (VR < 0.7)."""

    def test_vr_below_threshold(self):
        """c too large: VR should be below 0.7."""
        innovations, R = _make_c_too_large(n=500, scale=0.3, seed=600)
        result = innovation_variance_ratio(innovations, R)
        self.assertLess(result.current_vr, VR_LOWER_THRESHOLD)

    def test_flagged_too_large(self):
        """c too large: is_c_too_large should be True."""
        innovations, R = _make_c_too_large(n=500, scale=0.3, seed=610)
        result = innovation_variance_ratio(innovations, R)
        self.assertTrue(result.is_c_too_large)

    def test_needs_correction(self):
        innovations, R = _make_c_too_large(n=500, scale=0.3, seed=620)
        result = innovation_variance_ratio(innovations, R)
        self.assertTrue(result.needs_correction)

    def test_c_correction_deflates(self):
        """c too large: c_correction < 1.0 (deflate c)."""
        innovations, R = _make_c_too_large(n=500, scale=0.3, seed=630)
        result = innovation_variance_ratio(innovations, R)
        self.assertLess(result.c_correction, 1.0)

    def test_c_correction_is_sqrt_vr(self):
        """c_correction = VR^0.5 for deflation too."""
        innovations, R = _make_c_too_large(n=500, scale=0.3, seed=640)
        result = innovation_variance_ratio(innovations, R)
        expected = result.current_vr ** VR_DAMPENING_POWER
        self.assertAlmostEqual(result.c_correction, expected, places=6)


class TestVarianceRatioRolling(unittest.TestCase):
    """Tests on rolling VR behavior."""

    def test_rolling_vr_positive(self):
        """Rolling VR values should be positive."""
        innovations, R = _make_well_calibrated(n=300)
        result = innovation_variance_ratio(innovations, R)
        self.assertTrue(np.all(result.rolling_vr > 0))

    def test_rolling_vr_regime_shift(self):
        """VR should shift when innovations change mid-series."""
        rng = np.random.default_rng(700)
        n = 300
        R = np.ones(n) * 0.01
        innovations = np.zeros(n)
        # First half: well-calibrated
        innovations[:150] = rng.normal(0, 0.1, size=150)
        # Second half: c too small (innovations 3x bigger)
        innovations[150:] = rng.normal(0, 0.3, size=150)

        result = innovation_variance_ratio(innovations, R, window=60)
        # VR near end should be much higher than near t=100
        vr_early = result.rolling_vr[100]
        vr_late = result.rolling_vr[-1]
        self.assertGreater(vr_late, vr_early)


class TestVarianceRatioEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_short_series(self):
        """Series shorter than window: no correction."""
        innovations = np.array([0.1, -0.1, 0.05])
        R = np.array([0.01, 0.01, 0.01])
        result = innovation_variance_ratio(innovations, R, window=60)
        self.assertFalse(result.needs_correction)
        self.assertEqual(result.c_correction, 1.0)

    def test_exactly_window_length(self):
        """Series exactly window length: should still work."""
        innovations, R = _make_well_calibrated(n=60, seed=800)
        result = innovation_variance_ratio(innovations, R, window=60)
        self.assertIsInstance(result, VarianceRatioResult)

    def test_custom_window(self):
        """Custom window size."""
        innovations, R = _make_well_calibrated(n=200, seed=810)
        result = innovation_variance_ratio(innovations, R, window=30)
        self.assertEqual(result.window, 30)

    def test_zero_R(self):
        """R=0 edge case: should not crash."""
        innovations = np.ones(100) * 0.01
        R = np.zeros(100)
        result = innovation_variance_ratio(innovations, R, window=60)
        self.assertIsInstance(result, VarianceRatioResult)


class TestVarianceRatioConstants(unittest.TestCase):
    """Test configuration constants."""

    def test_upper_threshold(self):
        self.assertEqual(VR_UPPER_THRESHOLD, 1.5)

    def test_lower_threshold(self):
        self.assertEqual(VR_LOWER_THRESHOLD, 0.7)

    def test_default_window(self):
        self.assertEqual(VR_DEFAULT_WINDOW, 60)

    def test_dampening_power(self):
        self.assertEqual(VR_DAMPENING_POWER, 0.5)


if __name__ == '__main__':
    unittest.main()
