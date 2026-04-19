"""
Tests for Story 7.3: Overnight Gap Detector and Vol Adjustment.

Tests cover:
  - Gap detection API (shapes, types, edge cases)
  - Threshold calibration (2-sigma trailing vol)
  - Variance inflation on gap days (gap^2/4)
  - Filter P inflation on gap days
  - Synthetic earnings-gap scenarios
  - Edge cases (no gaps, all gaps, short series)
  - Reproducibility
"""
from __future__ import annotations

import numpy as np
import unittest

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.realized_volatility import (
    detect_overnight_gap,
    adjust_vol_for_gaps,
    adjust_filter_P_for_gaps,
    GapDetectionResult,
    GAP_THRESHOLD_SIGMA,
    GAP_TRAILING_WINDOW,
    GAP_VAR_FRACTION,
    GAP_P_INFLATE_FACTOR,
    MIN_VARIANCE,
)


# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

def _make_normal_prices(n: int = 200, seed: int = 42) -> dict:
    """Generate normal OHLC data with no significant gaps."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.015  # ~1.5% daily vol
    ret = daily_vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(ret))
    # Open tracks previous close with tiny noise
    open_ = np.copy(close)
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.001, n - 1))
    open_[0] = close[0]
    return {"open_": open_, "close": close, "returns": ret}


def _make_gap_prices(
    n: int = 200,
    gap_indices: list = None,
    gap_size: float = 0.05,
    seed: int = 42,
) -> dict:
    """Generate OHLC data with specific overnight gaps."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.015
    ret = daily_vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(ret))

    open_ = np.copy(close)
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.001, n - 1))
    open_[0] = close[0]

    if gap_indices is None:
        gap_indices = [50, 100, 150]

    for idx in gap_indices:
        if 0 < idx < n:
            # Add a large gap: open = previous close * exp(gap)
            direction = rng.choice([-1, 1])
            open_[idx] = close[idx - 1] * np.exp(direction * gap_size)

    return {
        "open_": open_,
        "close": close,
        "returns": ret,
        "gap_indices": gap_indices,
        "gap_size": gap_size,
    }


# ============================================================================
# TEST: DETECTION API & BASIC PROPERTIES
# ============================================================================

class TestDetectionAPI(unittest.TestCase):
    """Test basic API contract of detect_overnight_gap."""

    @classmethod
    def setUpClass(cls):
        cls.data = _make_gap_prices(n=200, gap_indices=[50, 100, 150], gap_size=0.05)
        cls.result = detect_overnight_gap(cls.data["open_"], cls.data["close"])

    def test_returns_gap_detection_result(self):
        self.assertIsInstance(self.result, GapDetectionResult)

    def test_is_gap_shape(self):
        self.assertEqual(len(self.result.is_gap), 200)

    def test_is_gap_dtype(self):
        self.assertEqual(self.result.is_gap.dtype, bool)

    def test_gap_return_shape(self):
        self.assertEqual(len(self.result.gap_return), 200)

    def test_gap_magnitude_shape(self):
        self.assertEqual(len(self.result.gap_magnitude), 200)

    def test_threshold_shape(self):
        self.assertEqual(len(self.result.threshold), 200)

    def test_gap_magnitude_non_negative(self):
        self.assertTrue(np.all(self.result.gap_magnitude >= 0))

    def test_first_point_never_gap(self):
        """First point has no prior close, should never be flagged."""
        self.assertFalse(self.result.is_gap[0])

    def test_n_gaps_matches_sum(self):
        self.assertEqual(self.result.n_gaps, int(np.sum(self.result.is_gap)))

    def test_gap_fraction_range(self):
        self.assertGreaterEqual(self.result.gap_fraction, 0.0)
        self.assertLessEqual(self.result.gap_fraction, 1.0)


# ============================================================================
# TEST: THRESHOLD CALIBRATION
# ============================================================================

class TestThreshold(unittest.TestCase):
    """Test gap threshold is properly calibrated at 2*sigma."""

    def test_large_gaps_detected(self):
        """5% gap on 1.5% vol data should always be detected."""
        d = _make_gap_prices(n=200, gap_indices=[50, 100, 150], gap_size=0.05)
        result = detect_overnight_gap(d["open_"], d["close"])
        # All 3 gap days should be detected
        for idx in [50, 100, 150]:
            self.assertTrue(result.is_gap[idx],
                            f"Gap at index {idx} not detected")

    def test_small_noise_not_detected(self):
        """Normal 0.1% overnight noise should not be flagged."""
        d = _make_normal_prices(n=200, seed=10)
        result = detect_overnight_gap(d["open_"], d["close"])
        # With 1.5% vol and 0.1% open noise, gaps should be very rare
        self.assertLess(result.n_gaps, 5,
                        f"Too many false positives: {result.n_gaps}")

    def test_custom_threshold(self):
        """Lower threshold should detect more gaps."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.03)
        result_strict = detect_overnight_gap(d["open_"], d["close"],
                                             threshold_sigma=3.0)
        result_loose = detect_overnight_gap(d["open_"], d["close"],
                                            threshold_sigma=1.0)
        self.assertGreaterEqual(result_loose.n_gaps, result_strict.n_gaps)

    def test_provided_vol_used(self):
        """When vol is provided, should use it instead of computing."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        # Very high vol -> gap won't be detected
        high_vol = np.full(200, 0.10)  # 10% vol, gap is only 5%
        result = detect_overnight_gap(d["open_"], d["close"], vol=high_vol)
        self.assertFalse(result.is_gap[50])

        # Very low vol -> gap will be detected
        low_vol = np.full(200, 0.005)
        result = detect_overnight_gap(d["open_"], d["close"], vol=low_vol)
        self.assertTrue(result.is_gap[50])

    def test_threshold_positive(self):
        """Threshold should always be positive."""
        d = _make_normal_prices(n=200, seed=20)
        result = detect_overnight_gap(d["open_"], d["close"])
        self.assertTrue(np.all(result.threshold > 0))


# ============================================================================
# TEST: VARIANCE INFLATION
# ============================================================================

class TestVarianceInflation(unittest.TestCase):
    """Test that variance is properly inflated on gap days."""

    def test_gap_days_have_higher_variance(self):
        """Adjusted variance should be higher on gap days."""
        d = _make_gap_prices(n=200, gap_indices=[50, 100], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        base_var = np.full(200, 0.0002)  # Base variance
        adjusted = adjust_vol_for_gaps(base_var, gap_result)

        for idx in [50, 100]:
            if gap_result.is_gap[idx]:
                self.assertGreater(adjusted[idx], base_var[idx])

    def test_non_gap_days_unchanged(self):
        """Non-gap days should have same variance."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        base_var = np.full(200, 0.0002)
        adjusted = adjust_vol_for_gaps(base_var, gap_result)

        non_gap = ~gap_result.is_gap
        np.testing.assert_array_equal(adjusted[non_gap], base_var[non_gap])

    def test_inflation_formula(self):
        """Inflation should be exactly gap_var_fraction * gap^2."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        base_var = np.full(200, 0.0002)
        adjusted = adjust_vol_for_gaps(base_var, gap_result, gap_var_fraction=0.25)

        if gap_result.is_gap[50]:
            expected = base_var[50] + 0.25 * gap_result.gap_return[50] ** 2
            self.assertAlmostEqual(adjusted[50], expected, places=15)

    def test_custom_fraction(self):
        """Different gap_var_fraction should scale the inflation."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        base_var = np.full(200, 0.0002)
        adj_small = adjust_vol_for_gaps(base_var, gap_result, gap_var_fraction=0.1)
        adj_large = adjust_vol_for_gaps(base_var, gap_result, gap_var_fraction=0.5)

        if gap_result.is_gap[50]:
            self.assertGreater(adj_large[50], adj_small[50])

    def test_variance_floor_maintained(self):
        """Adjusted variance should never go below MIN_VARIANCE."""
        d = _make_normal_prices(n=100, seed=30)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        base_var = np.full(100, MIN_VARIANCE)
        adjusted = adjust_vol_for_gaps(base_var, gap_result)
        self.assertTrue(np.all(adjusted >= MIN_VARIANCE))


# ============================================================================
# TEST: FILTER P INFLATION
# ============================================================================

class TestFilterPInflation(unittest.TestCase):
    """Test that filter uncertainty P is inflated on gap days."""

    def test_gap_days_have_higher_P(self):
        """P should be inflated on gap days."""
        d = _make_gap_prices(n=200, gap_indices=[50, 100], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        P = np.full(200, 1e-4)
        adjusted = adjust_filter_P_for_gaps(P, gap_result)

        for idx in [50, 100]:
            if gap_result.is_gap[idx]:
                self.assertGreater(adjusted[idx], P[idx])

    def test_non_gap_days_unchanged(self):
        """Non-gap days should have same P."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        P = np.full(200, 1e-4)
        adjusted = adjust_filter_P_for_gaps(P, gap_result)

        non_gap = ~gap_result.is_gap
        np.testing.assert_array_equal(adjusted[non_gap], P[non_gap])

    def test_inflation_factor_applied(self):
        """P should be multiplied by inflate_factor on gap days."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        P = np.full(200, 1e-4)
        factor = 3.0
        adjusted = adjust_filter_P_for_gaps(P, gap_result, inflate_factor=factor)

        if gap_result.is_gap[50]:
            self.assertAlmostEqual(adjusted[50], P[50] * factor, places=15)

    def test_default_factor_is_2(self):
        """Default inflation factor should be 2.0."""
        self.assertEqual(GAP_P_INFLATE_FACTOR, 2.0)

    def test_original_P_unchanged(self):
        """Original P array should not be modified (immutability)."""
        d = _make_gap_prices(n=200, gap_indices=[50], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])

        P = np.full(200, 1e-4)
        P_copy = np.copy(P)
        _ = adjust_filter_P_for_gaps(P, gap_result)
        np.testing.assert_array_equal(P, P_copy)


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and robustness."""

    def test_no_gaps_detected(self):
        """On smooth data, no gaps should be flagged (or very few)."""
        d = _make_normal_prices(n=500, seed=40)
        result = detect_overnight_gap(d["open_"], d["close"])
        # Allow a few false positives but not many
        self.assertLess(result.gap_fraction, 0.05)

    def test_all_gaps(self):
        """If every day has a huge gap, all should be detected."""
        n = 100
        rng = np.random.default_rng(50)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        # Open is always 10% away from previous close
        open_ = np.copy(close)
        open_[1:] = close[:-1] * np.exp(rng.choice([-0.10, 0.10], n - 1))
        open_[0] = close[0]

        result = detect_overnight_gap(open_, close)
        # Most days should be flagged (some early may not due to trailing vol)
        self.assertGreater(result.n_gaps, n * 0.5)

    def test_short_series(self):
        """Should not crash on very short data."""
        open_ = np.array([100.0, 103.0, 101.0])
        close = np.array([100.5, 102.0, 102.5])
        result = detect_overnight_gap(open_, close)
        self.assertEqual(len(result.is_gap), 3)

    def test_single_point(self):
        """Single data point should work."""
        result = detect_overnight_gap(np.array([100.0]), np.array([100.0]))
        self.assertEqual(len(result.is_gap), 1)
        self.assertFalse(result.is_gap[0])
        self.assertEqual(result.n_gaps, 0)

    def test_constant_prices(self):
        """Constant prices: no gaps."""
        n = 50
        p = np.full(n, 100.0)
        result = detect_overnight_gap(p, p)
        self.assertEqual(result.n_gaps, 0)

    def test_gap_return_sign_correct(self):
        """Positive gap (open > prev close) should have positive gap_return."""
        close = np.array([100.0, 100.0, 100.0])
        open_ = np.array([100.0, 110.0, 90.0])  # +10% gap, -10% gap
        result = detect_overnight_gap(open_, close)
        self.assertGreater(result.gap_return[1], 0)  # Up gap
        self.assertLess(result.gap_return[2], 0)      # Down gap

    def test_variance_adjustment_immutable(self):
        """adjust_vol_for_gaps should not modify input array."""
        d = _make_gap_prices(n=100, gap_indices=[30], gap_size=0.05)
        gap_result = detect_overnight_gap(d["open_"], d["close"])
        base_var = np.full(100, 0.0002)
        base_copy = np.copy(base_var)
        _ = adjust_vol_for_gaps(base_var, gap_result)
        np.testing.assert_array_equal(base_var, base_copy)


# ============================================================================
# TEST: CONFIGURATION DEFAULTS
# ============================================================================

class TestDefaults(unittest.TestCase):
    """Test default configuration values."""

    def test_threshold_default(self):
        self.assertEqual(GAP_THRESHOLD_SIGMA, 2.0)

    def test_trailing_window_default(self):
        self.assertEqual(GAP_TRAILING_WINDOW, 20)

    def test_var_fraction_default(self):
        self.assertEqual(GAP_VAR_FRACTION, 0.25)

    def test_p_inflate_default(self):
        self.assertEqual(GAP_P_INFLATE_FACTOR, 2.0)


# ============================================================================
# TEST: REPRODUCIBILITY
# ============================================================================

class TestReproducibility(unittest.TestCase):
    """Test deterministic output."""

    def test_deterministic(self):
        d = _make_gap_prices(n=200, gap_indices=[50, 100], gap_size=0.05, seed=60)
        r1 = detect_overnight_gap(d["open_"], d["close"])
        r2 = detect_overnight_gap(d["open_"], d["close"])
        np.testing.assert_array_equal(r1.is_gap, r2.is_gap)
        np.testing.assert_array_equal(r1.gap_return, r2.gap_return)


if __name__ == "__main__":
    unittest.main()
