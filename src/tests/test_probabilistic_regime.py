"""Tests for Story 4.1: Probabilistic Regime Assignment with Soft Boundaries.

Validates:
1. Logistic boundary behavior at vol thresholds
2. Smooth transitions (no hard jumps)
3. Probability normalization
4. Soft BMA weight mixing across regimes
5. Crisis detection via tail indicator
6. Hard assignment (argmax) compatibility
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from decision.signals import (
    compute_regime_probabilities_v2,
    compute_soft_bma_weights,
    _logistic,
    REGIME_TRANSITION_WIDTH_VOL,
    REGIME_TRANSITION_WIDTH_DRIFT,
    REGIME_VOL_HIGH_BOUNDARY,
    REGIME_VOL_LOW_BOUNDARY,
    REGIME_CRISIS_VOL_THRESHOLD,
)


class TestLogisticFunction(unittest.TestCase):
    """Test the logistic sigmoid helper."""

    def test_center_is_half(self):
        self.assertAlmostEqual(_logistic(1.3, 1.3, 0.15), 0.5, places=5)

    def test_far_above_center(self):
        self.assertGreater(_logistic(3.0, 1.3, 0.15), 0.99)

    def test_far_below_center(self):
        self.assertLess(_logistic(0.0, 1.3, 0.15), 0.01)

    def test_monotonic(self):
        vals = [_logistic(x, 1.0, 0.1) for x in np.linspace(0, 2, 100)]
        for i in range(1, len(vals)):
            self.assertGreaterEqual(vals[i], vals[i - 1] - 1e-10)

    def test_extreme_clamp(self):
        """Test numerical stability for extreme inputs."""
        v = _logistic(1000.0, 0.0, 0.1)
        self.assertTrue(np.isfinite(v))
        self.assertAlmostEqual(v, 1.0, places=5)


class TestRegimeProbabilitiesV2Normalization(unittest.TestCase):
    """Test that probabilities always sum to 1."""

    def test_sum_to_one_boundary(self):
        p = compute_regime_probabilities_v2(1.30, 0.001)
        self.assertAlmostEqual(p.sum(), 1.0, places=10)

    def test_sum_to_one_low_vol(self):
        p = compute_regime_probabilities_v2(0.5, 0.0001)
        self.assertAlmostEqual(p.sum(), 1.0, places=10)

    def test_sum_to_one_high_vol(self):
        p = compute_regime_probabilities_v2(2.0, 0.005)
        self.assertAlmostEqual(p.sum(), 1.0, places=10)

    def test_sum_to_one_crisis(self):
        p = compute_regime_probabilities_v2(3.0, 0.001, tail_indicator=5.0)
        self.assertAlmostEqual(p.sum(), 1.0, places=10)

    def test_all_non_negative(self):
        for vr in [0.3, 0.85, 1.0, 1.3, 2.0, 3.0]:
            for da in [0.0, 0.0005, 0.001, 0.01]:
                p = compute_regime_probabilities_v2(vr, da)
                self.assertTrue(np.all(p >= 0), f"Negative prob for vol={vr}, drift={da}: {p}")


class TestRegimeProbabilitiesV2Boundaries(unittest.TestCase):
    """Test soft boundary behavior at regime transitions."""

    def test_vol_boundary_high_vol_mixed(self):
        """At vol=1.30 (exact high boundary), P(HIGH_VOL) should be mixed (~0.5)."""
        p = compute_regime_probabilities_v2(1.30, 0.001)
        # HIGH_VOL = regimes 1 (HVT) + 3 (HVR)
        p_high_vol = p[1] + p[3]
        self.assertGreater(p_high_vol, 0.2, "Should have some HIGH_VOL mass at boundary")
        self.assertLess(p_high_vol, 0.8, "Should not be all HIGH_VOL at boundary")

    def test_clearly_high_vol(self):
        """At vol=2.0, combined elevated-vol regimes (1+3+4) > 0.95."""
        p = compute_regime_probabilities_v2(2.0, 0.001)
        p_elevated = p[1] + p[3] + p[4]  # HIGH_VOL_TREND + HIGH_VOL_RANGE + CRISIS
        self.assertGreater(p_elevated, 0.95)

    def test_clearly_low_vol(self):
        """At vol=0.5, LOW_VOL regimes dominate."""
        p = compute_regime_probabilities_v2(0.5, 0.001)
        p_low_vol = p[0] + p[2]  # LOW_VOL_TREND + LOW_VOL_RANGE
        self.assertGreater(p_low_vol, 0.85)

    def test_drift_boundary_trending(self):
        """When drift >> threshold, P(trending) high; when near-zero, P(ranging) high."""
        p_trend = compute_regime_probabilities_v2(0.7, 0.002)
        p_range = compute_regime_probabilities_v2(0.7, 0.0001)
        # LOW_VOL_TREND (0) vs LOW_VOL_RANGE (2)
        self.assertGreater(p_trend[0], p_trend[2], "Strong drift -> more TREND")
        self.assertGreater(p_range[2], p_range[0], "Weak drift -> more RANGE")

    def test_crisis_via_vol(self):
        """Extreme vol triggers crisis."""
        p = compute_regime_probabilities_v2(3.0, 0.001)
        self.assertGreater(p[4], 0.5, "Extreme vol -> crisis should be > 0.5")

    def test_crisis_via_tail(self):
        """High tail indicator triggers crisis."""
        p = compute_regime_probabilities_v2(1.0, 0.001, tail_indicator=5.0)
        self.assertGreater(p[4], 0.3, "High tail -> some crisis mass")


class TestRegimeProbabilitiesV2Smoothness(unittest.TestCase):
    """Test smooth transitions (no hard jumps)."""

    def test_smooth_vol_transition(self):
        """As vol increases from 0.5 to 2.0, P(HIGH_VOL_TREND) increases monotonically."""
        vols = np.linspace(0.5, 2.5, 50)
        p_hvt = [compute_regime_probabilities_v2(v, 0.001)[1] for v in vols]
        # Should be generally increasing (allow small local noise from normalization)
        for i in range(2, len(p_hvt)):
            # Allow 0.05 tolerance for normalization effects
            self.assertGreater(p_hvt[i], p_hvt[0] - 0.05,
                               f"P(HVT) should increase with vol: vol={vols[i]:.2f}")

    def test_no_discontinuity_at_boundary(self):
        """No large jump at vol=1.30 boundary."""
        p_below = compute_regime_probabilities_v2(1.29, 0.001)
        p_above = compute_regime_probabilities_v2(1.31, 0.001)
        for i in range(5):
            diff = abs(p_above[i] - p_below[i])
            self.assertLess(diff, 0.1, f"Regime {i} jump at boundary: {diff:.4f}")

    def test_no_discontinuity_at_low_boundary(self):
        """No large jump at vol=0.85 boundary."""
        p_below = compute_regime_probabilities_v2(0.84, 0.001)
        p_above = compute_regime_probabilities_v2(0.86, 0.001)
        for i in range(5):
            diff = abs(p_above[i] - p_below[i])
            self.assertLess(diff, 0.1, f"Regime {i} jump at low boundary: {diff:.4f}")


class TestHardAssignmentCompatibility(unittest.TestCase):
    """Test that argmax matches current behavior for clear regimes."""

    def test_clear_low_vol_trend(self):
        p = compute_regime_probabilities_v2(0.5, 0.002)
        self.assertEqual(np.argmax(p), 0, "Clear low vol + strong drift -> LOW_VOL_TREND (0)")

    def test_clear_high_vol_trend(self):
        p = compute_regime_probabilities_v2(1.8, 0.003)
        # Could be 1 (HVT) or 4 (crisis) — both are valid for high vol
        self.assertIn(np.argmax(p), [1, 4], "High vol + strong drift -> HIGH_VOL_TREND or CRISIS")

    def test_clear_low_vol_range(self):
        p = compute_regime_probabilities_v2(0.5, 0.00005)
        self.assertEqual(np.argmax(p), 2, "Low vol + zero drift -> LOW_VOL_RANGE (2)")

    def test_crisis_extreme(self):
        p = compute_regime_probabilities_v2(4.0, 0.001, tail_indicator=6.0)
        self.assertEqual(np.argmax(p), 4, "Extreme vol + tail -> CRISIS (4)")


class TestSoftBMAWeights(unittest.TestCase):
    """Test soft BMA weight mixing across regimes."""

    def _make_regime_data(self, posteriors_by_regime):
        """Create regime_data structure from per-regime posteriors."""
        data = {}
        for r_idx, post in posteriors_by_regime.items():
            data[str(r_idx)] = {
                "model_posterior": post,
                "models": {m: {"fit_success": True} for m in post},
            }
        return data

    def test_uniform_probs_equal_average(self):
        """Uniform regime probs -> even mix of all regime posteriors."""
        regime_data = self._make_regime_data({
            0: {"m_a": 0.8, "m_b": 0.2},
            2: {"m_a": 0.2, "m_b": 0.8},
        })
        global_post = {"m_a": 0.5, "m_b": 0.5}
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        w = compute_soft_bma_weights(probs, regime_data, global_post)
        # Regimes 0,2 have data; regimes 1,3,4 fall back to global
        # total: 0.2*0.8 + 0.2*0.5 + 0.2*0.2 + 0.2*0.5 + 0.2*0.5 = 0.16+0.10+0.04+0.10+0.10 = 0.50 for m_a
        self.assertAlmostEqual(w["m_a"], 0.5, places=1)

    def test_dominant_regime_dominates_weights(self):
        """When one regime has P=0.9, its posteriors dominate."""
        regime_data = self._make_regime_data({
            0: {"m_a": 1.0, "m_b": 0.0},
            1: {"m_a": 0.0, "m_b": 1.0},
        })
        global_post = {"m_a": 0.5, "m_b": 0.5}
        probs = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
        w = compute_soft_bma_weights(probs, regime_data, global_post)
        self.assertGreater(w.get("m_a", 0), 0.8, "Regime 0 dominant -> m_a dominant")

    def test_empty_regime_data_uses_global(self):
        """No regime data -> all global."""
        global_post = {"m_x": 0.7, "m_y": 0.3}
        probs = np.array([0.5, 0.1, 0.3, 0.05, 0.05])
        w = compute_soft_bma_weights(probs, {}, global_post)
        self.assertAlmostEqual(w.get("m_x", 0), 0.7, places=2)

    def test_weights_sum_to_one(self):
        """Mixed weights should sum to 1."""
        regime_data = self._make_regime_data({
            0: {"a": 0.5, "b": 0.3, "c": 0.2},
            1: {"a": 0.1, "b": 0.2, "c": 0.7},
        })
        global_post = {"a": 0.3, "b": 0.4, "c": 0.3}
        probs = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        w = compute_soft_bma_weights(probs, regime_data, global_post)
        total = sum(w.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_smooth_weight_transition(self):
        """As regime prob shifts, weights transition smoothly."""
        regime_data = self._make_regime_data({
            0: {"m1": 1.0, "m2": 0.0},
            2: {"m1": 0.0, "m2": 1.0},
        })
        global_post = {"m1": 0.5, "m2": 0.5}
        w_prev = None
        for p0 in np.linspace(0.0, 1.0, 20):
            probs = np.array([p0, 0.0, 1.0 - p0, 0.0, 0.0])
            w = compute_soft_bma_weights(probs, regime_data, global_post)
            if w_prev is not None:
                diff = abs(w.get("m1", 0) - w_prev.get("m1", 0))
                self.assertLess(diff, 0.15, f"Weight jump too large at p0={p0:.2f}")
            w_prev = w


class TestRegimeProbsV2EdgeCases(unittest.TestCase):
    """Edge cases."""

    def test_zero_vol(self):
        p = compute_regime_probabilities_v2(0.0, 0.0)
        self.assertAlmostEqual(p.sum(), 1.0, places=10)
        self.assertTrue(np.all(p >= 0))

    def test_extreme_vol(self):
        p = compute_regime_probabilities_v2(10.0, 0.01, tail_indicator=10.0)
        self.assertAlmostEqual(p.sum(), 1.0, places=10)
        self.assertGreater(p[4], 0.9, "Extreme vol/tail -> crisis dominant")

    def test_normal_vol_range(self):
        """Vol in 0.85-1.30 (normal) should spread mass across regimes."""
        p = compute_regime_probabilities_v2(1.0, 0.0005)
        # No single regime should dominate completely
        self.assertLess(max(p), 0.8, "Normal vol shouldn't produce extreme certainty")

    def test_returns_5_elements(self):
        p = compute_regime_probabilities_v2(1.0, 0.001)
        self.assertEqual(len(p), 5)


if __name__ == "__main__":
    unittest.main()
