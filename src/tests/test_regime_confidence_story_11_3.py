"""
Test Suite for Story 11.3: Regime-Conditional Confidence Adjustment
====================================================================

Tests regime-specific confidence scaling based on historical predictability.
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.regime_confidence import (
    RegimeConfidenceResult,
    RegimeHitRates,
    compute_regime_hit_rates,
    regime_confidence_scale,
    adjust_confidence_timeseries,
    compute_regime_ece,
    REGIME_NAMES,
    TRAILING_WINDOW,
    MIN_REGIME_SAMPLES,
    DEFAULT_GLOBAL_HIT_RATE,
    TREND_BOOST_THRESHOLD,
    TREND_BOOST_FACTOR,
    CRISIS_SCALE_FLOOR,
)


def _make_hit_rates(rates=None, counts=None, global_rate=0.52, total=500):
    """Helper to create RegimeHitRates."""
    if rates is None:
        rates = {r: 0.52 for r in REGIME_NAMES}
    if counts is None:
        counts = {r: 100 for r in REGIME_NAMES}
    return RegimeHitRates(rates=rates, counts=counts,
                          global_rate=global_rate, total_count=total)


def _generate_regime_data(n=1000, seed=42):
    """Generate synthetic multi-regime data."""
    rng = np.random.default_rng(seed)

    regimes = rng.choice(REGIME_NAMES, n)
    true_probs = np.full(n, 0.52)

    # Trend regimes are more predictable
    for i in range(n):
        if regimes[i] in ("LOW_VOL_TREND", "HIGH_VOL_TREND"):
            true_probs[i] = 0.58
        elif regimes[i] == "CRISIS_JUMP":
            true_probs[i] = 0.45
        elif regimes[i] in ("LOW_VOL_RANGE", "HIGH_VOL_RANGE"):
            true_probs[i] = 0.51

    predictions = true_probs + rng.normal(0, 0.05, n)
    predictions = np.clip(predictions, 0.01, 0.99)
    outcomes = (rng.random(n) < true_probs).astype(float)
    # Convert outcomes to returns-like
    returns = np.where(outcomes > 0.5, rng.uniform(0, 0.03, n), rng.uniform(-0.03, 0, n))

    return predictions, returns, regimes


# ===================================================================
# TestRegimeHitRates
# ===================================================================

class TestRegimeHitRates(unittest.TestCase):
    """Test compute_regime_hit_rates()."""

    def test_returns_result(self):
        preds, rets, regs = _generate_regime_data(500)
        result = compute_regime_hit_rates(preds, rets, regs)
        self.assertIsInstance(result, RegimeHitRates)

    def test_all_regimes_present(self):
        preds, rets, regs = _generate_regime_data(500)
        result = compute_regime_hit_rates(preds, rets, regs)
        for r in REGIME_NAMES:
            self.assertIn(r, result.rates)
            self.assertIn(r, result.counts)

    def test_global_rate_reasonable(self):
        preds, rets, regs = _generate_regime_data(1000)
        result = compute_regime_hit_rates(preds, rets, regs)
        self.assertGreater(result.global_rate, 0.3)
        self.assertLess(result.global_rate, 0.8)

    def test_rates_in_range(self):
        preds, rets, regs = _generate_regime_data(1000)
        result = compute_regime_hit_rates(preds, rets, regs)
        for r in REGIME_NAMES:
            self.assertGreaterEqual(result.rates[r], 0.0)
            self.assertLessEqual(result.rates[r], 1.0)

    def test_trailing_window_respected(self):
        preds, rets, regs = _generate_regime_data(1000)
        result = compute_regime_hit_rates(preds, rets, regs, trailing_window=100)
        self.assertEqual(result.total_count, 100)

    def test_small_regime_uses_global(self):
        """If a regime has fewer than MIN_REGIME_SAMPLES, use global rate."""
        n = 200
        rng = np.random.default_rng(42)
        preds = rng.uniform(0.4, 0.6, n)
        rets = rng.normal(0, 0.01, n)
        regs = np.array(["LOW_VOL_TREND"] * n)
        # Only LOW_VOL_TREND has data; others should fallback
        result = compute_regime_hit_rates(preds, rets, regs)
        # CRISIS_JUMP should use global rate
        self.assertEqual(result.counts["CRISIS_JUMP"], 0)
        self.assertAlmostEqual(result.rates["CRISIS_JUMP"], result.global_rate)


# ===================================================================
# TestRegimeConfidenceScale
# ===================================================================

class TestRegimeConfidenceScale(unittest.TestCase):
    """Test regime_confidence_scale()."""

    def test_returns_result(self):
        hr = _make_hit_rates()
        result = regime_confidence_scale(0.7, "LOW_VOL_TREND", hr)
        self.assertIsInstance(result, RegimeConfidenceResult)

    def test_neutral_when_rate_equals_global(self):
        """If regime rate == global rate, scale ~1."""
        hr = _make_hit_rates(
            rates={r: 0.52 for r in REGIME_NAMES},
            global_rate=0.52,
        )
        result = regime_confidence_scale(0.7, "LOW_VOL_RANGE", hr)
        self.assertAlmostEqual(result.adjusted_confidence, 0.7, places=2)

    def test_boost_when_regime_better(self):
        """If regime hit rate > global, confidence increases."""
        hr = _make_hit_rates(
            rates={"LOW_VOL_TREND": 0.60, **{r: 0.52 for r in REGIME_NAMES if r != "LOW_VOL_TREND"}},
            global_rate=0.52,
        )
        result = regime_confidence_scale(0.5, "LOW_VOL_TREND", hr)
        self.assertGreater(result.adjusted_confidence, 0.5)

    def test_reduce_when_regime_worse(self):
        """If regime hit rate < global, confidence decreases."""
        hr = _make_hit_rates(
            rates={"CRISIS_JUMP": 0.40, **{r: 0.55 for r in REGIME_NAMES if r != "CRISIS_JUMP"}},
            global_rate=0.55,
        )
        result = regime_confidence_scale(0.7, "CRISIS_JUMP", hr)
        self.assertLess(result.adjusted_confidence, 0.7)

    def test_trend_boost_applied(self):
        """Trend regime with hit_rate > 55% gets additional boost."""
        hr = _make_hit_rates(
            rates={"LOW_VOL_TREND": 0.58, **{r: 0.52 for r in REGIME_NAMES if r != "LOW_VOL_TREND"}},
            global_rate=0.52,
        )
        result = regime_confidence_scale(0.5, "LOW_VOL_TREND", hr)
        # Scale should be (0.58/0.52) * 1.05 = ~1.17
        expected = 0.5 * (0.58 / 0.52) * TREND_BOOST_FACTOR
        self.assertAlmostEqual(result.adjusted_confidence, expected, places=3)

    def test_crisis_floor(self):
        """Crisis regime confidence should not drop below floor."""
        hr = _make_hit_rates(
            rates={"CRISIS_JUMP": 0.20, **{r: 0.55 for r in REGIME_NAMES if r != "CRISIS_JUMP"}},
            global_rate=0.55,
        )
        result = regime_confidence_scale(0.7, "CRISIS_JUMP", hr)
        # Without floor: scale = 0.20/0.55 = 0.36
        # With floor: scale = max(0.36, 0.5) = 0.5
        self.assertGreaterEqual(result.scale_factor, CRISIS_SCALE_FLOOR)

    def test_confidence_capped(self):
        """Adjusted confidence should not exceed 1.0."""
        hr = _make_hit_rates(
            rates={"LOW_VOL_TREND": 0.80, **{r: 0.50 for r in REGIME_NAMES if r != "LOW_VOL_TREND"}},
            global_rate=0.50,
        )
        result = regime_confidence_scale(0.9, "LOW_VOL_TREND", hr)
        self.assertLessEqual(result.adjusted_confidence, 1.0)

    def test_confidence_non_negative(self):
        hr = _make_hit_rates(
            rates={"CRISIS_JUMP": 0.01, **{r: 0.50 for r in REGIME_NAMES if r != "CRISIS_JUMP"}},
            global_rate=0.50,
        )
        result = regime_confidence_scale(0.1, "CRISIS_JUMP", hr)
        self.assertGreaterEqual(result.adjusted_confidence, 0.0)

    def test_unknown_regime_uses_global(self):
        hr = _make_hit_rates()
        result = regime_confidence_scale(0.6, "UNKNOWN_REGIME", hr)
        self.assertAlmostEqual(result.scale_factor, 1.0)
        self.assertTrue(result.used_default)

    def test_scale_factor_stored(self):
        hr = _make_hit_rates(
            rates={"HIGH_VOL_TREND": 0.60, **{r: 0.50 for r in REGIME_NAMES if r != "HIGH_VOL_TREND"}},
            global_rate=0.50,
        )
        result = regime_confidence_scale(0.5, "HIGH_VOL_TREND", hr)
        self.assertGreater(result.scale_factor, 1.0)


# ===================================================================
# TestAdjustTimeseries
# ===================================================================

class TestAdjustTimeseries(unittest.TestCase):
    """Test adjust_confidence_timeseries()."""

    def test_output_shape(self):
        preds, rets, regs = _generate_regime_data(300)
        confs = np.full(300, 0.6)
        adjusted = adjust_confidence_timeseries(confs, regs, preds, rets)
        self.assertEqual(len(adjusted), 300)

    def test_early_samples_unchanged(self):
        """Before MIN_REGIME_SAMPLES, confidence should be unchanged."""
        preds, rets, regs = _generate_regime_data(300)
        confs = np.full(300, 0.6)
        adjusted = adjust_confidence_timeseries(confs, regs, preds, rets)
        np.testing.assert_array_equal(adjusted[:MIN_REGIME_SAMPLES], confs[:MIN_REGIME_SAMPLES])

    def test_all_finite(self):
        preds, rets, regs = _generate_regime_data(500)
        confs = np.random.default_rng(42).uniform(0.3, 0.8, 500)
        adjusted = adjust_confidence_timeseries(confs, regs, preds, rets)
        self.assertTrue(np.all(np.isfinite(adjusted)))

    def test_in_range(self):
        preds, rets, regs = _generate_regime_data(500)
        confs = np.random.default_rng(42).uniform(0.3, 0.8, 500)
        adjusted = adjust_confidence_timeseries(confs, regs, preds, rets)
        self.assertTrue(np.all(adjusted >= 0.0))
        self.assertTrue(np.all(adjusted <= 1.0))


# ===================================================================
# TestRegimeECE
# ===================================================================

class TestRegimeECE(unittest.TestCase):
    """Test compute_regime_ece()."""

    def test_returns_all_regimes(self):
        preds, rets, regs = _generate_regime_data(1000)
        labels = (rets > 0).astype(float)
        ece_dict = compute_regime_ece(preds, labels, regs)
        for r in REGIME_NAMES:
            self.assertIn(r, ece_dict)

    def test_ece_non_negative(self):
        preds, rets, regs = _generate_regime_data(1000)
        labels = (rets > 0).astype(float)
        ece_dict = compute_regime_ece(preds, labels, regs)
        for r, ece in ece_dict.items():
            if not np.isnan(ece):
                self.assertGreaterEqual(ece, 0.0)

    def test_small_regime_nan(self):
        """Regime with too few samples should return NaN."""
        n = 100
        regs = np.array(["LOW_VOL_TREND"] * n)
        probs = np.random.default_rng(42).uniform(0.3, 0.7, n)
        labels = np.random.default_rng(42).integers(0, 2, n).astype(float)
        ece_dict = compute_regime_ece(probs, labels, regs)
        # Only LOW_VOL_TREND has data
        self.assertTrue(np.isnan(ece_dict["CRISIS_JUMP"]))


# ===================================================================
# TestConstants
# ===================================================================

class TestConstants(unittest.TestCase):
    """Test constant values."""

    def test_regime_names_count(self):
        self.assertEqual(len(REGIME_NAMES), 5)

    def test_trailing_window(self):
        self.assertEqual(TRAILING_WINDOW, 252)

    def test_min_regime_samples(self):
        self.assertEqual(MIN_REGIME_SAMPLES, 20)

    def test_trend_boost(self):
        self.assertGreater(TREND_BOOST_FACTOR, 1.0)

    def test_crisis_floor(self):
        self.assertGreater(CRISIS_SCALE_FLOOR, 0.0)
        self.assertLess(CRISIS_SCALE_FLOOR, 1.0)


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_zero_confidence(self):
        hr = _make_hit_rates()
        result = regime_confidence_scale(0.0, "LOW_VOL_TREND", hr)
        self.assertAlmostEqual(result.adjusted_confidence, 0.0)

    def test_one_confidence(self):
        hr = _make_hit_rates()
        result = regime_confidence_scale(1.0, "LOW_VOL_TREND", hr)
        self.assertLessEqual(result.adjusted_confidence, 1.0)

    def test_all_same_regime(self):
        n = 300
        rng = np.random.default_rng(42)
        regs = np.array(["CRISIS_JUMP"] * n)
        preds = rng.uniform(0.3, 0.7, n)
        rets = rng.normal(0, 0.01, n)
        confs = np.full(n, 0.5)
        adjusted = adjust_confidence_timeseries(confs, regs, preds, rets)
        self.assertTrue(np.all(np.isfinite(adjusted)))

    def test_single_observation_per_regime(self):
        n = 5
        regs = np.array(REGIME_NAMES)
        preds = np.array([0.6, 0.4, 0.5, 0.55, 0.45])
        rets = np.array([0.01, -0.01, 0.005, -0.005, 0.002])
        result = compute_regime_hit_rates(preds, rets, regs)
        # All regimes have only 1 sample -> should use global
        for r in REGIME_NAMES:
            self.assertEqual(result.counts[r], 1)


if __name__ == "__main__":
    unittest.main()
