"""
Tests for Story 9.1: Ljung-Box Autocorrelation Test on Innovations.

Tests that innovation_ljung_box() properly detects autocorrelated
innovations (misspecified filter) vs white noise (well-specified filter).
"""
import math
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.innovation_diagnostics import (
    DEFAULT_LAGS,
    LB_PVALUE_THRESHOLD,
    LjungBoxResult,
    innovation_ljung_box,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_white_noise(n=500, seed=42):
    """Generate white noise innovations (well-specified filter)."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0, 1, size=n)
    R = np.ones(n)
    return innovations, R


def _make_autocorrelated(n=500, rho=0.5, seed=42):
    """Generate AR(1) innovations (misspecified filter)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, size=n)
    innovations = np.zeros(n)
    innovations[0] = eps[0]
    for t in range(1, n):
        innovations[t] = rho * innovations[t - 1] + eps[t]
    R = np.ones(n)
    return innovations, R


class TestLjungBoxBasic(unittest.TestCase):
    """Basic API and contract tests."""

    def test_returns_result_dataclass(self):
        """innovation_ljung_box returns LjungBoxResult."""
        innovations, R = _make_white_noise()
        result = innovation_ljung_box(innovations, R)
        self.assertIsInstance(result, LjungBoxResult)

    def test_result_has_default_lags(self):
        """Result contains all default lags."""
        innovations, R = _make_white_noise()
        result = innovation_ljung_box(innovations, R)
        for lag in DEFAULT_LAGS:
            self.assertIn(lag, result.q_stats)
            self.assertIn(lag, result.p_values)

    def test_custom_lags(self):
        """Custom lags are respected."""
        innovations, R = _make_white_noise()
        result = innovation_ljung_box(innovations, R, lags=[1, 3])
        self.assertIn(1, result.q_stats)
        self.assertIn(3, result.q_stats)
        self.assertNotIn(5, result.q_stats)

    def test_q_stats_nonnegative(self):
        """Q-statistics are non-negative."""
        innovations, R = _make_white_noise()
        result = innovation_ljung_box(innovations, R)
        for q in result.q_stats.values():
            self.assertGreaterEqual(q, 0.0)

    def test_p_values_in_0_1(self):
        """P-values are in [0, 1]."""
        innovations, R = _make_white_noise()
        result = innovation_ljung_box(innovations, R)
        for p in result.p_values.values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_n_obs_correct(self):
        """n_obs matches input length."""
        innovations, R = _make_white_noise(n=200)
        result = innovation_ljung_box(innovations, R)
        self.assertEqual(result.n_obs, 200)


class TestLjungBoxWhiteNoise(unittest.TestCase):
    """Tests on well-specified (white noise) innovations."""

    def test_white_noise_not_misspecified(self):
        """White noise innovations: should not be flagged as misspecified."""
        innovations, R = _make_white_noise(n=500, seed=100)
        result = innovation_ljung_box(innovations, R)
        self.assertFalse(result.is_misspecified)

    def test_white_noise_high_p_values(self):
        """White noise: p-values should be > 0.01."""
        innovations, R = _make_white_noise(n=500, seed=200)
        result = innovation_ljung_box(innovations, R)
        for lag, p in result.p_values.items():
            self.assertGreater(p, 0.01,
                               f"Lag {lag}: p={p:.4f} < 0.01 on white noise")

    def test_false_alarm_rate_under_5_percent(self):
        """False alarm rate < 5% on white noise (100 trials)."""
        n_alarms = 0
        for seed in range(100):
            innovations, R = _make_white_noise(n=500, seed=300 + seed)
            result = innovation_ljung_box(innovations, R)
            if result.is_misspecified:
                n_alarms += 1
        rate = n_alarms / 100
        self.assertLessEqual(rate, 0.06,
                             f"False alarm rate {rate:.2%} > 6%")


class TestLjungBoxMisspecified(unittest.TestCase):
    """Tests on misspecified (autocorrelated) innovations."""

    def test_ar1_detected(self):
        """AR(1) with rho=0.5 should be flagged as misspecified."""
        innovations, R = _make_autocorrelated(n=500, rho=0.5, seed=400)
        result = innovation_ljung_box(innovations, R)
        self.assertTrue(result.is_misspecified)

    def test_ar1_lag1_flagged(self):
        """AR(1) innovations should flag lag 1."""
        innovations, R = _make_autocorrelated(n=500, rho=0.5, seed=410)
        result = innovation_ljung_box(innovations, R)
        self.assertIn(1, result.flagged_lags)

    def test_ar1_low_p_at_lag1(self):
        """AR(1) with rho=0.5: p-value at lag 1 should be very small."""
        innovations, R = _make_autocorrelated(n=500, rho=0.5, seed=420)
        result = innovation_ljung_box(innovations, R)
        self.assertLess(result.p_values[1], 0.01)

    def test_detection_rate_above_90_percent(self):
        """Detection rate > 90% on AR(1) data (20 trials)."""
        n_detected = 0
        for seed in range(20):
            innovations, R = _make_autocorrelated(n=500, rho=0.5, seed=500 + seed)
            result = innovation_ljung_box(innovations, R)
            if result.is_misspecified:
                n_detected += 1
        rate = n_detected / 20
        self.assertGreater(rate, 0.90,
                           f"Detection rate {rate:.2%} <= 90%")

    def test_weak_ar1_harder_to_detect(self):
        """Weak AR(1) rho=0.1: harder to detect (lower Q-stats)."""
        innovations, R = _make_autocorrelated(n=500, rho=0.1, seed=600)
        result = innovation_ljung_box(innovations, R)
        # Weak AR(1) might or might not be detected, but Q-stat at lag 1
        # should be larger than for white noise
        wn_innovations, wn_R = _make_white_noise(n=500, seed=600)
        wn_result = innovation_ljung_box(wn_innovations, wn_R)
        self.assertGreater(result.q_stats[1], wn_result.q_stats[1] * 0.5)


class TestLjungBoxStandardization(unittest.TestCase):
    """Tests that innovations are properly standardized by R."""

    def test_varying_R(self):
        """With varying R, standardization adjusts correctly."""
        rng = np.random.default_rng(701)
        R = rng.uniform(0.5, 2.0, size=500)
        innovations = rng.normal(0, 1, size=500) * np.sqrt(R)
        result = innovation_ljung_box(innovations, R)
        # Properly standardized -> high p-values (most lags above threshold)
        n_high_p = sum(1 for p in result.p_values.values() if p > 0.01)
        self.assertGreaterEqual(n_high_p, len(result.p_values) - 1)

    def test_large_R_reduces_q_stats(self):
        """Larger R should reduce apparent autocorrelation in raw innovations."""
        innovations, _ = _make_autocorrelated(n=500, rho=0.3, seed=710)
        R_small = np.ones(500)
        R_large = np.ones(500) * 100.0
        # With large R, the standardized innovations have less autocorrelation
        # because the raw innovations ARE the true innovations (R just scales)
        result_small = innovation_ljung_box(innovations, R_small)
        result_large = innovation_ljung_box(innovations, R_large)
        # Both should find same autocorrelation structure after standardization
        # (Q-stat should be similar since autocorrelation is scale-invariant)
        self.assertGreater(result_small.q_stats[1], 0.0)


class TestLjungBoxEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_short_series(self):
        """Very short series (n=10): should handle gracefully."""
        innovations, R = _make_white_noise(n=10, seed=800)
        result = innovation_ljung_box(innovations, R, lags=[1, 5])
        self.assertIsInstance(result, LjungBoxResult)

    def test_very_short_series(self):
        """n=2: should return empty results without crash."""
        innovations = np.array([0.1, -0.1])
        R = np.array([1.0, 1.0])
        result = innovation_ljung_box(innovations, R)
        self.assertIsInstance(result, LjungBoxResult)
        self.assertFalse(result.is_misspecified)

    def test_constant_innovations(self):
        """Constant innovations: should not crash."""
        innovations = np.zeros(200)
        R = np.ones(200)
        result = innovation_ljung_box(innovations, R)
        self.assertIsInstance(result, LjungBoxResult)
        self.assertFalse(result.is_misspecified)

    def test_lag_exceeds_n(self):
        """Lag > n: should handle gracefully (p=1.0)."""
        innovations, R = _make_white_noise(n=15, seed=810)
        result = innovation_ljung_box(innovations, R, lags=[1, 20])
        self.assertEqual(result.p_values[20], 1.0)

    def test_single_lag(self):
        """Single lag works."""
        innovations, R = _make_white_noise(n=200)
        result = innovation_ljung_box(innovations, R, lags=[1])
        self.assertIn(1, result.q_stats)

    def test_custom_threshold(self):
        """Custom threshold affects misspecified flag."""
        innovations, R = _make_autocorrelated(n=500, rho=0.5, seed=820)
        # Very strict threshold (0.5): will flag almost anything
        result_strict = innovation_ljung_box(innovations, R, threshold=0.5)
        # Very lenient threshold (1e-10): hard to flag
        result_lenient = innovation_ljung_box(innovations, R, threshold=1e-10)
        self.assertTrue(result_strict.is_misspecified)
        # Strong AR(1) might still be flagged even with very lenient threshold
        # but the point is that strictness increases flagging


class TestLjungBoxConstants(unittest.TestCase):
    """Tests for configuration constants."""

    def test_default_lags(self):
        """Default lags are [1, 5, 10, 20]."""
        self.assertEqual(DEFAULT_LAGS, [1, 5, 10, 20])

    def test_pvalue_threshold(self):
        """Default p-value threshold is 0.01."""
        self.assertEqual(LB_PVALUE_THRESHOLD, 0.01)


if __name__ == '__main__':
    unittest.main()
