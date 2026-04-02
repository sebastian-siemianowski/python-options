"""
Test Story 1.2: Kalman Gain Monitoring and Adaptive Reset.

Validates:
  1. K_t computation post-hoc from P_filtered matches inline computation
  2. Stall detection triggers when K_t < threshold for N consecutive bars
  3. P-inflation reset increases K_t within 5 bars
  4. Maximum 3 resets per 252-bar window constraint
  5. Cooldown period prevents back-to-back resets
  6. Regime change at bar 200 triggers reset
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.signals import (
    K_MIN_THRESHOLD,
    GAIN_STALL_WINDOW,
    RESET_INFLATION_FACTOR,
    MAX_RESETS_PER_WINDOW,
    RESET_COOLDOWN_BARS,
    GAIN_RESET_WINDOW_SIZE,
    _compute_kalman_gain_from_filtered,
    _apply_gain_monitoring_reset,
)


class TestKalmanGainComputation(unittest.TestCase):
    """Test post-hoc K_t computation from P_filtered."""

    def test_gaussian_gain_matches_inline(self):
        """Post-hoc K_t should match inline Kalman gain for Gaussian filter (t>=1)."""
        rng = np.random.RandomState(42)
        T = 200
        sigma = np.full(T, 0.01)
        q = 1e-4
        c = 1.0
        phi = 1.0

        # Run inline filter
        mu_t = 0.0
        P_t = 1e-4
        y = rng.normal(0.001, 0.01, T)
        K_inline = np.zeros(T)
        P_filt = np.zeros(T)

        for t in range(T):
            P_pred = phi * phi * P_t + q
            R_t = c * sigma[t] ** 2
            S_t = P_pred + R_t
            K_t = P_pred / S_t
            mu_t = phi * mu_t + K_t * (y[t] - phi * mu_t)
            P_t = (1 - K_t) * P_pred
            K_inline[t] = K_t
            P_filt[t] = P_t

        # Compute post-hoc
        K_posthoc = _compute_kalman_gain_from_filtered(
            P_filt, sigma, phi, q, c, nu=None
        )

        # t=0 cannot match exactly (post-hoc uses P_filtered[0] not P_init)
        # From t=1 onwards output is exact
        np.testing.assert_allclose(K_posthoc[1:], K_inline[1:], atol=1e-10)

    def test_student_t_gain_has_nu_correction(self):
        """Student-t gain should have nu/(nu+3) correction factor."""
        T = 50
        sigma = np.full(T, 0.01)
        P_filtered = np.full(T, 1e-4)
        q = 1e-4
        c = 1.0
        phi = 1.0
        nu = 4.0

        K_t = _compute_kalman_gain_from_filtered(
            P_filtered, sigma, phi, q, c, nu=nu
        )
        K_gauss = _compute_kalman_gain_from_filtered(
            P_filtered, sigma, phi, q, c, nu=None
        )

        expected_ratio = nu / (nu + 3.0)
        np.testing.assert_allclose(K_t / K_gauss, expected_ratio, atol=1e-10)


class TestGainStallDetection(unittest.TestCase):
    """Test stall detection and P-inflation reset."""

    def _make_stalled_filter(self, T=800, stall_start=100, q=1e-9):
        """Create synthetic filter output that stalls at stall_start.

        Uses T=800 and stall_start=100 so that K_t has ~700 bars to
        decay below K_MIN_THRESHOLD (P decays as ~1/t after q drop).
        """
        rng = np.random.RandomState(123)
        sigma = np.full(T, 0.01)
        y = rng.normal(0.001, 0.01, T)
        phi = 1.0
        c = 1.0

        # Two-phase filter: normal q then tiny q
        mu_t = 0.0
        P_t = 1e-4
        mu_filtered = np.zeros(T)
        P_filtered = np.zeros(T)
        K_gain = np.zeros(T)

        for t in range(T):
            q_used = 1e-4 if t < stall_start else q
            P_pred = phi * phi * P_t + q_used
            R_t = c * sigma[t] ** 2
            S_t = P_pred + R_t
            K_t = P_pred / S_t
            innov = y[t] - phi * mu_t
            mu_t = phi * mu_t + K_t * innov
            P_t = max((1 - K_t) * P_pred, 1e-12)
            mu_filtered[t] = mu_t
            P_filtered[t] = P_t
            K_gain[t] = K_t

        return mu_filtered, P_filtered, K_gain, y, sigma, phi, q, c

    def test_stall_detected_after_regime_change(self):
        """Gain stall should be detected after q drops (regime change)."""
        mu, P, K, y, sigma, phi, q, c = self._make_stalled_filter()

        # Verify K_t eventually stalls below threshold
        # P decays as ~1/t so stall develops gradually; check tail quarter
        tail_quarter = K[len(K) * 3 // 4 :]
        self.assertTrue(
            np.all(tail_quarter < K_MIN_THRESHOLD),
            f"K_t should stall below {K_MIN_THRESHOLD} in tail quarter, "
            f"min={tail_quarter.min():.6f}, max={tail_quarter.max():.6f}"
        )

        result = _apply_gain_monitoring_reset(
            mu, P, K, y, sigma, phi=phi, q=q, c=c, nu=None
        )

        self.assertTrue(result["gain_stall_detected"])
        self.assertGreater(result["reset_count"], 0)

    def test_reset_recovers_gain_within_5_bars(self):
        """After reset, K_t should recover above 0.02 within 5 bars."""
        mu, P, K, y, sigma, phi, q, c = self._make_stalled_filter()

        result = _apply_gain_monitoring_reset(
            mu, P, K, y, sigma, phi=phi, q=q, c=c, nu=None
        )

        if result["reset_count"] > 0:
            for ri in result["reset_indices"]:
                # Check K_t within 5 bars after reset
                post_reset = K[ri:min(ri + 6, len(K))]
                self.assertTrue(
                    np.any(post_reset > 0.02),
                    f"K_t should recover above 0.02 within 5 bars after "
                    f"reset at {ri}, got max={post_reset.max():.6f}"
                )

    def test_no_reset_when_gain_healthy(self):
        """No reset should trigger when K_t is healthy."""
        rng = np.random.RandomState(42)
        T = 300
        sigma = np.full(T, 0.01)
        y = rng.normal(0.001, 0.01, T)
        phi = 1.0
        q = 1e-4  # Healthy q
        c = 1.0

        mu_t = 0.0
        P_t = 1e-4
        mu_filtered = np.zeros(T)
        P_filtered = np.zeros(T)
        K_gain = np.zeros(T)

        for t in range(T):
            P_pred = phi * phi * P_t + q
            R_t = c * sigma[t] ** 2
            S_t = P_pred + R_t
            K_t = P_pred / S_t
            innov = y[t] - phi * mu_t
            mu_t = phi * mu_t + K_t * innov
            P_t = max((1 - K_t) * P_pred, 1e-12)
            mu_filtered[t] = mu_t
            P_filtered[t] = P_t
            K_gain[t] = K_t

        result = _apply_gain_monitoring_reset(
            mu_filtered, P_filtered, K_gain,
            y, sigma, phi=phi, q=q, c=c, nu=None
        )

        self.assertEqual(result["reset_count"], 0)
        self.assertFalse(result["gain_stall_detected"])

    def test_max_resets_per_window_enforced(self):
        """Maximum MAX_RESETS_PER_WINDOW resets in a 252-bar window."""
        # Create a long series with many stall regions
        T = 1000
        rng = np.random.RandomState(789)
        sigma = np.full(T, 0.01)
        y = rng.normal(0.001, 0.01, T)
        phi = 1.0
        c = 1.0
        q_tiny = 1e-10  # Always stalled

        mu_t = 0.0
        P_t = 1e-4
        mu_filtered = np.zeros(T)
        P_filtered = np.zeros(T)
        K_gain = np.zeros(T)

        for t in range(T):
            P_pred = phi * phi * P_t + q_tiny
            R_t = c * sigma[t] ** 2
            S_t = P_pred + R_t
            K_t = P_pred / S_t
            innov = y[t] - phi * mu_t
            mu_t = phi * mu_t + K_t * innov
            P_t = max((1 - K_t) * P_pred, 1e-12)
            mu_filtered[t] = mu_t
            P_filtered[t] = P_t
            K_gain[t] = K_t

        result = _apply_gain_monitoring_reset(
            mu_filtered, P_filtered, K_gain,
            y, sigma, phi=phi, q=q_tiny, c=c, nu=None
        )

        # Count resets in any 252-bar window
        for ri in result["reset_indices"]:
            window_start = max(0, ri - GAIN_RESET_WINDOW_SIZE)
            resets_in_window = sum(
                1 for r in result["reset_indices"]
                if window_start <= r <= ri
            )
            self.assertLessEqual(
                resets_in_window, MAX_RESETS_PER_WINDOW,
                f"More than {MAX_RESETS_PER_WINDOW} resets in 252-bar window "
                f"ending at {ri}"
            )

    def test_cooldown_enforced(self):
        """Resets must be at least RESET_COOLDOWN_BARS apart."""
        T = 1000
        rng = np.random.RandomState(101)
        sigma = np.full(T, 0.01)
        y = rng.normal(0.001, 0.01, T)
        phi = 1.0
        c = 1.0
        q_tiny = 1e-10

        mu_t = 0.0
        P_t = 1e-4
        mu_filtered = np.zeros(T)
        P_filtered = np.zeros(T)
        K_gain = np.zeros(T)

        for t in range(T):
            P_pred = phi * phi * P_t + q_tiny
            R_t = c * sigma[t] ** 2
            S_t = P_pred + R_t
            K_t = P_pred / S_t
            innov = y[t] - phi * mu_t
            mu_t = phi * mu_t + K_t * innov
            P_t = max((1 - K_t) * P_pred, 1e-12)
            mu_filtered[t] = mu_t
            P_filtered[t] = P_t
            K_gain[t] = K_t

        result = _apply_gain_monitoring_reset(
            mu_filtered, P_filtered, K_gain,
            y, sigma, phi=phi, q=q_tiny, c=c, nu=None
        )

        indices = result["reset_indices"]
        for i in range(1, len(indices)):
            gap = indices[i] - indices[i - 1]
            self.assertGreaterEqual(
                gap, RESET_COOLDOWN_BARS,
                f"Resets at {indices[i-1]} and {indices[i]} are only "
                f"{gap} bars apart (minimum {RESET_COOLDOWN_BARS})"
            )

    def test_p_inflation_factor_applied(self):
        """P at reset point should be inflated by RESET_INFLATION_FACTOR."""
        mu, P, K, y, sigma, phi, q, c = self._make_stalled_filter()

        P_before = P.copy()
        result = _apply_gain_monitoring_reset(
            mu, P, K, y, sigma, phi=phi, q=q, c=c, nu=None
        )

        if result["reset_count"] > 0:
            ri = result["reset_indices"][0]
            # P at reset point should be inflated
            expected_P = P_before[ri] * RESET_INFLATION_FACTOR
            self.assertAlmostEqual(
                P[ri], expected_P, places=15,
                msg=f"P at reset point should be inflated by {RESET_INFLATION_FACTOR}x"
            )

    def test_short_series_no_crash(self):
        """Short series should not crash, just return no resets."""
        T = 5
        result = _apply_gain_monitoring_reset(
            np.zeros(T), np.ones(T) * 1e-4, np.zeros(T),
            np.zeros(T), np.ones(T) * 0.01,
            phi=1.0, q=1e-4, c=1.0, nu=None,
        )
        self.assertEqual(result["reset_count"], 0)


class TestGainMonitoringConstants(unittest.TestCase):
    """Test configuration constants are sensible."""

    def test_k_min_threshold_reasonable(self):
        """K_MIN_THRESHOLD should be in a sensible range."""
        self.assertGreater(K_MIN_THRESHOLD, 0)
        self.assertLess(K_MIN_THRESHOLD, 0.1)

    def test_stall_window_reasonable(self):
        """GAIN_STALL_WINDOW should be >= 5 bars."""
        self.assertGreaterEqual(GAIN_STALL_WINDOW, 5)
        self.assertLessEqual(GAIN_STALL_WINDOW, 50)

    def test_inflation_factor_reasonable(self):
        """Inflation factor should be > 1 (otherwise not inflating)."""
        self.assertGreater(RESET_INFLATION_FACTOR, 1.0)
        self.assertLessEqual(RESET_INFLATION_FACTOR, 100.0)

    def test_max_resets_reasonable(self):
        """Max resets should be 1-5 per window."""
        self.assertGreaterEqual(MAX_RESETS_PER_WINDOW, 1)
        self.assertLessEqual(MAX_RESETS_PER_WINDOW, 5)

    def test_cooldown_reasonable(self):
        """Cooldown should be > stall window."""
        self.assertGreater(RESET_COOLDOWN_BARS, GAIN_STALL_WINDOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
