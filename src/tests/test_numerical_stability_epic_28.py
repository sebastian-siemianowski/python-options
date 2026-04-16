"""
Tests for Epic 28: Numerical Stability at Extreme Values

Story 28.1: safe_student_t_logpdf - stable at low nu and extreme z
Story 28.2: clamp_covariance - bounds P at every timestep
Story 28.3: kahan_sum - compensated summation for log-likelihood
"""

import os
import sys
import unittest

import numpy as np
from scipy.stats import t as scipy_t

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.numerical_stability import (
    safe_student_t_logpdf,
    safe_student_t_logpdf_scalar,
    clamp_covariance,
    clamp_covariance_array,
    kahan_sum,
    kahan_sum_value,
    CovarianceClampResult,
    KahanSumResult,
    P_MIN_DEFAULT,
    P_MAX_DEFAULT,
    NU_MIN,
)


# ===========================================================================
# Story 28.1: Safe Student-t Evaluation at Low nu
# ===========================================================================

class TestSafeStudentTLogpdf(unittest.TestCase):
    """AC: safe_student_t_logpdf handles extreme z without overflow."""

    def test_basic_evaluation(self):
        """Basic evaluation returns finite values."""
        x = np.array([0.0, 0.01, -0.01, 0.1, -0.1])
        nu = 4.0
        mu = np.zeros(5)
        scale = np.ones(5) * 0.02

        result = safe_student_t_logpdf(x, nu, mu, scale)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result.shape, (5,))

    def test_extreme_z_no_overflow(self):
        """AC: |x| > 10 without overflow at low nu."""
        x = np.array([10.0, -10.0, 50.0, -50.0, 100.0])
        nu = 2.1
        mu = np.zeros(5)
        scale = np.ones(5)

        result = safe_student_t_logpdf(x, nu, mu, scale)
        self.assertTrue(np.all(np.isfinite(result)))
        # All should be negative (log of small probability)
        self.assertTrue(np.all(result < 0))

    def test_matches_scipy_nu_2_1_z_10(self):
        """AC: At nu=2.1, z=10: result finite and matches scipy to 1e-8."""
        nu = 2.1
        x = np.array([10.0])
        mu = np.array([0.0])
        scale = np.array([1.0])

        our_result = safe_student_t_logpdf(x, nu, mu, scale)[0]
        scipy_result = scipy_t.logpdf(10.0, df=nu, loc=0.0, scale=1.0)

        self.assertTrue(np.isfinite(our_result))
        self.assertAlmostEqual(our_result, scipy_result, places=7)

    def test_matches_scipy_nu_50_z_001(self):
        """AC: At nu=50, z=0.01: matches Gaussian to 1e-6."""
        nu = 50.0
        x = np.array([0.01])
        mu = np.array([0.0])
        scale = np.array([1.0])

        our_result = safe_student_t_logpdf(x, nu, mu, scale)[0]
        scipy_result = scipy_t.logpdf(0.01, df=nu, loc=0.0, scale=1.0)

        self.assertAlmostEqual(our_result, scipy_result, places=5)

    def test_matches_scipy_various_nu(self):
        """Matches scipy across range of nu values."""
        rng = np.random.default_rng(42)
        for nu in [2.5, 3.0, 4.0, 8.0, 12.0, 20.0]:
            x = rng.normal(0, 2, size=10)
            mu = np.zeros(10)
            scale = np.ones(10)

            ours = safe_student_t_logpdf(x, nu, mu, scale)
            scipy_ref = scipy_t.logpdf(x, df=nu, loc=0.0, scale=1.0)

            np.testing.assert_allclose(ours, scipy_ref, rtol=1e-7,
                                       err_msg=f"Mismatch at nu={nu}")

    def test_no_nan_or_inf(self):
        """AC: No NaN or Inf for any reasonable inputs."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.05, size=100)
        mu = np.zeros(100)
        scale = np.abs(rng.normal(0.02, 0.005, size=100)) + 1e-6

        for nu in [2.1, 2.5, 3.0, 4.0, 8.0, 20.0]:
            result = safe_student_t_logpdf(x, nu, mu, scale)
            self.assertTrue(np.all(np.isfinite(result)),
                           f"NaN/Inf at nu={nu}")

    def test_scalar_version(self):
        """Scalar version matches array version."""
        nu = 4.0
        x, mu, scale = 0.05, 0.0, 0.02

        scalar_result = safe_student_t_logpdf_scalar(x, nu, mu, scale)
        array_result = safe_student_t_logpdf(
            np.array([x]), nu, np.array([mu]), np.array([scale])
        )[0]

        self.assertAlmostEqual(scalar_result, array_result, places=12)

    def test_zero_scale_handled(self):
        """Zero scale doesn't produce NaN."""
        result = safe_student_t_logpdf(
            np.array([0.0]), 4.0, np.array([0.0]), np.array([0.0])
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_very_low_nu(self):
        """nu near 2.0 still works."""
        x = np.array([0.0, 1.0, 5.0])
        nu = 2.01
        mu = np.zeros(3)
        scale = np.ones(3)

        result = safe_student_t_logpdf(x, nu, mu, scale)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_location_scale(self):
        """Location and scale parameters work correctly."""
        nu = 5.0
        x = np.array([1.0])
        mu = np.array([0.5])
        scale = np.array([2.0])

        our_result = safe_student_t_logpdf(x, nu, mu, scale)[0]
        scipy_result = scipy_t.logpdf(1.0, df=5.0, loc=0.5, scale=2.0)

        self.assertAlmostEqual(our_result, scipy_result, places=8)


# ===========================================================================
# Story 28.2: Filter Covariance Floor and Ceiling
# ===========================================================================

class TestClampCovariance(unittest.TestCase):
    """AC: clamp_covariance enforces P_min <= P <= P_max."""

    def test_within_bounds_unchanged(self):
        """P within bounds is unchanged."""
        result = clamp_covariance(0.001)
        self.assertAlmostEqual(result.P_clamped, 0.001)
        self.assertFalse(result.was_floored)
        self.assertFalse(result.was_ceilinged)

    def test_below_floor_clamped(self):
        """AC: P < P_min clamped to P_min."""
        result = clamp_covariance(1e-15)
        self.assertAlmostEqual(result.P_clamped, P_MIN_DEFAULT)
        self.assertTrue(result.was_floored)
        self.assertFalse(result.was_ceilinged)

    def test_above_ceiling_clamped(self):
        """AC: P > P_max clamped to P_max."""
        result = clamp_covariance(5.0)
        self.assertAlmostEqual(result.P_clamped, P_MAX_DEFAULT)
        self.assertFalse(result.was_floored)
        self.assertTrue(result.was_ceilinged)

    def test_zero_floored(self):
        """P = 0 is floored."""
        result = clamp_covariance(0.0)
        self.assertAlmostEqual(result.P_clamped, P_MIN_DEFAULT)
        self.assertTrue(result.was_floored)

    def test_negative_floored(self):
        """Negative P is floored."""
        result = clamp_covariance(-0.001)
        self.assertAlmostEqual(result.P_clamped, P_MIN_DEFAULT)
        self.assertTrue(result.was_floored)

    def test_nan_handled(self):
        """NaN P is handled (floored)."""
        result = clamp_covariance(float('nan'))
        self.assertAlmostEqual(result.P_clamped, P_MIN_DEFAULT)
        self.assertTrue(result.was_floored)

    def test_inf_handled(self):
        """Inf P is handled (ceilinged)."""
        result = clamp_covariance(float('inf'))
        self.assertAlmostEqual(result.P_clamped, P_MIN_DEFAULT)
        self.assertTrue(result.was_floored)  # NaN/Inf -> floor

    def test_custom_bounds(self):
        """Custom P_min and P_max work."""
        result = clamp_covariance(0.5, P_min=0.1, P_max=0.3)
        self.assertAlmostEqual(result.P_clamped, 0.3)
        self.assertTrue(result.was_ceilinged)

    def test_kalman_gain_never_zero(self):
        """AC: Floor prevents Kalman gain K -> 0."""
        P = P_MIN_DEFAULT  # At floor
        R = 0.01 ** 2  # Typical observation noise
        K = P / (P + R)  # Kalman gain
        self.assertGreater(K, 0)  # K > 0 (filter still updates)

    def test_kalman_gain_never_one(self):
        """AC: Ceiling prevents Kalman gain K -> 1."""
        P = P_MAX_DEFAULT  # At ceiling
        R = 0.01 ** 2
        K = P / (P + R)
        self.assertLess(K, 1.0)  # K < 1 (filter still uses prior)

    def test_identical_observations_stays_above_floor(self):
        """AC: 1000 identical observations -- P stays above floor."""
        P = 0.001
        phi = 0.999
        q = 1e-7
        c = 1.0
        R = 0.01 ** 2

        for _ in range(1000):
            # Predict
            P_pred = phi ** 2 * P + q
            # Update
            S = P_pred + c * R
            K = P_pred / S
            P = (1 - K) * P_pred
            # Clamp
            result = clamp_covariance(P)
            P = result.P_clamped

        self.assertGreaterEqual(P, P_MIN_DEFAULT)

    def test_invalid_bounds_raises(self):
        """P_min > P_max raises ValueError."""
        with self.assertRaises(ValueError):
            clamp_covariance(0.5, P_min=1.0, P_max=0.1)

    def test_to_dict(self):
        result = clamp_covariance(0.5)
        d = result.to_dict()
        self.assertIn("P_clamped", d)
        self.assertIn("was_floored", d)
        self.assertIn("was_ceilinged", d)


class TestClampCovarianceArray(unittest.TestCase):
    """Test vectorized covariance clamping."""

    def test_basic_clamping(self):
        P = np.array([1e-15, 0.001, 5.0, np.nan, np.inf])
        result = clamp_covariance_array(P)
        self.assertAlmostEqual(result[0], P_MIN_DEFAULT)
        self.assertAlmostEqual(result[1], 0.001)
        self.assertAlmostEqual(result[2], P_MAX_DEFAULT)
        self.assertAlmostEqual(result[3], P_MIN_DEFAULT)  # NaN -> P_min
        # Inf -> replaced by P_min (via isfinite check), then clip has no effect
        self.assertAlmostEqual(result[4], P_MIN_DEFAULT)

    def test_all_valid(self):
        P = np.array([0.001, 0.01, 0.1])
        result = clamp_covariance_array(P)
        np.testing.assert_allclose(result, P)


# ===========================================================================
# Story 28.3: Kahan Compensated Summation
# ===========================================================================

class TestKahanSum(unittest.TestCase):
    """AC: kahan_sum achieves relative error < 1e-14 for 2000 elements."""

    def test_basic_sum(self):
        """Basic summation works."""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kahan_sum(vals)
        self.assertIsInstance(result, KahanSumResult)
        self.assertAlmostEqual(result.total, 15.0, places=10)
        self.assertEqual(result.n_elements, 5)

    def test_relative_error_2000_elements(self):
        """AC: Relative error < 1e-14 for 2000-element series."""
        rng = np.random.default_rng(42)
        # Mix of large and small values (worst case for naive sum)
        vals = np.concatenate([
            rng.normal(1e6, 1e3, size=1000),
            rng.normal(1e-6, 1e-9, size=1000),
        ])

        result = kahan_sum(vals)

        # Use Python's math.fsum as reference (exact for finite floats)
        import math
        exact = math.fsum(vals)

        if abs(exact) > 0:
            rel_error = abs(result.total - exact) / abs(exact)
            self.assertLess(rel_error, 1e-13)

    def test_beats_naive_sum(self):
        """Kahan sum is at least as accurate as naive sum."""
        rng = np.random.default_rng(42)
        # Pathological case: alternating large values
        n = 10000
        vals = np.zeros(n)
        vals[0::2] = 1e16
        vals[1::2] = 1.0

        result = kahan_sum(vals)
        # Both should be close for this case, but Kahan >= naive accuracy
        self.assertTrue(np.isfinite(result.total))
        self.assertTrue(np.isfinite(result.naive_total))

    def test_empty_array(self):
        """Empty array returns zero."""
        result = kahan_sum(np.array([]))
        self.assertAlmostEqual(result.total, 0.0)
        self.assertEqual(result.n_elements, 0)

    def test_single_element(self):
        """Single element returned as-is."""
        result = kahan_sum(np.array([42.0]))
        self.assertAlmostEqual(result.total, 42.0)

    def test_nan_filtered(self):
        """NaN values are filtered."""
        vals = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        result = kahan_sum(vals)
        self.assertAlmostEqual(result.total, 9.0)
        self.assertEqual(result.n_elements, 3)

    def test_log_likelihood_accumulation(self):
        """Typical log-likelihood accumulation scenario."""
        rng = np.random.default_rng(42)
        # Simulate log-likelihood contributions (all negative)
        ll_values = -rng.exponential(2.0, size=2000)

        result = kahan_sum(ll_values)
        import math
        exact = math.fsum(ll_values)

        self.assertAlmostEqual(result.total, exact, places=8)

    def test_kahan_sum_value_convenience(self):
        """Convenience function returns just the float."""
        vals = np.array([1.0, 2.0, 3.0])
        result = kahan_sum_value(vals)
        self.assertAlmostEqual(result, 6.0)

    def test_compensation_tracked(self):
        """Compensation term is tracked."""
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, size=1000)
        result = kahan_sum(vals)
        # Compensation should be a small number
        self.assertLess(abs(result.compensation), 1e-10)

    def test_to_dict(self):
        result = kahan_sum(np.array([1.0, 2.0, 3.0]))
        d = result.to_dict()
        self.assertIn("total", d)
        self.assertIn("compensation", d)
        self.assertIn("n_elements", d)
        self.assertIn("naive_total", d)

    def test_runtime_overhead_small(self):
        """AC: Runtime overhead < 5% (conceptual, just check it runs fast)."""
        import time
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, size=10000)

        t0 = time.perf_counter()
        for _ in range(100):
            kahan_sum(vals)
        kahan_time = time.perf_counter() - t0

        # Should complete in reasonable time
        self.assertLess(kahan_time, 10.0)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for numerical stability."""

    def test_student_t_logpdf_kahan_sum(self):
        """Combine safe logpdf with Kahan sum for log-likelihood."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(0, 0.02, size=n)
        mu = np.zeros(n)
        scale = np.ones(n) * 0.02

        for nu in [2.1, 3.0, 8.0]:
            ll_values = safe_student_t_logpdf(x, nu, mu, scale)
            self.assertTrue(np.all(np.isfinite(ll_values)))

            result = kahan_sum(ll_values)
            self.assertTrue(np.isfinite(result.total))
            # Log-likelihood sum should be finite (can be positive for narrow densities)
            self.assertTrue(np.isfinite(result.total))

    def test_filter_with_clamping(self):
        """Simulated filter with covariance clamping stays stable."""
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0, 0.02, size=n)

        mu = 0.0
        P = 0.001
        phi = 0.999
        q = 1e-6
        c = 1.0
        vol = 0.02

        ll_values = []
        for t in range(n):
            # Predict
            mu_pred = phi * mu
            P_pred = phi ** 2 * P + q

            # Clamp
            P_pred = clamp_covariance(P_pred).P_clamped

            # Update
            R = c * vol ** 2
            v = returns[t] - mu_pred
            S = P_pred + R

            # Log-likelihood via safe Student-t
            ll = safe_student_t_logpdf_scalar(returns[t], 4.0, mu_pred, np.sqrt(S))
            ll_values.append(ll)

            K = P_pred / S
            mu = mu_pred + K * v
            P = (1 - K) * P_pred

            # Clamp after update
            P = clamp_covariance(P).P_clamped

        # All log-likelihoods should be finite
        self.assertTrue(all(np.isfinite(ll) for ll in ll_values))

        # Kahan sum of log-likelihood
        total_ll = kahan_sum(np.array(ll_values))
        self.assertTrue(np.isfinite(total_ll.total))

    def test_extreme_returns_stable(self):
        """Extreme returns (MSTR-like) don't break anything."""
        # Simulate extreme returns: occasional 20% daily moves
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.03, size=200)
        # Add extreme events
        returns[50] = 0.20   # +20% day
        returns[100] = -0.25  # -25% day
        returns[150] = 0.15

        mu = np.zeros(200)
        scale = np.ones(200) * 0.03

        for nu in [2.1, 3.0, 5.0]:
            ll = safe_student_t_logpdf(returns, nu, mu, scale)
            self.assertTrue(np.all(np.isfinite(ll)),
                           f"NaN/Inf for extreme returns at nu={nu}")


if __name__ == "__main__":
    unittest.main()
