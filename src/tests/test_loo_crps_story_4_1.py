"""
Tests for Story 4.1: Leave-One-Out CRPS Computation per Model.

Acceptance Criteria:
  - loo_crps_gaussian(mu, sigma, returns) computes LOO-CRPS using Gaussian CDF
  - loo_crps_student_t(mu, sigma, nu, returns) computes LOO-CRPS using Student-t CDF
  - Both Numba-compiled with @njit(cache=True) for speed
  - LOO-CRPS matches properscoring.crps_gaussian to within 1e-8 on Gaussian test case
  - Runtime: < 10ms for 1000-step series (must not bottleneck tuning)
  - Validated on: SPY (Gaussian-like), MSTR (heavy-tailed), GC=F (trend + jumps)
"""
import os
import sys
import time
import unittest
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t as t_dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
for p in [SRC_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration.loo_crps import (
    loo_crps_gaussian,
    loo_crps_student_t,
    loo_crps_gaussian_mean,
    loo_crps_student_t_mean,
)


def _scipy_crps_gaussian(mu, sigma, y):
    """Reference Gaussian CRPS using scipy (Gneiting & Raftery 2005)."""
    n = len(mu)
    out = np.empty(n)
    for i in range(n):
        sig = max(sigma[i], 1e-12)
        z = (y[i] - mu[i]) / sig
        out[i] = sig * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return out


def _scipy_crps_student_t(mu, sigma, y, nu_val):
    """Reference Student-t CRPS using scipy (Thorarinsdottir & Gneiting 2010).

    g(nu) must be computed via numerical quadrature (the closed-form Beta
    approach is known to be incorrect -- see signals_calibration_numba.py).
    """
    n = len(mu)
    out = np.empty(n)
    # g(nu) via trapezoidal quadrature: g = 2 * integral of x * F(x) * f(x) dx
    L = min(30.0, max(10.0, 4.0 * np.sqrt(nu_val / max(nu_val - 2.0, 0.1))))
    n_quad = 500
    h = 2.0 * L / n_quad
    total = 0.0
    for i in range(n_quad + 1):
        x = -L + i * h
        fx = t_dist.pdf(x, nu_val)
        Fx = t_dist.cdf(x, nu_val)
        val = x * Fx * fx
        if i == 0 or i == n_quad:
            total += 0.5 * val
        else:
            total += val
    g_nu = 2.0 * total * h

    for i in range(n):
        sig = max(sigma[i], 1e-12)
        z = (y[i] - mu[i]) / sig
        cdf_z = t_dist.cdf(z, nu_val)
        pdf_z = t_dist.pdf(z, nu_val)
        out[i] = sig * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z * (nu_val + z * z) / (nu_val - 1.0) - g_nu)
    return out


class TestGaussianCRPS(unittest.TestCase):
    """Gaussian LOO-CRPS tests."""

    def test_output_shape(self):
        """Output shape matches input."""
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02
        returns = np.random.randn(n) * 0.02
        out = loo_crps_gaussian(mu, sigma, returns)
        self.assertEqual(out.shape, (n,))

    def test_all_finite(self):
        """No NaN or Inf in output."""
        n = 500
        mu = np.random.randn(n) * 0.001
        sigma = np.abs(np.random.randn(n)) * 0.02 + 0.005
        returns = np.random.randn(n) * 0.02
        out = loo_crps_gaussian(mu, sigma, returns)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_non_negative(self):
        """CRPS is always >= 0."""
        n = 500
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.abs(rng.randn(n)) * 0.02 + 0.005
        returns = rng.randn(n) * 0.02
        out = loo_crps_gaussian(mu, sigma, returns)
        self.assertTrue(np.all(out >= -1e-10), f"Min CRPS: {out.min()}")

    def test_matches_scipy_reference(self):
        """LOO-CRPS matches scipy-based reference within 1e-8."""
        n = 200
        rng = np.random.RandomState(123)
        mu = rng.randn(n) * 0.001
        sigma = np.abs(rng.randn(n)) * 0.02 + 0.005
        returns = rng.randn(n) * 0.02
        our_crps = loo_crps_gaussian(mu, sigma, returns)
        ref_crps = _scipy_crps_gaussian(mu, sigma, returns)
        np.testing.assert_allclose(our_crps, ref_crps, atol=1e-8, rtol=1e-7)

    def test_perfect_forecast_has_low_crps(self):
        """When mu == returns, CRPS should be small."""
        n = 100
        returns = np.random.randn(n) * 0.01
        sigma = np.ones(n) * 0.01
        out = loo_crps_gaussian(returns, sigma, returns)
        mean_crps = np.mean(out)
        # Expected: sigma * (2*phi(0) - 1/sqrt(pi)) = sigma * (2*0.3989 - 0.5642) = sigma * 0.2337
        expected = 0.01 * (2.0 * norm.pdf(0) - 1.0 / np.sqrt(np.pi))
        self.assertAlmostEqual(mean_crps, expected, delta=0.001)

    def test_wider_sigma_increases_crps(self):
        """CRPS increases with sigma when mean is correct."""
        n = 500
        rng = np.random.RandomState(42)
        returns = rng.randn(n) * 0.02
        mu = np.zeros(n)
        crps_tight = np.mean(loo_crps_gaussian(mu, np.ones(n) * 0.02, returns))
        crps_wide = np.mean(loo_crps_gaussian(mu, np.ones(n) * 0.10, returns))
        self.assertGreater(crps_wide, crps_tight)

    def test_mean_function_matches_manual(self):
        """loo_crps_gaussian_mean matches manual mean of loo_crps_gaussian."""
        n = 300
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.ones(n) * 0.02
        returns = rng.randn(n) * 0.02
        mean_val = loo_crps_gaussian_mean(mu, sigma, returns)
        manual = np.mean(loo_crps_gaussian(mu, sigma, returns))
        self.assertAlmostEqual(mean_val, manual, places=10)

    def test_empty_input(self):
        """Empty arrays return empty output."""
        out = loo_crps_gaussian(np.array([]), np.array([]), np.array([]))
        self.assertEqual(len(out), 0)

    def test_single_observation(self):
        """Works with single observation."""
        out = loo_crps_gaussian(
            np.array([0.01]), np.array([0.02]), np.array([0.005])
        )
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isfinite(out[0]))


class TestStudentTCRPS(unittest.TestCase):
    """Student-t LOO-CRPS tests."""

    def test_output_shape(self):
        """Output shape matches input."""
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02
        nu = np.ones(n) * 5.0
        returns = np.random.randn(n) * 0.02
        out = loo_crps_student_t(mu, sigma, nu, returns)
        self.assertEqual(out.shape, (n,))

    def test_all_finite(self):
        """No NaN or Inf in output."""
        n = 500
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.abs(rng.randn(n)) * 0.02 + 0.005
        nu = np.ones(n) * 5.0
        returns = rng.randn(n) * 0.02
        out = loo_crps_student_t(mu, sigma, nu, returns)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_non_negative(self):
        """CRPS is always >= 0."""
        n = 500
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.abs(rng.randn(n)) * 0.02 + 0.005
        nu = np.ones(n) * 5.0
        returns = rng.randn(n) * 0.02
        out = loo_crps_student_t(mu, sigma, nu, returns)
        self.assertTrue(np.all(out >= -1e-10), f"Min CRPS: {out.min()}")

    def test_matches_scipy_reference(self):
        """LOO-CRPS matches scipy-based reference within 1e-4."""
        n = 200
        rng = np.random.RandomState(123)
        mu = rng.randn(n) * 0.001
        sigma = np.abs(rng.randn(n)) * 0.02 + 0.005
        nu_val = 5.0
        nu = np.ones(n) * nu_val
        returns = rng.randn(n) * 0.02
        our_crps = loo_crps_student_t(mu, sigma, nu, returns)
        ref_crps = _scipy_crps_student_t(mu, sigma, returns, nu_val)
        np.testing.assert_allclose(our_crps, ref_crps, atol=1e-4, rtol=1e-3)

    def test_converges_to_gaussian_at_high_nu(self):
        """Student-t CRPS should converge to Gaussian CRPS as nu -> infinity."""
        n = 300
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.ones(n) * 0.02
        returns = rng.randn(n) * 0.02
        nu_high = np.ones(n) * 100.0
        crps_t = loo_crps_student_t(mu, sigma, nu_high, returns)
        crps_g = loo_crps_gaussian(mu, sigma, returns)
        np.testing.assert_allclose(crps_t, crps_g, atol=0.001, rtol=0.05)

    def test_heavier_tails_increase_crps(self):
        """Lower nu (heavier tails) gives higher CRPS for Gaussian data."""
        n = 500
        rng = np.random.RandomState(42)
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02
        returns = rng.randn(n) * 0.02  # Gaussian data
        crps_light = np.mean(loo_crps_student_t(mu, sigma, np.ones(n) * 20.0, returns))
        crps_heavy = np.mean(loo_crps_student_t(mu, sigma, np.ones(n) * 3.0, returns))
        self.assertGreater(crps_heavy, crps_light)

    def test_varying_nu(self):
        """Works with different nu values per observation."""
        n = 100
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.ones(n) * 0.02
        nu = rng.uniform(3.0, 15.0, n)
        returns = rng.randn(n) * 0.02
        out = loo_crps_student_t(mu, sigma, nu, returns)
        self.assertEqual(len(out), n)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_nu_clamp_at_2(self):
        """Nu < 2.01 is clamped to 2.01 (variance must exist for CRPS)."""
        n = 50
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02
        nu = np.ones(n) * 1.5  # Below threshold
        returns = np.random.randn(n) * 0.02
        out = loo_crps_student_t(mu, sigma, nu, returns)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_mean_function_matches_manual(self):
        """loo_crps_student_t_mean matches manual mean."""
        n = 300
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.ones(n) * 0.02
        nu = np.ones(n) * 5.0
        returns = rng.randn(n) * 0.02
        mean_val = loo_crps_student_t_mean(mu, sigma, nu, returns)
        manual = np.mean(loo_crps_student_t(mu, sigma, nu, returns))
        self.assertAlmostEqual(mean_val, manual, places=10)

    def test_multiple_nu_values_from_grid(self):
        """Validate with different nu from the actual tuning grid."""
        n = 200
        rng = np.random.RandomState(42)
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02
        returns = rng.randn(n) * 0.02

        nu_grid = [2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
        crps_by_nu = []
        for nu_val in nu_grid:
            nu = np.ones(n) * nu_val
            crps_mean = np.mean(loo_crps_student_t(mu, sigma, nu, returns))
            crps_by_nu.append(crps_mean)
            self.assertTrue(np.isfinite(crps_mean), f"NaN at nu={nu_val}")
        # All should be positive
        self.assertTrue(all(c > 0 for c in crps_by_nu))


class TestPerformance(unittest.TestCase):
    """Runtime tests: < 10ms for 1000-step series."""

    def test_gaussian_runtime_1000(self):
        """Gaussian LOO-CRPS < 10ms for 1000 observations."""
        n = 1000
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.ones(n) * 0.02
        returns = rng.randn(n) * 0.02

        # Warmup JIT
        _ = loo_crps_gaussian(mu[:10], sigma[:10], returns[:10])

        start = time.perf_counter()
        for _ in range(10):
            _ = loo_crps_gaussian(mu, sigma, returns)
        elapsed = (time.perf_counter() - start) / 10.0

        self.assertLess(elapsed, 0.010, f"Gaussian LOO-CRPS took {elapsed*1000:.1f}ms")

    def test_student_t_runtime_1000(self):
        """Student-t LOO-CRPS < 10ms for 1000 observations (uniform nu)."""
        n = 1000
        rng = np.random.RandomState(42)
        mu = rng.randn(n) * 0.001
        sigma = np.ones(n) * 0.02
        nu = np.ones(n) * 5.0
        returns = rng.randn(n) * 0.02

        # Warmup JIT
        _ = loo_crps_student_t(mu[:10], sigma[:10], nu[:10], returns[:10])

        start = time.perf_counter()
        for _ in range(10):
            _ = loo_crps_student_t(mu, sigma, nu, returns)
        elapsed = (time.perf_counter() - start) / 10.0

        self.assertLess(elapsed, 0.010, f"Student-t LOO-CRPS took {elapsed*1000:.1f}ms")


class TestNumbaCompilation(unittest.TestCase):
    """Verify functions are properly Numba-compiled."""

    def test_gaussian_is_njit(self):
        """loo_crps_gaussian should be a Numba dispatcher."""
        from numba.core.registry import CPUDispatcher
        self.assertIsInstance(loo_crps_gaussian, CPUDispatcher)

    def test_student_t_is_njit(self):
        """loo_crps_student_t should be a Numba dispatcher."""
        from numba.core.registry import CPUDispatcher
        self.assertIsInstance(loo_crps_student_t, CPUDispatcher)

    def test_gaussian_mean_is_njit(self):
        """loo_crps_gaussian_mean should be a Numba dispatcher."""
        from numba.core.registry import CPUDispatcher
        self.assertIsInstance(loo_crps_gaussian_mean, CPUDispatcher)

    def test_student_t_mean_is_njit(self):
        """loo_crps_student_t_mean should be a Numba dispatcher."""
        from numba.core.registry import CPUDispatcher
        self.assertIsInstance(loo_crps_student_t_mean, CPUDispatcher)


class TestGaussianReference(unittest.TestCase):
    """Match properscoring.crps_gaussian formula to within 1e-8."""

    def test_exact_match_standard_cases(self):
        """Exact match against hand-computed Gaussian CRPS."""
        # z=0: CRPS = sigma * (2*phi(0) - 1/sqrt(pi)) = sigma * 0.23369
        mu = np.array([0.0])
        sigma = np.array([1.0])
        y = np.array([0.0])
        expected = 1.0 * (2.0 * norm.pdf(0) - 1.0 / np.sqrt(np.pi))
        out = loo_crps_gaussian(mu, sigma, y)
        self.assertAlmostEqual(out[0], expected, places=8)

    def test_z_positive(self):
        """Correct CRPS when z > 0."""
        mu = np.array([0.0])
        sigma = np.array([1.0])
        y = np.array([1.0])
        z = 1.0
        expected = 1.0 * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        out = loo_crps_gaussian(mu, sigma, y)
        self.assertAlmostEqual(out[0], expected, places=8)

    def test_z_negative(self):
        """Correct CRPS when z < 0."""
        mu = np.array([0.0])
        sigma = np.array([1.0])
        y = np.array([-2.0])
        z = -2.0
        expected = 1.0 * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        out = loo_crps_gaussian(mu, sigma, y)
        self.assertAlmostEqual(out[0], expected, places=8)

    def test_large_batch_against_scipy(self):
        """1000 random cases match scipy reference within 1e-8."""
        rng = np.random.RandomState(999)
        n = 1000
        mu = rng.randn(n)
        sigma = np.abs(rng.randn(n)) + 0.1
        y = rng.randn(n)
        our = loo_crps_gaussian(mu, sigma, y)
        ref = _scipy_crps_gaussian(mu, sigma, y)
        np.testing.assert_allclose(our, ref, atol=1e-8, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
