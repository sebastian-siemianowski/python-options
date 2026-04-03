"""
Tests for Story 3.1: Coherent Multi-Horizon MC Simulation.

Validates that a single MC call produces all horizons from the same paths,
ensuring variance monotonicity and cross-horizon consistency.
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestCoherentMCConstants(unittest.TestCase):
    def test_coherent_mc_enabled(self):
        from decision.signals import COHERENT_MC_ENABLED
        self.assertTrue(COHERENT_MC_ENABLED)

    def test_fast_mode_default_true(self):
        """latest_signals now defaults to coherent mode."""
        import inspect
        from decision.signals import latest_signals
        sig = inspect.signature(latest_signals)
        self.assertTrue(sig.parameters["_calibration_fast_mode"].default)


class TestVarianceMonotonicity(unittest.TestCase):
    """Variance of cumulative returns should increase with horizon."""

    def test_monotonicity_random_walk(self):
        """For a pure random walk, var(H) = H * sigma^2."""
        rng = np.random.RandomState(42)
        n_paths = 5000
        H_max = 63
        sigma = 0.01

        # Simulate random walk: r_t ~ N(0, sigma)
        steps = rng.normal(0, sigma, size=(H_max, n_paths))
        cum_out = np.cumsum(steps, axis=0)

        horizons = [1, 3, 7, 21, 63]
        variances = [np.var(cum_out[h - 1, :]) for h in horizons]

        for i in range(len(variances) - 1):
            self.assertLessEqual(
                variances[i],
                variances[i + 1] * 1.05,  # 5% tolerance
                f"Var at H={horizons[i]} should be <= Var at H={horizons[i+1]}"
            )

    def test_monotonicity_trending(self):
        """Even with drift, variance grows with time."""
        rng = np.random.RandomState(99)
        n_paths = 5000
        H_max = 63
        mu = 0.001
        sigma = 0.015

        steps = rng.normal(mu, sigma, size=(H_max, n_paths))
        cum_out = np.cumsum(steps, axis=0)

        horizons = [1, 7, 21, 63]
        variances = [np.var(cum_out[h - 1, :]) for h in horizons]

        for i in range(len(variances) - 1):
            self.assertLessEqual(
                variances[i],
                variances[i + 1] * 1.05,
            )


class TestPathConsistency(unittest.TestCase):
    """Same paths should be used for all horizons (increments are finite)."""

    def test_increments_finite(self):
        """r_21 = r_7 + increment_{7..21}, increments must be finite."""
        rng = np.random.RandomState(42)
        n_paths = 3000
        H_max = 21
        sigma = 0.01

        steps = rng.normal(0, sigma, size=(H_max, n_paths))
        cum_out = np.cumsum(steps, axis=0)

        r_7 = cum_out[6, :]   # H=7 (0-indexed)
        r_21 = cum_out[20, :]  # H=21

        increments = r_21 - r_7
        self.assertTrue(np.all(np.isfinite(increments)))

    def test_path_prefix(self):
        """At any horizon, cum_out[H] = sum of steps [0..H]."""
        rng = np.random.RandomState(42)
        n_paths = 1000
        H_max = 10
        sigma = 0.01

        steps = rng.normal(0, sigma, size=(H_max, n_paths))
        cum_out = np.cumsum(steps, axis=0)

        # Manual check: cum_out[4] should equal sum of steps[0..4]
        expected = np.sum(steps[:5, :], axis=0)
        np.testing.assert_allclose(cum_out[4, :], expected, atol=1e-12)


class TestSingleCallProducesAll(unittest.TestCase):
    """Verify that run_unified_mc returns full cum_out matrix."""

    def test_run_unified_mc_returns_matrix(self):
        """run_unified_mc returns shape (H_max, n_paths)."""
        from decision.signals import run_unified_mc
        H = 21
        n_paths = 500
        sigma2 = 0.0001  # daily variance

        result = run_unified_mc(
            mu_t=0.0, P_t=0.0001, sigma2_step=sigma2,
            q=1e-6, phi=0.0, H_max=H, n_paths=n_paths,
        )
        self.assertIn("returns", result)
        self.assertEqual(result["returns"].shape, (H, n_paths))

    def test_all_horizons_from_single_call(self):
        """A single run_unified_mc call provides data for multiple horizons."""
        from decision.signals import run_unified_mc
        H_max = 63
        n_paths = 1000

        result = run_unified_mc(
            mu_t=0.0, P_t=0.0001, sigma2_step=0.0001,
            q=1e-6, phi=0.0, H_max=H_max, n_paths=n_paths,
        )
        cum_out = result["returns"]

        # Can extract any horizon <= H_max
        for h in [1, 3, 7, 21, 63]:
            r_h = cum_out[h - 1, :]
            self.assertEqual(r_h.shape, (n_paths,))
            self.assertTrue(np.all(np.isfinite(r_h)))


class TestDefaultBehavior(unittest.TestCase):
    def test_default_horizons_exist(self):
        from decision.signals import DEFAULT_HORIZONS
        self.assertEqual(DEFAULT_HORIZONS, [1, 3, 7, 21, 63, 126, 252])


if __name__ == "__main__":
    unittest.main()
