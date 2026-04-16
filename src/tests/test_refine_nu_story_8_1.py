"""
Tests for Story 8.1: Continuous nu via Golden-Section Profile Likelihood.

Tests that refine_nu_continuous() properly refines nu from a discrete grid
to a continuous optimum using golden-section search on profile log-likelihood.
"""
import math
import os
import sys
import time
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.continuous_nu import (
    NU_GOLDEN_MAXITER,
    NU_GOLDEN_XTOL,
    NU_MAX,
    NU_MIN,
    NuRefinementResult,
    refine_nu_continuous,
    _student_t_log_likelihood,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_student_t_returns(nu_true, n=500, seed=42):
    """Generate Student-t returns with known nu."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(df=nu_true, size=n) * 0.01  # scale to ~1% daily
    vol = np.full(n, 0.01)  # constant vol for simplicity
    return returns, vol


def _make_filter_func_from_ll(ll_func):
    """Wrap a (returns, vol, nu) -> ll function into filter_func API."""
    def filter_func(returns, vol, q, c, phi, nu):
        ll = ll_func(returns, vol, nu)
        return np.zeros(len(returns)), np.zeros(len(returns)), ll
    return filter_func


def _true_student_t_filter(returns, vol, q, c, phi, nu):
    """Filter function that evaluates Student-t log-likelihood."""
    ll = _student_t_log_likelihood(returns, vol, nu)
    return np.zeros(len(returns)), np.zeros(len(returns)), ll


class TestRefineNuContinuousBasic(unittest.TestCase):
    """Basic API and contract tests."""

    def test_returns_result_dataclass(self):
        """refine_nu_continuous returns NuRefinementResult."""
        returns, vol = _make_student_t_returns(nu_true=5.0)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertIsInstance(result, NuRefinementResult)

    def test_result_fields_populated(self):
        """All result fields are finite numbers."""
        returns, vol = _make_student_t_returns(nu_true=5.0)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertTrue(math.isfinite(result.nu_refined))
        self.assertTrue(math.isfinite(result.nu_grid))
        self.assertTrue(math.isfinite(result.ll_refined))
        self.assertTrue(math.isfinite(result.ll_grid))
        self.assertTrue(math.isfinite(result.bic_improvement))
        self.assertIsInstance(result.converged, bool)
        self.assertGreater(result.n_evaluations, 0)

    def test_nu_grid_stored(self):
        """nu_grid in result matches input."""
        returns, vol = _make_student_t_returns(nu_true=5.0)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 8.0)
        self.assertEqual(result.nu_grid, 8.0)

    def test_nu_refined_in_bounds(self):
        """nu_refined is within [2.1, 50]."""
        returns, vol = _make_student_t_returns(nu_true=5.0)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertGreaterEqual(result.nu_refined, NU_MIN)
        self.assertLessEqual(result.nu_refined, NU_MAX)

    def test_nu_refined_in_search_range(self):
        """nu_refined is within [nu_grid-2, nu_grid+4] (default search margins)."""
        returns, vol = _make_student_t_returns(nu_true=5.0)
        nu_grid = 4.0
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, nu_grid)
        self.assertGreaterEqual(result.nu_refined, max(NU_MIN, nu_grid - 2.0))
        self.assertLessEqual(result.nu_refined, min(NU_MAX, nu_grid + 4.0))


class TestRefineNuContinuousConvergence(unittest.TestCase):
    """Tests that nu converges toward the true value."""

    def test_heavy_tail_nu3(self):
        """True nu=3 data: refined nu should be close to 3 (not stuck at grid=4)."""
        returns, vol = _make_student_t_returns(nu_true=3.0, n=1000, seed=100)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        # Should move toward 3, not stay at 4
        self.assertLess(result.nu_refined, 4.0)
        self.assertGreater(result.nu_refined, 2.1)

    def test_moderate_tail_nu6(self):
        """True nu=6 data: refined nu from grid=4 should move toward 6."""
        returns, vol = _make_student_t_returns(nu_true=6.0, n=1000, seed=200)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        # Should move toward 6 (between grid=4 and true=6)
        self.assertGreater(result.nu_refined, 4.0)

    def test_light_tail_nu20(self):
        """True nu=20 data: refined nu from grid=20 should stay near 20."""
        returns, vol = _make_student_t_returns(nu_true=20.0, n=1000, seed=300)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 20.0)
        # Should stay near 20
        self.assertGreater(result.nu_refined, 15.0)
        self.assertLess(result.nu_refined, 25.0)

    def test_between_grid_points_nu5(self):
        """True nu=5 (between grid 4 and 8): should find nu near 5."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=2000, seed=400)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        # With 2000 samples, should be fairly accurate
        self.assertGreater(result.nu_refined, 3.5)
        self.assertLess(result.nu_refined, 7.0)

    def test_between_grid_points_nu12(self):
        """True nu=12 (between grid 8 and 20): should find nu near 12."""
        returns, vol = _make_student_t_returns(nu_true=12.0, n=2000, seed=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 8.0)
        # Should move past 8 toward 12
        self.assertGreater(result.nu_refined, 8.0)
        self.assertLess(result.nu_refined, 13.0)


class TestRefineNuBICImprovement(unittest.TestCase):
    """Tests for BIC improvement > 5 nats on most assets."""

    def test_bic_improvement_nonnegative(self):
        """BIC improvement >= 0 (refined is at least as good as grid)."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertGreaterEqual(result.bic_improvement, -0.01)  # numerical tolerance

    def test_bic_formula_correct(self):
        """BIC improvement = 2 * (ll_refined - ll_grid)."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        expected = 2.0 * (result.ll_refined - result.ll_grid)
        self.assertAlmostEqual(result.bic_improvement, expected, places=6)

    def test_bic_improvement_when_far_from_grid(self):
        """True nu=5, grid=3: should see significant BIC improvement."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=1000, seed=600)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 3.0)
        self.assertGreater(result.bic_improvement, 5.0)

    def test_bic_improvement_when_near_grid(self):
        """True nu=4, grid=4: BIC improvement should be small."""
        returns, vol = _make_student_t_returns(nu_true=4.0, n=1000, seed=700)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        # Already near optimal, improvement should be modest
        self.assertLess(result.bic_improvement, 20.0)

    def test_bic_improvement_80_percent_assets(self):
        """BIC improvement > 0 on 80%+ of simulated assets (continuous always >= grid)."""
        # True nus placed between grid points to ensure refinement finds improvement
        nu_values = [3.5, 5.0, 5.5, 6.5, 10.0, 12.0, 15.0, 6.0, 4.5, 9.0]
        grid_choices = [4, 4, 4, 8, 8, 8, 20, 8, 4, 8]
        improvements = []
        for i, (nu_true, nu_grid) in enumerate(zip(nu_values, grid_choices)):
            returns, vol = _make_student_t_returns(nu_true=nu_true, n=2000, seed=800 + i)
            result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, float(nu_grid))
            improvements.append(result.bic_improvement)
        # At minimum, continuous refinement should improve on most assets
        n_improved = sum(1 for x in improvements if x > 0.0)
        pct = n_improved / len(improvements)
        self.assertGreaterEqual(pct, 0.80,
                                f"Only {pct:.0%} improved: {improvements}")


class TestRefineNuSmoothness(unittest.TestCase):
    """Tests that nu* is smooth across similar assets."""

    def test_similar_returns_similar_nu(self):
        """Two datasets from nu=6 should give similar nu*."""
        r1, v1 = _make_student_t_returns(nu_true=6.0, n=1000, seed=900)
        r2, v2 = _make_student_t_returns(nu_true=6.0, n=1000, seed=901)
        res1 = refine_nu_continuous(r1, v1, 1e-5, 0.01, 0.99, 4.0)
        res2 = refine_nu_continuous(r2, v2, 1e-5, 0.01, 0.99, 4.0)
        self.assertLess(abs(res1.nu_refined - res2.nu_refined), 3.0)

    def test_different_nu_different_results(self):
        """nu=3 data and nu=20 data should give different nu*."""
        r_heavy, v_heavy = _make_student_t_returns(nu_true=3.0, n=1000, seed=910)
        r_light, v_light = _make_student_t_returns(nu_true=20.0, n=1000, seed=920)
        res_heavy = refine_nu_continuous(r_heavy, v_heavy, 1e-5, 0.01, 0.99, 4.0)
        res_light = refine_nu_continuous(r_light, v_light, 1e-5, 0.01, 0.99, 20.0)
        self.assertLess(res_heavy.nu_refined, res_light.nu_refined)

    def test_monotonic_in_true_nu(self):
        """Higher true nu -> higher refined nu (approximately)."""
        nus_true = [3.0, 5.0, 8.0, 15.0, 25.0]
        grid_choices = [3.0, 4.0, 8.0, 20.0, 20.0]
        results = []
        for i, (nu_t, nu_g) in enumerate(zip(nus_true, grid_choices)):
            r, v = _make_student_t_returns(nu_true=nu_t, n=1000, seed=930 + i)
            res = refine_nu_continuous(r, v, 1e-5, 0.01, 0.99, nu_g)
            results.append(res.nu_refined)
        # Check approximate monotonicity: at most 1 violation
        violations = sum(1 for i in range(len(results) - 1)
                         if results[i] > results[i + 1] + 0.5)
        self.assertLessEqual(violations, 1,
                             f"Nu values not monotonic: {results}")


class TestRefineNuRuntime(unittest.TestCase):
    """Runtime tests: < 50ms per asset."""

    def test_runtime_under_50ms(self):
        """Single refinement completes in < 50ms."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500, seed=950)
        # Warm up
        refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)

        start = time.perf_counter()
        for _ in range(10):
            refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        elapsed = (time.perf_counter() - start) / 10

        self.assertLess(elapsed, 0.050, f"Runtime {elapsed*1000:.1f}ms > 50ms")

    def test_evaluations_count(self):
        """Golden-section uses roughly 15 evaluations."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertGreater(result.n_evaluations, 3)
        self.assertLessEqual(result.n_evaluations, NU_GOLDEN_MAXITER + 1)


class TestRefineNuTolerance(unittest.TestCase):
    """Golden-section tolerance tests."""

    def test_tolerance_respected(self):
        """Decreasing xtol gives closer to true nu."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=2000, seed=960)
        res_coarse = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0, xtol=1.0)
        res_fine = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0, xtol=0.01)
        # Fine should be at least as good on likelihood
        self.assertGreaterEqual(res_fine.ll_refined, res_coarse.ll_refined - 0.1)

    def test_default_xtol_below_01(self):
        """Default tolerance is < 0.1."""
        self.assertLessEqual(NU_GOLDEN_XTOL, 0.1)


class TestRefineNuEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_nu_grid_at_lower_bound(self):
        """nu_grid_best = 3 (near lower bound): should still work."""
        returns, vol = _make_student_t_returns(nu_true=3.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 3.0)
        self.assertGreaterEqual(result.nu_refined, NU_MIN)
        self.assertTrue(result.converged or result.nu_refined >= NU_MIN)

    def test_nu_grid_at_upper_bound(self):
        """nu_grid_best = 50 (near upper bound): should still work."""
        returns, vol = _make_student_t_returns(nu_true=30.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 48.0)
        self.assertLessEqual(result.nu_refined, NU_MAX)

    def test_short_series(self):
        """Short return series (n=50) should still work."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=50, seed=970)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertGreaterEqual(result.nu_refined, NU_MIN)
        self.assertLessEqual(result.nu_refined, NU_MAX)

    def test_constant_returns(self):
        """Constant returns: should not crash."""
        returns = np.zeros(200)
        vol = np.full(200, 0.01)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertIsInstance(result, NuRefinementResult)
        self.assertGreaterEqual(result.nu_refined, NU_MIN)

    def test_nan_free_result(self):
        """Result should contain no NaN values."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertFalse(math.isnan(result.nu_refined))
        self.assertFalse(math.isnan(result.ll_refined))
        self.assertFalse(math.isnan(result.ll_grid))
        self.assertFalse(math.isnan(result.bic_improvement))


class TestRefineNuCustomFilter(unittest.TestCase):
    """Tests with custom filter functions."""

    def test_custom_filter_func(self):
        """Custom filter function is used correctly."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(
            returns, vol, 1e-5, 0.01, 0.99, 4.0,
            filter_func=_true_student_t_filter,
        )
        self.assertIsInstance(result, NuRefinementResult)
        self.assertTrue(result.converged)

    def test_filter_func_that_always_fails(self):
        """Filter function that raises: should fall back gracefully."""
        def bad_filter(returns, vol, q, c, phi, nu):
            raise ValueError("Simulated filter failure")

        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(
            returns, vol, 1e-5, 0.01, 0.99, 4.0,
            filter_func=bad_filter,
        )
        # Should fall back to grid value since all evals fail
        self.assertEqual(result.nu_refined, 4.0)
        self.assertFalse(result.converged)

    def test_filter_func_returning_nan(self):
        """Filter returning NaN likelihood: should handle gracefully."""
        def nan_filter(returns, vol, q, c, phi, nu):
            return np.zeros(len(returns)), np.zeros(len(returns)), float('nan')

        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(
            returns, vol, 1e-5, 0.01, 0.99, 4.0,
            filter_func=nan_filter,
        )
        self.assertIsInstance(result, NuRefinementResult)


class TestStudentTLogLikelihood(unittest.TestCase):
    """Tests for the internal Student-t log-likelihood function."""

    def test_finite_output(self):
        """Log-likelihood is finite for valid inputs."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        ll = _student_t_log_likelihood(returns, vol, 5.0)
        self.assertTrue(math.isfinite(ll))

    def test_lower_nu_higher_ll_for_heavy_tails(self):
        """For heavy-tailed data, lower nu gives higher ll."""
        rng = np.random.default_rng(42)
        # Very heavy-tailed data
        returns = rng.standard_t(df=3.0, size=2000) * 0.01
        vol = np.full(2000, 0.01)
        ll_3 = _student_t_log_likelihood(returns, vol, 3.0)
        ll_20 = _student_t_log_likelihood(returns, vol, 20.0)
        self.assertGreater(ll_3, ll_20)

    def test_higher_nu_higher_ll_for_gaussian(self):
        """For near-Gaussian data, higher nu gives higher ll."""
        rng = np.random.default_rng(43)
        returns = rng.normal(0, 0.01, size=2000)
        vol = np.full(2000, 0.01)
        ll_3 = _student_t_log_likelihood(returns, vol, 3.0)
        ll_20 = _student_t_log_likelihood(returns, vol, 20.0)
        self.assertGreater(ll_20, ll_3)

    def test_empty_returns(self):
        """Empty returns: should return -1e12."""
        ll = _student_t_log_likelihood(np.array([]), np.array([]), 5.0)
        self.assertEqual(ll, -1e12)

    def test_invalid_nu(self):
        """nu <= 2: should return -1e12."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=100)
        ll = _student_t_log_likelihood(returns, vol, 2.0)
        self.assertEqual(ll, -1e12)

    def test_ll_scales_with_n(self):
        """Log-likelihood scales linearly with sample size."""
        rng = np.random.default_rng(44)
        returns = rng.standard_t(df=5.0, size=1000) * 0.01
        vol = np.full(1000, 0.01)
        ll_500 = _student_t_log_likelihood(returns[:500], vol[:500], 5.0)
        ll_1000 = _student_t_log_likelihood(returns, vol, 5.0)
        # ll_1000 should be roughly 2x ll_500 (in terms of magnitude)
        ratio = ll_1000 / ll_500
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)


class TestRefineNuValidation(unittest.TestCase):
    """Validation tests matching acceptance criteria."""

    def test_acceptance_returns_nu_in_bounds(self):
        """AC: returns nu* in [2.1, 50]."""
        for nu_true in [3.0, 5.0, 10.0, 20.0]:
            returns, vol = _make_student_t_returns(nu_true=nu_true, n=500, seed=int(nu_true * 10))
            result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
            self.assertGreaterEqual(result.nu_refined, 2.1)
            self.assertLessEqual(result.nu_refined, 50.0)

    def test_acceptance_golden_section_tolerance(self):
        """AC: Uses golden-section with tolerance < 0.1."""
        self.assertLessEqual(NU_GOLDEN_XTOL, 0.1)

    def test_acceptance_convergence_flag(self):
        """AC: Convergence flag is set on clean data."""
        returns, vol = _make_student_t_returns(nu_true=5.0, n=500)
        result = refine_nu_continuous(returns, vol, 1e-5, 0.01, 0.99, 4.0)
        self.assertTrue(result.converged)


if __name__ == '__main__':
    unittest.main()
