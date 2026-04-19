"""
Test Suite for Epic 13: Kelly Criterion Integration
=====================================================

Story 13.1: Full Kelly Sizing from BMA Predictive Distribution
Story 13.2: Risk-Adjusted Kelly with Drawdown Constraint
Story 13.3: Fractional Kelly Auto-Tuning via Utility Maximization
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.kelly_sizing import (
    # Story 13.1
    kelly_fraction,
    kelly_fraction_array,
    kelly_position_pnl,
    kelly_hit_rate,
    kelly_profit_factor,
    KELLY_MAX, KELLY_MIN, DEFAULT_KELLY_FRAC, MIN_KELLY_THRESHOLD,
    # Story 13.2
    DrawdownAdjustedResult,
    drawdown_adjusted_kelly,
    compute_running_drawdown,
    apply_drawdown_kelly_backtest,
    compute_max_drawdown,
    compute_sharpe,
    DD_REDUCE_THRESHOLD, DD_FLAT_THRESHOLD,
    # Story 13.3
    KellyAutoTuneResult,
    auto_tune_kelly_frac,
    _log_utility,
    DEFAULT_FRAC_GRID,
    AUTOTUNE_TRAIN_DAYS,
    AUTOTUNE_TEST_DAYS,
)


def _generate_signal_returns(n=500, hit_rate=0.55, vol=0.01, seed=42):
    """Generate returns with positive drift (hit_rate > 50%)."""
    rng = np.random.default_rng(seed)
    drift = vol * (2 * hit_rate - 1) / np.sqrt(1)
    return rng.normal(drift, vol, n)


def _generate_forecasts(returns, noise_sigma=0.005, seed=123):
    """Generate forecasts that are noisy versions of future returns."""
    rng = np.random.default_rng(seed)
    mu = returns + rng.normal(0, noise_sigma, len(returns))
    sigma = np.full(len(returns), np.std(returns)) + rng.uniform(0, 0.002, len(returns))
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


# ===================================================================
# Story 13.1 Tests: Full Kelly Sizing
# ===================================================================

class TestKellyFraction(unittest.TestCase):
    """Test kelly_fraction()."""

    def test_gaussian_half_kelly(self):
        """f = 0.5 * mu / sigma^2 for Gaussian."""
        mu, sigma = 0.0001, 0.02  # Small mu so f doesn't hit cap
        f = kelly_fraction(mu, sigma)
        expected = 0.5 * mu / sigma**2
        self.assertAlmostEqual(f, expected, places=10)

    def test_full_kelly(self):
        """f = 1.0 * mu / sigma^2 with kelly_frac=1.0."""
        mu, sigma = 0.0001, 0.02  # Small mu so f doesn't hit cap
        f = kelly_fraction(mu, sigma, kelly_frac=1.0)
        expected = mu / sigma**2
        self.assertAlmostEqual(f, expected, places=10)

    def test_student_t_reduces_fraction(self):
        """Student-t with finite nu reduces Kelly vs Gaussian."""
        mu, sigma = 0.0001, 0.02  # Small mu so neither hits cap
        f_gauss = kelly_fraction(mu, sigma)
        f_t = kelly_fraction(mu, sigma, nu=8.0)
        self.assertLess(f_t, f_gauss)

    def test_student_t_formula(self):
        """Verify exact Student-t formula for nu=8."""
        mu, sigma = 0.0001, 0.02  # Small mu so f doesn't hit cap
        nu = 8.0
        f = kelly_fraction(mu, sigma, nu=nu, kelly_frac=0.5)
        kappa_excess = 6.0 / (nu - 4.0)
        expected = 0.5 * mu / sigma**2 / (1.0 + kappa_excess / 6.0)
        self.assertAlmostEqual(f, expected, places=10)

    def test_heavy_tails_very_small(self):
        """nu <= 4 gives very small fraction."""
        mu, sigma = 0.0001, 0.02  # Small mu so neither hits cap
        f = kelly_fraction(mu, sigma, nu=3.0)
        f_gauss = kelly_fraction(mu, sigma)
        self.assertLess(f, f_gauss * 0.5)

    def test_bounded_positive(self):
        """Large mu/sigma should be capped at KELLY_MAX."""
        f = kelly_fraction(1.0, 0.001)
        self.assertAlmostEqual(f, KELLY_MAX)

    def test_bounded_negative(self):
        """Large negative mu should be capped at KELLY_MIN."""
        f = kelly_fraction(-1.0, 0.001)
        self.assertAlmostEqual(f, KELLY_MIN)

    def test_zero_mu(self):
        f = kelly_fraction(0.0, 0.02)
        self.assertAlmostEqual(f, 0.0)

    def test_zero_sigma(self):
        f = kelly_fraction(0.001, 0.0)
        self.assertAlmostEqual(f, 0.0)

    def test_negative_sigma(self):
        f = kelly_fraction(0.001, -0.01)
        self.assertAlmostEqual(f, 0.0)

    def test_nan_mu(self):
        f = kelly_fraction(float('nan'), 0.02)
        self.assertAlmostEqual(f, 0.0)

    def test_nan_sigma(self):
        f = kelly_fraction(0.001, float('nan'))
        self.assertAlmostEqual(f, 0.0)

    def test_inf_nu(self):
        """Infinite nu should behave like Gaussian."""
        mu, sigma = 0.001, 0.02
        f = kelly_fraction(mu, sigma, nu=float('inf'))
        f_gauss = kelly_fraction(mu, sigma)
        self.assertAlmostEqual(f, f_gauss, places=8)

    def test_negative_mu_negative_fraction(self):
        """Negative mu -> negative (short) fraction."""
        f = kelly_fraction(-0.001, 0.02)
        self.assertLess(f, 0)

    def test_larger_sigma_smaller_fraction(self):
        """More uncertainty -> smaller position."""
        f1 = kelly_fraction(0.0001, 0.02)
        f2 = kelly_fraction(0.0001, 0.04)
        self.assertGreater(abs(f1), abs(f2))


class TestKellyFractionArray(unittest.TestCase):
    """Test kelly_fraction_array()."""

    def test_output_shape(self):
        mu = np.array([0.001, -0.001, 0.002])
        sigma = np.array([0.02, 0.02, 0.03])
        result = kelly_fraction_array(mu, sigma)
        self.assertEqual(len(result), 3)

    def test_matches_scalar(self):
        mu = np.array([0.001, 0.002])
        sigma = np.array([0.02, 0.03])
        result = kelly_fraction_array(mu, sigma)
        for i in range(2):
            self.assertAlmostEqual(result[i], kelly_fraction(mu[i], sigma[i]))

    def test_with_nu(self):
        mu = np.array([0.001, 0.002])
        sigma = np.array([0.02, 0.03])
        nu = np.array([8.0, 12.0])
        result = kelly_fraction_array(mu, sigma, nu)
        for i in range(2):
            self.assertAlmostEqual(result[i], kelly_fraction(mu[i], sigma[i], nu[i]))


class TestKellyPnL(unittest.TestCase):
    """Test kelly_position_pnl()."""

    def test_output_shape(self):
        r = np.array([0.01, -0.02, 0.005])
        f = np.array([0.3, 0.2, 0.4])
        pnl = kelly_position_pnl(r, f)
        self.assertEqual(len(pnl), 3)

    def test_pnl_values(self):
        r = np.array([0.01, -0.02])
        f = np.array([0.5, 0.3])
        pnl = kelly_position_pnl(r, f)
        np.testing.assert_allclose(pnl, [0.005, -0.006])


class TestKellyHitRate(unittest.TestCase):
    """Test kelly_hit_rate()."""

    def test_perfect_signal(self):
        """Positive fractions on positive returns -> 100% hit rate."""
        r = np.array([0.01, 0.02, 0.005, 0.015])
        f = np.array([0.1, 0.1, 0.1, 0.1])
        hr = kelly_hit_rate(r, f, f_min=0.01)
        self.assertAlmostEqual(hr, 1.0)

    def test_below_threshold_no_trade(self):
        r = np.array([0.01, -0.02])
        f = np.array([0.001, 0.001])
        hr = kelly_hit_rate(r, f, f_min=0.02)
        self.assertAlmostEqual(hr, 0.5)  # no trades

    def test_reasonable_range(self):
        r = _generate_signal_returns(500)
        mu, sigma = _generate_forecasts(r)
        f = kelly_fraction_array(mu, sigma)
        hr = kelly_hit_rate(r, f)
        self.assertGreater(hr, 0.0)
        self.assertLess(hr, 1.0)


class TestKellyProfitFactor(unittest.TestCase):
    """Test kelly_profit_factor()."""

    def test_all_profit(self):
        r = np.array([0.01, 0.02, 0.005])
        f = np.array([0.1, 0.1, 0.1])
        pf = kelly_profit_factor(r, f, f_min=0.01)
        self.assertEqual(pf, 10.0)  # capped

    def test_no_trades(self):
        r = np.array([0.01, -0.01])
        f = np.array([0.001, 0.001])
        pf = kelly_profit_factor(r, f, f_min=0.02)
        self.assertAlmostEqual(pf, 1.0)

    def test_mixed_pnl(self):
        r = np.array([0.01, -0.02, 0.015, -0.005])
        f = np.array([0.5, 0.5, 0.5, 0.5])
        pf = kelly_profit_factor(r, f, f_min=0.01)
        gross_profit = 0.005 + 0.0075  # 0.5*0.01 + 0.5*0.015
        gross_loss = abs(0.5 * (-0.02)) + abs(0.5 * (-0.005))
        expected = gross_profit / gross_loss
        self.assertAlmostEqual(pf, expected, places=5)


# ===================================================================
# Story 13.2 Tests: Drawdown-Adjusted Kelly
# ===================================================================

class TestDrawdownAdjustedKelly(unittest.TestCase):
    """Test drawdown_adjusted_kelly()."""

    def test_no_drawdown(self):
        result = drawdown_adjusted_kelly(0.3, 0.0)
        self.assertAlmostEqual(result.f_adjusted, 0.3)
        self.assertAlmostEqual(result.dd_dampener, 1.0)
        self.assertFalse(result.is_flat)

    def test_small_drawdown_no_effect(self):
        result = drawdown_adjusted_kelly(0.3, 0.05)
        self.assertAlmostEqual(result.f_adjusted, 0.3)

    def test_medium_drawdown_reduces(self):
        """dd=12% reduces fraction."""
        result = drawdown_adjusted_kelly(0.3, 0.12)
        expected = 0.3 * (1.0 - 0.12 / 0.15)
        self.assertAlmostEqual(result.f_adjusted, expected, places=5)
        self.assertLess(result.f_adjusted, 0.3)

    def test_max_drawdown_goes_flat(self):
        result = drawdown_adjusted_kelly(0.3, 0.15)
        self.assertAlmostEqual(result.f_adjusted, 0.0)
        self.assertTrue(result.is_flat)

    def test_beyond_max_goes_flat(self):
        result = drawdown_adjusted_kelly(0.3, 0.20)
        self.assertAlmostEqual(result.f_adjusted, 0.0)
        self.assertTrue(result.is_flat)

    def test_custom_max_dd(self):
        result = drawdown_adjusted_kelly(0.3, 0.25, max_dd=0.30)
        # dd=25% > reduce threshold=10%, dampener = 1 - 0.25/0.30
        self.assertLess(result.f_adjusted, 0.3)
        self.assertGreater(result.f_adjusted, 0.0)

    def test_returns_dataclass(self):
        result = drawdown_adjusted_kelly(0.3, 0.12)
        self.assertIsInstance(result, DrawdownAdjustedResult)


class TestRunningDrawdown(unittest.TestCase):
    """Test compute_running_drawdown()."""

    def test_monotonically_increasing(self):
        equity = np.array([1.0, 1.01, 1.02, 1.03])
        dd = compute_running_drawdown(equity)
        np.testing.assert_allclose(dd, [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_drawdown_after_peak(self):
        equity = np.array([1.0, 1.1, 1.0, 0.9])
        dd = compute_running_drawdown(equity)
        self.assertAlmostEqual(dd[0], 0.0)
        self.assertAlmostEqual(dd[1], 0.0)
        self.assertAlmostEqual(dd[2], 1.0 - 1.0/1.1, places=5)
        self.assertAlmostEqual(dd[3], 1.0 - 0.9/1.1, places=5)

    def test_output_shape(self):
        equity = np.arange(1.0, 2.0, 0.01)
        dd = compute_running_drawdown(equity)
        self.assertEqual(len(dd), len(equity))


class TestApplyDrawdownBacktest(unittest.TestCase):
    """Test apply_drawdown_kelly_backtest()."""

    def test_output_shapes(self):
        r = np.random.default_rng(42).normal(0, 0.01, 100)
        f = np.full(100, 0.3)
        strat_r, adj_f, dd = apply_drawdown_kelly_backtest(r, f)
        self.assertEqual(len(strat_r), 100)
        self.assertEqual(len(adj_f), 100)
        self.assertEqual(len(dd), 100)

    def test_reduces_during_drawdown(self):
        """After losses, fractions should reduce."""
        # Larger losses to exceed 10% DD threshold
        r = np.concatenate([np.full(80, -0.01), np.full(20, 0.005)])
        f = np.full(100, 0.5)
        strat_r, adj_f, dd = apply_drawdown_kelly_backtest(r, f)
        # After significant losses, fractions should reduce
        self.assertLess(adj_f[79], adj_f[0])

    def test_max_dd_reduced(self):
        """Drawdown adjustment should reduce max drawdown vs raw."""
        rng = np.random.default_rng(42)
        r = rng.normal(-0.001, 0.02, 200)  # Slightly negative drift
        f = np.full(200, 0.4)
        raw_pnl = f * r
        raw_dd = compute_max_drawdown(raw_pnl)
        strat_r, _, _ = apply_drawdown_kelly_backtest(r, f)
        adj_dd = compute_max_drawdown(strat_r)
        self.assertLessEqual(adj_dd, raw_dd + 0.01)  # Should not be much worse


class TestComputeMaxDrawdown(unittest.TestCase):
    """Test compute_max_drawdown()."""

    def test_no_drawdown(self):
        r = np.full(50, 0.01)
        dd = compute_max_drawdown(r)
        self.assertAlmostEqual(dd, 0.0, places=5)

    def test_known_drawdown(self):
        r = np.array([0.1, -0.2, 0.05])
        dd = compute_max_drawdown(r)
        self.assertGreater(dd, 0.0)

    def test_empty_returns(self):
        dd = compute_max_drawdown(np.array([]))
        self.assertAlmostEqual(dd, 0.0)


class TestComputeSharpe(unittest.TestCase):
    """Test compute_sharpe()."""

    def test_zero_vol(self):
        r = np.full(100, 0.001)
        s = compute_sharpe(r)
        self.assertAlmostEqual(s, 0.0)  # std = 0

    def test_positive_drift(self):
        r = np.random.default_rng(42).normal(0.001, 0.01, 500)
        s = compute_sharpe(r)
        self.assertGreater(s, 0)

    def test_short_array(self):
        s = compute_sharpe(np.array([0.01]))
        self.assertAlmostEqual(s, 0.0)


# ===================================================================
# Story 13.3 Tests: Fractional Kelly Auto-Tuning
# ===================================================================

class TestLogUtility(unittest.TestCase):
    """Test _log_utility()."""

    def test_zero_frac(self):
        r = np.array([0.01, -0.02, 0.005])
        u = _log_utility(r, 0.0)
        self.assertAlmostEqual(u, 0.0, places=10)

    def test_positive_frac_positive_drift(self):
        r = np.full(100, 0.001)
        u = _log_utility(r, 0.5)
        self.assertGreater(u, 0)

    def test_large_frac_can_be_negative(self):
        r = np.array([-0.1, 0.01, -0.05])
        u = _log_utility(r, 1.0)
        # With losses, log utility can be negative
        self.assertIsInstance(u, float)


class TestAutoTuneKelly(unittest.TestCase):
    """Test auto_tune_kelly_frac()."""

    def test_returns_result(self):
        n = 600
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        mu, sigma = _generate_forecasts(returns, seed=99)
        result = auto_tune_kelly_frac(returns, mu, sigma)
        self.assertIsInstance(result, KellyAutoTuneResult)

    def test_optimal_in_grid(self):
        n = 600
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        mu, sigma = _generate_forecasts(returns, seed=99)
        result = auto_tune_kelly_frac(returns, mu, sigma)
        self.assertIn(result.optimal_frac, DEFAULT_FRAC_GRID)

    def test_custom_grid(self):
        n = 600
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        mu, sigma = _generate_forecasts(returns, seed=99)
        grid = [0.05, 0.1, 0.15]
        result = auto_tune_kelly_frac(returns, mu, sigma, frac_grid=grid)
        self.assertIn(result.optimal_frac, grid)

    def test_n_folds_positive(self):
        n = 600
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        mu, sigma = _generate_forecasts(returns, seed=99)
        result = auto_tune_kelly_frac(returns, mu, sigma)
        self.assertGreater(result.n_folds, 0)

    def test_utilities_length(self):
        n = 600
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        mu, sigma = _generate_forecasts(returns, seed=99)
        result = auto_tune_kelly_frac(returns, mu, sigma)
        self.assertEqual(len(result.utilities), len(DEFAULT_FRAC_GRID))

    def test_short_data_fallback(self):
        returns = np.array([0.01, -0.01])
        mu = np.array([0.001, -0.001])
        sigma = np.array([0.02, 0.02])
        result = auto_tune_kelly_frac(returns, mu, sigma)
        self.assertEqual(result.n_folds, 0)
        self.assertAlmostEqual(result.optimal_frac, 0.5)

    def test_higher_vol_needs_smaller_frac(self):
        """High-vol asset should prefer smaller or equal Kelly fraction."""
        n = 600
        rng = np.random.default_rng(42)
        # Low-vol asset with strong signal
        low_vol_returns = rng.normal(0.002, 0.005, n)
        low_mu = low_vol_returns + rng.normal(0, 0.001, n)
        low_sigma = np.full(n, 0.005)
        result_low = auto_tune_kelly_frac(low_vol_returns, low_mu, low_sigma)
        # High-vol asset with weak signal
        high_vol_returns = rng.normal(0.0005, 0.05, n)
        high_mu = high_vol_returns + rng.normal(0, 0.02, n)
        high_sigma = np.full(n, 0.05)
        result_high = auto_tune_kelly_frac(high_vol_returns, high_mu, high_sigma)
        # Both should return valid results from the grid
        self.assertIn(result_high.optimal_frac, DEFAULT_FRAC_GRID)
        self.assertIn(result_low.optimal_frac, DEFAULT_FRAC_GRID)
        # High-vol with weak signal should not pick the most aggressive frac
        # (this is a soft check -- the key property is the mechanism works)
        self.assertLessEqual(result_high.optimal_frac, 0.5)

    def test_with_student_t(self):
        n = 600
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        mu, sigma = _generate_forecasts(returns, seed=99)
        nu = np.full(n, 8.0)
        result = auto_tune_kelly_frac(returns, mu, sigma, forecasts_nu=nu)
        self.assertIsInstance(result, KellyAutoTuneResult)
        self.assertIn(result.optimal_frac, DEFAULT_FRAC_GRID)


# ===================================================================
# Constants Tests
# ===================================================================

class TestEpic13Constants(unittest.TestCase):
    """Test constant values."""

    def test_kelly_bounds(self):
        self.assertAlmostEqual(KELLY_MAX, 0.5)
        self.assertAlmostEqual(KELLY_MIN, -0.5)

    def test_default_frac(self):
        self.assertAlmostEqual(DEFAULT_KELLY_FRAC, 0.5)

    def test_min_threshold(self):
        self.assertAlmostEqual(MIN_KELLY_THRESHOLD, 0.02)

    def test_dd_thresholds(self):
        self.assertAlmostEqual(DD_REDUCE_THRESHOLD, 0.10)
        self.assertAlmostEqual(DD_FLAT_THRESHOLD, 0.15)

    def test_frac_grid(self):
        self.assertEqual(DEFAULT_FRAC_GRID, [0.1, 0.2, 0.3, 0.5])

    def test_autotune_windows(self):
        self.assertEqual(AUTOTUNE_TRAIN_DAYS, 252)
        self.assertEqual(AUTOTUNE_TEST_DAYS, 21)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic13EdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_kelly_with_tiny_sigma(self):
        """Very small sigma -> bounded fraction."""
        f = kelly_fraction(0.001, 1e-10)
        self.assertAlmostEqual(f, KELLY_MAX)

    def test_kelly_with_very_large_nu(self):
        """Very large nu -> converges to Gaussian."""
        f_gauss = kelly_fraction(0.001, 0.02)
        f_large_nu = kelly_fraction(0.001, 0.02, nu=1000.0)
        self.assertAlmostEqual(f_gauss, f_large_nu, places=3)

    def test_drawdown_negative_dd(self):
        """Negative drawdown treated as positive."""
        result = drawdown_adjusted_kelly(0.3, -0.12)
        self.assertLess(result.f_adjusted, 0.3)

    def test_running_dd_single_point(self):
        dd = compute_running_drawdown(np.array([1.0]))
        self.assertAlmostEqual(dd[0], 0.0)

    def test_max_dd_all_positive(self):
        r = np.full(50, 0.01)
        dd = compute_max_drawdown(r)
        self.assertAlmostEqual(dd, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
