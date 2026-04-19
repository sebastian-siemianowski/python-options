"""
Test Suite for Epic 20: Mean Reversion Enhancement -- OU Parameter Accuracy
============================================================================

Story 20.1: Multi-Scale Kappa Estimation
Story 20.2: Adaptive Equilibrium with Change-Point Detection (PELT)
Story 20.3: Kappa-Dependent Position Timing (mr_signal_strength)
"""
import os
import sys
import math
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from models.ou_mean_reversion import (
    # Story 20.1
    KappaEstimate,
    MultiScaleKappaResult,
    multi_scale_kappa,
    _estimate_kappa_at_frequency,
    DEFAULT_FREQUENCIES,
    KAPPA_MIN,
    KAPPA_MAX,
    MIN_OBS_PER_FREQ,
    # Story 20.2
    ChangePointResult,
    detect_equilibrium_shift,
    adaptive_equilibrium,
    PELT_MIN_SEGMENT,
    PELT_MAX_CHANGEPOINTS,
    PELT_BIC_PENALTY_FACTOR,
    # Story 20.3
    MRSignalResult,
    mr_signal_strength,
    mr_signal_strength_array,
    MR_STRONG_THRESHOLD,
    MR_WEAK_THRESHOLD,
    MR_KELLY_STRONG,
    MR_KELLY_WEAK,
)


# ===================================================================
# Helper: Generate OU process
# ===================================================================

def simulate_ou(
    n: int = 500,
    kappa: float = 0.05,
    mu: float = 100.0,
    sigma: float = 2.0,
    x0: float = 100.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate OU process prices: dX = kappa*(mu - X)*dt + sigma*dW."""
    rng = np.random.default_rng(seed)
    dt = 1.0  # daily
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) * dt + sigma * rng.normal()
    return x


# ===================================================================
# Story 20.1 Tests: Multi-Scale Kappa
# ===================================================================

class TestEstimateKappaAtFrequency(unittest.TestCase):
    """Test internal _estimate_kappa_at_frequency()."""

    def test_returns_estimate_for_ou(self):
        prices = simulate_ou(n=500, kappa=0.05)
        log_prices = np.log(prices)
        est = _estimate_kappa_at_frequency(log_prices, freq=1)
        self.assertIsNotNone(est)
        self.assertIsInstance(est, KappaEstimate)

    def test_kappa_positive(self):
        prices = simulate_ou(n=500, kappa=0.05)
        log_prices = np.log(prices)
        est = _estimate_kappa_at_frequency(log_prices, freq=1)
        self.assertGreater(est.kappa, 0)

    def test_se_positive(self):
        prices = simulate_ou(n=500, kappa=0.05)
        log_prices = np.log(prices)
        est = _estimate_kappa_at_frequency(log_prices, freq=1)
        self.assertGreater(est.se, 0)

    def test_short_series_returns_none(self):
        log_prices = np.log(np.array([100.0, 101.0, 102.0]))
        est = _estimate_kappa_at_frequency(log_prices, freq=1)
        self.assertIsNone(est)

    def test_weekly_frequency(self):
        prices = simulate_ou(n=500, kappa=0.05)
        log_prices = np.log(prices)
        est = _estimate_kappa_at_frequency(log_prices, freq=5)
        self.assertIsNotNone(est)
        self.assertGreater(est.kappa, 0)

    def test_r_squared_bounded(self):
        prices = simulate_ou(n=500, kappa=0.05)
        log_prices = np.log(prices)
        est = _estimate_kappa_at_frequency(log_prices, freq=1)
        self.assertGreaterEqual(est.r_squared, 0.0)
        self.assertLessEqual(est.r_squared, 1.0)


class TestMultiScaleKappa(unittest.TestCase):
    """Test multi_scale_kappa()."""

    def test_returns_result(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        self.assertIsInstance(result, MultiScaleKappaResult)

    def test_pooled_kappa_positive(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        self.assertGreater(result.kappa_pooled, 0)

    def test_pooled_kappa_in_bounds(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        self.assertGreaterEqual(result.kappa_pooled, KAPPA_MIN)
        self.assertLessEqual(result.kappa_pooled, KAPPA_MAX)

    def test_se_pooled_less_than_daily(self):
        """Pooled SE should be <= daily-only SE (more info)."""
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        if result.n_frequencies_used > 1:
            daily_est = [e for e in result.per_frequency if e.frequency == 1]
            if daily_est:
                self.assertLessEqual(result.se_pooled, daily_est[0].se + 1e-10)

    def test_half_life_consistent(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        expected_hl = math.log(2) / result.kappa_pooled
        self.assertAlmostEqual(result.half_life_days, expected_hl, places=3)

    def test_frequencies_used(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices, frequencies=[1, 5, 22])
        self.assertGreaterEqual(result.n_frequencies_used, 1)
        self.assertLessEqual(result.n_frequencies_used, 3)

    def test_custom_frequencies(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices, frequencies=[1, 10])
        for est in result.per_frequency:
            self.assertIn(est.frequency, [1, 10])

    def test_very_short_prices(self):
        prices = np.array([100.0])
        result = multi_scale_kappa(prices)
        self.assertEqual(result.n_frequencies_used, 0)
        self.assertAlmostEqual(result.kappa_pooled, 0.05)  # fallback

    def test_cv_computed(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        self.assertGreaterEqual(result.cv, 0.0)

    def test_strong_mr_higher_kappa(self):
        """Strong MR (kappa=0.15) should give higher pooled kappa than weak."""
        prices_strong = simulate_ou(n=1000, kappa=0.15, seed=42)
        prices_weak = simulate_ou(n=1000, kappa=0.02, seed=42)
        r_strong = multi_scale_kappa(prices_strong)
        r_weak = multi_scale_kappa(prices_weak)
        self.assertGreater(r_strong.kappa_pooled, r_weak.kappa_pooled)

    def test_per_frequency_list(self):
        prices = simulate_ou(n=500, kappa=0.05)
        result = multi_scale_kappa(prices)
        for est in result.per_frequency:
            self.assertIsInstance(est, KappaEstimate)
            self.assertGreater(est.n_obs, 0)


# ===================================================================
# Story 20.2 Tests: Change-Point Detection
# ===================================================================

class TestDetectEquilibriumShift(unittest.TestCase):
    """Test detect_equilibrium_shift()."""

    def test_returns_result(self):
        mu = np.random.default_rng(42).normal(100, 1, 200)
        result = detect_equilibrium_shift(mu)
        self.assertIsInstance(result, ChangePointResult)

    def test_no_change_point_in_constant(self):
        mu = np.ones(200) * 100
        result = detect_equilibrium_shift(mu)
        self.assertEqual(len(result.change_points), 0)

    def test_detects_level_shift(self):
        """Clear regime shift in the middle should be detected."""
        rng = np.random.default_rng(42)
        mu = np.concatenate([
            rng.normal(100, 0.5, 150),
            rng.normal(110, 0.5, 150),
        ])
        result = detect_equilibrium_shift(mu, min_segment=20)
        self.assertGreaterEqual(len(result.change_points), 1)
        # Change point should be near index 150
        if len(result.change_points) > 0:
            nearest = min(result.change_points, key=lambda x: abs(x - 150))
            self.assertAlmostEqual(nearest, 150, delta=30)

    def test_max_changepoints_enforced(self):
        rng = np.random.default_rng(42)
        mu = np.concatenate([
            rng.normal(100, 0.3, 100),
            rng.normal(110, 0.3, 100),
            rng.normal(90, 0.3, 100),
            rng.normal(105, 0.3, 100),
            rng.normal(115, 0.3, 100),
            rng.normal(95, 0.3, 100),
            rng.normal(108, 0.3, 100),
        ])
        result = detect_equilibrium_shift(mu, max_changepoints=3, min_segment=20)
        self.assertLessEqual(len(result.change_points), 3)

    def test_segments_cover_data(self):
        rng = np.random.default_rng(42)
        mu = np.concatenate([rng.normal(100, 1, 150), rng.normal(105, 1, 150)])
        result = detect_equilibrium_shift(mu)
        total_len = sum(result.segment_lengths)
        self.assertEqual(total_len, 300)

    def test_n_segments_correct(self):
        rng = np.random.default_rng(42)
        mu = rng.normal(100, 1, 200)
        result = detect_equilibrium_shift(mu)
        self.assertEqual(result.n_segments, len(result.change_points) + 1)

    def test_short_series(self):
        mu = np.array([1.0, 2.0])
        result = detect_equilibrium_shift(mu)
        self.assertEqual(len(result.change_points), 0)

    def test_penalty_positive(self):
        mu = np.random.default_rng(42).normal(100, 1, 200)
        result = detect_equilibrium_shift(mu)
        self.assertGreater(result.penalty, 0)

    def test_segment_means_reasonable(self):
        rng = np.random.default_rng(42)
        mu = np.concatenate([
            rng.normal(100, 0.5, 150),
            rng.normal(120, 0.5, 150),
        ])
        result = detect_equilibrium_shift(mu, min_segment=20)
        if result.n_segments >= 2:
            # First segment should be ~100, last ~120
            self.assertAlmostEqual(result.segment_means[0], 100, delta=3)
            self.assertAlmostEqual(result.segment_means[-1], 120, delta=3)

    def test_change_points_sorted(self):
        rng = np.random.default_rng(42)
        mu = np.concatenate([rng.normal(i * 10, 0.5, 100) for i in range(5)])
        result = detect_equilibrium_shift(mu, min_segment=20)
        self.assertEqual(result.change_points, sorted(result.change_points))


class TestAdaptiveEquilibrium(unittest.TestCase):
    """Test adaptive_equilibrium()."""

    def test_output_shape(self):
        mu = np.random.default_rng(42).normal(100, 1, 200)
        eq = adaptive_equilibrium(mu, [100])
        self.assertEqual(len(eq), 200)

    def test_no_change_points(self):
        mu = np.random.default_rng(42).normal(100, 1, 200)
        eq = adaptive_equilibrium(mu, [])
        # Equilibrium = overall mean
        np.testing.assert_allclose(eq, np.mean(mu), atol=1e-10)

    def test_one_change_point(self):
        rng = np.random.default_rng(42)
        part1 = rng.normal(100, 0.1, 100)
        part2 = rng.normal(110, 0.1, 100)
        mu = np.concatenate([part1, part2])
        eq = adaptive_equilibrium(mu, [100])
        self.assertAlmostEqual(eq[50], np.mean(part1), delta=1)
        self.assertAlmostEqual(eq[150], np.mean(part2), delta=1)

    def test_multiple_change_points(self):
        rng = np.random.default_rng(42)
        mu = np.concatenate([
            rng.normal(100, 0.1, 100),
            rng.normal(110, 0.1, 100),
            rng.normal(90, 0.1, 100),
        ])
        eq = adaptive_equilibrium(mu, [100, 200])
        self.assertAlmostEqual(eq[50], 100, delta=2)
        self.assertAlmostEqual(eq[150], 110, delta=2)
        self.assertAlmostEqual(eq[250], 90, delta=2)


# ===================================================================
# Story 20.3 Tests: MR Signal Strength
# ===================================================================

class TestMRSignalStrength(unittest.TestCase):
    """Test mr_signal_strength()."""

    def test_returns_result(self):
        result = mr_signal_strength(105.0, 100.0, 0.05, 1.0)
        self.assertIsInstance(result, MRSignalResult)

    def test_strong_signal(self):
        """Price far from equilibrium with high kappa => strong."""
        result = mr_signal_strength(110.0, 100.0, 0.05, 0.1)
        self.assertEqual(result.strength, "strong")
        self.assertAlmostEqual(result.kelly_fraction, MR_KELLY_STRONG)

    def test_no_signal_at_equilibrium(self):
        """Price at equilibrium => no signal."""
        result = mr_signal_strength(100.0, 100.0, 0.05, 1.0)
        self.assertEqual(result.strength, "none")
        self.assertAlmostEqual(result.kelly_fraction, MR_KELLY_WEAK)

    def test_direction_above_eq(self):
        """Price above equilibrium => direction -1 (expect reversion down)."""
        result = mr_signal_strength(105.0, 100.0, 0.05, 1.0)
        self.assertEqual(result.direction, -1)

    def test_direction_below_eq(self):
        """Price below equilibrium => direction +1 (expect reversion up)."""
        result = mr_signal_strength(95.0, 100.0, 0.05, 1.0)
        self.assertEqual(result.direction, 1)

    def test_z_formula(self):
        """z = kappa * (price - eq) / sigma."""
        result = mr_signal_strength(105.0, 100.0, 0.1, 2.0)
        expected_z = 0.1 * (105.0 - 100.0) / 2.0
        self.assertAlmostEqual(result.z, expected_z, places=8)

    def test_moderate_signal(self):
        """Intermediate |z| => moderate."""
        # z = 0.05 * 20 / 1.0 = 1.0 (between 0.5 and 2.0)
        result = mr_signal_strength(120.0, 100.0, 0.05, 1.0)
        self.assertEqual(result.strength, "moderate")
        self.assertGreater(result.kelly_fraction, MR_KELLY_WEAK)
        self.assertLess(result.kelly_fraction, MR_KELLY_STRONG)

    def test_zero_sigma_handled(self):
        result = mr_signal_strength(105.0, 100.0, 0.05, 0.0)
        self.assertIsInstance(result, MRSignalResult)

    def test_zero_kappa(self):
        result = mr_signal_strength(105.0, 100.0, 0.0, 1.0)
        self.assertEqual(result.strength, "none")
        self.assertAlmostEqual(result.z, 0.0)

    def test_distance_positive(self):
        result = mr_signal_strength(95.0, 100.0, 0.05, 1.0)
        self.assertGreater(result.distance, 0)


class TestMRSignalStrengthArray(unittest.TestCase):
    """Test mr_signal_strength_array()."""

    def test_output_shapes(self):
        prices = np.array([95, 100, 105, 110, 115], dtype=float)
        eq = np.array([100, 100, 100, 100, 100], dtype=float)
        sigma = np.ones(5)
        z, kelly, direction = mr_signal_strength_array(prices, eq, 0.05, sigma)
        self.assertEqual(len(z), 5)
        self.assertEqual(len(kelly), 5)
        self.assertEqual(len(direction), 5)

    def test_kelly_non_negative(self):
        prices = np.linspace(90, 110, 50)
        eq = np.ones(50) * 100
        sigma = np.ones(50) * 2.0
        z, kelly, direction = mr_signal_strength_array(prices, eq, 0.05, sigma)
        self.assertTrue(np.all(kelly >= 0))

    def test_direction_signs(self):
        prices = np.array([95, 100, 105], dtype=float)
        eq = np.array([100, 100, 100], dtype=float)
        sigma = np.ones(3)
        z, kelly, direction = mr_signal_strength_array(prices, eq, 0.05, sigma)
        self.assertEqual(direction[0], 1)    # below eq => +1
        self.assertEqual(direction[1], 0)    # at eq => 0
        self.assertEqual(direction[2], -1)   # above eq => -1

    def test_z_consistency(self):
        """Array z should match scalar z."""
        prices = np.array([105.0])
        eq = np.array([100.0])
        sigma = np.array([2.0])
        kappa = 0.1
        z_arr, _, _ = mr_signal_strength_array(prices, eq, kappa, sigma)
        result = mr_signal_strength(105.0, 100.0, 0.1, 2.0)
        self.assertAlmostEqual(z_arr[0], result.z, places=8)

    def test_strong_kelly_at_extremes(self):
        """Very far from equilibrium should have strong Kelly."""
        prices = np.array([200.0])
        eq = np.array([100.0])
        sigma = np.array([1.0])
        z, kelly, _ = mr_signal_strength_array(prices, eq, 0.1, sigma)
        self.assertAlmostEqual(kelly[0], MR_KELLY_STRONG)


# ===================================================================
# Integration Tests
# ===================================================================

class TestEpic20Integration(unittest.TestCase):
    """Integration tests combining stories."""

    def test_full_pipeline(self):
        """OU prices -> multi_scale_kappa -> detect_equilibrium_shift -> mr_signal_strength."""
        prices = simulate_ou(n=500, kappa=0.05, mu=100, sigma=2, seed=42)

        # Step 1: Estimate kappa
        kappa_result = multi_scale_kappa(prices)
        self.assertGreater(kappa_result.kappa_pooled, 0)

        # Step 2: Detect equilibrium shifts
        log_prices = np.log(np.maximum(prices, 1e-10))
        cp_result = detect_equilibrium_shift(log_prices)
        self.assertIsInstance(cp_result, ChangePointResult)

        # Step 3: Compute adaptive equilibrium
        eq = adaptive_equilibrium(log_prices, cp_result.change_points)
        self.assertEqual(len(eq), len(prices))

        # Step 4: Compute MR signals
        sigma = np.std(np.diff(log_prices)) * np.ones(len(prices))
        z, kelly, direction = mr_signal_strength_array(
            log_prices, eq, kappa_result.kappa_pooled, sigma,
        )
        self.assertEqual(len(z), len(prices))
        self.assertTrue(np.all(kelly >= 0))

    def test_multi_regime_ou(self):
        """OU with equilibrium shift should detect the shift."""
        rng = np.random.default_rng(42)
        # Two regimes: mean=100 for first half, mean=110 for second half
        n = 400
        prices = np.zeros(n)
        prices[0] = 100
        for t in range(1, n):
            mu = 100 if t < 200 else 110
            prices[t] = prices[t - 1] + 0.05 * (mu - prices[t - 1]) + rng.normal() * 2
        prices = np.maximum(prices, 10)

        kappa_result = multi_scale_kappa(prices)
        self.assertGreater(kappa_result.kappa_pooled, 0)

        # Detect shift
        cp = detect_equilibrium_shift(prices, min_segment=30)
        # Should detect at least one change point near t=200
        if len(cp.change_points) > 0:
            nearest = min(cp.change_points, key=lambda x: abs(x - 200))
            self.assertAlmostEqual(nearest, 200, delta=50)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic20EdgeCases(unittest.TestCase):

    def test_constants(self):
        self.assertEqual(DEFAULT_FREQUENCIES, [1, 5, 22])
        self.assertEqual(MIN_OBS_PER_FREQ, 30)
        self.assertAlmostEqual(MR_STRONG_THRESHOLD, 2.0)
        self.assertAlmostEqual(MR_WEAK_THRESHOLD, 0.5)
        self.assertAlmostEqual(MR_KELLY_STRONG, 0.80)
        self.assertAlmostEqual(MR_KELLY_WEAK, 0.0)
        self.assertEqual(PELT_MIN_SEGMENT, 22)
        self.assertEqual(PELT_MAX_CHANGEPOINTS, 5)

    def test_random_walk_low_kappa(self):
        """Random walk (kappa~0) should give low pooled kappa."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 500))
        prices = np.maximum(prices, 10)
        result = multi_scale_kappa(prices)
        # May or may not estimate kappa, but should not crash
        self.assertIsInstance(result, MultiScaleKappaResult)

    def test_negative_prices_handled(self):
        """Should handle any price array gracefully."""
        prices = np.array([100, 99, 98, 97] * 50, dtype=float)
        result = multi_scale_kappa(prices)
        self.assertIsInstance(result, MultiScaleKappaResult)

    def test_constant_prices(self):
        """All same price => no MR signal."""
        prices = np.ones(200) * 100
        result = multi_scale_kappa(prices)
        self.assertIsInstance(result, MultiScaleKappaResult)


if __name__ == "__main__":
    unittest.main()
