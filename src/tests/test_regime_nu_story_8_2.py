"""
Tests for Story 8.2: Regime-Conditional nu Estimation.

Tests that regime_nu_estimates() properly estimates per-regime nu values,
with crisis regimes getting heavier tails (lower nu) than calm regimes.
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

from calibration.continuous_nu import (
    MIN_REGIME_SAMPLES,
    REGIME_CRISIS_JUMP,
    REGIME_HIGH_VOL_RANGE,
    REGIME_HIGH_VOL_TREND,
    REGIME_LOW_VOL_RANGE,
    REGIME_LOW_VOL_TREND,
    RegimeNuResult,
    regime_nu_estimates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_regime_data(
    regime_nus,
    regime_sizes,
    seed=42,
):
    """
    Generate synthetic regime-tagged Student-t returns.

    Parameters
    ----------
    regime_nus : dict
        {regime_id: nu} mapping.
    regime_sizes : dict
        {regime_id: n_samples} mapping.

    Returns
    -------
    returns, vol, regime_labels : np.ndarray
    """
    rng = np.random.default_rng(seed)
    all_returns = []
    all_vol = []
    all_labels = []

    for regime_id in sorted(regime_nus.keys()):
        nu = regime_nus[regime_id]
        n = regime_sizes[regime_id]
        # Student-t with regime-specific nu
        r = rng.standard_t(df=nu, size=n) * 0.01
        v = np.full(n, 0.01)
        labels = np.full(n, regime_id, dtype=int)
        all_returns.append(r)
        all_vol.append(v)
        all_labels.append(labels)

    returns = np.concatenate(all_returns)
    vol = np.concatenate(all_vol)
    regime_labels = np.concatenate(all_labels)

    # Shuffle to interleave regimes (more realistic)
    idx = rng.permutation(len(returns))
    return returns[idx], vol[idx], regime_labels[idx]


def _make_crisis_vs_calm_data(n_per_regime=200, seed=42):
    """Standard test data with crisis (nu=3) vs calm (nu=20) regimes."""
    regime_nus = {
        REGIME_LOW_VOL_TREND: 20.0,
        REGIME_HIGH_VOL_TREND: 8.0,
        REGIME_LOW_VOL_RANGE: 15.0,
        REGIME_HIGH_VOL_RANGE: 6.0,
        REGIME_CRISIS_JUMP: 3.0,
    }
    regime_sizes = {r: n_per_regime for r in regime_nus}
    return _make_regime_data(regime_nus, regime_sizes, seed=seed)


class TestRegimeNuBasic(unittest.TestCase):
    """Basic API and contract tests."""

    def test_returns_result_dataclass(self):
        """regime_nu_estimates returns RegimeNuResult."""
        returns, vol, labels = _make_crisis_vs_calm_data()
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertIsInstance(result, RegimeNuResult)

    def test_result_has_all_regimes(self):
        """Result contains nu for all observed regimes."""
        returns, vol, labels = _make_crisis_vs_calm_data()
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        for r in range(5):
            self.assertIn(r, result.regime_nus)

    def test_regime_nus_in_valid_range(self):
        """All regime nus are in [2.1, 50]."""
        returns, vol, labels = _make_crisis_vs_calm_data()
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        for r, nu in result.regime_nus.items():
            self.assertGreaterEqual(nu, 2.1, f"Regime {r}: nu={nu} < 2.1")
            self.assertLessEqual(nu, 50.0, f"Regime {r}: nu={nu} > 50")

    def test_global_nu_is_finite(self):
        """Global fallback nu is finite."""
        returns, vol, labels = _make_crisis_vs_calm_data()
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertTrue(math.isfinite(result.global_nu))

    def test_regime_counts_match(self):
        """Regime counts match actual label counts."""
        returns, vol, labels = _make_crisis_vs_calm_data(n_per_regime=100)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        for r in range(5):
            expected = int(np.sum(labels == r))
            self.assertEqual(result.regime_counts[r], expected)


class TestRegimeNuOrdering(unittest.TestCase):
    """Tests that nu_CRISIS < nu_LOW_VOL_TREND."""

    def test_crisis_heavier_than_calm(self):
        """nu_CRISIS < nu_LOW_VOL_TREND with clear separation in data."""
        returns, vol, labels = _make_crisis_vs_calm_data(n_per_regime=300, seed=100)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        nu_crisis = result.regime_nus[REGIME_CRISIS_JUMP]
        nu_calm = result.regime_nus[REGIME_LOW_VOL_TREND]
        self.assertLess(nu_crisis, nu_calm,
                        f"Crisis nu={nu_crisis:.2f} not < Calm nu={nu_calm:.2f}")

    def test_crisis_heavier_90_percent(self):
        """nu_CRISIS < nu_LOW_VOL_TREND for 90%+ of assets (10 trials)."""
        n_success = 0
        for seed in range(10):
            returns, vol, labels = _make_crisis_vs_calm_data(
                n_per_regime=300, seed=200 + seed
            )
            result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
            if result.regime_nus[REGIME_CRISIS_JUMP] < result.regime_nus[REGIME_LOW_VOL_TREND]:
                n_success += 1
        self.assertGreaterEqual(n_success / 10, 0.90,
                                f"Only {n_success}/10 had crisis < calm")

    def test_high_vol_heavier_than_low_vol(self):
        """nu_HIGH_VOL_RANGE < nu_LOW_VOL_RANGE with clear separation."""
        returns, vol, labels = _make_crisis_vs_calm_data(n_per_regime=300, seed=300)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        nu_high = result.regime_nus[REGIME_HIGH_VOL_RANGE]
        nu_low = result.regime_nus[REGIME_LOW_VOL_RANGE]
        self.assertLess(nu_high, nu_low,
                        f"HighVol nu={nu_high:.2f} not < LowVol nu={nu_low:.2f}")


class TestRegimeNuBorrowing(unittest.TestCase):
    """Tests for regime sample minimum and borrowing."""

    def test_small_regime_borrows_from_global(self):
        """Regime with < 50 samples borrows global nu."""
        regime_nus = {0: 10.0, 1: 8.0, 4: 3.0}
        regime_sizes = {0: 200, 1: 200, 4: 30}  # Regime 4 too small
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=400)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertIn(REGIME_CRISIS_JUMP, result.borrowed_regimes)

    def test_large_regime_not_borrowed(self):
        """Regime with >= 50 samples is NOT borrowed."""
        regime_nus = {0: 10.0, 4: 3.0}
        regime_sizes = {0: 200, 4: 200}
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=410)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertNotIn(0, result.borrowed_regimes)
        self.assertNotIn(REGIME_CRISIS_JUMP, result.borrowed_regimes)

    def test_borrowed_regime_gets_global_nu(self):
        """Borrowed regime gets the global nu value."""
        regime_nus = {0: 10.0, 4: 3.0}
        regime_sizes = {0: 200, 4: 20}  # Regime 4 too small
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=420)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertAlmostEqual(
            result.regime_nus[REGIME_CRISIS_JUMP],
            result.global_nu,
            places=6,
        )

    def test_all_regimes_too_small(self):
        """All regimes with < min_samples: all borrow from global."""
        regime_nus = {0: 5.0, 1: 8.0}
        regime_sizes = {0: 30, 1: 30}
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=430)
        result = regime_nu_estimates(
            returns, vol, labels, 1e-5, 0.01, 0.99, min_samples=50
        )
        self.assertEqual(len(result.borrowed_regimes), 2)

    def test_custom_min_samples(self):
        """Custom min_samples threshold is respected."""
        regime_nus = {0: 10.0, 4: 3.0}
        regime_sizes = {0: 200, 4: 80}
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=440)

        # With min_samples=100, regime 4 borrows
        result_100 = regime_nu_estimates(
            returns, vol, labels, 1e-5, 0.01, 0.99, min_samples=100
        )
        self.assertIn(REGIME_CRISIS_JUMP, result_100.borrowed_regimes)

        # With min_samples=50, regime 4 is estimated
        result_50 = regime_nu_estimates(
            returns, vol, labels, 1e-5, 0.01, 0.99, min_samples=50
        )
        self.assertNotIn(REGIME_CRISIS_JUMP, result_50.borrowed_regimes)


class TestRegimeNuBIC(unittest.TestCase):
    """BIC improvement tests."""

    def test_bic_improvement_with_regime_switching(self):
        """Regime-switching data should show positive BIC improvement."""
        # Extreme nu differences across regimes
        regime_nus = {
            REGIME_LOW_VOL_TREND: 25.0,
            REGIME_CRISIS_JUMP: 3.0,
        }
        regime_sizes = {
            REGIME_LOW_VOL_TREND: 500,
            REGIME_CRISIS_JUMP: 500,
        }
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=500)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertGreater(result.total_bic_improvement, 0.0)

    def test_bic_improvement_finite(self):
        """BIC improvement is always finite."""
        returns, vol, labels = _make_crisis_vs_calm_data()
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertTrue(math.isfinite(result.total_bic_improvement))

    def test_regime_ll_finite(self):
        """Per-regime log-likelihoods are finite."""
        returns, vol, labels = _make_crisis_vs_calm_data()
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        for r, ll in result.regime_ll.items():
            self.assertTrue(math.isfinite(ll), f"Regime {r}: ll={ll}")


class TestRegimeNuEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_single_regime(self):
        """All data in one regime: should still work."""
        rng = np.random.default_rng(600)
        returns = rng.standard_t(df=5.0, size=200) * 0.01
        vol = np.full(200, 0.01)
        labels = np.zeros(200, dtype=int)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertIn(0, result.regime_nus)
        self.assertGreaterEqual(result.regime_nus[0], 2.1)

    def test_two_regimes(self):
        """Two regimes with different tails."""
        regime_nus = {0: 15.0, 4: 3.0}
        regime_sizes = {0: 300, 4: 300}
        returns, vol, labels = _make_regime_data(regime_nus, regime_sizes, seed=610)
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        self.assertIn(0, result.regime_nus)
        self.assertIn(4, result.regime_nus)

    def test_no_crash_on_empty_regime(self):
        """If a regime has 0 samples, it's simply absent from results."""
        rng = np.random.default_rng(620)
        returns = rng.standard_t(df=5.0, size=200) * 0.01
        vol = np.full(200, 0.01)
        labels = np.zeros(200, dtype=int)  # Only regime 0
        result = regime_nu_estimates(returns, vol, labels, 1e-5, 0.01, 0.99)
        # Regime 4 not present
        self.assertNotIn(4, result.regime_nus)

    def test_custom_filter_func(self):
        """Works with custom filter function."""
        from calibration.continuous_nu import _student_t_log_likelihood

        def custom_filter(returns, vol, q, c, phi, nu):
            ll = _student_t_log_likelihood(returns, vol, nu)
            return np.zeros(len(returns)), np.zeros(len(returns)), ll

        returns, vol, labels = _make_crisis_vs_calm_data(n_per_regime=100)
        result = regime_nu_estimates(
            returns, vol, labels, 1e-5, 0.01, 0.99,
            filter_func=custom_filter,
        )
        self.assertIsInstance(result, RegimeNuResult)


class TestRegimeNuConstants(unittest.TestCase):
    """Tests for configuration constants."""

    def test_min_regime_samples(self):
        """MIN_REGIME_SAMPLES is 50."""
        self.assertEqual(MIN_REGIME_SAMPLES, 50)

    def test_regime_labels_defined(self):
        """All 5 regime constants are defined."""
        self.assertEqual(REGIME_LOW_VOL_TREND, 0)
        self.assertEqual(REGIME_HIGH_VOL_TREND, 1)
        self.assertEqual(REGIME_LOW_VOL_RANGE, 2)
        self.assertEqual(REGIME_HIGH_VOL_RANGE, 3)
        self.assertEqual(REGIME_CRISIS_JUMP, 4)


if __name__ == '__main__':
    unittest.main()
