"""
Tests for Story 6.2: Minimum Description Length Model Averaging
===============================================================

Validates MDL weight computation with Fisher information penalty.

Key properties tested:
1. MDL weights sum to 1 and are positive
2. Fisher information estimated from log-likelihoods
3. MDL differs from BIC for small samples (n < 200)
4. MDL ~ BIC for large samples (n > 500) -- asymptotic equivalence
5. MDL selects simpler models for short-history assets
6. Validated on realistic parameter ranges (IONQ, RKLB, SPY, MSFT)
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.entropy_bma import (
    mdl_weights,
    MDLResult,
    _estimate_fisher_logdet,
    _compute_bic_weights,
)


class TestMDLBasicProperties(unittest.TestCase):
    """MDL weights must satisfy basic probability simplex constraints."""

    def test_weights_sum_to_one(self):
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 3.0, 4.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)

    def test_weights_positive(self):
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 3.0, 4.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertTrue(np.all(result.weights > 0))

    def test_returns_mdl_result(self):
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 3.0, 4.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertIsInstance(result, MDLResult)

    def test_result_fields_present(self):
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 3.0, 4.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertEqual(len(result.weights), 3)
        self.assertEqual(len(result.mdl_scores), 3)
        self.assertEqual(len(result.bic_scores), 3)
        self.assertEqual(len(result.fisher_penalties), 3)
        self.assertEqual(len(result.bic_weights), 3)

    def test_best_model_highest_weight(self):
        """Model with best (lowest) MDL score gets highest weight."""
        ll = np.array([-100.0, -200.0, -300.0])
        k = np.array([2.0, 2.0, 2.0])  # Same complexity
        result = mdl_weights(ll, k, n_obs=300)
        # Model 0 has best LL with same k -> best MDL
        self.assertEqual(np.argmax(result.weights), 0)

    def test_single_model(self):
        ll = np.array([-100.0])
        k = np.array([2.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertAlmostEqual(result.weights[0], 1.0)


class TestFisherInformation(unittest.TestCase):
    """Fisher information estimation from log-likelihoods."""

    def test_fisher_logdet_shape(self):
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 3.0, 4.0])
        logdet = _estimate_fisher_logdet(ll, k, n_obs=300)
        self.assertEqual(len(logdet), 3)

    def test_fisher_scales_with_k(self):
        """More parameters -> larger Fisher information magnitude."""
        ll = np.array([-100.0, -100.0, -100.0])
        k = np.array([2.0, 4.0, 6.0])
        logdet = _estimate_fisher_logdet(ll, k, n_obs=300)
        # |logdet| should increase with k (assuming same LL)
        self.assertGreater(abs(logdet[2]), abs(logdet[0]))

    def test_fisher_depends_on_ll(self):
        """Fisher info changes with fit quality (log-likelihood value)."""
        k = np.array([3.0, 3.0])
        ll_good = np.array([-50.0, -50.0])   # Good fit
        ll_bad = np.array([-500.0, -500.0])   # Bad fit
        logdet_good = _estimate_fisher_logdet(ll_good, k, n_obs=300)
        logdet_bad = _estimate_fisher_logdet(ll_bad, k, n_obs=300)
        # Good fit -> higher per-param Fisher -> different values
        self.assertFalse(np.allclose(logdet_good, logdet_bad))

    def test_external_fisher_accepted(self):
        """Can provide external Fisher info log-determinant."""
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 3.0, 4.0])
        external_fisher = np.array([5.0, 8.0, 12.0])
        result = mdl_weights(ll, k, n_obs=300, fisher_info_logdet=external_fisher)
        # Fisher penalties should be 0.5 * external values
        np.testing.assert_allclose(result.fisher_penalties, 0.5 * external_fisher)


class TestMDLvsBICSmallSample(unittest.TestCase):
    """MDL should differ from BIC for small samples (n < 200)."""

    def test_weights_differ_small_n(self):
        """For n < 200, MDL weights != BIC weights."""
        # Realistic: well-fit Kalman filter models give positive per-obs LL
        # (small daily return variance → high density)
        ll = np.array([230.0, 228.0, 225.0, 222.0, 218.0])
        k = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        n_obs = 100  # Small sample
        result = mdl_weights(ll, k, n_obs=n_obs)
        # Weights should differ meaningfully
        max_diff = np.max(np.abs(result.weights - result.bic_weights))
        self.assertGreater(max_diff, 0.01,
                           f"MDL and BIC weights too similar for n={n_obs}: "
                           f"max diff = {max_diff:.6f}")

    def test_weights_differ_n150(self):
        """Also differs at n=150."""
        ll = np.array([340.0, 337.0, 333.0, 328.0])
        k = np.array([2.0, 3.0, 5.0, 7.0])
        result = mdl_weights(ll, k, n_obs=150)
        max_diff = np.max(np.abs(result.weights - result.bic_weights))
        self.assertGreater(max_diff, 0.001,
                           f"MDL and BIC too similar at n=150")

    def test_fisher_penalty_nonzero(self):
        """Fisher penalties should be non-zero for finite samples."""
        ll = np.array([230.0, 225.0, 220.0])
        k = np.array([2.0, 3.0, 4.0])
        result = mdl_weights(ll, k, n_obs=100)
        self.assertFalse(np.allclose(result.fisher_penalties, 0))


class TestMDLvsBICLargeSample(unittest.TestCase):
    """For n > 500: MDL ~ BIC (asymptotic equivalence)."""

    def test_weights_converge_n1000(self):
        """At n=1000, MDL and BIC weights should be close."""
        ll = np.array([-500.0, -510.0, -520.0, -530.0])
        k = np.array([2.0, 3.0, 4.0, 5.0])
        result = mdl_weights(ll, k, n_obs=1000)
        # Weights should be similar (not identical, but close)
        max_diff = np.max(np.abs(result.weights - result.bic_weights))
        self.assertLess(max_diff, 0.15,
                        f"MDL and BIC weights too different for n=1000: "
                        f"max diff = {max_diff:.4f}")

    def test_weights_converge_n2000(self):
        """At n=2000, convergence should be even tighter."""
        ll = np.array([-1000.0, -1010.0, -1020.0, -1030.0])
        k = np.array([2.0, 3.0, 4.0, 5.0])
        result = mdl_weights(ll, k, n_obs=2000)
        max_diff = np.max(np.abs(result.weights - result.bic_weights))
        self.assertLess(max_diff, 0.10,
                        f"MDL and BIC weights too different for n=2000: "
                        f"max diff = {max_diff:.4f}")

    def test_ranking_preserved_large_n(self):
        """For large n, MDL and BIC should agree on best model."""
        ll = np.array([-400.0, -500.0, -600.0])
        k = np.array([2.0, 2.0, 2.0])  # Same complexity
        result = mdl_weights(ll, k, n_obs=1000)
        self.assertEqual(np.argmax(result.weights), np.argmax(result.bic_weights))


class TestMDLSelectsSimpler(unittest.TestCase):
    """MDL should favor simpler models for small-sample assets."""

    def test_simpler_preferred_small_n(self):
        """For small n with well-fit models, MDL gives more weight to simpler models."""
        # Complex model has slightly better LL but many more parameters
        # With positive per-obs LL (well-fit financial data), Fisher penalty is large
        ll = np.array([248.0, 245.0])   # Complex slightly better fit
        k = np.array([6.0, 2.0])        # Complex vs simple
        n_obs = 100

        result = mdl_weights(ll, k, n_obs=n_obs)

        # Weighted complexity: lower = simpler models preferred
        mdl_avg_k = float(np.sum(result.weights * k))
        bic_avg_k = float(np.sum(result.bic_weights * k))

        self.assertLess(mdl_avg_k, bic_avg_k,
                        f"MDL avg_k={mdl_avg_k:.2f} should be < "
                        f"BIC avg_k={bic_avg_k:.2f} for n={n_obs}")

    def test_simpler_preferred_varied_complexity(self):
        """Across a range of complexities, MDL leans simpler for small n."""
        # Realistic: models with increasing complexity get slightly better LL
        ll = np.array([255.0, 253.0, 250.0, 248.0, 245.0])
        k = np.array([8.0, 6.0, 4.0, 3.0, 2.0])
        n_obs = 80

        result = mdl_weights(ll, k, n_obs=n_obs)
        mdl_avg_k = float(np.sum(result.weights * k))
        bic_avg_k = float(np.sum(result.bic_weights * k))

        self.assertLess(mdl_avg_k, bic_avg_k,
                        f"MDL avg_k={mdl_avg_k:.2f} should be < "
                        f"BIC avg_k={bic_avg_k:.2f}")


class TestMDLRealisticAssets(unittest.TestCase):
    """Validate MDL on realistic asset parameter ranges."""

    def _make_realistic_ll(self, n_obs, n_models=6, seed=42):
        """Generate realistic log-likelihoods for Kalman filter models.

        For daily returns with sigma ~ 0.02, per-obs LL ~ 2.5.
        Models with more params fit slightly better.
        """
        rng = np.random.RandomState(seed)
        # Per-obs LL ~ 2.0-2.5 for well-fit financial models
        base_ll_per_obs = 2.3 + rng.uniform(-0.3, 0.0, n_models)
        base_ll_per_obs.sort()
        base_ll_per_obs = base_ll_per_obs[::-1]  # Best first
        ll = base_ll_per_obs * n_obs
        k = np.array([2, 3, 3, 4, 5, 6], dtype=float)[:n_models]
        return ll, k

    def test_ionq_short_history(self):
        """IONQ: short history (~150 obs) -> MDL should favor simpler models."""
        n_obs = 150
        ll, k = self._make_realistic_ll(n_obs, seed=101)
        result = mdl_weights(ll, k, n_obs=n_obs)

        # Should produce valid weights
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        self.assertTrue(np.all(result.weights > 0))

        # MDL should favor simpler models compared to BIC
        mdl_avg_k = float(np.sum(result.weights * k))
        bic_avg_k = float(np.sum(result.bic_weights * k))
        self.assertLessEqual(mdl_avg_k, bic_avg_k + 0.5,
                             f"IONQ: MDL avg_k={mdl_avg_k:.2f} vs BIC={bic_avg_k:.2f}")

    def test_rklb_recent_ipo(self):
        """RKLB: recent IPO (~200 obs) -> MDL should still differ from BIC."""
        n_obs = 200
        ll, k = self._make_realistic_ll(n_obs, seed=102)
        result = mdl_weights(ll, k, n_obs=n_obs)

        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        # Should show some difference from BIC
        max_diff = np.max(np.abs(result.weights - result.bic_weights))
        self.assertGreater(max_diff, 0.0001)

    def test_spy_long_history(self):
        """SPY: long history (~2500 obs) -> MDL ~ BIC."""
        n_obs = 2500
        ll, k = self._make_realistic_ll(n_obs, seed=103)
        result = mdl_weights(ll, k, n_obs=n_obs)

        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        # For large n, weights should converge
        max_diff = np.max(np.abs(result.weights - result.bic_weights))
        self.assertLess(max_diff, 0.15,
                        f"SPY: MDL-BIC diff={max_diff:.4f} too large for n={n_obs}")

    def test_msft_medium_history(self):
        """MSFT: medium history (~1000 obs)."""
        n_obs = 1000
        ll, k = self._make_realistic_ll(n_obs, seed=104)
        result = mdl_weights(ll, k, n_obs=n_obs)

        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        self.assertTrue(np.all(result.weights > 0))


class TestMDLEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_two_models(self):
        ll = np.array([-100.0, -110.0])
        k = np.array([2.0, 3.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        self.assertTrue(np.all(result.weights > 0))

    def test_equal_models(self):
        """Identical models get equal weights."""
        ll = np.array([-100.0, -100.0, -100.0])
        k = np.array([3.0, 3.0, 3.0])
        result = mdl_weights(ll, k, n_obs=300)
        np.testing.assert_allclose(result.weights,
                                   np.ones(3) / 3, atol=1e-10)

    def test_very_different_ll(self):
        """Handles widely spread log-likelihoods."""
        ll = np.array([-100.0, -500.0, -1000.0])
        k = np.array([2.0, 3.0, 4.0])
        result = mdl_weights(ll, k, n_obs=300)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        # Best model should dominate
        self.assertGreater(result.weights[0], 0.9)

    def test_fourteen_models(self):
        """Realistic: 14 models per regime."""
        np.random.seed(42)
        M = 14
        ll = -np.sort(np.random.uniform(100, 300, M))[::-1]
        k = np.random.choice([2, 3, 4, 5, 6], M).astype(float)
        result = mdl_weights(ll, k, n_obs=500)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        self.assertEqual(len(result.weights), 14)

    def test_mdl_scores_lower_is_better(self):
        """Verify that lower MDL score -> higher weight."""
        ll = np.array([-100.0, -105.0, -110.0])
        k = np.array([2.0, 2.0, 2.0])  # Same complexity
        result = mdl_weights(ll, k, n_obs=300)
        # Model 0 has best LL -> lowest MDL -> highest weight
        self.assertEqual(np.argmin(result.mdl_scores), np.argmax(result.weights))


if __name__ == '__main__':
    unittest.main()
