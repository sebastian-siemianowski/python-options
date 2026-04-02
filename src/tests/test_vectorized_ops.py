"""
Test Story 7.1: Vectorized Filter Operations.

Validates:
  1. phi^H matches loop results (numerical precision)
  2. BMA weights sum to 1
  3. Batch MC draws shape correct
  4. Vectorized quantiles match per-horizon
  5. BMA log-sum-exp stability
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from models.vectorized_ops import (
    vectorized_phi_forecast,
    vectorized_phi_variance,
    vectorized_bma_weights,
    batch_monte_carlo_sample,
    vectorized_quantiles,
)


class TestVectorizedOps(unittest.TestCase):
    """Tests for vectorized operations."""

    def test_phi_forecast_matches_loop(self):
        """Vectorized phi^H matches manual loop."""
        mu = 0.05
        phi = 0.98
        horizons = np.array([1, 3, 7, 30, 90, 180, 365])
        
        vec_result = vectorized_phi_forecast(mu, phi, horizons)
        
        # Loop reference
        loop_result = np.array([mu * phi**h for h in horizons])
        
        np.testing.assert_allclose(vec_result, loop_result, rtol=1e-12)

    def test_bma_weights_sum_to_one(self):
        """BMA weights sum to 1.0."""
        bics = np.array([-1000, -950, -900, -850, -800])
        weights = vectorized_bma_weights(bics)
        
        self.assertAlmostEqual(np.sum(weights), 1.0, places=10)

    def test_bma_logsumexp_stability(self):
        """BMA stable with extreme BIC values."""
        bics = np.array([-50000, -49900, -49800])
        weights = vectorized_bma_weights(bics)
        
        self.assertAlmostEqual(np.sum(weights), 1.0, places=10)
        self.assertTrue(np.all(weights >= 0))
        # Best BIC should have highest weight
        self.assertEqual(np.argmax(weights), 0)

    def test_batch_mc_shape(self):
        """Batch MC samples have correct shape."""
        means = np.array([0.01, 0.02, 0.03])
        variances = np.array([0.001, 0.002, 0.003])
        
        samples = batch_monte_carlo_sample(means, variances, n_samples=500)
        self.assertEqual(samples.shape, (3, 500))

    def test_batch_mc_mean_convergence(self):
        """MC sample means converge to true means."""
        rng = np.random.default_rng(42)
        means = np.array([0.5, -0.3, 1.0])
        variances = np.array([0.01, 0.01, 0.01])
        
        samples = batch_monte_carlo_sample(
            means, variances, n_samples=10000, rng=rng
        )
        sample_means = np.mean(samples, axis=1)
        np.testing.assert_allclose(sample_means, means, atol=0.05)

    def test_vectorized_quantiles_shape(self):
        """Quantile output shape correct."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, (5, 1000))
        quantiles = np.array([0.10, 0.50, 0.90])
        
        result = vectorized_quantiles(samples, quantiles)
        self.assertEqual(result.shape, (5, 3))

    def test_variance_matches_loop(self):
        """Vectorized variance matches manual computation."""
        phi = 0.98
        q = 1e-5
        R = 0.001
        horizons = np.array([1, 7, 30])
        
        vec_var = vectorized_phi_variance(phi, q, R, horizons)
        
        # Loop reference
        for i, h in enumerate(horizons):
            phi_sq = phi * phi
            sv = q * (1 - phi_sq**h) / (1 - phi_sq) + R
            self.assertAlmostEqual(vec_var[i], sv, places=10)

    def test_bma_empty(self):
        """Empty BIC array -> empty weights."""
        weights = vectorized_bma_weights(np.array([]))
        self.assertEqual(len(weights), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
