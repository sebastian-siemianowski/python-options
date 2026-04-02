"""
Test Story 3.4: OU Parameter Estimation in Tuning Pipeline.

Validates:
  1. Known OU params recovered from simulation
  2. Random walk produces long half-life
  3. Half-life within valid bounds
  4. Short series returns None
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from tuning.tune import (
    fit_ou_params,
    OU_HALF_LIFE_MIN_TUNE,
    OU_HALF_LIFE_MAX_TUNE,
)


class TestOUEstimation(unittest.TestCase):
    """Tests for OU parameter estimation."""

    def _simulate_ou(self, kappa=0.05, theta=100.0, sigma=0.01, n=1000, seed=42):
        """Simulate OU process: dp = kappa * (theta - p) * dt + sigma * dW."""
        np.random.seed(seed)
        prices = np.zeros(n)
        prices[0] = theta
        for t in range(1, n):
            dp = kappa * (theta - prices[t-1]) + sigma * np.random.randn() * prices[t-1]
            prices[t] = max(prices[t-1] + dp, 0.01)
        return prices

    def test_recover_kappa(self):
        """Recover approximate kappa from OU simulation."""
        prices = self._simulate_ou(kappa=0.05, n=2000)
        result = fit_ou_params(prices)
        self.assertIsNotNone(result)
        # kappa should be in the right ballpark (within 3x)
        self.assertGreater(result["kappa"], 0.01)
        self.assertLess(result["kappa"], 0.15)

    def test_random_walk_long_halflife(self):
        """Random walk (no mean reversion) produces long half-life."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))
        result = fit_ou_params(prices)
        self.assertIsNotNone(result)
        # Random walk -> long half-life (near max)
        self.assertGreater(result["half_life_days"], 50)

    def test_halflife_bounds(self):
        """Half-life must be in [5, 252]."""
        prices = self._simulate_ou(kappa=0.05, n=500)
        result = fit_ou_params(prices)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["half_life_days"], OU_HALF_LIFE_MIN_TUNE)
        self.assertLessEqual(result["half_life_days"], OU_HALF_LIFE_MAX_TUNE)

    def test_result_structure(self):
        """Result has required keys."""
        prices = self._simulate_ou()
        result = fit_ou_params(prices)
        self.assertIsNotNone(result)
        for key in ["kappa", "theta", "sigma_ou", "half_life_days"]:
            self.assertIn(key, result)

    def test_short_series_none(self):
        """Too few prices -> None."""
        result = fit_ou_params(np.ones(20) * 100)
        self.assertIsNone(result)

    def test_theta_near_mean(self):
        """Estimated theta should be near the true mean level."""
        prices = self._simulate_ou(theta=150.0, kappa=0.03, n=2000)
        result = fit_ou_params(prices)
        self.assertIsNotNone(result)
        # Theta EWMA should be near 150
        self.assertAlmostEqual(result["theta"], 150.0, delta=30.0)

    def test_positive_sigma(self):
        """sigma_ou should be positive."""
        prices = self._simulate_ou()
        result = fit_ou_params(prices)
        self.assertIsNotNone(result)
        self.assertGreater(result["sigma_ou"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
