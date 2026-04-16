"""
Test Story 3.3: GARCH(1,1) MLE Parameter Fitting.

Validates:
  1. Known GARCH params recovered from simulation
  2. Stationarity constraint enforced
  3. Returns correct structure
  4. Handles edge cases gracefully
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from tuning.tune import (
    fit_garch_mle,
    garch_log_likelihood,
    GARCH_PERSISTENCE_MAX,
)


def simulate_garch_returns(omega, alpha, beta, n=2000, seed=42):
    """Simulate returns from known GARCH(1,1) process."""
    np.random.seed(seed)
    sigma2 = omega / (1 - alpha - beta) if alpha + beta < 1 else 0.01**2
    returns = np.zeros(n)
    for t in range(n):
        returns[t] = np.sqrt(sigma2) * np.random.randn()
        sigma2 = omega + alpha * returns[t]**2 + beta * sigma2
    return returns


class TestGarchMLE(unittest.TestCase):
    """Tests for GARCH(1,1) MLE fitting."""

    def test_recover_known_params(self):
        """Recover GARCH params from simulation (within tolerance)."""
        true_omega = 1e-6
        true_alpha = 0.10
        true_beta = 0.85
        returns = simulate_garch_returns(true_omega, true_alpha, true_beta, n=5000)
        
        result = fit_garch_mle(returns)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["alpha"], true_alpha, delta=0.05)
        self.assertAlmostEqual(result["beta"], true_beta, delta=0.05)

    def test_stationarity_enforced(self):
        """Persistence must be < 1."""
        # Use extreme returns that might push toward non-stationarity
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        returns[100:110] *= 10  # Extreme cluster
        
        result = fit_garch_mle(returns)
        if result is not None:
            self.assertLess(result["persistence"], 1.0)

    def test_result_structure(self):
        """Result has correct keys."""
        returns = simulate_garch_returns(1e-6, 0.10, 0.85)
        result = fit_garch_mle(returns)
        self.assertIsNotNone(result)
        for key in ["omega", "alpha", "beta", "persistence", "long_run_vol", "converged"]:
            self.assertIn(key, result)

    def test_short_series_returns_none(self):
        """Too few returns -> None."""
        result = fit_garch_mle(np.random.randn(10))
        self.assertIsNone(result)

    def test_constant_returns_none(self):
        """Constant returns (zero var) -> None."""
        result = fit_garch_mle(np.zeros(100))
        self.assertIsNone(result)

    def test_positive_params(self):
        """All fitted params should be positive."""
        returns = simulate_garch_returns(1e-6, 0.08, 0.88, n=3000)
        result = fit_garch_mle(returns)
        self.assertIsNotNone(result)
        self.assertGreater(result["omega"], 0)
        self.assertGreater(result["alpha"], 0)
        self.assertGreater(result["beta"], 0)

    def test_long_run_vol_reasonable(self):
        """Long-run vol should be reasonable (1%-100% annualized)."""
        returns = simulate_garch_returns(1e-6, 0.10, 0.85, n=3000)
        result = fit_garch_mle(returns)
        self.assertIsNotNone(result)
        self.assertGreater(result["long_run_vol"], 0.01)
        self.assertLess(result["long_run_vol"], 2.0)

    def test_neg_log_likelihood_finite(self):
        """Log-likelihood should be finite."""
        returns = simulate_garch_returns(1e-6, 0.10, 0.85, n=500)
        nll = garch_log_likelihood([1e-6, 0.10, 0.85], returns)
        self.assertTrue(np.isfinite(nll))


if __name__ == "__main__":
    unittest.main(verbosity=2)
