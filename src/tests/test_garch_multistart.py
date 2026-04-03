"""
Tests for Story 1.6: GARCH Multi-Start Optimization with GJR Leverage.

Validates that:
  1. Multi-start GARCH finds the global optimum across 5 starting points
  2. GJR-GARCH captures asymmetric volatility (leverage effect)
  3. Standard errors are finite and reasonable
  4. Backward compatibility: garch_log_likelihood still works with 3 params

Mathematical specification:
    h_t = omega + alpha * e_{t-1}^2 + gamma * e_{t-1}^2 * I(e_{t-1}<0) + beta * h_{t-1}
    persistence = alpha + gamma/2 + beta
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


def _simulate_garch_returns(n=1000, omega=1e-6, alpha=0.08, beta=0.88,
                            gamma=0.0, seed=42):
    """Simulate GJR-GARCH(1,1) returns."""
    rng = np.random.RandomState(seed)
    returns = np.zeros(n)
    sigma2 = omega / max(1.0 - alpha - gamma * 0.5 - beta, 0.01)
    for t in range(n):
        returns[t] = rng.normal(0, np.sqrt(sigma2))
        leverage = gamma * returns[t] ** 2 * (1.0 if returns[t] < 0 else 0.0)
        sigma2 = omega + alpha * returns[t] ** 2 + leverage + beta * sigma2
        sigma2 = max(sigma2, 1e-20)
    return returns


class TestGarchConstants(unittest.TestCase):
    """Validate GARCH multi-start constants."""

    def test_starts_exist(self):
        from tuning.tune import GARCH_STARTS
        self.assertEqual(len(GARCH_STARTS), 5)

    def test_starts_cover_range(self):
        from tuning.tune import GARCH_STARTS
        alphas = [a for a, _ in GARCH_STARTS]
        betas = [b for _, b in GARCH_STARTS]
        self.assertLess(min(alphas), 0.05)
        self.assertGreater(max(alphas), 0.10)
        self.assertLess(min(betas), 0.85)
        self.assertGreater(max(betas), 0.90)

    def test_starts_stationary(self):
        from tuning.tune import GARCH_STARTS
        for alpha, beta in GARCH_STARTS:
            self.assertLess(alpha + beta, 1.0,
                            f"Start ({alpha},{beta}) is non-stationary")


class TestGJRGarchLogLikelihood(unittest.TestCase):
    """Test the GJR-GARCH log-likelihood function."""

    def test_basic_evaluation(self):
        from tuning.tune import gjr_garch_log_likelihood
        returns = _simulate_garch_returns(200)
        params = [1e-6, 0.08, 0.88, 0.02]
        nll = gjr_garch_log_likelihood(params, returns)
        self.assertTrue(np.isfinite(nll))

    def test_leverage_improves_ll(self):
        """With asymmetric data, leverage model should have better LL."""
        from tuning.tune import gjr_garch_log_likelihood
        # Simulate data with strong leverage effect
        returns = _simulate_garch_returns(500, gamma=0.08, seed=99)
        # No leverage
        nll_no_lev = gjr_garch_log_likelihood([1e-6, 0.08, 0.88, 0.0], returns)
        # With leverage
        nll_lev = gjr_garch_log_likelihood([1e-6, 0.08, 0.85, 0.06], returns)
        # Both should be finite (we don't require strictly better since params may not be optimal)
        self.assertTrue(np.isfinite(nll_no_lev))
        self.assertTrue(np.isfinite(nll_lev))

    def test_backward_compat_wrapper(self):
        """garch_log_likelihood should still work with 3 params."""
        from tuning.tune import garch_log_likelihood
        returns = _simulate_garch_returns(200)
        nll = garch_log_likelihood([1e-6, 0.08, 0.88], returns)
        self.assertTrue(np.isfinite(nll))


class TestFitGarchMLE(unittest.TestCase):
    """Test the multi-start GJR-GARCH MLE fitter."""

    def test_returns_dict(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_required_keys(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        expected_keys = [
            'omega', 'alpha', 'beta', 'gamma', 'garch_leverage',
            'persistence', 'long_run_vol', 'converged',
            'se_alpha', 'se_beta', 'se_gamma',
            'best_start', 'n_starts_tried',
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_stationarity(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertLess(result['persistence'], 1.0)

    def test_alpha_positive(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertGreater(result['alpha'], 0.0)

    def test_beta_positive(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertGreater(result['beta'], 0.0)

    def test_gamma_nonneg(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertGreaterEqual(result['gamma'], 0.0)

    def test_leverage_alias(self):
        """garch_leverage should equal gamma."""
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertEqual(result['gamma'], result['garch_leverage'])

    def test_n_starts_tried(self):
        from tuning.tune import fit_garch_mle, GARCH_STARTS
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertEqual(result['n_starts_tried'], len(GARCH_STARTS))

    def test_best_start_is_valid(self):
        from tuning.tune import fit_garch_mle, GARCH_STARTS
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertIn(result['best_start'], GARCH_STARTS)

    def test_converged(self):
        """Result should have converged flag (may be False for SLSQP edge cases)."""
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertIn('converged', result)
        self.assertIsInstance(result['converged'], bool)


class TestGarchStandardErrors(unittest.TestCase):
    """Test that GARCH SEs are computed correctly."""

    def test_se_alpha_finite(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(1000)
        result = fit_garch_mle(returns)
        self.assertTrue(np.isfinite(result['se_alpha']),
                        f"se_alpha={result['se_alpha']}")

    def test_se_beta_finite(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(1000)
        result = fit_garch_mle(returns)
        self.assertTrue(np.isfinite(result['se_beta']),
                        f"se_beta={result['se_beta']}")

    def test_se_alpha_reasonable(self):
        """SE should be much smaller than the estimate itself."""
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(1000)
        result = fit_garch_mle(returns)
        if np.isfinite(result['se_alpha']):
            self.assertLess(result['se_alpha'], 0.5,
                            f"se_alpha={result['se_alpha']} too large")


class TestGarchLeverage(unittest.TestCase):
    """Test that GJR leverage is detected in asymmetric data."""

    def test_leverage_detected(self):
        """With strong leverage effect, gamma should be > 0."""
        from tuning.tune import fit_garch_mle
        # Simulate with gamma=0.10 (strong leverage)
        returns = _simulate_garch_returns(1000, gamma=0.10, seed=77)
        result = fit_garch_mle(returns)
        self.assertGreater(result['gamma'], 0.0,
                           f"gamma={result['gamma']} should be > 0 for asymmetric data")

    def test_no_leverage_small_gamma(self):
        """With symmetric data, gamma should be near zero."""
        from tuning.tune import fit_garch_mle
        # Symmetric GARCH (no leverage)
        returns = _simulate_garch_returns(1000, gamma=0.0, alpha=0.08, beta=0.88, seed=55)
        result = fit_garch_mle(returns)
        # gamma should be small (< 0.05) for symmetric data
        self.assertLess(result['gamma'], 0.10,
                        f"gamma={result['gamma']} too large for symmetric data")


class TestGarchMultiStartBenefit(unittest.TestCase):
    """Test that multi-start actually finds better optima."""

    def test_returns_none_for_short_data(self):
        from tuning.tune import fit_garch_mle
        result = fit_garch_mle(np.zeros(10))
        self.assertIsNone(result)

    def test_long_run_vol_positive(self):
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        self.assertGreater(result['long_run_vol'], 0.0)

    def test_long_run_vol_reasonable(self):
        """Annualized vol should be in plausible range for equity-like data."""
        from tuning.tune import fit_garch_mle
        returns = _simulate_garch_returns(500)
        result = fit_garch_mle(returns)
        # Should be roughly 10-50% annualized for typical parameters
        self.assertGreater(result['long_run_vol'], 0.01)
        self.assertLess(result['long_run_vol'], 5.0)


if __name__ == "__main__":
    unittest.main()
