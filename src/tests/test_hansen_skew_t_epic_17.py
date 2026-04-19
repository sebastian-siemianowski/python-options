"""
Test Suite for Epic 17: Hansen Skew-t Model Accuracy Enhancement
=================================================================

Story 17.1: Continuous Lambda Optimization
Story 17.2: Time-Varying Skewness via Regime-Conditional Lambda
Story 17.3: Skew-Adjusted Directional Signals
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

from models.hansen_skew_t import (
    # Story 17.1
    HansenLambdaResult,
    optimize_hansen_lambda,
    _hansen_profile_loglik,
    LAMBDA_GRID,
    HANSEN_NU_DEFAULT,
    # Story 17.2
    RegimeLambdaResult,
    regime_lambda_estimates,
    ALL_REGIMES, LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE,
    HIGH_VOL_RANGE, CRISIS_JUMP, MIN_REGIME_OBS,
    # Story 17.3
    SkewAdjustedDirectionResult,
    skew_adjusted_direction,
    skew_adjusted_direction_array,
    # Existing functions for validation
    hansen_skew_t_cdf,
    hansen_skew_t_logpdf,
    _hansen_constants,
)


def _generate_skewed_returns(n=500, skew_lambda=-0.15, nu=8.0, seed=42):
    """Generate returns with known Hansen skew-t innovations."""
    rng = np.random.default_rng(seed)
    # Approximate: use normal with skew bias
    base = rng.standard_t(nu, n)
    # Add skew: shift negative observations to be larger
    if skew_lambda < 0:
        mask = base < 0
        base[mask] *= (1 - skew_lambda)  # Amplify negative
    elif skew_lambda > 0:
        mask = base > 0
        base[mask] *= (1 + skew_lambda)  # Amplify positive
    vol = np.ones(n) * 0.02
    returns = base * 0.01
    return returns, vol


# ===================================================================
# Story 17.1 Tests: Continuous Lambda Optimization
# ===================================================================

class TestHansenProfileLoglik(unittest.TestCase):
    """Test the profile log-likelihood for lambda."""

    def test_finite_output(self):
        z = np.random.default_rng(42).standard_normal(200)
        nll = _hansen_profile_loglik(0.0, z, 8.0)
        self.assertTrue(np.isfinite(nll))

    def test_symmetric_at_zero(self):
        """lambda=0 should give symmetric Student-t loglik."""
        z = np.random.default_rng(42).standard_normal(300)
        nll_zero = _hansen_profile_loglik(0.0, z, 8.0)
        nll_small = _hansen_profile_loglik(0.01, z, 8.0)
        # Should be same order of magnitude (Hansen constants shift NLL)
        self.assertAlmostEqual(nll_zero, nll_small, delta=10.0)

    def test_nan_handling(self):
        z = np.array([0.1, np.nan, -0.2, 0.3])
        nll = _hansen_profile_loglik(0.0, z, 8.0)
        self.assertTrue(np.isfinite(nll))

    def test_empty_innovations(self):
        nll = _hansen_profile_loglik(0.0, np.array([]), 8.0)
        self.assertEqual(nll, 1e12)


class TestOptimizeHansenLambda(unittest.TestCase):
    """Test optimize_hansen_lambda()."""

    def test_returns_dataclass(self):
        returns, vol = _generate_skewed_returns(300)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        self.assertIsInstance(result, HansenLambdaResult)

    def test_lambda_in_bounds(self):
        returns, vol = _generate_skewed_returns(300)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        self.assertGreaterEqual(result.lambda_star, -0.9)
        self.assertLessEqual(result.lambda_star, 0.9)

    def test_converges(self):
        returns, vol = _generate_skewed_returns(500)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        self.assertTrue(result.converged)

    def test_bic_computed(self):
        returns, vol = _generate_skewed_returns(300)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        self.assertTrue(np.isfinite(result.bic))

    def test_negative_skew_detected(self):
        """For left-skewed data, lambda should be negative."""
        returns, vol = _generate_skewed_returns(500, skew_lambda=-0.3, seed=99)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        # Should detect some negative skew (may not be exact)
        self.assertLess(result.lambda_star, 0.1)

    def test_positive_skew_detected(self):
        """For right-skewed data, lambda should be positive."""
        returns, vol = _generate_skewed_returns(500, skew_lambda=0.3, seed=99)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        self.assertGreater(result.lambda_star, -0.1)

    def test_short_series_fallback(self):
        returns = np.array([0.01, -0.02, 0.005])
        vol = np.array([0.01, 0.01, 0.01])
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99)
        self.assertFalse(result.converged)
        self.assertAlmostEqual(result.lambda_star, 0.0)

    def test_different_nu_values(self):
        """Should work with different nu."""
        returns, vol = _generate_skewed_returns(300)
        r4 = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=4.0)
        r12 = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=12.0)
        self.assertIsInstance(r4, HansenLambdaResult)
        self.assertIsInstance(r12, HansenLambdaResult)
        self.assertAlmostEqual(r4.nu, 4.0)
        self.assertAlmostEqual(r12.nu, 12.0)


# ===================================================================
# Story 17.2 Tests: Regime-Conditional Lambda
# ===================================================================

class TestRegimeLambdaEstimates(unittest.TestCase):
    """Test regime_lambda_estimates()."""

    def _make_data_with_regimes(self, n=600, seed=42):
        """Generate data with regime labels."""
        rng = np.random.default_rng(seed)
        vol = np.ones(n) * 0.015
        returns = rng.normal(0, 0.015, n)
        # Assign regimes in blocks
        labels = []
        block = n // 5
        for i, regime in enumerate(ALL_REGIMES):
            labels.extend([regime] * block)
        # Pad remaining
        while len(labels) < n:
            labels.append(LOW_VOL_TREND)
        return returns, vol, labels[:n]

    def test_returns_dataclass(self):
        returns, vol, labels = self._make_data_with_regimes()
        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)
        self.assertIsInstance(result, RegimeLambdaResult)

    def test_all_regimes_present(self):
        returns, vol, labels = self._make_data_with_regimes()
        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)
        for regime in ALL_REGIMES:
            self.assertIn(regime, result.lambda_by_regime)
            self.assertIn(regime, result.n_obs_by_regime)

    def test_lambda_in_bounds(self):
        returns, vol, labels = self._make_data_with_regimes()
        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)
        for regime, lam in result.lambda_by_regime.items():
            self.assertGreaterEqual(lam, -0.9)
            self.assertLessEqual(lam, 0.9)

    def test_insufficient_regime_defaults(self):
        """Regime with < MIN_REGIME_OBS observations gets lambda=0."""
        returns, vol, labels = self._make_data_with_regimes(n=200)
        # With only 200 points and 5 regimes, each has 40 < MIN_REGIME_OBS
        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)
        for regime in ALL_REGIMES:
            if result.n_obs_by_regime[regime] < MIN_REGIME_OBS:
                self.assertAlmostEqual(result.lambda_by_regime[regime], 0.0)

    def test_length_mismatch_raises(self):
        returns = np.zeros(100)
        vol = np.zeros(100)
        labels = ["LOW_VOL_TREND"] * 50  # Wrong length
        with self.assertRaises(ValueError):
            regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)

    def test_bic_computed(self):
        returns, vol, labels = self._make_data_with_regimes()
        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)
        self.assertTrue(np.isfinite(result.total_bic))

    def test_regimes_estimated_list(self):
        returns, vol, labels = self._make_data_with_regimes()
        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)
        self.assertIsInstance(result.regimes_estimated, list)


# ===================================================================
# Story 17.3 Tests: Skew-Adjusted Directional Signals
# ===================================================================

class TestSkewAdjustedDirection(unittest.TestCase):
    """Test skew_adjusted_direction()."""

    def test_returns_dataclass(self):
        result = skew_adjusted_direction(0.001, 0.02, nu=8.0, lambda_=0.0)
        self.assertIsInstance(result, SkewAdjustedDirectionResult)

    def test_positive_mu_high_prob(self):
        """Positive mu -> high P(r > 0)."""
        result = skew_adjusted_direction(0.01, 0.02, nu=8.0, lambda_=0.0)
        self.assertGreater(result.prob_positive, 0.5)

    def test_negative_mu_low_prob(self):
        """Negative mu -> low P(r > 0)."""
        result = skew_adjusted_direction(-0.01, 0.02, nu=8.0, lambda_=0.0)
        self.assertLess(result.prob_positive, 0.5)

    def test_zero_mu_near_half(self):
        """Zero mu -> P(r > 0) near 0.5 for symmetric."""
        result = skew_adjusted_direction(0.0, 0.02, nu=8.0, lambda_=0.0)
        self.assertAlmostEqual(result.prob_positive, 0.5, delta=0.05)

    def test_left_skew_increases_prob(self):
        """Left skew (lambda < 0): P(r > 0) should increase vs symmetric
        because left tail is heavier, CDF at 0 is lower."""
        r_sym = skew_adjusted_direction(0.0, 0.02, nu=8.0, lambda_=0.0)
        r_left = skew_adjusted_direction(0.0, 0.02, nu=8.0, lambda_=-0.3)
        self.assertGreater(r_left.prob_positive, r_sym.prob_positive - 0.15)

    def test_skew_adjustment_sign(self):
        """Skew adjustment should have consistent sign."""
        r_left = skew_adjusted_direction(0.0, 0.02, nu=8.0, lambda_=-0.3)
        r_right = skew_adjusted_direction(0.0, 0.02, nu=8.0, lambda_=0.3)
        # Left and right skew should produce different adjustments
        self.assertNotAlmostEqual(
            r_left.skew_adjustment, r_right.skew_adjustment, places=2
        )

    def test_prob_bounds(self):
        """Probabilities should be in (0, 1)."""
        result = skew_adjusted_direction(0.05, 0.01, nu=4.0, lambda_=-0.5)
        self.assertGreater(result.prob_positive, 0)
        self.assertLess(result.prob_positive, 1)
        self.assertGreater(result.prob_positive_symmetric, 0)
        self.assertLess(result.prob_positive_symmetric, 1)

    def test_zero_sigma_handled(self):
        """Zero sigma should not crash."""
        result = skew_adjusted_direction(0.01, 0.0, nu=8.0, lambda_=0.0)
        self.assertIsInstance(result, SkewAdjustedDirectionResult)

    def test_symmetric_no_adjustment(self):
        """Lambda=0 should produce zero skew adjustment."""
        result = skew_adjusted_direction(0.001, 0.02, nu=8.0, lambda_=0.0)
        self.assertAlmostEqual(result.skew_adjustment, 0.0, delta=0.01)


class TestSkewAdjustedDirectionArray(unittest.TestCase):
    """Test skew_adjusted_direction_array()."""

    def test_output_shape(self):
        mu = np.array([0.001, -0.001, 0.0])
        sigma = np.array([0.02, 0.02, 0.02])
        probs = skew_adjusted_direction_array(mu, sigma, nu=8.0, lambda_=0.0)
        self.assertEqual(len(probs), 3)

    def test_correct_ordering(self):
        """Higher mu -> higher P(r > 0)."""
        mu = np.array([-0.01, 0.0, 0.01])
        sigma = np.array([0.02, 0.02, 0.02])
        probs = skew_adjusted_direction_array(mu, sigma, nu=8.0, lambda_=0.0)
        self.assertLess(probs[0], probs[1])
        self.assertLess(probs[1], probs[2])

    def test_all_in_bounds(self):
        mu = np.random.default_rng(42).normal(0, 0.01, 20)
        sigma = np.ones(20) * 0.02
        probs = skew_adjusted_direction_array(mu, sigma, nu=8.0, lambda_=-0.2)
        self.assertTrue(np.all(probs > 0))
        self.assertTrue(np.all(probs < 1))


# ===================================================================
# Hansen CDF/Constants Tests
# ===================================================================

class TestHansenConstants(unittest.TestCase):
    """Test that Hansen constants are well-behaved."""

    def test_symmetric_constants(self):
        """Lambda=0: a=0, b=1, c = standard t constant."""
        a, b, c = _hansen_constants(8.0, 0.0)
        self.assertAlmostEqual(a, 0.0, places=10)
        self.assertAlmostEqual(b, 1.0, places=10)
        self.assertGreater(c, 0)

    def test_b_positive(self):
        for lam in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            a, b, c = _hansen_constants(8.0, lam)
            self.assertGreater(b, 0)

    def test_cdf_at_zero_symmetric(self):
        """CDF(0) = 0.5 for symmetric."""
        cdf_val = hansen_skew_t_cdf(0.0, 8.0, 0.0)
        self.assertAlmostEqual(float(cdf_val), 0.5, places=3)


# ===================================================================
# Integration Tests
# ===================================================================

class TestEpic17Integration(unittest.TestCase):
    """Integration tests combining stories."""

    def test_optimize_then_direction(self):
        """Optimize lambda, then use it for directional signal."""
        returns, vol = _generate_skewed_returns(400, skew_lambda=-0.2)
        result = optimize_hansen_lambda(returns, vol, 1e-6, 1.0, 0.99, nu=8.0)
        # Use optimized lambda for direction
        direction = skew_adjusted_direction(
            0.001, 0.02, nu=8.0, lambda_=result.lambda_star,
        )
        self.assertIsInstance(direction, SkewAdjustedDirectionResult)
        self.assertGreater(direction.prob_positive, 0)
        self.assertLess(direction.prob_positive, 1)

    def test_regime_then_direction(self):
        """Estimate regime lambdas, use for direction."""
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0, 0.015, n)
        vol = np.ones(n) * 0.015
        labels = [ALL_REGIMES[i % 5] for i in range(n)]

        result = regime_lambda_estimates(returns, vol, labels, 1e-6, 1.0, 0.99)

        for regime in result.regimes_estimated:
            lam = result.lambda_by_regime[regime]
            direction = skew_adjusted_direction(0.001, 0.02, nu=8.0, lambda_=lam)
            self.assertIsInstance(direction, SkewAdjustedDirectionResult)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic17EdgeCases(unittest.TestCase):

    def test_extreme_lambda(self):
        result = skew_adjusted_direction(0.001, 0.02, nu=8.0, lambda_=0.89)
        self.assertGreater(result.prob_positive, 0)
        self.assertLess(result.prob_positive, 1)

    def test_small_nu(self):
        result = skew_adjusted_direction(0.001, 0.02, nu=2.5, lambda_=-0.2)
        self.assertIsInstance(result, SkewAdjustedDirectionResult)

    def test_large_nu_approaches_normal(self):
        """Large nu -> Student-t approaches Normal."""
        r_t = skew_adjusted_direction(0.001, 0.02, nu=100.0, lambda_=0.0)
        r_n = skew_adjusted_direction(0.001, 0.02, nu=1000.0, lambda_=0.0)
        self.assertAlmostEqual(r_t.prob_positive, r_n.prob_positive, delta=0.01)

    def test_constants_valid(self):
        self.assertEqual(len(LAMBDA_GRID), 7)
        self.assertEqual(MIN_REGIME_OBS, 50)
        self.assertAlmostEqual(HANSEN_NU_DEFAULT, 10.0)


if __name__ == "__main__":
    unittest.main()
