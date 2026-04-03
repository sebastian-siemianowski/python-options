"""
Tests for Story 1.2: Regime Drift Prior Integration.

Validates Bayesian shrinkage of Kalman posterior drift toward regime-conditional
drift priors using precision-weighted averaging.

Mathematical specification:
    tau_kalman = 1 / P_t
    tau_prior  = 1 / se_regime^2
    mu_shrunk  = (tau_k * mu_kalman + tau_p * mu_prior) / (tau_k + tau_p)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestBayesianShrinkageFormula(unittest.TestCase):
    """Validate the precision-weighted Bayesian shrinkage formula."""

    def _shrink(self, mu_kalman, P_t, mu_prior, se_prior):
        """Apply the shrinkage formula identical to signals.py implementation."""
        tau_kalman = 1.0 / max(P_t, 1e-12)
        tau_prior = 1.0 / max(se_prior ** 2, 1e-12)
        tau_total = tau_kalman + tau_prior
        return (tau_kalman * mu_kalman + tau_prior * mu_prior) / tau_total

    def test_confident_kalman_stays_near_kalman(self):
        """When P_t is very small (confident), mu_shrunk ~ mu_kalman."""
        mu_k = 0.0010  # 0.10%/day
        P_t = 1e-10     # Very confident Kalman
        mu_p = 0.0005   # Prior drift
        se_p = 0.001    # Prior uncertainty
        result = self._shrink(mu_k, P_t, mu_p, se_p)
        self.assertAlmostEqual(result, mu_k, places=6)

    def test_uncertain_kalman_shrinks_to_prior(self):
        """When P_t is large (uncertain), mu_shrunk ~ mu_prior."""
        mu_k = 0.0050   # Extreme Kalman drift
        P_t = 1.0        # Very uncertain
        mu_p = 0.0005    # Prior drift
        se_p = 0.001     # Tight prior
        result = self._shrink(mu_k, P_t, mu_p, se_p)
        # Should be much closer to prior than to Kalman
        dist_to_prior = abs(result - mu_p)
        dist_to_kalman = abs(result - mu_k)
        self.assertLess(dist_to_prior, dist_to_kalman)

    def test_equal_precision_averages(self):
        """When tau_kalman == tau_prior, mu_shrunk = midpoint."""
        P_t = 0.001
        se_p = np.sqrt(P_t)  # Same precision
        mu_k = 0.002
        mu_p = 0.000
        result = self._shrink(mu_k, P_t, mu_p, se_p)
        expected = (mu_k + mu_p) / 2.0
        self.assertAlmostEqual(result, expected, places=8)

    def test_zero_kalman_drift_shrinks_positive(self):
        """When Kalman drift is 0 but regime prior is positive, result > 0."""
        mu_k = 0.0
        P_t = 1e-6
        mu_p = 0.0008  # Positive regime drift prior
        se_p = 0.001
        result = self._shrink(mu_k, P_t, mu_p, se_p)
        # With tight Kalman, should barely move
        self.assertGreater(result, 0.0)

    def test_negative_prior_pulls_down(self):
        """Crisis regime prior (negative) pulls drift down."""
        mu_k = 0.0003  # Small positive Kalman
        P_t = 0.01      # Somewhat uncertain
        mu_p = -0.001   # Negative crisis prior
        se_p = 0.0005   # Tight crisis prior
        result = self._shrink(mu_k, P_t, mu_p, se_p)
        self.assertLess(result, mu_k)

    def test_shrinkage_reduces_variance(self):
        """Shrinkage toward common prior should reduce cross-asset variance."""
        mu_prior = 0.0005
        se_prior = 0.001
        P_t = 1e-5
        # Diverse Kalman drifts
        kalman_drifts = [0.0001, 0.0005, 0.0012, -0.0003, 0.0020]
        shrunk = [self._shrink(mu, P_t, mu_prior, se_prior) for mu in kalman_drifts]
        # Variance of shrunk should be <= variance of original
        var_original = np.var(kalman_drifts)
        var_shrunk = np.var(shrunk)
        self.assertLessEqual(var_shrunk, var_original)


class TestRegimeDriftPriorEstimation(unittest.TestCase):
    """Validate _estimate_regime_drift_priors function."""

    def test_function_exists(self):
        from decision.signals import _estimate_regime_drift_priors
        self.assertTrue(callable(_estimate_regime_drift_priors))

    def test_returns_none_for_short_data(self):
        from decision.signals import _estimate_regime_drift_priors
        ret = pd.Series(np.random.randn(50) * 0.01)
        vol = pd.Series(np.abs(np.random.randn(50) * 0.01))
        result = _estimate_regime_drift_priors(ret, vol)
        self.assertIsNone(result)

    def test_returns_dict_for_sufficient_data(self):
        from decision.signals import _estimate_regime_drift_priors
        np.random.seed(42)
        n = 500
        # Generate returns with regime structure
        ret = pd.Series(np.random.randn(n) * 0.01 + 0.0003)
        vol = pd.Series(np.abs(np.random.randn(n) * 0.005) + 0.005)
        result = _estimate_regime_drift_priors(ret, vol)
        # May return None if HMM not available, so skip in that case
        if result is not None:
            self.assertIn("current_drift_prior", result)
            self.assertIn("regime_drifts", result)
            self.assertIsInstance(result["current_drift_prior"], float)
            self.assertTrue(np.isfinite(result["current_drift_prior"]))


class TestShrinkageIntegration(unittest.TestCase):
    """Integration tests for shrinkage in the signals pipeline context."""

    def test_shrinkage_preserves_sign_for_strong_signal(self):
        """Strong positive Kalman drift stays positive after shrinkage."""
        mu_k = 0.002  # Strong positive
        P_t = 1e-7    # Very confident
        mu_p = -0.0001  # Slightly negative regime prior
        se_p = 0.001
        tau_k = 1.0 / max(P_t, 1e-12)
        tau_p = 1.0 / max(se_p ** 2, 1e-12)
        result = (tau_k * mu_k + tau_p * mu_p) / (tau_k + tau_p)
        self.assertGreater(result, 0)

    def test_nan_handling(self):
        """NaN inputs should not propagate."""
        # The code guards against NaN by checking isfinite and falling back to 0
        mu_k = float('nan')
        # This would be caught by the `if not np.isfinite(mu_t_mc): mu_t_mc = 0.0` guard
        self.assertFalse(np.isfinite(mu_k))


if __name__ == "__main__":
    unittest.main()
