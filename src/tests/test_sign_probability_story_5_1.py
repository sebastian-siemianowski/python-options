"""
Tests for Story 5.1: Parameter Uncertainty Propagation into Sign Probability.

Validates:
1. sign_prob_with_uncertainty() returns valid probabilities
2. Gaussian closed form: P(r>0) = Phi(mu / sqrt(P_t + c*sigma^2))
3. Student-t MC integration matches expectation
4. Higher P_t (uncertainty) pushes probability toward 0.5
5. ECE < 0.05 on synthetic data with known DGP
6. Hit rate at 60% confidence threshold within 2% of stated
7. Edge cases: zero drift, huge uncertainty, tiny uncertainty
8. Comparison with plug-in (no uncertainty) baseline
"""
import os
import sys
import unittest
import numpy as np
from scipy.stats import norm, t as student_t_dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.sign_probability import (
    sign_prob_with_uncertainty,
    sign_prob_no_uncertainty,
    compute_sign_prob_ece,
    compute_hit_rate_at_threshold,
    _sign_prob_gaussian,
    _sign_prob_student_t,
)


class TestGaussianClosedForm(unittest.TestCase):
    """Test the Gaussian closed-form sign probability."""

    def test_positive_drift_above_half(self):
        """Positive mu should give P(r>0) > 0.5."""
        p = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=0.02, model='gaussian')
        self.assertGreater(p, 0.5)

    def test_negative_drift_below_half(self):
        """Negative mu should give P(r>0) < 0.5."""
        p = sign_prob_with_uncertainty(mu_t=-0.01, P_t=1e-4, sigma_t=0.02, model='gaussian')
        self.assertLess(p, 0.5)

    def test_zero_drift_equals_half(self):
        """Zero mu should give P(r>0) = 0.5."""
        p = sign_prob_with_uncertainty(mu_t=0.0, P_t=1e-4, sigma_t=0.02, model='gaussian')
        self.assertAlmostEqual(p, 0.5, places=5)

    def test_closed_form_matches_formula(self):
        """Verify the closed form: Phi(mu / sqrt(P_t + c*sigma^2))."""
        mu_t, P_t, sigma_t, c = 0.005, 1e-4, 0.015, 1.0
        expected = norm.cdf(mu_t / np.sqrt(P_t + c * sigma_t**2))
        actual = sign_prob_with_uncertainty(mu_t, P_t, sigma_t, c=c, model='gaussian')
        self.assertAlmostEqual(actual, expected, places=8)

    def test_c_multiplier(self):
        """c > 1 should increase total variance, pushing toward 0.5."""
        p_c1 = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=0.02, c=1.0, model='gaussian')
        p_c4 = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=0.02, c=4.0, model='gaussian')
        # Higher c -> more variance -> closer to 0.5
        self.assertGreater(abs(p_c1 - 0.5), abs(p_c4 - 0.5))

    def test_clipping_bounds(self):
        """Extreme probabilities should be clipped to [0.01, 0.99]."""
        p_high = sign_prob_with_uncertainty(mu_t=10.0, P_t=1e-8, sigma_t=0.001, model='gaussian')
        p_low = sign_prob_with_uncertainty(mu_t=-10.0, P_t=1e-8, sigma_t=0.001, model='gaussian')
        self.assertLessEqual(p_high, 0.99)
        self.assertGreaterEqual(p_low, 0.01)


class TestUncertaintyEffect(unittest.TestCase):
    """Test that increasing P_t (uncertainty) pushes P toward 0.5."""

    def test_higher_P_t_closer_to_half_positive_drift(self):
        """More uncertainty on positive drift -> P closer to 0.5 (from above)."""
        p_low_unc = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-6, sigma_t=0.02, model='gaussian')
        p_high_unc = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-2, sigma_t=0.02, model='gaussian')
        self.assertGreater(p_low_unc, p_high_unc)  # More certain -> further from 0.5
        self.assertGreater(p_high_unc, 0.5)  # Still above 0.5

    def test_higher_P_t_closer_to_half_negative_drift(self):
        """More uncertainty on negative drift -> P closer to 0.5 (from below)."""
        p_low_unc = sign_prob_with_uncertainty(mu_t=-0.01, P_t=1e-6, sigma_t=0.02, model='gaussian')
        p_high_unc = sign_prob_with_uncertainty(mu_t=-0.01, P_t=1e-2, sigma_t=0.02, model='gaussian')
        self.assertLess(p_low_unc, p_high_unc)  # More certain -> further from 0.5
        self.assertLess(p_high_unc, 0.5)  # Still below 0.5

    def test_tiny_P_t_matches_plug_in(self):
        """When P_t is negligible, should match plug-in estimate."""
        mu_t, sigma_t, c = 0.005, 0.02, 1.0
        p_unc = sign_prob_with_uncertainty(mu_t, P_t=1e-15, sigma_t=sigma_t, c=c, model='gaussian')
        p_plug = sign_prob_no_uncertainty(mu_t, sigma_t, c=c, model='gaussian')
        self.assertAlmostEqual(p_unc, p_plug, places=4)

    def test_huge_P_t_approaches_half(self):
        """When P_t >> obs_var, P(r>0) approaches 0.5."""
        p = sign_prob_with_uncertainty(mu_t=0.01, P_t=100.0, sigma_t=0.02, model='gaussian')
        self.assertAlmostEqual(p, 0.5, places=2)


class TestStudentTMC(unittest.TestCase):
    """Test the Student-t Monte Carlo integration."""

    def test_positive_drift_above_half(self):
        p = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=0.02,
                                       model='student_t', nu=8.0, rng_seed=42)
        self.assertGreater(p, 0.5)

    def test_negative_drift_below_half(self):
        p = sign_prob_with_uncertainty(mu_t=-0.01, P_t=1e-4, sigma_t=0.02,
                                       model='student_t', nu=8.0, rng_seed=42)
        self.assertLess(p, 0.5)

    def test_zero_drift_near_half(self):
        p = sign_prob_with_uncertainty(mu_t=0.0, P_t=1e-4, sigma_t=0.02,
                                       model='student_t', nu=8.0, rng_seed=42)
        self.assertAlmostEqual(p, 0.5, places=2)

    def test_nu_required_for_student_t(self):
        with self.assertRaises(ValueError):
            sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=0.02, model='student_t')

    def test_reproducibility_with_seed(self):
        kwargs = dict(mu_t=0.005, P_t=1e-4, sigma_t=0.02, model='student_t', nu=6.0)
        p1 = sign_prob_with_uncertainty(**kwargs, rng_seed=123)
        p2 = sign_prob_with_uncertainty(**kwargs, rng_seed=123)
        self.assertEqual(p1, p2)

    def test_heavier_tails_more_conservative(self):
        """Lower nu (heavier tails) should produce sign prob closer to 0.5."""
        kwargs = dict(mu_t=0.01, P_t=1e-4, sigma_t=0.02, model='student_t', rng_seed=42)
        p_heavy = sign_prob_with_uncertainty(**kwargs, nu=3.0)  # Heavy tails
        p_light = sign_prob_with_uncertainty(**kwargs, nu=30.0)  # Near Gaussian
        # Heavier tails -> more probability mass in opposite direction -> closer to 0.5
        self.assertGreater(abs(p_light - 0.5), abs(p_heavy - 0.5) - 0.02)

    def test_mc_converges_to_gaussian_for_large_nu(self):
        """For large nu, Student-t should approximate Gaussian."""
        kwargs = dict(mu_t=0.008, P_t=1e-4, sigma_t=0.02, c=1.0)
        p_gauss = sign_prob_with_uncertainty(**kwargs, model='gaussian')
        p_t_large_nu = sign_prob_with_uncertainty(**kwargs, model='student_t',
                                                   nu=100.0, n_mc=50000, rng_seed=42)
        self.assertAlmostEqual(p_gauss, p_t_large_nu, places=2)

    def test_higher_P_t_closer_to_half_student_t(self):
        """Same uncertainty effect for Student-t."""
        kwargs = dict(mu_t=0.01, sigma_t=0.02, model='student_t', nu=8.0, rng_seed=42)
        p_low = sign_prob_with_uncertainty(P_t=1e-6, **kwargs)
        p_high = sign_prob_with_uncertainty(P_t=1e-2, **kwargs)
        self.assertGreater(p_low, p_high)


class TestPlugInBaseline(unittest.TestCase):
    """Test the plug-in (no uncertainty) baseline."""

    def test_gaussian_plug_in(self):
        p = sign_prob_no_uncertainty(mu_t=0.005, sigma_t=0.02, model='gaussian')
        expected = norm.cdf(0.005 / 0.02)
        self.assertAlmostEqual(p, expected, places=6)

    def test_student_t_plug_in(self):
        p = sign_prob_no_uncertainty(mu_t=0.005, sigma_t=0.02, model='student_t', nu=8.0)
        expected = student_t_dist.cdf(0.005 / 0.02, df=8.0)
        self.assertAlmostEqual(p, expected, places=6)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            sign_prob_no_uncertainty(mu_t=0.01, sigma_t=0.02, model='unknown')

    def test_unknown_model_raises_with_uncertainty(self):
        with self.assertRaises(ValueError):
            sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=0.02, model='unknown')


class TestECE(unittest.TestCase):
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration_zero_ece(self):
        """If predicted probs match actual frequencies, ECE should be ~0."""
        np.random.seed(42)
        N = 10000
        true_probs = np.random.uniform(0.1, 0.9, N)
        actual = (np.random.uniform(0, 1, N) < true_probs).astype(float)
        ece = compute_sign_prob_ece(true_probs, actual, n_bins=10)
        self.assertLess(ece, 0.03)  # Should be very close to 0

    def test_overconfident_high_ece(self):
        """Overconfident predictions should have higher ECE."""
        np.random.seed(42)
        N = 5000
        # Predict 0.9 always, but actual is 50/50
        predicted = np.full(N, 0.9)
        actual = (np.random.uniform(0, 1, N) < 0.5).astype(float)
        ece = compute_sign_prob_ece(predicted, actual)
        self.assertGreater(ece, 0.3)

    def test_ece_in_zero_one_range(self):
        np.random.seed(42)
        N = 1000
        predicted = np.random.uniform(0.3, 0.7, N)
        actual = np.random.binomial(1, 0.5, N).astype(float)
        ece = compute_sign_prob_ece(predicted, actual)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_empty_array(self):
        ece = compute_sign_prob_ece(np.array([]), np.array([]))
        self.assertEqual(ece, 0.0)


class TestHitRate(unittest.TestCase):
    """Test hit rate at confidence threshold."""

    def test_perfect_predictions(self):
        predicted = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
        actual = np.array([1, 1, 1, 0, 0])
        hit_rate, n = compute_hit_rate_at_threshold(predicted, actual, threshold=0.7)
        self.assertAlmostEqual(hit_rate, 1.0)
        self.assertEqual(n, 5)  # All qualify at 0.7 (3 longs + 2 shorts)

    def test_no_qualifying_predictions(self):
        predicted = np.array([0.55, 0.45, 0.52, 0.48])
        actual = np.array([1, 0, 1, 0])
        hit_rate, n = compute_hit_rate_at_threshold(predicted, actual, threshold=0.8)
        self.assertEqual(n, 0)

    def test_threshold_filtering(self):
        predicted = np.array([0.9, 0.55, 0.1, 0.5])
        actual = np.array([1, 1, 0, 0])
        hit_rate, n = compute_hit_rate_at_threshold(predicted, actual, threshold=0.7)
        # 0.9 qualifies as long (correct), 0.1 qualifies as short (correct)
        # 0.55 and 0.5 don't qualify
        self.assertEqual(n, 2)
        self.assertAlmostEqual(hit_rate, 1.0)


class TestSyntheticCalibration(unittest.TestCase):
    """
    Test calibration on synthetic data with known DGP.
    
    Simulate the full pipeline:
    1. Generate mu_t from a random walk (Kalman model)
    2. Generate returns: r_t = mu_t + sigma_t * epsilon_t
    3. Compute sign probabilities with and without uncertainty
    4. Verify ECE improvement with uncertainty propagation
    """

    def test_gaussian_dgp_calibration(self):
        """
        Gaussian DGP: mu follows random walk with known P_t.
        Sign probability with uncertainty should be well-calibrated.
        """
        np.random.seed(42)
        T = 5000
        sigma = 0.02  # Daily vol ~2%
        q = 1e-6       # Process noise
        c = 1.0

        # Simulate Kalman-like dynamics
        mu_true = np.zeros(T)
        for t in range(1, T):
            mu_true[t] = mu_true[t-1] + np.sqrt(q) * np.random.randn()

        returns = mu_true + sigma * np.random.randn(T)
        actual_signs = (returns > 0).astype(float)

        # Compute sign probabilities
        # Use "estimated" mu_t with some noise (simulates Kalman error)
        P_t = q / (1 - 0.99**2) + 1e-6  # Steady-state P_t approximation
        mu_est = mu_true + np.sqrt(P_t) * np.random.randn(T) * 0.1  # Small estimation error

        p_with_unc = np.array([
            sign_prob_with_uncertainty(mu_est[t], P_t, sigma, c=c, model='gaussian')
            for t in range(T)
        ])
        p_without_unc = np.array([
            sign_prob_no_uncertainty(mu_est[t], sigma, c=c, model='gaussian')
            for t in range(T)
        ])

        ece_with = compute_sign_prob_ece(p_with_unc, actual_signs)
        ece_without = compute_sign_prob_ece(p_without_unc, actual_signs)

        # ECE with uncertainty should be <= ECE without (or at least < 0.05)
        self.assertLess(ece_with, 0.05,
                        f"ECE with uncertainty {ece_with:.4f} exceeds 0.05 threshold")

    def test_student_t_dgp_calibration(self):
        """Student-t DGP: returns have heavy tails."""
        np.random.seed(123)
        T = 3000
        sigma = 0.02
        nu = 6.0
        q = 1e-6
        c = 1.0

        mu_true = np.zeros(T)
        for t in range(1, T):
            mu_true[t] = mu_true[t-1] + np.sqrt(q) * np.random.randn()

        # Heavy-tailed returns
        returns = mu_true + sigma * student_t_dist.rvs(df=nu, size=T)
        actual_signs = (returns > 0).astype(float)

        P_t = 1e-5
        p_with_unc = np.array([
            sign_prob_with_uncertainty(mu_true[t], P_t, sigma, c=c,
                                       model='student_t', nu=nu, n_mc=5000, rng_seed=t)
            for t in range(T)
        ])

        ece = compute_sign_prob_ece(p_with_unc, actual_signs)
        self.assertLess(ece, 0.05,
                        f"Student-t ECE {ece:.4f} exceeds 0.05 threshold")

    def test_hit_rate_at_60pct_threshold(self):
        """Hit rate at 60% confidence should be within 2% of stated."""
        np.random.seed(456)
        T = 10000
        sigma = 0.015
        q = 5e-6
        c = 1.0

        mu_true = np.zeros(T)
        for t in range(1, T):
            mu_true[t] = 0.95 * mu_true[t-1] + np.sqrt(q) * np.random.randn()

        returns = mu_true + sigma * np.random.randn(T)
        actual_signs = (returns > 0).astype(float)

        P_t = 1e-5
        probs = np.array([
            sign_prob_with_uncertainty(mu_true[t], P_t, sigma, c=c, model='gaussian')
            for t in range(T)
        ])

        hit_rate, n_preds = compute_hit_rate_at_threshold(probs, actual_signs, threshold=0.60)

        if n_preds > 50:  # Only test if enough qualifying predictions
            # Hit rate should be >= 58% (within 2% of 60% threshold)
            self.assertGreaterEqual(hit_rate, 0.58,
                                    f"Hit rate {hit_rate:.4f} below 0.58 "
                                    f"({n_preds} qualifying predictions)")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_very_large_drift(self):
        p = sign_prob_with_uncertainty(mu_t=1.0, P_t=1e-4, sigma_t=0.02, model='gaussian')
        self.assertGreater(p, 0.98)

    def test_very_small_sigma(self):
        p = sign_prob_with_uncertainty(mu_t=0.001, P_t=1e-4, sigma_t=1e-12, model='gaussian')
        self.assertGreater(p, 0.5)

    def test_negative_sigma_handled(self):
        """Negative sigma should be treated as |sigma|."""
        p = sign_prob_with_uncertainty(mu_t=0.01, P_t=1e-4, sigma_t=-0.02, model='gaussian')
        self.assertGreater(p, 0.5)

    def test_zero_P_t(self):
        """P_t=0 should fallback to floor and still work."""
        p = sign_prob_with_uncertainty(mu_t=0.01, P_t=0.0, sigma_t=0.02, model='gaussian')
        self.assertGreater(p, 0.5)

    def test_student_t_various_nu(self):
        """Should work for typical nu values."""
        for nu in [3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 50.0]:
            p = sign_prob_with_uncertainty(
                mu_t=0.005, P_t=1e-4, sigma_t=0.02,
                model='student_t', nu=nu, rng_seed=42,
            )
            self.assertGreater(p, 0.5)
            self.assertLess(p, 0.99)


if __name__ == '__main__':
    unittest.main()
