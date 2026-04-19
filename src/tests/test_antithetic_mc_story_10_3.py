"""
Test Suite for Story 10.3: Antithetic Variates for MC Variance Reduction
========================================================================

Tests antithetic variate sampling that generates (z, -z) pairs for
variance reduction in posterior predictive MC.
"""
import os
import sys
import unittest
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.mc_variance_reduction import (
    AntitheticResult,
    antithetic_mc_sample,
    antithetic_tail_prob,
    _estimate_av_variance_reduction,
    enhanced_mc_sample,
    AV_UNIFORM_CLIP,
)


# ===================================================================
# TestAVResult: Dataclass contract
# ===================================================================

class TestAVResult(unittest.TestCase):
    """Test AntitheticResult contract."""

    def test_fields_exist(self):
        r = AntitheticResult(
            samples=np.zeros(10),
            mean_estimate=0.0,
            var_reduction=0.3,
            n_pairs=5,
            is_symmetric=True,
        )
        self.assertEqual(r.n_pairs, 5)
        self.assertTrue(r.is_symmetric)

    def test_samples_from_function(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=100,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 100)
        self.assertEqual(result.n_pairs, 50)


# ===================================================================
# TestAntitheticGaussian: Gaussian antithetic sampling
# ===================================================================

class TestAntitheticGaussian(unittest.TestCase):
    """Test antithetic sampling with Gaussian distribution."""

    def test_n_samples_correct(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 1000)

    def test_n_pairs_half_of_n(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertEqual(result.n_pairs, 500)

    def test_odd_n_rounded_down(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=1001,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 1000)

    def test_gaussian_symmetric(self):
        """Gaussian antithetic pairs should be exactly symmetric about mu."""
        result = antithetic_mc_sample(0.0, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertTrue(result.is_symmetric)

    def test_mean_close_to_mu(self):
        result = antithetic_mc_sample(0.001, 0.01, n_samples=10000,
                                       rng=np.random.default_rng(42))
        self.assertAlmostEqual(result.mean_estimate, 0.001, delta=1e-10)

    def test_gaussian_mean_exact(self):
        """For Gaussian, antithetic mean should be exactly mu (by construction)."""
        result = antithetic_mc_sample(0.005, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        # (x_i + x_anti_i)/2 = mu exactly for Gaussian (symmetric CDF)
        self.assertAlmostEqual(result.mean_estimate, 0.005, places=10)

    def test_variance_reduction_achieved(self):
        """Variance reduction ratio should be < 1 (less than iid)."""
        result = antithetic_mc_sample(0.0, 0.01, n_samples=10000,
                                       rng=np.random.default_rng(42))
        self.assertLess(result.var_reduction, 0.5,
                        f"Variance reduction {result.var_reduction:.3f} not < 0.5")

    def test_variance_reduction_significant(self):
        """MC variance of mean reduced by > 30%."""
        result = antithetic_mc_sample(0.0, 0.01, n_samples=10000,
                                       rng=np.random.default_rng(42))
        # var_reduction < 0.7 means > 30% reduction
        self.assertLess(result.var_reduction, 0.7)

    def test_samples_finite(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertTrue(np.all(np.isfinite(result.samples)))


# ===================================================================
# TestAntitheticStudentT: Student-t antithetic sampling
# ===================================================================

class TestAntitheticStudentT(unittest.TestCase):
    """Test antithetic sampling with Student-t distribution."""

    def test_n_samples_correct(self):
        result = antithetic_mc_sample(0.0, 0.01, nu=8.0, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 1000)

    def test_student_t_symmetric(self):
        """Student-t is symmetric, so antithetic pairs should be symmetric."""
        result = antithetic_mc_sample(0.0, 0.01, nu=8.0, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertTrue(result.is_symmetric)

    def test_mean_close_to_mu(self):
        result = antithetic_mc_sample(0.001, 0.01, nu=8.0, n_samples=10000,
                                       rng=np.random.default_rng(42))
        # For Student-t, exact symmetry holds too
        self.assertAlmostEqual(result.mean_estimate, 0.001, places=10)

    def test_heavier_tails_wider_range(self):
        """Lower nu should produce wider sample range."""
        result_4 = antithetic_mc_sample(0.0, 0.01, nu=4.0, n_samples=10000,
                                         rng=np.random.default_rng(42))
        result_30 = antithetic_mc_sample(0.0, 0.01, nu=30.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        range_4 = result_4.samples.max() - result_4.samples.min()
        range_30 = result_30.samples.max() - result_30.samples.min()
        self.assertGreater(range_4, range_30)

    def test_variance_reduction_student_t(self):
        result = antithetic_mc_sample(0.0, 0.01, nu=8.0, n_samples=10000,
                                       rng=np.random.default_rng(42))
        self.assertLess(result.var_reduction, 0.5)


# ===================================================================
# TestVarianceReduction: Quantified MC improvement
# ===================================================================

class TestVarianceReduction(unittest.TestCase):
    """Test MC variance reduction is quantifiably better."""

    def test_mean_variance_reduction_30pct(self):
        """MC variance of mean estimate reduced by > 30% vs iid."""
        rng = np.random.default_rng(42)
        mu, sigma = 0.0, 0.01
        n = 10000
        n_trials = 100

        av_means = []
        iid_means = []
        for trial in range(n_trials):
            seed = 1000 + trial
            av_result = antithetic_mc_sample(mu, sigma, n_samples=n,
                                              rng=np.random.default_rng(seed))
            av_means.append(av_result.mean_estimate)

            iid_samples = np.random.default_rng(seed).normal(mu, sigma, n)
            iid_means.append(float(np.mean(iid_samples)))

        var_av = np.var(av_means)
        var_iid = np.var(iid_means)

        # Antithetic should have lower variance
        reduction = 1.0 - var_av / max(var_iid, 1e-30)
        self.assertGreater(reduction, 0.30,
                           f"Variance reduction {reduction:.3f} < 0.30")

    def test_tail_prob_no_worse_than_iid(self):
        """Antithetic tail probability variance should not be significantly worse than iid.

        Note: Antithetic variates primarily reduce variance for monotone functions
        (like the mean). For indicator functions (tail probabilities), the benefit
        is smaller and can be neutral. The key property is that antithetic sampling
        does NOT degrade tail probability estimates while greatly improving mean.
        """
        mu, sigma = 0.0, 0.01
        threshold = mu - 2 * sigma
        n = 10000
        n_trials = 100

        av_probs = []
        iid_probs = []
        for trial in range(n_trials):
            seed = 2000 + trial
            av_result = antithetic_mc_sample(mu, sigma, n_samples=n,
                                              rng=np.random.default_rng(seed))
            av_probs.append(antithetic_tail_prob(av_result, threshold, "left"))

            iid_samples = np.random.default_rng(seed).normal(mu, sigma, n)
            iid_probs.append(float(np.mean(iid_samples < threshold)))

        var_av = np.var(av_probs)
        var_iid = np.var(iid_probs)

        # Antithetic should not be more than 50% worse for tail probs
        ratio = var_av / max(var_iid, 1e-30)
        self.assertLess(ratio, 1.5,
                        f"Antithetic tail variance ratio {ratio:.3f} > 1.5x worse")


# ===================================================================
# TestSymmetry: Distribution symmetry
# ===================================================================

class TestSymmetry(unittest.TestCase):
    """Test symmetry preservation in antithetic sampling."""

    def test_gaussian_exact_symmetry(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        # Pairs: samples[0::2] + samples[1::2] should = 2*mu = 0
        x = result.samples[0::2]
        x_anti = result.samples[1::2]
        pair_sums = x + x_anti
        np.testing.assert_allclose(pair_sums, 0.0, atol=1e-10)

    def test_student_t_exact_symmetry(self):
        result = antithetic_mc_sample(0.0, 0.01, nu=8.0, n_samples=1000,
                                       rng=np.random.default_rng(42))
        x = result.samples[0::2]
        x_anti = result.samples[1::2]
        pair_sums = x + x_anti
        np.testing.assert_allclose(pair_sums, 0.0, atol=1e-10)

    def test_nonzero_mu_symmetry(self):
        mu = 0.005
        result = antithetic_mc_sample(mu, 0.01, n_samples=1000,
                                       rng=np.random.default_rng(42))
        x = result.samples[0::2]
        x_anti = result.samples[1::2]
        pair_sums = x + x_anti
        np.testing.assert_allclose(pair_sums, 2 * mu, atol=1e-10)

    def test_empirical_cdf_symmetric(self):
        """Empirical CDF should be symmetric about mu."""
        result = antithetic_mc_sample(0.0, 0.01, n_samples=10000,
                                       rng=np.random.default_rng(42))
        # P(X < -t) should equal P(X > t) exactly
        t = 0.01
        p_left = np.mean(result.samples < -t)
        p_right = np.mean(result.samples > t)
        self.assertAlmostEqual(p_left, p_right, places=10)


# ===================================================================
# TestTailProb: Antithetic tail probability
# ===================================================================

class TestTailProb(unittest.TestCase):
    """Test tail probability with antithetic samples."""

    def test_left_tail(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=10000,
                                       rng=np.random.default_rng(42))
        prob = antithetic_tail_prob(result, -0.02, "left")
        self.assertGreater(prob, 0)
        self.assertLess(prob, 0.5)

    def test_right_tail(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=10000,
                                       rng=np.random.default_rng(42))
        prob = antithetic_tail_prob(result, 0.02, "right")
        self.assertGreater(prob, 0)
        self.assertLess(prob, 0.5)

    def test_gaussian_tail_accuracy(self):
        """Left tail at -2sigma should be ~2.3%."""
        from scipy.stats import norm
        mu, sigma = 0.0, 0.01
        threshold = mu - 2 * sigma
        true_prob = norm.cdf(threshold, loc=mu, scale=sigma)

        result = antithetic_mc_sample(mu, sigma, n_samples=10000,
                                       rng=np.random.default_rng(42))
        est_prob = antithetic_tail_prob(result, threshold, "left")

        rel_error = abs(est_prob - true_prob) / true_prob
        self.assertLess(rel_error, 0.15)


# ===================================================================
# TestAVVarianceReductionEstimate: Internal helper
# ===================================================================

class TestAVVarianceReductionEstimate(unittest.TestCase):
    """Test _estimate_av_variance_reduction()."""

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_anti = -x
        vr = _estimate_av_variance_reduction(x, x_anti)
        self.assertLess(vr, 0.1)

    def test_zero_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        x_anti = rng.normal(0, 1, 1000)  # Independent
        vr = _estimate_av_variance_reduction(x, x_anti)
        self.assertAlmostEqual(vr, 0.5, delta=0.15)

    def test_single_element(self):
        vr = _estimate_av_variance_reduction(np.array([1.0]), np.array([-1.0]))
        self.assertAlmostEqual(vr, 1.0)  # Not enough data

    def test_bounded_0_1(self):
        x = np.array([1.0, 2.0, 3.0])
        x_anti = np.array([-1.0, -2.0, -3.0])
        vr = _estimate_av_variance_reduction(x, x_anti)
        self.assertGreaterEqual(vr, 0.0)
        self.assertLessEqual(vr, 1.0)


# ===================================================================
# TestEnhancedMC: Combined sampler
# ===================================================================

class TestEnhancedMC(unittest.TestCase):
    """Test enhanced_mc_sample() combined sampler."""

    def test_returns_correct_length(self):
        samples = enhanced_mc_sample(0.0, 0.01, n_samples=1000,
                                      rng=np.random.default_rng(42))
        self.assertEqual(len(samples), 1000)

    def test_gaussian_with_antithetic(self):
        samples = enhanced_mc_sample(0.0, 0.01, nu=None, n_samples=1000,
                                      use_antithetic=True,
                                      rng=np.random.default_rng(42))
        self.assertEqual(len(samples), 1000)

    def test_student_t_with_antithetic(self):
        samples = enhanced_mc_sample(0.0, 0.01, nu=8.0, n_samples=1000,
                                      use_antithetic=True,
                                      rng=np.random.default_rng(42))
        self.assertEqual(len(samples), 1000)

    def test_no_antithetic_fallback(self):
        samples = enhanced_mc_sample(0.0, 0.01, nu=8.0, n_samples=1000,
                                      use_antithetic=False,
                                      rng=np.random.default_rng(42))
        self.assertEqual(len(samples), 1000)

    def test_gaussian_no_antithetic(self):
        samples = enhanced_mc_sample(0.0, 0.01, nu=None, n_samples=1000,
                                      use_antithetic=False,
                                      rng=np.random.default_rng(42))
        self.assertEqual(len(samples), 1000)

    def test_all_finite(self):
        samples = enhanced_mc_sample(0.0, 0.01, nu=4.0, n_samples=5000,
                                      rng=np.random.default_rng(42))
        self.assertTrue(np.all(np.isfinite(samples)))


# ===================================================================
# TestEdgeCases: Edge cases
# ===================================================================

class TestAVEdgeCases(unittest.TestCase):
    """Test edge cases for antithetic variates."""

    def test_very_small_sigma(self):
        result = antithetic_mc_sample(0.0, 1e-10, n_samples=100,
                                       rng=np.random.default_rng(42))
        self.assertTrue(np.all(np.isfinite(result.samples)))

    def test_large_sigma(self):
        result = antithetic_mc_sample(0.0, 100.0, n_samples=100,
                                       rng=np.random.default_rng(42))
        self.assertTrue(np.all(np.isfinite(result.samples)))

    def test_two_samples(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=2,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(result.n_pairs, 1)

    def test_large_n_samples(self):
        result = antithetic_mc_sample(0.0, 0.01, n_samples=100000,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 100000)

    def test_low_nu(self):
        result = antithetic_mc_sample(0.0, 0.01, nu=3.0, n_samples=1000,
                                       rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 1000)
        self.assertTrue(np.all(np.isfinite(result.samples)))


if __name__ == "__main__":
    unittest.main()
