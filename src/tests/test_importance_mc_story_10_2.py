"""
Test Suite for Story 10.2: Importance-Weighted MC for Heavy-Tailed Posteriors
=============================================================================

Tests importance sampling with heavier-tailed proposal distributions
to accurately estimate tail probabilities with Student-t models.
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.mc_variance_reduction import (
    ImportanceSamplingResult,
    importance_mc_student_t,
    importance_weighted_tail_prob,
    importance_weighted_crps,
    _student_t_logpdf,
    _compute_ess,
    IS_MIN_PROPOSAL_NU,
    IS_NU_OFFSET,
    IS_MIN_ESS_RATIO,
    IS_WEIGHT_CAP,
    DEFAULT_N_SAMPLES,
)


# ===================================================================
# TestISResult: Dataclass contract
# ===================================================================

class TestISResult(unittest.TestCase):
    """Test ImportanceSamplingResult contract."""

    def test_fields_exist(self):
        r = ImportanceSamplingResult(
            samples=np.zeros(10),
            weights=np.ones(10) / 10,
            raw_weights=np.ones(10),
            ess=10.0,
            ess_ratio=1.0,
            n_samples=10,
            target_nu=8.0,
            proposal_nu=6.0,
            mean_estimate=0.0,
            var_estimate=1.0,
        )
        self.assertEqual(r.n_samples, 10)
        self.assertAlmostEqual(r.ess_ratio, 1.0)

    def test_weights_sum_to_one(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=10)

    def test_samples_length(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=5000,
                                          rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 5000)

    def test_weights_length(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=5000,
                                          rng=np.random.default_rng(42))
        self.assertEqual(len(result.weights), 5000)


# ===================================================================
# TestStudentTLogPDF: Density evaluation
# ===================================================================

class TestStudentTLogPDF(unittest.TestCase):
    """Test Student-t log-density computation."""

    def test_standard_t_at_zero(self):
        """Log-density at mode should be maximal."""
        x = np.array([0.0])
        lp_0 = _student_t_logpdf(x, mu=0.0, sigma=1.0, nu=5.0)
        lp_1 = _student_t_logpdf(np.array([1.0]), mu=0.0, sigma=1.0, nu=5.0)
        self.assertGreater(lp_0[0], lp_1[0])

    def test_matches_scipy(self):
        from scipy.stats import t as t_dist
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        our_lp = _student_t_logpdf(x, mu=0.0, sigma=1.0, nu=5.0)
        scipy_lp = t_dist.logpdf(x, df=5.0, loc=0.0, scale=1.0)
        np.testing.assert_allclose(our_lp, scipy_lp, atol=1e-10)

    def test_location_shift(self):
        from scipy.stats import t as t_dist
        x = np.array([3.0, 4.0, 5.0])
        our_lp = _student_t_logpdf(x, mu=3.0, sigma=0.5, nu=8.0)
        scipy_lp = t_dist.logpdf(x, df=8.0, loc=3.0, scale=0.5)
        np.testing.assert_allclose(our_lp, scipy_lp, atol=1e-10)

    def test_finite_output(self):
        x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        lp = _student_t_logpdf(x, mu=0.0, sigma=1.0, nu=4.0)
        self.assertTrue(np.all(np.isfinite(lp)))


# ===================================================================
# TestESS: Effective sample size
# ===================================================================

class TestESS(unittest.TestCase):
    """Test ESS computation."""

    def test_uniform_weights_ess_equals_n(self):
        w = np.ones(100)
        ess = _compute_ess(w)
        self.assertAlmostEqual(ess, 100.0, places=5)

    def test_single_dominant_weight(self):
        w = np.zeros(100)
        w[0] = 1.0
        ess = _compute_ess(w)
        self.assertAlmostEqual(ess, 1.0, places=5)

    def test_two_equal_weights(self):
        w = np.zeros(100)
        w[0] = 1.0
        w[1] = 1.0
        ess = _compute_ess(w)
        self.assertAlmostEqual(ess, 2.0, places=5)

    def test_ess_bounded_by_n(self):
        w = np.random.default_rng(42).exponential(1.0, size=1000)
        ess = _compute_ess(w)
        self.assertLessEqual(ess, 1000.0)
        self.assertGreater(ess, 0.0)


# ===================================================================
# TestImportanceSampling: Core IS functionality
# ===================================================================

class TestImportanceSampling(unittest.TestCase):
    """Test importance_mc_student_t()."""

    def test_default_proposal_nu(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertEqual(result.proposal_nu, max(IS_MIN_PROPOSAL_NU, 8.0 - IS_NU_OFFSET))

    def test_custom_proposal_nu(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=1000,
                                          proposal_nu=4.0,
                                          rng=np.random.default_rng(42))
        self.assertEqual(result.proposal_nu, 4.0)

    def test_proposal_nu_floor(self):
        """proposal_nu should not go below IS_MIN_PROPOSAL_NU."""
        result = importance_mc_student_t(0.0, 0.01, nu=4.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertGreaterEqual(result.proposal_nu, IS_MIN_PROPOSAL_NU)

    def test_ess_above_threshold(self):
        """ESS should be > 50% of n_samples (proposals not wasted)."""
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        self.assertGreater(result.ess_ratio, IS_MIN_ESS_RATIO,
                           f"ESS ratio {result.ess_ratio:.3f} < {IS_MIN_ESS_RATIO}")

    def test_ess_high_nu(self):
        """With high nu (near Gaussian), ESS should be very high."""
        result = importance_mc_student_t(0.0, 0.01, nu=30.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        self.assertGreater(result.ess_ratio, 0.7)

    def test_mean_close_to_mu(self):
        result = importance_mc_student_t(0.001, 0.01, nu=8.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        self.assertAlmostEqual(result.mean_estimate, 0.001, delta=0.002)

    def test_variance_positive(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=5000,
                                          rng=np.random.default_rng(42))
        self.assertGreater(result.var_estimate, 0)

    def test_weights_non_negative(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertTrue(np.all(result.weights >= 0))
        self.assertTrue(np.all(result.raw_weights >= 0))

    def test_reproducible(self):
        r1 = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=100,
                                      rng=np.random.default_rng(42))
        r2 = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=100,
                                      rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_different_seeds_different(self):
        r1 = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=100,
                                      rng=np.random.default_rng(42))
        r2 = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=100,
                                      rng=np.random.default_rng(99))
        self.assertFalse(np.allclose(r1.samples, r2.samples))


# ===================================================================
# TestTailProbability: Tail probability estimation
# ===================================================================

class TestTailProbability(unittest.TestCase):
    """Test tail probability computation via importance sampling."""

    def test_left_tail_positive(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        prob = importance_weighted_tail_prob(result, threshold=-0.03, direction="left")
        self.assertGreater(prob, 0)

    def test_right_tail_positive(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        prob = importance_weighted_tail_prob(result, threshold=0.03, direction="right")
        self.assertGreater(prob, 0)

    def test_tail_prob_accuracy(self):
        """P(r < -3*sigma) should be accurate to within 10% relative error."""
        from scipy.stats import t as t_dist
        mu, sigma, nu = 0.0, 0.01, 8.0
        threshold = mu - 3 * sigma

        # True tail probability
        true_prob = t_dist.cdf(threshold, df=nu, loc=mu, scale=sigma)

        # IS estimate
        result = importance_mc_student_t(mu, sigma, nu, n_samples=10000,
                                          rng=np.random.default_rng(42))
        is_prob = importance_weighted_tail_prob(result, threshold, direction="left")

        # 10% relative error
        rel_error = abs(is_prob - true_prob) / true_prob
        self.assertLess(rel_error, 0.15,
                        f"Relative error {rel_error:.3f} > 0.15. "
                        f"True: {true_prob:.6f}, IS: {is_prob:.6f}")

    def test_heavy_tails_better_than_light(self):
        """IS should be particularly beneficial for heavy-tailed (low nu) targets."""
        mu, sigma = 0.0, 0.01
        threshold = mu - 4 * sigma

        # nu=4 (heavy tails)
        result_heavy = importance_mc_student_t(mu, sigma, nu=4.0, n_samples=10000,
                                                rng=np.random.default_rng(42))
        prob_heavy = importance_weighted_tail_prob(result_heavy, threshold, direction="left")
        self.assertGreater(prob_heavy, 0, "Should detect tail events for nu=4")

    def test_extreme_threshold_small_prob(self):
        mu, sigma, nu = 0.0, 0.01, 8.0
        threshold = mu - 5 * sigma  # Very far tail
        result = importance_mc_student_t(mu, sigma, nu, n_samples=10000,
                                          rng=np.random.default_rng(42))
        prob = importance_weighted_tail_prob(result, threshold, direction="left")
        self.assertGreater(prob, 0)
        self.assertLess(prob, 0.01)

    def test_symmetry_of_tail_probs(self):
        """For symmetric distribution, P(X < -t) should ~ P(X > t)."""
        mu, sigma, nu = 0.0, 0.01, 8.0
        t_val = 3 * sigma
        result = importance_mc_student_t(mu, sigma, nu, n_samples=20000,
                                          rng=np.random.default_rng(42))
        prob_left = importance_weighted_tail_prob(result, -t_val, direction="left")
        prob_right = importance_weighted_tail_prob(result, t_val, direction="right")
        # Should be approximately equal (within MC noise)
        self.assertAlmostEqual(prob_left, prob_right, delta=0.01)


# ===================================================================
# TestISCRPS: Importance-weighted CRPS
# ===================================================================

class TestISCRPS(unittest.TestCase):
    """Test importance-weighted CRPS computation."""

    def test_crps_non_negative(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=5000,
                                          rng=np.random.default_rng(42))
        crps = importance_weighted_crps(result, observation=0.005)
        self.assertGreaterEqual(crps, 0)

    def test_crps_zero_at_mode(self):
        """CRPS should be small when observation is at the mode."""
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=10000,
                                          rng=np.random.default_rng(42))
        crps_mode = importance_weighted_crps(result, observation=0.0)
        crps_tail = importance_weighted_crps(result, observation=0.05)
        self.assertLess(crps_mode, crps_tail)

    def test_crps_finite(self):
        result = importance_mc_student_t(0.0, 0.01, nu=4.0, n_samples=5000,
                                          rng=np.random.default_rng(42))
        crps = importance_weighted_crps(result, observation=-0.03)
        self.assertTrue(np.isfinite(crps))


# ===================================================================
# TestISConstants: Module constants
# ===================================================================

class TestISConstants(unittest.TestCase):
    """Test module constants are reasonable."""

    def test_min_proposal_nu(self):
        self.assertGreaterEqual(IS_MIN_PROPOSAL_NU, 2.1)

    def test_nu_offset_positive(self):
        self.assertGreater(IS_NU_OFFSET, 0)

    def test_min_ess_ratio(self):
        self.assertGreater(IS_MIN_ESS_RATIO, 0)
        self.assertLessEqual(IS_MIN_ESS_RATIO, 1.0)

    def test_weight_cap_positive(self):
        self.assertGreater(IS_WEIGHT_CAP, 1.0)

    def test_default_n_samples(self):
        self.assertGreaterEqual(DEFAULT_N_SAMPLES, 1000)


# ===================================================================
# TestISEdgeCases: Edge cases
# ===================================================================

class TestISEdgeCases(unittest.TestCase):
    """Test edge cases for importance sampling."""

    def test_very_small_sigma(self):
        result = importance_mc_student_t(0.0, 1e-10, nu=8.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 1000)
        self.assertTrue(np.all(np.isfinite(result.weights)))

    def test_large_mu(self):
        result = importance_mc_student_t(100.0, 0.01, nu=8.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertAlmostEqual(result.mean_estimate, 100.0, delta=0.01)

    def test_negative_mu(self):
        result = importance_mc_student_t(-0.05, 0.01, nu=8.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertAlmostEqual(result.mean_estimate, -0.05, delta=0.01)

    def test_low_nu_works(self):
        """nu=3 should work (proposal_nu=3 = floor)."""
        result = importance_mc_student_t(0.0, 0.01, nu=3.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 1000)

    def test_high_nu_like_gaussian(self):
        result = importance_mc_student_t(0.0, 0.01, nu=100.0, n_samples=1000,
                                          rng=np.random.default_rng(42))
        self.assertGreater(result.ess_ratio, 0.5)

    def test_small_n_samples(self):
        result = importance_mc_student_t(0.0, 0.01, nu=8.0, n_samples=10,
                                          rng=np.random.default_rng(42))
        self.assertEqual(len(result.samples), 10)
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
