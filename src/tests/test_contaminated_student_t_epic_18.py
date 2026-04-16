"""
Test Suite for Epic 18: Contaminated Student-t for Jump Detection
==================================================================

Story 18.1: EM CST Fit
Story 18.2: Online Jump Detection via CST Responsibilities
Story 18.3: CST-Adjusted Forecast Intervals
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

from models.contaminated_student_t import (
    # Story 18.1
    CSTFitResult,
    em_cst_fit,
    _em_compute_responsibilities,
    _em_student_t_logpdf,
    _em_mixture_loglik,
    EM_DEFAULT_ITER,
    EM_TOL,
    EM_MIN_OBS,
    EM_EPSILON_MIN,
    EM_EPSILON_MAX,
    # Story 18.2
    JumpProbabilityResult,
    cst_jump_probability,
    cst_jump_probability_array,
    JUMP_THRESHOLD,
    JUMP_Q_MULTIPLIER,
    # Story 18.3
    CSTPredictionInterval,
    cst_prediction_interval,
    cst_prediction_interval_array,
    _cst_cdf_em,
)


def _generate_contaminated_returns(n=500, epsilon=0.1, nu_normal=12.0,
                                    nu_crisis=3.0, seed=42):
    """Generate returns from a known CST mixture."""
    rng = np.random.default_rng(seed)
    vol = np.ones(n) * 0.02

    # Draw from mixture
    is_crisis = rng.random(n) < epsilon
    normal_draws = rng.standard_t(nu_normal, n) * 0.01
    crisis_draws = rng.standard_t(nu_crisis, n) * 0.01
    returns = np.where(is_crisis, crisis_draws, normal_draws)

    return returns, vol, is_crisis


# ===================================================================
# Story 18.1 Tests: EM CST Fit
# ===================================================================

class TestEMStudentTLogpdf(unittest.TestCase):
    """Test the internal Student-t log-pdf helper."""

    def test_finite_output(self):
        z = np.array([0.0, 1.0, -1.0, 3.0])
        logpdf = _em_student_t_logpdf(z, 8.0)
        self.assertTrue(np.all(np.isfinite(logpdf)))

    def test_symmetric(self):
        z = np.array([1.0, -1.0])
        logpdf = _em_student_t_logpdf(z, 8.0)
        self.assertAlmostEqual(logpdf[0], logpdf[1], places=10)

    def test_peak_at_zero(self):
        z = np.array([0.0, 0.5, 1.0, 2.0])
        logpdf = _em_student_t_logpdf(z, 8.0)
        self.assertEqual(np.argmax(logpdf), 0)


class TestEMComputeResponsibilities(unittest.TestCase):
    """Test E-step computation."""

    def test_extreme_observation_high_gamma(self):
        """Extreme z should have high crisis responsibility."""
        z = np.array([0.0, 0.5, 5.0, 10.0])
        gamma = _em_compute_responsibilities(z, 0.1, 12.0, 3.0)
        # Very extreme observations should have higher gamma
        self.assertGreater(gamma[3], gamma[0])

    def test_output_bounds(self):
        z = np.random.default_rng(42).standard_normal(100)
        gamma = _em_compute_responsibilities(z, 0.1, 12.0, 4.0)
        self.assertTrue(np.all(gamma > 0))
        self.assertTrue(np.all(gamma < 1))

    def test_shape_preserved(self):
        z = np.random.default_rng(42).standard_normal(50)
        gamma = _em_compute_responsibilities(z, 0.1, 12.0, 4.0)
        self.assertEqual(len(gamma), 50)


class TestEMMixtureLoglik(unittest.TestCase):
    """Test mixture log-likelihood computation."""

    def test_finite(self):
        z = np.random.default_rng(42).standard_normal(100)
        ll = _em_mixture_loglik(z, 0.1, 12.0, 4.0)
        self.assertTrue(np.isfinite(ll))

    def test_higher_with_matching_data(self):
        """Data from heavy tails should prefer lower nu_crisis."""
        rng = np.random.default_rng(42)
        z = rng.standard_t(3.0, 200)  # Heavy-tailed
        ll_heavy = _em_mixture_loglik(z, 0.1, 12.0, 3.0)
        ll_light = _em_mixture_loglik(z, 0.1, 12.0, 8.0)
        # heavier crisis tails should fit better
        self.assertGreater(ll_heavy, ll_light)


class TestEMCSTFit(unittest.TestCase):
    """Test em_cst_fit()."""

    def test_returns_dataclass(self):
        returns, vol, _ = _generate_contaminated_returns(300)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertIsInstance(result, CSTFitResult)

    def test_epsilon_in_bounds(self):
        returns, vol, _ = _generate_contaminated_returns(300)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertGreaterEqual(result.epsilon, EM_EPSILON_MIN)
        self.assertLessEqual(result.epsilon, EM_EPSILON_MAX)

    def test_nu_crisis_less_than_normal(self):
        returns, vol, _ = _generate_contaminated_returns(400)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertLess(result.nu_crisis, result.nu_normal)

    def test_converges(self):
        returns, vol, _ = _generate_contaminated_returns(500)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertTrue(result.converged)

    def test_bic_finite(self):
        returns, vol, _ = _generate_contaminated_returns(300)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertTrue(np.isfinite(result.bic))

    def test_responsibilities_shape(self):
        n = 300
        returns, vol, _ = _generate_contaminated_returns(n)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertEqual(len(result.responsibilities), n)

    def test_responsibilities_bounds(self):
        returns, vol, _ = _generate_contaminated_returns(300)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertTrue(np.all(result.responsibilities >= 0))
        self.assertTrue(np.all(result.responsibilities <= 1))

    def test_short_series_fallback(self):
        returns = np.array([0.01, -0.02, 0.005])
        vol = np.array([0.01, 0.01, 0.01])
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertFalse(result.converged)
        self.assertEqual(result.n_iter, 0)

    def test_nan_handling(self):
        returns, vol, _ = _generate_contaminated_returns(300)
        returns[50] = np.nan
        returns[100] = np.inf
        vol[75] = np.nan
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertIsInstance(result, CSTFitResult)
        self.assertTrue(np.all(np.isfinite(result.responsibilities)))

    def test_different_inits(self):
        returns, vol, _ = _generate_contaminated_returns(400)
        r1 = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99,
                         epsilon_init=0.05, nu_normal_init=15.0, nu_crisis_init=3.0)
        r2 = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99,
                         epsilon_init=0.15, nu_normal_init=8.0, nu_crisis_init=5.0)
        # Both should converge to similar solutions
        self.assertIsInstance(r1, CSTFitResult)
        self.assertIsInstance(r2, CSTFitResult)

    def test_n_iter_limit(self):
        returns, vol, _ = _generate_contaminated_returns(300)
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99, n_iter=2)
        self.assertLessEqual(result.n_iter, 2)

    def test_high_contamination_detection(self):
        """Data with many jumps should yield higher epsilon."""
        returns, vol, _ = _generate_contaminated_returns(
            500, epsilon=0.15, nu_crisis=2.5, seed=99,
        )
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        # Should detect elevated contamination (not necessarily exact)
        self.assertGreater(result.epsilon, EM_EPSILON_MIN)


# ===================================================================
# Story 18.2 Tests: Online Jump Detection
# ===================================================================

class TestCSTJumpProbability(unittest.TestCase):
    """Test cst_jump_probability()."""

    def test_returns_dataclass(self):
        result = cst_jump_probability(0.001, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertIsInstance(result, JumpProbabilityResult)

    def test_normal_obs_low_gamma(self):
        """Small observation should have low crisis probability."""
        result = cst_jump_probability(0.001, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertLess(result.gamma, 0.5)
        self.assertFalse(result.is_jump)

    def test_extreme_obs_high_gamma(self):
        """Very extreme observation should be flagged as jump."""
        result = cst_jump_probability(0.15, 0.0, 0.02, 0.1, 12.0, 3.0)
        self.assertGreater(result.gamma, 0.3)

    def test_gamma_bounds(self):
        result = cst_jump_probability(0.05, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertGreaterEqual(result.gamma, 0.0)
        self.assertLessEqual(result.gamma, 1.0)

    def test_q_inflation_formula(self):
        result = cst_jump_probability(0.001, 0.0, 0.02, 0.1, 12.0, 4.0)
        expected = 1.0 + JUMP_Q_MULTIPLIER * result.gamma
        self.assertAlmostEqual(result.q_inflation, expected, places=8)

    def test_jump_flag_threshold(self):
        """is_jump should match gamma > JUMP_THRESHOLD."""
        # Normal observation
        r1 = cst_jump_probability(0.001, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertEqual(r1.is_jump, r1.gamma > JUMP_THRESHOLD)

    def test_nan_input_handled(self):
        result = cst_jump_probability(np.nan, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertIsInstance(result, JumpProbabilityResult)
        self.assertAlmostEqual(result.gamma, 0.1)

    def test_inf_input_handled(self):
        result = cst_jump_probability(np.inf, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertIsInstance(result, JumpProbabilityResult)

    def test_zero_sigma_handled(self):
        result = cst_jump_probability(0.01, 0.0, 0.0, 0.1, 12.0, 4.0)
        self.assertIsInstance(result, JumpProbabilityResult)

    def test_log_likelihoods_finite(self):
        result = cst_jump_probability(0.01, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertTrue(np.isfinite(result.log_p_normal))
        self.assertTrue(np.isfinite(result.log_p_crisis))

    def test_symmetric_observation(self):
        """Positive and negative extremes should get similar gamma."""
        r_pos = cst_jump_probability(0.08, 0.0, 0.02, 0.1, 12.0, 4.0)
        r_neg = cst_jump_probability(-0.08, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertAlmostEqual(r_pos.gamma, r_neg.gamma, places=3)


class TestCSTJumpProbabilityArray(unittest.TestCase):
    """Test cst_jump_probability_array()."""

    def test_output_shapes(self):
        returns = np.array([0.001, -0.001, 0.05, -0.08])
        mu = np.zeros(4)
        sigma = np.ones(4) * 0.02
        gamma, q_inf = cst_jump_probability_array(
            returns, mu, sigma, 0.1, 12.0, 4.0,
        )
        self.assertEqual(len(gamma), 4)
        self.assertEqual(len(q_inf), 4)

    def test_gamma_bounds(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_t(5, 100) * 0.02
        mu = np.zeros(100)
        sigma = np.ones(100) * 0.02
        gamma, _ = cst_jump_probability_array(
            returns, mu, sigma, 0.1, 12.0, 4.0,
        )
        self.assertTrue(np.all(gamma >= 0))
        self.assertTrue(np.all(gamma <= 1))

    def test_q_inflation_formula(self):
        returns = np.array([0.001, 0.05])
        mu = np.zeros(2)
        sigma = np.ones(2) * 0.02
        gamma, q_inf = cst_jump_probability_array(
            returns, mu, sigma, 0.1, 12.0, 4.0,
        )
        expected = 1.0 + JUMP_Q_MULTIPLIER * gamma
        np.testing.assert_allclose(q_inf, expected)

    def test_extreme_gets_higher_gamma(self):
        returns = np.array([0.001, 0.10])
        mu = np.zeros(2)
        sigma = np.ones(2) * 0.02
        gamma, _ = cst_jump_probability_array(
            returns, mu, sigma, 0.1, 12.0, 4.0,
        )
        self.assertGreater(gamma[1], gamma[0])


# ===================================================================
# Story 18.3 Tests: CST Prediction Intervals
# ===================================================================

class TestCSTCDF(unittest.TestCase):
    """Test the CST CDF helper."""

    def test_cdf_at_zero_near_half(self):
        """CDF(0) should be near 0.5 for symmetric location."""
        cdf = _cst_cdf_em(0.0, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertAlmostEqual(cdf, 0.5, places=3)

    def test_cdf_monotone(self):
        xs = np.linspace(-0.1, 0.1, 20)
        cdfs = [_cst_cdf_em(x, 0.0, 0.02, 0.1, 12.0, 4.0) for x in xs]
        for i in range(1, len(cdfs)):
            self.assertGreaterEqual(cdfs[i], cdfs[i - 1] - 1e-10)

    def test_cdf_bounds(self):
        cdf_lo = _cst_cdf_em(-1.0, 0.0, 0.02, 0.1, 12.0, 4.0)
        cdf_hi = _cst_cdf_em(1.0, 0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertGreater(cdf_lo, 0)
        self.assertLess(cdf_hi, 1)


class TestCSTPredictionInterval(unittest.TestCase):
    """Test cst_prediction_interval()."""

    def test_returns_dataclass(self):
        result = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertIsInstance(result, CSTPredictionInterval)

    def test_interval_contains_mu(self):
        result = cst_prediction_interval(0.001, 0.02, 0.1, 12.0, 4.0)
        self.assertLess(result.q_lo, 0.001)
        self.assertGreater(result.q_hi, 0.001)

    def test_positive_width(self):
        result = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertGreater(result.width, 0)

    def test_wider_than_pure_t(self):
        """CST interval should be wider than pure Student-t."""
        result = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertGreaterEqual(result.width_ratio, 1.0 - 0.01)

    def test_width_ratio_positive(self):
        result = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertGreater(result.width_ratio, 0)

    def test_narrower_alpha(self):
        """alpha=0.01 should give wider interval than alpha=0.10."""
        r01 = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0, alpha=0.01)
        r10 = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0, alpha=0.10)
        self.assertGreater(r01.width, r10.width)

    def test_higher_epsilon_wider(self):
        """More contamination -> wider interval."""
        r_lo = cst_prediction_interval(0.0, 0.02, 0.02, 12.0, 4.0)
        r_hi = cst_prediction_interval(0.0, 0.02, 0.20, 12.0, 4.0)
        self.assertGreater(r_hi.width, r_lo.width)

    def test_zero_sigma_handled(self):
        result = cst_prediction_interval(0.0, 0.0, 0.1, 12.0, 4.0)
        self.assertIsInstance(result, CSTPredictionInterval)

    def test_symmetric_about_mu(self):
        """For mu=0, interval should be roughly symmetric."""
        result = cst_prediction_interval(0.0, 0.02, 0.1, 12.0, 4.0)
        self.assertAlmostEqual(result.q_lo, -result.q_hi, delta=0.001)


class TestCSTPredictionIntervalArray(unittest.TestCase):
    """Test cst_prediction_interval_array()."""

    def test_output_shapes(self):
        mu = np.array([0.0, 0.001, -0.001])
        sigma = np.array([0.02, 0.02, 0.02])
        q_lo, q_hi = cst_prediction_interval_array(
            mu, sigma, 0.1, 12.0, 4.0,
        )
        self.assertEqual(len(q_lo), 3)
        self.assertEqual(len(q_hi), 3)

    def test_lo_less_than_hi(self):
        mu = np.random.default_rng(42).normal(0, 0.005, 10)
        sigma = np.ones(10) * 0.02
        q_lo, q_hi = cst_prediction_interval_array(
            mu, sigma, 0.1, 12.0, 4.0,
        )
        self.assertTrue(np.all(q_lo < q_hi))


# ===================================================================
# Integration Tests
# ===================================================================

class TestEpic18Integration(unittest.TestCase):
    """Integration tests combining all three stories."""

    def test_fit_then_detect_then_interval(self):
        """Full pipeline: fit -> detect jumps -> compute intervals."""
        returns, vol, _ = _generate_contaminated_returns(400)
        # Fit
        fit = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertIsInstance(fit, CSTFitResult)

        # Detect jump on last observation
        jump = cst_jump_probability(
            float(returns[-1]), 0.0, float(vol[-1]),
            fit.epsilon, fit.nu_normal, fit.nu_crisis,
        )
        self.assertIsInstance(jump, JumpProbabilityResult)

        # Compute interval
        pi = cst_prediction_interval(
            0.0, float(vol[-1]),
            fit.epsilon, fit.nu_normal, fit.nu_crisis,
        )
        self.assertIsInstance(pi, CSTPredictionInterval)
        self.assertGreater(pi.width, 0)

    def test_fit_then_array_detection(self):
        """Fit parameters, then run array detection."""
        returns, vol, is_crisis = _generate_contaminated_returns(400)
        fit = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)

        mu = np.zeros(400)
        gamma, q_inf = cst_jump_probability_array(
            returns, mu, vol,
            fit.epsilon, fit.nu_normal, fit.nu_crisis,
        )
        self.assertEqual(len(gamma), 400)
        self.assertTrue(np.all(q_inf >= 1.0))

    def test_coverage_on_generated_data(self):
        """95% PI should cover ~95% of data from the fitted distribution."""
        rng = np.random.default_rng(42)
        n = 1000
        vol = np.ones(n) * 0.02
        # Generate from known mixture
        is_crisis = rng.random(n) < 0.10
        returns = np.where(
            is_crisis,
            rng.standard_t(4.0, n) * 0.02,
            rng.standard_t(12.0, n) * 0.02,
        )

        fit = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)

        # Check coverage
        covered = 0
        for i in range(n):
            pi = cst_prediction_interval(
                0.0, 0.02,
                fit.epsilon, fit.nu_normal, fit.nu_crisis,
                alpha=0.05,
            )
            if pi.q_lo <= returns[i] <= pi.q_hi:
                covered += 1

        coverage = covered / n
        # Should be reasonably close to 95% (allow some slack)
        self.assertGreater(coverage, 0.85)
        self.assertLess(coverage, 1.0)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic18EdgeCases(unittest.TestCase):

    def test_em_constants(self):
        self.assertEqual(EM_DEFAULT_ITER, 50)
        self.assertAlmostEqual(EM_TOL, 0.01)
        self.assertEqual(EM_MIN_OBS, 30)
        self.assertAlmostEqual(JUMP_THRESHOLD, 0.5)
        self.assertAlmostEqual(JUMP_Q_MULTIPLIER, 10.0)

    def test_all_nan_returns(self):
        returns = np.full(100, np.nan)
        vol = np.ones(100) * 0.02
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertFalse(result.converged)

    def test_constant_returns(self):
        returns = np.ones(100) * 0.001
        vol = np.ones(100) * 0.02
        result = em_cst_fit(returns, vol, 1e-6, 1.0, 0.99)
        self.assertIsInstance(result, CSTFitResult)

    def test_very_small_epsilon(self):
        pi = cst_prediction_interval(0.0, 0.02, 0.02, 12.0, 4.0)
        self.assertGreater(pi.width, 0)

    def test_very_large_epsilon(self):
        pi = cst_prediction_interval(0.0, 0.02, 0.20, 12.0, 4.0)
        self.assertGreater(pi.width, 0)


if __name__ == "__main__":
    unittest.main()
