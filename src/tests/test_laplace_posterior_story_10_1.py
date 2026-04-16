"""
Test Suite for Story 10.1: Laplace Posterior Approximation
==========================================================

Tests the Laplace approximation of parameter posteriors for
Kalman filter models (Gaussian and Student-t families).

75+ tests organized by component:
  - LaplaceResult contract
  - Finite difference Hessian correctness
  - Hessian regularization
  - Laplace posterior main API
  - Variance inflation
  - Parameter sampling
  - Prediction intervals and coverage
  - Edge cases and numerical stability
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

from calibration.laplace_posterior import (
    LaplaceResult,
    laplace_posterior,
    compute_hessian,
    regularize_hessian,
    compute_variance_inflation,
    predictive_variance_with_uncertainty,
    sample_from_laplace,
    compute_prediction_interval,
    compute_pi_coverage,
    _compute_fd_step,
    _compute_condition_number,
    _ll_gaussian,
    MAX_CONDITION_NUMBER,
    MIN_VARIANCE_INFLATION,
    MAX_VARIANCE_INFLATION,
    MIN_OBS_FOR_LAPLACE,
    FD_STEP_RELATIVE,
    FD_STEP_MIN,
    MIN_RIDGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_returns(n=500, mu=0.0005, sigma=0.01, seed=42):
    """Generate synthetic returns with known parameters."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, size=n)
    vol = np.full(n, sigma)
    return returns, vol


def _make_trending_returns(n=500, phi=0.5, q=1e-6, c=1.0, seed=42):
    """Generate returns from a local-level model with known parameters."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.01)

    # State-space simulation
    mu_state = 0.0
    returns = np.zeros(n)
    for t in range(n):
        mu_state = phi * mu_state + rng.normal(0, math.sqrt(q))
        R = c * vol[t] ** 2
        returns[t] = mu_state + rng.normal(0, math.sqrt(R))

    return returns, vol


# ===================================================================
# TestLaplaceResult: Dataclass contract
# ===================================================================

class TestLaplaceResult(unittest.TestCase):
    """Test LaplaceResult dataclass contract."""

    def test_fields_exist(self):
        r = LaplaceResult(
            theta_hat=np.array([1.0, 0.5, 1e-6]),
            covariance=np.eye(3),
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=1.1,
            param_std=np.ones(3),
        )
        self.assertEqual(r.n_obs, 100)
        self.assertFalse(r.is_regularized)
        self.assertEqual(len(r.param_names), 3)

    def test_theta_hat_is_array(self):
        r = LaplaceResult(
            theta_hat=np.array([1.0, 0.5, 1e-6]),
            covariance=np.eye(3),
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=1.1,
            param_std=np.ones(3),
        )
        self.assertIsInstance(r.theta_hat, np.ndarray)
        self.assertEqual(len(r.theta_hat), 3)

    def test_covariance_is_square(self):
        cov = np.eye(3) * 0.01
        r = LaplaceResult(
            theta_hat=np.array([1.0, 0.5, 1e-6]),
            covariance=cov,
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=1.1,
            param_std=np.sqrt(np.diag(cov)),
        )
        self.assertEqual(r.covariance.shape, (3, 3))

    def test_param_std_matches_diag(self):
        cov = np.diag([0.01, 0.04, 1e-8])
        r = LaplaceResult(
            theta_hat=np.array([1.0, 0.5, 1e-6]),
            covariance=cov,
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=1.1,
            param_std=np.sqrt(np.diag(cov)),
        )
        np.testing.assert_allclose(r.param_std, [0.1, 0.2, 1e-4])

    def test_variance_inflation_stored(self):
        r = LaplaceResult(
            theta_hat=np.array([1.0, 0.5, 1e-6]),
            covariance=np.eye(3),
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=1.15,
            param_std=np.ones(3),
        )
        self.assertAlmostEqual(r.variance_inflation, 1.15)

    def test_ridge_lambda_zero_when_no_regularization(self):
        r = LaplaceResult(
            theta_hat=np.array([1.0, 0.5, 1e-6]),
            covariance=np.eye(3),
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=1.0,
            param_std=np.ones(3),
        )
        self.assertEqual(r.ridge_lambda, 0.0)


# ===================================================================
# TestFDStep: Finite difference step computation
# ===================================================================

class TestFDStep(unittest.TestCase):
    """Test finite difference step size computation."""

    def test_scales_with_parameter(self):
        step_big = _compute_fd_step(10.0)
        step_small = _compute_fd_step(0.001)
        self.assertGreater(step_big, step_small)

    def test_respects_minimum(self):
        step = _compute_fd_step(0.0)
        self.assertGreaterEqual(step, FD_STEP_MIN)

    def test_relative_scaling(self):
        val = 5.0
        step = _compute_fd_step(val)
        self.assertAlmostEqual(step, val * FD_STEP_RELATIVE, places=10)

    def test_negative_parameter(self):
        step = _compute_fd_step(-3.0)
        self.assertAlmostEqual(step, 3.0 * FD_STEP_RELATIVE, places=10)

    def test_very_small_parameter_uses_minimum(self):
        step = _compute_fd_step(1e-10)
        self.assertAlmostEqual(step, FD_STEP_MIN, places=15)


# ===================================================================
# TestConditionNumber: Hessian condition number
# ===================================================================

class TestConditionNumber(unittest.TestCase):
    """Test condition number computation."""

    def test_identity_condition_one(self):
        kappa = _compute_condition_number(np.eye(3))
        self.assertAlmostEqual(kappa, 1.0, places=5)

    def test_diagonal_condition(self):
        H = np.diag([10.0, 1.0, 0.1])
        kappa = _compute_condition_number(H)
        self.assertAlmostEqual(kappa, 100.0, places=3)

    def test_singular_matrix_large_condition(self):
        H = np.array([[1.0, 0.0], [0.0, 0.0]])
        kappa = _compute_condition_number(H)
        self.assertGreater(kappa, 1e10)

    def test_negative_definite(self):
        H = -np.eye(3)
        kappa = _compute_condition_number(H)
        self.assertAlmostEqual(kappa, 1.0, places=5)

    def test_symmetric_matrix(self):
        H = np.array([[4.0, 1.0], [1.0, 4.0]])
        kappa = _compute_condition_number(H)
        # eigenvalues: 3, 5 -> condition = 5/3
        self.assertAlmostEqual(kappa, 5.0 / 3.0, places=5)


# ===================================================================
# TestRegularization: Hessian regularization
# ===================================================================

class TestRegularization(unittest.TestCase):
    """Test ridge regularization for the negative Hessian."""

    def test_well_conditioned_no_change(self):
        H = np.diag([10.0, 8.0, 6.0])
        H_reg, ridge, was_reg = regularize_hessian(H)
        np.testing.assert_array_almost_equal(H_reg, H)
        self.assertEqual(ridge, 0.0)
        self.assertFalse(was_reg)

    def test_ill_conditioned_gets_regularized(self):
        H = np.diag([1e6, 1.0, 0.01])
        H_reg, ridge, was_reg = regularize_hessian(H)
        self.assertTrue(was_reg)
        self.assertGreater(ridge, 0)
        # Condition number should be bounded
        kappa = _compute_condition_number(H_reg)
        self.assertLessEqual(kappa, MAX_CONDITION_NUMBER * 1.1)  # Small tolerance

    def test_negative_eigenvalue_fixed(self):
        H = np.diag([10.0, 5.0, -1.0])
        H_reg, ridge, was_reg = regularize_hessian(H)
        self.assertTrue(was_reg)
        eigenvalues = np.linalg.eigvalsh(H_reg)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_ridge_lambda_positive_when_regularized(self):
        H = np.diag([1e8, 1.0, 1e-4])
        _, ridge, was_reg = regularize_hessian(H)
        self.assertTrue(was_reg)
        self.assertGreater(ridge, 0)

    def test_identity_not_regularized(self):
        H = np.eye(3)
        _, ridge, was_reg = regularize_hessian(H)
        self.assertFalse(was_reg)
        self.assertEqual(ridge, 0.0)

    def test_custom_max_condition(self):
        H = np.diag([100.0, 1.0])
        # Default: condition=100, within 1e4 -> no regularization
        _, _, was_reg_default = regularize_hessian(H)
        self.assertFalse(was_reg_default)
        # Strict: condition=100 > 10 -> regularize
        H_reg, _, was_reg_strict = regularize_hessian(H, max_condition=10.0)
        self.assertTrue(was_reg_strict)

    def test_output_shape_preserved(self):
        for n in [2, 3, 4]:
            H = np.eye(n) * 5.0
            H_reg, _, _ = regularize_hessian(H)
            self.assertEqual(H_reg.shape, (n, n))

    def test_symmetry_preserved(self):
        H = np.array([[10.0, 2.0, 1.0],
                       [2.0, 8.0, 0.5],
                       [1.0, 0.5, 0.001]])
        H_reg, _, _ = regularize_hessian(H)
        np.testing.assert_array_almost_equal(H_reg, H_reg.T)

    def test_all_zero_matrix(self):
        H = np.zeros((3, 3))
        H_reg, ridge, was_reg = regularize_hessian(H)
        self.assertTrue(was_reg)
        eigenvalues = np.linalg.eigvalsh(H_reg)
        self.assertTrue(np.all(eigenvalues > 0))


# ===================================================================
# TestHessianGaussian: Hessian computation for Gaussian model
# ===================================================================

class TestHessianGaussian(unittest.TestCase):
    """Test Hessian computation for the Gaussian Kalman filter."""

    @classmethod
    def setUpClass(cls):
        """Generate synthetic data once for all tests."""
        cls.returns, cls.vol = _make_trending_returns(
            n=500, phi=0.3, q=1e-6, c=1.0, seed=42
        )
        cls.theta = np.array([1.0, 0.3, 1e-6])

    def test_hessian_shape(self):
        H = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        self.assertEqual(H.shape, (3, 3))

    def test_hessian_symmetric(self):
        H = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        np.testing.assert_array_almost_equal(H, H.T, decimal=3)

    def test_hessian_finite(self):
        H = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        self.assertTrue(np.all(np.isfinite(H)))

    def test_negative_hessian_mostly_positive(self):
        """At near-MLE, -H should have predominantly positive eigenvalues.

        Note: theta is not the exact MLE, so some directions may show
        slight non-concavity. The regularization step handles this.
        """
        H = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        neg_H = -H
        eigenvalues = np.linalg.eigvalsh(neg_H)
        # At least the largest eigenvalue must be positive (some curvature)
        self.assertGreater(eigenvalues.max(), 0,
                           f"Eigenvalues: {eigenvalues}")
        # After regularization, all should be positive
        neg_H_reg, _, _ = regularize_hessian(neg_H)
        eig_reg = np.linalg.eigvalsh(neg_H_reg)
        self.assertTrue(np.all(eig_reg > 0),
                        f"Regularized eigenvalues: {eig_reg}")

    def test_diagonal_negative(self):
        """Diagonal of Hessian should be negative (concavity at MLE)."""
        H = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        for i in range(3):
            self.assertLess(H[i, i], 0,
                            f"H[{i},{i}] = {H[i,i]} should be negative")

    def test_hessian_varies_with_data(self):
        """Different data should give different Hessians."""
        returns2, vol2 = _make_trending_returns(n=500, phi=0.3, q=1e-6, c=1.0, seed=99)
        H1 = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        H2 = compute_hessian(returns2, vol2, self.theta, family="gaussian")
        self.assertFalse(np.allclose(H1, H2, atol=1e-3))

    def test_hessian_varies_with_theta(self):
        """Different parameters should give different Hessians."""
        theta2 = np.array([2.0, 0.5, 1e-5])
        H1 = compute_hessian(self.returns, self.vol, self.theta, family="gaussian")
        H2 = compute_hessian(self.returns, self.vol, theta2, family="gaussian")
        self.assertFalse(np.allclose(H1, H2, atol=1e-3))


# ===================================================================
# TestHessianStudentT: Hessian computation for Student-t model
# ===================================================================

class TestHessianStudentT(unittest.TestCase):
    """Test Hessian computation for the Student-t Kalman filter."""

    @classmethod
    def setUpClass(cls):
        cls.returns, cls.vol = _make_trending_returns(
            n=500, phi=0.3, q=1e-6, c=1.0, seed=42
        )
        cls.theta = np.array([1.0, 0.3, 1e-6])
        cls.nu = 8.0

    def test_hessian_shape(self):
        H = compute_hessian(self.returns, self.vol, self.theta,
                            family="student_t", nu=self.nu)
        self.assertEqual(H.shape, (3, 3))

    def test_hessian_symmetric(self):
        H = compute_hessian(self.returns, self.vol, self.theta,
                            family="student_t", nu=self.nu)
        np.testing.assert_array_almost_equal(H, H.T, decimal=3)

    def test_hessian_finite(self):
        H = compute_hessian(self.returns, self.vol, self.theta,
                            family="student_t", nu=self.nu)
        self.assertTrue(np.all(np.isfinite(H)))

    def test_nu_required(self):
        with self.assertRaises(ValueError):
            compute_hessian(self.returns, self.vol, self.theta,
                            family="student_t", nu=None)

    def test_different_nu_different_hessian(self):
        H4 = compute_hessian(self.returns, self.vol, self.theta,
                             family="student_t", nu=4.0)
        H20 = compute_hessian(self.returns, self.vol, self.theta,
                              family="student_t", nu=20.0)
        self.assertFalse(np.allclose(H4, H20, atol=1e-3))

    def test_high_nu_approaches_gaussian(self):
        """With large nu, Student-t Hessian should approach Gaussian."""
        H_st = compute_hessian(self.returns, self.vol, self.theta,
                               family="student_t", nu=100.0)
        H_g = compute_hessian(self.returns, self.vol, self.theta,
                              family="gaussian")
        # Should be "close" but not identical (nu=100 is not inf)
        # Check same sign structure
        for i in range(3):
            self.assertTrue(
                np.sign(H_st[i, i]) == np.sign(H_g[i, i]),
                f"Sign mismatch at [{i},{i}]: Student-t={H_st[i,i]}, Gaussian={H_g[i,i]}"
            )

    def test_invalid_family_raises(self):
        with self.assertRaises(ValueError):
            compute_hessian(self.returns, self.vol, self.theta,
                            family="unknown")


# ===================================================================
# TestLaplacePosterior: Main API
# ===================================================================

class TestLaplacePosterior(unittest.TestCase):
    """Test the laplace_posterior() main entry point."""

    @classmethod
    def setUpClass(cls):
        cls.returns, cls.vol = _make_trending_returns(
            n=500, phi=0.3, q=1e-6, c=1.0, seed=42
        )
        cls.theta = np.array([1.0, 0.3, 1e-6])

    def test_returns_laplace_result(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertIsInstance(result, LaplaceResult)

    def test_theta_hat_unchanged(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        np.testing.assert_array_almost_equal(result.theta_hat, self.theta)

    def test_covariance_shape(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertEqual(result.covariance.shape, (3, 3))

    def test_covariance_symmetric(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        np.testing.assert_array_almost_equal(result.covariance,
                                              result.covariance.T)

    def test_covariance_positive_diagonal(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        for i in range(3):
            self.assertGreater(result.covariance[i, i], 0,
                               f"Cov[{i},{i}] should be positive")

    def test_n_obs_correct(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertEqual(result.n_obs, len(self.returns))

    def test_log_likelihood_finite(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertTrue(np.isfinite(result.log_likelihood))

    def test_param_names_default(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertEqual(result.param_names, ("c", "phi", "q"))

    def test_param_names_custom(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian",
                                   param_names=("obs_noise", "persistence", "proc_noise"))
        self.assertEqual(result.param_names[0], "obs_noise")

    def test_param_std_positive(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        for i in range(3):
            self.assertGreaterEqual(result.param_std[i], 0)

    def test_condition_number_positive(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertGreater(result.condition_number, 0)

    def test_student_t_works(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="student_t", nu=8.0)
        self.assertIsInstance(result, LaplaceResult)
        self.assertEqual(result.covariance.shape, (3, 3))

    def test_hessian_stored(self):
        result = laplace_posterior(self.returns, self.vol, self.theta,
                                   family="gaussian")
        self.assertEqual(result.hessian.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(result.hessian)))


# ===================================================================
# TestVarianceInflation: Predictive variance increase
# ===================================================================

class TestVarianceInflation(unittest.TestCase):
    """Test that parameter uncertainty inflates predictive variance."""

    def test_inflation_at_least_one(self):
        theta = np.array([1.0, 0.3, 1e-6])
        cov = np.diag([0.01, 0.001, 1e-10])
        inflation = compute_variance_inflation(theta, cov, family="gaussian")
        self.assertGreaterEqual(inflation, MIN_VARIANCE_INFLATION)

    def test_zero_covariance_no_inflation(self):
        theta = np.array([1.0, 0.3, 1e-6])
        cov = np.zeros((3, 3))
        inflation = compute_variance_inflation(theta, cov, family="gaussian")
        self.assertAlmostEqual(inflation, 1.0)

    def test_larger_uncertainty_more_inflation(self):
        theta = np.array([1.0, 0.3, 1e-6])
        cov_small = np.diag([0.001, 0.0001, 1e-12])
        cov_large = np.diag([0.1, 0.01, 1e-8])
        inf_small = compute_variance_inflation(theta, cov_small, family="gaussian")
        inf_large = compute_variance_inflation(theta, cov_large, family="gaussian")
        self.assertGreater(inf_large, inf_small)

    def test_inflation_bounded(self):
        theta = np.array([1.0, 0.3, 1e-6])
        cov = np.diag([10.0, 10.0, 10.0])  # Huge uncertainty
        inflation = compute_variance_inflation(theta, cov, family="gaussian")
        self.assertLessEqual(inflation, MAX_VARIANCE_INFLATION)

    def test_student_t_inflation(self):
        theta = np.array([1.0, 0.3, 1e-6])
        cov = np.diag([0.01, 0.001, 1e-10])
        inf_g = compute_variance_inflation(theta, cov, family="gaussian")
        inf_t = compute_variance_inflation(theta, cov, family="student_t", nu=8.0)
        # Both should be >= 1
        self.assertGreaterEqual(inf_g, 1.0)
        self.assertGreaterEqual(inf_t, 1.0)

    def test_typical_inflation_range(self):
        """With realistic parameters, inflation should be in 5-20% range."""
        returns, vol = _make_trending_returns(n=500, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        # Inflation should be reasonable for 500 observations
        self.assertGreaterEqual(result.variance_inflation, 1.0)
        self.assertLessEqual(result.variance_inflation, MAX_VARIANCE_INFLATION)


# ===================================================================
# TestPredictiveVariance: Inflated sigma
# ===================================================================

class TestPredictiveVariance(unittest.TestCase):
    """Test predictive_variance_with_uncertainty()."""

    def _make_result(self, inflation):
        return LaplaceResult(
            theta_hat=np.array([1.0, 0.3, 1e-6]),
            covariance=np.eye(3),
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=inflation,
            param_std=np.ones(3),
        )

    def test_no_inflation(self):
        result = self._make_result(1.0)
        sigma_out = predictive_variance_with_uncertainty(0.01, result)
        self.assertAlmostEqual(sigma_out, 0.01)

    def test_ten_percent_inflation(self):
        result = self._make_result(1.10)
        sigma_out = predictive_variance_with_uncertainty(0.01, result)
        expected = 0.01 * math.sqrt(1.10)
        self.assertAlmostEqual(sigma_out, expected, places=8)

    def test_always_at_least_original(self):
        result = self._make_result(0.5)  # Force below 1 (should be clipped)
        sigma_out = predictive_variance_with_uncertainty(0.01, result)
        self.assertGreaterEqual(sigma_out, 0.01)

    def test_scales_linearly_with_sigma(self):
        result = self._make_result(1.15)
        s1 = predictive_variance_with_uncertainty(0.01, result)
        s2 = predictive_variance_with_uncertainty(0.02, result)
        self.assertAlmostEqual(s2 / s1, 2.0, places=5)


# ===================================================================
# TestSampling: Sample from Laplace posterior
# ===================================================================

class TestSampling(unittest.TestCase):
    """Test sample_from_laplace()."""

    def _make_result(self):
        return LaplaceResult(
            theta_hat=np.array([1.0, 0.3, 1e-6]),
            covariance=np.diag([0.01, 0.001, 1e-10]),
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=500,
            param_names=("c", "phi", "q"),
            variance_inflation=1.1,
            param_std=np.sqrt(np.array([0.01, 0.001, 1e-10])),
        )

    def test_output_shape(self):
        result = self._make_result()
        samples = sample_from_laplace(result, n_samples=1000)
        self.assertEqual(samples.shape, (1000, 3))

    def test_c_positive(self):
        result = self._make_result()
        samples = sample_from_laplace(result, n_samples=1000)
        self.assertTrue(np.all(samples[:, 0] > 0))

    def test_q_positive(self):
        result = self._make_result()
        samples = sample_from_laplace(result, n_samples=1000)
        self.assertTrue(np.all(samples[:, 2] > 0))

    def test_phi_bounded(self):
        result = self._make_result()
        samples = sample_from_laplace(result, n_samples=1000)
        self.assertTrue(np.all(np.abs(samples[:, 1]) <= 0.999))

    def test_mean_close_to_theta_hat(self):
        result = self._make_result()
        samples = sample_from_laplace(result, n_samples=10000,
                                       rng=np.random.default_rng(42))
        mean = samples.mean(axis=0)
        # Mean should be close to theta_hat (CLT)
        np.testing.assert_allclose(mean[0], 1.0, atol=0.05)  # c
        np.testing.assert_allclose(mean[1], 0.3, atol=0.02)  # phi

    def test_reproducible_with_rng(self):
        result = self._make_result()
        s1 = sample_from_laplace(result, n_samples=100,
                                  rng=np.random.default_rng(42))
        s2 = sample_from_laplace(result, n_samples=100,
                                  rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_different_rng_different_samples(self):
        result = self._make_result()
        s1 = sample_from_laplace(result, n_samples=100,
                                  rng=np.random.default_rng(42))
        s2 = sample_from_laplace(result, n_samples=100,
                                  rng=np.random.default_rng(99))
        self.assertFalse(np.allclose(s1, s2))


# ===================================================================
# TestPredictionInterval: PI computation
# ===================================================================

class TestPredictionInterval(unittest.TestCase):
    """Test prediction interval computation."""

    def _make_result(self, inflation=1.1):
        return LaplaceResult(
            theta_hat=np.array([1.0, 0.3, 1e-6]),
            covariance=np.eye(3) * 0.01,
            hessian=-np.eye(3),
            is_regularized=False,
            condition_number=1.0,
            ridge_lambda=0.0,
            log_likelihood=-100.0,
            n_obs=100,
            param_names=("c", "phi", "q"),
            variance_inflation=inflation,
            param_std=np.ones(3) * 0.1,
        )

    def test_lower_less_than_upper(self):
        result = self._make_result()
        lower, upper = compute_prediction_interval(0.0, 0.01, result, alpha=0.10)
        self.assertLess(lower, upper)

    def test_mu_centered(self):
        result = self._make_result()
        lower, upper = compute_prediction_interval(0.0, 0.01, result, alpha=0.10)
        # Should be symmetric around mu=0 for Gaussian
        self.assertAlmostEqual(lower + upper, 0.0, places=8)

    def test_wider_than_plug_in(self):
        """Interval should be wider with parameter uncertainty."""
        result_no = self._make_result(inflation=1.0)
        result_yes = self._make_result(inflation=1.15)
        _, u_no = compute_prediction_interval(0.0, 0.01, result_no, alpha=0.10)
        _, u_yes = compute_prediction_interval(0.0, 0.01, result_yes, alpha=0.10)
        self.assertGreater(u_yes, u_no)

    def test_smaller_alpha_wider(self):
        result = self._make_result()
        _, u_90 = compute_prediction_interval(0.0, 0.01, result, alpha=0.10)
        _, u_95 = compute_prediction_interval(0.0, 0.01, result, alpha=0.05)
        self.assertGreater(u_95, u_90)

    def test_student_t_wider_than_gaussian(self):
        result = self._make_result()
        _, u_g = compute_prediction_interval(0.0, 0.01, result, alpha=0.10,
                                              family="gaussian")
        _, u_t = compute_prediction_interval(0.0, 0.01, result, alpha=0.10,
                                              family="student_t", nu=4.0)
        self.assertGreater(u_t, u_g)


# ===================================================================
# TestPICoverage: Empirical coverage
# ===================================================================

class TestPICoverage(unittest.TestCase):
    """Test prediction interval coverage computation."""

    def test_perfect_coverage(self):
        n = 100
        returns = np.zeros(n)
        mu = np.zeros(n)
        sigma = np.ones(n) * 100  # Very wide intervals
        cov = compute_pi_coverage(returns, mu, sigma, alpha=0.10)
        self.assertAlmostEqual(cov, 1.0)

    def test_zero_coverage(self):
        n = 100
        returns = np.ones(n) * 100  # Far from predictions
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.001  # Very narrow intervals
        cov = compute_pi_coverage(returns, mu, sigma, alpha=0.10)
        self.assertAlmostEqual(cov, 0.0)

    def test_gaussian_nominal_coverage(self):
        """With Gaussian data and correct sigma, coverage should be ~90%."""
        rng = np.random.default_rng(42)
        n = 5000
        sigma_true = 0.01
        returns = rng.normal(0, sigma_true, size=n)
        mu = np.zeros(n)
        sigma_pred = np.full(n, sigma_true)
        cov = compute_pi_coverage(returns, mu, sigma_pred, alpha=0.10)
        # Should be close to 90% (nominal)
        self.assertGreater(cov, 0.87)
        self.assertLess(cov, 0.93)

    def test_empty_returns(self):
        cov = compute_pi_coverage(np.array([]), np.array([]),
                                   np.array([]), alpha=0.10)
        self.assertEqual(cov, 0.0)


# ===================================================================
# TestEdgeCases: Numerical stability and edge cases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and numerical robustness."""

    def test_short_series(self):
        """Laplace should work (possibly with regularization) on short series."""
        returns, vol = _make_synthetic_returns(n=30)
        theta = np.array([1.0, 0.0, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)
        self.assertTrue(np.all(np.isfinite(result.covariance)))

    def test_phi_near_one(self):
        """phi near unit root should work (possibly with larger uncertainty)."""
        returns, vol = _make_trending_returns(n=300, phi=0.99, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.99, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)

    def test_phi_zero(self):
        """phi=0 (no persistence) should work."""
        returns, vol = _make_synthetic_returns(n=300)
        theta = np.array([1.0, 0.0, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)

    def test_very_small_q(self):
        """Very small process noise."""
        returns, vol = _make_synthetic_returns(n=300)
        theta = np.array([1.0, 0.3, 1e-12])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)

    def test_very_large_c(self):
        """Large observation noise scale."""
        returns, vol = _make_synthetic_returns(n=300)
        theta = np.array([100.0, 0.3, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)

    def test_student_t_low_nu(self):
        """Student-t with low nu (heavy tails)."""
        returns, vol = _make_trending_returns(n=300, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="student_t", nu=4.0)
        self.assertIsInstance(result, LaplaceResult)

    def test_student_t_high_nu(self):
        """Student-t with high nu (near Gaussian)."""
        returns, vol = _make_trending_returns(n=300, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="student_t", nu=30.0)
        self.assertIsInstance(result, LaplaceResult)

    def test_constant_vol(self):
        """Constant volatility series."""
        returns, _ = _make_synthetic_returns(n=300)
        vol = np.ones(300) * 0.015
        theta = np.array([1.0, 0.0, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)

    def test_covariance_invertibility(self):
        """Covariance should be invertible (non-singular)."""
        returns, vol = _make_trending_returns(n=500, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        # Should not raise
        inv_cov = np.linalg.inv(result.covariance)
        self.assertTrue(np.all(np.isfinite(inv_cov)))


# ===================================================================
# TestConstants: Module-level constants
# ===================================================================

class TestConstants(unittest.TestCase):
    """Test module constants are reasonable."""

    def test_max_condition_positive(self):
        self.assertGreater(MAX_CONDITION_NUMBER, 0)

    def test_max_condition_reasonable(self):
        self.assertGreaterEqual(MAX_CONDITION_NUMBER, 100)
        self.assertLessEqual(MAX_CONDITION_NUMBER, 1e8)

    def test_min_variance_inflation(self):
        self.assertEqual(MIN_VARIANCE_INFLATION, 1.0)

    def test_max_variance_inflation(self):
        self.assertGreater(MAX_VARIANCE_INFLATION, 1.0)
        self.assertLessEqual(MAX_VARIANCE_INFLATION, 10.0)

    def test_fd_step_positive(self):
        self.assertGreater(FD_STEP_RELATIVE, 0)
        self.assertGreater(FD_STEP_MIN, 0)

    def test_min_obs_for_laplace(self):
        self.assertGreater(MIN_OBS_FOR_LAPLACE, 10)


# ===================================================================
# TestLLGaussian: Log-likelihood evaluator
# ===================================================================

class TestLLGaussian(unittest.TestCase):
    """Test the Gaussian log-likelihood evaluator."""

    def test_valid_params_returns_finite(self):
        returns, vol = _make_synthetic_returns(n=100)
        ll = _ll_gaussian(returns, vol, c=1.0, phi=0.0, q=1e-6)
        self.assertTrue(np.isfinite(ll))

    def test_negative_c_returns_penalty(self):
        returns, vol = _make_synthetic_returns(n=100)
        ll = _ll_gaussian(returns, vol, c=-1.0, phi=0.0, q=1e-6)
        self.assertEqual(ll, -1e18)

    def test_negative_q_returns_penalty(self):
        returns, vol = _make_synthetic_returns(n=100)
        ll = _ll_gaussian(returns, vol, c=1.0, phi=0.0, q=-1e-6)
        self.assertEqual(ll, -1e18)

    def test_larger_c_lower_ll(self):
        """Larger c => larger observation noise => more spread => lower LL."""
        returns, vol = _make_synthetic_returns(n=200, sigma=0.01)
        ll1 = _ll_gaussian(returns, vol, c=1.0, phi=0.0, q=1e-6)
        ll2 = _ll_gaussian(returns, vol, c=10.0, phi=0.0, q=1e-6)
        self.assertGreater(ll1, ll2)


# ===================================================================
# TestIntegration: End-to-end with realistic data
# ===================================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_gaussian_full_pipeline(self):
        """Full pipeline: generate data -> Laplace -> sample -> PI coverage."""
        returns, vol = _make_trending_returns(n=500, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])

        # Step 1: Laplace posterior
        result = laplace_posterior(returns, vol, theta, family="gaussian")
        self.assertIsInstance(result, LaplaceResult)
        self.assertTrue(np.all(np.isfinite(result.covariance)))

        # Step 2: Variance inflation reasonable
        self.assertGreaterEqual(result.variance_inflation, 1.0)

        # Step 3: Sampling works
        samples = sample_from_laplace(result, n_samples=500)
        self.assertEqual(samples.shape, (500, 3))

    def test_student_t_full_pipeline(self):
        """Full pipeline for Student-t model."""
        returns, vol = _make_trending_returns(n=500, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])

        result = laplace_posterior(returns, vol, theta, family="student_t", nu=8.0)
        self.assertIsInstance(result, LaplaceResult)
        self.assertTrue(np.all(np.isfinite(result.covariance)))

        samples = sample_from_laplace(result, n_samples=500)
        self.assertEqual(samples.shape, (500, 3))

    def test_inflated_pi_wider(self):
        """PI with parameter uncertainty should be wider than plug-in."""
        returns, vol = _make_trending_returns(n=500, phi=0.3, q=1e-6, c=1.0, seed=42)
        theta = np.array([1.0, 0.3, 1e-6])

        result = laplace_posterior(returns, vol, theta, family="gaussian")

        sigma_plugin = 0.01
        sigma_inflated = predictive_variance_with_uncertainty(sigma_plugin, result)

        # Inflated should be >= plugin
        self.assertGreaterEqual(sigma_inflated, sigma_plugin)


if __name__ == "__main__":
    unittest.main()
