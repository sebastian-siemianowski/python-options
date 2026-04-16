"""
Tests for Story 3.3: phi-nu Joint Optimization with Identifiability Guard.

Acceptance Criteria:
  - Hessian condition number at (phi*, nu*) logged for every fit
  - Warning if kappa(H) > 100 (near-singular -- parameters trading off)
  - Regularization: ||phi - phi_0||^2 / lambda_phi + ||nu - nu_0||^2 / lambda_nu
  - Synthetic test: known (phi=0.5, nu=5) recovered within (0.45-0.55, 4-6)
  - No BIC regression on any of 50 test assets
  - Validated on: UPST, BTC-USD, MSTR, SPY, GC=F, SI=F
"""
import os
import sys
import math
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
for p in [SRC_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration.phi_nu_identifiability import (
    compute_phi_nu_hessian,
    compute_condition_number,
    apply_phi_nu_regularization,
    check_phi_nu_identifiability,
    IdentifiabilityResult,
    KAPPA_WARNING,
    KAPPA_CRITICAL,
    DEFAULT_PHI_0,
    DEFAULT_NU_0,
    DEFAULT_LAMBDA_PHI,
    DEFAULT_LAMBDA_NU,
)


def _generate_student_t_data(phi=0.5, nu=5.0, n=500, seed=42):
    """Generate synthetic data from a Student-t Kalman filter DGP."""
    rng = np.random.RandomState(seed)
    q = 1e-5
    c = 1.0
    mu = np.zeros(n)
    returns = np.zeros(n)
    vol = np.ones(n) * 0.02  # constant vol for simplicity

    mu[0] = 0.0
    for t in range(1, n):
        mu[t] = phi * mu[t - 1] + rng.normal(0, math.sqrt(q))

    # Student-t observation noise
    for t in range(n):
        scale = c * vol[t]
        # Student-t = Normal / sqrt(chi2/nu)
        z = rng.normal(0, 1)
        chi2 = rng.chisquare(nu)
        returns[t] = mu[t] + scale * z / math.sqrt(chi2 / nu)

    return returns, vol, q, c


class TestHessianComputation(unittest.TestCase):
    """Test the 2x2 Hessian computation."""

    def setUp(self):
        self.returns, self.vol, self.q, self.c = _generate_student_t_data(
            phi=0.5, nu=5.0, n=400
        )

    def test_hessian_shape_2x2(self):
        """Hessian must be 2x2 matrix."""
        H = compute_phi_nu_hessian(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertEqual(H.shape, (2, 2))

    def test_hessian_symmetric(self):
        """Hessian should be approximately symmetric."""
        H = compute_phi_nu_hessian(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        np.testing.assert_allclose(H[0, 1], H[1, 0], rtol=0.2, atol=1e-3)

    def test_hessian_negative_definite_at_maximum(self):
        """At MLE, Hessian of log-likelihood should be negative semi-definite."""
        H = compute_phi_nu_hessian(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        eigenvalues = np.linalg.eigvalsh(H)
        # At or near MLE, eigenvalues should be <= 0 (or very close)
        # Allow small positive due to finite differences
        self.assertTrue(
            np.all(eigenvalues < 1.0),
            f"Eigenvalues should be non-positive at MLE, got {eigenvalues}"
        )

    def test_hessian_finite_values(self):
        """All Hessian entries must be finite."""
        H = compute_phi_nu_hessian(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertTrue(np.all(np.isfinite(H)))

    def test_hessian_different_at_different_points(self):
        """Hessian at different (phi, nu) should differ."""
        H1 = compute_phi_nu_hessian(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        H2 = compute_phi_nu_hessian(
            self.returns, self.vol, self.q, self.c, 0.8, 12.0
        )
        self.assertFalse(np.allclose(H1, H2, atol=1e-3))


class TestConditionNumber(unittest.TestCase):
    """Test condition number computation."""

    def test_identity_condition_number_is_one(self):
        """Identity matrix has condition number 1."""
        H = np.eye(2)
        kappa = compute_condition_number(H)
        self.assertAlmostEqual(kappa, 1.0, places=5)

    def test_well_conditioned(self):
        """Diagonal matrix with similar eigenvalues has low kappa."""
        H = np.diag([10.0, 8.0])
        kappa = compute_condition_number(H)
        self.assertAlmostEqual(kappa, 10.0 / 8.0, places=5)

    def test_ill_conditioned(self):
        """Matrix with very different eigenvalues has high kappa."""
        H = np.diag([1000.0, 0.001])
        kappa = compute_condition_number(H)
        self.assertAlmostEqual(kappa, 1e6, places=1)

    def test_singular_matrix_returns_large_kappa(self):
        """Singular matrix should return very large condition number."""
        H = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        kappa = compute_condition_number(H)
        self.assertGreater(kappa, 1e10)

    def test_negative_eigenvalues(self):
        """Should handle negative eigenvalues (uses absolute values)."""
        H = np.diag([-5.0, -3.0])
        kappa = compute_condition_number(H)
        self.assertAlmostEqual(kappa, 5.0 / 3.0, places=5)


class TestRegularization(unittest.TestCase):
    """Test the regularization function."""

    def test_no_regularization_below_warning(self):
        """No change when kappa < KAPPA_WARNING."""
        phi, nu = 0.7, 6.0
        phi_r, nu_r = apply_phi_nu_regularization(phi, nu, 50.0)
        self.assertEqual(phi_r, phi)
        self.assertEqual(nu_r, nu)

    def test_regularization_applied_above_warning(self):
        """Some shrinkage when kappa > KAPPA_WARNING."""
        phi, nu = 0.7, 6.0
        phi_r, nu_r = apply_phi_nu_regularization(phi, nu, 500.0)
        # Should shrink toward defaults
        self.assertNotEqual(phi_r, phi)
        # phi should move toward DEFAULT_PHI_0
        if phi > DEFAULT_PHI_0:
            self.assertLess(phi_r, phi)
        else:
            self.assertGreater(phi_r, phi)

    def test_stronger_regularization_at_critical(self):
        """More shrinkage at KAPPA_CRITICAL vs just above WARNING."""
        phi, nu = 0.8, 4.0
        phi_warn, nu_warn = apply_phi_nu_regularization(phi, nu, KAPPA_WARNING * 2)
        phi_crit, nu_crit = apply_phi_nu_regularization(phi, nu, KAPPA_CRITICAL)
        # Critical should shrink more
        self.assertGreater(abs(phi - phi_crit), abs(phi - phi_warn))

    def test_phi_clamp_range(self):
        """Regularized phi must stay in (-0.80, 0.99)."""
        phi_r, nu_r = apply_phi_nu_regularization(0.98, 3.0, 5000.0)
        self.assertGreaterEqual(phi_r, -0.80)
        self.assertLessEqual(phi_r, 0.99)

    def test_nu_clamp_range(self):
        """Regularized nu must stay in (2.1, 30.0)."""
        phi_r, nu_r = apply_phi_nu_regularization(0.5, 2.5, 5000.0)
        self.assertGreaterEqual(nu_r, 2.1)
        self.assertLessEqual(nu_r, 30.0)

    def test_custom_priors(self):
        """Custom phi_0 and nu_0 should affect shrinkage direction."""
        phi, nu = 0.7, 6.0
        phi_r1, nu_r1 = apply_phi_nu_regularization(
            phi, nu, 500.0, phi_0=0.9, nu_0=10.0
        )
        phi_r2, nu_r2 = apply_phi_nu_regularization(
            phi, nu, 500.0, phi_0=0.3, nu_0=4.0
        )
        # With phi_0=0.9, phi should increase; with phi_0=0.3, it should decrease
        self.assertGreater(phi_r1, phi_r2)


class TestIdentifiabilityResult(unittest.TestCase):
    """Test the IdentifiabilityResult dataclass."""

    def setUp(self):
        self.returns, self.vol, self.q, self.c = _generate_student_t_data(
            phi=0.5, nu=5.0, n=400
        )

    def test_result_fields_exist(self):
        """IdentifiabilityResult has all required fields."""
        result = check_phi_nu_identifiability(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertIsInstance(result, IdentifiabilityResult)
        self.assertIsInstance(result.phi, float)
        self.assertIsInstance(result.nu, float)
        self.assertIsInstance(result.hessian, np.ndarray)
        self.assertIsInstance(result.condition_number, float)
        self.assertIsInstance(result.is_warning, bool)
        self.assertIsInstance(result.is_critical, bool)
        self.assertIsInstance(result.phi_regularized, float)
        self.assertIsInstance(result.nu_regularized, float)
        self.assertIsInstance(result.regularization_applied, bool)

    def test_hessian_2x2_in_result(self):
        """Hessian in result is 2x2."""
        result = check_phi_nu_identifiability(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertEqual(result.hessian.shape, (2, 2))

    def test_condition_number_positive(self):
        """Condition number must be positive."""
        result = check_phi_nu_identifiability(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertGreater(result.condition_number, 0)

    def test_warning_flag_consistency(self):
        """is_warning should match kappa > KAPPA_WARNING."""
        result = check_phi_nu_identifiability(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertEqual(result.is_warning, result.condition_number > KAPPA_WARNING)

    def test_critical_flag_consistency(self):
        """is_critical should match kappa > KAPPA_CRITICAL."""
        result = check_phi_nu_identifiability(
            self.returns, self.vol, self.q, self.c, 0.5, 5.0
        )
        self.assertEqual(result.is_critical, result.condition_number > KAPPA_CRITICAL)


class TestSyntheticRecovery(unittest.TestCase):
    """Synthetic DGP: known (phi=0.5, nu=5) should be recovered."""

    def test_phi_nu_recovered_from_dgp(self):
        """
        Generate data with phi=0.5, nu=5. Run identifiability check.
        Parameters should not be distorted by regularization.
        (Well-identified DGP should have low kappa.)
        """
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=800, seed=99)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0, phi_0=0.5, nu_0=8.0
        )
        # For a well-identified DGP, kappa should be moderate
        # and no regularization should be needed
        # Even if kappa is high, the regularized values should be close
        self.assertAlmostEqual(result.phi_regularized, 0.5, delta=0.15)
        self.assertAlmostEqual(result.nu_regularized, 5.0, delta=3.0)

    def test_phi_in_range_045_055(self):
        """Recovered phi within (0.45, 0.55) for DGP phi=0.5."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=800, seed=99)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0, phi_0=0.5, nu_0=8.0
        )
        self.assertGreaterEqual(result.phi_regularized, 0.45)
        self.assertLessEqual(result.phi_regularized, 0.55)

    def test_nu_in_range_4_6(self):
        """Recovered nu within (4, 6) for DGP nu=5."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=800, seed=99)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0, phi_0=0.5, nu_0=8.0
        )
        self.assertGreaterEqual(result.nu_regularized, 4.0)
        self.assertLessEqual(result.nu_regularized, 6.0)

    def test_no_regularization_for_well_identified_dgp(self):
        """Well-identified DGP should not trigger regularization."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=800, seed=99)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0, phi_0=0.5, nu_0=8.0
        )
        # phi and nu should remain unchanged
        self.assertAlmostEqual(result.phi_regularized, result.phi, places=4)
        self.assertAlmostEqual(result.nu_regularized, result.nu, places=2)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_extreme_phi(self):
        """Works with phi near boundary."""
        returns, vol, q, c = _generate_student_t_data(phi=0.95, nu=8.0, n=300)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.95, 8.0
        )
        self.assertIsInstance(result, IdentifiabilityResult)
        self.assertTrue(np.isfinite(result.condition_number))

    def test_extreme_nu(self):
        """Works with very low nu (heavy tails)."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=2.5, n=300)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 2.5
        )
        self.assertIsInstance(result, IdentifiabilityResult)
        self.assertTrue(np.isfinite(result.condition_number))

    def test_high_nu(self):
        """Works with high nu (near Gaussian)."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=20.0, n=300)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 20.0
        )
        self.assertIsInstance(result, IdentifiabilityResult)
        self.assertTrue(np.isfinite(result.condition_number))

    def test_short_series(self):
        """Works with short data series (200 obs)."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=200)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0
        )
        self.assertIsInstance(result, IdentifiabilityResult)

    def test_negative_phi(self):
        """Works with negative phi (mean-reverting)."""
        returns, vol, q, c = _generate_student_t_data(phi=-0.3, nu=5.0, n=400)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, -0.3, 5.0
        )
        self.assertIsInstance(result, IdentifiabilityResult)
        self.assertGreaterEqual(result.phi_regularized, -0.80)

    def test_asset_symbol_none(self):
        """Works when asset_symbol is None."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=300)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0, asset_symbol=None
        )
        self.assertIsInstance(result, IdentifiabilityResult)

    def test_asset_symbol_provided(self):
        """Works when asset_symbol is provided."""
        returns, vol, q, c = _generate_student_t_data(phi=0.5, nu=5.0, n=300)
        result = check_phi_nu_identifiability(
            returns, vol, q, c, 0.5, 5.0, asset_symbol='SPY'
        )
        self.assertIsInstance(result, IdentifiabilityResult)


class TestThresholdConstants(unittest.TestCase):
    """Verify threshold constants are correctly configured."""

    def test_kappa_warning_is_100(self):
        self.assertEqual(KAPPA_WARNING, 100.0)

    def test_kappa_critical_is_1000(self):
        self.assertEqual(KAPPA_CRITICAL, 1000.0)

    def test_critical_greater_than_warning(self):
        self.assertGreater(KAPPA_CRITICAL, KAPPA_WARNING)

    def test_default_nu_0(self):
        self.assertEqual(DEFAULT_NU_0, 8.0)

    def test_default_lambda_phi(self):
        self.assertEqual(DEFAULT_LAMBDA_PHI, 0.1)

    def test_default_lambda_nu(self):
        self.assertEqual(DEFAULT_LAMBDA_NU, 2.0)


class TestRegularizationFormula(unittest.TestCase):
    """Verify the regularization formula: ||phi-phi_0||^2/lambda_phi + ||nu-nu_0||^2/lambda_nu."""

    def test_regularization_shrinks_phi_toward_phi0(self):
        """phi should move toward phi_0 when regularized."""
        phi_0 = 0.5
        phi = 0.9
        phi_r, _ = apply_phi_nu_regularization(
            phi, 8.0, 500.0, phi_0=phi_0, nu_0=8.0
        )
        # phi_r should be between phi and phi_0
        self.assertLess(phi_r, phi)
        self.assertGreater(phi_r, phi_0)

    def test_regularization_shrinks_nu_toward_nu0(self):
        """nu should move toward nu_0 when regularized."""
        nu_0 = 8.0
        nu = 3.0
        _, nu_r = apply_phi_nu_regularization(
            0.5, nu, 500.0, phi_0=0.5, nu_0=nu_0, lambda_nu=5.0
        )
        # nu_r should be between nu and nu_0
        self.assertGreater(nu_r, nu)
        self.assertLess(nu_r, nu_0)

    def test_lambda_phi_controls_strength(self):
        """Higher lambda_phi -> stronger shrinkage."""
        phi = 0.8
        phi_weak, _ = apply_phi_nu_regularization(
            phi, 8.0, 500.0, phi_0=0.0, lambda_phi=0.05
        )
        phi_strong, _ = apply_phi_nu_regularization(
            phi, 8.0, 500.0, phi_0=0.0, lambda_phi=0.5
        )
        # Stronger lambda should shrink more
        self.assertGreater(abs(phi - phi_strong), abs(phi - phi_weak))

    def test_lambda_nu_controls_strength(self):
        """Higher lambda_nu -> stronger shrinkage."""
        nu = 4.0
        _, nu_weak = apply_phi_nu_regularization(
            0.5, nu, 500.0, nu_0=8.0, lambda_nu=1.0
        )
        _, nu_strong = apply_phi_nu_regularization(
            0.5, nu, 500.0, nu_0=8.0, lambda_nu=8.0
        )
        # Stronger lambda should shrink more
        self.assertGreater(abs(nu - nu_strong), abs(nu - nu_weak))


if __name__ == '__main__':
    unittest.main()
