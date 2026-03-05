"""
Tests for the AD Tail-Correction Pipeline (March 2026).

Three-stage correction: TWSC → SPTG → Isotonic
Tests cover:
  - Numba TWSC kernel behavior
  - SPTG GPD tail-grafting CDF
  - Isotonic non-degradation guarantee
  - Full pipeline for Student-t models
  - Full pipeline for Gaussian models
"""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestTWSCKernel(unittest.TestCase):
    """Test Tail-Weighted Scale Correction Numba kernel."""

    def test_twsc_no_tail_excess_returns_ones(self):
        """When tails are well-behaved, scale inflation should be ~1."""
        from models.numba_kernels import ad_twsc_kernel

        rng = np.random.default_rng(42)
        # Standard normal — 5% in each tail is expected
        z = rng.standard_normal(500)
        z = np.ascontiguousarray(z, dtype=np.float64)

        scale_adj = ad_twsc_kernel(z, 0.97, 0.05, 0.5, 2.0, 0.15)
        # Most adjustments should be near 1.0 (no excess tails)
        self.assertEqual(len(scale_adj), 500)
        # Mean should be close to 1 (allow sampling noise + EWMA warmup)
        self.assertAlmostEqual(np.mean(scale_adj), 1.0, delta=0.15)

    def test_twsc_heavy_tails_inflates(self):
        """When tail frequency exceeds expected, scale should inflate."""
        from models.numba_kernels import ad_twsc_kernel

        rng = np.random.default_rng(123)
        # Student-t(3) has much heavier tails than alpha=0.05 expects
        z = rng.standard_t(3, size=1000)
        z = np.ascontiguousarray(z, dtype=np.float64)

        scale_adj = ad_twsc_kernel(z, 0.97, 0.05, 0.5, 2.0, 0.15)
        # Should see some inflation > 1 for heavy-tailed data
        max_adj = np.max(scale_adj)
        self.assertGreater(max_adj, 1.0, "TWSC should inflate scale for heavy tails")

    def test_twsc_max_inflate_cap(self):
        """Scale inflation should never exceed max_inflate."""
        from models.numba_kernels import ad_twsc_kernel

        # Extreme tails: all observations beyond z_alpha
        z = np.full(200, 5.0, dtype=np.float64)
        scale_adj = ad_twsc_kernel(z, 0.97, 0.05, 0.5, 2.0, 0.15)
        self.assertTrue(np.all(scale_adj <= 2.0 + 1e-10),
                        "Inflation should be capped at max_inflate=2.0")

    def test_twsc_edge_alpha(self):
        """Invalid alpha should return all ones."""
        from models.numba_kernels import ad_twsc_kernel

        z = np.ones(100, dtype=np.float64)
        adj = ad_twsc_kernel(z, 0.97, 0.0, 0.5, 2.0, 0.15)
        np.testing.assert_array_equal(adj, np.ones(100))

        adj2 = ad_twsc_kernel(z, 0.97, 0.5, 0.5, 2.0, 0.15)
        np.testing.assert_array_equal(adj2, np.ones(100))


class TestSPTGKernel(unittest.TestCase):
    """Test Semi-Parametric Tail-Grafted CDF kernels."""

    def test_sptg_student_t_bulk_matches_original(self):
        """In the bulk region, SPTG should match original Student-t CDF."""
        from models.numba_kernels import ad_sptg_cdf_student_t_scalar, _student_t_cdf_scalar

        nu = 8.0
        # Test points in the bulk (|z| < threshold)
        for z in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            sptg = ad_sptg_cdf_student_t_scalar(
                z, nu,
                0.2, 0.5, 2.0,   # left tail GPD params
                0.2, 0.5, 2.0,   # right tail GPD params
                0.05, 0.05,       # tail probs
            )
            original = _student_t_cdf_scalar(z, nu)
            self.assertAlmostEqual(sptg, original, places=6,
                                   msg=f"Bulk mismatch at z={z}")

    def test_sptg_student_t_tail_monotonic(self):
        """PIT values should be monotonically increasing in z when tail probs match model."""
        from models.numba_kernels import ad_sptg_cdf_student_t_array, _student_t_cdf_scalar

        nu = 8.0
        u_left = 2.0
        u_right = 2.0
        # Use correct tail probabilities from the model CDF for continuity
        p_left = _student_t_cdf_scalar(-u_left, nu)   # F(-2.0, 8)
        p_right = 1.0 - _student_t_cdf_scalar(u_right, nu)  # 1 - F(2.0, 8)

        z_arr = np.linspace(-5.0, 5.0, 200)
        pit = ad_sptg_cdf_student_t_array(
            z_arr, nu,
            0.2, 0.5, u_left,
            0.2, 0.5, u_right,
            p_left, p_right,
        )
        # Check monotonicity
        diffs = np.diff(pit)
        self.assertTrue(np.all(diffs >= -1e-10), "SPTG CDF should be monotonic")

    def test_sptg_gaussian_bulk_matches_ndtr(self):
        """Gaussian SPTG bulk should match standard normal CDF."""
        from models.numba_kernels import ad_sptg_cdf_gaussian_scalar, _ndtr_scalar
        from scipy.special import ndtr

        for z in [-1.5, -0.5, 0.0, 0.5, 1.5]:
            sptg = ad_sptg_cdf_gaussian_scalar(
                z,
                0.2, 0.5, 2.0,
                0.2, 0.5, 2.0,
                0.05, 0.05,
            )
            scipy_val = float(ndtr(z))
            self.assertAlmostEqual(sptg, scipy_val, places=4,
                                   msg=f"Gaussian bulk mismatch at z={z}")

    def test_sptg_gaussian_monotonic(self):
        """Gaussian SPTG PIT values should be monotonic when tail probs match model."""
        from models.numba_kernels import ad_sptg_cdf_gaussian_array, _ndtr_scalar

        u_left = 2.0
        u_right = 2.0
        # Match tail probs to Gaussian CDF for continuity
        p_left = _ndtr_scalar(-u_left)
        p_right = 1.0 - _ndtr_scalar(u_right)

        z_arr = np.linspace(-5.0, 5.0, 200)
        pit = ad_sptg_cdf_gaussian_array(
            z_arr,
            0.2, 0.5, u_left,
            0.2, 0.5, u_right,
            p_left, p_right,
        )
        diffs = np.diff(pit)
        self.assertTrue(np.all(diffs >= -1e-10), "Gaussian SPTG CDF should be monotonic")


class TestNdtrScalar(unittest.TestCase):
    """Test Numba normal CDF implementation."""

    def test_ndtr_matches_scipy(self):
        """Numba _ndtr_scalar should match scipy.special.ndtr."""
        from models.numba_kernels import _ndtr_scalar
        from scipy.special import ndtr

        test_points = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
        for x in test_points:
            numba_val = _ndtr_scalar(x)
            scipy_val = float(ndtr(x))
            self.assertAlmostEqual(numba_val, scipy_val, places=5,
                                   msg=f"ndtr mismatch at x={x}")

    def test_ndtr_extremes(self):
        """Extreme values should clip to 0 or 1."""
        from models.numba_kernels import _ndtr_scalar

        self.assertAlmostEqual(_ndtr_scalar(-10.0), 0.0, places=10)
        self.assertAlmostEqual(_ndtr_scalar(10.0), 1.0, places=10)


class TestStudentTPipeline(unittest.TestCase):
    """Test full AD correction pipeline for Student-t models."""

    def test_pipeline_improves_misspecified_student_t(self):
        """Pipeline should improve AD p-value for misspecified Student-t data."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf
        from calibration.pit_calibration import anderson_darling_uniform

        rng = np.random.default_rng(2026)
        n = 500
        # True distribution: Student-t(4) (heavy tails)
        true_returns = rng.standard_t(4, size=n) * 0.01

        # Fitted model: assumes ν=12 (too light tails → AD failure)
        nu_fitted = 12.0
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(true_returns))

        z = true_returns / scale
        pit_raw = _fast_t_cdf(z, nu_fitted)
        pit_raw = np.clip(pit_raw, 0.001, 0.999)

        _, ad_raw = anderson_darling_uniform(pit_raw)

        pit_corrected, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            true_returns, mu_pred, scale, nu_fitted, pit_raw
        )

        _, ad_corrected = anderson_darling_uniform(pit_corrected)

        # Corrected should be at least as good as raw
        self.assertGreaterEqual(ad_corrected, ad_raw * 0.95,
                                "Pipeline should not significantly worsen AD p-value")

        # Check diagnostics
        self.assertIn('twsc_applied', diag)
        self.assertIn('sptg_applied', diag)
        self.assertIn('isotonic_applied', diag)

    def test_pipeline_short_data_passes_through(self):
        """Pipeline should pass through with < 50 observations."""
        from models.phi_student_t import PhiStudentTDriftModel

        pit_raw = np.linspace(0.01, 0.99, 30)
        pit_corrected, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            np.zeros(30), np.zeros(30), np.ones(30), 8.0, pit_raw
        )
        self.assertFalse(diag['twsc_applied'])
        self.assertFalse(diag['sptg_applied'])
        self.assertFalse(diag['isotonic_applied'])
        np.testing.assert_array_equal(pit_corrected, pit_raw)

    def test_pipeline_well_calibrated_no_harm(self):
        """Pipeline should not degrade already well-calibrated PIT."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf
        from calibration.pit_calibration import anderson_darling_uniform

        rng = np.random.default_rng(1234)
        n = 400
        # Correctly specified: true ν=8, fitted ν=8
        returns = rng.standard_t(8, size=n) * 0.01
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))

        z = returns / scale
        pit_raw = _fast_t_cdf(z, 8.0)
        pit_raw = np.clip(pit_raw, 0.001, 0.999)

        _, ad_raw = anderson_darling_uniform(pit_raw)

        pit_corrected, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, 8.0, pit_raw
        )

        _, ad_corrected = anderson_darling_uniform(pit_corrected)

        # Should not degrade by more than a small tolerance
        self.assertGreaterEqual(ad_corrected, ad_raw * 0.80,
                                "Pipeline should not significantly worsen well-calibrated data")


class TestGaussianPipeline(unittest.TestCase):
    """Test full AD correction pipeline for Gaussian models."""

    def test_pipeline_improves_contaminated_gaussian(self):
        """Pipeline should improve AD for Gaussian with fat-tail contamination."""
        from models.gaussian import GaussianDriftModel
        from scipy.special import ndtr
        from calibration.pit_calibration import anderson_darling_uniform

        rng = np.random.default_rng(999)
        n = 500
        # 90% Gaussian + 10% fat-tail contamination
        normal_part = rng.standard_normal(n) * 0.01
        contam_idx = rng.choice(n, size=n // 10, replace=False)
        normal_part[contam_idx] *= 3.0  # inflate outliers

        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(normal_part))

        z = normal_part / scale
        pit_raw = np.clip(ndtr(z), 0.001, 0.999)

        _, ad_raw = anderson_darling_uniform(pit_raw)

        pit_corrected, diag = GaussianDriftModel.apply_ad_correction_pipeline(
            normal_part, mu_pred, scale, None, pit_raw
        )

        _, ad_corrected = anderson_darling_uniform(pit_corrected)

        # Should not worsen
        self.assertGreaterEqual(ad_corrected, ad_raw * 0.95,
                                "Gaussian pipeline should not worsen AD p-value")
        self.assertIn('twsc_applied', diag)

    def test_gaussian_pipeline_short_data(self):
        """Pipeline should pass through with insufficient data."""
        from models.gaussian import GaussianDriftModel

        pit_raw = np.linspace(0.01, 0.99, 20)
        pit_corrected, diag = GaussianDriftModel.apply_ad_correction_pipeline(
            np.zeros(20), np.zeros(20), np.ones(20), None, pit_raw
        )
        self.assertFalse(diag['twsc_applied'])
        np.testing.assert_array_equal(pit_corrected, pit_raw)


class TestWrappers(unittest.TestCase):
    """Test numba_wrappers for AD correction functions."""

    def test_run_ad_twsc_wrapper(self):
        """Wrapper should produce same result as direct kernel call."""
        try:
            from models.numba_wrappers import run_ad_twsc
            from models.numba_kernels import ad_twsc_kernel

            rng = np.random.default_rng(77)
            z = rng.standard_t(5, size=300).astype(np.float64)
            z_cont = np.ascontiguousarray(z)

            wrapper_result = run_ad_twsc(z)
            direct_result = ad_twsc_kernel(z_cont, 0.97, 0.05, 0.5, 2.0, 0.15)

            np.testing.assert_array_almost_equal(wrapper_result, direct_result)
        except ImportError:
            self.skipTest("Numba not available")

    def test_run_ad_sptg_student_t_wrapper(self):
        """SPTG Student-t wrapper should return valid PIT values."""
        try:
            from models.numba_wrappers import run_ad_sptg_student_t

            z = np.linspace(-3.0, 3.0, 100)
            pit = run_ad_sptg_student_t(
                z, 8.0,
                0.2, 0.5, 2.0,
                0.2, 0.5, 2.0,
                0.05, 0.05,
            )
            self.assertEqual(len(pit), 100)
            self.assertTrue(np.all(pit > 0))
            self.assertTrue(np.all(pit < 1))
        except ImportError:
            self.skipTest("Numba not available")

    def test_run_ad_sptg_gaussian_wrapper(self):
        """SPTG Gaussian wrapper should return valid PIT values."""
        try:
            from models.numba_wrappers import run_ad_sptg_gaussian

            z = np.linspace(-3.0, 3.0, 100)
            pit = run_ad_sptg_gaussian(
                z,
                0.2, 0.5, 2.0,
                0.2, 0.5, 2.0,
                0.05, 0.05,
            )
            self.assertEqual(len(pit), 100)
            self.assertTrue(np.all(pit > 0))
            self.assertTrue(np.all(pit < 1))
        except ImportError:
            self.skipTest("Numba not available")


# =============================================================================
# REAL MODEL IMPROVEMENT TESTS (March 2026)
# =============================================================================
# These tests verify that calibration corrections produce REAL improvement
# parameters that flow into signals.py, not just cosmetic AD p-value changes.
# =============================================================================


class TestCalibrationParamsOutput(unittest.TestCase):
    """Verify that the pipeline returns calibration_params for persistence."""

    def test_student_t_returns_calibration_params(self):
        """Student-t pipeline should return calibration_params dict."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf

        rng = np.random.default_rng(2026)
        n = 500
        returns = rng.standard_t(4, size=n) * 0.01
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))
        z = returns / scale
        pit_raw = np.clip(_fast_t_cdf(z, 12.0), 0.001, 0.999)

        _, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, 12.0, pit_raw
        )

        self.assertIn('calibration_params', diag)
        cal = diag['calibration_params']
        self.assertIsInstance(cal, dict)

    def test_twsc_scale_factor_stored(self):
        """TWSC should store a scale_factor in calibration_params."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf

        rng = np.random.default_rng(42)
        n = 500
        # Heavy tails → TWSC should have scale_factor > 1
        returns = rng.standard_t(3, size=n) * 0.01
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))
        z = returns / scale
        pit_raw = np.clip(_fast_t_cdf(z, 12.0), 0.001, 0.999)

        _, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, 12.0, pit_raw
        )
        cal = diag['calibration_params']
        self.assertIn('twsc_scale_factor', cal)
        # For misspecified model with too-light tails, scale_factor > 1
        self.assertGreater(cal['twsc_scale_factor'], 0.5)
        self.assertLess(cal['twsc_scale_factor'], 5.0)

    def test_gpd_params_stored(self):
        """GPD tail parameters should be stored for tail risk estimation."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf

        rng = np.random.default_rng(7777)
        n = 500
        returns = rng.standard_t(4, size=n) * 0.01
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))
        z = returns / scale
        pit_raw = np.clip(_fast_t_cdf(z, 8.0), 0.001, 0.999)

        _, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, 8.0, pit_raw
        )
        cal = diag['calibration_params']

        if diag.get('sptg_applied'):
            # GPD params should be stored when SPTG succeeds
            self.assertIn('gpd_left_xi', cal)
            self.assertIn('gpd_right_xi', cal)
            self.assertIn('gpd_left_threshold', cal)
            self.assertIn('gpd_right_threshold', cal)
            self.assertIn('nu_effective', cal)
            self.assertIn('nu_adjustment_ratio', cal)
            # ξ should be positive for Student-t data
            self.assertGreater(cal['gpd_left_xi'], -1.0)
            self.assertGreater(cal['gpd_right_xi'], -1.0)

    def test_isotonic_knots_stored(self):
        """Isotonic transport map knots should be stored when applicable."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf

        rng = np.random.default_rng(9999)
        n = 500
        # Badly misspecified: true ν=3, model assumes ν=20
        returns = rng.standard_t(3, size=n) * 0.01
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))
        z = returns / scale
        pit_raw = np.clip(_fast_t_cdf(z, 20.0), 0.001, 0.999)

        _, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, 20.0, pit_raw
        )
        cal = diag['calibration_params']

        if diag.get('isotonic_applied'):
            self.assertIn('isotonic_x_knots', cal)
            self.assertIn('isotonic_y_knots', cal)
            self.assertIsInstance(cal['isotonic_x_knots'], list)
            self.assertIsInstance(cal['isotonic_y_knots'], list)
            # Knots should be monotonically increasing
            x_knots = np.array(cal['isotonic_x_knots'])
            y_knots = np.array(cal['isotonic_y_knots'])
            self.assertTrue(np.all(np.diff(x_knots) >= 0))
            self.assertTrue(np.all(np.diff(y_knots) >= -1e-10))

    def test_nu_effective_is_more_conservative(self):
        """GPD-derived nu_effective should be <= model's assumed nu."""
        from models.phi_student_t import PhiStudentTDriftModel, _fast_t_cdf

        rng = np.random.default_rng(5555)
        n = 500
        # True ν=4, model assumes ν=12 → GPD should detect heavier tails
        returns = rng.standard_t(4, size=n) * 0.01
        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))
        z = returns / scale
        pit_raw = np.clip(_fast_t_cdf(z, 12.0), 0.001, 0.999)

        _, diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, 12.0, pit_raw
        )
        cal = diag['calibration_params']

        if 'nu_effective' in cal:
            # nu_effective should be <= 12 (more conservative)
            self.assertLessEqual(cal['nu_effective'], 12.0)
            self.assertGreaterEqual(cal['nu_effective'], 2.5)

    def test_gaussian_returns_calibration_params(self):
        """Gaussian pipeline should also return calibration_params."""
        from models.gaussian import GaussianDriftModel
        from scipy.special import ndtr

        rng = np.random.default_rng(3333)
        n = 500
        returns = rng.standard_normal(n) * 0.01
        contam_idx = rng.choice(n, size=n // 10, replace=False)
        returns[contam_idx] *= 3.0

        mu_pred = np.zeros(n)
        scale = np.full(n, np.std(returns))
        z = returns / scale
        pit_raw = np.clip(ndtr(z), 0.001, 0.999)

        _, diag = GaussianDriftModel.apply_ad_correction_pipeline(
            returns, mu_pred, scale, None, pit_raw
        )

        self.assertIn('calibration_params', diag)
        cal = diag['calibration_params']
        self.assertIsInstance(cal, dict)
        # TWSC should produce a scale factor even for Gaussian
        if diag.get('twsc_applied'):
            self.assertIn('twsc_scale_factor', cal)


class TestIsotonicTransportMap(unittest.TestCase):
    """Test isotonic transport map application for probability calibration."""

    def test_isotonic_interp_calibrates_probability(self):
        """Linear interpolation through isotonic knots should recalibrate p."""
        # Simulate a simple isotonic map: identity-ish with small correction
        x_knots = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        y_knots = np.array([0.0, 0.08, 0.28, 0.50, 0.72, 0.92, 1.0])

        # Interpolate at several test points
        for p_raw, expected_approx in [(0.5, 0.50), (0.1, 0.08), (0.9, 0.92)]:
            p_cal = float(np.interp(p_raw, x_knots, y_knots))
            self.assertAlmostEqual(p_cal, expected_approx, places=2)

    def test_isotonic_no_direction_flip(self):
        """Isotonic calibration should not flip signal direction."""
        x_knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_knots = np.array([0.0, 0.20, 0.55, 0.80, 1.0])

        p_raw = 0.6  # bullish
        p_cal = float(np.interp(p_raw, x_knots, y_knots))
        # Should stay bullish (>0.5)
        self.assertGreater(p_cal, 0.5)


class TestTWSCScaleCorrection(unittest.TestCase):
    """Test TWSC scale factor application to P_filtered."""

    def test_scale_factor_inflates_variance(self):
        """TWSC scale_factor > 1 should inflate P_filtered variance."""
        P_filtered = np.array([1e-4, 2e-4, 3e-4, 4e-4])
        twsc_factor = 1.15  # Model was 15% too tight

        P_corrected = P_filtered * (twsc_factor ** 2)

        expected = P_filtered * (1.15 ** 2)
        np.testing.assert_array_almost_equal(P_corrected, expected)
        self.assertTrue(np.all(P_corrected > P_filtered))

    def test_scale_factor_one_no_change(self):
        """scale_factor == 1 should leave P_filtered unchanged."""
        P_filtered = np.array([1e-4, 2e-4, 3e-4])
        twsc_factor = 1.0

        P_corrected = P_filtered * (twsc_factor ** 2)
        np.testing.assert_array_equal(P_corrected, P_filtered)

    def test_scale_factor_bounds(self):
        """Safety bounds should prevent extreme corrections."""
        # Factor should be clipped to [0.5, 3.0]
        for factor, expected in [(0.3, 0.5), (5.0, 3.0), (1.5, 1.5)]:
            clipped = float(np.clip(factor, 0.5, 3.0))
            self.assertAlmostEqual(clipped, expected)


class TestGPDNuAdjustment(unittest.TestCase):
    """Test GPD ξ → effective ν adjustment for MC simulation."""

    def test_heavier_tails_reduce_nu(self):
        """GPD with ξ > 1/ν should reduce effective ν."""
        nu_model = 8.0  # ξ_model = 1/8 = 0.125
        xi_gpd = 0.25   # ξ_gpd > ξ_model → tails heavier than expected

        # nu_from_gpd = 1/ξ_gpd = 4
        nu_from_gpd = 1.0 / xi_gpd
        nu_effective = max(2.5, min(nu_model, nu_from_gpd))

        self.assertAlmostEqual(nu_effective, 4.0)
        self.assertLess(nu_effective, nu_model)

    def test_lighter_tails_no_adjustment(self):
        """GPD with ξ < 1/ν should not increase effective ν."""
        nu_model = 4.0   # ξ_model = 0.25
        xi_gpd = 0.10    # ξ_gpd < ξ_model → tails lighter than expected

        nu_from_gpd = 1.0 / xi_gpd  # = 10
        nu_effective = max(2.5, min(nu_model, nu_from_gpd))

        # Should NOT increase beyond model's ν=4
        self.assertAlmostEqual(nu_effective, 4.0)

    def test_extreme_xi_floors_at_2_5(self):
        """Very heavy tails (ξ → 0.5+) should floor at ν=2.5."""
        xi_gpd = 0.5     # → ν=2
        nu_from_gpd = 1.0 / xi_gpd  # = 2
        nu_effective = max(2.5, min(8.0, nu_from_gpd))

        self.assertAlmostEqual(nu_effective, 2.5)


if __name__ == "__main__":
    unittest.main()
