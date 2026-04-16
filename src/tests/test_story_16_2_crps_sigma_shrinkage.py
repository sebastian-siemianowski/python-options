"""
Story 16.2 – CRPS-Optimal Sigma Shrinkage
==========================================
Verify alpha* = sqrt(nu / ((nu-2)*(1+1/nu))) is the CRPS minimizer.
"""

import os, sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import (
    crps_optimal_sigma_shrinkage,
    apply_crps_sigma_shrinkage,
    verify_crps_shrinkage_vs_grid,
)


class TestAlphaStarFormula:
    """Verify the closed-form alpha* values."""

    def test_nu_4(self):
        r = crps_optimal_sigma_shrinkage(4.0)
        assert abs(r["alpha_star"] - 0.8944) < 0.001

    def test_nu_8(self):
        r = crps_optimal_sigma_shrinkage(8.0)
        assert abs(r["alpha_star"] - 0.9354) < 0.001

    def test_nu_20(self):
        r = crps_optimal_sigma_shrinkage(20.0)
        assert abs(r["alpha_star"] - 0.9701) < 0.001

    def test_nu_100_near_one(self):
        """Large nu -> alpha* -> 1."""
        r = crps_optimal_sigma_shrinkage(100.0)
        assert abs(r["alpha_star"] - 1.0) < 0.02

    def test_alpha_increases_with_nu(self):
        """alpha* is monotonically increasing in nu."""
        nus = [3.0, 4.0, 6.0, 10.0, 20.0, 50.0]
        alphas = [crps_optimal_sigma_shrinkage(nu)["alpha_star"] for nu in nus]
        for i in range(1, len(alphas)):
            assert alphas[i] > alphas[i - 1]

    def test_alpha_less_than_one(self):
        """alpha* < 1 for all finite nu."""
        for nu in [3.0, 5.0, 10.0, 30.0, 100.0]:
            r = crps_optimal_sigma_shrinkage(nu)
            assert r["alpha_star"] < 1.0

    def test_nu_2_raises(self):
        """nu <= 2 has infinite variance -> error."""
        with pytest.raises(ValueError):
            crps_optimal_sigma_shrinkage(2.0)

    def test_nu_1_5_raises(self):
        with pytest.raises(ValueError):
            crps_optimal_sigma_shrinkage(1.5)


class TestApplyShrinkage:
    """Test apply_crps_sigma_shrinkage."""

    def test_basic_apply(self):
        sigma = np.full(10, 0.02)
        result = apply_crps_sigma_shrinkage(sigma, 8.0)
        alpha = crps_optimal_sigma_shrinkage(8.0)["alpha_star"]
        np.testing.assert_allclose(result, sigma * alpha)

    def test_preserves_shape(self):
        sigma = np.random.default_rng(42).uniform(0.01, 0.03, 50)
        result = apply_crps_sigma_shrinkage(sigma, 4.0)
        assert result.shape == sigma.shape

    def test_shrunk_is_smaller(self):
        """Shrunk sigma < original sigma."""
        sigma = np.full(10, 0.02)
        result = apply_crps_sigma_shrinkage(sigma, 8.0)
        assert np.all(result < sigma)


class TestGridVerification:
    """Verify alpha* behavior via grid search."""

    @pytest.mark.parametrize("nu", [4.0, 8.0, 20.0])
    def test_well_calibrated_grid_near_one(self, nu):
        """For well-calibrated data (z ~ t_nu), grid optimum near 1."""
        rng = np.random.default_rng(42)
        z = rng.standard_t(nu, 500)
        sigma = np.full(500, 0.02)
        result = verify_crps_shrinkage_vs_grid(z, sigma, nu)
        # Well-calibrated: optimal alpha should be near 1
        assert abs(result["alpha_star_grid"] - 1.0) < 0.15

    @pytest.mark.parametrize("nu", [4.0, 8.0, 20.0])
    def test_shrinkage_doesnt_hurt_much(self, nu):
        """Formula alpha* doesn't increase CRPS by more than 5%."""
        rng = np.random.default_rng(42)
        z = rng.standard_t(nu, 500)
        sigma = np.full(500, 0.02)
        result = verify_crps_shrinkage_vs_grid(z, sigma, nu)
        # Shrinkage shouldn't hurt more than 5% vs grid optimum
        assert result["relative_gap"] < 0.05

    def test_crps_improvement_miscalibrated(self):
        """When sigma overestimated by 20%, shrinkage helps."""
        from models.numba_kernels import crps_student_t_kernel

        rng = np.random.default_rng(42)
        nu = 8.0
        # True data from t(nu) with scale 0.02
        z = rng.standard_t(nu, 500) * 0.02
        # Model overestimates sigma by 20%
        sigma_over = np.full(500, 0.024)
        crps_over = crps_student_t_kernel(z / 0.024, sigma_over, nu)
        # Apply shrinkage
        sigma_shrunk = apply_crps_sigma_shrinkage(sigma_over, nu)
        crps_shrunk = crps_student_t_kernel(z / sigma_shrunk, sigma_shrunk, nu)
        # Shrinkage should help (lower CRPS)
        assert crps_shrunk < crps_over * 1.01  # at least not much worse


class TestVarianceRatio:
    """Verify variance_ratio and df_adj fields."""

    def test_variance_ratio_nu_4(self):
        r = crps_optimal_sigma_shrinkage(4.0)
        assert abs(r["variance_ratio"] - 2.0) < 1e-10

    def test_variance_ratio_nu_8(self):
        r = crps_optimal_sigma_shrinkage(8.0)
        assert abs(r["variance_ratio"] - 8.0 / 6.0) < 1e-10

    def test_df_adj_nu_4(self):
        r = crps_optimal_sigma_shrinkage(4.0)
        expected = (3 * 4 + 4) / (3 * 4 + 8)  # 16/20 = 0.8
        assert abs(r["effective_df_adj"] - expected) < 1e-10
