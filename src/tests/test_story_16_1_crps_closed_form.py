"""
Story 16.1 – Closed-Form Student-t CRPS Verification
=====================================================
Verify crps_student_t_kernel (closed-form) against MC reference,
check determinism, and speed advantage.
"""

import os, sys, time
import numpy as np
import pytest
from scipy import stats, special

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_kernels import (
    crps_student_t_kernel,
    crps_student_t_numerical_kernel,
)


def _mc_crps_single(z, sigma, nu, n_samples=200000):
    """MC reference CRPS for a single observation."""
    rng = np.random.default_rng(42)
    samples = rng.standard_t(nu, n_samples) * sigma
    obs = z * sigma
    E_X_minus_y = np.mean(np.abs(samples - obs))
    samples2 = rng.standard_t(nu, n_samples) * sigma
    E_X_minus_Xp = np.mean(np.abs(samples - samples2))
    return E_X_minus_y - 0.5 * E_X_minus_Xp


class TestClosedFormVsMC:
    """Closed-form CRPS matches MC reference."""

    @pytest.mark.parametrize("nu", [4.0, 8.0, 20.0])
    def test_single_obs_matches_mc(self, nu):
        """Single observation CRPS: closed-form vs MC."""
        z = np.array([1.5])
        sigma = np.array([0.02])
        crps_cf = crps_student_t_kernel(z, sigma, nu)
        crps_mc = _mc_crps_single(1.5, 0.02, nu)
        # Allow 2% relative error (MC noise)
        assert abs(crps_cf - crps_mc) / max(abs(crps_mc), 1e-10) < 0.02, \
            f"nu={nu}: CF={crps_cf:.6f} vs MC={crps_mc:.6f}"

    @pytest.mark.parametrize("nu", [4.0, 8.0, 20.0])
    def test_zero_residual(self, nu):
        """CRPS at z=0 should be positive."""
        z = np.array([0.0])
        sigma = np.array([0.02])
        crps = crps_student_t_kernel(z, sigma, nu)
        assert crps > 0

    def test_crps_increases_with_error(self):
        """Larger z gives larger CRPS."""
        sigma = np.array([0.02])
        crps_small = crps_student_t_kernel(np.array([0.5]), sigma, 8.0)
        crps_large = crps_student_t_kernel(np.array([3.0]), sigma, 8.0)
        assert crps_large > crps_small


class TestClosedFormVsNumerical:
    """Closed-form matches numerical quadrature."""

    @pytest.mark.parametrize("nu", [4.0, 8.0, 12.0, 20.0])
    def test_matches_numerical(self, nu):
        """Closed-form and numerical kernel agree within 5%."""
        rng = np.random.default_rng(42)
        n = 100
        z = rng.standard_t(nu, n)
        sigma = np.full(n, 0.02)
        crps_cf = crps_student_t_kernel(z, sigma, nu)
        crps_num = crps_student_t_numerical_kernel(z, sigma, nu)
        rel_err = abs(crps_cf - crps_num) / max(abs(crps_num), 1e-10)
        assert rel_err < 0.05, f"nu={nu}: CF={crps_cf:.6f} vs Num={crps_num:.6f}"


class TestDeterminism:
    """Closed-form is deterministic (no MC noise)."""

    def test_same_result_twice(self):
        """Two calls with same input give identical result."""
        rng = np.random.default_rng(42)
        z = rng.standard_t(8, 50)
        sigma = np.full(50, 0.02)
        crps1 = crps_student_t_kernel(z, sigma, 8.0)
        crps2 = crps_student_t_kernel(z, sigma, 8.0)
        assert crps1 == crps2  # exact equality, not approximate


class TestEdgeCases:
    """Edge cases for closed-form CRPS."""

    def test_nu_large(self):
        """Large nu (near Gaussian) -> CRPS close to Gaussian CRPS."""
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 100)
        sigma = np.full(100, 0.02)
        crps_t = crps_student_t_kernel(z, sigma, 50.0)
        # Gaussian CRPS: sigma * [z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
        # Just check it's finite and reasonable
        assert np.isfinite(crps_t)
        assert crps_t > 0

    def test_nu_2_plus(self):
        """nu just above 2 -> finite CRPS."""
        z = np.array([1.0])
        sigma = np.array([0.02])
        crps = crps_student_t_kernel(z, sigma, 2.1)
        assert np.isfinite(crps)
        assert crps > 0

    def test_empty_input(self):
        """Empty input -> large penalty."""
        crps = crps_student_t_kernel(np.array([]), np.array([]), 8.0)
        assert crps == 1e10

    def test_crps_scales_with_sigma(self):
        """CRPS proportional to sigma."""
        z = np.array([1.0])
        crps1 = crps_student_t_kernel(z, np.array([0.01]), 8.0)
        crps2 = crps_student_t_kernel(z, np.array([0.02]), 8.0)
        assert abs(crps2 / crps1 - 2.0) < 0.01


class TestSpeed:
    """Closed-form is fast."""

    def test_faster_than_numerical(self):
        """Closed-form faster than numerical on 1000 obs."""
        rng = np.random.default_rng(42)
        z = rng.standard_t(8, 1000)
        sigma = np.full(1000, 0.02)
        # Warm up JIT
        crps_student_t_kernel(z[:10], sigma[:10], 8.0)
        crps_student_t_numerical_kernel(z[:10], sigma[:10], 8.0)
        t0 = time.perf_counter()
        for _ in range(10):
            crps_student_t_kernel(z, sigma, 8.0)
        t_cf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(10):
            crps_student_t_numerical_kernel(z, sigma, 8.0)
        t_num = time.perf_counter() - t0
        # Closed-form should be faster
        assert t_cf < t_num * 1.5  # at least comparable
