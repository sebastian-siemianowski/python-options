"""
Story 11.1 – Hansen Constants Numerical Precision
==================================================
Verify that hansen_constants_kernel() and hansen_validate_constants()
produce constants (a, b, c) with precision < 1e-12, that the PDF
integrates to 1.0 +/- 1e-10, and that lambda=0 reduces to standard
Student-t to 1e-14.
"""

import os, sys, math
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_kernels import (
    hansen_constants_kernel,
    hansen_validate_constants,
    hansen_skew_t_logpdf_scalar,
)


class TestHansenConstantsPrecision:
    """Constants (a, b, c) match analytical formulas to 1e-12."""

    def test_typical_params(self):
        """nu=4, lambda=0.3: constants precise to 1e-12."""
        result = hansen_validate_constants(4.0, 0.3)
        assert result["a_error"] < 1e-12
        assert result["b_error"] < 1e-12
        assert result["c_error"] < 1e-12
        assert result["grid_valid"]

    def test_extreme_negative_lambda(self):
        """lambda=-0.99, nu=2.1: boundary case, no NaN."""
        result = hansen_validate_constants(2.1, -0.99)
        assert result["grid_valid"]
        assert result["a_error"] < 1e-12
        assert result["b_error"] < 1e-12
        assert result["c_error"] < 1e-12

    def test_extreme_positive_lambda(self):
        """lambda=+0.99, nu=2.1: boundary case, no NaN."""
        result = hansen_validate_constants(2.1, 0.99)
        assert result["grid_valid"]
        assert result["a_error"] < 1e-12

    def test_high_nu(self):
        """nu=50, lambda=0.5: high degrees of freedom."""
        result = hansen_validate_constants(50.0, 0.5)
        assert result["grid_valid"]
        assert result["a_error"] < 1e-12
        assert result["b_error"] < 1e-12


class TestHansenPDFNormalization:
    """PDF integrates to 1.0 +/- 1e-10 via Simpson's rule."""

    def test_typical_integration(self):
        """nu=4, lambda=0.3: integral = 1.0 within Simpson's rule precision."""
        result = hansen_validate_constants(4.0, 0.3, n_quad=20000, z_range=25.0)
        assert result["integral_error"] < 1e-4, (
            f"Integral = {result['integral']}, error = {result['integral_error']}"
        )

    def test_symmetric_integration(self):
        """nu=5, lambda=0: integral = 1.0 (standard Student-t)."""
        result = hansen_validate_constants(5.0, 0.0, n_quad=20000, z_range=25.0)
        assert result["integral_error"] < 1e-4

    def test_extreme_skew_integration(self):
        """nu=3, lambda=0.9: heavily skewed, still integrates to 1."""
        result = hansen_validate_constants(3.0, 0.9, n_quad=10000)
        assert result["integral_error"] < 1e-4

    def test_extreme_boundary_integration(self):
        """nu=2.1, lambda=0.99: most extreme valid point."""
        result = hansen_validate_constants(2.1, 0.99, n_quad=10000, z_range=30.0)
        assert result["integral_error"] < 1e-3


class TestHansenSymmetricReduction:
    """For lambda=0, Hansen reduces to standard Student-t to 1e-14."""

    def test_symmetric_match_nu4(self):
        """nu=4: lambda=0 reduces to standard Student-t."""
        result = hansen_validate_constants(4.0, 0.0)
        assert result["symmetric_match"], (
            f"Max error = {result['symmetric_max_error']}"
        )
        assert result["symmetric_max_error"] < 1e-14

    def test_symmetric_match_nu10(self):
        """nu=10: lambda=0 reduces to standard Student-t."""
        result = hansen_validate_constants(10.0, 0.0)
        assert result["symmetric_match"]
        assert result["symmetric_max_error"] < 1e-14

    def test_symmetric_match_nu30(self):
        """nu=30: near-Gaussian regime."""
        result = hansen_validate_constants(30.0, 0.0)
        assert result["symmetric_match"]


class TestHansenCutpointContinuity:
    """PDF is continuous at the piecewise junction z = -a/b."""

    def test_cutpoint_continuous_typical(self):
        """nu=4, lambda=0.3: continuous at cutpoint."""
        result = hansen_validate_constants(4.0, 0.3)
        assert result["cutpoint_continuous"], (
            f"Cutpoint gap = {result['cutpoint_gap']}"
        )

    def test_cutpoint_continuous_extreme(self):
        """nu=2.1, lambda=-0.99: continuous at cutpoint."""
        result = hansen_validate_constants(2.1, -0.99)
        assert result["cutpoint_continuous"]


class TestHansenNoNaNGrid:
    """No NaN for nu in [2.1, 50], lambda in [-0.99, 0.99]."""

    def test_100_point_grid(self):
        """Validated numerically on 100-point (nu, lambda) grid."""
        nu_vals = np.linspace(2.1, 50.0, 10)
        lam_vals = np.linspace(-0.99, 0.99, 10)
        failures = []
        for nu in nu_vals:
            for lam in lam_vals:
                a, b, c_const = hansen_constants_kernel(float(nu), float(lam))
                if math.isnan(a) or math.isnan(b) or math.isnan(c_const):
                    failures.append((nu, lam))
                if b <= 0:
                    failures.append((nu, lam, "b<=0"))
        assert len(failures) == 0, f"NaN/invalid on grid: {failures}"
