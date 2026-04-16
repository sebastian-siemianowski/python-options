"""
Story 1.3: Student-t Quantile Function (CDF Inverse) Implementation
====================================================================
Verify student_t_ppf_scalar matches scipy.stats.t.ppf to 1e-6
across a nu x p grid (40+ combinations), works in Numba @njit.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _numba_available():
    try:
        from models.numba_kernels import student_t_ppf_scalar
        return True
    except (ImportError, Exception):
        return False


@pytest.mark.skipif(not _numba_available(), reason="Numba not available")
class TestStudentTPPF:
    """Acceptance criteria for Story 1.3."""

    def test_ppf_matches_scipy_grid(self):
        """AC1: ppf matches scipy.stats.t.ppf to 1e-6 for nu x p grid."""
        from models.numba_kernels import student_t_ppf_scalar
        from scipy.stats import t as scipy_t

        p_values = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
        nu_values = [2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 50.0]

        for nu in nu_values:
            for p in p_values:
                result = student_t_ppf_scalar(p, nu)
                expected = float(scipy_t.ppf(p, nu))
                abs_err = abs(result - expected)
                assert abs_err < 1e-6, (
                    f"ppf(p={p}, nu={nu}): got {result:.8f}, "
                    f"expected {expected:.8f}, err={abs_err:.2e}"
                )

    def test_ppf_no_nan_inf(self):
        """AC2: No NaN or Inf for nu in [2.1, 50]."""
        from models.numba_kernels import student_t_ppf_scalar

        for nu in np.arange(2.1, 50.1, 0.5):
            for p in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
                result = student_t_ppf_scalar(p, nu)
                assert np.isfinite(result), (
                    f"ppf(p={p}, nu={nu}) = {result} (non-finite)"
                )

    def test_ppf_symmetry(self):
        """Verify ppf(p) = -ppf(1-p) (symmetry of Student-t)."""
        from models.numba_kernels import student_t_ppf_scalar

        for nu in [3.0, 6.0, 12.0]:
            for p in [0.01, 0.05, 0.1, 0.25]:
                left = student_t_ppf_scalar(p, nu)
                right = student_t_ppf_scalar(1.0 - p, nu)
                assert abs(left + right) < 1e-10, (
                    f"Symmetry: ppf({p}, {nu})={left:.8f}, "
                    f"ppf({1-p}, {nu})={right:.8f}"
                )

    def test_ppf_cdf_roundtrip(self):
        """Verify F^{-1}(F(z)) = z to 1e-6 for |z|<=8."""
        from models.numba_kernels import student_t_ppf_scalar, _student_t_cdf_scalar

        for nu in [3.0, 4.0, 8.0, 20.0]:
            for z in [-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0]:
                p = _student_t_cdf_scalar(z, nu)
                z_recovered = student_t_ppf_scalar(p, nu)
                assert abs(z_recovered - z) < 1e-6, (
                    f"Roundtrip: z={z}, nu={nu}, p={p:.12f}, "
                    f"z_recovered={z_recovered:.8f}, err={abs(z_recovered - z):.2e}"
                )

    def test_ppf_array_kernel(self):
        """Test vectorized PPF array kernel."""
        from models.numba_kernels import student_t_ppf_array_kernel
        from scipy.stats import t as scipy_t

        p_arr = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        nu = 4.0
        result = student_t_ppf_array_kernel(p_arr, nu)
        expected = scipy_t.ppf(p_arr, nu)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_ppf_usable_in_njit(self):
        """AC3: Can be used inside Numba @njit loops."""
        from numba import njit
        from models.numba_kernels import student_t_ppf_scalar

        @njit(cache=False)
        def _use_ppf(p, nu):
            return student_t_ppf_scalar(p, nu)

        result = _use_ppf(0.975, 4.0)
        assert np.isfinite(result)
        assert result > 0  # Upper quantile should be positive

    def test_ppf_performance(self):
        """Benchmark: PPF should be < 50 microseconds per call."""
        from models.numba_kernels import student_t_ppf_scalar
        import time

        # Warm up JIT
        for _ in range(100):
            student_t_ppf_scalar(0.5, 4.0)

        n_calls = 10_000
        start = time.perf_counter()
        for i in range(n_calls):
            p = 0.01 + 0.98 * (i / n_calls)
            student_t_ppf_scalar(p, 4.0)
        elapsed = time.perf_counter() - start

        us_per_call = elapsed / n_calls * 1e6
        assert us_per_call < 50.0, f"PPF: {us_per_call:.1f} us/call (too slow)"
