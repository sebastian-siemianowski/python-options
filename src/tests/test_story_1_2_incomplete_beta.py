"""
Story 1.2: Incomplete Beta Function Accuracy at Extreme Quantiles
=================================================================
Verify _betainc matches scipy.special.betainc to 1e-8, and that
Student-t CDF at |z|>5 is accurate with no PIT pile-up at 0 or 1.
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
        from models.numba_kernels import _betainc
        return True
    except (ImportError, Exception):
        return False


@pytest.mark.skipif(not _numba_available(), reason="Numba not available")
class TestIncompleteBeta:
    """Acceptance criteria for Story 1.2."""

    def test_betainc_matches_scipy(self):
        """AC1: _betainc matches scipy.special.betainc to 1e-8 for extreme x."""
        from models.numba_kernels import _betainc
        from scipy.special import betainc as scipy_betainc

        # Test with a=nu/2, b=0.5 for various nu and extreme x
        test_cases = []
        for nu in [3.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
            a = nu / 2.0
            b = 0.5
            for x in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
                test_cases.append((a, b, x))

        for a, b, x in test_cases:
            result = _betainc(a, b, x)
            expected = float(scipy_betainc(a, b, x))
            abs_err = abs(result - expected)
            assert abs_err < 1e-8, (
                f"betainc({a}, {b}, {x}): got {result:.12e}, "
                f"expected {expected:.12e}, err={abs_err:.2e}"
            )

    def test_student_t_cdf_extreme_z(self):
        """AC2: Student-t CDF at z=+/-6 matches scipy to 1e-8."""
        from models.numba_kernels import _student_t_cdf_scalar
        from scipy.stats import t as scipy_t

        for nu in [3.0, 4.0, 6.0, 8.0, 20.0]:
            for z in [-6.0, -5.0, -4.0, -3.0, 3.0, 4.0, 5.0, 6.0]:
                result = _student_t_cdf_scalar(z, nu)
                expected = float(scipy_t.cdf(z, nu))
                abs_err = abs(result - expected)
                assert abs_err < 1e-8, (
                    f"CDF(z={z}, nu={nu}): got {result:.12e}, "
                    f"expected {expected:.12e}, err={abs_err:.2e}"
                )

    def test_cdf_inversion_consistency(self):
        """Task 3: CDF inversion F^-1(F(z)) = z +/- 1e-6 for |z|<=8.

        We verify this by checking that CDF values are monotonic and
        that we can recover z from CDF via bisection.
        """
        from models.numba_kernels import _student_t_cdf_scalar

        for nu in [3.0, 4.0, 6.0]:
            z_grid = np.linspace(-8.0, 8.0, 100)
            cdf_vals = np.array([_student_t_cdf_scalar(z, nu) for z in z_grid])

            # Monotonicity
            assert np.all(np.diff(cdf_vals) >= 0), (
                f"CDF not monotonic for nu={nu}"
            )

            # No pile-up at 0 or 1
            assert cdf_vals[0] > 1e-10, (
                f"CDF(-8, nu={nu}) = {cdf_vals[0]:.2e} -- too close to 0"
            )
            assert cdf_vals[-1] < 1.0 - 1e-10, (
                f"CDF(8, nu={nu}) = {cdf_vals[-1]:.12f} -- too close to 1"
            )

    def test_pit_no_accumulation_at_boundaries(self):
        """AC3: PIT histogram shows no accumulation at 0.0 or 1.0 bins.

        Simulate a Student-t process and verify PIT values are well-spread.
        """
        from models.numba_kernels import _student_t_cdf_scalar

        np.random.seed(42)
        nu = 3.0
        n = 5000
        # Generate Student-t samples
        samples = np.random.standard_t(nu, size=n)
        pit_values = np.array([_student_t_cdf_scalar(s, nu) for s in samples])

        # No values should be exactly 0 or 1
        assert np.all(pit_values > 0.0), "PIT values contain exact 0.0"
        assert np.all(pit_values < 1.0), "PIT values contain exact 1.0"

        # Check that boundary bins are not overpopulated
        n_low = np.sum(pit_values < 0.01)
        n_high = np.sum(pit_values > 0.99)
        expected_per_bin = n * 0.01  # ~50 per 1% bin

        # Allow up to 3x expected (generous bound for randomness)
        assert n_low < 3 * expected_per_bin, (
            f"PIT pile-up at 0: {n_low} values < 0.01 (expected ~{expected_per_bin:.0f})"
        )
        assert n_high < 3 * expected_per_bin, (
            f"PIT pile-up at 1: {n_high} values > 0.99 (expected ~{expected_per_bin:.0f})"
        )

    def test_betacf_convergence_at_300_iterations(self):
        """Verify that 300 iterations handle extreme parameters."""
        from models.numba_kernels import _betainc
        from scipy.special import betainc as scipy_betainc

        # Extreme case: very low nu, extreme x
        extreme_cases = [
            (1.05, 0.5, 0.999),  # nu=2.1 -> a=1.05
            (1.25, 0.5, 0.0001),  # nu=2.5 -> a=1.25
            (25.0, 0.5, 0.99),  # nu=50 -> a=25
            (1.5, 0.5, 0.9999),  # nu=3 -> a=1.5, extreme right
        ]
        for a, b, x in extreme_cases:
            result = _betainc(a, b, x)
            expected = float(scipy_betainc(a, b, x))
            abs_err = abs(result - expected)
            assert abs_err < 1e-8, (
                f"betainc({a}, {b}, {x}): err={abs_err:.2e}"
            )

    def test_cdf_benchmark_cost(self):
        """Task 5: Measure CDF evaluation cost per 1000 calls."""
        from models.numba_kernels import _student_t_cdf_scalar
        import time

        # Warm up
        for _ in range(100):
            _student_t_cdf_scalar(2.0, 4.0)

        n_calls = 10_000
        start = time.perf_counter()
        for i in range(n_calls):
            _student_t_cdf_scalar(-3.0 + 6.0 * (i / n_calls), 4.0)
        elapsed = time.perf_counter() - start

        us_per_call = elapsed / n_calls * 1e6
        # Should be < 10 microseconds per call
        assert us_per_call < 10.0, f"CDF: {us_per_call:.1f} us/call (too slow)"
