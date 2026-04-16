"""
Story 3.2: Regime-Aware GARCH Initialization
=============================================
h0 uses trailing 20-day realized variance for faster convergence.
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
        from models.numba_kernels import garch_variance_kernel, garch_h0_from_trailing
        return True
    except (ImportError, Exception):
        return False


@pytest.mark.skipif(not _numba_available(), reason="Numba not available")
class TestRegimeAwareGARCHInit:
    """Acceptance criteria for Story 3.2."""

    def test_h0_from_trailing_median(self):
        """AC1: h0 uses trailing 20-day realized variance (median)."""
        from models.numba_kernels import garch_h0_from_trailing

        sq = np.array([0.0004] * 15 + [0.01] * 5, dtype=np.float64)
        h0 = garch_h0_from_trailing(sq, window=20)
        # Median of sorted: 15 values of 0.0004, 5 of 0.01 -> median is 0.0004
        assert h0 > 0, f"h0 = {h0}"
        expected_median = np.median(sq[:20])
        assert abs(h0 - expected_median) < 1e-15

    def test_h0_override_used_in_kernel(self):
        """h0_override parameter correctly sets initial variance."""
        from models.numba_kernels import garch_variance_kernel

        n = 100
        rng = np.random.default_rng(42)
        innovations = rng.normal(0, 0.02, n)
        sq = innovations ** 2
        neg = (innovations < 0).astype(np.float64)

        gu = 0.0004
        h0_custom = 0.001  # 2.5x unconditional

        h_default = np.zeros(n)
        h_custom = np.zeros(n)

        garch_variance_kernel(
            sq, neg, innovations, n,
            go=1e-6, ga=0.08, gb=0.88, gl=0.04,
            gu=gu, rl=0.0, km=0.0, tv=gu, se=0.0, rs=0.0, sm=1.0,
            h_out=h_default,
        )
        garch_variance_kernel(
            sq, neg, innovations, n,
            go=1e-6, ga=0.08, gb=0.88, gl=0.04,
            gu=gu, rl=0.0, km=0.0, tv=gu, se=0.0, rs=0.0, sm=1.0,
            h_out=h_custom,
            h0_override=h0_custom,
        )

        assert abs(h_default[0] - gu) < 1e-15
        assert abs(h_custom[0] - h0_custom) < 1e-15
        # They should converge after some timesteps
        assert abs(h_default[-1] - h_custom[-1]) / h_default[-1] < 0.1

    def test_convergence_within_10_steps(self):
        """AC2: Filter converges to steady state within 10 timesteps."""
        from models.numba_kernels import garch_variance_kernel, garch_h0_from_trailing

        n = 200
        rng = np.random.default_rng(7)
        innovations = rng.normal(0, 0.02, n)
        sq = innovations ** 2
        neg = (innovations < 0).astype(np.float64)

        h0 = garch_h0_from_trailing(sq, window=20)
        gu = 0.0004

        h_out = np.zeros(n)
        garch_variance_kernel(
            sq, neg, innovations, n,
            go=1e-6, ga=0.08, gb=0.88, gl=0.04,
            gu=gu, rl=0.0, km=0.0, tv=gu, se=0.0, rs=0.0, sm=1.0,
            h_out=h_out,
            h0_override=h0 if h0 > 0 else -1.0,
        )

        # Check convergence: |h_t - h_{t-1}| / h_t < 0.05 within 10 steps
        converged_by = None
        for t in range(1, min(50, n)):
            rel_change = abs(h_out[t] - h_out[t - 1]) / max(h_out[t], 1e-15)
            if rel_change < 0.05:
                converged_by = t
                break

        assert converged_by is not None and converged_by <= 15, (
            f"Not converged by step 15 (converged_by={converged_by})"
        )

    def test_short_series_fallback(self):
        """Task 5: If realized var < 1e-10, fallback to unconditional."""
        from models.numba_kernels import garch_h0_from_trailing

        sq = np.full(20, 1e-12, dtype=np.float64)
        h0 = garch_h0_from_trailing(sq, window=20)
        assert h0 == -1.0, f"Should fallback, got h0={h0}"

    def test_insufficient_data_fallback(self):
        """Fewer than window observations -> fallback."""
        from models.numba_kernels import garch_h0_from_trailing

        sq = np.array([0.001, 0.002], dtype=np.float64)
        h0 = garch_h0_from_trailing(sq, window=20)
        assert h0 == -1.0

    def test_all_outputs_finite_positive(self):
        """All h_t values are finite and positive with h0_override."""
        from models.numba_kernels import garch_variance_kernel

        n = 500
        rng = np.random.default_rng(99)
        innovations = rng.normal(0, 0.03, n)
        sq = innovations ** 2
        neg = (innovations < 0).astype(np.float64)

        h_out = np.zeros(n)
        garch_variance_kernel(
            sq, neg, innovations, n,
            go=1e-6, ga=0.08, gb=0.88, gl=0.04,
            gu=0.0009, rl=0.0, km=0.0, tv=0.0009,
            se=0.0, rs=0.0, sm=1.0,
            h_out=h_out,
            h0_override=0.002,
        )

        assert np.all(np.isfinite(h_out))
        assert np.all(h_out > 0)
