"""
Story 2.1: Numerically Stable Geometric Series for Near-Unit-Root Phi
=====================================================================
Taylor expansion for phi in [0.98, 1.0] avoiding catastrophic cancellation.
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

from models.vectorized_ops import vectorized_phi_variance


class TestStableGeometricSeries:
    """Acceptance criteria for Story 2.1."""

    def test_phi_0999_H365_matches_reference(self):
        """AC1: phi=0.999, H=365, q=1e-6 matches double-precision reference."""
        phi = 0.999
        q = 1e-6
        R = 0.0
        H = np.array([365.0])

        result = vectorized_phi_variance(phi, q, R, H)[0]

        # Reference: direct sum to avoid cancellation
        phi_sq = phi * phi
        ref = q * sum(phi_sq ** j for j in range(365))

        assert abs(result - ref) < 1e-10, (
            f"phi=0.999 H=365: got {result:.15e}, ref={ref:.15e}, "
            f"err={abs(result - ref):.2e}"
        )

    def test_phi_near_one_no_nan_inf(self):
        """AC2: phi = 1 - 1e-8: no NaN, no Inf, smooth transition."""
        phi = 1.0 - 1e-8
        q = 1e-6
        R = 1e-4
        horizons = np.array([1, 10, 50, 100, 365, 730])

        result = vectorized_phi_variance(phi, q, R, horizons)
        assert np.all(np.isfinite(result)), f"Non-finite: {result}"
        assert np.all(result > 0), f"Non-positive: {result}"

    def test_phi_exactly_one(self):
        """phi = 1.0 (exact random walk) should give H * q + R."""
        phi = 1.0
        q = 1e-6
        R = 1e-4
        horizons = np.array([1, 10, 100, 365])

        result = vectorized_phi_variance(phi, q, R, horizons)
        expected = q * horizons + R
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_monotonicity_sweep(self):
        """AC2 (Task 2): Sweep phi in [0.990, 1.000], variance is monotonic in phi."""
        q = 1e-6
        R = 0.0
        H = np.array([365.0])
        phis = np.arange(0.990, 1.0005, 0.001)

        variances = [vectorized_phi_variance(phi, q, R, H)[0] for phi in phis]

        for i in range(1, len(variances)):
            assert variances[i] >= variances[i - 1] - 1e-15, (
                f"Non-monotonic: phi={phis[i]:.4f} var={variances[i]:.10e} < "
                f"phi={phis[i-1]:.4f} var={variances[i-1]:.10e}"
            )

    def test_stress_H1000_phi09999(self):
        """Task 5: H=1000, phi=0.9999 should be finite and positive."""
        result = vectorized_phi_variance(0.9999, 1e-6, 0.0, np.array([1000.0]))
        assert np.isfinite(result[0]), f"Non-finite: {result[0]}"
        assert result[0] > 0, f"Non-positive: {result[0]}"

    def test_transition_continuity(self):
        """Verify smooth transition at boundary delta = 1e-6."""
        q = 1e-6
        R = 0.0
        H = np.array([100.0, 365.0])

        # Just inside Taylor zone
        phi_in = np.sqrt(1.0 - 0.5e-6)
        v_in = vectorized_phi_variance(phi_in, q, R, H)

        # Just outside Taylor zone
        phi_out = np.sqrt(1.0 - 2e-6)
        v_out = vectorized_phi_variance(phi_out, q, R, H)

        # Both should be close and continuous
        assert np.all(np.isfinite(v_in))
        assert np.all(np.isfinite(v_out))
        # Relative difference should be small (order of delta)
        rel_diff = np.abs(v_in - v_out) / np.maximum(v_in, v_out)
        assert np.all(rel_diff < 0.01), f"Discontinuity: rel_diff={rel_diff}"

    def test_matches_direct_sum_grid(self):
        """Compare Taylor vs direct sum for a phi x H grid."""
        q = 1e-5
        R = 0.0
        for phi in [0.995, 0.998, 0.999, 0.9995, 0.9999]:
            for H_val in [10, 50, 100, 365]:
                H = np.array([float(H_val)])
                result = vectorized_phi_variance(phi, q, R, H)[0]

                phi_sq = phi * phi
                ref = q * sum(phi_sq ** j for j in range(H_val))

                assert abs(result - ref) / max(abs(ref), 1e-20) < 1e-8, (
                    f"phi={phi}, H={H_val}: result={result:.12e}, "
                    f"ref={ref:.12e}"
                )
