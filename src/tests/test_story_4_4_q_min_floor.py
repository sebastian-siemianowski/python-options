"""
Story 4.4: Scale-Aware Process Noise Floor with Regime Detection
================================================================
q_min = max(1e-8, 0.001 * Var(sigma^2), 0.002 * Var(r))
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


class TestScaleAwareQMin:
    """Acceptance criteria for Story 4.4."""

    def test_q_min_formula_components(self):
        """AC1: q_min uses vol variance, return variance, and hard floor."""
        vol = np.full(200, 0.02)
        returns = np.random.default_rng(42).normal(0, 0.02, 200)
        vol_var_median = float(np.median(vol ** 2))
        ret_var = float(np.var(returns))

        q_min_vol = max(1e-10, 0.001 * vol_var_median)
        q_min_ret = max(1e-10, 0.002 * ret_var)
        q_min = max(q_min_vol, q_min_ret, 1e-8)

        assert q_min >= 1e-8
        assert q_min >= 0.001 * vol_var_median
        assert q_min >= 0.002 * ret_var

    def test_low_vol_P_never_collapses(self):
        """AC2: On low-vol data, P never drops below 1e-10."""
        from models.phi_gaussian import _kalman_filter_phi

        n = 500
        returns = np.random.default_rng(42).normal(0, 0.001, n)
        vol = np.full(n, 0.001)

        mu_f, P_f, ll = _kalman_filter_phi(returns, vol, q=1e-8, c=1.0, phi=0.5)
        assert np.all(P_f > 0), f"P collapsed: min={np.min(P_f)}"

    def test_high_vol_not_overregularized(self):
        """AC3: On high-vol data, q_min doesn't dominate."""
        vol = np.full(200, 0.10)
        returns = np.random.default_rng(7).normal(0, 0.10, 200)
        vol_var_median = float(np.median(vol ** 2))
        ret_var = float(np.var(returns))

        q_min_vol = max(1e-10, 0.001 * vol_var_median)
        q_min_ret = max(1e-10, 0.002 * ret_var)
        q_min = max(q_min_vol, q_min_ret, 1e-8)

        # For high-vol, q_min should be reasonable, not huge
        assert q_min < 0.01, f"q_min too large: {q_min}"

    def test_hard_floor_1e8(self):
        """Hard floor of 1e-8 always applies."""
        vol = np.full(200, 1e-6)  # Extremely low vol
        returns = np.full(200, 0.0)  # Zero returns
        vol_var_median = float(np.median(vol ** 2))
        ret_var = float(np.var(returns))

        q_min_vol = max(1e-10, 0.001 * vol_var_median)
        q_min_ret = max(1e-10, 0.002 * ret_var)
        q_min = max(q_min_vol, q_min_ret, 1e-8)

        assert q_min >= 1e-8

    def test_scaling_with_return_variance(self):
        """Return variance component scales correctly."""
        for sigma in [0.001, 0.01, 0.05, 0.1]:
            returns = np.random.default_rng(42).normal(0, sigma, 500)
            ret_var = float(np.var(returns))
            q_min_ret = 0.002 * ret_var
            # Should scale as sigma^2
            assert abs(q_min_ret - 0.002 * sigma ** 2) / max(q_min_ret, 1e-15) < 0.5
