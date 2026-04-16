"""
Story 14.3 – State-Space Equilibrium Estimation Accuracy
=========================================================
Verify RTS smoother produces lag-free equilibrium that adapts to structural
shifts, and MR signal is approximately mean-zero.
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
    rts_smoother,
    compute_equilibrium_and_mr_signal,
    run_gaussian_filter,
)


def _run_filter_for_smoother(returns, vol, q=1e-5, c=0.01, P0=1e-4):
    """Run Gaussian filter to get filtered states for smoother input."""
    mu_f, P_f, _ = run_gaussian_filter(returns, vol, q, c, P0)
    return mu_f, P_f


class TestRTSSmoother:
    """Rauch-Tung-Striebel smoother basic properties."""

    def test_output_shape(self):
        """Smoother output has same length as input."""
        mu_f = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        P_f = np.full(5, 0.001)
        mu_s, P_s = rts_smoother(mu_f, P_f, 0.98, 1e-5)
        assert len(mu_s) == 5
        assert len(P_s) == 5

    def test_last_point_unchanged(self):
        """Final smoothed state equals filtered state (no future data)."""
        mu_f = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        P_f = np.full(5, 0.001)
        mu_s, P_s = rts_smoother(mu_f, P_f, 0.98, 1e-5)
        assert mu_s[-1] == mu_f[-1]
        assert P_s[-1] == P_f[-1]

    def test_smoother_variance_le_filtered(self):
        """Smoothed variance <= filtered variance (smoother uses more data)."""
        rng = np.random.default_rng(42)
        n = 200
        vol = np.full(n, 0.02)
        returns = rng.normal(0.001, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol)
        mu_s, P_s = rts_smoother(mu_f, P_f, 0.98, 1e-5)
        # Smoother variance should be <= filtered for all t < T
        # Allow small numerical tolerance
        assert np.all(P_s[:-1] <= P_f[:-1] + 1e-15)

    def test_constant_state_recovered(self):
        """If true state is constant, smoother recovers it."""
        n = 300
        true_mu = 0.005
        vol = np.full(n, 0.02)
        rng = np.random.default_rng(42)
        returns = rng.normal(true_mu, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol, q=1e-6)
        mu_s, _ = rts_smoother(mu_f, P_f, 0.999, 1e-6)
        # Smoothed estimate in the middle should be close to true_mu
        assert abs(np.mean(mu_s[50:250]) - true_mu) < 0.005


class TestStructuralShiftAdaptation:
    """Smoother adapts to level shifts within 5 days."""

    def test_level_shift(self):
        """Inject a level shift; equilibrium adapts within 5 days."""
        n = 200
        vol = np.full(n, 0.02)
        returns = np.full(n, 0.001)
        # Structural break at t=100: mean shifts from 0.001 to 0.01
        returns[100:] = 0.01
        mu_f, P_f = _run_filter_for_smoother(returns, vol, q=1e-4)
        mu_s, _ = rts_smoother(mu_f, P_f, 0.98, 1e-4)
        # By t=105 (5 days after shift), equilibrium should be closer to 0.01 than 0.001
        mid_old = np.mean(mu_s[80:100])
        at_shift_5 = mu_s[105]
        mid_new = np.mean(mu_s[120:180])
        # The smoothed value at t=105 should be more than halfway toward new level
        assert at_shift_5 > (mid_old + mid_new) / 2.0


class TestMeanReversionSignal:
    """MR signal is approximately mean-zero over long horizons."""

    def test_mr_mean_near_zero(self):
        """Mean of MR signal < 0.001 in absolute value."""
        rng = np.random.default_rng(42)
        n = 500
        vol = np.full(n, 0.02)
        returns = rng.normal(0.001, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol)
        result = compute_equilibrium_and_mr_signal(mu_f, P_f, 0.98, 1e-5, kappa=0.1)
        assert abs(result["mr_mean"]) < 0.001

    def test_mr_std_positive(self):
        """MR signal has positive standard deviation (not degenerate)."""
        rng = np.random.default_rng(42)
        n = 500
        vol = np.full(n, 0.02)
        returns = rng.normal(0.001, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol)
        result = compute_equilibrium_and_mr_signal(mu_f, P_f, 0.98, 1e-5, kappa=0.1)
        assert result["mr_std"] > 0

    def test_mr_kappa_scaling(self):
        """Doubling kappa doubles MR signal magnitude."""
        rng = np.random.default_rng(42)
        n = 300
        vol = np.full(n, 0.02)
        returns = rng.normal(0.001, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol)
        r1 = compute_equilibrium_and_mr_signal(mu_f, P_f, 0.98, 1e-5, kappa=0.1)
        r2 = compute_equilibrium_and_mr_signal(mu_f, P_f, 0.98, 1e-5, kappa=0.2)
        np.testing.assert_allclose(r2["mr_signal"], 2.0 * r1["mr_signal"], atol=1e-15)


class TestEquilibriumSmoothness:
    """Equilibrium is smoother than filtered state."""

    def test_equilibrium_lower_variance(self):
        """Smoothed equilibrium has lower variance than filtered state."""
        rng = np.random.default_rng(42)
        n = 500
        vol = np.full(n, 0.02)
        returns = rng.normal(0.001, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol, q=1e-5)
        mu_s, _ = rts_smoother(mu_f, P_f, 0.98, 1e-5)
        # Smoothed should be less variable (uses all data, not just past)
        var_filtered = np.var(np.diff(mu_f))
        var_smooth = np.var(np.diff(mu_s))
        assert var_smooth <= var_filtered

    def test_equilibrium_correlated_with_filtered(self):
        """Smoothed equilibrium is highly correlated with filtered state."""
        rng = np.random.default_rng(42)
        n = 500
        vol = np.full(n, 0.02)
        returns = rng.normal(0.001, 0.02, n)
        mu_f, P_f = _run_filter_for_smoother(returns, vol, q=1e-5)
        mu_s, _ = rts_smoother(mu_f, P_f, 0.98, 1e-5)
        corr = np.corrcoef(mu_f, mu_s)[0, 1]
        assert corr > 0.95
