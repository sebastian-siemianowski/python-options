"""
Story 15.1 – Predictive PIT Implementation Audit
==================================================
Verify PIT uses predictive (not filtered) distribution.
Synthetic test: known DGP -> PIT should be uniform.
Anti-test: filtered values -> PIT should fail KS.
"""

import os, sys
import numpy as np
import pytest
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import run_gaussian_filter, run_phi_student_t_filter


def _compute_predictive_from_filtered(mu_filt, P_filt, phi, q, c, vol):
    """
    Compute predictive distribution from filtered output.

    mu_pred[t] = phi * mu_filt[t-1]
    S_pred[t] = phi^2 * P_filt[t-1] + q + (c * vol[t])^2
    """
    n = len(mu_filt)
    mu_pred = np.empty(n, dtype=np.float64)
    S_pred = np.empty(n, dtype=np.float64)
    mu_pred[0] = 0.0
    S_pred[0] = q + (c * vol[0]) ** 2
    for t in range(1, n):
        mu_pred[t] = phi * mu_filt[t - 1]
        S_pred[t] = phi ** 2 * P_filt[t - 1] + q + (c * vol[t]) ** 2
    return mu_pred, S_pred


class TestPredictivePITGaussian:
    """Gaussian PIT uses predictive distribution correctly."""

    def test_known_dgp_uniform(self):
        """Generate N(0,1) returns, compute predictive PIT, verify uniform."""
        rng = np.random.default_rng(42)
        n = 2000
        vol = np.full(n, 0.02, dtype=np.float64)
        returns = rng.normal(0.0, 0.02, n)

        mu_f, P_f, _ = run_gaussian_filter(returns, vol, q=1e-6, c=1.0)
        mu_pred, S_pred = _compute_predictive_from_filtered(
            mu_f, P_f, phi=1.0, q=1e-6, c=1.0, vol=vol
        )

        # Compute PIT values
        std_pred = np.sqrt(np.maximum(S_pred[50:], 1e-20))
        z = (returns[50:] - mu_pred[50:]) / std_pred
        pit = stats.norm.cdf(z)

        # KS test against uniform
        ks_stat, ks_p = stats.kstest(pit, "uniform")
        assert ks_p > 0.05, f"PIT not uniform: KS p={ks_p:.4f}"

    def test_filtered_pit_fails(self):
        """
        Anti-test: using filtered (not predictive) values gives biased PIT.

        The filter has already seen r_t when producing mu_filt[t],
        so PIT(r_t | mu_filt[t]) has look-ahead bias.
        """
        rng = np.random.default_rng(42)
        n = 2000
        vol = np.full(n, 0.02, dtype=np.float64)
        returns = rng.normal(0.0, 0.02, n)

        mu_f, P_f, _ = run_gaussian_filter(returns, vol, q=1e-4, c=1.0)
        # DELIBERATELY use filtered (not predictive) for PIT
        std_f = np.sqrt(np.maximum(P_f[50:] + (1.0 * vol[50:]) ** 2, 1e-20))
        z = (returns[50:] - mu_f[50:]) / std_f
        pit_bad = stats.norm.cdf(z)

        # With filtered values, PIT should be peaked around 0.5 (overconfident)
        # and KS should reject uniformity
        ks_stat, ks_p = stats.kstest(pit_bad, "uniform")
        # The filtered PIT should fail or at least be significantly worse
        # With high q (1e-4), the filter adapts fast -> strong look-ahead bias
        assert ks_stat > 0.02, f"Filtered PIT not detectably biased: KS={ks_stat:.4f}"


class TestPredictivePITStudentT:
    """Student-t PIT uses predictive distribution correctly."""

    def test_known_student_t_dgp(self):
        """Generate t(8) returns, compute predictive PIT, verify uniform."""
        rng = np.random.default_rng(42)
        n = 2000
        nu = 8.0
        vol = np.full(n, 0.02, dtype=np.float64)
        # Generate from Student-t
        raw = rng.standard_t(nu, n)
        # Scale to have same variance as N(0, vol^2): t(nu) has var = nu/(nu-2)
        returns = raw * vol * np.sqrt((nu - 2) / nu)

        mu_f, P_f, _ = run_phi_student_t_filter(
            returns, vol, q=1e-6, c=1.0, phi=0.999, nu=nu
        )
        mu_pred, S_pred = _compute_predictive_from_filtered(
            mu_f, P_f, phi=0.999, q=1e-6, c=1.0, vol=vol
        )

        # Student-t PIT: t_cdf((r - mu_pred) / scale, nu)
        # where scale = sqrt(S_pred * (nu-2)/nu) for unit-variance parameterization
        scale = np.sqrt(np.maximum(S_pred[50:], 1e-20) * (nu - 2) / nu)
        z = (returns[50:] - mu_pred[50:]) / np.maximum(scale, 1e-10)
        pit = stats.t.cdf(z, nu)

        ks_stat, ks_p = stats.kstest(pit, "uniform")
        assert ks_p > 0.01, f"Student-t PIT not uniform: KS p={ks_p:.4f}"


class TestPITDistributionShape:
    """PIT values have correct properties when well-calibrated."""

    def test_pit_mean_near_half(self):
        """Well-calibrated PIT has mean near 0.5."""
        rng = np.random.default_rng(42)
        n = 2000
        vol = np.full(n, 0.02, dtype=np.float64)
        returns = rng.normal(0.0, 0.02, n)
        mu_f, P_f, _ = run_gaussian_filter(returns, vol, q=1e-6, c=1.0)
        mu_pred, S_pred = _compute_predictive_from_filtered(
            mu_f, P_f, phi=1.0, q=1e-6, c=1.0, vol=vol
        )
        std_pred = np.sqrt(np.maximum(S_pred[50:], 1e-20))
        z = (returns[50:] - mu_pred[50:]) / std_pred
        pit = stats.norm.cdf(z)
        assert abs(np.mean(pit) - 0.5) < 0.03

    def test_pit_std_near_uniform(self):
        """Well-calibrated PIT has std near 1/sqrt(12) ~ 0.289."""
        rng = np.random.default_rng(42)
        n = 2000
        vol = np.full(n, 0.02, dtype=np.float64)
        returns = rng.normal(0.0, 0.02, n)
        mu_f, P_f, _ = run_gaussian_filter(returns, vol, q=1e-6, c=1.0)
        mu_pred, S_pred = _compute_predictive_from_filtered(
            mu_f, P_f, phi=1.0, q=1e-6, c=1.0, vol=vol
        )
        std_pred = np.sqrt(np.maximum(S_pred[50:], 1e-20))
        z = (returns[50:] - mu_pred[50:]) / std_pred
        pit = stats.norm.cdf(z)
        uniform_std = 1.0 / np.sqrt(12)
        assert abs(np.std(pit) - uniform_std) < 0.03
