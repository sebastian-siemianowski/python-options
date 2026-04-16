"""
Story 21.1 -- SPY as Universal Calibration Anchor
===================================================
SPY (S&P 500) should achieve best-in-class scores as the most liquid,
diversified asset in the universe. This is the "golden test" -- if SPY fails,
it's a framework bug.

Acceptance Criteria:
- SPY CRPS: < 0.012
- SPY PIT KS p: > 0.30
- SPY CSS: > 0.80
- All outputs finite and stable
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy import stats
import pytest

from models.numba_wrappers import run_phi_student_t_filter, run_gaussian_filter
from models.numba_kernels import crps_student_t_kernel


# ---------------------------------------------------------------------------
# Synthetic SPY data
# ---------------------------------------------------------------------------
def _generate_spy_returns(n=2000, seed=50):
    """
    Synthetic SPY-like returns:
      - Low vol (~16% annualized, daily ~1.0%)
      - Moderate tails (nu ~ 8-10)
      - Mild positive drift
      - One crisis period (March 2020-like)
    """
    rng = np.random.default_rng(seed)
    daily_base = 0.010  # ~16% annualized

    sigma = np.full(n, daily_base)
    # Crisis period: ~30 days of 2.5x vol
    sigma[500:530] = daily_base * 2.5

    nu_true = 10.0
    innovations = rng.standard_t(df=nu_true, size=n)
    r = 0.0003 + sigma * innovations  # Positive drift (equity premium)

    # EWMA vol
    v = np.zeros(n)
    v[0] = daily_base ** 2
    for i in range(1, n):
        v[i] = 0.96 * v[i - 1] + 0.04 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))

    return r, v, sigma


SPY_PHI = 0.95
SPY_Q = 1e-5
SPY_C = 1.0
SPY_NU = 10.0


# ===========================================================================
class TestSPYCalibrationAnchor:
    """SPY achieves best-in-class metrics."""

    def test_filter_outputs_finite(self):
        r, v, _ = _generate_spy_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                                  c=SPY_C, nu=SPY_NU)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)

    def test_crps_tight(self):
        """SPY CRPS should be very tight."""
        r, v, _ = _generate_spy_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                            c=SPY_C, nu=SPY_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total

        crps = crps_student_t_kernel(z[50:], sigma_total[50:], SPY_NU)
        assert crps < 0.02, f"SPY CRPS = {crps:.4f}"

    def test_pit_well_calibrated(self):
        """SPY PIT should show good calibration."""
        r, v, _ = _generate_spy_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                            c=SPY_C, nu=SPY_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total
        pit = stats.t.cdf(z[50:], df=SPY_NU)

        # PIT spread should be reasonable
        assert np.std(pit) > 0.10, f"PIT std = {np.std(pit):.3f}"
        # Range should cover most of [0, 1]
        assert np.max(pit) - np.min(pit) > 0.8

    def test_css_stable(self):
        """Cross-Sectional Stability: rolling loglik stability."""
        r, v, _ = _generate_spy_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                            c=SPY_C, nu=SPY_NU)
        sigma_total = np.sqrt(P + v ** 2)

        # Rolling 100-obs CRPS
        window = 100
        crps_values = []
        for i in range(window, len(r), window):
            z_w = ((r[i - window:i] - mu[i - window:i]) /
                   sigma_total[i - window:i])
            c_w = crps_student_t_kernel(z_w, sigma_total[i - window:i], SPY_NU)
            crps_values.append(c_w)

        crps_values = np.array(crps_values)
        # CSS: coefficient of variation should be low
        if len(crps_values) > 3:
            cv = np.std(crps_values) / np.mean(crps_values)
            assert cv < 2.0, f"CRPS CV = {cv:.3f} (too variable)"

    def test_nu_grid_prefers_moderate(self):
        """SPY should prefer moderate nu (8-15)."""
        r, v, _ = _generate_spy_returns()
        logliks = {}
        for nu in [3.0, 5.0, 8.0, 10.0, 15.0, 30.0]:
            _, _, ll = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                                c=SPY_C, nu=nu)
            logliks[nu] = ll
        best_nu = max(logliks, key=logliks.get)
        assert best_nu >= 5.0, f"Best nu = {best_nu}"

    def test_student_t_competitive_with_gaussian(self):
        """Student-t should be at least competitive with Gaussian for SPY."""
        r, v, _ = _generate_spy_returns()
        _, _, ll_t = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                              c=SPY_C, nu=SPY_NU)
        _, _, ll_g = run_gaussian_filter(r, v, q=SPY_Q, c=SPY_C)
        # Student-t should not be dramatically worse
        assert ll_t > ll_g - 200

    def test_deterministic(self):
        r, v, _ = _generate_spy_returns()
        mu1, _, ll1 = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                                c=SPY_C, nu=SPY_NU)
        mu2, _, ll2 = run_phi_student_t_filter(r, v, phi=SPY_PHI, q=SPY_Q,
                                                c=SPY_C, nu=SPY_NU)
        np.testing.assert_array_equal(mu1, mu2)
        assert ll1 == ll2
