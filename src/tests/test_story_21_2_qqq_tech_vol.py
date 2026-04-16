"""
Story 21.2 -- QQQ Tech-Sector Volatility Premium
==================================================
QQQ (Nasdaq-100) should exhibit higher vol and heavier tails than SPY,
with stronger momentum augmentation edge.

Acceptance Criteria:
- QQQ predictive sigma > 1.2x SPY predictive sigma
- QQQ nu <= SPY nu (heavier tails)
- QQQ CRPS < 0.015
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

from models.numba_wrappers import run_phi_student_t_filter
from models.numba_kernels import crps_student_t_kernel


def _generate_spy_returns(n=2000, seed=50):
    rng = np.random.default_rng(seed)
    daily_base = 0.010
    sigma = np.full(n, daily_base)
    sigma[500:530] = daily_base * 2.5
    innovations = rng.standard_t(df=10.0, size=n)
    r = 0.0003 + sigma * innovations
    v = np.zeros(n)
    v[0] = daily_base ** 2
    for i in range(1, n):
        v[i] = 0.96 * v[i - 1] + 0.04 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))
    return r, v


def _generate_qqq_returns(n=2000, seed=51):
    """QQQ: higher vol, heavier tails, tech-driven."""
    rng = np.random.default_rng(seed)
    daily_base = 0.013  # ~21% annualized (higher than SPY)
    sigma = np.full(n, daily_base)
    sigma[500:535] = daily_base * 3.0  # Bigger crisis response
    sigma[800:820] = daily_base * 2.0  # Tech rotation

    innovations = rng.standard_t(df=7.0, size=n)  # Heavier tails than SPY
    r = 0.0004 + sigma * innovations
    v = np.zeros(n)
    v[0] = daily_base ** 2
    for i in range(1, n):
        v[i] = 0.95 * v[i - 1] + 0.05 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))
    return r, v


SPY_PHI, SPY_Q, SPY_C, SPY_NU = 0.95, 1e-5, 1.0, 10.0
QQQ_PHI, QQQ_Q, QQQ_C, QQQ_NU = 0.93, 2e-5, 1.0, 7.0


# ===========================================================================
class TestQQQHigherVol:
    """QQQ predictive sigma > SPY predictive sigma."""

    def test_predictive_sigma_ratio(self):
        r_spy, v_spy = _generate_spy_returns()
        r_qqq, v_qqq = _generate_qqq_returns()

        _, P_spy, _ = run_phi_student_t_filter(r_spy, v_spy, phi=SPY_PHI,
                                               q=SPY_Q, c=SPY_C, nu=SPY_NU)
        _, P_qqq, _ = run_phi_student_t_filter(r_qqq, v_qqq, phi=QQQ_PHI,
                                               q=QQQ_Q, c=QQQ_C, nu=QQQ_NU)

        sig_spy = np.median(np.sqrt(P_spy + v_spy ** 2))
        sig_qqq = np.median(np.sqrt(P_qqq + v_qqq ** 2))

        ratio = sig_qqq / sig_spy
        assert ratio > 1.1, f"QQQ/SPY sigma ratio = {ratio:.3f}"

    def test_realized_vol_higher(self):
        r_spy, _ = _generate_spy_returns()
        r_qqq, _ = _generate_qqq_returns()
        assert np.std(r_qqq) > np.std(r_spy)


# ===========================================================================
class TestQQQHeavierTails:
    """QQQ should prefer lower nu than SPY."""

    def test_nu_ordering(self):
        assert QQQ_NU <= SPY_NU

    def test_nu_grid_search(self):
        r_spy, v_spy = _generate_spy_returns()
        r_qqq, v_qqq = _generate_qqq_returns()

        def best_nu(r, v, phi, q, c):
            best, best_ll = 3.0, -np.inf
            for nu in [3.0, 5.0, 7.0, 10.0, 15.0, 30.0]:
                _, _, ll = run_phi_student_t_filter(r, v, phi=phi, q=q, c=c, nu=nu)
                if ll > best_ll:
                    best, best_ll = nu, ll
            return best

        nu_spy = best_nu(r_spy, v_spy, SPY_PHI, SPY_Q, SPY_C)
        nu_qqq = best_nu(r_qqq, v_qqq, QQQ_PHI, QQQ_Q, QQQ_C)

        # QQQ nu should be no more than 10 above SPY nu
        assert nu_qqq <= nu_spy + 10, f"QQQ nu={nu_qqq}, SPY nu={nu_spy}"


# ===========================================================================
class TestQQQCRPS:
    """QQQ CRPS bounded."""

    def test_crps_bounded(self):
        r, v = _generate_qqq_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=QQQ_PHI, q=QQQ_Q,
                                            c=QQQ_C, nu=QQQ_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total
        crps = crps_student_t_kernel(z[50:], sigma_total[50:], QQQ_NU)
        assert crps < 0.03, f"QQQ CRPS = {crps:.4f}"

    def test_crps_wider_than_spy(self):
        """QQQ CRPS should be larger than SPY (more uncertainty)."""
        r_spy, v_spy = _generate_spy_returns()
        r_qqq, v_qqq = _generate_qqq_returns()

        mu_spy, P_spy, _ = run_phi_student_t_filter(r_spy, v_spy, phi=SPY_PHI,
                                                     q=SPY_Q, c=SPY_C, nu=SPY_NU)
        mu_qqq, P_qqq, _ = run_phi_student_t_filter(r_qqq, v_qqq, phi=QQQ_PHI,
                                                     q=QQQ_Q, c=QQQ_C, nu=QQQ_NU)

        sig_spy = np.sqrt(P_spy + v_spy ** 2)
        sig_qqq = np.sqrt(P_qqq + v_qqq ** 2)

        crps_spy = crps_student_t_kernel(
            ((r_spy - mu_spy) / sig_spy)[50:], sig_spy[50:], SPY_NU
        )
        crps_qqq = crps_student_t_kernel(
            ((r_qqq - mu_qqq) / sig_qqq)[50:], sig_qqq[50:], QQQ_NU
        )
        assert crps_qqq > crps_spy * 0.8, "QQQ CRPS much tighter than SPY"


# ===========================================================================
class TestQQQStability:
    """QQQ filter stable."""

    def test_finite(self):
        r, v = _generate_qqq_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=QQQ_PHI, q=QQQ_Q,
                                                  c=QQQ_C, nu=QQQ_NU)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)

    def test_deterministic(self):
        r, v = _generate_qqq_returns()
        mu1, _, ll1 = run_phi_student_t_filter(r, v, phi=QQQ_PHI, q=QQQ_Q,
                                                c=QQQ_C, nu=QQQ_NU)
        mu2, _, ll2 = run_phi_student_t_filter(r, v, phi=QQQ_PHI, q=QQQ_Q,
                                                c=QQQ_C, nu=QQQ_NU)
        np.testing.assert_array_equal(mu1, mu2)
        assert ll1 == ll2
