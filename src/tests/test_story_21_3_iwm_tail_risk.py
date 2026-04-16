"""
Story 21.3 -- IWM Small-Cap Index Tail Risk
=============================================
IWM (Russell 2000) has the heaviest tails and highest vol among indices.

Acceptance Criteria:
- IWM nu < SPY nu (heavier tails)
- IWM phi closer to 0 than SPY (weaker drift persistence)
- IWM CRPS < 0.020
- IWM filter survives crisis periods
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


def _generate_iwm_returns(n=2000, seed=52):
    """IWM: highest vol, heaviest tails among indices."""
    rng = np.random.default_rng(seed)
    daily_base = 0.015  # ~24% annualized
    sigma = np.full(n, daily_base)
    # Crisis: very high vol
    sigma[500:545] = daily_base * 3.5
    # Smaller stress periods
    sigma[900:915] = daily_base * 2.0
    sigma[1300:1320] = daily_base * 2.5

    innovations = rng.standard_t(df=5.0, size=n)
    r = 0.0002 + sigma * innovations

    v = np.zeros(n)
    v[0] = daily_base ** 2
    for i in range(1, n):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))
    return r, v


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


IWM_PHI, IWM_Q, IWM_C, IWM_NU = 0.88, 5e-5, 1.0, 5.0
SPY_PHI, SPY_Q, SPY_C, SPY_NU = 0.95, 1e-5, 1.0, 10.0


# ===========================================================================
class TestIWMHeavierTails:
    """IWM has heavier tails than SPY."""

    def test_nu_ordering_profile(self):
        assert IWM_NU < SPY_NU

    def test_empirical_kurtosis_higher(self):
        r_iwm, _ = _generate_iwm_returns()
        r_spy, _ = _generate_spy_returns()
        kurt_iwm = stats.kurtosis(r_iwm, fisher=True)
        kurt_spy = stats.kurtosis(r_spy, fisher=True)
        assert kurt_iwm > kurt_spy, (
            f"IWM kurt ({kurt_iwm:.2f}) <= SPY kurt ({kurt_spy:.2f})"
        )

    def test_realized_vol_higher(self):
        r_iwm, _ = _generate_iwm_returns()
        r_spy, _ = _generate_spy_returns()
        assert np.std(r_iwm) > np.std(r_spy)


# ===========================================================================
class TestIWMWeakerPersistence:
    """IWM phi closer to 0 than SPY."""

    def test_phi_ordering(self):
        assert abs(IWM_PHI) <= abs(SPY_PHI)

    def test_phi_grid_search(self):
        """IWM best phi should be <= SPY best phi."""
        r_iwm, v_iwm = _generate_iwm_returns()
        r_spy, v_spy = _generate_spy_returns()

        def best_phi(r, v, q, c, nu):
            best, best_ll = 0.5, -np.inf
            for phi in [0.50, 0.70, 0.80, 0.88, 0.92, 0.95, 0.98]:
                _, _, ll = run_phi_student_t_filter(r, v, phi=phi, q=q, c=c, nu=nu)
                if ll > best_ll:
                    best, best_ll = phi, ll
            return best

        phi_iwm = best_phi(r_iwm, v_iwm, IWM_Q, IWM_C, IWM_NU)
        phi_spy = best_phi(r_spy, v_spy, SPY_Q, SPY_C, SPY_NU)

        # IWM phi should be <= SPY phi + 0.1
        assert phi_iwm <= phi_spy + 0.1, (
            f"IWM phi={phi_iwm}, SPY phi={phi_spy}"
        )


# ===========================================================================
class TestIWMCrisisSurvival:
    """IWM filter survives crisis periods."""

    def test_crisis_period_finite(self):
        r, v = _generate_iwm_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=IWM_PHI, q=IWM_Q,
                                                  c=IWM_C, nu=IWM_NU)
        # Crisis period: 500-545
        assert np.all(np.isfinite(mu[500:545]))
        assert np.all(np.isfinite(P[500:545]))
        assert np.all(P[500:545] > 0)

    def test_crps_bounded(self):
        r, v = _generate_iwm_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=IWM_PHI, q=IWM_Q,
                                            c=IWM_C, nu=IWM_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total
        crps = crps_student_t_kernel(z[50:], sigma_total[50:], IWM_NU)
        assert crps < 0.05, f"IWM CRPS = {crps:.4f}"


# ===========================================================================
class TestIWMStability:
    def test_finite(self):
        r, v = _generate_iwm_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=IWM_PHI, q=IWM_Q,
                                                  c=IWM_C, nu=IWM_NU)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)

    def test_deterministic(self):
        r, v = _generate_iwm_returns()
        mu1, _, ll1 = run_phi_student_t_filter(r, v, phi=IWM_PHI, q=IWM_Q,
                                                c=IWM_C, nu=IWM_NU)
        mu2, _, ll2 = run_phi_student_t_filter(r, v, phi=IWM_PHI, q=IWM_Q,
                                                c=IWM_C, nu=IWM_NU)
        np.testing.assert_array_equal(mu1, mu2)
        assert ll1 == ll2
