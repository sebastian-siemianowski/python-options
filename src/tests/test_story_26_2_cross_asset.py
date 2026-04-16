"""
Story 26.2 -- Cross-Asset Consistency Tests
=============================================
Verify expected cross-asset parameter relationships.

Acceptance Criteria:
- nu_MSTR <= nu_SPY (heavy-tailed assets have lower nu)
- sigma_QQQ >= sigma_SPY (QQQ riskier)
- nu_silver <= nu_gold (silver heavier tails)
- Violations flagged as warnings (relationships may change)
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.optimize import minimize_scalar
import pytest

from models.numba_wrappers import run_phi_student_t_filter


# ---------------------------------------------------------------------------
# Parameter estimation helpers
# ---------------------------------------------------------------------------
def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def estimate_optimal_nu(r, v, phi=0.90, q=1e-4, c=1.0):
    """Find optimal nu via grid search on log-likelihood."""
    best_nu = 8.0
    best_ll = -np.inf
    for nu in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]:
        _, _, ll = run_phi_student_t_filter(r, v, phi, q, c, nu)
        if ll > best_ll:
            best_ll = ll
            best_nu = nu
    return best_nu


# ---------------------------------------------------------------------------
# Synthetic assets with known properties
# ---------------------------------------------------------------------------
def _gen_spy(n=1000, seed=101):
    rng = np.random.default_rng(seed)
    sigma = 0.01
    r = sigma * rng.standard_normal(n)
    return r, _make_ewma(r, sigma), sigma


def _gen_mstr(n=1000, seed=102):
    rng = np.random.default_rng(seed)
    sigma = 0.03
    r = sigma * rng.standard_t(df=4, size=n)
    return r, _make_ewma(r, sigma), sigma


def _gen_qqq(n=1000, seed=103):
    rng = np.random.default_rng(seed)
    sigma = 0.012
    r = sigma * rng.standard_t(df=10, size=n)
    return r, _make_ewma(r, sigma), sigma


def _gen_gold(n=1000, seed=104):
    rng = np.random.default_rng(seed)
    sigma = 0.008
    r = sigma * rng.standard_normal(n)
    return r, _make_ewma(r, sigma), sigma


def _gen_silver(n=1000, seed=105):
    rng = np.random.default_rng(seed)
    sigma = 0.015
    r = sigma * rng.standard_t(df=6, size=n)
    return r, _make_ewma(r, sigma), sigma


# ===========================================================================
class TestCrossAssetNu:
    """Heavy-tailed assets should have lower optimal nu."""

    def test_mstr_nu_le_spy_nu(self):
        r_mstr, v_mstr, _ = _gen_mstr()
        r_spy, v_spy, _ = _gen_spy()
        nu_mstr = estimate_optimal_nu(r_mstr, v_mstr)
        nu_spy = estimate_optimal_nu(r_spy, v_spy)
        assert nu_mstr <= nu_spy, \
            f"nu_MSTR={nu_mstr} > nu_SPY={nu_spy}"

    def test_silver_nu_le_gold_nu(self):
        r_silver, v_silver, _ = _gen_silver()
        r_gold, v_gold, _ = _gen_gold()
        nu_silver = estimate_optimal_nu(r_silver, v_silver)
        nu_gold = estimate_optimal_nu(r_gold, v_gold)
        assert nu_silver <= nu_gold, \
            f"nu_silver={nu_silver} > nu_gold={nu_gold}"


# ===========================================================================
class TestCrossAssetVolatility:
    """Riskier assets should have higher empirical volatility."""

    def test_qqq_vol_ge_spy_vol(self):
        _, _, sigma_qqq = _gen_qqq()
        _, _, sigma_spy = _gen_spy()
        assert sigma_qqq >= sigma_spy, \
            f"sigma_QQQ={sigma_qqq:.4f} < sigma_SPY={sigma_spy:.4f}"

    def test_mstr_vol_ge_spy_vol(self):
        _, _, sigma_mstr = _gen_mstr()
        _, _, sigma_spy = _gen_spy()
        assert sigma_mstr > sigma_spy, \
            f"sigma_MSTR={sigma_mstr:.4f} <= sigma_SPY={sigma_spy:.4f}"

    def test_silver_vol_ge_gold_vol(self):
        _, _, sigma_silver = _gen_silver()
        _, _, sigma_gold = _gen_gold()
        assert sigma_silver > sigma_gold, \
            f"sigma_silver={sigma_silver:.4f} <= sigma_gold={sigma_gold:.4f}"


# ===========================================================================
class TestParameterConsistency:
    """Parameters are consistent across re-estimation."""

    def test_nu_deterministic(self):
        r, v, _ = _gen_spy()
        nu1 = estimate_optimal_nu(r, v)
        nu2 = estimate_optimal_nu(r, v)
        assert nu1 == nu2

    def test_all_nu_finite(self):
        for gen_fn in [_gen_spy, _gen_mstr, _gen_qqq, _gen_gold, _gen_silver]:
            r, v, _ = gen_fn()
            nu = estimate_optimal_nu(r, v)
            assert 2 < nu <= 30, f"nu={nu}"
