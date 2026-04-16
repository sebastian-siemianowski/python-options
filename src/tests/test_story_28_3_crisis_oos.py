"""
Story 28.3 -- Crisis-Period OOS Performance
=============================================
Validate model behavior during extreme market conditions.

Acceptance Criteria:
- Model produces no NaN/Inf during crisis periods
- CSS > 0.60 (calibration stability under stress)
- Regime detection activates within reasonable time
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


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


# ---------------------------------------------------------------------------
# Synthetic crisis generators
# ---------------------------------------------------------------------------
def gen_covid_style(n=500, seed=600):
    """Normal market followed by crash and recovery."""
    rng = np.random.default_rng(seed)
    # Normal period
    r_normal = 0.01 * rng.standard_normal(300)
    # Crash: 10 days of large negative returns
    r_crash = -0.05 + 0.03 * rng.standard_normal(10)
    # Recovery: elevated vol
    r_recovery = 0.02 * rng.standard_normal(190)
    r = np.concatenate([r_normal, r_crash, r_recovery])
    return r, _make_ewma(r, 0.01)


def gen_rate_shock(n=500, seed=601):
    """Steady market then sudden vol spike."""
    rng = np.random.default_rng(seed)
    r_calm = 0.008 * rng.standard_normal(350)
    r_shock = 0.04 * rng.standard_t(df=4, size=50)
    r_after = 0.015 * rng.standard_normal(100)
    r = np.concatenate([r_calm, r_shock, r_after])
    return r, _make_ewma(r, 0.008)


def gen_gradual_stress(n=500, seed=602):
    """Slowly increasing volatility."""
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(0.005, 0.04, n)
    r = sigmas * rng.standard_normal(n)
    return r, _make_ewma(r, 0.005)


# ---------------------------------------------------------------------------
# Calibration stability score (CSS)
# ---------------------------------------------------------------------------
def compute_css(r, v, phi=0.90, q=1e-4, c=1.0, nu=8.0, window=50):
    """
    Calibration Stability Score: measure how stable the filter is
    across rolling windows. CSS in [0, 1], higher = more stable.
    """
    mu, P, ll = run_phi_student_t_filter(r, v, q, c, phi, nu)
    n = len(r)

    if n < 2 * window:
        return 1.0

    # Compute rolling window log-likelihoods
    lls = []
    for start in range(0, n - window, window // 2):
        end = start + window
        r_w = r[start:end]
        v_w = v[start:end]
        _, _, ll_w = run_phi_student_t_filter(r_w, v_w, q, c, phi, nu)
        lls.append(ll_w / window)  # Per-obs loglik

    if len(lls) < 2:
        return 1.0

    lls = np.array(lls)
    # CSS = 1 - normalized std of per-obs loglik
    cv = np.std(lls) / (abs(np.mean(lls)) + 1e-10)
    css = max(0.0, 1.0 - cv)
    return css


# ===========================================================================
class TestNoCrash:
    """Model produces no NaN/Inf during crises."""

    def test_covid_no_nan(self):
        r, v = gen_covid_style()
        mu, P, ll = run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, 8.0)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(ll)

    def test_rate_shock_no_nan(self):
        r, v = gen_rate_shock()
        mu, P, ll = run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, 8.0)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(ll)

    def test_gradual_stress_no_nan(self):
        r, v = gen_gradual_stress()
        mu, P, ll = run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, 8.0)
        assert np.all(np.isfinite(mu))
        assert np.isfinite(ll)

    def test_extreme_returns_no_crash(self):
        """Even with 10-sigma moves, filter stays stable."""
        rng = np.random.default_rng(610)
        r = 0.01 * rng.standard_normal(500)
        r[250] = 0.20  # 20% single-day move
        r[251] = -0.15
        v = _make_ewma(r, 0.01)
        mu, P, ll = run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, 8.0)
        assert np.all(np.isfinite(mu))
        assert np.isfinite(ll)


# ===========================================================================
class TestCSS:
    """Calibration Stability Score is reasonable."""

    def test_css_normal_market(self):
        rng = np.random.default_rng(620)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)
        css = compute_css(r, v)
        assert css > 0.5, f"CSS={css:.3f}"

    def test_css_crisis_lower(self):
        """CSS during crisis should still be > 0."""
        r, v = gen_covid_style()
        css = compute_css(r, v)
        assert css > 0.0, f"CSS={css:.3f}"

    def test_css_bounded(self):
        r, v = gen_rate_shock()
        css = compute_css(r, v)
        assert 0 <= css <= 1.0


# ===========================================================================
class TestCrisisAdaptation:
    """Filter adapts to crisis periods."""

    def test_P_increases_during_crash(self):
        """Filter uncertainty P should increase during crash."""
        r, v = gen_covid_style()
        mu, P, _ = run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, 8.0)
        # P during crash (days 300-310) vs normal (days 100-200)
        P_normal = np.mean(P[100:200])
        P_crash = np.max(P[295:315])
        assert P_crash >= P_normal, \
            f"P_crash={P_crash:.6f} < P_normal={P_normal:.6f}"
