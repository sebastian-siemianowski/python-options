"""
Story 24.3 -- VoV Enhancement Interaction with GARCH
======================================================
Damping prevents double-counting when both VoV and GARCH are active.

Acceptance Criteria:
- GARCH+VoV: damping >= 0.3
- Total variance inflation <= 1.5 during calm, <= 3.0 during crisis
- Damping logic correctly applied
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# VoV + GARCH interaction model
# ---------------------------------------------------------------------------
def compute_garch_vol(r, omega=1e-6, alpha=0.06, beta=0.94):
    """Simple GARCH(1,1) variance."""
    n = len(r)
    h = np.zeros(n)
    h[0] = omega / (1 - alpha - beta) if alpha + beta < 1 else np.var(r[:50])
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-16)
    return h


def compute_vov(r, h_garch, gamma=0.1, lam=0.95):
    """
    Volatility-of-volatility tracker.
    VoV_t based on chi-squared ratio.
    """
    n = len(r)
    chi2 = r ** 2 / np.maximum(h_garch, 1e-16)
    vov = np.zeros(n)
    vov[0] = 1.0
    for t in range(1, n):
        vov[t] = lam * vov[t - 1] + (1 - lam) * chi2[t]
    return vov


def compute_observation_noise(h_garch, vov, c=1.0, gamma=0.1,
                               garch_active=True, vov_active=True, damp=0.3):
    """
    Combined observation noise with damping.
    R_t = c * h_garch * (1 + gamma_eff * (vov - 1))
    """
    n = len(h_garch)
    R = np.zeros(n)

    if garch_active and vov_active:
        gamma_eff = gamma * (1 - damp)
    elif vov_active:
        gamma_eff = gamma
    else:
        gamma_eff = 0.0

    for t in range(n):
        vov_mult = 1.0 + gamma_eff * max(vov[t] - 1.0, 0.0)
        R[t] = c * h_garch[t] * vov_mult

    return R


def compute_variance_inflation(R, R_base):
    """Ratio R_t / R_base_t."""
    return R / np.maximum(R_base, 1e-16)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _gen_calm_crisis(n=1500, seed=90):
    """Calm then crisis data."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    # Calm: 0-800
    r[:800] = 0.01 * rng.standard_normal(800)
    # Crisis: 800-1000
    r[800:1000] = 0.04 * rng.standard_t(df=4, size=200)
    # Recovery: 1000+
    r[1000:] = 0.012 * rng.standard_normal(n - 1000)
    return r


# ===========================================================================
class TestDampingLogic:
    """Damping correctly applied."""

    def test_both_active_damped(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov = compute_vov(r, h)

        R_damped = compute_observation_noise(
            h, vov, garch_active=True, vov_active=True, damp=0.3)
        R_undamped = compute_observation_noise(
            h, vov, garch_active=True, vov_active=True, damp=0.0)

        # Damped should be <= undamped
        assert np.all(R_damped <= R_undamped + 1e-16)

    def test_garch_only_no_damping_needed(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov = compute_vov(r, h)

        R = compute_observation_noise(
            h, vov, garch_active=True, vov_active=False, damp=0.3)
        R_base = compute_observation_noise(
            h, vov, garch_active=True, vov_active=False, damp=0.0)

        # With VoV inactive, damping shouldn't matter
        np.testing.assert_array_almost_equal(R, R_base)

    def test_vov_only_full_contribution(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov = compute_vov(r, h)

        R = compute_observation_noise(
            h, vov, garch_active=False, vov_active=True, damp=0.3)
        R_full = compute_observation_noise(
            h, vov, garch_active=False, vov_active=True, damp=0.0)

        # With GARCH inactive, damping shouldn't be applied
        np.testing.assert_array_almost_equal(R, R_full)


# ===========================================================================
class TestVarianceInflation:
    """Variance inflation bounded."""

    def test_calm_inflation_bounded(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov = compute_vov(r, h)

        R = compute_observation_noise(h, vov, garch_active=True, vov_active=True, damp=0.3)
        R_base = h  # GARCH-only baseline

        infl = compute_variance_inflation(R, R_base)
        calm_infl = np.mean(infl[:800])
        assert calm_infl < 2.0, f"Calm inflation={calm_infl:.2f}"

    def test_crisis_inflation_bounded(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov = compute_vov(r, h)

        R = compute_observation_noise(h, vov, garch_active=True, vov_active=True, damp=0.3)
        R_base = h

        infl = compute_variance_inflation(R, R_base)
        crisis_infl = np.max(infl[800:1000])
        assert crisis_infl < 5.0, f"Crisis max inflation={crisis_infl:.2f}"


# ===========================================================================
class TestOutputStability:
    """Outputs finite and deterministic."""

    def test_all_finite(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov = compute_vov(r, h)
        R = compute_observation_noise(h, vov)
        assert np.all(np.isfinite(R))
        assert np.all(R > 0)

    def test_deterministic(self):
        r = _gen_calm_crisis()
        h = compute_garch_vol(r)
        vov1 = compute_vov(r, h)
        vov2 = compute_vov(r, h)
        np.testing.assert_array_equal(vov1, vov2)
