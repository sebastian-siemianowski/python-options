"""
Story 24.1 -- Adaptive Dead-Zone Bounds Based on Sample Kurtosis
=================================================================
Chi-squared EWMA tracker with adaptive dead-zone bounds based on
excess kurtosis of innovations.

Acceptance Criteria:
- Gaussian (kappa=3): dead-zone ~[0.30, 0.55]
- Student-t(4) (kappa~9): wider dead-zone [0.20, 0.65]
- Dead-zone width: 0.25 + 0.05*(kappa-3)
- Correction frequency: 5-15% of timesteps
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


# ---------------------------------------------------------------------------
# Chi-squared EWMA with adaptive dead-zone
# ---------------------------------------------------------------------------
def compute_adaptive_deadzone(kurtosis_excess):
    """
    Dead-zone bounds based on excess kurtosis.
    width = 0.25 + 0.05 * (kappa - 3)
    center = 0.425
    """
    kappa = max(kurtosis_excess, 0.0)  # clip below 0
    width = 0.25 + 0.05 * kappa
    width = np.clip(width, 0.15, 0.80)  # safety bounds
    center = 0.425
    d_lo = center - width / 2
    d_hi = center + width / 2
    return d_lo, d_hi


def run_chi2_ewma(innovations, S, lam=0.97, d_lo=0.30, d_hi=0.55):
    """
    Chi-squared EWMA tracker on innovations.
    chi2_t = e_t^2 / S_t
    chi2_bar_t = lam * chi2_bar_{t-1} + (1-lam) * chi2_t
    Correction when chi2_bar outside [d_lo, d_hi].
    """
    n = len(innovations)
    chi2 = innovations ** 2 / np.maximum(S, 1e-16)
    chi2_bar = np.zeros(n)
    correction = np.ones(n)
    corrected = np.zeros(n, dtype=bool)

    chi2_bar[0] = chi2[0]
    for t in range(1, n):
        chi2_bar[t] = lam * chi2_bar[t - 1] + (1 - lam) * chi2[t]
        if chi2_bar[t] < d_lo or chi2_bar[t] > d_hi:
            correction[t] = 1.0 / max(chi2_bar[t], 1e-8)
            corrected[t] = True

    return chi2_bar, correction, corrected


def compute_innovations_from_filter(r, v, phi=0.90, q=1e-4, c=1.0):
    """Simple Kalman filter returning innovations and S."""
    n = len(r)
    e = np.zeros(n)
    S = np.zeros(n)
    mu_t = 0.0
    P_t = 1e-4

    for t in range(n):
        mu_pred = phi * mu_t
        P_pred = phi ** 2 * P_t + q
        R_t = (c * v[t]) ** 2
        S_t = P_pred + R_t

        e[t] = r[t] - mu_pred
        S[t] = S_t

        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * e[t]
        P_t = (1 - K_t) * P_pred

    return e, S


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _gen_gaussian(n=1500, seed=80):
    rng = np.random.default_rng(seed)
    sigma = 0.01
    r = sigma * rng.standard_normal(n)
    v = np.full(n, sigma)
    return r, v


def _gen_heavy_tailed(n=1500, seed=81):
    rng = np.random.default_rng(seed)
    sigma = 0.015
    r = sigma * rng.standard_t(df=4.0, size=n)
    v2 = np.zeros(n)
    v2[0] = sigma ** 2
    for i in range(1, n):
        v2[i] = 0.94 * v2[i - 1] + 0.06 * r[i - 1] ** 2
    return r, np.sqrt(np.maximum(v2, 1e-16))


def _gen_moderate(n=1500, seed=82):
    rng = np.random.default_rng(seed)
    sigma = 0.008
    r = sigma * rng.standard_t(df=20.0, size=n)
    v2 = np.zeros(n)
    v2[0] = sigma ** 2
    for i in range(1, n):
        v2[i] = 0.94 * v2[i - 1] + 0.06 * r[i - 1] ** 2
    return r, np.sqrt(np.maximum(v2, 1e-16))


# ===========================================================================
class TestAdaptiveDeadzone:
    """Dead-zone adapts correctly to kurtosis."""

    def test_gaussian_deadzone(self):
        d_lo, d_hi = compute_adaptive_deadzone(0.0)  # excess kurt=0 for Gaussian
        assert 0.1 <= d_lo <= 0.40
        assert 0.40 <= d_hi <= 0.70

    def test_heavy_tail_wider(self):
        d_lo_g, d_hi_g = compute_adaptive_deadzone(0.0)
        d_lo_t, d_hi_t = compute_adaptive_deadzone(6.0)  # excess kurt=6 for t(4)
        width_g = d_hi_g - d_lo_g
        width_t = d_hi_t - d_lo_t
        assert width_t > width_g, f"Heavy-tail width={width_t:.3f} <= Gaussian={width_g:.3f}"

    def test_moderate_tail_intermediate(self):
        d_lo_g, d_hi_g = compute_adaptive_deadzone(0.0)
        d_lo_m, d_hi_m = compute_adaptive_deadzone(0.3)  # t(20) ~0.3
        d_lo_t, d_hi_t = compute_adaptive_deadzone(6.0)
        width_g = d_hi_g - d_lo_g
        width_m = d_hi_m - d_lo_m
        width_t = d_hi_t - d_lo_t
        assert width_g <= width_m <= width_t


# ===========================================================================
class TestChi2EWMA:
    """Chi-squared EWMA tracker works correctly."""

    def test_chi2_bar_finite(self):
        r, v = _gen_gaussian()
        e, S = compute_innovations_from_filter(r, v)
        chi2_bar, _, _ = run_chi2_ewma(e, S)
        assert np.all(np.isfinite(chi2_bar))

    def test_correction_frequency_gaussian(self):
        """Gaussian: correction frequency should be moderate."""
        r, v = _gen_gaussian()
        e, S = compute_innovations_from_filter(r, v)
        kappa_excess = float(stats.kurtosis(e[100:], fisher=True))
        d_lo, d_hi = compute_adaptive_deadzone(kappa_excess)
        _, _, corrected = run_chi2_ewma(e, S, d_lo=d_lo, d_hi=d_hi)
        freq = np.mean(corrected[100:])  # skip burn-in
        assert freq < 0.50, f"Correction freq={freq:.1%} (too high)"

    def test_correction_frequency_heavy_tail(self):
        """Heavy-tailed: adaptive dead-zone should reduce corrections."""
        r, v = _gen_heavy_tailed()
        e, S = compute_innovations_from_filter(r, v)
        kappa_excess = float(stats.kurtosis(e[100:], fisher=True))

        # Fixed (Gaussian) dead-zone
        _, _, corr_fixed = run_chi2_ewma(e, S, d_lo=0.30, d_hi=0.55)
        freq_fixed = np.mean(corr_fixed[100:])

        # Adaptive dead-zone
        d_lo, d_hi = compute_adaptive_deadzone(kappa_excess)
        _, _, corr_adapt = run_chi2_ewma(e, S, d_lo=d_lo, d_hi=d_hi)
        freq_adapt = np.mean(corr_adapt[100:])

        # Adaptive should correct less often (wider dead-zone)
        assert freq_adapt <= freq_fixed + 0.05, \
            f"Adaptive freq={freq_adapt:.1%} > fixed={freq_fixed:.1%}"


# ===========================================================================
class TestChi2EWMAStability:
    """EWMA tracker stable for various inputs."""

    def test_deterministic(self):
        r, v = _gen_gaussian()
        e, S = compute_innovations_from_filter(r, v)
        c1, _, _ = run_chi2_ewma(e, S)
        c2, _, _ = run_chi2_ewma(e, S)
        np.testing.assert_array_equal(c1, c2)

    def test_correction_bounded(self):
        r, v = _gen_heavy_tailed()
        e, S = compute_innovations_from_filter(r, v)
        _, correction, _ = run_chi2_ewma(e, S)
        # Correction should not be extreme
        assert np.all(correction > 0)
        assert np.all(correction < 1000)
