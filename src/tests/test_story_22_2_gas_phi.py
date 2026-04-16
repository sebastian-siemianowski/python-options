"""
Story 22.2 -- GAS-Phi (Score-Driven Drift Persistence)
========================================================
phi_t evolves via score-driven dynamics to adapt persistence:
  phi_{t+1} = omega_phi + alpha_phi * s_{phi,t} + beta_phi * phi_t

Acceptance Criteria:
- GAS-phi update rule implemented
- phi_t constrained to [-0.5, 1.05] via sigmoid
- Trending markets get higher phi, ranging markets lower
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


def _sigmoid_constrain(x, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_unconstrain(y, lo, hi):
    t = np.clip((y - lo) / (hi - lo), 1e-10, 1 - 1e-10)
    return np.log(t / (1 - t))


def run_gas_phi_filter(r, v, q, c, nu, omega_phi, alpha_phi, beta_phi, phi_init=0.90):
    """
    Run Student-t filter with GAS-phi dynamics.
    Score: s_{phi,t} = mu_{t-1} * (r_t - mu_pred) / S_t
    """
    n = len(r)
    phi_path = np.zeros(n)
    mu_path = np.zeros(n)
    P_path = np.zeros(n)
    loglik = 0.0

    phi_unc = _sigmoid_unconstrain(phi_init, -0.5, 1.05)
    phi_t = phi_init
    mu_t = 0.0
    P_t = 1e-4

    for t in range(n):
        mu_prev = mu_t

        # Predict
        mu_pred = phi_t * mu_t
        P_pred = phi_t ** 2 * P_t + q

        # Observation noise
        R_t = (c * v[t]) ** 2
        S_t = P_pred + R_t

        # Innovation
        innov = r[t] - mu_pred
        z = innov / np.sqrt(S_t)

        # Log-likelihood (Student-t)
        ll_t = stats.t.logpdf(z, df=nu) - 0.5 * np.log(S_t)
        loglik += ll_t

        # Kalman update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = (1 - K_t) * P_pred

        mu_path[t] = mu_t
        P_path[t] = P_t
        phi_path[t] = phi_t

        # GAS score for phi: mu_{t-1} * innovation / S_t
        score_phi = mu_prev * innov / S_t

        # Update in unconstrained space
        phi_unc = omega_phi + alpha_phi * score_phi + beta_phi * phi_unc
        phi_t = _sigmoid_constrain(phi_unc, -0.5, 1.05)

    return mu_path, P_path, loglik, phi_path


def _generate_trending_then_ranging(n=1500, seed=56):
    """Trending then ranging data to test phi adaptation."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    sigma = 0.012

    # Trending: 0-500 (drift = 0.002/day, phi should be high)
    for i in range(500):
        r[i] = 0.002 + sigma * rng.standard_t(df=8.0)

    # Ranging: 500-1000 (mean-reverting, drift = 0)
    r[500:1000] = sigma * rng.standard_t(df=8.0, size=500)

    # Trending again: 1000-1500 (negative drift)
    for i in range(1000, n):
        r[i] = -0.001 + sigma * rng.standard_t(df=8.0)

    v = np.zeros(n)
    v[0] = sigma ** 2
    for i in range(1, n):
        v[i] = 0.95 * v[i - 1] + 0.05 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))

    return r, v


# ===========================================================================
class TestGASPhiConstraints:
    """phi_t stays in [-0.5, 1.05]."""

    def test_sigmoid_bounds(self):
        for x in [-100, -10, 0, 10, 100]:
            y = _sigmoid_constrain(x, -0.5, 1.05)
            assert -0.5 <= y <= 1.05, f"x={x} -> y={y}"

    def test_sigmoid_invertible(self):
        for phi in [-0.3, 0.0, 0.5, 0.90, 1.0]:
            x = _sigmoid_unconstrain(phi, -0.5, 1.05)
            phi_back = _sigmoid_constrain(x, -0.5, 1.05)
            np.testing.assert_allclose(phi_back, phi, rtol=1e-6)

    def test_gas_phi_bounded(self):
        r, v = _generate_trending_then_ranging()
        _, _, _, phi_path = run_gas_phi_filter(
            r, v, q=1e-4, c=1.0, nu=8.0,
            omega_phi=0.0, alpha_phi=0.5, beta_phi=0.95, phi_init=0.90
        )
        assert np.all(phi_path >= -0.5)
        assert np.all(phi_path <= 1.05)


# ===========================================================================
class TestGASPhiDynamics:
    """phi_t adapts to market regime."""

    def test_phi_not_constant(self):
        r, v = _generate_trending_then_ranging()
        _, _, _, phi_path = run_gas_phi_filter(
            r, v, q=1e-4, c=1.0, nu=8.0,
            omega_phi=0.0, alpha_phi=1.0, beta_phi=0.90, phi_init=0.90
        )
        phi_std = np.std(phi_path[50:])
        assert phi_std > 0.001, f"phi_t std = {phi_std:.6f} (flat)"

    def test_phi_path_smooth(self):
        """phi_t should evolve smoothly, not jump erratically."""
        r, v = _generate_trending_then_ranging()
        _, _, _, phi_path = run_gas_phi_filter(
            r, v, q=1e-4, c=1.0, nu=8.0,
            omega_phi=0.0, alpha_phi=0.3, beta_phi=0.95, phi_init=0.90
        )
        # Max 1-step change should be reasonable
        max_change = np.max(np.abs(np.diff(phi_path[50:])))
        assert max_change < 1.0, f"Max phi change = {max_change:.4f}"


# ===========================================================================
class TestGASPhiFilterStability:
    """GAS-phi filter produces valid outputs."""

    def test_outputs_finite(self):
        r, v = _generate_trending_then_ranging()
        mu, P, loglik, phi_path = run_gas_phi_filter(
            r, v, q=1e-4, c=1.0, nu=8.0,
            omega_phi=0.0, alpha_phi=0.3, beta_phi=0.95, phi_init=0.90
        )
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)
        assert np.all(np.isfinite(phi_path))

    def test_deterministic(self):
        r, v = _generate_trending_then_ranging()
        params = dict(q=1e-4, c=1.0, nu=8.0,
                      omega_phi=0.0, alpha_phi=0.3, beta_phi=0.95, phi_init=0.90)
        _, _, ll1, phi1 = run_gas_phi_filter(r, v, **params)
        _, _, ll2, phi2 = run_gas_phi_filter(r, v, **params)
        assert ll1 == ll2
        np.testing.assert_array_equal(phi1, phi2)

    def test_loglik_finite_various_params(self):
        """Filter stable for various GAS parameter settings."""
        r, v = _generate_trending_then_ranging()
        configs = [
            (0.0, 0.1, 0.99),
            (0.0, 1.0, 0.80),
            (0.5, 0.2, 0.90),
            (-0.5, 0.5, 0.95),
        ]
        for omega, alpha, beta in configs:
            _, _, ll, phi = run_gas_phi_filter(
                r, v, q=1e-4, c=1.0, nu=8.0,
                omega_phi=omega, alpha_phi=alpha, beta_phi=beta
            )
            assert np.isfinite(ll), f"NaN loglik for omega={omega}, alpha={alpha}, beta={beta}"
            assert np.all(np.isfinite(phi))
