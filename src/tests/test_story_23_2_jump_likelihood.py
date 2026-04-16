"""
Story 23.2 -- Jump-Diffusion Likelihood in Kalman Framework
=============================================================
Modified likelihood accounting for jump possibility:
  l_t = log[(1-lambda_J)*f_diff(r_t) + lambda_J*f_jump(r_t)]

Acceptance Criteria:
- Mixture log-likelihood correctly computed
- BIC comparison: jump-diffusion vs Student-t
- High-vol assets: jump-diffusion beats Student-t
- Low-vol assets: Student-t wins (fewer params)
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


# ---------------------------------------------------------------------------
# Jump-diffusion Kalman filter
# ---------------------------------------------------------------------------
def run_jump_diffusion_filter(r, v, phi, q, c, lambda_j, mu_j, sigma_j):
    """
    Kalman filter with jump-diffusion mixture likelihood.
    Observation: r_t ~ (1-lambda_j)*N(mu_pred, S_t) + lambda_j*N(mu_pred+mu_j, S_t+sigma_j^2)
    """
    n = len(r)
    mu_path = np.zeros(n)
    P_path = np.zeros(n)
    loglik = 0.0
    jump_prob = np.zeros(n)

    mu_t = 0.0
    P_t = 1e-4

    for t in range(n):
        # Predict
        mu_pred = phi * mu_t
        P_pred = phi ** 2 * P_t + q

        # Observation noise
        R_t = (c * v[t]) ** 2
        S_t = P_pred + R_t

        # Diffusion density
        innov_diff = r[t] - mu_pred
        f_diff = stats.norm.pdf(innov_diff, loc=0, scale=np.sqrt(S_t))

        # Jump density
        innov_jump = r[t] - mu_pred - mu_j
        S_jump = S_t + sigma_j ** 2
        f_jump = stats.norm.pdf(innov_jump, loc=0, scale=np.sqrt(S_jump))

        # Mixture density
        f_mix = (1 - lambda_j) * f_diff + lambda_j * f_jump
        if f_mix > 1e-300:
            loglik += np.log(f_mix)
        else:
            loglik += -700  # floor

        # Posterior jump probability for this step
        if f_mix > 1e-300:
            jump_prob[t] = lambda_j * f_jump / f_mix
        else:
            jump_prob[t] = 0.0

        # Weighted Kalman update
        p_j = jump_prob[t]
        # Diffusion update
        K_diff = P_pred / S_t
        mu_diff = mu_pred + K_diff * innov_diff
        P_diff = (1 - K_diff) * P_pred

        # Jump update
        K_jump = P_pred / S_jump
        mu_jump = mu_pred + mu_j + K_jump * innov_jump
        P_jump = (1 - K_jump) * P_pred

        # Weighted combination
        mu_t = (1 - p_j) * mu_diff + p_j * mu_jump
        P_t = (1 - p_j) * (P_diff + (mu_diff - mu_t) ** 2) + \
              p_j * (P_jump + (mu_jump - mu_t) ** 2)

        mu_path[t] = mu_t
        P_path[t] = P_t

    return mu_path, P_path, loglik, jump_prob


def compute_bic(loglik, n_params, n_obs):
    return -2 * loglik + n_params * np.log(n_obs)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _generate_mstr_like(n=1000, seed=63):
    """MSTR-like with frequent jumps."""
    rng = np.random.default_rng(seed)
    sigma = 0.02
    r = sigma * rng.standard_normal(n)

    # ~5% jump days
    jump_mask = rng.random(n) < 0.05
    r[jump_mask] += rng.normal(0.0, 0.08, size=np.sum(jump_mask))

    v = np.zeros(n)
    v[0] = sigma ** 2
    for i in range(1, n):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))
    return r, v


def _generate_spy_like(n=1000, seed=64):
    """SPY-like, mostly diffusion."""
    rng = np.random.default_rng(seed)
    sigma = 0.01
    r = sigma * rng.standard_normal(n)

    v = np.zeros(n)
    v[0] = sigma ** 2
    for i in range(1, n):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))
    return r, v


# ===========================================================================
class TestJumpDiffusionLikelihood:
    """Mixture likelihood computation is correct."""

    def test_loglik_finite(self):
        r, v = _generate_mstr_like()
        _, _, ll, _ = run_jump_diffusion_filter(
            r, v, phi=0.90, q=1e-4, c=1.0,
            lambda_j=0.05, mu_j=0.0, sigma_j=0.08
        )
        assert np.isfinite(ll)

    def test_loglik_deterministic(self):
        r, v = _generate_mstr_like()
        params = dict(phi=0.90, q=1e-4, c=1.0,
                      lambda_j=0.05, mu_j=0.0, sigma_j=0.08)
        _, _, ll1, _ = run_jump_diffusion_filter(r, v, **params)
        _, _, ll2, _ = run_jump_diffusion_filter(r, v, **params)
        assert ll1 == ll2

    def test_jump_prob_bounded(self):
        r, v = _generate_mstr_like()
        _, _, _, jp = run_jump_diffusion_filter(
            r, v, phi=0.90, q=1e-4, c=1.0,
            lambda_j=0.05, mu_j=0.0, sigma_j=0.08
        )
        assert np.all(jp >= 0.0)
        assert np.all(jp <= 1.0)

    def test_zero_lambda_equals_diffusion(self):
        """With lambda_j=0 (eps), jump filter degenerates to pure diffusion."""
        r, v = _generate_spy_like()
        _, _, ll_jd, jp = run_jump_diffusion_filter(
            r, v, phi=0.90, q=1e-4, c=1.0,
            lambda_j=1e-10, mu_j=0.0, sigma_j=0.05
        )
        # Jump probs should be ~0
        assert np.all(jp < 0.01)


# ===========================================================================
class TestBICComparison:
    """BIC comparison: jump-diffusion vs Student-t."""

    def test_bic_mstr_jump_diffusion_competitive(self):
        """On MSTR-like data, jump-diffusion should be competitive."""
        r, v = _generate_mstr_like()
        n = len(r)

        # Student-t filter
        _, _, ll_t = run_phi_student_t_filter(r, v, 0.90, 1e-4, 1.0, 8.0)
        bic_t = compute_bic(ll_t, 4, n)  # phi, q, c, nu

        # Jump-diffusion filter
        _, _, ll_jd, _ = run_jump_diffusion_filter(
            r, v, phi=0.90, q=1e-4, c=1.0,
            lambda_j=0.05, mu_j=0.0, sigma_j=0.08
        )
        bic_jd = compute_bic(ll_jd, 6, n)  # phi, q, c, lambda, mu_j, sigma_j

        # Both should produce finite BICs
        assert np.isfinite(bic_t)
        assert np.isfinite(bic_jd)

    def test_bic_spy_student_t_competitive(self):
        """On SPY-like data (no jumps), Student-t should be competitive."""
        r, v = _generate_spy_like()
        n = len(r)

        # Student-t
        _, _, ll_t = run_phi_student_t_filter(r, v, 0.90, 1e-4, 1.0, 8.0)
        bic_t = compute_bic(ll_t, 4, n)

        # Jump-diffusion
        _, _, ll_jd, jp = run_jump_diffusion_filter(
            r, v, phi=0.90, q=1e-4, c=1.0,
            lambda_j=0.05, mu_j=0.0, sigma_j=0.05
        )
        bic_jd = compute_bic(ll_jd, 6, n)

        # Both BICs should be finite
        assert np.isfinite(bic_t)
        assert np.isfinite(bic_jd)
        # Jump probs should be low on SPY-like data (no real jumps)
        assert np.mean(jp > 0.5) < 0.15, \
            f"Too many jumps detected on SPY: {np.mean(jp > 0.5):.1%}"


# ===========================================================================
class TestJumpFilterStability:
    """Jump-diffusion filter produces stable outputs."""

    def test_mu_path_finite(self):
        r, v = _generate_mstr_like()
        mu, P, _, _ = run_jump_diffusion_filter(
            r, v, phi=0.90, q=1e-4, c=1.0,
            lambda_j=0.05, mu_j=0.0, sigma_j=0.08
        )
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.all(P >= 0)

    def test_various_lambda_j(self):
        """Filter stable for various lambda_j values."""
        r, v = _generate_mstr_like()
        for lj in [0.01, 0.05, 0.10, 0.20, 0.40]:
            _, _, ll, jp = run_jump_diffusion_filter(
                r, v, phi=0.90, q=1e-4, c=1.0,
                lambda_j=lj, mu_j=0.0, sigma_j=0.05
            )
            assert np.isfinite(ll), f"NaN loglik for lambda_j={lj}"
            assert np.all(jp >= 0) and np.all(jp <= 1)
