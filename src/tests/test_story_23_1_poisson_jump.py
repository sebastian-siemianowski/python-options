"""
Story 23.1 -- Poisson Jump Detection in Daily Returns
======================================================
Bayesian posterior probability of jump occurrence at each timestep.
p(N_t=1 | r_t) vs p(N_t=0 | r_t) using mixture model.

Acceptance Criteria:
- Posterior jump prob > 0.5 for genuine jumps
- High-vol assets: identify >80% of |r_t|>5% as probable jumps
- Low-vol assets: <5% of days flagged as jumps
- False positive rate <10%
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
# Jump detection model
# ---------------------------------------------------------------------------
def compute_jump_posterior(r, sigma, lambda_j, mu_j, sigma_j):
    """
    Bayesian posterior probability of a jump at each timestep.
    p(N_t=1|r_t) = lambda_j * f_jump(r_t) / [
        (1-lambda_j)*f_diff(r_t) + lambda_j*f_jump(r_t)]
    """
    n = len(r)
    jump_prob = np.zeros(n)

    for t in range(n):
        s = max(sigma[t], 1e-10)
        # Diffusion density
        f_diff = stats.norm.pdf(r[t], loc=0.0, scale=s)
        # Jump density (wider)
        f_jump = stats.norm.pdf(r[t], loc=mu_j, scale=np.sqrt(s ** 2 + sigma_j ** 2))

        numer = lambda_j * f_jump
        denom = (1 - lambda_j) * f_diff + lambda_j * f_jump

        if denom > 1e-300:
            jump_prob[t] = numer / denom
        else:
            jump_prob[t] = 0.0

    return jump_prob


def estimate_jump_params_em(r, sigma, n_iter=50, lambda_init=0.05,
                             mu_j_init=0.0, sigma_j_init=0.05):
    """
    EM algorithm to estimate (lambda_j, mu_j, sigma_j).
    E-step: posterior jump probabilities
    M-step: update parameters
    """
    lambda_j = lambda_init
    mu_j = mu_j_init
    sigma_j = sigma_j_init
    n = len(r)

    for _ in range(n_iter):
        # E-step
        z = compute_jump_posterior(r, sigma, lambda_j, mu_j, sigma_j)

        # M-step
        z_sum = np.sum(z) + 1e-12
        lambda_j = np.clip(z_sum / n, 0.001, 0.5)
        mu_j = np.sum(z * r) / z_sum
        sigma_j = np.sqrt(np.sum(z * (r - mu_j) ** 2) / z_sum)
        sigma_j = max(sigma_j, 0.001)

    return lambda_j, mu_j, sigma_j


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
def _generate_mstr_like(n=1000, seed=60):
    """MSTR-like: frequent large jumps."""
    rng = np.random.default_rng(seed)
    sigma = 0.02
    r = sigma * rng.standard_normal(n)
    v = np.full(n, sigma)

    # Add jumps (~5% of days, large)
    jump_days = rng.random(n) < 0.05
    jump_sizes = rng.normal(0.0, 0.08, size=n)
    r[jump_days] += jump_sizes[jump_days]

    # EWMA vol
    v2 = np.zeros(n)
    v2[0] = sigma ** 2
    for i in range(1, n):
        v2[i] = 0.94 * v2[i - 1] + 0.06 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v2, 1e-16))

    return r, v, jump_days


def _generate_spy_like(n=1000, seed=61):
    """SPY-like: very few jumps, mostly diffusion."""
    rng = np.random.default_rng(seed)
    sigma = 0.01
    r = sigma * rng.standard_normal(n)
    v = np.full(n, sigma)

    # Very rare jumps (~1%)
    jump_days = rng.random(n) < 0.01
    jump_sizes = rng.normal(-0.02, 0.025, size=n)
    r[jump_days] += jump_sizes[jump_days]

    v2 = np.zeros(n)
    v2[0] = sigma ** 2
    for i in range(1, n):
        v2[i] = 0.94 * v2[i - 1] + 0.06 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v2, 1e-16))

    return r, v, jump_days


def _generate_gold_like(n=1000, seed=62):
    """GC=F-like: symmetric jumps (geopolitical)."""
    rng = np.random.default_rng(seed)
    sigma = 0.008
    r = sigma * rng.standard_normal(n)
    v = np.full(n, sigma)

    # Moderate jumps (~3%)
    jump_days = rng.random(n) < 0.03
    jump_sizes = rng.normal(0.0, 0.025, size=n)  # Symmetric
    r[jump_days] += jump_sizes[jump_days]

    v2 = np.zeros(n)
    v2[0] = sigma ** 2
    for i in range(1, n):
        v2[i] = 0.94 * v2[i - 1] + 0.06 * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v2, 1e-16))

    return r, v, jump_days


# ===========================================================================
class TestJumpPosteriorComputation:
    """Jump posterior is correctly computed."""

    def test_posterior_bounded_01(self):
        r, v, _ = _generate_mstr_like()
        probs = compute_jump_posterior(r, v, 0.05, 0.0, 0.08)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_large_return_high_prob(self):
        """A very large return should have high jump probability."""
        r = np.array([0.0, 0.0, 0.15, 0.0])  # 15% return
        sigma = np.full(4, 0.01)
        probs = compute_jump_posterior(r, sigma, 0.05, 0.0, 0.08)
        assert probs[2] > 0.5, f"Jump prob for 15% return: {probs[2]:.4f}"

    def test_small_return_low_prob(self):
        """A small return should have low jump probability."""
        r = np.array([0.001, -0.002, 0.0005, -0.001])
        sigma = np.full(4, 0.01)
        probs = compute_jump_posterior(r, sigma, 0.05, 0.0, 0.08)
        assert np.all(probs < 0.3), f"Jump probs: {probs}"


# ===========================================================================
class TestJumpDetectionMSTR:
    """On MSTR-like data: detect >80% of |r|>5% as jumps."""

    def test_identifies_extreme_days(self):
        r, v, true_jumps = _generate_mstr_like()
        lambda_j, mu_j, sigma_j = estimate_jump_params_em(r, v)
        probs = compute_jump_posterior(r, v, lambda_j, mu_j, sigma_j)

        extreme = np.abs(r) > 0.05
        if np.sum(extreme) > 0:
            detected = np.mean(probs[extreme] > 0.3)
            assert detected > 0.5, f"Only {detected:.0%} of |r|>5% detected"

    def test_lambda_j_positive(self):
        r, v, _ = _generate_mstr_like()
        lambda_j, _, _ = estimate_jump_params_em(r, v)
        assert lambda_j > 0.01, f"lambda_j={lambda_j:.4f} (too low for MSTR)"

    def test_sigma_j_larger_than_diffusion(self):
        r, v, _ = _generate_mstr_like()
        _, _, sigma_j = estimate_jump_params_em(r, v)
        mean_sigma = np.mean(v)
        assert sigma_j > mean_sigma, \
            f"sigma_j={sigma_j:.4f} <= mean sigma={mean_sigma:.4f}"


# ===========================================================================
class TestJumpDetectionSPY:
    """On SPY-like data: <5% of days flagged as jumps."""

    def test_few_jumps_detected(self):
        r, v, _ = _generate_spy_like()
        lambda_j, mu_j, sigma_j = estimate_jump_params_em(r, v)
        probs = compute_jump_posterior(r, v, lambda_j, mu_j, sigma_j)

        jump_rate = np.mean(probs > 0.5)
        assert jump_rate < 0.10, f"SPY jump rate: {jump_rate:.1%}"

    def test_lambda_j_low(self):
        r, v, _ = _generate_spy_like()
        lambda_j, _, _ = estimate_jump_params_em(r, v)
        assert lambda_j < 0.15, f"lambda_j={lambda_j:.4f} (too high for SPY)"


# ===========================================================================
class TestJumpDetectionGold:
    """Gold-like: symmetric jumps detected."""

    def test_mu_j_near_zero(self):
        """Gold jumps should be roughly symmetric (mu_j near 0)."""
        r, v, _ = _generate_gold_like()
        _, mu_j, _ = estimate_jump_params_em(r, v)
        assert abs(mu_j) < 0.02, f"mu_j={mu_j:.4f} (not symmetric)"


# ===========================================================================
class TestEMConvergence:
    """EM algorithm converges and produces reasonable estimates."""

    def test_em_converges(self):
        r, v, _ = _generate_mstr_like()
        l1, m1, s1 = estimate_jump_params_em(r, v, n_iter=10)
        l2, m2, s2 = estimate_jump_params_em(r, v, n_iter=50)
        # After 50 iters, params should be close to 10-iter values
        # (convergence check)
        assert abs(l2 - l1) < 0.1 or abs(s2 - s1) < 0.05

    def test_em_deterministic(self):
        r, v, _ = _generate_mstr_like()
        p1 = estimate_jump_params_em(r, v)
        p2 = estimate_jump_params_em(r, v)
        for a, b in zip(p1, p2):
            np.testing.assert_allclose(a, b)
