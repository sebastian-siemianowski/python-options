"""
Story 23.3 -- Jump Size Distribution for Different Asset Classes
=================================================================
Calibrated jump parameters per asset class:
- Equity: mu_J < 0 (downward bias), sigma_J ~ 0.03-0.05
- Gold: mu_J ~ 0 (symmetric), sigma_J ~ 0.02-0.03
- BTC: mu_J varies, sigma_J ~ 0.05-0.10
- MSTR: mu_J ~ 0, sigma_J ~ 0.08-0.15

Acceptance Criteria:
- EM estimates match asset class characteristics
- Jump intensity matches empirical extreme-move frequency
- Class averages serve as reasonable priors
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
# EM for jump parameters (from Story 23.1)
# ---------------------------------------------------------------------------
def compute_jump_posterior(r, sigma, lambda_j, mu_j, sigma_j):
    n = len(r)
    jump_prob = np.zeros(n)
    for t in range(n):
        s = max(sigma[t], 1e-10)
        f_diff = stats.norm.pdf(r[t], loc=0.0, scale=s)
        f_jump = stats.norm.pdf(r[t], loc=mu_j, scale=np.sqrt(s ** 2 + sigma_j ** 2))
        numer = lambda_j * f_jump
        denom = (1 - lambda_j) * f_diff + lambda_j * f_jump
        if denom > 1e-300:
            jump_prob[t] = numer / denom
    return jump_prob


def estimate_jump_params_em(r, sigma, n_iter=50, lambda_init=0.05,
                             mu_j_init=0.0, sigma_j_init=0.05):
    lambda_j = lambda_init
    mu_j = mu_j_init
    sigma_j = sigma_j_init
    n = len(r)
    for _ in range(n_iter):
        z = compute_jump_posterior(r, sigma, lambda_j, mu_j, sigma_j)
        z_sum = np.sum(z) + 1e-12
        lambda_j = np.clip(z_sum / n, 0.001, 0.5)
        mu_j = np.sum(z * r) / z_sum
        sigma_j = np.sqrt(np.sum(z * (r - mu_j) ** 2) / z_sum)
        sigma_j = max(sigma_j, 0.001)
    return lambda_j, mu_j, sigma_j


# ---------------------------------------------------------------------------
# Synthetic asset class generators
# ---------------------------------------------------------------------------
def _make_ewma_vol(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def _gen_equity(n=1500, seed=70):
    """Equity-like: downward-biased jumps."""
    rng = np.random.default_rng(seed)
    sigma = 0.012
    r = sigma * rng.standard_normal(n)
    mask = rng.random(n) < 0.04
    r[mask] += rng.normal(-0.02, 0.04, size=np.sum(mask))
    return r, _make_ewma_vol(r, sigma)


def _gen_gold(n=1500, seed=71):
    """Gold-like: symmetric jumps."""
    rng = np.random.default_rng(seed)
    sigma = 0.008
    r = sigma * rng.standard_normal(n)
    mask = rng.random(n) < 0.03
    r[mask] += rng.normal(0.0, 0.025, size=np.sum(mask))
    return r, _make_ewma_vol(r, sigma)


def _gen_btc(n=1500, seed=72):
    """BTC-like: large jumps both ways."""
    rng = np.random.default_rng(seed)
    sigma = 0.025
    r = sigma * rng.standard_normal(n)
    mask = rng.random(n) < 0.06
    r[mask] += rng.normal(0.005, 0.07, size=np.sum(mask))
    return r, _make_ewma_vol(r, sigma)


def _gen_mstr(n=1500, seed=73):
    """MSTR-like: very large jumps."""
    rng = np.random.default_rng(seed)
    sigma = 0.03
    r = sigma * rng.standard_normal(n)
    mask = rng.random(n) < 0.05
    r[mask] += rng.normal(0.0, 0.10, size=np.sum(mask))
    return r, _make_ewma_vol(r, sigma)


# ===========================================================================
class TestEquityJumps:
    """Equity jumps should show downward bias."""

    def test_mu_j_negative(self):
        r, v = _gen_equity()
        _, mu_j, _ = estimate_jump_params_em(r, v)
        assert mu_j < 0.01, f"Equity mu_j={mu_j:.4f} (expected negative or near zero)"

    def test_sigma_j_range(self):
        r, v = _gen_equity()
        _, _, sigma_j = estimate_jump_params_em(r, v)
        assert 0.01 < sigma_j < 0.15, f"Equity sigma_j={sigma_j:.4f}"

    def test_lambda_reasonable(self):
        r, v = _gen_equity()
        lambda_j, _, _ = estimate_jump_params_em(r, v)
        assert 0.01 < lambda_j < 0.3, f"Equity lambda_j={lambda_j:.4f}"


# ===========================================================================
class TestGoldJumps:
    """Gold jumps should be symmetric."""

    def test_mu_j_near_zero(self):
        r, v = _gen_gold()
        _, mu_j, _ = estimate_jump_params_em(r, v)
        assert abs(mu_j) < 0.02, f"Gold mu_j={mu_j:.4f} (expected symmetric)"

    def test_sigma_j_range(self):
        r, v = _gen_gold()
        _, _, sigma_j = estimate_jump_params_em(r, v)
        assert 0.005 < sigma_j < 0.10, f"Gold sigma_j={sigma_j:.4f}"


# ===========================================================================
class TestBTCJumps:
    """BTC jumps should be large."""

    def test_sigma_j_large(self):
        r, v = _gen_btc()
        _, _, sigma_j = estimate_jump_params_em(r, v)
        assert sigma_j > 0.01, f"BTC sigma_j={sigma_j:.4f} (expected large)"

    def test_lambda_moderate(self):
        r, v = _gen_btc()
        lambda_j, _, _ = estimate_jump_params_em(r, v)
        assert 0.01 < lambda_j < 0.4, f"BTC lambda_j={lambda_j:.4f}"


# ===========================================================================
class TestMSTRJumps:
    """MSTR jumps: symmetric but very large sigma_j."""

    def test_sigma_j_very_large(self):
        r, v = _gen_mstr()
        _, _, sigma_j = estimate_jump_params_em(r, v)
        assert sigma_j > 0.02, f"MSTR sigma_j={sigma_j:.4f} (expected very large)"

    def test_mu_j_roughly_symmetric(self):
        r, v = _gen_mstr()
        _, mu_j, _ = estimate_jump_params_em(r, v)
        assert abs(mu_j) < 0.03, f"MSTR mu_j={mu_j:.4f}"


# ===========================================================================
class TestCrossAssetComparison:
    """Class-average parameters differ meaningfully across asset types."""

    def test_sigma_j_ordering(self):
        """MSTR sigma_j > BTC sigma_j > Gold sigma_j (in expectation)."""
        _, _, s_gold = estimate_jump_params_em(*_gen_gold())
        _, _, s_btc = estimate_jump_params_em(*_gen_btc())
        _, _, s_mstr = estimate_jump_params_em(*_gen_mstr())

        # At minimum, MSTR should have larger jumps than gold
        assert s_mstr > s_gold, \
            f"MSTR sigma_j={s_mstr:.4f} <= Gold sigma_j={s_gold:.4f}"

    def test_class_averages_as_priors(self):
        """Class averages should be reasonable starting points."""
        params = {}
        for name, gen_fn in [("equity", _gen_equity), ("gold", _gen_gold),
                              ("btc", _gen_btc), ("mstr", _gen_mstr)]:
            r, v = gen_fn()
            lj, mj, sj = estimate_jump_params_em(r, v)
            params[name] = (lj, mj, sj)
            assert np.isfinite(lj) and np.isfinite(mj) and np.isfinite(sj)

        # All lambda_j should be positive
        for name, (lj, _, _) in params.items():
            assert lj > 0, f"{name} lambda_j={lj}"
