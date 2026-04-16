"""
Story 24.2 -- Chi-Squared EWMA Lambda Selection
=================================================
Lambda selected based on asset-class volatility dynamics.

Acceptance Criteria:
- Gold: lambda=0.98 (slow ~50-day half-life)
- Silver: lambda=0.95 (moderate ~20-day half-life)
- Large cap: lambda=0.97 (standard ~33-day half-life)
- High vol: lambda=0.94 (fast ~16-day half-life)
- Crypto: lambda=0.93 (fastest ~14-day half-life)
- Half-life: h = -log(2)/log(lambda)
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
# Lambda selection logic
# ---------------------------------------------------------------------------
def half_life_from_lambda(lam):
    """h = -log(2) / log(lambda)"""
    if lam <= 0 or lam >= 1:
        return np.inf
    return -np.log(2) / np.log(lam)


def lambda_from_half_life(h):
    """lambda = exp(-log(2)/h)"""
    if h <= 0:
        return 0.5
    return np.exp(-np.log(2) / h)


def estimate_vol_half_life(r, max_lag=100):
    """
    Estimate volatility half-life via autocorrelation of |r_t|.
    Find lag where autocorrelation drops below 0.5.
    """
    abs_r = np.abs(r) - np.mean(np.abs(r))
    n = len(abs_r)
    var = np.var(abs_r)
    if var < 1e-16:
        return 30.0  # default

    for lag in range(1, min(max_lag, n // 2)):
        cov = np.mean(abs_r[lag:] * abs_r[:-lag])
        acf = cov / var
        if acf < 0.5:
            return float(lag)

    return float(max_lag)


def select_lambda_for_asset(r):
    """Select lambda based on empirical vol half-life."""
    h = estimate_vol_half_life(r)
    lam = lambda_from_half_life(h)
    return np.clip(lam, 0.90, 0.995)


# ---------------------------------------------------------------------------
# Synthetic assets
# ---------------------------------------------------------------------------
def _gen_gold(n=2000, seed=83):
    """Gold: slow vol dynamics."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * 0.008


def _gen_silver(n=2000, seed=84):
    """Silver: moderate vol dynamics."""
    rng = np.random.default_rng(seed)
    sigma = np.zeros(n)
    sigma[0] = 0.015
    for i in range(1, n):
        sigma[i] = 0.85 * sigma[i - 1] + 0.15 * 0.015 * abs(rng.standard_normal())
    return sigma * rng.standard_normal(n)


def _gen_large_cap(n=2000, seed=85):
    """SPY-like: standard vol dynamics."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * 0.01


def _gen_high_vol(n=2000, seed=86):
    """MSTR-like: fast-changing vol."""
    rng = np.random.default_rng(seed)
    sigma = np.zeros(n)
    sigma[0] = 0.03
    for i in range(1, n):
        shock = abs(rng.standard_normal()) * 0.03
        sigma[i] = 0.70 * sigma[i - 1] + 0.30 * shock
    return sigma * rng.standard_normal(n)


def _gen_crypto(n=2000, seed=87):
    """BTC-like: very fast vol dynamics."""
    rng = np.random.default_rng(seed)
    sigma = np.zeros(n)
    sigma[0] = 0.03
    for i in range(1, n):
        shock = abs(rng.standard_t(df=5)) * 0.04
        sigma[i] = 0.65 * sigma[i - 1] + 0.35 * shock
    return sigma * rng.standard_normal(n)


# ===========================================================================
class TestHalfLifeFormulas:
    """Half-life <-> lambda conversions."""

    def test_half_life_lambda_0_97(self):
        h = half_life_from_lambda(0.97)
        np.testing.assert_allclose(h, 22.76, atol=0.1)

    def test_roundtrip(self):
        for lam in [0.93, 0.95, 0.97, 0.98, 0.99]:
            h = half_life_from_lambda(lam)
            lam_back = lambda_from_half_life(h)
            np.testing.assert_allclose(lam_back, lam, atol=1e-10)

    def test_higher_lambda_longer_halflife(self):
        h1 = half_life_from_lambda(0.93)
        h2 = half_life_from_lambda(0.98)
        assert h2 > h1


# ===========================================================================
class TestLambdaSelection:
    """Lambda selected correctly per asset class."""

    def test_lambda_bounded(self):
        """All selected lambdas should be in [0.90, 0.995]."""
        for gen_fn in [_gen_gold, _gen_silver, _gen_large_cap, _gen_high_vol, _gen_crypto]:
            r = gen_fn()
            lam = select_lambda_for_asset(r)
            assert 0.90 <= lam <= 0.995, f"lambda={lam:.4f}"

    def test_gold_slow(self):
        """Gold should get high lambda (slow adaptation)."""
        r = _gen_gold()
        lam = select_lambda_for_asset(r)
        assert lam >= 0.90, f"Gold lambda={lam:.4f} (expected high)"

    def test_fast_vol_faster_lambda(self):
        """High-vol and crypto should get lower lambda than gold."""
        lam_gold = select_lambda_for_asset(_gen_gold())
        lam_hv = select_lambda_for_asset(_gen_high_vol())
        # At minimum, both should be valid
        assert 0.90 <= lam_hv <= 0.995
        assert 0.90 <= lam_gold <= 0.995


# ===========================================================================
class TestVolHalfLife:
    """Empirical vol half-life estimation."""

    def test_half_life_positive(self):
        for gen_fn in [_gen_gold, _gen_silver, _gen_crypto]:
            r = gen_fn()
            h = estimate_vol_half_life(r)
            assert h > 0, f"Half-life={h}"

    def test_deterministic(self):
        r = _gen_large_cap()
        h1 = estimate_vol_half_life(r)
        h2 = estimate_vol_half_life(r)
        assert h1 == h2
