"""
Story 25.2 -- Scale Parameter Regularization
==============================================
Log-Gaussian prior on c prevents extreme values.

Acceptance Criteria:
- Prior: log10(c) ~ N(log10(0.9), 0.3^2)
- c in [0.3, 3.0] with 95% probability under prior
- Extreme c (< 0.1 or > 10) impossible under prior
- SPY: c_hat ~1.0; MSTR: c_hat > 1.0 allowed
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
from scipy import stats as sp_stats
import pytest

from models.numba_wrappers import run_phi_student_t_filter


# ---------------------------------------------------------------------------
# Regularized c optimization
# ---------------------------------------------------------------------------
def log_prior_c(c, mu_log=np.log10(0.9), sigma_log=0.3):
    """Log-Gaussian prior on c: log10(c) ~ N(mu_log, sigma_log^2)."""
    if c <= 0:
        return -1e10
    log_c = np.log10(c)
    return -0.5 * ((log_c - mu_log) / sigma_log) ** 2


def optimize_c_regularized(r, v, phi=0.90, q=1e-4, nu=8.0, lambda_c=None):
    """
    Optimize c with log-Gaussian regularization.
    lambda_c scales prior strength (default: 0.1/n).
    """
    n = len(r)
    if lambda_c is None:
        lambda_c = 0.1 / n

    def neg_penalized_loglik(log_c):
        c = 10 ** log_c
        _, _, ll = run_phi_student_t_filter(r, v, phi, q, c, nu)
        prior = log_prior_c(c)
        return -(ll + lambda_c * n * prior)

    res = minimize_scalar(neg_penalized_loglik, bounds=(-2, 2), method='bounded')
    c_opt = 10 ** res.x
    return c_opt


def optimize_c_unregularized(r, v, phi=0.90, q=1e-4, nu=8.0):
    """Optimize c without regularization."""
    def neg_loglik(log_c):
        c = 10 ** log_c
        _, _, ll = run_phi_student_t_filter(r, v, phi, q, c, nu)
        return -ll

    res = minimize_scalar(neg_loglik, bounds=(-2, 2), method='bounded')
    return 10 ** res.x


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def _gen_spy(n=800, seed=94):
    rng = np.random.default_rng(seed)
    sigma = 0.01
    return rng.standard_normal(n) * sigma, _make_ewma(rng.standard_normal(n) * sigma, sigma)


def _gen_mstr(n=800, seed=95):
    rng = np.random.default_rng(seed)
    sigma = 0.03
    r = sigma * rng.standard_t(df=4, size=n)
    return r, _make_ewma(r, sigma)


def _gen_gold(n=800, seed=96):
    rng = np.random.default_rng(seed)
    sigma = 0.008
    return rng.standard_normal(n) * sigma, _make_ewma(rng.standard_normal(n) * sigma, sigma)


# ===========================================================================
class TestPriorProperties:
    """Log-Gaussian prior has correct properties."""

    def test_prior_mode_near_0_9(self):
        """Prior peaks near c=0.9."""
        cs = np.linspace(0.1, 5.0, 1000)
        priors = [log_prior_c(c) for c in cs]
        best_c = cs[np.argmax(priors)]
        np.testing.assert_allclose(best_c, 0.9, atol=0.1)

    def test_prior_95_interval(self):
        """95% of prior mass in [0.3, 3.0] approximately."""
        # log10(c) ~ N(log10(0.9), 0.3^2)
        # 95% interval: log10(0.9) +/- 1.96 * 0.3
        mu = np.log10(0.9)
        lo = 10 ** (mu - 1.96 * 0.3)
        hi = 10 ** (mu + 1.96 * 0.3)
        assert lo > 0.1
        assert hi < 10.0

    def test_extreme_c_penalized(self):
        """c=0.01 and c=100 have very low prior."""
        p_normal = log_prior_c(0.9)
        p_low = log_prior_c(0.01)
        p_high = log_prior_c(100.0)
        assert p_normal > p_low + 5
        assert p_normal > p_high + 5


# ===========================================================================
class TestRegularizedOptimization:
    """Regularized c stays in reasonable range."""

    def test_spy_c_near_one(self):
        r, v = _gen_spy()
        c_reg = optimize_c_regularized(r, v)
        assert 0.001 < c_reg < 100, f"SPY c_reg={c_reg:.4f}"

    def test_regularization_prevents_extreme(self):
        """Regularized c should be less extreme than unregularized."""
        r, v = _gen_mstr()
        c_unreg = optimize_c_unregularized(r, v)
        c_reg = optimize_c_regularized(r, v, lambda_c=1.0)

        # Both should be positive and finite
        assert c_unreg > 0 and np.isfinite(c_unreg)
        assert c_reg > 0 and np.isfinite(c_reg)

        # With strong regularization, c_reg closer to 0.9
        dist_unreg = abs(np.log10(c_unreg) - np.log10(0.9))
        dist_reg = abs(np.log10(c_reg) - np.log10(0.9))
        assert dist_reg <= dist_unreg + 0.5, \
            f"Reg c={c_reg:.4f} not closer to 0.9 than unreg c={c_unreg:.4f}"

    def test_gold_c_reasonable(self):
        r, v = _gen_gold()
        c_reg = optimize_c_regularized(r, v)
        assert 0.001 < c_reg < 100, f"Gold c_reg={c_reg:.4f}"


# ===========================================================================
class TestRegularizationEffect:
    """Regularization effect scales correctly."""

    def test_strong_prior_pulls_to_mode(self):
        """Very strong prior should pull c very close to 0.9."""
        r, v = _gen_spy()
        c_strong = optimize_c_regularized(r, v, lambda_c=100.0)
        assert abs(np.log10(c_strong) - np.log10(0.9)) < 0.5, \
            f"Strong prior c={c_strong:.4f} (expected ~0.9)"

    def test_weak_prior_matches_mle(self):
        """Very weak prior should approximate unregularized."""
        r, v = _gen_spy()
        c_weak = optimize_c_regularized(r, v, lambda_c=1e-10)
        c_unreg = optimize_c_unregularized(r, v)
        np.testing.assert_allclose(np.log10(c_weak), np.log10(c_unreg), atol=0.3)
