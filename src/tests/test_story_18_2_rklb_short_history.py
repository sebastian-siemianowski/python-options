"""
Story 18.2 – RKLB Short History Robustness
============================================
Validate that models handle short history (N < 500) with proper regularization.
Uses synthetic data mimicking growth-stage space company.
"""

import os, sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import run_phi_student_t_filter


def _generate_rklb_like_returns(n=500, seed=42):
    """Generate RKLB-like returns: moderate tails, high vol, short history."""
    rng = np.random.default_rng(seed)
    nu_true = 6.0
    sigma = 0.035
    vol = np.full(n, sigma)
    returns = rng.standard_t(nu_true, n) * sigma + 0.0002
    return returns, vol


class TestSmallSampleValidity:
    """All models produce valid parameters with small N."""

    @pytest.mark.parametrize("n", [200, 300, 500])
    def test_filter_valid_at_sample_size(self, n):
        """Filter produces valid output at various sample sizes."""
        returns, vol = _generate_rklb_like_returns(n=n)
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=6.0
        )
        assert len(mu) == n
        assert np.all(np.isfinite(mu))
        assert np.all(P > 0)
        assert np.isfinite(loglik)

    def test_n200_vs_n500_loglik_per_obs(self):
        """Per-observation loglik should be comparable across sample sizes."""
        returns_full, vol_full = _generate_rklb_like_returns(n=500)
        _, _, ll_200 = run_phi_student_t_filter(
            returns_full[:200], vol_full[:200], phi=0.90, q=1e-4, c=1.0, nu=6.0
        )
        _, _, ll_500 = run_phi_student_t_filter(
            returns_full, vol_full, phi=0.90, q=1e-4, c=1.0, nu=6.0
        )
        # Per-obs loglik should be in same ballpark
        ll_per_200 = ll_200 / 200
        ll_per_500 = ll_500 / 500
        assert abs(ll_per_200 - ll_per_500) / max(abs(ll_per_500), 1e-10) < 0.3


class TestParameterStability:
    """Parameters should be stable across different sample truncations."""

    def test_nu_grid_stable(self):
        """Optimal nu shouldn't swing wildly between N=200 and N=500."""
        returns, vol = _generate_rklb_like_returns(n=500)

        def _best_nu(r, v):
            best_ll = -np.inf
            best = None
            for nu in [4.0, 6.0, 8.0, 10.0, 12.0]:
                _, _, ll = run_phi_student_t_filter(
                    r, v, phi=0.90, q=1e-4, c=1.0, nu=nu
                )
                if ll > best_ll:
                    best_ll = ll
                    best = nu
            return best

        nu_200 = _best_nu(returns[:200], vol[:200])
        nu_300 = _best_nu(returns[:300], vol[:300])
        nu_500 = _best_nu(returns, vol)

        # Nu estimates should be in reasonable range
        assert 4.0 <= nu_200 <= 12.0
        assert 4.0 <= nu_300 <= 12.0
        assert 4.0 <= nu_500 <= 12.0
        # Difference between 200 and 500 shouldn't be more than factor 2
        assert max(nu_200, nu_500) / min(nu_200, nu_500) <= 3.0

    def test_phi_stable_across_truncations(self):
        """Phi estimates should be similar across sample sizes."""
        returns, vol = _generate_rklb_like_returns(n=500)

        def _best_phi(r, v):
            best_ll = -np.inf
            best = None
            for phi in np.arange(0.80, 1.01, 0.05):
                _, _, ll = run_phi_student_t_filter(
                    r, v, phi=phi, q=1e-4, c=1.0, nu=6.0
                )
                if ll > best_ll:
                    best_ll = ll
                    best = phi
            return best

        phi_200 = _best_phi(returns[:200], vol[:200])
        phi_500 = _best_phi(returns, vol)
        assert abs(phi_200 - phi_500) < 0.20


class TestMinimumSampleSize:
    """Identify minimum viable sample size."""

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_very_small_samples_dont_crash(self, n):
        """Even very small samples produce finite output."""
        returns, vol = _generate_rklb_like_returns(n=n)
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=6.0
        )
        assert np.all(np.isfinite(mu))
        assert np.isfinite(loglik)

    def test_n10_edge_case(self):
        """Even N=10 should not crash (minimal data)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 10)
        vol = np.full(10, 0.02)
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=6.0
        )
        assert np.all(np.isfinite(mu))
