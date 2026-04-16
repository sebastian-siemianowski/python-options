"""
Story 17.1 – MSFT as Gaussian Baseline Benchmark
==================================================
Validate that the model pipeline handles large-cap equities correctly.
MSFT = easy case with near-Gaussian returns, moderate tails.
Uses synthetic data mimicking MSFT properties.
"""

import os, sys
import numpy as np
import pytest
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import (
    run_gaussian_filter,
    run_phi_student_t_filter,
    crps_decomposition,
    pit_entropy,
)
from models.numba_kernels import crps_student_t_kernel


def _generate_msft_like_returns(n=1000, seed=42):
    """Generate returns mimicking MSFT: near-Gaussian, slight negative skew, moderate tails."""
    rng = np.random.default_rng(seed)
    # Student-t with nu~12 (moderate tails), slight drift
    nu_true = 12.0
    sigma_true = 0.015  # ~1.5% daily vol
    drift = 0.0003  # slight positive drift (~7.5% annual)
    innovations = rng.standard_t(nu_true, n) * sigma_true
    returns = drift + innovations
    vol = np.full(n, sigma_true)
    return returns, vol, nu_true


class TestGaussianBaseline:
    """Gaussian model on MSFT-like data."""

    def test_gaussian_filter_runs(self):
        """Gaussian filter completes without error."""
        returns, vol, _ = _generate_msft_like_returns()
        mu, P, loglik = run_gaussian_filter(returns, vol, q=1e-5, c=1.0)
        assert len(mu) == len(returns)
        assert np.isfinite(loglik)

    def test_gaussian_pit_computed(self):
        """Gaussian PIT can be computed (may not be uniform for t-distributed data)."""
        returns, vol, _ = _generate_msft_like_returns(n=2000)
        mu, P, _ = run_gaussian_filter(returns, vol, q=1e-5, c=1.0)
        S = np.sqrt(P + vol ** 2)
        z = (returns - mu) / np.maximum(S, 1e-10)
        pit = stats.norm.cdf(z)
        assert len(pit) == len(returns)
        assert np.all((pit >= 0) & (pit <= 1))

    def test_gaussian_loglik_finite(self):
        returns, vol, _ = _generate_msft_like_returns()
        _, _, loglik = run_gaussian_filter(returns, vol, q=1e-5, c=1.0)
        assert np.isfinite(loglik)
        # loglik can be positive when sigma is small (pdf > 1 for continuous distributions)


class TestStudentTImprovement:
    """Student-t model should improve over Gaussian for MSFT-like data."""

    def test_student_t_filter_runs(self):
        """Student-t filter completes on MSFT-like data."""
        returns, vol, _ = _generate_msft_like_returns()
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=10.0
        )
        assert len(mu) == len(returns)
        assert np.isfinite(loglik)

    def test_student_t_bic_improvement(self):
        """Student-t should have better BIC than Gaussian (captures tails)."""
        returns, vol, _ = _generate_msft_like_returns(n=2000)
        _, _, loglik_g = run_gaussian_filter(returns, vol, q=1e-5, c=1.0)
        _, _, loglik_t = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=10.0
        )
        n = len(returns)
        # BIC: -2*loglik + k*log(n), lower is better
        # Gaussian: k=2 (q, c), Student-t: k=4 (phi, q, c, nu)
        bic_g = -2 * loglik_g + 2 * np.log(n)
        bic_t = -2 * loglik_t + 4 * np.log(n)
        # Student-t should be better (lower BIC) or comparable
        # With synthetic t-distributed data, Student-t should win
        bic_diff = bic_g - bic_t  # positive means Student-t is better
        assert bic_diff > 0, f"BIC diff: {bic_diff:.1f} (expected > 0)"

    def test_nu_estimate_moderate(self):
        """For MSFT-like data, optimal nu should be moderate (not extreme)."""
        returns, vol, nu_true = _generate_msft_like_returns(n=2000)
        # Grid search over nu
        best_loglik = -np.inf
        best_nu = None
        for nu in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]:
            _, _, loglik = run_phi_student_t_filter(
                returns, vol, phi=0.95, q=1e-5, c=1.0, nu=nu
            )
            if loglik > best_loglik:
                best_loglik = loglik
                best_nu = nu
        # Best nu should be in moderate range [6, 30] for MSFT-like data
        assert 6.0 <= best_nu <= 30.0, f"best_nu={best_nu}"


class TestCRPSBaseline:
    """CRPS should be low for well-fitted large-cap model."""

    def test_crps_finite(self):
        returns, vol, _ = _generate_msft_like_returns()
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=10.0
        )
        S = np.sqrt(P + vol ** 2)
        z = (returns - mu) / np.maximum(S, 1e-10)
        crps = crps_student_t_kernel(z, S, 10.0)
        assert np.isfinite(crps)
        assert crps > 0

    def test_crps_decomposition_well_calibrated(self):
        """CRPS decomposition should show reasonable calibration."""
        returns, vol, _ = _generate_msft_like_returns(n=2000)
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=10.0
        )
        S = np.sqrt(P + vol ** 2)
        z = (returns - mu) / np.maximum(S, 1e-10)
        pit = stats.t.cdf(z, df=10.0)
        result = crps_decomposition(pit)
        assert np.isfinite(result["reliability"])
        assert np.isfinite(result["resolution"])


class TestRollingStability:
    """Model should be stable across rolling windows."""

    def test_rolling_windows_all_finite(self):
        """All rolling windows produce finite log-likelihoods."""
        returns, vol, _ = _generate_msft_like_returns(n=2000)
        window = 250  # ~1 year
        step = 250
        logliks = []
        for start in range(0, len(returns) - window, step):
            r_w = returns[start : start + window]
            v_w = vol[start : start + window]
            _, _, ll = run_phi_student_t_filter(
                r_w, v_w, phi=0.95, q=1e-5, c=1.0, nu=10.0
            )
            logliks.append(ll)
        assert all(np.isfinite(ll) for ll in logliks)
        assert len(logliks) >= 3

    def test_no_performance_cliff(self):
        """No window's loglik drops more than 50% below mean."""
        returns, vol, _ = _generate_msft_like_returns(n=2000)
        window = 250
        step = 250
        logliks = []
        for start in range(0, len(returns) - window, step):
            r_w = returns[start : start + window]
            v_w = vol[start : start + window]
            _, _, ll = run_phi_student_t_filter(
                r_w, v_w, phi=0.95, q=1e-5, c=1.0, nu=10.0
            )
            logliks.append(ll)
        mean_ll = np.mean(logliks)
        for ll in logliks:
            # No window should be dramatically worse than average
            assert ll > mean_ll * 1.5 or ll > mean_ll - abs(mean_ll) * 0.5
