"""
Story 18.1 – MSTR Tail Distribution Validation
================================================
Validate heavy-tail modeling for extreme-vol assets like MSTR.
Uses synthetic data mimicking MSTR properties (BTC-correlated, heavy tails).
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

from models.numba_wrappers import run_phi_student_t_filter
from models.numba_kernels import crps_student_t_kernel


def _generate_mstr_like_returns(n=1500, seed=42):
    """
    Generate MSTR-like returns: heavy tails (nu~3.5), high vol, occasional 10%+ moves.
    """
    rng = np.random.default_rng(seed)
    nu_true = 3.5
    sigma_base = 0.045  # ~4.5% daily vol
    # Regime-switching vol: normal + stress
    vol = np.full(n, sigma_base)
    # Add stress periods (~10% of time with 2-3x vol)
    for i in range(n):
        if rng.random() < 0.10:
            vol[i] = sigma_base * rng.uniform(2.0, 3.0)

    innovations = rng.standard_t(nu_true, n) * vol
    returns = 0.001 + innovations
    return returns, vol, nu_true


class TestHeavyTailCapture:
    """Model captures heavy tails without numerical failure."""

    def test_low_nu_preferred(self):
        """For MSTR-like data, low nu should be preferred."""
        returns, vol, _ = _generate_mstr_like_returns()
        best_ll = -np.inf
        best_nu = None
        for nu in [3.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
            _, _, ll = run_phi_student_t_filter(
                returns, vol, phi=0.90, q=1e-4, c=1.0, nu=nu
            )
            if ll > best_ll:
                best_ll = ll
                best_nu = nu
        # Heavy-tailed data should prefer low nu
        assert best_nu <= 8.0, f"best_nu={best_nu}"

    def test_no_nan_inf_in_trajectory(self):
        """No NaN or Inf in Kalman trajectory during extreme data."""
        returns, vol, _ = _generate_mstr_like_returns()
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=3.5
        )
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.all(P > 0)
        assert np.isfinite(loglik)

    def test_handles_10pct_moves(self):
        """Filter doesn't crash on 10%+ daily moves."""
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.standard_t(3, n) * 0.04
        # Inject 10%+ moves
        returns[50] = 0.12
        returns[200] = -0.15
        returns[350] = 0.18
        vol = np.full(n, 0.04)
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=3.5
        )
        assert np.all(np.isfinite(mu))
        assert np.isfinite(loglik)


class TestVaRCoverage:
    """VaR backtest for extreme-vol assets."""

    def test_var_1pct_reasonable(self):
        """1% VaR violations should be near 1%."""
        returns, vol, _ = _generate_mstr_like_returns(n=2000)
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        S = np.sqrt(P + vol ** 2)
        # 1% VaR: mu + t_inv(0.01, nu) * sigma
        var_1pct = mu + stats.t.ppf(0.01, df=4.0) * S
        violations = np.sum(returns < var_1pct) / len(returns)
        # Should be near 1% (allow 0.5% - 3%)
        assert 0.005 <= violations <= 0.03, f"violations={violations:.3f}"


class TestCRPSForHighVol:
    """CRPS should be higher but still finite for high-vol assets."""

    def test_crps_finite(self):
        returns, vol, _ = _generate_mstr_like_returns()
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        S = np.sqrt(P + vol ** 2)
        z = (returns - mu) / np.maximum(S, 1e-10)
        crps = crps_student_t_kernel(z, S, 4.0)
        assert np.isfinite(crps)
        assert crps > 0

    def test_crps_higher_than_large_cap(self):
        """MSTR-like CRPS should be higher than MSFT-like CRPS."""
        rng = np.random.default_rng(42)
        # MSFT-like
        r_msft = rng.standard_t(12, 1000) * 0.015
        v_msft = np.full(1000, 0.015)
        mu_m, P_m, _ = run_phi_student_t_filter(
            r_msft, v_msft, phi=0.95, q=1e-5, c=1.0, nu=12.0
        )
        S_m = np.sqrt(P_m + v_msft ** 2)
        z_m = (r_msft - mu_m) / np.maximum(S_m, 1e-10)
        crps_msft = crps_student_t_kernel(z_m, S_m, 12.0)

        # MSTR-like
        r_mstr, v_mstr, _ = _generate_mstr_like_returns(n=1000)
        mu_x, P_x, _ = run_phi_student_t_filter(
            r_mstr, v_mstr, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        S_x = np.sqrt(P_x + v_mstr ** 2)
        z_x = (r_mstr - mu_x) / np.maximum(S_x, 1e-10)
        crps_mstr = crps_student_t_kernel(z_x, S_x, 4.0)

        # MSTR CRPS should be higher (harder to model)
        assert crps_mstr > crps_msft


class TestStressFiltering:
    """Filter survives stress periods (March 2020-like crash)."""

    def test_crash_then_recovery(self):
        """Filter handles crash followed by recovery."""
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.standard_t(4, n) * 0.04
        # Simulate crash: days 100-115 have extreme negative returns
        returns[100:115] = rng.standard_t(3, 15) * 0.10 - 0.05
        # Recovery: days 115-130
        returns[115:130] = rng.standard_t(3, 15) * 0.08 + 0.03
        vol = np.full(n, 0.04)
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        assert np.all(np.isfinite(mu))
        assert np.all(P > 0)
        assert np.isfinite(loglik)
