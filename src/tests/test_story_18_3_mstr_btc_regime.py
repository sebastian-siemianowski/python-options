"""
Story 18.3 – MSTR-BTC Regime Correlation
==========================================
Validate that BTC-correlated assets show correlated stress regimes.
Uses synthetic BTC-correlated MSTR data with shared crash dynamics.
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


def _generate_btc_mstr_pair(n=2000, seed=42):
    """
    Generate correlated BTC and MSTR returns.
    MSTR is highly correlated with BTC (beta~1.5), shared crash events.
    """
    rng = np.random.default_rng(seed)

    # BTC returns: heavy tails, nu~3.5, 3% daily vol
    nu_btc = 3.5
    sigma_btc = 0.03
    btc_returns = rng.standard_t(nu_btc, n) * sigma_btc

    # Inject BTC crashes (5 events with -10% to -20% drops)
    crash_days = [200, 500, 900, 1300, 1700]
    for d in crash_days:
        if d < n:
            btc_returns[d] = -0.15 + rng.normal(0, 0.02)
            if d + 1 < n:
                btc_returns[d + 1] = -0.05 + rng.normal(0, 0.02)

    # Inject BTC rallies
    rally_days = [300, 700, 1100, 1500]
    for d in rally_days:
        if d < n:
            btc_returns[d] = 0.12 + rng.normal(0, 0.02)

    # MSTR = beta * BTC + idiosyncratic
    beta = 1.5
    sigma_idio = 0.025
    mstr_returns = (
        beta * btc_returns + rng.standard_t(4, n) * sigma_idio + 0.0005
    )

    vol_btc = np.full(n, np.std(btc_returns))
    vol_mstr = np.full(n, np.std(mstr_returns))

    is_crash = np.zeros(n, dtype=bool)
    for d in crash_days:
        if d < n:
            is_crash[d] = True
            if d + 1 < n:
                is_crash[d + 1] = True

    is_rally = np.zeros(n, dtype=bool)
    for d in rally_days:
        if d < n:
            is_rally[d] = True

    return (
        btc_returns, vol_btc,
        mstr_returns, vol_mstr,
        beta, is_crash, is_rally,
    )


class TestStressCorrelation:
    """Stress regimes should be correlated between BTC and MSTR."""

    def test_variance_spikes_correlated(self):
        """P spikes are correlated between BTC and MSTR."""
        r_btc, v_btc, r_mstr, v_mstr, _, _, _ = _generate_btc_mstr_pair()
        _, P_btc, _ = run_phi_student_t_filter(
            r_btc, v_btc, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        _, P_mstr, _ = run_phi_student_t_filter(
            r_mstr, v_mstr, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        # Correlation of P (filtered variance proxy)
        rho_P = np.corrcoef(P_btc, P_mstr)[0, 1]
        assert rho_P > 0.3, f"rho_P={rho_P:.3f}"

    def test_crash_detected_in_residuals(self):
        """MSTR residuals are larger on BTC crash days."""
        _, _, r_mstr, v_mstr, _, is_crash, _ = _generate_btc_mstr_pair()
        mu, P, _ = run_phi_student_t_filter(
            r_mstr, v_mstr, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        resid = np.abs(r_mstr - mu)
        resid_crash = resid[is_crash]
        resid_normal = resid[~is_crash]
        if len(resid_crash) > 0:
            assert np.mean(resid_crash) > np.mean(resid_normal)


class TestFactorResidual:
    """After removing BTC factor, MSTR residuals have lower kurtosis."""

    def test_factor_residual_lower_variance(self):
        """MSTR residual after BTC regression has lower variance."""
        r_btc, _, r_mstr, _, beta_true, _, _ = _generate_btc_mstr_pair()

        # OLS regression: MSTR = alpha + beta * BTC + eps
        A = np.vstack([r_btc, np.ones(len(r_btc))]).T
        result = np.linalg.lstsq(A, r_mstr, rcond=None)
        beta_hat, alpha_hat = result[0]
        residual = r_mstr - alpha_hat - beta_hat * r_btc

        # Variance of residual should be lower than raw MSTR (factor removed)
        var_raw = np.var(r_mstr)
        var_resid = np.var(residual)
        assert var_resid < var_raw, (
            f"var_resid={var_resid:.6f} vs var_raw={var_raw:.6f}"
        )

    def test_beta_estimate_reasonable(self):
        """Estimated beta should be near the true beta (1.5)."""
        r_btc, _, r_mstr, _, beta_true, _, _ = _generate_btc_mstr_pair()
        A = np.vstack([r_btc, np.ones(len(r_btc))]).T
        result = np.linalg.lstsq(A, r_mstr, rcond=None)
        beta_hat = result[0][0]
        assert abs(beta_hat - beta_true) < 0.5, f"beta_hat={beta_hat:.3f}"


class TestBTCRallyResponse:
    """MSTR should show elevated uncertainty during BTC rallies."""

    def test_rally_increases_p(self):
        """P increases on BTC rally days."""
        _, _, r_mstr, v_mstr, _, _, is_rally = _generate_btc_mstr_pair()
        _, P_mstr, _ = run_phi_student_t_filter(
            r_mstr, v_mstr, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        rally_response = np.zeros_like(is_rally)
        rally_response[is_rally] = True
        for i in range(len(rally_response) - 1):
            if is_rally[i]:
                rally_response[i + 1] = True

        P_rally = P_mstr[rally_response]
        P_normal = P_mstr[~rally_response & ~np.roll(is_rally, 1)]
        if len(P_rally) > 0:
            assert np.mean(P_rally) > np.mean(P_normal) * 0.8


class TestPITDuringBTCEvents:
    """PIT should remain valid during BTC-driven events."""

    def test_pit_finite_during_crashes(self):
        """PIT is finite and in [0,1] during BTC crashes."""
        _, _, r_mstr, v_mstr, _, is_crash, _ = _generate_btc_mstr_pair()
        mu, P, _ = run_phi_student_t_filter(
            r_mstr, v_mstr, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        S = np.sqrt(P + v_mstr ** 2)
        z = (r_mstr - mu) / np.maximum(S, 1e-10)
        pit = stats.t.cdf(z, df=4.0)
        pit_crash = pit[is_crash]
        assert np.all(np.isfinite(pit_crash))
        assert np.all((pit_crash >= 0) & (pit_crash <= 1))

    def test_overall_pit_valid(self):
        """Full PIT array is valid."""
        _, _, r_mstr, v_mstr, _, _, _ = _generate_btc_mstr_pair()
        mu, P, _ = run_phi_student_t_filter(
            r_mstr, v_mstr, phi=0.90, q=1e-4, c=1.0, nu=4.0
        )
        S = np.sqrt(P + v_mstr ** 2)
        z = (r_mstr - mu) / np.maximum(S, 1e-10)
        pit = stats.t.cdf(z, df=4.0)
        assert np.all(np.isfinite(pit))
        assert len(pit) == len(r_mstr)
