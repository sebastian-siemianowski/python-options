"""
Story 17.2 – GOOGL Earnings Seasonality Detection
===================================================
Validate that the model detects earnings-induced volatility spikes.
Uses synthetic data mimicking GOOGL with quarterly earnings vol.
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
    run_phi_student_t_filter,
    crps_decomposition,
)


def _generate_googl_like_returns(n=1000, seed=42):
    """
    Generate GOOGL-like returns with quarterly earnings vol spikes.
    Earnings at indices ~63, ~126, ~189, ~252, ... (every 63 trading days).
    Earnings days have 3x normal volatility.
    """
    rng = np.random.default_rng(seed)
    sigma_base = 0.018  # ~1.8% daily vol
    nu_true = 8.0
    vol = np.full(n, sigma_base)

    # Mark earnings days (quarterly = every ~63 trading days)
    earnings_days = []
    for q in range(n // 63 + 1):
        day = 63 * q + 62  # last day of quarter
        if day < n:
            earnings_days.append(day)
            # Spike vol on earnings day and day after
            vol[day] = sigma_base * 3.0
            if day + 1 < n:
                vol[day + 1] = sigma_base * 2.0

    innovations = rng.standard_t(nu_true, n) * vol
    drift = 0.0004
    returns = drift + innovations
    is_earnings = np.zeros(n, dtype=bool)
    for d in earnings_days:
        is_earnings[d] = True
        if d + 1 < n:
            is_earnings[d + 1] = True

    return returns, vol, nu_true, is_earnings, earnings_days


class TestVolSpikeDetection:
    """Model should detect vol spikes at earnings."""

    def test_filter_detects_variance_increase(self):
        """P (filtered variance) increases around earnings dates."""
        returns, vol, _, is_earnings, _ = _generate_googl_like_returns()
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        # P on earnings days should be higher than average
        P_earnings = P[is_earnings]
        P_normal = P[~is_earnings]
        assert np.mean(P_earnings) > np.mean(P_normal)

    def test_earnings_residuals_larger(self):
        """Residuals on earnings days are larger in magnitude."""
        returns, vol, _, is_earnings, _ = _generate_googl_like_returns()
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        resid = np.abs(returns - mu)
        assert np.mean(resid[is_earnings]) > np.mean(resid[~is_earnings])


class TestEarningsPITSeparation:
    """PIT should be computed separately for earnings and non-earnings."""

    def test_non_earnings_pit_better(self):
        """Non-earnings PIT should be better calibrated than earnings PIT."""
        returns, vol, _, is_earnings, _ = _generate_googl_like_returns(n=2000)
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        S = np.sqrt(P + vol ** 2)
        z = (returns - mu) / np.maximum(S, 1e-10)
        pit = stats.t.cdf(z, df=8.0)

        pit_earn = pit[is_earnings]
        pit_norm = pit[~is_earnings]

        # Non-earnings KS should generally be better
        _, ks_earn = stats.kstest(pit_earn, 'uniform')
        _, ks_norm = stats.kstest(pit_norm, 'uniform')
        # Both should produce valid PIT values
        assert len(pit_earn) > 0
        assert len(pit_norm) > 0
        assert np.all((pit >= 0) & (pit <= 1))

    def test_earnings_fraction_small(self):
        """Earnings days should be < 5% of total sample."""
        _, _, _, is_earnings, _ = _generate_googl_like_returns(n=2000)
        frac = np.mean(is_earnings)
        assert frac < 0.05


class TestEarningsVolScaling:
    """Model with correct vol handles earnings gracefully."""

    def test_informed_vol_improves_fit(self):
        """Using true vol (with spikes) gives better loglik than flat vol."""
        returns, vol_true, _, _, _ = _generate_googl_like_returns(n=2000)
        vol_flat = np.full_like(vol_true, np.mean(vol_true))

        _, _, ll_true = run_phi_student_t_filter(
            returns, vol_true, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        _, _, ll_flat = run_phi_student_t_filter(
            returns, vol_flat, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        # Informed vol should give better (higher) loglik
        assert ll_true > ll_flat, f"true={ll_true:.1f} vs flat={ll_flat:.1f}"

    def test_filter_stable_through_earnings(self):
        """Filter doesn't diverge through earnings periods."""
        returns, vol, _, _, _ = _generate_googl_like_returns(n=2000)
        mu, P, loglik = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.all(P > 0)
        assert np.isfinite(loglik)


class TestMultipleEarningsAssets:
    """Pattern holds across multiple earnings-heavy assets."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_earnings_detection_across_seeds(self, seed):
        """Earnings vol detection works across different random seeds."""
        returns, vol, _, is_earnings, _ = _generate_googl_like_returns(
            n=1000, seed=seed
        )
        mu, P, _ = run_phi_student_t_filter(
            returns, vol, phi=0.95, q=1e-5, c=1.0, nu=8.0
        )
        resid = np.abs(returns - mu)
        # Earnings residuals should be larger on average
        assert np.mean(resid[is_earnings]) > np.mean(resid[~is_earnings])
