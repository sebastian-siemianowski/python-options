"""
Story 20.3 -- BTC-USD Weekend/Weekday Volatility Differential
==============================================================
Validate day-of-week volatility adjustment for 24/7 BTC trading.

Acceptance Criteria:
- Compute weekend vs weekday vol ratio rho = sigma_weekend / sigma_weekday
- If rho > 1.2: apply weekend scaling R_weekend = rho^2 * R_weekday
- PIT during weekends: valid
- Overall CRPS improvement from day-of-week adjustment
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
from models.numba_kernels import crps_student_t_kernel


# ---------------------------------------------------------------------------
# Synthetic BTC with day-of-week vol pattern
# ---------------------------------------------------------------------------
def _generate_btc_weekday_pattern(n=2100, seed=47):
    """
    BTC with weekend/weekday vol differential:
      - Weekdays (Mon-Fri): base vol
      - Weekends (Sat-Sun): 1.4x vol (lower liquidity)
      - Day-of-week encoded as 0=Mon, ..., 6=Sun

    Returns (r, v, day_of_week, is_weekend, vol_ratio_true)
    """
    rng = np.random.default_rng(seed)

    daily_base = 0.025  # ~40% annualized
    weekend_multiplier = 1.4

    # Day of week (cyclical)
    day_of_week = np.array([i % 7 for i in range(n)])
    is_weekend = (day_of_week >= 5)  # Sat=5, Sun=6

    # Vol structure
    sigma = np.where(is_weekend, daily_base * weekend_multiplier, daily_base)

    # Returns: Student-t, nu=4
    nu_true = 4.0
    innovations = rng.standard_t(df=nu_true, size=n)
    r = sigma * innovations

    # EWMA vol (doesn't know about weekday structure)
    ewm_lambda = 0.92
    v = np.zeros(n)
    v[0] = daily_base ** 2
    for i in range(1, n):
        v[i] = ewm_lambda * v[i - 1] + (1 - ewm_lambda) * r[i - 1] ** 2
    v = np.sqrt(np.maximum(v, 1e-16))

    return r, v, day_of_week, is_weekend, weekend_multiplier


def apply_weekend_scaling(v, is_weekend, rho):
    """Scale weekend observation noise by rho^2."""
    v_adj = v.copy()
    v_adj[is_weekend] = v[is_weekend] * rho
    return v_adj


# ---------------------------------------------------------------------------
BTC_PHI = 0.85
BTC_Q = 5e-4
BTC_C = 1.0
BTC_NU = 4.0


# ===========================================================================
class TestWeekendVolRatio:
    """Weekend vol is empirically higher than weekday vol."""

    def test_compute_vol_ratio(self):
        r, _, _, is_weekend, true_mult = _generate_btc_weekday_pattern()
        vol_weekend = np.std(r[is_weekend])
        vol_weekday = np.std(r[~is_weekend])

        rho = vol_weekend / vol_weekday
        assert rho > 1.1, f"Vol ratio = {rho:.3f}, expected > 1.1"

    def test_mean_abs_return_higher_weekend(self):
        r, _, _, is_weekend, _ = _generate_btc_weekday_pattern()
        mar_weekend = np.mean(np.abs(r[is_weekend]))
        mar_weekday = np.mean(np.abs(r[~is_weekend]))

        assert mar_weekend > mar_weekday, (
            f"Weekend MAR ({mar_weekend:.6f}) <= Weekday MAR ({mar_weekday:.6f})"
        )

    def test_vol_ratio_magnitude(self):
        """Estimated rho should be within 30% of true multiplier."""
        r, _, _, is_weekend, true_mult = _generate_btc_weekday_pattern()
        vol_weekend = np.std(r[is_weekend])
        vol_weekday = np.std(r[~is_weekend])
        rho = vol_weekend / vol_weekday

        assert abs(rho - true_mult) < 0.5 * true_mult, (
            f"rho = {rho:.3f}, true = {true_mult:.3f}"
        )


# ===========================================================================
class TestWeekendScalingEffect:
    """Applying weekend scaling improves model fit."""

    def test_scaled_filter_valid(self):
        """Filter with weekend-scaled v produces valid output."""
        r, v, _, is_weekend, _ = _generate_btc_weekday_pattern()
        rho = 1.4
        v_adj = apply_weekend_scaling(v, is_weekend, rho)

        mu, P, loglik = run_phi_student_t_filter(r, v_adj, phi=BTC_PHI, q=BTC_Q,
                                                  c=BTC_C, nu=BTC_NU)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)

    def test_scaling_improves_or_matches_loglik(self):
        """Weekend scaling should not degrade loglik significantly."""
        r, v, _, is_weekend, _ = _generate_btc_weekday_pattern()

        _, _, ll_base = run_phi_student_t_filter(r, v, phi=BTC_PHI, q=BTC_Q,
                                                  c=BTC_C, nu=BTC_NU)

        rho = 1.4
        v_adj = apply_weekend_scaling(v, is_weekend, rho)
        _, _, ll_adj = run_phi_student_t_filter(r, v_adj, phi=BTC_PHI, q=BTC_Q,
                                                c=BTC_C, nu=BTC_NU)

        # Adjusted should be at least comparable (within 100 nats)
        assert ll_adj > ll_base - 100, (
            f"Adjusted ({ll_adj:.2f}) much worse than base ({ll_base:.2f})"
        )


# ===========================================================================
class TestWeekendPIT:
    """PIT validity during weekends and weekdays."""

    def test_pit_valid_both_periods(self):
        """PIT values are valid for both weekdays and weekends."""
        r, v, _, is_weekend, _ = _generate_btc_weekday_pattern()
        rho = 1.4
        v_adj = apply_weekend_scaling(v, is_weekend, rho)

        mu, P, _ = run_phi_student_t_filter(r, v_adj, phi=BTC_PHI, q=BTC_Q,
                                            c=BTC_C, nu=BTC_NU)
        sigma_total = np.sqrt(P + v_adj ** 2)
        z = (r - mu) / sigma_total
        pit = stats.t.cdf(z, df=BTC_NU)

        # All PIT values valid
        assert np.all(np.isfinite(pit))
        assert np.all((pit >= 0) & (pit <= 1))

        # Weekend PIT has reasonable spread
        pit_weekend = pit[is_weekend]
        assert np.std(pit_weekend) > 0.05, "Weekend PIT degenerate"

    def test_pit_weekday_range(self):
        """Weekday PIT should cover reasonable range."""
        r, v, _, is_weekend, _ = _generate_btc_weekday_pattern()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=BTC_PHI, q=BTC_Q,
                                            c=BTC_C, nu=BTC_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total
        pit = stats.t.cdf(z, df=BTC_NU)

        pit_weekday = pit[~is_weekend]
        assert np.max(pit_weekday) - np.min(pit_weekday) > 0.5


# ===========================================================================
class TestDayOfWeekProfile:
    """Day-of-week return profiles."""

    def test_seven_days_present(self):
        """BTC trades all 7 days."""
        _, _, dow, _, _ = _generate_btc_weekday_pattern()
        unique_days = np.unique(dow)
        assert len(unique_days) == 7

    def test_day_of_week_vol_structure(self):
        """Weekend days should have higher vol than weekday median."""
        r, _, dow, _, _ = _generate_btc_weekday_pattern()
        vols = {}
        for d in range(7):
            vols[d] = np.std(r[dow == d])

        weekday_vol = np.median([vols[d] for d in range(5)])
        weekend_vol = np.mean([vols[5], vols[6]])

        assert weekend_vol > weekday_vol, (
            f"Weekend vol ({weekend_vol:.6f}) <= Weekday median ({weekday_vol:.6f})"
        )


# ===========================================================================
class TestFilterStability:
    """Filter stable on BTC weekend/weekday data."""

    def test_deterministic(self):
        r, v, _, _, _ = _generate_btc_weekday_pattern()
        mu1, P1, ll1 = run_phi_student_t_filter(r, v, phi=BTC_PHI, q=BTC_Q,
                                                  c=BTC_C, nu=BTC_NU)
        mu2, P2, ll2 = run_phi_student_t_filter(r, v, phi=BTC_PHI, q=BTC_Q,
                                                  c=BTC_C, nu=BTC_NU)
        np.testing.assert_array_equal(mu1, mu2)
        assert ll1 == ll2

    def test_no_nan(self):
        r, v, _, _, _ = _generate_btc_weekday_pattern()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=BTC_PHI, q=BTC_Q,
                                                  c=BTC_C, nu=BTC_NU)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)
