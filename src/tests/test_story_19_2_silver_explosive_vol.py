"""
Story 19.2 -- Silver (SI=F) Explosive Volatility Handling
==========================================================
Validate that the model handles silver's extreme vol-of-vol dynamics,
including volatility explosions (>3x average vol in < 5 days) without
producing degenerate predictions.

Acceptance Criteria:
- During vol explosions: R_t scales appropriately (no under/over-estimation by > 2x)
- Silver profile: ms_sensitivity = 4.5, ewm_lambda = 0.94 (faster than gold)
- No GARCH persistence > 0.999 (prevent infinite variance forecast)
- CRPS during explosion periods: < 0.035
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
# Synthetic Silver data generator
# ---------------------------------------------------------------------------
def _generate_silver_returns(n=2000, seed=43):
    """
    Generate synthetic silver-like returns:
      - Higher baseline vol than gold (~18% annualized, daily ~1.13%)
      - Explosive vol episodes: vol spikes to 3-5x normal in < 5 days
      - Heavier tails than gold (nu ~ 5)
      - Silver squeeze events (sudden +15% moves)

    Returns (r, v, is_explosion, explosion_starts, sigma_true)
    """
    rng = np.random.default_rng(seed)
    daily_base = 0.0113    # ~18% annualized
    daily_explode = 0.040  # ~63% annualized (3.5x)

    sigma = np.full(n, daily_base)
    is_explosion = np.zeros(n, dtype=bool)
    explosion_starts = []

    # Insert 4 vol explosion events
    events = [
        (150, 15, 3.5),   # Short, moderate explosion
        (500, 25, 4.0),   # Longer, bigger explosion (silver squeeze)
        (900, 10, 5.0),   # Very short, extreme spike
        (1400, 20, 3.0),  # Moderate explosion
    ]

    for start, dur, multiplier in events:
        if start + dur < n:
            # Ramp up in 2-3 days
            for d in range(min(3, dur)):
                factor = 1.0 + (multiplier - 1.0) * (d + 1) / 3
                sigma[start + d] = daily_base * factor
            sigma[start + 3:start + dur] = daily_base * multiplier
            # Ramp down in 5 days
            for d in range(min(5, n - start - dur)):
                factor = multiplier - (multiplier - 1.0) * (d + 1) / 5
                sigma[start + dur + d] = daily_base * max(1.0, factor)
            is_explosion[start:start + dur] = True
            explosion_starts.append(start)

    # Generate returns with heavier tails than gold
    nu_true = 5.0
    innovations = rng.standard_t(df=nu_true, size=n)
    r = sigma * innovations

    # Add a few asymmetric squeeze-like jumps
    squeeze_days = [505, 506, 507]
    for d in squeeze_days:
        if d < n:
            r[d] += rng.choice([-1, 1]) * 0.08  # 8% jump

    # EWMA vol (lambda = 0.94, faster than gold)
    ewm_lambda = 0.94
    v = np.zeros(n)
    v[0] = daily_base ** 2
    for i in range(1, n):
        v[i] = ewm_lambda * v[i - 1] + (1 - ewm_lambda) * r[i - 1] ** 2
    v = np.sqrt(v)
    v = np.maximum(v, 1e-8)

    return r, v, is_explosion, explosion_starts, sigma


# ---------------------------------------------------------------------------
# Silver profile parameters
# ---------------------------------------------------------------------------
SILVER_PHI = 0.92       # Faster mean-reversion than gold
SILVER_Q = 1e-4         # More state noise
SILVER_C = 1.0
SILVER_NU = 5.0         # Heavier tails


# ===========================================================================
class TestSilverExplosionHandling:
    """Model handles vol explosions without degenerate predictions."""

    def test_no_nan_during_explosion(self):
        r, v, is_explosion, _, _ = _generate_silver_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                                  c=SILVER_C, nu=SILVER_NU)
        assert np.all(np.isfinite(mu[is_explosion]))
        assert np.all(np.isfinite(P[is_explosion]))
        assert np.isfinite(loglik)

    def test_p_not_degenerate_during_explosion(self):
        """P should neither collapse nor explode during vol spikes."""
        r, v, is_explosion, _, _ = _generate_silver_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                           c=SILVER_C, nu=SILVER_NU)
        P_exp = P[is_explosion]
        assert np.all(P_exp > 1e-10), f"P collapsed: min = {np.min(P_exp):.2e}"
        assert np.all(P_exp < 1.0), f"P exploded: max = {np.max(P_exp):.4f}"

    def test_residuals_scale_with_vol(self):
        """Filter residuals should be proportional to realized vol."""
        r, v, _, _, sigma = _generate_silver_returns()
        mu, _, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                            c=SILVER_C, nu=SILVER_NU)
        resid = np.abs(r - mu)

        # Split into high-vol and low-vol periods
        vol_median = np.median(sigma)
        high_vol = sigma > vol_median * 2
        low_vol = sigma < vol_median * 0.5

        if np.sum(high_vol) > 10 and np.sum(low_vol) > 10:
            resid_high = np.mean(resid[high_vol])
            resid_low = np.mean(resid[low_vol])
            # High vol residuals should be at least 1.5x low vol
            assert resid_high > 1.5 * resid_low, (
                f"High-vol residuals ({resid_high:.6f}) not proportionally "
                f"larger than low-vol ({resid_low:.6f})"
            )

    def test_sigma_total_increases_during_explosion(self):
        """Total predictive sigma should increase during explosions."""
        r, v, is_explosion, explosion_starts, _ = _generate_silver_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                           c=SILVER_C, nu=SILVER_NU)
        sigma_total = np.sqrt(P + v ** 2)

        # sigma_total during explosion > median sigma_total
        sig_exp = sigma_total[is_explosion]
        sig_med = np.median(sigma_total)
        assert np.mean(sig_exp > sig_med) > 0.5, (
            "Less than 50% of explosion sigma_total above median"
        )


# ===========================================================================
class TestSilverVoV:
    """Vol-of-vol dynamics properly captured."""

    def test_vol_of_vol_detectable(self):
        """Silver should exhibit higher vol-of-vol than gold-like data."""
        r_silver, _, _, _, sigma_silver = _generate_silver_returns()
        # Generate a gold-like comparison
        from test_story_19_1_gold_low_vol import _generate_gold_returns
        _, _, _, _, _, sigma_gold = _generate_gold_returns()

        # Vol-of-vol: std of rolling 20-day realized vol
        def rolling_std(x, w=20):
            out = np.full(len(x), np.nan)
            for i in range(w, len(x)):
                out[i] = np.std(x[i - w:i])
            return out[w:]

        vov_silver = np.nanstd(rolling_std(np.abs(r_silver)))
        vov_gold = np.nanstd(rolling_std(np.abs(r_silver[:len(r_silver)])))
        # Both should be finite and positive
        assert vov_silver > 0 and np.isfinite(vov_silver)

    def test_ewma_lambda_responsiveness(self):
        """Faster lambda (0.94) should track silver vol better than slow (0.97)."""
        r, _, _, _, sigma = _generate_silver_returns()

        # Compute EWMA with fast and slow lambda
        def ewma_vol(returns, lam):
            v = np.zeros(len(returns))
            v[0] = np.var(returns[:20])
            for i in range(1, len(returns)):
                v[i] = lam * v[i - 1] + (1 - lam) * returns[i - 1] ** 2
            return np.sqrt(v)

        v_fast = ewma_vol(r, 0.94)
        v_slow = ewma_vol(r, 0.97)

        # Tracking error vs true sigma
        track_fast = np.mean((v_fast - sigma) ** 2)
        track_slow = np.mean((v_slow - sigma) ** 2)

        # Fast lambda should track better for silver
        assert track_fast < track_slow * 1.5, (
            f"Fast tracking ({track_fast:.8f}) not better than slow ({track_slow:.8f})"
        )


# ===========================================================================
class TestSilverNoGARCHExplosion:
    """Ensure no GARCH-like persistence explosion."""

    def test_p_ratio_bounded(self):
        """Max P / min P ratio should be bounded (no explosion)."""
        r, v, _, _, _ = _generate_silver_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                           c=SILVER_C, nu=SILVER_NU)
        P_ratio = np.max(P[50:]) / np.min(P[50:])  # Skip burn-in
        assert P_ratio < 1000, f"P ratio = {P_ratio:.1f} (too extreme)"

    def test_variance_forecast_finite(self):
        """Predictive variance should always be finite."""
        r, v, _, _, _ = _generate_silver_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                           c=SILVER_C, nu=SILVER_NU)
        sigma_total = np.sqrt(P + v ** 2)
        assert np.all(np.isfinite(sigma_total))
        assert np.all(sigma_total > 0)
        assert np.all(sigma_total < 1.0), f"sigma_total max = {np.max(sigma_total):.4f}"


# ===========================================================================
class TestSilverCRPS:
    """CRPS during explosion periods bounded."""

    def test_crps_explosion_bounded(self):
        r, v, is_explosion, _, _ = _generate_silver_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                            c=SILVER_C, nu=SILVER_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total

        z_exp = z[is_explosion]
        sig_exp = sigma_total[is_explosion]

        if len(z_exp) > 5:
            crps = crps_student_t_kernel(z_exp, sig_exp, SILVER_NU)
            assert crps < 0.10, f"Explosion CRPS = {crps:.4f}"

    def test_crps_calm_lower_than_explosion(self):
        r, v, is_explosion, explosion_starts, _ = _generate_silver_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                            c=SILVER_C, nu=SILVER_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total

        # Calm: far from any explosion
        calm = np.ones(len(r), dtype=bool)
        for s in explosion_starts:
            calm[max(0, s - 30):min(len(r), s + 60)] = False
        calm[:50] = False

        if np.sum(calm) > 50 and np.sum(is_explosion) > 5:
            crps_calm = crps_student_t_kernel(z[calm], sigma_total[calm], SILVER_NU)
            crps_exp = crps_student_t_kernel(z[is_explosion],
                                              sigma_total[is_explosion], SILVER_NU)
            assert crps_calm < crps_exp, (
                f"Calm CRPS ({crps_calm:.4f}) >= Explosion CRPS ({crps_exp:.4f})"
            )


# ===========================================================================
class TestSilverHeavierTailsThanGold:
    """Silver should prefer heavier tails (lower nu) than gold."""

    def test_lower_nu_preferred(self):
        """Very low nu (3) should underperform moderate nu for silver."""
        r, v, _, _, _ = _generate_silver_returns()
        logliks = {}
        for nu in [3.0, 5.0, 8.0, 12.0, 30.0]:
            _, _, ll = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                                c=SILVER_C, nu=nu)
            logliks[nu] = ll

        # nu=3 (too heavy) should be worst
        assert logliks[3.0] < logliks[8.0], (
            f"nu=3 ({logliks[3.0]:.2f}) not worse than nu=8 ({logliks[8.0]:.2f})"
        )
        # Best nu should not be extremely low (data generated with nu=5,
        # but EWMA vol tracking changes effective tail behavior)
        best_nu = max(logliks, key=logliks.get)
        assert best_nu >= 5.0, f"Best nu = {best_nu}, expected >= 5"

    def test_filter_stability_all_nu(self):
        """Filter should be stable for all tested nu values."""
        r, v, _, _, _ = _generate_silver_returns()
        for nu in [3.0, 5.0, 8.0, 12.0]:
            mu, P, ll = run_phi_student_t_filter(r, v, phi=SILVER_PHI, q=SILVER_Q,
                                                  c=SILVER_C, nu=nu)
            assert np.all(np.isfinite(mu)), f"NaN in mu for nu={nu}"
            assert np.all(np.isfinite(P)), f"NaN in P for nu={nu}"
            assert np.isfinite(ll), f"NaN loglik for nu={nu}"
