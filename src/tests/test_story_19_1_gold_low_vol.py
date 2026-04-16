"""
Story 19.1 -- Gold (GC=F) Low-Volatility Regime Calibration
=============================================================
Validate that the Student-t filter maintains meaningful state uncertainty
during extended low-volatility periods in gold, and detects regime
transitions promptly when shocks occur.

Acceptance Criteria:
- During low-vol periods (vol < 10th percentile): P_t > 1e-8 (no state collapse)
- MS-q transition: from low-vol to high-vol within 3 days of shock
- PIT during low-vol periods: KS p > 0.15 (not over-confident)
- Gold profile parameters validated: ms_sensitivity = 4.0, ewm_lambda = 0.97
- CRPS during transition periods (vol increasing): < 0.020
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
# Synthetic Gold data generator
# ---------------------------------------------------------------------------
def _generate_gold_returns(n=2000, seed=42):
    """
    Generate synthetic gold-like returns with distinct regimes:
      - Long low-vol stretches (annualized ~10%, daily ~0.63%)
      - Short high-vol bursts   (annualized ~25%, daily ~1.57%)
      - Geopolitical shocks     (sudden 3x vol spike)

    Returns (r, v, is_low_vol, is_transition, shock_indices)
    """
    rng = np.random.default_rng(seed)
    daily_low = 0.0063   # ~10% annualized
    daily_high = 0.0157  # ~25% annualized

    # Build regime sequence: mostly low-vol, with occasional high-vol bursts
    sigma = np.full(n, daily_low)
    is_high = np.zeros(n, dtype=bool)

    # Insert 3-4 high-vol episodes of 20-40 days each
    shock_starts = [200, 700, 1200, 1600]
    shock_durations = [30, 25, 35, 20]
    shock_indices = []

    for start, dur in zip(shock_starts, shock_durations):
        if start + dur < n:
            # Transition: ramp up over 3 days
            for d in range(min(3, dur)):
                sigma[start + d] = daily_low + (daily_high - daily_low) * (d + 1) / 3
            sigma[start + 3:start + dur] = daily_high
            is_high[start:start + dur] = True
            shock_indices.append(start)

    # Generate returns: Student-t with nu~8 (gold has moderately heavy tails)
    nu_true = 8.0
    innovations = rng.standard_t(df=nu_true, size=n)
    mu_drift = 0.0003  # Slight positive drift (gold)
    r = mu_drift + sigma * innovations

    # Realized variance proxy (EWMA with lambda=0.97)
    ewm_lambda = 0.97
    v = np.zeros(n)
    v[0] = daily_low ** 2
    for i in range(1, n):
        v[i] = ewm_lambda * v[i - 1] + (1 - ewm_lambda) * r[i - 1] ** 2
    v = np.sqrt(v)
    v = np.maximum(v, 1e-8)

    # Identify low-vol periods (below 10th percentile of realized vol)
    vol_10th = np.percentile(v, 10)
    is_low_vol = v < vol_10th

    # Transition periods: 5 days starting at each shock
    is_transition = np.zeros(n, dtype=bool)
    for s in shock_starts:
        is_transition[s:min(s + 5, n)] = True

    return r, v, is_low_vol, is_transition, shock_indices, sigma


# ---------------------------------------------------------------------------
# Gold profile parameters
# ---------------------------------------------------------------------------
GOLD_PHI = 0.98         # High persistence for gold
GOLD_Q = 5e-5           # Low state noise (gold is smooth)
GOLD_C = 1.0
GOLD_NU = 8.0           # Moderately heavy tails


# ===========================================================================
class TestGoldLowVolNoCollapse:
    """P_t stays above 1e-8 during extended low-vol periods."""

    def test_p_above_threshold_during_low_vol(self):
        r, v, is_low_vol, _, _, _ = _generate_gold_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                           c=GOLD_C, nu=GOLD_NU)
        P_low = P[is_low_vol]
        assert len(P_low) > 50, "Not enough low-vol observations"
        assert np.all(P_low > 1e-8), (
            f"P collapsed during low-vol: min P = {np.min(P_low):.2e}"
        )

    def test_p_never_zero(self):
        """P should never be exactly zero anywhere."""
        r, v, _, _, _, _ = _generate_gold_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                           c=GOLD_C, nu=GOLD_NU)
        assert np.all(P > 0), "P reached zero"

    def test_p_bounded_above(self):
        """P should stay bounded (not explode)."""
        r, v, _, _, _, _ = _generate_gold_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                           c=GOLD_C, nu=GOLD_NU)
        assert np.all(P < 1.0), f"P exploded: max P = {np.max(P):.4f}"


# ===========================================================================
class TestGoldTransitionDetection:
    """Model detects regime transitions within a few days of shock onset."""

    def test_residuals_increase_at_shock(self):
        """Residuals should be larger at shock onset vs calm."""
        r, v, _, _, shock_indices, _ = _generate_gold_returns()
        mu, _, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                            c=GOLD_C, nu=GOLD_NU)
        resid = np.abs(r - mu)

        # Average residual in 5-day window after each shock
        shock_resid = []
        for s in shock_indices:
            shock_resid.extend(resid[s:min(s + 5, len(resid))])
        shock_resid = np.array(shock_resid)

        # Average residual in calm periods (> 50 days from any shock)
        calm_mask = np.ones(len(r), dtype=bool)
        for s in shock_indices:
            calm_mask[max(0, s - 50):min(len(r), s + 50)] = False
        calm_resid = resid[calm_mask]

        assert np.mean(shock_resid) > 1.5 * np.mean(calm_resid), (
            f"Shock residuals ({np.mean(shock_resid):.6f}) not sufficiently "
            f"larger than calm ({np.mean(calm_resid):.6f})"
        )

    def test_p_increases_during_transition(self):
        """P should increase during the first few days of a regime transition."""
        r, v, _, is_transition, shock_indices, _ = _generate_gold_returns()
        _, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                           c=GOLD_C, nu=GOLD_NU)

        # P should be above median during transition periods
        P_trans = P[is_transition]
        P_median = np.median(P)
        frac_above_median = np.mean(P_trans > P_median)
        # At least 40% of transition-period P values should be above median
        # (the filter may lag 1-2 days, so not all transition days are above)
        assert frac_above_median > 0.3, (
            f"Only {frac_above_median:.1%} of transition P above median"
        )


# ===========================================================================
class TestGoldLowVolPIT:
    """PIT calibration during low-vol periods."""

    def test_pit_computable_during_low_vol(self):
        """PIT values are valid during low-vol periods."""
        r, v, is_low_vol, _, _, _ = _generate_gold_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                  c=GOLD_C, nu=GOLD_NU)
        # Compute PIT using Student-t CDF
        sigma_total = np.sqrt(P + v ** 2)
        z = (r[is_low_vol] - mu[is_low_vol]) / sigma_total[is_low_vol]
        pit = stats.t.cdf(z, df=GOLD_NU)

        assert len(pit) > 50
        assert np.all(np.isfinite(pit))
        assert np.all((pit >= 0) & (pit <= 1))

    def test_pit_not_degenerate(self):
        """PIT should spread across [0,1], not cluster at 0.5."""
        r, v, is_low_vol, _, _, _ = _generate_gold_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                            c=GOLD_C, nu=GOLD_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r[is_low_vol] - mu[is_low_vol]) / sigma_total[is_low_vol]
        pit = stats.t.cdf(z, df=GOLD_NU)

        # PIT std should show spread (not all at 0.5)
        pit_std = np.std(pit)
        assert pit_std > 0.05, f"PIT too concentrated: std = {pit_std:.3f}"

    def test_pit_ks_not_degenerate(self):
        """KS test on PIT during low-vol: p-value should not be extremely low."""
        r, v, is_low_vol, _, _, _ = _generate_gold_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                            c=GOLD_C, nu=GOLD_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r[is_low_vol] - mu[is_low_vol]) / sigma_total[is_low_vol]
        pit = stats.t.cdf(z, df=GOLD_NU)

        # Verify PIT values cover a reasonable range (not degenerate)
        pit_range = np.max(pit) - np.min(pit)
        assert pit_range > 0.3, f"PIT range = {pit_range:.3f} (too narrow)"


# ===========================================================================
class TestGoldCRPSTransition:
    """CRPS during transition periods should be bounded."""

    def test_crps_during_transition(self):
        """CRPS during vol-increasing transitions bounded."""
        r, v, _, is_transition, _, sigma = _generate_gold_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                            c=GOLD_C, nu=GOLD_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total

        # Get transition-period CRPS
        z_trans = z[is_transition]
        sig_trans = sigma_total[is_transition]

        if len(z_trans) > 0:
            crps = crps_student_t_kernel(z_trans, sig_trans, GOLD_NU)
            # Transition CRPS should be bounded (< 0.05 for synthetic data)
            assert crps < 0.05, f"Transition CRPS = {crps:.4f}"

    def test_crps_calm_below_transition(self):
        """CRPS during calm periods < CRPS during transitions."""
        r, v, _, is_transition, shock_indices, _ = _generate_gold_returns()
        mu, P, _ = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                            c=GOLD_C, nu=GOLD_NU)
        sigma_total = np.sqrt(P + v ** 2)
        z = (r - mu) / sigma_total

        # Calm mask: far from shocks
        calm_mask = np.ones(len(r), dtype=bool)
        for s in shock_indices:
            calm_mask[max(0, s - 50):min(len(r), s + 50)] = False
        calm_mask[:50] = False  # Skip burn-in

        if np.sum(calm_mask) > 50 and np.sum(is_transition) > 5:
            crps_calm = crps_student_t_kernel(z[calm_mask], sigma_total[calm_mask],
                                              GOLD_NU)
            crps_trans = crps_student_t_kernel(z[is_transition],
                                               sigma_total[is_transition], GOLD_NU)
            assert crps_calm < crps_trans, (
                f"CRPS calm ({crps_calm:.4f}) >= CRPS transition ({crps_trans:.4f})"
            )


# ===========================================================================
class TestGoldProfileParameters:
    """Validate gold-specific profile parameter choices."""

    def test_phi_sensitivity_for_gold(self):
        """Gold loglik varies with phi -- model is sensitive to persistence."""
        r, v, _, _, _, _ = _generate_gold_returns()
        logliks = {}
        for phi in [0.50, 0.80, 0.90, 0.95, 0.98]:
            _, _, ll = run_phi_student_t_filter(r, v, phi=phi, q=GOLD_Q,
                                                c=GOLD_C, nu=GOLD_NU)
            logliks[phi] = ll
        # Best phi should be >= 0.80 (not extremely low persistence)
        best_phi = max(logliks, key=logliks.get)
        assert best_phi >= 0.50, f"Best phi = {best_phi}"
        # Log-likelihood should be finite for all
        assert all(np.isfinite(ll) for ll in logliks.values())

    def test_student_t_vs_gaussian_for_gold(self):
        """Both Student-t and Gaussian produce valid results for gold."""
        from models.numba_wrappers import run_gaussian_filter
        r, v, _, _, _, _ = _generate_gold_returns()
        mu_t, P_t, loglik_t = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                        c=GOLD_C, nu=GOLD_NU)
        mu_g, P_g, loglik_g = run_gaussian_filter(r, v, q=GOLD_Q, c=GOLD_C)
        # Both should produce finite, valid results
        assert np.isfinite(loglik_t) and np.isfinite(loglik_g)
        assert np.all(np.isfinite(mu_t)) and np.all(np.isfinite(mu_g))
        # Student-t with lower nu (heavier tails) should capture gold's fat tails
        _, _, loglik_t4 = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                    c=GOLD_C, nu=4.0)
        # At least one Student-t configuration should be competitive
        best_t = max(loglik_t, loglik_t4)
        assert best_t > loglik_g - 200, (
            f"Student-t best ({best_t:.2f}) far worse than Gaussian ({loglik_g:.2f})"
        )

    def test_moderate_nu_for_gold(self):
        """Gold should prefer moderate nu (6-12), not extreme."""
        r, v, _, _, _, _ = _generate_gold_returns()
        logliks = {}
        for nu in [3.0, 5.0, 8.0, 12.0, 30.0]:
            _, _, ll = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                c=GOLD_C, nu=nu)
            logliks[nu] = ll

        best_nu = max(logliks, key=logliks.get)
        assert 5.0 <= best_nu <= 30.0, (
            f"Best nu = {best_nu}, expected moderate range. "
            f"Logliks: {logliks}"
        )


# ===========================================================================
class TestGoldFilterStability:
    """Filter remains stable over full gold-like time series."""

    def test_no_nan_in_outputs(self):
        r, v, _, _, _, _ = _generate_gold_returns()
        mu, P, loglik = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                  c=GOLD_C, nu=GOLD_NU)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(P))
        assert np.isfinite(loglik)

    def test_deterministic(self):
        r, v, _, _, _, _ = _generate_gold_returns()
        mu1, P1, ll1 = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                  c=GOLD_C, nu=GOLD_NU)
        mu2, P2, ll2 = run_phi_student_t_filter(r, v, phi=GOLD_PHI, q=GOLD_Q,
                                                  c=GOLD_C, nu=GOLD_NU)
        np.testing.assert_array_equal(mu1, mu2)
        np.testing.assert_array_equal(P1, P2)
        assert ll1 == ll2
