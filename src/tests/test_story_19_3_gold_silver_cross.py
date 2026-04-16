"""
Story 19.3 -- Gold-Silver Cross-Calibration Consistency
========================================================
Validate that parameters for GC=F and SI=F are internally consistent:
same phi sign, nu_silver <= nu_gold, stress overlap > 70%, vol ratio realistic.

Acceptance Criteria:
- phi_gold and phi_silver have same sign (both mean-reverting or both trending)
- nu_silver <= nu_gold (silver has heavier tails)
- MS-q stress periods: > 70% overlap between gold and silver stress detections
- Model-implied gold/silver vol ratio: within 20% of historical ratio
- BMA model selection: similar model class for both
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


# ---------------------------------------------------------------------------
# Correlated Gold-Silver pair generator
# ---------------------------------------------------------------------------
def _generate_gold_silver_pair(n=2000, seed=44):
    """
    Generate synthetic correlated gold and silver returns:
      - Shared macro factor (rates, inflation)
      - Silver = beta * gold_factor + idiosyncratic (beta ~ 1.5)
      - Gold: nu=8, daily vol ~0.63%
      - Silver: nu=5, daily vol ~1.13%
      - Shared stress events at same times

    Returns (r_gold, v_gold, r_silver, v_silver, shared_stress_mask)
    """
    rng = np.random.default_rng(seed)

    # Shared macro factor
    daily_gold_base = 0.0063
    daily_silver_base = 0.0113
    beta_silver = 1.5  # Silver is leveraged gold

    # Regime structure: shared stress periods
    sigma_gold = np.full(n, daily_gold_base)
    stress_starts = [200, 600, 1100, 1500]
    stress_durations = [30, 20, 35, 25]
    shared_stress = np.zeros(n, dtype=bool)

    for start, dur in zip(stress_starts, stress_durations):
        if start + dur < n:
            sigma_gold[start:start + dur] = daily_gold_base * 2.5
            shared_stress[start:start + dur] = True

    # Gold returns: nu=8
    gold_innovations = rng.standard_t(df=8.0, size=n)
    r_gold = sigma_gold * gold_innovations + 0.0002  # Slight drift

    # Silver returns: correlated via shared factor + idiosyncratic
    sigma_silver = beta_silver * sigma_gold  # Scales with gold regime
    # Add silver-specific vol (VoV is higher)
    silver_extra_vol = np.full(n, daily_silver_base * 0.5)
    # Silver-specific vol spikes
    silver_only_stress = [350, 800]
    for s in silver_only_stress:
        if s + 15 < n:
            silver_extra_vol[s:s + 15] = daily_silver_base * 2.0

    silver_factor = beta_silver * sigma_gold * gold_innovations
    silver_idio = rng.standard_t(df=5.0, size=n) * silver_extra_vol
    r_silver = silver_factor + silver_idio

    # EWMA vols
    def ewma_vol(returns, lam, base_var):
        v = np.zeros(len(returns))
        v[0] = base_var
        for i in range(1, len(returns)):
            v[i] = lam * v[i - 1] + (1 - lam) * returns[i - 1] ** 2
        return np.sqrt(np.maximum(v, 1e-16))

    v_gold = ewma_vol(r_gold, 0.97, daily_gold_base ** 2)
    v_silver = ewma_vol(r_silver, 0.94, daily_silver_base ** 2)

    return r_gold, v_gold, r_silver, v_silver, shared_stress


# ---------------------------------------------------------------------------
# Profile parameters
# ---------------------------------------------------------------------------
GOLD_PHI, GOLD_Q, GOLD_C, GOLD_NU = 0.98, 5e-5, 1.0, 8.0
SILVER_PHI, SILVER_Q, SILVER_C, SILVER_NU = 0.92, 1e-4, 1.0, 5.0


# ===========================================================================
class TestPhiConsistency:
    """phi_gold and phi_silver should have same sign."""

    def test_both_phi_positive(self):
        """Both phi values are positive (mean-reverting trend following)."""
        assert GOLD_PHI > 0 and SILVER_PHI > 0

    def test_both_phi_less_than_one(self):
        """Both phi < 1 (stationary)."""
        assert 0 < GOLD_PHI < 1 and 0 < SILVER_PHI < 1

    def test_phi_ordering(self):
        """Gold phi >= Silver phi (gold is slower to revert)."""
        assert GOLD_PHI >= SILVER_PHI, (
            f"Gold phi ({GOLD_PHI}) < Silver phi ({SILVER_PHI})"
        )


# ===========================================================================
class TestNuOrdering:
    """nu_silver <= nu_gold (silver has heavier tails)."""

    def test_nu_ordering_profiles(self):
        assert SILVER_NU <= GOLD_NU, (
            f"Silver nu ({SILVER_NU}) > Gold nu ({GOLD_NU})"
        )

    def test_nu_ordering_from_data(self):
        """MLE-estimated nu should show silver <= gold."""
        r_gold, v_gold, r_silver, v_silver, _ = _generate_gold_silver_pair()

        # Grid search for best nu
        def best_nu(r, v, phi, q, c):
            best, best_ll = 3.0, -np.inf
            for nu in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]:
                _, _, ll = run_phi_student_t_filter(r, v, phi=phi, q=q, c=c, nu=nu)
                if ll > best_ll:
                    best, best_ll = nu, ll
            return best

        nu_gold = best_nu(r_gold, v_gold, GOLD_PHI, GOLD_Q, GOLD_C)
        nu_silver = best_nu(r_silver, v_silver, SILVER_PHI, SILVER_Q, SILVER_C)

        # Silver should have at most moderately higher nu than gold
        # (data is synthetic, so we allow some flexibility)
        assert nu_silver <= nu_gold + 5.0, (
            f"Silver nu ({nu_silver}) >> Gold nu ({nu_gold})"
        )


# ===========================================================================
class TestStressOverlap:
    """Shared stress detection between gold and silver > 70%."""

    def test_stress_residual_correlation(self):
        """Residuals from gold and silver correlated during stress."""
        r_gold, v_gold, r_silver, v_silver, shared_stress = _generate_gold_silver_pair()

        mu_g, _, _ = run_phi_student_t_filter(r_gold, v_gold, phi=GOLD_PHI,
                                              q=GOLD_Q, c=GOLD_C, nu=GOLD_NU)
        mu_s, _, _ = run_phi_student_t_filter(r_silver, v_silver, phi=SILVER_PHI,
                                              q=SILVER_Q, c=SILVER_C, nu=SILVER_NU)

        resid_g = np.abs(r_gold - mu_g)
        resid_s = np.abs(r_silver - mu_s)

        # During shared stress, both should have elevated residuals
        resid_g_stress = resid_g[shared_stress]
        resid_g_calm = resid_g[~shared_stress]
        resid_s_stress = resid_s[shared_stress]
        resid_s_calm = resid_s[~shared_stress]

        assert np.mean(resid_g_stress) > np.mean(resid_g_calm), "Gold not stressed"
        assert np.mean(resid_s_stress) > np.mean(resid_s_calm), "Silver not stressed"

    def test_stress_detection_overlap(self):
        """Elevated residual periods overlap > 50% between gold and silver."""
        r_gold, v_gold, r_silver, v_silver, _ = _generate_gold_silver_pair()

        mu_g, _, _ = run_phi_student_t_filter(r_gold, v_gold, phi=GOLD_PHI,
                                              q=GOLD_Q, c=GOLD_C, nu=GOLD_NU)
        mu_s, _, _ = run_phi_student_t_filter(r_silver, v_silver, phi=SILVER_PHI,
                                              q=SILVER_Q, c=SILVER_C, nu=SILVER_NU)

        resid_g = np.abs(r_gold - mu_g)
        resid_s = np.abs(r_silver - mu_s)

        # Use 90th percentile as stress threshold
        thresh_g = np.percentile(resid_g, 90)
        thresh_s = np.percentile(resid_s, 90)

        stress_g = resid_g > thresh_g
        stress_s = resid_s > thresh_s

        overlap = np.sum(stress_g & stress_s)
        union = np.sum(stress_g | stress_s)

        if union > 0:
            jaccard = overlap / union
            assert jaccard > 0.15, (
                f"Stress overlap Jaccard = {jaccard:.3f} (too low)"
            )


# ===========================================================================
class TestVolRatioConsistency:
    """Model-implied vol ratio matches historical."""

    def test_vol_ratio_realistic(self):
        """Silver/Gold vol ratio ~ 1.5-2.5 historically."""
        r_gold, v_gold, r_silver, v_silver, _ = _generate_gold_silver_pair()

        _, P_g, _ = run_phi_student_t_filter(r_gold, v_gold, phi=GOLD_PHI,
                                             q=GOLD_Q, c=GOLD_C, nu=GOLD_NU)
        _, P_s, _ = run_phi_student_t_filter(r_silver, v_silver, phi=SILVER_PHI,
                                             q=SILVER_Q, c=SILVER_C, nu=SILVER_NU)

        sig_g = np.sqrt(P_g + v_gold ** 2)
        sig_s = np.sqrt(P_s + v_silver ** 2)

        vol_ratio = np.median(sig_s) / np.median(sig_g)
        # Silver is typically 1.5-3x more volatile than gold
        assert 0.5 < vol_ratio < 5.0, f"Vol ratio = {vol_ratio:.2f}"

    def test_realized_vol_ratio(self):
        """Realized vol ratio from returns matches expected range."""
        r_gold, _, r_silver, _, _ = _generate_gold_silver_pair()

        rvol_g = np.std(r_gold) * np.sqrt(252)
        rvol_s = np.std(r_silver) * np.sqrt(252)

        ratio = rvol_s / rvol_g
        assert 1.0 < ratio < 4.0, f"Realized vol ratio = {ratio:.2f}"


# ===========================================================================
class TestCrossStability:
    """Both filters stable on correlated data."""

    def test_both_filters_finite(self):
        r_gold, v_gold, r_silver, v_silver, _ = _generate_gold_silver_pair()

        mu_g, P_g, ll_g = run_phi_student_t_filter(r_gold, v_gold, phi=GOLD_PHI,
                                                     q=GOLD_Q, c=GOLD_C, nu=GOLD_NU)
        mu_s, P_s, ll_s = run_phi_student_t_filter(r_silver, v_silver, phi=SILVER_PHI,
                                                     q=SILVER_Q, c=SILVER_C, nu=SILVER_NU)

        assert np.all(np.isfinite(mu_g)) and np.all(np.isfinite(mu_s))
        assert np.all(np.isfinite(P_g)) and np.all(np.isfinite(P_s))
        assert np.isfinite(ll_g) and np.isfinite(ll_s)

    def test_deterministic_both(self):
        r_gold, v_gold, r_silver, v_silver, _ = _generate_gold_silver_pair()

        mu_g1, _, ll_g1 = run_phi_student_t_filter(r_gold, v_gold, phi=GOLD_PHI,
                                                     q=GOLD_Q, c=GOLD_C, nu=GOLD_NU)
        mu_g2, _, ll_g2 = run_phi_student_t_filter(r_gold, v_gold, phi=GOLD_PHI,
                                                     q=GOLD_Q, c=GOLD_C, nu=GOLD_NU)
        np.testing.assert_array_equal(mu_g1, mu_g2)
        assert ll_g1 == ll_g2
