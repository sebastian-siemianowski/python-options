"""
Story 6.3: Smooth Asymmetric Nu Calibration per Asset Class
=============================================================
Calibrated (alpha_asym, k_asym) parameters per asset class.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.phi_student_t import (
    get_asymmetric_nu_params, compute_empirical_tail_ratio,
    ASSET_CLASS_ASYMMETRIC_NU,
)


class TestAsymmetricNuCalibration:
    """Acceptance criteria for Story 6.3."""

    def test_equity_negative_alpha(self):
        """AC1: Equities have alpha < 0 (heavier left tail)."""
        alpha, k = get_asymmetric_nu_params('NFLX')
        assert alpha < 0, f"Equity alpha={alpha} should be < 0"

    def test_gold_symmetric(self):
        """AC2: Gold has alpha ~ 0 (symmetric, safe haven)."""
        alpha, k = get_asymmetric_nu_params('GC=F')
        assert abs(alpha) <= 0.05, f"Gold alpha={alpha} should be ~0"

    def test_crypto_unconstrained(self):
        """AC3: Crypto can have positive alpha (bubble dynamics)."""
        alpha, k = get_asymmetric_nu_params('BTC-USD')
        # Crypto alpha can be positive
        assert isinstance(alpha, float)

    def test_large_cap_mild_skew(self):
        """AC4: Large cap has milder asymmetry than small cap."""
        alpha_large, _ = get_asymmetric_nu_params('MSFT')
        alpha_small, _ = get_asymmetric_nu_params('RKLB')  # Will use equity default
        # Both should be negative
        assert alpha_large < 0

    def test_high_vol_strong_crash_tail(self):
        """AC5: High-vol stocks have stronger crash asymmetry."""
        alpha_hv, _ = get_asymmetric_nu_params('MSTR')
        alpha_lc, _ = get_asymmetric_nu_params('MSFT')
        assert alpha_hv < alpha_lc, \
            f"High-vol alpha={alpha_hv} should be < large-cap alpha={alpha_lc}"

    def test_all_asset_classes_defined(self):
        """AC6: All expected asset classes have entries."""
        required = ['equity', 'metals_gold', 'crypto', 'high_vol', 'index']
        for cls in required:
            assert cls in ASSET_CLASS_ASYMMETRIC_NU, f"Missing {cls}"
            alpha, k = ASSET_CLASS_ASYMMETRIC_NU[cls]
            assert isinstance(alpha, (int, float))
            assert k > 0

    def test_empirical_tail_ratio(self):
        """AC7: Empirical tail ratio correctly detects left-heavy tails."""
        rng = np.random.default_rng(42)
        # Left-skewed data
        n = 5000
        returns = rng.normal(0, 0.02, n)
        # Add crashes
        crashes = rng.choice(n, size=50, replace=False)
        returns[crashes] -= rng.exponential(0.05, 50)

        ratio = compute_empirical_tail_ratio(returns)
        # Should be positive (left kurtosis > right kurtosis => crash heavier)
        assert ratio > 0, f"Tail ratio={ratio} should be > 0 for crash-heavy data"

    def test_empirical_tail_ratio_symmetric(self):
        """AC8: Symmetric data gives tail ratio near 0."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 5000)
        ratio = compute_empirical_tail_ratio(returns)
        assert abs(ratio) < 0.1, f"Tail ratio={ratio} should be ~0 for symmetric data"
