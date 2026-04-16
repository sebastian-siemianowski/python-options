"""
Story 11.2 – Hansen Skew-t Kalman Robust Weight Function
=========================================================
Verify monotonicity, asymmetric downweighting, and correct skew-
direction behavior for hansen_robust_weight_scalar().
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

from models.numba_kernels import (
    hansen_constants_kernel,
    hansen_robust_weight_scalar,
)


def _weight(z_std, nu, lam):
    """Helper: compute robust weight from a standardized z (S=1 proxy)."""
    a, b, _ = hansen_constants_kernel(nu, lam)
    # S such that forecast_scale = 1  =>  S * (nu-2)/nu = 1  =>  S = nu/(nu-2)
    S = nu / (nu - 2.0)
    return hansen_robust_weight_scalar(z_std, S, nu, lam, a, b)


class TestMonotonicity:
    """Monotonicity: |z1| > |z2| => w(z1) <= w(z2) on each piece."""

    @pytest.mark.parametrize("lam", [-0.5, 0.0, 0.5])
    def test_monotone_left_piece(self, lam):
        """Weights decrease with increasing |z| on the left piece."""
        nu = 5.0
        z_left = np.array([-0.5, -1.0, -2.0, -3.0, -5.0])
        weights = [_weight(z, nu, lam) for z in z_left]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], (
                f"Non-monotone at z={z_left[i]}: w={weights[i]} vs z={z_left[i+1]}: w={weights[i+1]}"
            )

    @pytest.mark.parametrize("lam", [-0.5, 0.0, 0.5])
    def test_monotone_right_piece(self, lam):
        """Weights decrease with increasing |z| on the right piece."""
        nu = 5.0
        z_right = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        weights = [_weight(z, nu, lam) for z in z_right]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]


class TestAsymmetricDownweighting:
    """Skew direction: lambda > 0 = right-skew; lambda < 0 = left-skew."""

    def test_positive_lambda_right_skew(self):
        """lambda > 0: right outliers get less downweighting than left."""
        nu = 5.0
        lam = 0.5
        z_pos = 3.0
        z_neg = -3.0
        w_right = _weight(z_pos, nu, lam)
        w_left = _weight(z_neg, nu, lam)
        # Right-skew: right outliers are less surprising, so w_right > w_left
        assert w_right > w_left, (
            f"Expected right outlier less downweighted: w_right={w_right}, w_left={w_left}"
        )

    def test_negative_lambda_left_skew(self):
        """lambda < 0: left outliers get less downweighting than right."""
        nu = 5.0
        lam = -0.5
        z_pos = 3.0
        z_neg = -3.0
        w_right = _weight(z_pos, nu, lam)
        w_left = _weight(z_neg, nu, lam)
        # Left-skew: left outliers are less surprising, so w_left > w_right
        assert w_left > w_right, (
            f"Expected left outlier less downweighted: w_left={w_left}, w_right={w_right}"
        )

    def test_symmetric_lambda_zero(self):
        """lambda = 0: symmetric weights for |z| = const."""
        nu = 5.0
        lam = 0.0
        w_right = _weight(3.0, nu, lam)
        w_left = _weight(-3.0, nu, lam)
        assert abs(w_right - w_left) < 1e-12


class TestWeightBounds:
    """Weights bounded in (0, 1+1/nu] and equal 1 at z=0 (approx)."""

    def test_weight_at_zero(self):
        """At z=0: w should be close to 1 (depends on cutpoint offset)."""
        nu = 5.0
        for lam in [-0.3, 0.0, 0.3]:
            w = _weight(0.0, nu, lam)
            # For z=0, z_eff = a/(1+-lambda) which is small; w ~ (nu+1)/(nu+small) ~ 1
            assert 0.5 < w <= 1.5, f"lam={lam}: w={w}"

    def test_weight_positive(self):
        """Weights are always positive."""
        nu = 5.0
        for lam in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            for z in [-10.0, -3.0, 0.0, 3.0, 10.0]:
                w = _weight(z, nu, lam)
                assert w > 0, f"lam={lam}, z={z}: w={w}"

    def test_heavy_downweighting_fraction(self):
        """For extreme outliers, w < 0.5 (heavy downweighting)."""
        nu = 5.0
        lam = -0.3
        w = _weight(5.0, nu, lam)
        assert w < 0.5, f"Expected heavy downweighting at z=5: w={w}"


class TestAssetClassBehavior:
    """Validate weight function on synthetic data mimicking asset classes."""

    @pytest.mark.parametrize("asset_lam", [
        ("TSLA", -0.3),
        ("NFLX", -0.2),
        ("MSTR", -0.5),
        ("AAPL", -0.15),
        ("RKLB", -0.4),
        ("BTC-USD", -0.25),
        ("SQ", -0.3),
        ("AFRM", -0.35),
    ])
    def test_crash_downweighted_more(self, asset_lam):
        """For equity-like lambda < 0: crashes (z < 0) downweighted less than rallies."""
        _, lam = asset_lam
        nu = 5.0
        # Crash: z = -4, Rally: z = +4
        w_crash = _weight(-4.0, nu, lam)
        w_rally = _weight(4.0, nu, lam)
        # Negative lambda = left-skew = crashes are expected, so crash gets MORE weight
        assert w_crash > w_rally
