"""
Story 12.2 – CST Robust Kalman Weight Accuracy
================================================
Verify that cst_robust_weight_scalar() correctly blends normal and
crisis component weights based on posterior probability.
"""

import os, sys, math
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_kernels import cst_robust_weight_scalar


NU_NORMAL = 8.0
NU_CRISIS = 3.0
EPSILON = 0.05


def _w(z, S=1.0):
    """Convenience: CST weight at standardized z with S=1."""
    innovation = z * math.sqrt(S)
    return cst_robust_weight_scalar(innovation, S, NU_NORMAL, NU_CRISIS, EPSILON)


class TestNormalRegimeWeights:
    """|z| < 2: w_CST approx w_normal."""

    def test_small_z_near_normal_weight(self):
        """At z=0, weight ~ (nu_n+1)/(nu_n) = 1.125."""
        w = _w(0.0)
        w_normal = (NU_NORMAL + 1.0) / NU_NORMAL
        assert abs(w - w_normal) < 0.2

    def test_moderate_z_dominated_by_normal(self):
        """At z=1.5, normal component dominates."""
        w = _w(1.5)
        z_sq = 1.5 ** 2
        w_normal = (NU_NORMAL + 1.0) / (NU_NORMAL + z_sq)
        # CST weight should be close to normal weight
        assert abs(w - w_normal) < 0.15


class TestCrisisRegimeWeights:
    """|z| > 4: w_CST closer to w_crisis."""

    def test_large_z_crisis_dominates(self):
        """At z=6, crisis component has higher posterior."""
        w = _w(6.0)
        z_sq = 36.0
        w_crisis = (NU_CRISIS + 1.0) / (NU_CRISIS + z_sq)
        w_normal = (NU_NORMAL + 1.0) / (NU_NORMAL + z_sq)
        # w should be between w_normal and w_crisis, closer to w_crisis
        assert w >= w_crisis - 0.01
        assert w <= w_normal + 0.01


class TestSmoothTransition:
    """Transition between normal and crisis is smooth and monotone."""

    def test_monotone_decreasing(self):
        """Weights decrease with |z|."""
        z_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        weights = [_w(z) for z in z_values]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], (
                f"Non-monotone: w({z_values[i]})={weights[i]} > w({z_values[i+1]})={weights[i+1]}"
            )

    def test_symmetric_weights(self):
        """w(z) = w(-z) (CST is symmetric)."""
        for z in [1.0, 2.0, 3.0, 5.0]:
            w_pos = _w(z)
            w_neg = _w(-z)
            assert abs(w_pos - w_neg) < 1e-10


class TestPosteriorConsistency:
    """Weight matches posterior-blended components."""

    def test_weight_bounded(self):
        """w_CST is between w_crisis and w_normal for all z."""
        for z in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
            w = _w(z)
            z_sq = z * z
            w_normal = (NU_NORMAL + 1.0) / (NU_NORMAL + z_sq)
            w_crisis = (NU_CRISIS + 1.0) / (NU_CRISIS + z_sq)
            w_lo = min(w_normal, w_crisis)
            w_hi = max(w_normal, w_crisis)
            assert w_lo - 0.01 <= w <= w_hi + 0.01, (
                f"z={z}: w={w} not in [{w_lo}, {w_hi}]"
            )


class TestEdgeCases:
    """Edge cases: S near zero, extreme z."""

    def test_s_near_zero(self):
        """When S is tiny, return 1.0 (no update)."""
        w = cst_robust_weight_scalar(0.01, 1e-20, NU_NORMAL, NU_CRISIS, EPSILON)
        assert w == 1.0

    def test_extreme_z(self):
        """Very large z: weight is small but positive."""
        w = _w(20.0)
        assert w > 0
        assert w < 0.1
