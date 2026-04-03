"""
Tests for Story 1.7: Vol-Proportional Process Noise Floor Calibration.

Validates that q floor scales quadratically with asset volatility:
    q_floor(regime, vol) = Q_FLOOR_BASE[regime] * (vol / REFERENCE_VOL)^2

This ensures currencies (low-vol) get proportionally smaller q floors
and high-vol tech stocks get proportionally larger floors.
"""

import os
import sys
import unittest
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestQFloorConstants(unittest.TestCase):
    """Validate module-level constants."""

    def test_reference_vol_exists(self):
        from tuning.tune import REFERENCE_VOL
        self.assertGreater(REFERENCE_VOL, 0.0)
        self.assertLess(REFERENCE_VOL, 1.0)

    def test_abs_min_exists(self):
        from tuning.tune import Q_FLOOR_ABS_MIN
        self.assertGreater(Q_FLOOR_ABS_MIN, 0.0)
        self.assertEqual(Q_FLOOR_ABS_MIN, 1e-8)

    def test_q_floor_by_regime_exists(self):
        from tuning.tune import Q_FLOOR_BY_REGIME
        self.assertEqual(len(Q_FLOOR_BY_REGIME), 5)
        for regime in range(5):
            self.assertIn(regime, Q_FLOOR_BY_REGIME)
            self.assertGreater(Q_FLOOR_BY_REGIME[regime], 0.0)


class TestVolProportionalQFloor(unittest.TestCase):
    """Test compute_vol_proportional_q_floor."""

    def test_reference_vol_returns_base(self):
        """At reference vol, floor should equal base."""
        from tuning.tune import compute_vol_proportional_q_floor, Q_FLOOR_BY_REGIME, REFERENCE_VOL
        for regime in range(5):
            q = compute_vol_proportional_q_floor(regime, REFERENCE_VOL)
            self.assertAlmostEqual(q, Q_FLOOR_BY_REGIME[regime], places=12)

    def test_currency_smaller(self):
        """Currency (vol~5%) floor should be ~10x smaller than equity."""
        from tuning.tune import compute_vol_proportional_q_floor, REFERENCE_VOL
        q_curr = compute_vol_proportional_q_floor(0, 0.05)
        q_eq = compute_vol_proportional_q_floor(0, REFERENCE_VOL)
        ratio = q_eq / q_curr
        self.assertGreater(ratio, 8.0, f"Equity/Currency ratio {ratio:.1f} should be ~10x")
        self.assertLess(ratio, 12.0)

    def test_highvol_larger(self):
        """High-vol tech (vol~50%) floor should be ~10x larger than equity."""
        from tuning.tune import compute_vol_proportional_q_floor, REFERENCE_VOL
        q_hv = compute_vol_proportional_q_floor(0, 0.50)
        q_eq = compute_vol_proportional_q_floor(0, REFERENCE_VOL)
        ratio = q_hv / q_eq
        self.assertGreater(ratio, 8.0, f"HiVol/Equity ratio {ratio:.1f} should be ~10x")
        self.assertLess(ratio, 12.0)

    def test_quadratic_scaling(self):
        """Scaling should be exactly quadratic in vol ratio."""
        from tuning.tune import compute_vol_proportional_q_floor, REFERENCE_VOL
        v1 = 0.10
        v2 = 0.20
        q1 = compute_vol_proportional_q_floor(0, v1)
        q2 = compute_vol_proportional_q_floor(0, v2)
        expected_ratio = (v2 / v1) ** 2  # should be 4.0
        actual_ratio = q2 / q1
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=6)

    def test_abs_min_floor(self):
        """Very low vol should still get at least Q_FLOOR_ABS_MIN."""
        from tuning.tune import compute_vol_proportional_q_floor, Q_FLOOR_ABS_MIN
        q = compute_vol_proportional_q_floor(2, 0.001)  # 0.1% vol
        self.assertGreaterEqual(q, Q_FLOOR_ABS_MIN)

    def test_zero_vol_safe(self):
        from tuning.tune import compute_vol_proportional_q_floor, Q_FLOOR_ABS_MIN
        q = compute_vol_proportional_q_floor(0, 0.0)
        self.assertGreaterEqual(q, Q_FLOOR_ABS_MIN)

    def test_all_regimes(self):
        """All regimes should return positive floor."""
        from tuning.tune import compute_vol_proportional_q_floor
        for regime in range(5):
            q = compute_vol_proportional_q_floor(regime, 0.20)
            self.assertGreater(q, 0.0)

    def test_crisis_largest(self):
        """CRISIS_JUMP (regime 4) should have the largest floor."""
        from tuning.tune import compute_vol_proportional_q_floor
        vol = 0.20
        floors = [compute_vol_proportional_q_floor(r, vol) for r in range(5)]
        self.assertEqual(max(floors), floors[4],
                         "CRISIS_JUMP should have largest q floor")


class TestApplyRegimeQFloorWithVol(unittest.TestCase):
    """Test that apply_regime_q_floor uses vol-proportional floor."""

    def test_backward_compatible(self):
        """Without asset_vol, should use hardcoded Q_FLOOR_BY_REGIME."""
        from tuning.tune import apply_regime_q_floor, Q_FLOOR_BY_REGIME
        models = {
            "test_model": {
                "fit_success": True,
                "q": 1e-10,  # way below floor
                "c": 1.0,
                "phi": 0.1,
                "nu": 8.0,
                "log_likelihood": -100.0,
                "bic": 200.0,
            }
        }
        returns = np.random.randn(200) * 0.01
        vol = np.full(200, 0.01)
        n_floored, n_total = apply_regime_q_floor(models, 0, returns, vol)
        self.assertEqual(n_floored, 1)

    def test_with_asset_vol(self):
        """With asset_vol, should use vol-proportional floor."""
        from tuning.tune import apply_regime_q_floor, compute_vol_proportional_q_floor
        q_expected = compute_vol_proportional_q_floor(0, 0.05)  # currency
        models = {
            "test_model": {
                "fit_success": True,
                "q": q_expected * 0.1,  # below computed floor
                "c": 1.0,
                "phi": 0.1,
                "nu": 8.0,
                "log_likelihood": -100.0,
                "bic": 200.0,
            }
        }
        returns = np.random.randn(200) * 0.003
        vol = np.full(200, 0.003)
        n_floored, _ = apply_regime_q_floor(models, 0, returns, vol, asset_vol=0.05)
        self.assertEqual(n_floored, 1)


if __name__ == "__main__":
    unittest.main()
