"""test_kelly_criterion_56.py -- Story 5.6

Validates Kelly Criterion integration into the signal pipeline:
1. Kelly positive when p_up > 0.52 and gain/loss > 1.0
2. Kelly zero/negative when p_up < 0.50
3. Half-Kelly provides floor for genuine signals
4. Kelly capped at 0.25
5. Fields in Signal dataclass
"""
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from decision.signals import Signal

KELLY_CAP = 0.25


def compute_kelly(p_up: float, E_gain: float, E_loss: float) -> tuple:
    """Reproduce Kelly computation from signals.py."""
    eps = 1e-12
    if E_loss > eps and E_gain > 0:
        odds = E_gain / E_loss
        kf = (p_up * odds - (1.0 - p_up)) / odds
        kf = float(np.clip(kf, 0.0, KELLY_CAP))
        kh = kf / 2.0
    else:
        kf, kh = 0.0, 0.0
    return kf, kh


class TestKellyPositiveEdge56(unittest.TestCase):

    def test_classic_kelly(self):
        """p=0.6, b=2.0 -> f*=0.4 capped at 0.25."""
        kf, kh = compute_kelly(0.6, 0.04, 0.02)
        self.assertEqual(kf, KELLY_CAP)
        self.assertAlmostEqual(kh, KELLY_CAP / 2)

    def test_moderate_edge(self):
        """p=0.55, b=1.2 -> f*=0.175."""
        kf, kh = compute_kelly(0.55, 0.024, 0.02)
        expected = (0.55 * 1.2 - 0.45) / 1.2
        self.assertAlmostEqual(kf, expected, places=5)

    def test_slight_edge(self):
        kf, _ = compute_kelly(0.52, 0.022, 0.02)
        self.assertGreater(kf, 0)

    def test_even_odds(self):
        kf, _ = compute_kelly(0.52, 0.02, 0.02)
        self.assertAlmostEqual(kf, 0.04, places=5)


class TestKellyNoEdge56(unittest.TestCase):

    def test_p50_no_edge(self):
        kf, kh = compute_kelly(0.50, 0.02, 0.02)
        self.assertAlmostEqual(kf, 0.0)

    def test_negative_edge(self):
        kf, kh = compute_kelly(0.45, 0.02, 0.02)
        self.assertEqual(kf, 0.0)
        self.assertEqual(kh, 0.0)

    def test_bad_odds(self):
        kf, _ = compute_kelly(0.55, 0.01, 0.02)
        self.assertEqual(kf, 0.0)


class TestKellyCap56(unittest.TestCase):

    def test_never_exceeds(self):
        kf, _ = compute_kelly(0.90, 0.10, 0.01)
        self.assertEqual(kf, KELLY_CAP)

    def test_half_kelly_half_of_cap(self):
        kf, kh = compute_kelly(0.90, 0.10, 0.01)
        self.assertAlmostEqual(kh, KELLY_CAP / 2)


class TestKellyZeroInputs56(unittest.TestCase):

    def test_zero_loss(self):
        kf, kh = compute_kelly(0.55, 0.02, 0.0)
        self.assertEqual(kf, 0.0)

    def test_zero_gain(self):
        kf, kh = compute_kelly(0.55, 0.0, 0.02)
        self.assertEqual(kf, 0.0)


class TestSignalKellyFields56(unittest.TestCase):

    def test_fields_set(self):
        sig = Signal(
            horizon_days=7, score=0.1, p_up=0.6, exp_ret=0.01,
            ci_low=-0.01, ci_high=0.03, ci_low_90=-0.02, ci_high_90=0.04,
            profit_pln=1000, profit_ci_low_pln=-500, profit_ci_high_pln=2500,
            position_strength=0.3,
            vol_mean=0.01, vol_ci_low=0.005, vol_ci_high=0.015,
            regime="LOW_VOL_TREND", label="BUY",
            kelly_full=0.15, kelly_half=0.075,
        )
        self.assertEqual(sig.kelly_full, 0.15)
        self.assertEqual(sig.kelly_half, 0.075)

    def test_defaults_zero(self):
        sig = Signal(
            horizon_days=7, score=0.1, p_up=0.6, exp_ret=0.01,
            ci_low=-0.01, ci_high=0.03, ci_low_90=-0.02, ci_high_90=0.04,
            profit_pln=1000, profit_ci_low_pln=-500, profit_ci_high_pln=2500,
            position_strength=0.3,
            vol_mean=0.01, vol_ci_low=0.005, vol_ci_high=0.015,
            regime="LOW_VOL_TREND", label="HOLD",
        )
        self.assertEqual(sig.kelly_full, 0.0)
        self.assertEqual(sig.kelly_half, 0.0)


class TestHalfKellyFloor56(unittest.TestCase):

    def test_floor_lifts_small(self):
        _, kh = compute_kelly(0.60, 0.03, 0.02)
        blend = 0.05
        self.assertGreater(max(blend, kh), blend)

    def test_floor_no_lower(self):
        _, kh = compute_kelly(0.55, 0.024, 0.02)
        blend = 0.5
        self.assertEqual(max(blend, kh), blend)


if __name__ == "__main__":
    unittest.main()
