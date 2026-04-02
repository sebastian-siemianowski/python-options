"""
Test Story 1.13: Per-Asset Forecast Calibration Feedback Loop.

Validates:
  1. compute_asset_amplification() with Bayesian shrinkage
  2. n=10, accuracy=60% -> amp ~ 1.03 (heavily shrunk)
  3. n=250, accuracy=60% -> amp ~ 1.16 (converged)
  4. n=250, accuracy=45% -> amp ~ 0.92 (dampened)
  5. Minimum sample guard: n < 20 -> amp = 1.0
  6. Bounds enforced: amp in [0.5, 1.5]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.forecast_scorecard import (
    compute_asset_amplification,
    AMP_MIN,
    AMP_MAX,
    AMP_MIN_SAMPLES,
    AMP_N_PRIOR,
    AMP_SENSITIVITY,
)


class TestComputeAssetAmplification(unittest.TestCase):
    """Tests for compute_asset_amplification (task 1.13.2)."""

    def test_n10_accuracy_60(self):
        """n=10, accuracy=60% -> amp barely moves from 1.0 (task 1.13.7)."""
        # n=10 < AMP_MIN_SAMPLES(20), so amp = 1.0
        amp = compute_asset_amplification(0.60, 10)
        self.assertAlmostEqual(amp, 1.0,
                               msg=f"n=10 below min samples, amp should be 1.0, got {amp}")

    def test_n250_accuracy_60(self):
        """n=250, accuracy=60% -> amp ~ 1.16 (task 1.13.8)."""
        # shrinkage = 250 / (250 + 60) = 0.806
        # amp = 1.0 + (0.60 - 0.50) * 0.806 * 2.0 = 1.161
        amp = compute_asset_amplification(0.60, 250)
        self.assertAlmostEqual(amp, 1.161, places=2,
                               msg=f"n=250, 60% accuracy: expected ~1.16, got {amp:.4f}")

    def test_n250_accuracy_45(self):
        """n=250, accuracy=45% -> amp ~ 0.92 (task 1.13.9)."""
        # shrinkage = 250 / 310 = 0.806
        # amp = 1.0 + (0.45 - 0.50) * 0.806 * 2.0 = 1.0 - 0.0806 = 0.919
        amp = compute_asset_amplification(0.45, 250)
        self.assertAlmostEqual(amp, 0.919, places=2,
                               msg=f"n=250, 45% accuracy: expected ~0.92, got {amp:.4f}")

    def test_min_sample_guard(self):
        """n < 20 -> amp = 1.0 regardless of accuracy (task 1.13.5)."""
        for n in [0, 1, 5, 10, 19]:
            amp = compute_asset_amplification(0.90, n)
            self.assertEqual(amp, 1.0, f"n={n} should return 1.0, got {amp}")

    def test_n20_activates(self):
        """n=20 is the minimum for amplification to kick in."""
        amp = compute_asset_amplification(0.70, 20)
        # shrinkage = 20/80 = 0.25
        # amp = 1.0 + (0.70 - 0.50) * 0.25 * 2.0 = 1.10
        self.assertGreater(amp, 1.0, f"n=20 with 70% accuracy should amplify, got {amp}")

    def test_upper_bound(self):
        """Even perfect accuracy can't exceed AMP_MAX."""
        amp = compute_asset_amplification(1.0, 10000)
        self.assertLessEqual(amp, AMP_MAX)

    def test_lower_bound(self):
        """Even terrible accuracy can't go below AMP_MIN."""
        amp = compute_asset_amplification(0.0, 10000)
        self.assertGreaterEqual(amp, AMP_MIN)

    def test_neutral_accuracy(self):
        """50% accuracy -> amp = 1.0 regardless of n."""
        for n in [20, 100, 500, 1000]:
            amp = compute_asset_amplification(0.50, n)
            self.assertAlmostEqual(amp, 1.0, places=6,
                                   msg=f"50% accuracy with n={n} should be neutral")

    def test_monotonically_increasing_with_accuracy(self):
        """Higher accuracy -> higher amp at fixed n."""
        n = 200
        accs = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        amps = [compute_asset_amplification(a, n) for a in accs]
        for i in range(len(amps) - 1):
            self.assertLessEqual(amps[i], amps[i + 1],
                                 f"amp({accs[i]}) = {amps[i]} should <= amp({accs[i+1]}) = {amps[i+1]}")

    def test_shrinkage_increases_with_n(self):
        """More data -> amp moves further from 1.0 (for fixed accuracy > 50%)."""
        acc = 0.65
        amp_small = compute_asset_amplification(acc, 25)
        amp_large = compute_asset_amplification(acc, 500)
        self.assertLess(amp_small, amp_large,
                        f"n=25 amp={amp_small} should be less extreme than n=500 amp={amp_large}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
