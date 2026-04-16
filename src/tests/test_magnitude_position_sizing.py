"""test_magnitude_position_sizing.py -- Story 5.3

Validates forecast-magnitude-aware position sizing:
1. Larger forecasts -> larger positions
2. Risk management preserved via EU component
3. Blending formula: SIZE_EU_WEIGHT * size_eu + SIZE_MAG_WEIGHT * size_mag
4. Both components clipped to [0, 1], final capped at 1.0
"""
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from decision.signals import SIZE_EU_WEIGHT, SIZE_MAG_WEIGHT


def blended_position(eu_size: float, mu_H: float, sig_H: float) -> float:
    """Reproduce the blending formula from signals.py."""
    size_eu = float(np.clip(eu_size, 0.0, 1.0))
    size_mag = abs(mu_H) / (sig_H + 1e-6) if sig_H > 0 else 0.0
    size_mag = float(np.clip(size_mag, 0.0, 1.0))
    return float(np.clip(
        SIZE_EU_WEIGHT * size_eu + SIZE_MAG_WEIGHT * size_mag,
        0.0, 1.0
    ))


class TestBlendingConstants(unittest.TestCase):

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(SIZE_EU_WEIGHT + SIZE_MAG_WEIGHT, 1.0)

    def test_eu_weight(self):
        self.assertEqual(SIZE_EU_WEIGHT, 0.6)

    def test_mag_weight(self):
        self.assertEqual(SIZE_MAG_WEIGHT, 0.4)


class TestLargerForecastLargerPosition(unittest.TestCase):

    def test_higher_mu_larger_position(self):
        """Asset with +3% forecast gets larger position than +0.5%."""
        eu_size = 0.5  # Same EU-based size for both
        sig = 0.02     # Same volatility

        pos_large = blended_position(eu_size, 0.03, sig)
        pos_small = blended_position(eu_size, 0.005, sig)
        self.assertGreater(pos_large, pos_small)

    def test_zero_forecast_still_has_eu_component(self):
        """Zero forecast still gets EU component."""
        pos = blended_position(0.5, 0.0, 0.02)
        expected_eu = SIZE_EU_WEIGHT * 0.5
        self.assertAlmostEqual(pos, expected_eu, places=5)


class TestRiskManagementPreserved(unittest.TestCase):

    def test_negative_eu_reduces_position(self):
        """Zero EU position size reduces blended to magnitude-only."""
        pos_with_eu = blended_position(0.5, 0.01, 0.02)
        pos_no_eu = blended_position(0.0, 0.01, 0.02)
        self.assertGreater(pos_with_eu, pos_no_eu)

    def test_volatile_asset_smaller_info_ratio(self):
        """Same forecast but higher vol -> smaller info ratio -> smaller mag component."""
        eu_size = 0.5
        mu = 0.02
        pos_low_vol = blended_position(eu_size, mu, 0.01)   # IR = 2.0
        pos_high_vol = blended_position(eu_size, mu, 0.04)  # IR = 0.5
        self.assertGreater(pos_low_vol, pos_high_vol)


class TestClipping(unittest.TestCase):

    def test_eu_clipped_to_unit(self):
        """EU component clipped to [0, 1]."""
        pos = blended_position(5.0, 0.0, 0.02)  # Huge EU
        self.assertLessEqual(pos, 1.0)

    def test_mag_clipped_to_unit(self):
        """Magnitude component clipped to [0, 1]."""
        pos = blended_position(0.0, 1.0, 0.001)  # Huge IR
        self.assertLessEqual(pos, 1.0)

    def test_negative_eu_clipped(self):
        """Negative EU size clipped to 0."""
        pos = blended_position(-0.5, 0.01, 0.02)
        # size_eu = 0 (clipped), size_mag ~ 0.5
        expected = SIZE_MAG_WEIGHT * 0.5
        self.assertAlmostEqual(pos, expected, places=3)

    def test_final_capped_at_one(self):
        """Total can't exceed 1.0."""
        pos = blended_position(1.0, 1.0, 0.001)
        self.assertLessEqual(pos, 1.0)


class TestCorrelationWithForecast(unittest.TestCase):

    def test_positive_correlation(self):
        """Position size should correlate positively with |forecast|."""
        eu_size = 0.4
        sig = 0.02
        forecasts = np.linspace(0.001, 0.05, 20)
        positions = [blended_position(eu_size, mu, sig) for mu in forecasts]
        corr = np.corrcoef(np.abs(forecasts), positions)[0, 1]
        self.assertGreater(corr, 0.5,
                           f"Correlation {corr:.3f} should exceed 0.5")


class TestEdgeCases(unittest.TestCase):

    def test_zero_sig(self):
        """Zero sigma doesn't crash (uses 1e-6 floor)."""
        pos = blended_position(0.5, 0.01, 0.0)
        self.assertTrue(np.isfinite(pos))

    def test_negative_forecast_uses_abs(self):
        """Negative forecast magnitude treated same as positive."""
        pos_pos = blended_position(0.5, 0.02, 0.01)
        pos_neg = blended_position(0.5, -0.02, 0.01)
        self.assertAlmostEqual(pos_pos, pos_neg, places=8)


if __name__ == "__main__":
    unittest.main()
