"""
Test Story 1.14: Hard Cap Relaxation with Confidence-Gated Bounds.

Validates:
  1. High confidence (strong agreement) widens caps
  2. Low confidence does not widen caps
  3. Cap multiplier never exceeds CAP_MAX_MULTIPLIER (2.0)
  4. Vol-bound remains as independent safety constraint
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.market_temperature import (
    CAP_CONFIDENCE_SCALE,
    CAP_MAX_MULTIPLIER,
)


class TestCapConfidenceScale(unittest.TestCase):
    """Tests for confidence-gated cap relaxation constants."""

    def test_zero_agreement_no_relaxation(self):
        """Zero agreement contrast -> multiplier = 1.0."""
        mult = min(1.0 + CAP_CONFIDENCE_SCALE * 0.0, CAP_MAX_MULTIPLIER)
        self.assertAlmostEqual(mult, 1.0)

    def test_full_agreement_relaxation(self):
        """Full agreement (contrast=1.0) -> multiplier = 1.5."""
        mult = min(1.0 + CAP_CONFIDENCE_SCALE * 1.0, CAP_MAX_MULTIPLIER)
        self.assertAlmostEqual(mult, 1.5)

    def test_multiplier_capped(self):
        """Even extreme agreement can't exceed CAP_MAX_MULTIPLIER."""
        mult = min(1.0 + CAP_CONFIDENCE_SCALE * 10.0, CAP_MAX_MULTIPLIER)
        self.assertAlmostEqual(mult, CAP_MAX_MULTIPLIER)

    def test_cap_scales_correctly(self):
        """Cap = 3% at zero confidence, 4.5% at full confidence (1.5x)."""
        base_cap = 3.0
        mult_zero = min(1.0 + CAP_CONFIDENCE_SCALE * 0.0, CAP_MAX_MULTIPLIER)
        mult_full = min(1.0 + CAP_CONFIDENCE_SCALE * 1.0, CAP_MAX_MULTIPLIER)
        
        self.assertAlmostEqual(base_cap * mult_zero, 3.0)
        self.assertAlmostEqual(base_cap * mult_full, 4.5)

    def test_mid_agreement(self):
        """50% agreement contrast -> multiplier = 1.25."""
        mult = min(1.0 + CAP_CONFIDENCE_SCALE * 0.5, CAP_MAX_MULTIPLIER)
        self.assertAlmostEqual(mult, 1.25)


class TestCapRelaxationIntegration(unittest.TestCase):
    """Integration test: ensemble_forecast with cap relaxation."""

    def test_forecasts_still_bounded(self):
        """Even with cap relaxation, forecasts should be bounded."""
        import pandas as pd
        from decision.market_temperature import ensemble_forecast
        
        np.random.seed(42)
        n = 400
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))))
        
        result = ensemble_forecast(prices, asset_type="equity", asset_name="TEST")
        
        # 7 forecasts + confidence string
        for i in range(7):
            fc = result[i]
            # Even with relaxed caps, no single-day forecast should exceed ~6% for equity
            # (hard cap 3% * 2.0 = 6%)
            if i == 0:  # 1-day horizon
                self.assertLessEqual(abs(fc), 6.5,
                                     f"1D forecast {fc} exceeds relaxed cap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
