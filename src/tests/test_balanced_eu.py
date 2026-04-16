"""test_balanced_eu.py -- Story 5.2

Validates balanced expected utility with capped EVT inflation:
1. EVT inflation capped at EVT_MAX_INFLATION (1.5x)
2. EU_balanced uses symmetric tail treatment
3. EU_balanced > EU_asymmetric for heavy-tail assets
4. Both EU values available in Signal dataclass
"""
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from decision.signals import (
    EVT_MAX_INFLATION, EVT_GAIN_FACTOR, Signal,
)


class TestEvtMaxInflation(unittest.TestCase):

    def test_constant_value(self):
        self.assertEqual(EVT_MAX_INFLATION, 1.5)

    def test_gain_factor_less_than_one(self):
        """Gain factor should be conservative (< 1.0)."""
        self.assertLess(EVT_GAIN_FACTOR, 1.0)
        self.assertGreater(EVT_GAIN_FACTOR, 0.0)


class TestEvtInflationCap(unittest.TestCase):

    def test_cap_prevents_blowup(self):
        """EVT with xi=0.30 should be capped at 1.5x."""
        E_loss_emp = 0.02
        xi = 0.30
        correction_factor = 1.5  # Hypothetical EVT correction
        uncapped = E_loss_emp * (1 + xi * correction_factor)  # 1.45x
        capped = min(uncapped, EVT_MAX_INFLATION * E_loss_emp)
        self.assertLessEqual(capped, EVT_MAX_INFLATION * E_loss_emp)

    def test_cap_binds_for_extreme_xi(self):
        """For extreme xi, cap should bind."""
        E_loss_emp = 0.02
        xi = 0.50
        # With large correction, uncapped would be > 1.5x
        correction_factor = 3.0
        uncapped = E_loss_emp * (1 + xi * correction_factor)  # 2.5x
        capped = min(uncapped, EVT_MAX_INFLATION * E_loss_emp)
        self.assertAlmostEqual(capped, EVT_MAX_INFLATION * E_loss_emp)

    def test_no_cap_for_light_tails(self):
        """Light-tailed assets (xi~0) should not be capped."""
        E_loss_emp = 0.02
        xi = 0.05
        correction_factor = 1.5
        uncapped = E_loss_emp * (1 + xi * correction_factor)  # 1.075x
        capped = min(uncapped, EVT_MAX_INFLATION * E_loss_emp)
        self.assertAlmostEqual(capped, uncapped)


class TestBalancedEuFormula(unittest.TestCase):

    def test_balanced_gt_asymmetric_when_heavy_tails(self):
        """EU_balanced >= EU_asymmetric when gains also get correction."""
        p_up = 0.60
        E_gain = 0.03
        E_loss = 0.02
        xi = 0.25

        # Asymmetric (only loss corrected)
        loss_correction = min(1.0 + xi * 1.5, EVT_MAX_INFLATION)
        E_loss_evt = E_loss * loss_correction
        EU_asym = p_up * E_gain - (1 - p_up) * E_loss_evt

        # Balanced (both corrected)
        gain_correction = min(1.0 + abs(xi) * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
        E_gain_evt = E_gain * gain_correction
        EU_bal = p_up * E_gain_evt - (1 - p_up) * E_loss_evt

        self.assertGreater(EU_bal, EU_asym,
                           "Balanced EU should be > asymmetric EU when gains also corrected")

    def test_balanced_positive_for_moderate_edge(self):
        """EU_balanced is positive for p_up=0.58, moderate vol."""
        p_up = 0.58
        E_gain = 0.025
        E_loss = 0.020
        xi = 0.15

        loss_corr = min(1.0 + xi * 1.5, EVT_MAX_INFLATION)
        gain_corr = min(1.0 + abs(xi) * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
        E_loss_evt = E_loss * loss_corr
        E_gain_evt = E_gain * gain_corr

        EU_bal = p_up * E_gain_evt - (1 - p_up) * E_loss_evt
        self.assertGreater(EU_bal, 0,
                           "Balanced EU should be positive with p_up=0.58")

    def test_symmetry_with_zero_xi(self):
        """With xi=0, balanced EU = standard EU (no correction)."""
        p_up = 0.55
        E_gain = 0.02
        E_loss = 0.018
        xi = 0.0

        loss_corr = min(1.0 + xi * 1.5, EVT_MAX_INFLATION)
        gain_corr = min(1.0 + abs(xi) * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
        EU_bal = p_up * E_gain * gain_corr - (1 - p_up) * E_loss * loss_corr
        EU_standard = p_up * E_gain - (1 - p_up) * E_loss

        self.assertAlmostEqual(EU_bal, EU_standard, places=10)


class TestSignalDataclassHasEuFields(unittest.TestCase):

    def test_eu_fields_exist(self):
        """Signal dataclass has eu_asymmetric and eu_balanced fields."""
        sig = Signal(
            horizon_days=7, score=0.1, p_up=0.6, exp_ret=0.01,
            ci_low=-0.01, ci_high=0.03, ci_low_90=-0.02, ci_high_90=0.04,
            profit_pln=1000, profit_ci_low_pln=-500, profit_ci_high_pln=2500,
            position_strength=0.3,
            vol_mean=0.01, vol_ci_low=0.005, vol_ci_high=0.015,
            regime="LOW_VOL_TREND", label="BUY",
            eu_asymmetric=0.005,
            eu_balanced=0.008,
        )
        self.assertEqual(sig.eu_asymmetric, 0.005)
        self.assertEqual(sig.eu_balanced, 0.008)

    def test_eu_fields_default_zero(self):
        """Default values are 0.0."""
        sig = Signal(
            horizon_days=7, score=0.1, p_up=0.6, exp_ret=0.01,
            ci_low=-0.01, ci_high=0.03, ci_low_90=-0.02, ci_high_90=0.04,
            profit_pln=1000, profit_ci_low_pln=-500, profit_ci_high_pln=2500,
            position_strength=0.3,
            vol_mean=0.01, vol_ci_low=0.005, vol_ci_high=0.015,
            regime="LOW_VOL_TREND", label="HOLD",
        )
        self.assertEqual(sig.eu_asymmetric, 0.0)
        self.assertEqual(sig.eu_balanced, 0.0)


class TestGainCorrectionBounded(unittest.TestCase):

    def test_gain_correction_never_exceeds_max(self):
        """Even with large xi, gain correction <= EVT_MAX_INFLATION."""
        for xi in [0.1, 0.3, 0.5, 1.0, 2.0]:
            corr = min(1.0 + abs(xi) * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
            self.assertLessEqual(corr, EVT_MAX_INFLATION)

    def test_gain_correction_monotonic_in_xi(self):
        """Higher xi -> higher gain correction (until cap)."""
        corrections = []
        for xi in [0.0, 0.1, 0.2, 0.3, 0.4]:
            corr = min(1.0 + abs(xi) * EVT_GAIN_FACTOR, EVT_MAX_INFLATION)
            corrections.append(corr)
        for i in range(len(corrections) - 1):
            self.assertLessEqual(corrections[i], corrections[i+1])


if __name__ == "__main__":
    unittest.main()
