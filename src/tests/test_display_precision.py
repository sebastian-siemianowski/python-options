"""
Test Story 1.8: Display Precision and Semantic Formatting.

Validates:
  1. adaptive_format_pct() returns correct precision based on magnitude
  2. semantic_signal_label() returns correct tier labels
  3. format_forecast_rich() applies correct Rich markup
  4. Edge cases: NaN, infinity, zero, negative values
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from decision.signals_ux import (
    adaptive_format_pct,
    semantic_signal_label,
    format_forecast_rich,
)


class TestAdaptiveFormatPct(unittest.TestCase):
    """Test adaptive precision formatting."""

    def test_small_value_2_decimals(self):
        """Values < 0.1% should show 2 decimal places."""
        result = adaptive_format_pct(0.047)
        self.assertEqual(result, "+0.05%")

    def test_medium_value_1_decimal(self):
        """Values 0.1-1.0% should show 1 decimal place."""
        result = adaptive_format_pct(0.3)
        self.assertEqual(result, "+0.3%")

    def test_large_value_1_decimal(self):
        """Values >= 1.0% should show 1 decimal place."""
        result = adaptive_format_pct(2.3)
        self.assertEqual(result, "+2.3%")

    def test_negative_small(self):
        """Negative small values show 2 decimal places."""
        result = adaptive_format_pct(-0.05)
        self.assertEqual(result, "-0.05%")

    def test_negative_large(self):
        """Negative large values show 1 decimal place."""
        result = adaptive_format_pct(-3.7)
        self.assertEqual(result, "-3.7%")

    def test_zero(self):
        """Zero should format properly."""
        result = adaptive_format_pct(0.0)
        self.assertEqual(result, "+0.00%")

    def test_nan_returns_na(self):
        """NaN should return N/A."""
        result = adaptive_format_pct(float('nan'))
        self.assertEqual(result, "N/A")

    def test_inf_returns_na(self):
        """Infinity should return N/A."""
        result = adaptive_format_pct(float('inf'))
        self.assertEqual(result, "N/A")


class TestSemanticSignalLabel(unittest.TestCase):
    """Test 6-tier semantic labeling."""

    def test_flat(self):
        self.assertEqual(semantic_signal_label(0.01), "FLAT")
        self.assertEqual(semantic_signal_label(-0.01), "FLAT")

    def test_slight(self):
        self.assertEqual(semantic_signal_label(0.05), "SLIGHT")
        self.assertEqual(semantic_signal_label(-0.05), "SLIGHT")

    def test_lean(self):
        self.assertEqual(semantic_signal_label(0.3), "LEAN")
        self.assertEqual(semantic_signal_label(-0.3), "LEAN")

    def test_moderate(self):
        self.assertEqual(semantic_signal_label(1.0), "MODERATE")
        self.assertEqual(semantic_signal_label(-1.5), "MODERATE")

    def test_strong(self):
        self.assertEqual(semantic_signal_label(3.0), "STRONG")
        self.assertEqual(semantic_signal_label(-4.5), "STRONG")

    def test_extreme(self):
        self.assertEqual(semantic_signal_label(7.0), "EXTREME")
        self.assertEqual(semantic_signal_label(-10.0), "EXTREME")

    def test_nan(self):
        self.assertEqual(semantic_signal_label(float('nan')), "N/A")

    def test_boundary_slight_lean(self):
        """0.1% should be LEAN (>= 0.1)."""
        self.assertEqual(semantic_signal_label(0.1), "LEAN")

    def test_boundary_lean_moderate(self):
        """0.5% should be MODERATE (>= 0.5)."""
        self.assertEqual(semantic_signal_label(0.5), "MODERATE")

    def test_boundary_strong_extreme(self):
        """5% should be EXTREME (>= 5)."""
        self.assertEqual(semantic_signal_label(5.0), "EXTREME")


class TestFormatForecastRich(unittest.TestCase):
    """Test Rich-formatted forecast output."""

    def test_positive_flat_is_dim(self):
        """Small positive values should be dim."""
        result = format_forecast_rich(0.01)
        self.assertIn("dim", result)

    def test_positive_moderate_is_bold_green(self):
        """Moderate positive should be bold green."""
        result = format_forecast_rich(1.0)
        self.assertIn("bold", result)
        self.assertIn("00d700", result)

    def test_negative_strong_is_bold_red(self):
        """Strong negative should be bold red."""
        result = format_forecast_rich(-3.0)
        self.assertIn("bold", result)
        self.assertIn("bright_red", result)

    def test_nan_returns_dim_na(self):
        """NaN should return dim N/A."""
        result = format_forecast_rich(float('nan'))
        self.assertIn("N/A", result)
        self.assertIn("dim", result)

    def test_contains_percentage(self):
        """Output should contain % symbol."""
        result = format_forecast_rich(2.5)
        self.assertIn("%", result)

    def test_uses_adaptive_precision(self):
        """Small values should show 2 decimal places in Rich output."""
        result = format_forecast_rich(0.05)
        self.assertIn("0.05%", result)

    def test_combination_0047(self):
        """Story spec: 0.047% -> "+0.05% SLIGHT" equivalent."""
        pct = adaptive_format_pct(0.047)
        label = semantic_signal_label(0.047)
        self.assertEqual(pct, "+0.05%")
        self.assertEqual(label, "SLIGHT")

    def test_combination_2_3(self):
        """Story spec: 2.3% -> "+2.3% STRONG" equivalent."""
        pct = adaptive_format_pct(2.3)
        label = semantic_signal_label(2.3)
        self.assertEqual(pct, "+2.3%")
        self.assertEqual(label, "STRONG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
