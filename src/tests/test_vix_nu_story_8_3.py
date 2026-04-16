"""
Tests for Story 8.3: VIX-Conditional nu (Tail Thickness Coupling).

Tests that vix_conditional_nu() properly adjusts nu based on VIX level,
with higher VIX producing heavier tails (lower nu).
"""
import math
import os
import sys
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.continuous_nu import (
    NU_FLOOR,
    VIX_KAPPA_DEFAULT,
    VIX_MEDIAN_DEFAULT,
    vix_conditional_nu,
)


class TestVixConditionalNuBasic(unittest.TestCase):
    """Basic API and return value tests."""

    def test_returns_float(self):
        """vix_conditional_nu returns a float."""
        result = vix_conditional_nu(8.0, 20.0)
        self.assertIsInstance(result, float)

    def test_finite_result(self):
        """Result is always finite."""
        result = vix_conditional_nu(8.0, 20.0)
        self.assertTrue(math.isfinite(result))

    def test_result_above_nu_min(self):
        """Result is always >= nu_min (2.1)."""
        result = vix_conditional_nu(8.0, 100.0)  # Extreme VIX
        self.assertGreaterEqual(result, NU_FLOOR)


class TestVixConditionalNuFormula(unittest.TestCase):
    """Tests for the VIX-conditional formula."""

    def test_formula_at_median(self):
        """When VIX = median: nu_t = nu_base (no change)."""
        result = vix_conditional_nu(8.0, 18.0, vix_median=18.0)
        self.assertAlmostEqual(result, 8.0, places=6)

    def test_formula_below_median(self):
        """When VIX < median: nu_t = nu_base (no unnecessary fattening)."""
        result = vix_conditional_nu(8.0, 12.0, vix_median=18.0)
        self.assertAlmostEqual(result, 8.0, places=6)

    def test_formula_above_median(self):
        """When VIX > median: nu_t < nu_base."""
        result = vix_conditional_nu(8.0, 25.0, vix_median=18.0)
        self.assertLess(result, 8.0)

    def test_formula_exact_value(self):
        """Check exact formula: nu_t = nu_base - kappa * (VIX - VIX_med)."""
        nu_base = 10.0
        vix = 25.0
        vix_med = 18.0
        kappa = 0.20
        expected = nu_base - kappa * (vix - vix_med)
        result = vix_conditional_nu(nu_base, vix, vix_median=vix_med, kappa=kappa)
        self.assertAlmostEqual(result, expected, places=6)

    def test_monotonic_in_vix(self):
        """Higher VIX -> lower nu (monotonic decrease)."""
        vix_values = [10, 15, 18, 22, 25, 30, 40, 50]
        results = [vix_conditional_nu(8.0, v) for v in vix_values]
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i], results[i + 1],
                                    f"VIX={vix_values[i]}: {results[i]} < VIX={vix_values[i+1]}: {results[i+1]}")


class TestVixConditionalNuAcceptance(unittest.TestCase):
    """Tests matching acceptance criteria from Tune.md."""

    def test_vix_above_30_drops_by_2(self):
        """AC: When VIX > 30, nu_t drops by at least 2."""
        nu_base = 8.0
        result = vix_conditional_nu(nu_base, 30.0)
        drop = nu_base - result
        self.assertGreaterEqual(drop, 2.0,
                                f"Drop at VIX=30: {drop:.2f} < 2.0")

    def test_vix_above_30_drops_by_2_various_bases(self):
        """Drop >= 2 at VIX=30 for various nu_base values."""
        for nu_base in [6.0, 8.0, 10.0, 15.0, 20.0]:
            result = vix_conditional_nu(nu_base, 30.0)
            drop = nu_base - result
            self.assertGreaterEqual(drop, 2.0,
                                    f"nu_base={nu_base}: drop={drop:.2f} < 2.0")

    def test_vix_below_15_unchanged(self):
        """AC: When VIX < 15, nu_t unchanged."""
        nu_base = 8.0
        result = vix_conditional_nu(nu_base, 14.0)
        self.assertAlmostEqual(result, nu_base, places=6,
                               msg="VIX=14 should not change nu")

    def test_vix_at_10_unchanged(self):
        """Very low VIX: no change."""
        nu_base = 8.0
        result = vix_conditional_nu(nu_base, 10.0)
        self.assertAlmostEqual(result, nu_base, places=6)


class TestVixConditionalNuFloor(unittest.TestCase):
    """Tests for nu floor behavior."""

    def test_floor_at_extreme_vix(self):
        """Extreme VIX (80): nu doesn't go below floor."""
        result = vix_conditional_nu(8.0, 80.0)
        self.assertGreaterEqual(result, NU_FLOOR)

    def test_floor_with_low_nu_base(self):
        """Low nu_base + high VIX: floor prevents sub-2.1."""
        result = vix_conditional_nu(3.0, 40.0)
        self.assertGreaterEqual(result, NU_FLOOR)

    def test_custom_nu_min(self):
        """Custom nu_min is respected."""
        result = vix_conditional_nu(8.0, 80.0, nu_min=3.0)
        self.assertGreaterEqual(result, 3.0)

    def test_floor_value(self):
        """NU_FLOOR is 2.1 (> 2 for finite variance)."""
        self.assertEqual(NU_FLOOR, 2.1)


class TestVixConditionalNuKappa(unittest.TestCase):
    """Tests for kappa sensitivity parameter."""

    def test_higher_kappa_more_drop(self):
        """Higher kappa produces larger nu drop."""
        low_kappa = vix_conditional_nu(8.0, 30.0, kappa=0.10)
        high_kappa = vix_conditional_nu(8.0, 30.0, kappa=0.30)
        self.assertGreater(low_kappa, high_kappa)

    def test_zero_kappa_no_change(self):
        """kappa=0: no change regardless of VIX."""
        result = vix_conditional_nu(8.0, 50.0, kappa=0.0)
        self.assertAlmostEqual(result, 8.0, places=6)

    def test_default_kappa(self):
        """Default kappa gives meaningful drop at VIX=30."""
        result = vix_conditional_nu(8.0, 30.0)
        self.assertLess(result, 8.0)
        self.assertGreater(result, NU_FLOOR)


class TestVixConditionalNuVixMedian(unittest.TestCase):
    """Tests for custom VIX median."""

    def test_custom_median_higher(self):
        """Higher median: same VIX produces less drop."""
        low_med = vix_conditional_nu(8.0, 25.0, vix_median=15.0)
        high_med = vix_conditional_nu(8.0, 25.0, vix_median=22.0)
        self.assertLess(low_med, high_med)

    def test_vix_equals_custom_median(self):
        """VIX = custom median: no change."""
        result = vix_conditional_nu(8.0, 25.0, vix_median=25.0)
        self.assertAlmostEqual(result, 8.0, places=6)

    def test_default_median_is_18(self):
        """Default VIX median is 18."""
        self.assertEqual(VIX_MEDIAN_DEFAULT, 18.0)


class TestVixConditionalNuEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_vix_zero(self):
        """VIX = 0: no change (below median)."""
        result = vix_conditional_nu(8.0, 0.0)
        self.assertAlmostEqual(result, 8.0, places=6)

    def test_very_high_nu_base(self):
        """Very high nu_base (Gaussian-like): still adjusts."""
        result = vix_conditional_nu(50.0, 30.0)
        self.assertLess(result, 50.0)

    def test_nu_base_at_floor(self):
        """nu_base at floor + any VIX: stays at floor."""
        result = vix_conditional_nu(NU_FLOOR, 30.0)
        self.assertAlmostEqual(result, NU_FLOOR, places=6)


if __name__ == '__main__':
    unittest.main()
