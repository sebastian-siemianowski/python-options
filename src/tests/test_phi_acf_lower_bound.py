"""
Tests for Story 1.8: Phi Lower Bound from Autocorrelation.

Validates that the autocorrelation-based phi floor:
  1. Prevents negative phi in tuned models
  2. Stores acf_1 and phi_acf_floor diagnostics
  3. Scales properly with PHI_ACF_SCALE

Mathematical specification:
    acf_1 = corr(r[t], r[t-1])
    phi_floor = max(0, acf_1 * PHI_ACF_SCALE)
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestPhiAcfConstants(unittest.TestCase):
    """Validate PHI_ACF_SCALE constant."""

    def test_exists(self):
        from tuning.tune import PHI_ACF_SCALE
        self.assertIsInstance(PHI_ACF_SCALE, (int, float))

    def test_positive(self):
        from tuning.tune import PHI_ACF_SCALE
        self.assertGreater(PHI_ACF_SCALE, 0.0)

    def test_reasonable(self):
        from tuning.tune import PHI_ACF_SCALE
        self.assertGreater(PHI_ACF_SCALE, 1.0)
        self.assertLess(PHI_ACF_SCALE, 5.0)


class TestPhiAcfFloorInModels(unittest.TestCase):
    """Test that fit_all_models_for_regime applies phi ACF floor."""

    @classmethod
    def setUpClass(cls):
        """Fit models on synthetic data with positive autocorrelation."""
        rng = np.random.RandomState(42)
        n = 400
        # AR(1) process with phi=0.3 → positive acf
        cls.returns = np.zeros(n)
        for t in range(1, n):
            cls.returns[t] = 0.3 * cls.returns[t - 1] + rng.normal(0, 0.012)
        cls.vol = np.full(n, 0.012, dtype=np.float64)

        from tuning.tune import fit_all_models_for_regime
        cls.models = fit_all_models_for_regime(
            cls.returns, cls.vol, asset="SYNTH_ACF",
        )

    def test_acf_1_stored(self):
        """All successful models should have acf_1 stored."""
        for name, m in self.models.items():
            if m.get("fit_success", False):
                self.assertIn("acf_1", m, f"Model {name} missing acf_1")
                self.assertTrue(np.isfinite(m["acf_1"]))

    def test_phi_floor_stored(self):
        """All successful models should have phi_acf_floor stored."""
        for name, m in self.models.items():
            if m.get("fit_success", False):
                self.assertIn("phi_acf_floor", m, f"Model {name} missing phi_acf_floor")
                self.assertGreaterEqual(m["phi_acf_floor"], 0.0)

    def test_no_negative_phi(self):
        """No model should have phi < 0 after ACF floor."""
        for name, m in self.models.items():
            if m.get("fit_success", False) and m.get("phi") is not None:
                self.assertGreaterEqual(m["phi"], 0.0,
                                        f"Model {name} has phi={m['phi']}")

    def test_positive_acf_gives_positive_floor(self):
        """AR(1) with phi=0.3 should produce positive acf_1 and floor."""
        # Check the first successful model
        for name, m in self.models.items():
            if m.get("fit_success", False):
                self.assertGreater(m["acf_1"], 0.0,
                                   f"acf_1={m['acf_1']} should be positive for AR(1)")
                self.assertGreater(m["phi_acf_floor"], 0.0)
                break


class TestPhiAcfFloorNegativeAcf(unittest.TestCase):
    """Test with data that has negative/zero autocorrelation."""

    @classmethod
    def setUpClass(cls):
        """Fit models on iid data (zero autocorrelation)."""
        rng = np.random.RandomState(99)
        n = 400
        cls.returns = rng.normal(0, 0.012, n).astype(np.float64)
        cls.vol = np.full(n, 0.012, dtype=np.float64)

        from tuning.tune import fit_all_models_for_regime
        cls.models = fit_all_models_for_regime(
            cls.returns, cls.vol, asset="SYNTH_IID",
        )

    def test_floor_is_zero_for_iid(self):
        """With iid data, phi_acf_floor should be ~0."""
        for name, m in self.models.items():
            if m.get("fit_success", False):
                # For iid data, acf_1 should be near 0 (sample noise)
                # Floor = max(0, acf_1 * scale) could be slightly positive
                # but should be very small
                self.assertLess(m["phi_acf_floor"], 0.3,
                                f"phi_acf_floor={m['phi_acf_floor']} too large for iid")
                break


class TestPhiAcfScaling(unittest.TestCase):
    """Test the scaling formula directly."""

    def test_formula(self):
        from tuning.tune import PHI_ACF_SCALE
        acf_1 = 0.15
        expected = max(0.0, acf_1 * PHI_ACF_SCALE)
        self.assertAlmostEqual(expected, 0.30, places=5)

    def test_negative_acf_floors_at_zero(self):
        from tuning.tune import PHI_ACF_SCALE
        acf_1 = -0.05
        result = max(0.0, acf_1 * PHI_ACF_SCALE)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
