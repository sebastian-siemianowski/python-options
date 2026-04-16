"""Tests for Story 3.6: MC Path Diagnostics and Anomaly Detection.

Validates that:
1. NaN paths are counted and excluded from statistics
2. Extreme paths (|return| > threshold) are flagged
3. Trimmed mean is used when extreme fraction > 10%
4. Warning is logged for anomalous MC runs
5. MCDiagnostics.to_dict() produces valid JSON-serializable output
6. Edge cases: empty array, all NaN, single path
"""
import os
import sys
import unittest
import warnings
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from decision.signals import (
    diagnose_mc_paths,
    MCDiagnostics,
    MC_EXTREME_THRESHOLD,
    MC_EXTREME_WARN_FRAC,
    MC_TRIM_PERCENTILE,
)


class TestMCDiagnosticsConstants(unittest.TestCase):
    """Test constants are sensible."""

    def test_extreme_threshold(self):
        self.assertAlmostEqual(MC_EXTREME_THRESHOLD, 5.0)

    def test_warn_frac(self):
        self.assertAlmostEqual(MC_EXTREME_WARN_FRAC, 0.10)

    def test_trim_percentile(self):
        self.assertAlmostEqual(MC_TRIM_PERCENTILE, 2.5)


class TestMCDiagnosticsCleanData(unittest.TestCase):
    """Test diagnostics on clean (normal) MC paths."""

    def test_no_nan_no_extreme(self):
        rng = np.random.default_rng(42)
        cum = rng.normal(0, 0.1, size=(10, 5000))
        d = diagnose_mc_paths(cum, H=10)
        self.assertEqual(d.n_nan, 0)
        self.assertEqual(d.n_extreme, 0)
        self.assertAlmostEqual(d.extreme_frac, 0.0)
        self.assertFalse(d.used_trimmed)

    def test_median_and_mean_close(self):
        rng = np.random.default_rng(42)
        cum = rng.normal(0, 0.01, size=(5, 10000))
        d = diagnose_mc_paths(cum, H=5)
        # For symmetric normal, mean and median should be close
        self.assertAlmostEqual(d.mean, d.median, delta=0.01)

    def test_n_paths_correct(self):
        cum = np.zeros((3, 200))
        d = diagnose_mc_paths(cum, H=3)
        self.assertEqual(d.n_paths, 200)


class TestMCDiagnosticsNaN(unittest.TestCase):
    """Test NaN path handling."""

    def test_nan_detected(self):
        cum = np.zeros((5, 100))
        cum[4, 10:15] = np.nan
        d = diagnose_mc_paths(cum, H=5)
        self.assertEqual(d.n_nan, 5)

    def test_inf_detected(self):
        cum = np.zeros((5, 100))
        cum[4, 0] = np.inf
        cum[4, 1] = -np.inf
        d = diagnose_mc_paths(cum, H=5)
        self.assertEqual(d.n_nan, 2)

    def test_nan_excluded_from_mean(self):
        cum = np.ones((3, 100)) * 0.05
        cum[2, 0:5] = np.nan
        d = diagnose_mc_paths(cum, H=3)
        # Mean should be 0.05 (NaN excluded)
        self.assertAlmostEqual(d.mean, 0.05, delta=1e-10)

    def test_all_nan_warning(self):
        cum = np.full((3, 50), np.nan)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = diagnose_mc_paths(cum, H=3, asset_name="TEST_NAN")
            nan_warnings = [x for x in w if "ALL" in str(x.message) and "NaN" in str(x.message)]
            self.assertTrue(len(nan_warnings) > 0, "Should warn when all paths are NaN")
        self.assertEqual(d.n_nan, 50)

    def test_partial_nan_warning(self):
        cum = np.zeros((3, 100))
        cum[2, :10] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = diagnose_mc_paths(cum, H=3, asset_name="TEST_PARTIAL")
            nan_warnings = [x for x in w if "NaN" in str(x.message) and "excluded" in str(x.message)]
            self.assertTrue(len(nan_warnings) > 0, "Should warn about NaN paths")


class TestMCDiagnosticsExtremePaths(unittest.TestCase):
    """Test extreme path detection and trimming."""

    def test_extreme_paths_counted(self):
        cum = np.zeros((5, 100))
        # 15 paths with extreme returns (|r| > 5.0)
        cum[4, :15] = 6.0  # >500% return
        d = diagnose_mc_paths(cum, H=5)
        self.assertEqual(d.n_extreme, 15)
        self.assertAlmostEqual(d.extreme_frac, 0.15)

    def test_trimmed_used_above_threshold(self):
        """When >10% extreme, trimmed mean should be used."""
        rng = np.random.default_rng(99)
        cum = np.zeros((5, 1000))
        cum[4, :] = rng.normal(0.05, 0.01, 1000)  # Normal returns
        # 12% extreme with VARYING magnitudes (some above 97.5th pctl)
        cum[4, :120] = rng.uniform(5.5, 15.0, 120)  # Varying extreme values
        d = diagnose_mc_paths(cum, H=5)
        self.assertTrue(d.used_trimmed, "Should use trimmed when >10% extreme")
        # Trimmed mean should be lower than raw mean (excludes top outliers)
        self.assertLess(d.trimmed_mean, d.mean)

    def test_trimmed_not_used_below_threshold(self):
        """When <10% extreme, raw mean is used."""
        cum = np.zeros((5, 1000))
        cum[4, :] = 0.05
        cum[4, :5] = 6.0  # Only 0.5% extreme
        d = diagnose_mc_paths(cum, H=5)
        self.assertFalse(d.used_trimmed)
        self.assertAlmostEqual(d.trimmed_mean, d.mean)

    def test_extreme_warning_logged(self):
        cum = np.zeros((5, 100))
        cum[4, :15] = 7.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = diagnose_mc_paths(cum, H=5, asset_name="EXTREME_TEST")
            extreme_warnings = [x for x in w if "extreme" in str(x.message).lower()]
            self.assertTrue(len(extreme_warnings) > 0, "Should warn about extreme paths")

    def test_negative_extreme_detected(self):
        cum = np.zeros((3, 50))
        cum[2, :5] = -6.0  # Extreme negative
        d = diagnose_mc_paths(cum, H=3)
        self.assertEqual(d.n_extreme, 5)


class TestMCDiagnosticsEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_invalid_horizon(self):
        cum = np.zeros((5, 100))
        d = diagnose_mc_paths(cum, H=0)
        self.assertEqual(d.n_paths, 0)

    def test_horizon_exceeds_array(self):
        cum = np.zeros((5, 100))
        d = diagnose_mc_paths(cum, H=10)
        self.assertEqual(d.n_paths, 0)

    def test_single_path(self):
        cum = np.array([[0.05]])
        d = diagnose_mc_paths(cum, H=1)
        self.assertEqual(d.n_paths, 1)
        self.assertAlmostEqual(d.mean, 0.05)

    def test_mean_median_gap_computed(self):
        rng = np.random.default_rng(77)
        # Skewed distribution has large mean-median gap
        cum = np.zeros((3, 1000))
        cum[2, :] = rng.exponential(0.1, 1000)  # Right-skewed
        d = diagnose_mc_paths(cum, H=3)
        self.assertGreater(d.mean_median_gap, 0.0)


class TestMCDiagnosticsToDict(unittest.TestCase):
    """Test serialization."""

    def test_to_dict_all_keys(self):
        cum = np.zeros((3, 100))
        d = diagnose_mc_paths(cum, H=3)
        result = d.to_dict()
        expected_keys = {"n_paths", "n_nan", "n_extreme", "extreme_frac",
                         "median", "mean", "trimmed_mean", "mean_median_gap",
                         "used_trimmed"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_to_dict_json_serializable(self):
        cum = np.zeros((3, 100))
        cum[2, :5] = np.nan
        cum[2, 5:10] = 7.0
        d = diagnose_mc_paths(cum, H=3)
        # Should serialize without errors
        json_str = json.dumps(d.to_dict())
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)

    def test_to_dict_values_finite(self):
        rng = np.random.default_rng(42)
        cum = rng.normal(0, 0.1, size=(5, 1000))
        d = diagnose_mc_paths(cum, H=5)
        for k, v in d.to_dict().items():
            if isinstance(v, (int, float)):
                self.assertTrue(np.isfinite(v), f"Key {k} is not finite: {v}")


class TestSignalMCDiagnosticsField(unittest.TestCase):
    """Test that mc_diagnostics field exists on Signal."""

    def test_signal_has_mc_diagnostics_field(self):
        from decision.signals import Signal
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Signal)}
        self.assertIn("mc_diagnostics", fields)

    def test_mc_diagnostics_default_none(self):
        from decision.signals import Signal
        import dataclasses
        for f in dataclasses.fields(Signal):
            if f.name == "mc_diagnostics":
                self.assertIsNone(f.default)


if __name__ == "__main__":
    unittest.main()
