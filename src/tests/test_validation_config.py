"""
Story 8.1: Test that validation config is correct and importable.
"""
import unittest
import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestValidationConfig(unittest.TestCase):
    """Validate the validation universe configuration."""

    def test_universe_has_12_symbols(self):
        from tests.validation_config import VALIDATION_UNIVERSE
        self.assertEqual(len(VALIDATION_UNIVERSE), 12)

    def test_all_symbols_have_required_fields(self):
        from tests.validation_config import VALIDATION_UNIVERSE
        for sym, meta in VALIDATION_UNIVERSE.items():
            self.assertIn("sector", meta, f"{sym} missing sector")
            self.assertIn("expected_vol", meta, f"{sym} missing expected_vol")
            self.assertIn("class", meta, f"{sym} missing class")
            self.assertGreater(meta["expected_vol"], 0, f"{sym} vol <= 0")

    def test_profitability_targets_quantitative(self):
        from tests.validation_config import PROFITABILITY_TARGETS
        self.assertGreater(len(PROFITABILITY_TARGETS), 5)
        for key, val in PROFITABILITY_TARGETS.items():
            self.assertIsInstance(val, (int, float), f"{key} is not numeric")

    def test_methodology_defined_for_all_targets(self):
        from tests.validation_config import PROFITABILITY_TARGETS, METRIC_METHODOLOGY
        for key in PROFITABILITY_TARGETS:
            self.assertIn(key, METRIC_METHODOLOGY,
                f"No methodology defined for {key}")

    def test_baseline_json_exists(self):
        path = os.path.join(SCRIPT_DIR, "validation_baseline.json")
        self.assertTrue(os.path.isfile(path), "validation_baseline.json missing")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("hit_rate_7d", data)
        self.assertIn("hit_rate_21d", data)

    def test_symbol_classes_partition(self):
        from tests.validation_config import (
            VALIDATION_UNIVERSE, EQUITY_SYMBOLS, ETF_SYMBOLS, METAL_SYMBOLS,
        )
        all_syms = set(EQUITY_SYMBOLS + ETF_SYMBOLS + METAL_SYMBOLS)
        self.assertEqual(all_syms, set(VALIDATION_UNIVERSE.keys()))

    def test_config_importable_from_all_contexts(self):
        from tests.validation_config import (
            VALIDATION_UNIVERSE,
            PROFITABILITY_TARGETS,
            VALIDATION_HORIZONS,
            HIGH_VOL_SYMBOLS,
            LOW_VOL_SYMBOLS,
        )
        self.assertEqual(len(VALIDATION_HORIZONS), 7)
        self.assertGreater(len(HIGH_VOL_SYMBOLS), 0)
        self.assertGreater(len(LOW_VOL_SYMBOLS), 0)


if __name__ == "__main__":
    unittest.main()
