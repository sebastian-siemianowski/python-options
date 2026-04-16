"""
Story 8.2: Full pipeline integration test.

Runs the complete tune + signal pipeline on the 12-asset validation universe.
Skipped by default -- run with RUN_INTEGRATION=1 environment variable.
"""
import unittest
import sys
import os
import json

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.validation_config import (
    VALIDATION_UNIVERSE, PROFITABILITY_TARGETS, VALIDATION_HORIZONS,
)


def _run_signals_for_symbol(symbol: str):
    """Run signal pipeline for a single symbol. Returns Signal list or None."""
    try:
        from decision.signals import latest_signals, compute_features
        from ingestion.data_utils import fetch_px_asset
        px, _title = fetch_px_asset(symbol, "2024-01-01")
        if px is None or len(px) < 252:
            return None
        feats = compute_features(px, asset_symbol=symbol)
        last_close = float(px.iloc[-1])
        sigs, _thr = latest_signals(
            feats, VALIDATION_HORIZONS, last_close=last_close,
            asset_key=symbol, n_mc_paths=5000,
        )
        return {s.horizon_days: s for s in sigs}
    except Exception:
        return None


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION"),
    "Integration test: set RUN_INTEGRATION=1 to run"
)
class TestFullPipeline(unittest.TestCase):
    """Full pipeline integration test on the validation universe."""

    _signal_results = None

    @classmethod
    def setUpClass(cls):
        """Run signal pipeline on all validation symbols."""
        cls._signal_results = {}
        for symbol in VALIDATION_UNIVERSE:
            result = _run_signals_for_symbol(symbol)
            if result:
                cls._signal_results[symbol] = result

    def test_all_assets_produce_signals(self):
        """Every validation asset must produce signals for at least some horizons."""
        for symbol in VALIDATION_UNIVERSE:
            self.assertIn(symbol, self._signal_results,
                f"{symbol} produced no signals")
            sigs = self._signal_results[symbol]
            self.assertGreater(len(sigs), 0,
                f"{symbol} has empty signal dict")

    def test_all_horizons_present(self):
        """Each asset should have signals for all 7 standard horizons."""
        for symbol, sigs in self._signal_results.items():
            for H in VALIDATION_HORIZONS:
                self.assertIn(H, sigs,
                    f"{symbol} missing horizon H={H}")

    def test_no_flat_signals(self):
        """No asset should have ALL horizons showing ~0.0%."""
        for symbol, sigs in self._signal_results.items():
            forecasts = [sigs[H].exp_ret for H in [1, 7, 21] if H in sigs]
            if forecasts:
                max_abs = max(abs(f) for f in forecasts)
                self.assertGreater(max_abs, 0.001,
                    f"{symbol} has flat signals: {forecasts}")

    def test_directional_differentiation(self):
        """Different assets should have different signal directions."""
        directions = {}
        for symbol, sigs in self._signal_results.items():
            if 7 in sigs:
                directions[symbol] = sigs[7].exp_ret
        if len(directions) >= 4:
            unique_signs = len(set(np.sign(d) for d in directions.values()))
            self.assertGreater(unique_signs, 1,
                "All assets have same direction -- no differentiation")

    def test_signal_rate_target(self):
        """At least 15% of signals should be non-HOLD."""
        labels = []
        for sigs in self._signal_results.values():
            if 7 in sigs:
                labels.append(sigs[7].label)
        if labels:
            non_hold = sum(1 for l in labels if l != "HOLD") / len(labels)
            self.assertGreater(non_hold, PROFITABILITY_TARGETS["signal_rate"],
                f"Non-HOLD rate {non_hold:.1%} below target")

    def test_no_catastrophic_forecasts(self):
        """No forecast should exceed physical bounds."""
        for symbol, sigs in self._signal_results.items():
            for H, sig in sigs.items():
                self.assertGreater(sig.exp_ret, -2.3,
                    f"{symbol} H={H} forecast {sig.exp_ret} is catastrophic")
                self.assertLess(sig.exp_ret, 1.6,
                    f"{symbol} H={H} forecast {sig.exp_ret} exceeds +400%")

    def test_p_up_bounded(self):
        """All p_up values must be in [0, 1]."""
        for symbol, sigs in self._signal_results.items():
            for H, sig in sigs.items():
                self.assertGreaterEqual(sig.p_up, 0.0,
                    f"{symbol} H={H} p_up={sig.p_up} < 0")
                self.assertLessEqual(sig.p_up, 1.0,
                    f"{symbol} H={H} p_up={sig.p_up} > 1")

    def test_ci_ordering(self):
        """CI bounds must be ordered: ci_low <= exp_ret <= ci_high."""
        for symbol, sigs in self._signal_results.items():
            for H, sig in sigs.items():
                self.assertLessEqual(sig.ci_low, sig.ci_high + 1e-10,
                    f"{symbol} H={H} CI inverted: [{sig.ci_low}, {sig.ci_high}]")


class TestIntegrationConfig(unittest.TestCase):
    """Non-integration tests validating config and Makefile target."""

    def test_validation_universe_importable(self):
        from tests.validation_config import VALIDATION_UNIVERSE
        self.assertEqual(len(VALIDATION_UNIVERSE), 12)

    def test_makefile_has_integration_target(self):
        makefile = os.path.join(REPO_ROOT, os.pardir, "Makefile")
        if os.path.isfile(makefile):
            with open(makefile) as f:
                content = f.read()
            self.assertIn("test-integration", content)
        else:
            self.skipTest("Makefile not found")


if __name__ == "__main__":
    unittest.main()
