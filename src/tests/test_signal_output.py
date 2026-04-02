"""
Test Story 4.1: Unified Signal Output Contract.

Validates:
  1. Round-trip serialization (dataclass -> JSON -> dataclass)
  2. HorizonForecast sub-dataclass
  3. to_rich_row formatting
  4. from_dict / to_dict
  5. Missing horizon graceful handling
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import unittest

from decision.signal_output import SignalOutput, HorizonForecast


class TestSignalOutput(unittest.TestCase):
    """Tests for unified signal output contract."""

    def _make_signal(self):
        hf = {
            7: HorizonForecast(
                horizon_days=7,
                point_forecast_pct=2.5,
                p10=-3.0, p25=-1.0, p50=2.0, p75=5.0, p90=8.0,
                direction_label="bullish",
                confidence_score=0.72,
                model_breakdown={"Kalman": 1.2, "GARCH": 0.8, "OU": 0.5},
            ),
            30: HorizonForecast(
                horizon_days=30,
                point_forecast_pct=-1.3,
                direction_label="bearish",
            ),
        }
        return SignalOutput(
            symbol="AAPL",
            sector="Technology",
            crash_risk=0.15,
            momentum=0.32,
            horizon_forecasts=hf,
            confidence=0.72,
            regime="LOW_VOL_TREND",
            model_explanation="Kalman dominates (48%), GARCH agrees",
            generated_at="2026-02-15T10:30:00",
            data_version="abc123",
        )

    def test_roundtrip_json(self):
        """dataclass -> JSON -> dataclass preserves all fields."""
        signal = self._make_signal()
        json_str = signal.to_json()
        restored = SignalOutput.from_json(json_str)
        self.assertEqual(restored.symbol, "AAPL")
        self.assertEqual(restored.sector, "Technology")
        self.assertAlmostEqual(restored.crash_risk, 0.15)
        self.assertEqual(len(restored.horizon_forecasts), 2)
        hf7 = restored.horizon_forecasts[7]
        self.assertAlmostEqual(hf7.point_forecast_pct, 2.5)
        self.assertEqual(hf7.direction_label, "bullish")

    def test_roundtrip_dict(self):
        """dataclass -> dict -> dataclass."""
        signal = self._make_signal()
        d = signal.to_dict()
        restored = SignalOutput.from_dict(d)
        self.assertEqual(restored.symbol, "AAPL")
        self.assertAlmostEqual(restored.horizon_forecasts[7].p90, 8.0)

    def test_horizon_forecast_from_dict(self):
        """HorizonForecast.from_dict ignores unknown keys."""
        d = {"horizon_days": 7, "point_forecast_pct": 1.0, "extra_key": "ignored"}
        hf = HorizonForecast.from_dict(d)
        self.assertEqual(hf.horizon_days, 7)
        self.assertAlmostEqual(hf.point_forecast_pct, 1.0)

    def test_to_rich_row(self):
        """Rich row has correct format."""
        signal = self._make_signal()
        row = signal.to_rich_row()
        self.assertEqual(row[0], "AAPL")
        self.assertEqual(row[1], "Technology")
        self.assertIn("+2.50%", row[5])  # 7d forecast
        self.assertIn("-1.30%", row[6])  # 30d forecast
        self.assertEqual(row[7], "--")   # 90d missing

    def test_missing_horizon(self):
        """get_direction_label returns neutral for missing horizon."""
        signal = self._make_signal()
        self.assertEqual(signal.get_direction_label(365), "neutral")

    def test_empty_signal(self):
        """Minimal signal serializes correctly."""
        signal = SignalOutput(symbol="TEST")
        json_str = signal.to_json()
        restored = SignalOutput.from_json(json_str)
        self.assertEqual(restored.symbol, "TEST")
        self.assertEqual(len(restored.horizon_forecasts), 0)

    def test_json_valid(self):
        """to_json produces valid JSON."""
        signal = self._make_signal()
        parsed = json.loads(signal.to_json())
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["symbol"], "AAPL")

    def test_model_breakdown_preserved(self):
        """Model breakdown dict preserved through serialization."""
        signal = self._make_signal()
        restored = SignalOutput.from_json(signal.to_json())
        breakdown = restored.horizon_forecasts[7].model_breakdown
        self.assertAlmostEqual(breakdown["Kalman"], 1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
