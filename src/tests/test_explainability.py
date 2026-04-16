"""
Test Story 2.9: Ensemble Model Explainability Output.

Validates:
  1. Per-horizon explanation exists after forecast
  2. 5 positive models -> bullish explanation
  3. Mixed models -> dissenter detected
  4. Template-based reason strings
  5. Contributions sum to ensemble
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest

from decision.market_temperature import (
    ensemble_forecast,
    get_forecast_explanation,
    _build_explanation,
    MODEL_NAMES,
    EXPLAIN_DRIVER_TEMPLATE,
    EXPLAIN_DISSENT_TEMPLATE,
)


class TestExplainability(unittest.TestCase):
    """Tests for model explainability output."""

    def test_explanation_exists(self):
        """Explanation cache populated after ensemble_forecast."""
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 400))))
        ensemble_forecast(prices, asset_type="equity", asset_name="EXPLAIN_TEST")
        
        explain = get_forecast_explanation("EXPLAIN_TEST")
        self.assertIsNotNone(explain)
        self.assertIn("horizons", explain)
        self.assertIn("explanations", explain)
        self.assertEqual(len(explain["explanations"]), 7)

    def test_explanation_structure(self):
        """Each horizon explanation has required fields."""
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 400))))
        ensemble_forecast(prices, asset_type="equity", asset_name="STRUCT_EX")
        
        explain = get_forecast_explanation("STRUCT_EX")
        for ex in explain["explanations"]:
            self.assertIn("model_forecasts", ex)
            self.assertIn("weights", ex)
            self.assertIn("contributions", ex)
            self.assertIn("top_contributor", ex)
            self.assertIn("reason", ex)
            self.assertEqual(len(ex["model_forecasts"]), 5)
            self.assertEqual(len(ex["weights"]), 5)

    def test_all_positive_bullish(self):
        """All positive forecasts -> bullish explanation."""
        forecasts = [1.0, 2.0, 0.5, 1.5, 0.8]
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        ensemble = sum(w * f for w, f in zip(weights, forecasts))
        
        ex = _build_explanation(forecasts, weights, ensemble)
        self.assertIn("bullish", ex["reason"])
        self.assertIsNone(ex["top_dissenter"])

    def test_mixed_has_dissenter(self):
        """3 positive + 2 negative -> dissenter detected."""
        forecasts = [2.0, 1.0, -1.5, 0.5, -0.8]
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]
        ensemble = sum(w * f for w, f in zip(weights, forecasts))
        
        ex = _build_explanation(forecasts, weights, ensemble)
        self.assertIsNotNone(ex["top_dissenter"])
        self.assertIn("opposes consensus", ex["reason"])

    def test_all_negative_bearish(self):
        """All negative forecasts -> bearish explanation."""
        forecasts = [-1.0, -2.0, -0.5, -1.5, -0.8]
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        ensemble = sum(w * f for w, f in zip(weights, forecasts))
        
        ex = _build_explanation(forecasts, weights, ensemble)
        self.assertIn("bearish", ex["reason"])

    def test_contributions_sum_close_to_ensemble(self):
        """Sum of contributions should equal ensemble."""
        forecasts = [1.0, -0.5, 2.0, 0.3, -1.0]
        weights = [0.3, 0.1, 0.25, 0.2, 0.15]
        ensemble = sum(w * f for w, f in zip(weights, forecasts))
        
        ex = _build_explanation(forecasts, weights, ensemble)
        self.assertAlmostEqual(sum(ex["contributions"]), ensemble, places=6)

    def test_top_contributor_is_model_name(self):
        """Top contributor should be one of the model names."""
        forecasts = [0.1, 0.2, 5.0, 0.3, 0.1]
        weights = [0.1, 0.1, 0.5, 0.2, 0.1]
        ensemble = sum(w * f for w, f in zip(weights, forecasts))
        
        ex = _build_explanation(forecasts, weights, ensemble)
        self.assertIn(ex["top_contributor"], MODEL_NAMES)

    def test_nonexistent_asset_returns_none(self):
        """Unknown asset returns None."""
        self.assertIsNone(get_forecast_explanation("NONEXISTENT_ASSET_XYZ"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
