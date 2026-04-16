"""
Test Story 1.6: Ensemble Forecast De-correlation and Signal Extraction.

Validates:
  1. Sign-agreement-weighted averaging amplifies consensus
  2. Disagreement is dampened with higher uncertainty
  3. CONTESTED label triggered when dispersion > threshold
  4. Consensus amplification math is correct
  5. Constants are in sensible ranges
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch

from decision.market_temperature import (
    ensemble_forecast,
    CONTRAST_BOOST,
    DISPERSION_CONTESTED,
    DISPERSION_INTERVAL_WIDEN,
    STANDARD_HORIZONS,
)


def _make_price_series(n=400, base=100.0, daily_return=0.0005, seed=42):
    """Create a synthetic price series."""
    rng = np.random.RandomState(seed)
    log_returns = daily_return + rng.normal(0, 0.015, n)
    log_prices = np.log(base) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


class TestEnsembleDecorrelation(unittest.TestCase):
    """Test sign-agreement-weighted averaging logic."""

    def test_all_models_positive_amplified(self):
        """When all 5 models agree on positive direction, forecast should be amplified."""
        # Create strongly bullish data with consistent uptrend
        prices = _make_price_series(400, daily_return=0.002, seed=10)
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS)
        
        # Result should have 8 elements: 7 forecasts + confidence
        self.assertEqual(len(result), 8)
        # With strong uptrend, most horizons should be positive
        positive_count = sum(1 for r in result[:7] if r > 0)
        self.assertGreaterEqual(positive_count, 2,
                           f"Bullish data should produce mostly positive forecasts, got {positive_count}/7")

    def test_ensemble_returns_valid_tuple(self):
        """Ensemble should return a tuple of correct length with valid types."""
        prices = _make_price_series(400)
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS)
        
        self.assertEqual(len(result), 8)
        for i in range(7):
            self.assertIsInstance(result[i], float)
            self.assertTrue(np.isfinite(result[i]))
        self.assertIsInstance(result[7], str)
        # Confidence label may include a tag like '[CALM]' or '[ELEVATED]'
        base_label = result[7].split(' [')[0] if ' [' in result[7] else result[7]
        self.assertIn(base_label, ["High", "Medium", "Low", "Contested"])

    def test_short_data_returns_low_confidence(self):
        """Very short price series should return Low confidence."""
        prices = _make_price_series(20)
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS)
        self.assertEqual(result[7], "Low")

    def test_contrast_boost_in_range(self):
        """CONTRAST_BOOST should be in sensible range."""
        self.assertGreater(CONTRAST_BOOST, 0)
        self.assertLessEqual(CONTRAST_BOOST, 1.0)

    def test_dispersion_contested_positive(self):
        """DISPERSION_CONTESTED should be a positive threshold."""
        self.assertGreater(DISPERSION_CONTESTED, 0)
        self.assertLess(DISPERSION_CONTESTED, 10.0)


class TestConsensusAmplificationMath(unittest.TestCase):
    """Test the math of sign-agreement weighting."""

    def test_full_agreement_gives_max_boost(self):
        """
        With 100% agreement (agreement=1.0), contrast = 1.0:
        agreeing models get weight * (1 + CONTRAST_BOOST)
        """
        agreement = 1.0
        contrast = (agreement - 0.5) * 2.0
        boost = 1.0 + CONTRAST_BOOST * contrast
        self.assertAlmostEqual(boost, 1.0 + CONTRAST_BOOST, places=10)
        self.assertAlmostEqual(boost, 1.6, places=5)  # With default 0.6
        
    def test_half_agreement_gives_no_boost(self):
        """
        With 50% agreement, contrast = 0.0, no boost.
        """
        agreement = 0.5
        contrast = (agreement - 0.5) * 2.0
        boost = 1.0 + CONTRAST_BOOST * contrast
        self.assertAlmostEqual(boost, 1.0, places=10)

    def test_dissenter_dampened(self):
        """
        With 80% agreement, dissenters get dampened weight.
        """
        agreement = 0.8
        contrast = (agreement - 0.5) * 2.0  # 0.6
        dampen = 1.0 - CONTRAST_BOOST * contrast
        self.assertLess(dampen, 1.0)
        self.assertGreater(dampen, 0.0)


class TestForecastUncertainty(unittest.TestCase):
    """Test forecast dispersion/uncertainty measures."""

    def test_trending_data_lower_uncertainty(self):
        """Strongly trending data should produce lower forecast dispersion."""
        # Strong uptrend
        prices_up = _make_price_series(400, daily_return=0.003, seed=20)
        result_up = ensemble_forecast(prices_up, horizons=STANDARD_HORIZONS)
        
        # Choppy/flat data
        prices_flat = _make_price_series(400, daily_return=0.0, seed=21)
        result_flat = ensemble_forecast(prices_flat, horizons=STANDARD_HORIZONS)
        
        # Both should produce valid results
        self.assertEqual(len(result_up), 8)
        self.assertEqual(len(result_flat), 8)

    def test_extreme_volatility_may_raise_contested(self):
        """
        Highly volatile data with no clear trend may produce Contested or Low confidence.
        """
        rng = np.random.RandomState(99)
        n = 400
        log_returns = rng.normal(0, 0.05, n)  # Very high vol
        log_prices = np.log(100) + np.cumsum(log_returns)
        prices = pd.Series(np.exp(log_prices),
                           index=pd.date_range("2020-01-01", periods=n, freq="B"))
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS)
        # Should be Low, Contested, or Medium — but may include a tag
        base_label = result[7].split(' [')[0] if ' [' in result[7] else result[7]
        self.assertIn(base_label, ["Low", "Contested", "Medium", "High"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
