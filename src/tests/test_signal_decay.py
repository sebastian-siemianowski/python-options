"""
Test Story 8.8: Signal Decay / TTL.

Validates:
  1. Fresh signal has full strength
  2. Signal at half-life has 50% strength
  3. 3-day-old 7-day forecast has reduced strength
  4. Expired signal detection
  5. Refresh recommendation
  6. TTL remaining computation
  7. Predictive power decay analysis
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

from decision.signal_decay import (
    compute_half_life,
    compute_signal_strength,
    compute_ttl_remaining,
    decay_signal,
    compute_predictive_power_decay,
    EXPIRED_THRESHOLD,
    REFRESH_THRESHOLD,
)


class TestSignalDecay(unittest.TestCase):
    """Tests for signal decay and TTL."""

    def test_fresh_signal_full_strength(self):
        """Age 0 -> strength 1.0."""
        s = compute_signal_strength(age_days=0, half_life=3.5)
        self.assertAlmostEqual(s, 1.0)

    def test_half_life_strength(self):
        """At half-life -> strength 0.5."""
        hl = 3.5
        s = compute_signal_strength(age_days=hl, half_life=hl)
        self.assertAlmostEqual(s, 0.5, places=5)

    def test_3day_old_7day_forecast(self):
        """3-day-old 7-day forecast has reduced strength."""
        sig = decay_signal("SPY", forecast_pct=2.0, confidence=0.8,
                           horizon_days=7, age_days=3.0)
        
        self.assertLess(sig.current_strength, 1.0)
        self.assertGreater(sig.current_strength, 0.3)
        self.assertLess(sig.forecast_pct, 2.0)
        self.assertLess(sig.confidence, 0.8)

    def test_expired_signal(self):
        """Very old signal is expired."""
        sig = decay_signal("SPY", forecast_pct=1.0, confidence=0.5,
                           horizon_days=1, age_days=10)
        
        self.assertTrue(sig.is_expired)
        self.assertAlmostEqual(sig.ttl_remaining_days, 0.0)

    def test_refresh_recommendation(self):
        """Signal below threshold triggers refresh."""
        sig = decay_signal("SPY", forecast_pct=1.0, confidence=0.5,
                           horizon_days=3, age_days=3.0)
        
        # Half-life = 1.5 days, age 3 = 2 half-lives -> strength ~0.25
        self.assertTrue(sig.needs_refresh)

    def test_ttl_remaining_fresh(self):
        """Fresh signal has positive TTL."""
        ttl = compute_ttl_remaining(1.0, half_life=3.5, threshold=0.1)
        self.assertGreater(ttl, 0)
        # Should be about 3.5 * log(10)/log(2) ~ 11.6 days
        expected = 3.5 * math.log(10) / math.log(2)
        self.assertAlmostEqual(ttl, expected, places=1)

    def test_half_life_from_horizon(self):
        """Half-life = 50% of horizon by default."""
        hl = compute_half_life(horizon_days=30)
        self.assertAlmostEqual(hl, 15.0)

    def test_predictive_power_decay(self):
        """Newer signals have better hit rate than older."""
        # Newer signals (age 0-1): always correct
        realized = [0.01] * 10 + [0.01] * 10
        forecast = [0.005] * 10 + [0.005] * 10
        ages = [0.5] * 10 + [10] * 10
        
        result = compute_predictive_power_decay(
            realized, forecast, ages,
            age_bins=[0, 1, 7, 14]
        )
        # Both buckets 100% because all same direction
        # Use mixed to test properly
        realized2 = [0.01] * 5 + [-0.01] * 5 + [0.01] * 3 + [-0.01] * 7
        forecast2 = [0.01] * 5 + [-0.01] * 5 + [0.01] * 5 + [0.01] * 5
        ages2 = [0.5] * 10 + [10] * 10
        
        result2 = compute_predictive_power_decay(
            realized2, forecast2, ages2,
            age_bins=[0, 1, 7, 14]
        )
        # Fresh (0-1d): 100% correct
        self.assertAlmostEqual(result2["0-1d"], 1.0)
        # Older (7-14d): only 60% correct (3/5 + 0/5 = 3/10)
        self.assertLess(result2["7-14d"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
