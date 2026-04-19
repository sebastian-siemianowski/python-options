"""
Tests for Epic 29: Missing Data, Halts, and Market Closures

Story 29.1: gap_aware_predict - advances Kalman state by k steps
Story 29.2: market_gap_days - holiday calendar integration
Story 29.3: data_quality_score - graceful degradation
"""

import os
import sys
import unittest
from datetime import date, timedelta

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.missing_data import (
    gap_aware_predict,
    market_gap_days,
    nyse_holidays,
    data_quality_score,
    GapAwarePredictResult,
    DataQualityResult,
    QUALITY_NORMAL,
    QUALITY_REDUCED,
    QUALITY_LOW,
    INTERVAL_MULTIPLIER_REDUCED,
)


# ===========================================================================
# Story 29.1: Gap-Aware Kalman Prediction Step
# ===========================================================================

class TestGapAwarePredict(unittest.TestCase):
    """AC: gap_aware_predict advances state by k steps."""

    def test_single_step(self):
        """k=1: standard prediction step."""
        result = gap_aware_predict(mu=0.01, P=0.001, phi=0.999, q=1e-6, gap_days=1)
        self.assertIsInstance(result, GapAwarePredictResult)
        self.assertEqual(result.gap_days, 1)
        self.assertAlmostEqual(result.mu_predicted, 0.999 * 0.01, places=6)

    def test_mu_decays_with_gap(self):
        """AC: mu_{t+k} = phi^k * mu_t (drift decays toward zero)."""
        mu = 0.01
        phi = 0.99

        r1 = gap_aware_predict(mu=mu, P=0.001, phi=phi, q=1e-6, gap_days=1)
        r3 = gap_aware_predict(mu=mu, P=0.001, phi=phi, q=1e-6, gap_days=3)
        r10 = gap_aware_predict(mu=mu, P=0.001, phi=phi, q=1e-6, gap_days=10)

        self.assertAlmostEqual(r1.mu_predicted, phi ** 1 * mu, places=10)
        self.assertAlmostEqual(r3.mu_predicted, phi ** 3 * mu, places=10)
        self.assertAlmostEqual(r10.mu_predicted, phi ** 10 * mu, places=10)

        # mu should decrease with larger gaps
        self.assertGreater(abs(r1.mu_predicted), abs(r10.mu_predicted))

    def test_P_growth_increases_with_gap(self):
        """AC: P_growth increases with larger gaps."""
        q = 1e-6

        r1 = gap_aware_predict(mu=0, P=0.0, phi=0.999, q=q, gap_days=1)
        r3 = gap_aware_predict(mu=0, P=0.0, phi=0.999, q=q, gap_days=3)
        r10 = gap_aware_predict(mu=0, P=0.0, phi=0.999, q=q, gap_days=10)

        # With P_init=0, P_predicted = P_growth only
        self.assertGreater(r3.P_predicted, r1.P_predicted)
        self.assertGreater(r10.P_predicted, r3.P_predicted)

    def test_10_day_halt_large_P(self):
        """AC: After 10-day halt, P increases significantly."""
        P_init = 0.001
        q = 1e-5

        r1 = gap_aware_predict(mu=0, P=P_init, phi=0.999, q=q, gap_days=1)
        r10 = gap_aware_predict(mu=0, P=P_init, phi=0.999, q=q, gap_days=10)

        # P should grow substantially
        self.assertGreater(r10.P_growth, r1.P_growth * 5)

    def test_random_walk_case(self):
        """phi=1.0 (random walk): P grows by exactly k*q."""
        P_init = 0.001
        q = 1e-6
        k = 5

        result = gap_aware_predict(mu=0.01, P=P_init, phi=1.0, q=q, gap_days=k)
        # mu should be unchanged (phi^k = 1)
        self.assertAlmostEqual(result.mu_predicted, 0.01, places=10)
        # P_growth should be k * q
        self.assertAlmostEqual(result.P_growth, k * q, places=12)

    def test_continuous_time_limit(self):
        """AC: As k -> infinity, P converges to q / (1 - phi^2)."""
        phi = 0.95
        q = 1e-5
        P_limit = q / (1.0 - phi ** 2)

        result = gap_aware_predict(mu=0, P=0.0, phi=phi, q=q, gap_days=365)
        # Should be close to limit
        self.assertAlmostEqual(result.P_predicted, P_limit, delta=P_limit * 0.01)

    def test_gap_capped(self):
        """Gap is capped at MAX_GAP_DAYS."""
        result = gap_aware_predict(mu=0, P=0.001, phi=0.999, q=1e-6, gap_days=10000)
        self.assertEqual(result.gap_days, 365)

    def test_negative_gap_floored(self):
        """Negative gap treated as 1."""
        result = gap_aware_predict(mu=0, P=0.001, phi=0.999, q=1e-6, gap_days=-5)
        self.assertEqual(result.gap_days, 1)

    def test_to_dict(self):
        result = gap_aware_predict(mu=0.01, P=0.001, phi=0.999, q=1e-6, gap_days=3)
        d = result.to_dict()
        self.assertIn("mu_predicted", d)
        self.assertIn("P_predicted", d)
        self.assertIn("gap_days", d)


# ===========================================================================
# Story 29.2: Holiday Calendar Integration
# ===========================================================================

class TestNYSEHolidays(unittest.TestCase):
    """Test NYSE holiday calendar."""

    def test_2024_holidays(self):
        """Check known 2024 NYSE holidays."""
        holidays = nyse_holidays(2024)
        # New Year: Jan 1 (Monday)
        self.assertIn(date(2024, 1, 1), holidays)
        # MLK Day: Jan 15
        self.assertIn(date(2024, 1, 15), holidays)
        # Presidents: Feb 19
        self.assertIn(date(2024, 2, 19), holidays)
        # Good Friday: Mar 29
        self.assertIn(date(2024, 3, 29), holidays)
        # Memorial Day: May 27
        self.assertIn(date(2024, 5, 27), holidays)
        # Juneteenth: Jun 19
        self.assertIn(date(2024, 6, 19), holidays)
        # July 4: Thursday
        self.assertIn(date(2024, 7, 4), holidays)
        # Labor Day: Sep 2
        self.assertIn(date(2024, 9, 2), holidays)
        # Thanksgiving: Nov 28
        self.assertIn(date(2024, 11, 28), holidays)
        # Christmas: Dec 25
        self.assertIn(date(2024, 12, 25), holidays)

    def test_holiday_count(self):
        """NYSE has ~10 holidays per year (post-2022 with Juneteenth)."""
        holidays = nyse_holidays(2024)
        self.assertGreaterEqual(len(holidays), 10)
        self.assertLessEqual(len(holidays), 12)

    def test_saturday_observed_friday(self):
        """Holiday on Saturday -> observed Friday."""
        # July 4, 2020 was Saturday -> observed Friday July 3
        holidays = nyse_holidays(2020)
        self.assertIn(date(2020, 7, 3), holidays)

    def test_sunday_observed_monday(self):
        """Holiday on Sunday -> observed Monday."""
        # Juneteenth 2022: June 19 was Sunday -> observed June 20
        holidays = nyse_holidays(2022)
        self.assertIn(date(2022, 6, 20), holidays)

    def test_no_juneteenth_before_2022(self):
        """Juneteenth not observed before 2022."""
        holidays = nyse_holidays(2021)
        self.assertNotIn(date(2021, 6, 18), holidays)
        self.assertNotIn(date(2021, 6, 19), holidays)
        self.assertNotIn(date(2021, 6, 21), holidays)


class TestMarketGapDays(unittest.TestCase):
    """AC: market_gap_days returns array of gap lengths."""

    def test_consecutive_weekdays(self):
        """Consecutive weekdays: gap = 1."""
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        gaps = market_gap_days(np.array(dates), market='nyse')
        np.testing.assert_array_equal(gaps, [1, 1])

    def test_weekend_gap(self):
        """Friday to Monday: gap = 1 (one trading day: Monday)."""
        dates = [date(2024, 1, 5), date(2024, 1, 8)]  # Fri to Mon
        gaps = market_gap_days(np.array(dates), market='nyse')
        np.testing.assert_array_equal(gaps, [1])

    def test_holiday_gap(self):
        """Gap including a holiday adds extra day."""
        # Jan 12 (Fri) to Jan 16 (Tue) 2024 -- MLK Monday Jan 15
        dates = [date(2024, 1, 12), date(2024, 1, 16)]
        gaps = market_gap_days(np.array(dates), market='nyse')
        # Sat 13, Sun 14 = not trading; Mon 15 = holiday; Tue 16 = 1 trading day
        np.testing.assert_array_equal(gaps, [1])

    def test_crypto_no_gaps(self):
        """AC: Crypto gap_days reflects calendar days."""
        dates = [date(2024, 1, 5), date(2024, 1, 8)]  # Fri to Mon
        gaps = market_gap_days(np.array(dates), market='crypto')
        np.testing.assert_array_equal(gaps, [3])  # 3 calendar days

    def test_single_date_empty(self):
        """Single date returns empty array."""
        gaps = market_gap_days(np.array([date(2024, 1, 1)]), market='nyse')
        self.assertEqual(len(gaps), 0)

    def test_two_dates(self):
        """Two dates returns one gap."""
        dates = [date(2024, 1, 2), date(2024, 1, 3)]
        gaps = market_gap_days(np.array(dates), market='nyse')
        self.assertEqual(len(gaps), 1)

    def test_nasdaq_same_as_nyse(self):
        """NASDAQ uses same calendar as NYSE."""
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        gaps_nyse = market_gap_days(np.array(dates), market='nyse')
        gaps_nasdaq = market_gap_days(np.array(dates), market='nasdaq')
        np.testing.assert_array_equal(gaps_nyse, gaps_nasdaq)


# ===========================================================================
# Story 29.3: Graceful Degradation on Extreme Missing Data
# ===========================================================================

class TestDataQualityScore(unittest.TestCase):
    """AC: data_quality_score returns quality assessment."""

    def test_full_data_normal(self):
        """AC: Quality > 95% -> NORMAL."""
        returns = np.random.default_rng(42).normal(0, 0.02, size=252)
        result = data_quality_score(returns, expected_obs=252)
        self.assertEqual(result.quality_flag, "NORMAL")
        self.assertGreater(result.quality_score, QUALITY_NORMAL)
        self.assertAlmostEqual(result.interval_multiplier, 1.0)
        self.assertFalse(result.suppress_direction)

    def test_reduced_confidence(self):
        """AC: Quality 80-95% -> REDUCED_CONFIDENCE, intervals * 1.3."""
        returns = np.ones(220) * 0.01
        # 220 / 252 = 87.3%
        result = data_quality_score(returns, expected_obs=252)
        self.assertEqual(result.quality_flag, "REDUCED_CONFIDENCE")
        self.assertAlmostEqual(result.interval_multiplier, INTERVAL_MULTIPLIER_REDUCED)
        self.assertFalse(result.suppress_direction)

    def test_low_quality(self):
        """AC: Quality < 80% -> LOW_QUALITY, suppress direction."""
        returns = np.ones(150) * 0.01
        # 150 / 252 = 59.5%
        result = data_quality_score(returns, expected_obs=252)
        self.assertEqual(result.quality_flag, "LOW_QUALITY")
        self.assertTrue(result.suppress_direction)

    def test_unusable(self):
        """AC: Quality < 50% -> UNUSABLE."""
        returns = np.ones(50) * 0.01
        # 50 / 252 = 19.8%
        result = data_quality_score(returns, expected_obs=252)
        self.assertEqual(result.quality_flag, "UNUSABLE")
        self.assertTrue(result.suppress_direction)

    def test_nan_counted_as_missing(self):
        """NaN values are counted as missing."""
        returns = np.ones(252) * 0.01
        returns[:50] = np.nan
        result = data_quality_score(returns, expected_obs=252)
        self.assertEqual(result.n_available, 202)
        self.assertAlmostEqual(result.quality_score, 202 / 252, places=3)

    def test_quality_capped_at_1(self):
        """Quality score is capped at 1.0 even if more data than expected."""
        returns = np.ones(300) * 0.01
        result = data_quality_score(returns, expected_obs=252)
        self.assertAlmostEqual(result.quality_score, 1.0)
        self.assertEqual(result.quality_flag, "NORMAL")

    def test_empty_returns(self):
        """Empty returns -> UNUSABLE."""
        result = data_quality_score(np.array([]), expected_obs=252)
        self.assertEqual(result.quality_flag, "UNUSABLE")
        self.assertEqual(result.n_available, 0)

    def test_to_dict(self):
        result = data_quality_score(np.ones(100), expected_obs=252)
        d = result.to_dict()
        self.assertIn("quality_score", d)
        self.assertIn("quality_flag", d)
        self.assertIn("interval_multiplier", d)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for missing data handling."""

    def test_gap_predict_then_quality(self):
        """Full pipeline: detect gaps -> predict -> assess quality."""
        # Simulate dates with a 10-day halt
        dates = []
        d = date(2024, 1, 2)
        for i in range(100):
            if 40 <= i < 50:
                continue  # 10-day halt
            dates.append(d + timedelta(days=i))

        gaps = market_gap_days(np.array(dates), market='nyse')

        # Verify some gaps > 1 exist
        self.assertTrue(np.any(gaps > 1))

        # Run gap-aware prediction at the halt
        halt_idx = np.argmax(gaps)
        k = int(gaps[halt_idx])

        result = gap_aware_predict(mu=0.01, P=0.0, phi=0.999, q=1e-6, gap_days=k)
        # P should grow from zero due to accumulated process noise
        self.assertGreater(result.P_predicted, 0.0)

    def test_btc_no_gaps(self):
        """BTC-USD (crypto) has no weekend gaps."""
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        gaps = market_gap_days(np.array(dates), market='crypto')
        np.testing.assert_array_equal(gaps, np.ones(29, dtype=np.int64))

    def test_quality_driven_interval_expansion(self):
        """Low quality data expands prediction intervals."""
        rng = np.random.default_rng(42)
        full_returns = rng.normal(0, 0.02, size=252)
        partial_returns = full_returns[:180]  # 71.4% quality

        full_q = data_quality_score(full_returns, expected_obs=252)
        partial_q = data_quality_score(partial_returns, expected_obs=252)

        self.assertEqual(full_q.quality_flag, "NORMAL")
        self.assertEqual(partial_q.quality_flag, "LOW_QUALITY")
        self.assertGreater(partial_q.interval_multiplier, full_q.interval_multiplier)


if __name__ == "__main__":
    unittest.main()
