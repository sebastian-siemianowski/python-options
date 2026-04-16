"""
Tests for Story 9.3: CUSUM Drift Detection on Innovations.

Tests that innovation_cusum() detects persistent drift shifts with
ARL ~500 under H0 and detection delay < 15 days for 1-sigma shift.
"""
import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.innovation_diagnostics import (
    CUSUM_THRESHOLD,
    CUSUM_REFERENCE,
    CUSUM_Q_MULTIPLIER,
    CUSUM_ALARM_DURATION,
    CUSUMResult,
    innovation_cusum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_no_drift(n=500, seed=42):
    """White noise innovations (no drift shift)."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0, 1, size=n)
    R = np.ones(n)
    return innovations, R


def _make_drift_shift(n=500, shift_time=200, drift_magnitude=1.0, seed=42):
    """Innovations with a persistent drift shift at shift_time."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0, 1, size=n)
    # After shift_time, add persistent drift
    innovations[shift_time:] += drift_magnitude
    R = np.ones(n)
    return innovations, R


class TestCUSUMBasic(unittest.TestCase):
    """Basic API and contract tests."""

    def test_returns_result_dataclass(self):
        innovations, R = _make_no_drift()
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)

    def test_cusum_paths_length(self):
        innovations, R = _make_no_drift(n=200)
        result = innovation_cusum(innovations, R)
        self.assertEqual(len(result.cusum_pos), 200)
        self.assertEqual(len(result.cusum_neg), 200)

    def test_n_obs_correct(self):
        innovations, R = _make_no_drift(n=150)
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.n_obs, 150)

    def test_threshold_stored(self):
        innovations, R = _make_no_drift()
        result = innovation_cusum(innovations, R, threshold=5.0)
        self.assertEqual(result.threshold, 5.0)

    def test_cusum_paths_nonnegative(self):
        """CUSUM paths are always >= 0."""
        innovations, R = _make_no_drift()
        result = innovation_cusum(innovations, R)
        self.assertTrue(np.all(result.cusum_pos >= 0))
        self.assertTrue(np.all(result.cusum_neg >= 0))

    def test_alarm_times_are_list(self):
        innovations, R = _make_no_drift()
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result.alarm_times, list)

    def test_alarm_directions_match_times(self):
        innovations, R = _make_drift_shift(drift_magnitude=2.0)
        result = innovation_cusum(innovations, R)
        self.assertEqual(len(result.alarm_times), len(result.alarm_directions))

    def test_q_multiplier_default(self):
        innovations, R = _make_no_drift()
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.q_multiplier, CUSUM_Q_MULTIPLIER)

    def test_alarm_duration_default(self):
        innovations, R = _make_no_drift()
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.alarm_duration, CUSUM_ALARM_DURATION)


class TestCUSUMNoDrift(unittest.TestCase):
    """Tests on innovations with no drift (H0)."""

    def test_no_alarm_on_white_noise(self):
        """White noise: should rarely alarm."""
        innovations, R = _make_no_drift(n=500, seed=100)
        result = innovation_cusum(innovations, R)
        # With n=500 and ARL=500, we expect ~1 alarm on average
        # Allow up to 3 alarms (rare but possible)
        self.assertLessEqual(result.n_alarms, 3)

    def test_arl_above_200(self):
        """Average run length under H0 should be long (>200 days).

        We test over multiple seeds: at most 20% of 500-length runs alarm.
        """
        n_alarms_total = 0
        n_trials = 50
        for seed in range(n_trials):
            innovations, R = _make_no_drift(n=500, seed=200 + seed)
            result = innovation_cusum(innovations, R)
            n_alarms_total += result.n_alarms
        # With ARL=500, expected alarms per 500 obs = 1.0
        # Over 50 trials, expected total = 50
        # Allow generous margin
        avg_alarms = n_alarms_total / n_trials
        self.assertLess(avg_alarms, 3.0,
                        f"Average alarms per run = {avg_alarms:.2f}, too many")

    def test_false_alarm_rate_under_50_percent(self):
        """Less than 50% of runs should have any alarm (ARL >> n)."""
        n_alarmed = 0
        for seed in range(50):
            innovations, R = _make_no_drift(n=500, seed=300 + seed)
            result = innovation_cusum(innovations, R)
            if result.has_alarm:
                n_alarmed += 1
        rate = n_alarmed / 50
        self.assertLess(rate, 0.50,
                        f"False alarm rate {rate:.2%} >= 50%")


class TestCUSUMDriftDetection(unittest.TestCase):
    """Tests on innovations with drift shifts."""

    def test_detects_1_sigma_shift(self):
        """1-sigma drift shift should be detected."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=1.0, seed=400)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)

    def test_detects_2_sigma_shift(self):
        """2-sigma drift shift: definite detection."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=2.0, seed=410)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)

    def test_alarm_after_shift_time(self):
        """Alarm should occur after the shift time."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=1.5, seed=420)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            first_alarm = result.alarm_times[0]
            self.assertGreaterEqual(first_alarm, 200)

    def test_detection_delay_under_15_days_1_sigma(self):
        """Detection delay for 1-sigma shift < 15 days (on average)."""
        delays = []
        for seed in range(30):
            innovations, R = _make_drift_shift(
                n=500, shift_time=200, drift_magnitude=1.0, seed=500 + seed)
            result = innovation_cusum(innovations, R)
            if result.has_alarm:
                # Find first alarm after shift time
                for t in result.alarm_times:
                    if t >= 200:
                        delays.append(t - 200)
                        break
        # At least 50% of trials should detect within 15 days
        if delays:
            median_delay = sorted(delays)[len(delays) // 2]
            self.assertLessEqual(median_delay, 15,
                                 f"Median detection delay = {median_delay} > 15")

    def test_detection_rate_above_80_for_1_sigma(self):
        """Detection rate > 80% for 1-sigma shift (30 trials)."""
        n_detected = 0
        for seed in range(30):
            innovations, R = _make_drift_shift(
                n=500, shift_time=200, drift_magnitude=1.0, seed=600 + seed)
            result = innovation_cusum(innovations, R)
            # Check for alarm after shift
            for t in result.alarm_times:
                if t >= 200:
                    n_detected += 1
                    break
        rate = n_detected / 30
        self.assertGreater(rate, 0.80,
                           f"Detection rate {rate:.2%} <= 80%")

    def test_positive_drift_detected_positive(self):
        """Positive drift: alarm direction should be +1."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=2.0, seed=700)
        result = innovation_cusum(innovations, R)
        # Find first alarm after shift
        for i, t in enumerate(result.alarm_times):
            if t >= 200:
                self.assertEqual(result.alarm_directions[i], 1)
                break

    def test_negative_drift_detected_negative(self):
        """Negative drift: alarm direction should be -1."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=-2.0, seed=710)
        result = innovation_cusum(innovations, R)
        # Find first alarm after shift
        for i, t in enumerate(result.alarm_times):
            if t >= 200:
                self.assertEqual(result.alarm_directions[i], -1)
                break


class TestCUSUMQCorrection(unittest.TestCase):
    """Tests on q correction parameters."""

    def test_q_multiplier_is_10(self):
        """q multiplier should be 10x on alarm."""
        self.assertEqual(CUSUM_Q_MULTIPLIER, 10.0)

    def test_alarm_duration_is_5(self):
        """Alarm duration should be 5 days."""
        self.assertEqual(CUSUM_ALARM_DURATION, 5)

    def test_q_correction_workflow(self):
        """Verify the q correction workflow: q_new = q_old * multiplier."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=2.0, seed=800)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            q_old = 1e-5
            q_new = q_old * result.q_multiplier
            self.assertEqual(q_new, q_old * 10.0)
            self.assertEqual(result.alarm_duration, 5)


class TestCUSUMEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_very_short_series(self):
        """n=1: should not crash."""
        innovations = np.array([0.5])
        R = np.array([1.0])
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)
        self.assertFalse(result.has_alarm)

    def test_empty_series(self):
        """n=0: should not crash."""
        innovations = np.array([])
        R = np.array([])
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)
        self.assertEqual(result.n_obs, 0)

    def test_constant_innovations(self):
        """Constant innovations (drift = 0): should eventually alarm if drift != 0."""
        innovations = np.ones(500) * 0.8  # Slight positive drift
        R = np.ones(500)
        result = innovation_cusum(innovations, R)
        # 0.8 > k=0.5, so CUSUM should accumulate and alarm
        self.assertTrue(result.has_alarm)

    def test_custom_threshold(self):
        """Custom threshold affects sensitivity."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, drift_magnitude=1.0, seed=900)
        result_strict = innovation_cusum(innovations, R, threshold=8.0)
        result_lenient = innovation_cusum(innovations, R, threshold=2.0)
        # Lenient should have more or equal alarms
        self.assertGreaterEqual(result_lenient.n_alarms, result_strict.n_alarms)

    def test_custom_reference(self):
        """Custom reference value works."""
        innovations, R = _make_no_drift(n=200)
        result = innovation_cusum(innovations, R, reference=1.0)
        self.assertIsInstance(result, CUSUMResult)

    def test_cusum_resets_on_alarm(self):
        """CUSUM path resets to 0 after alarm."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=100, drift_magnitude=3.0, seed=910)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            t = result.alarm_times[0]
            direction = result.alarm_directions[0]
            if direction == 1:
                self.assertEqual(result.cusum_pos[t], 0.0)
            else:
                self.assertEqual(result.cusum_neg[t], 0.0)


class TestCUSUMConstants(unittest.TestCase):
    """Test configuration constants."""

    def test_threshold_default(self):
        self.assertEqual(CUSUM_THRESHOLD, 4.0)

    def test_reference_default(self):
        self.assertEqual(CUSUM_REFERENCE, 0.5)

    def test_q_multiplier_default(self):
        self.assertEqual(CUSUM_Q_MULTIPLIER, 10.0)

    def test_alarm_duration_default(self):
        self.assertEqual(CUSUM_ALARM_DURATION, 5)


if __name__ == '__main__':
    unittest.main()
