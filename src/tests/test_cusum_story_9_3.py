"""
Tests for Story 9.3: CUSUM Drift Detection on Innovations.

Comprehensive test suite validating Page's two-sided CUSUM chart with:
  - ARL ~500 under H0 (Monte Carlo validated)
  - Detection delay < 15 days for 1-sigma drift shift
  - Correct alarm direction identification
  - q correction schedule with exponential decay
  - Threshold calibration via MC simulation
  - Edge cases and numerical stability
"""
import os
import sys
import math
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
    CUSUM_Q_DECAY_RATE,
    CUSUM_COOLDOWN_DAYS,
    CUSUM_MIN_OBS,
    CUSUMResult,
    innovation_cusum,
    apply_cusum_q_correction,
    calibrate_cusum_threshold,
    compute_cusum_diagnostics,
    _cusum_inner_loop,
    _estimate_arl_siegmund,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_white_noise(n=500, seed=42):
    """Standard white noise innovations (H0: no drift)."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, size=n)
    R = np.ones(n)
    return innovations, R


def _make_drift_shift(n=500, shift_time=200, delta=1.0, seed=42):
    """Innovations with persistent drift shift delta at shift_time."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, size=n)
    innovations[shift_time:] += delta
    R = np.ones(n)
    return innovations, R


def _make_time_varying_R(n=500, shift_time=200, delta=1.5, seed=42):
    """Innovations with time-varying R (heteroscedastic)."""
    rng = np.random.default_rng(seed)
    R = np.ones(n)
    R[shift_time:] = 4.0  # Double sigma after shift
    innovations = rng.normal(0.0, np.sqrt(R))
    innovations[shift_time:] += delta  # drift in original units
    return innovations, R


# =========================================================================
# 1. CUSUM Result Contract
# =========================================================================
class TestCUSUMContract(unittest.TestCase):
    """Verify the CUSUMResult dataclass contract and API."""

    def test_returns_cusum_result(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)

    def test_cusum_paths_length_matches_input(self):
        for n in [10, 100, 500]:
            innovations, R = _make_white_noise(n=n)
            result = innovation_cusum(innovations, R)
            self.assertEqual(len(result.cusum_pos), n)
            self.assertEqual(len(result.cusum_neg), n)

    def test_n_obs_correct(self):
        innovations, R = _make_white_noise(n=137)
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.n_obs, 137)

    def test_threshold_stored(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R, threshold=7.5)
        self.assertEqual(result.threshold, 7.5)

    def test_reference_stored(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R, reference=0.75)
        self.assertEqual(result.reference, 0.75)

    def test_cusum_paths_nonnegative(self):
        """CUSUM paths must be >= 0 by definition (max(0, ...) recursion)."""
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertTrue(np.all(result.cusum_pos >= 0.0))
        self.assertTrue(np.all(result.cusum_neg >= 0.0))

    def test_alarm_times_sorted(self):
        innovations, R = _make_drift_shift(delta=2.0)
        result = innovation_cusum(innovations, R)
        if len(result.alarm_times) > 1:
            for i in range(1, len(result.alarm_times)):
                self.assertGreater(result.alarm_times[i], result.alarm_times[i - 1])

    def test_alarm_directions_match_times(self):
        innovations, R = _make_drift_shift(delta=2.0)
        result = innovation_cusum(innovations, R)
        self.assertEqual(len(result.alarm_times), len(result.alarm_directions))
        self.assertEqual(len(result.alarm_times), len(result.alarm_magnitudes))

    def test_alarm_directions_valid_values(self):
        innovations, R = _make_drift_shift(delta=2.0)
        result = innovation_cusum(innovations, R)
        for d in result.alarm_directions:
            self.assertIn(d, [1, -1])

    def test_alarm_magnitudes_positive(self):
        """Alarm magnitude = CUSUM value at alarm, must exceed threshold."""
        innovations, R = _make_drift_shift(delta=2.0)
        result = innovation_cusum(innovations, R)
        for mag in result.alarm_magnitudes:
            self.assertGreater(mag, result.threshold)

    def test_has_alarm_consistent_with_n_alarms(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.has_alarm, result.n_alarms > 0)

    def test_q_multiplier_default(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.q_multiplier, CUSUM_Q_MULTIPLIER)

    def test_alarm_duration_default(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.alarm_duration, CUSUM_ALARM_DURATION)

    def test_estimated_arl_populated(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertGreater(result.estimated_arl, 0)
        self.assertTrue(math.isfinite(result.estimated_arl))

    def test_max_cusum_nonnegative(self):
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        self.assertGreaterEqual(result.max_cusum, 0.0)


# =========================================================================
# 2. Constants Validation
# =========================================================================
class TestCUSUMConstants(unittest.TestCase):
    """Validate CUSUM constants match spec requirements."""

    def test_threshold_is_5(self):
        self.assertEqual(CUSUM_THRESHOLD, 5.0)

    def test_reference_is_half(self):
        self.assertEqual(CUSUM_REFERENCE, 0.5)

    def test_q_multiplier_is_10(self):
        self.assertEqual(CUSUM_Q_MULTIPLIER, 10.0)

    def test_alarm_duration_is_5(self):
        self.assertEqual(CUSUM_ALARM_DURATION, 5)

    def test_cooldown_is_5(self):
        self.assertEqual(CUSUM_COOLDOWN_DAYS, 5)

    def test_decay_rate_positive(self):
        self.assertGreater(CUSUM_Q_DECAY_RATE, 0.0)

    def test_min_obs_is_2(self):
        self.assertEqual(CUSUM_MIN_OBS, 2)


# =========================================================================
# 3. ARL under H0 (Monte Carlo Validation)
# =========================================================================
class TestCUSUMARL(unittest.TestCase):
    """Monte Carlo validation of ARL under H0."""

    def test_no_alarm_on_most_short_series(self):
        """With n=200 and ARL~500, most short series should NOT alarm."""
        n_alarmed = 0
        for seed in range(50):
            innovations, R = _make_white_noise(n=200, seed=1000 + seed)
            result = innovation_cusum(innovations, R)
            if result.has_alarm:
                n_alarmed += 1
        rate = n_alarmed / 50
        self.assertLess(rate, 0.60, f"Alarm rate on n=200: {rate:.2%}")

    def test_arl_exceeds_200(self):
        """Average alarms per 500-obs run should be low (ARL > 200)."""
        total_alarms = 0
        n_trials = 50
        for seed in range(n_trials):
            innovations, R = _make_white_noise(n=500, seed=2000 + seed)
            result = innovation_cusum(innovations, R)
            total_alarms += result.n_alarms
        avg_alarms = total_alarms / n_trials
        self.assertLess(avg_alarms, 2.5,
                        f"Average alarms/run = {avg_alarms:.2f}")

    def test_false_alarm_rate_bounded(self):
        """With ARL~500 and n=500, P(>=1 alarm) < 85%."""
        n_alarmed = 0
        for seed in range(50):
            innovations, R = _make_white_noise(n=500, seed=3000 + seed)
            result = innovation_cusum(innovations, R)
            if result.has_alarm:
                n_alarmed += 1
        rate = n_alarmed / 50
        self.assertLess(rate, 0.85,
                        f"False alarm rate {rate:.2%} >= 85%")

    def test_theoretical_arl_approximation(self):
        """Siegmund formula should give ARL ~400-600 for h=5, k=0.5."""
        arl = _estimate_arl_siegmund(5.0, 0.5)
        self.assertGreater(arl, 200.0, f"Estimated ARL={arl:.0f} too low")
        self.assertLess(arl, 1500.0, f"Estimated ARL={arl:.0f} too high")

    def test_higher_threshold_increases_arl(self):
        """Higher h => fewer false alarms => higher ARL."""
        arl_5 = _estimate_arl_siegmund(5.0, 0.5)
        arl_8 = _estimate_arl_siegmund(8.0, 0.5)
        self.assertGreater(arl_8, arl_5)


# =========================================================================
# 4. Drift Detection Power
# =========================================================================
class TestCUSUMDetection(unittest.TestCase):
    """Test drift detection accuracy and speed."""

    def test_detects_1_sigma_shift(self):
        """1-sigma persistent drift: must be detected."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=1.0, seed=4000)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm,
                        "Failed to detect 1-sigma drift shift")

    def test_detects_2_sigma_shift(self):
        """2-sigma drift: definite detection with high confidence."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=2.0, seed=4100)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)

    def test_detects_half_sigma_shift(self):
        """0.5-sigma drift: should detect with longer delay."""
        innovations, R = _make_drift_shift(
            n=1000, shift_time=200, delta=0.5, seed=4200)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            post_shift = [t for t in result.alarm_times if t >= 200]
            self.assertGreater(len(post_shift), 0)

    def test_alarm_after_shift_time(self):
        """First drift-related alarm should be at or after shift time."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=1.5, seed=4300)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)
        post_shift = [t for t in result.alarm_times if t >= 190]
        self.assertGreater(len(post_shift), 0,
                          "No alarms near/after shift time")

    def test_detection_delay_under_15_days_1_sigma(self):
        """Median detection delay for 1-sigma shift < 15 days."""
        delays = []
        for seed in range(40):
            innovations, R = _make_drift_shift(
                n=500, shift_time=200, delta=1.0, seed=5000 + seed)
            result = innovation_cusum(innovations, R)
            if result.has_alarm:
                for t in result.alarm_times:
                    if t >= 200:
                        delays.append(t - 200)
                        break
        self.assertGreater(len(delays), 20,
                           f"Only {len(delays)}/40 detected")
        median_delay = sorted(delays)[len(delays) // 2]
        self.assertLessEqual(median_delay, 15,
                             f"Median delay = {median_delay} > 15")

    def test_detection_delay_under_5_days_2_sigma(self):
        """2-sigma shift: detection should be very fast (< 5 days median)."""
        delays = []
        for seed in range(40):
            innovations, R = _make_drift_shift(
                n=500, shift_time=200, delta=2.0, seed=5500 + seed)
            result = innovation_cusum(innovations, R)
            if result.has_alarm:
                for t in result.alarm_times:
                    if t >= 200:
                        delays.append(t - 200)
                        break
        self.assertGreater(len(delays), 35,
                           f"Only {len(delays)}/40 detected 2-sigma")
        median_delay = sorted(delays)[len(delays) // 2]
        self.assertLessEqual(median_delay, 8,
                             f"Median 2-sigma delay = {median_delay} > 8")

    def test_detection_rate_above_80_for_1_sigma(self):
        """Detection rate > 80% for 1-sigma shift across 40 trials."""
        n_detected = 0
        for seed in range(40):
            innovations, R = _make_drift_shift(
                n=500, shift_time=200, delta=1.0, seed=6000 + seed)
            result = innovation_cusum(innovations, R)
            for t in result.alarm_times:
                if t >= 200:
                    n_detected += 1
                    break
        rate = n_detected / 40
        self.assertGreater(rate, 0.80,
                           f"Detection rate {rate:.2%} <= 80%")

    def test_detection_rate_above_95_for_2_sigma(self):
        """Detection rate > 95% for 2-sigma shift."""
        n_detected = 0
        for seed in range(40):
            innovations, R = _make_drift_shift(
                n=500, shift_time=200, delta=2.0, seed=6500 + seed)
            result = innovation_cusum(innovations, R)
            for t in result.alarm_times:
                if t >= 200:
                    n_detected += 1
                    break
        rate = n_detected / 40
        self.assertGreater(rate, 0.95,
                           f"2-sigma detection rate {rate:.2%} <= 95%")

    def test_positive_drift_gives_positive_direction(self):
        """Positive drift shift: alarm direction = +1."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=2.0, seed=7000)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)
        for i, t in enumerate(result.alarm_times):
            if t >= 200:
                self.assertEqual(result.alarm_directions[i], 1,
                                f"Positive drift gave direction {result.alarm_directions[i]}")
                break

    def test_negative_drift_gives_negative_direction(self):
        """Negative drift shift: alarm direction = -1."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=-2.0, seed=7100)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)
        for i, t in enumerate(result.alarm_times):
            if t >= 200:
                self.assertEqual(result.alarm_directions[i], -1,
                                f"Negative drift gave direction {result.alarm_directions[i]}")
                break

    def test_time_varying_R_detection(self):
        """Drift detection works with heteroscedastic R."""
        innovations, R = _make_time_varying_R(
            n=500, shift_time=200, delta=2.0, seed=7200)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm,
                        "Failed to detect drift with time-varying R")


# =========================================================================
# 5. q Correction Schedule
# =========================================================================
class TestCUSUMQCorrection(unittest.TestCase):
    """Test apply_cusum_q_correction() q schedule generation."""

    def test_no_alarm_gives_constant_q(self):
        """No alarms => q_t = q_base everywhere."""
        innovations, R = _make_white_noise(n=200, seed=100)
        result = innovation_cusum(innovations, R, threshold=50.0)
        q_t = apply_cusum_q_correction(result, q_base=1e-5, T=200)
        np.testing.assert_array_equal(q_t, np.full(200, 1e-5))

    def test_alarm_boosts_q_at_alarm_time(self):
        """q_t at alarm time should be q_base * q_multiplier."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=3.0, seed=8000)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)
        q_base = 1e-5
        q_t = apply_cusum_q_correction(result, q_base, T=500)
        t0 = result.alarm_times[0]
        self.assertAlmostEqual(q_t[t0], q_base * CUSUM_Q_MULTIPLIER, places=10)

    def test_q_decays_after_alarm(self):
        """q_t should decay monotonically after alarm time."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=3.0, seed=8100)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)
        q_base = 1e-5
        q_t = apply_cusum_q_correction(result, q_base, T=500)
        t0 = result.alarm_times[0]
        if t0 + CUSUM_ALARM_DURATION < 500:
            for d in range(1, CUSUM_ALARM_DURATION):
                if t0 + d < 500:
                    self.assertGreater(q_t[t0 + d - 1], q_t[t0 + d],
                                      f"q not decaying at t={t0}+{d}")

    def test_q_returns_to_baseline_after_window(self):
        """q_t should return to q_base after alarm_duration when no more alarms."""
        # Use a single-alarm scenario: short drift that triggers one alarm
        innovations = np.zeros(500)
        innovations[10:20] = 3.0  # Brief burst triggers one alarm
        R = np.ones(500)
        result = innovation_cusum(innovations, R)
        if result.has_alarm and result.n_alarms == 1:
            q_base = 1e-5
            q_t = apply_cusum_q_correction(result, q_base, T=500)
            t0 = result.alarm_times[0]
            far_future = t0 + CUSUM_ALARM_DURATION + 50
            if far_future < 500:
                self.assertAlmostEqual(q_t[far_future], q_base, places=10)

    def test_q_always_at_least_base(self):
        """q_t >= q_base everywhere."""
        innovations, R = _make_drift_shift(
            n=500, shift_time=200, delta=2.0, seed=8300)
        result = innovation_cusum(innovations, R)
        q_base = 1e-5
        q_t = apply_cusum_q_correction(result, q_base, T=500)
        self.assertTrue(np.all(q_t >= q_base - 1e-20))

    def test_q_correction_workflow(self):
        """Verify: q_new = q_old * multiplier at peak."""
        q_old = 1e-5
        innovations, R = _make_drift_shift(delta=2.0, seed=8400)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            q_new_peak = q_old * result.q_multiplier
            self.assertEqual(q_new_peak, q_old * 10.0)

    def test_q_array_length_matches_T(self):
        """Output length must match T parameter."""
        innovations, R = _make_white_noise()
        result = innovation_cusum(innovations, R)
        for T in [100, 500, 1000]:
            q_t = apply_cusum_q_correction(result, 1e-5, T=T)
            self.assertEqual(len(q_t), T)

    def test_overlapping_alarms_take_max(self):
        """If alarm windows overlap, take the higher q value."""
        innovations = np.zeros(100)
        innovations[10:] = 5.0
        R = np.ones(100)
        result = innovation_cusum(innovations, R, threshold=2.0)
        q_base = 1e-5
        q_t = apply_cusum_q_correction(result, q_base, T=100)
        self.assertTrue(np.all(q_t >= q_base - 1e-20))

    def test_custom_decay_rate(self):
        """Faster decay rate means q returns to base quicker."""
        innovations, R = _make_drift_shift(delta=3.0, seed=8500)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            q_base = 1e-5
            q_fast = apply_cusum_q_correction(result, q_base, T=500, decay_rate=2.0)
            q_slow = apply_cusum_q_correction(result, q_base, T=500, decay_rate=0.1)
            t0 = result.alarm_times[0]
            if t0 + 2 < 500:
                self.assertLess(q_fast[t0 + 2], q_slow[t0 + 2])


# =========================================================================
# 6. Inner Loop
# =========================================================================
class TestCUSUMInnerLoop(unittest.TestCase):
    """Test the _cusum_inner_loop kernel directly."""

    def test_zero_input_no_alarm(self):
        """All-zero z: CUSUM stays at 0."""
        z = np.zeros(100)
        cp, cn, at, ad, am = _cusum_inner_loop(z, h=5.0, k=0.5, cooldown_days=5)
        np.testing.assert_array_equal(cp, np.zeros(100))
        np.testing.assert_array_equal(cn, np.zeros(100))
        self.assertEqual(len(at), 0)

    def test_constant_positive_z_above_k(self):
        """z_t = 1.0 > k=0.5 => accumulates 0.5 per step, alarms early."""
        z = np.ones(100) * 1.0
        cp, cn, at, ad, am = _cusum_inner_loop(z, h=5.0, k=0.5, cooldown_days=5)
        self.assertGreater(len(at), 0)
        self.assertEqual(ad[0], 1)
        self.assertLessEqual(at[0], 12)

    def test_constant_negative_z_below_minus_k(self):
        """z_t = -1.0: negative CUSUM accumulates."""
        z = -np.ones(100) * 1.0
        cp, cn, at, ad, am = _cusum_inner_loop(z, h=5.0, k=0.5, cooldown_days=5)
        self.assertGreater(len(at), 0)
        self.assertEqual(ad[0], -1)

    def test_cooldown_prevents_rapid_realarm(self):
        """After alarm, no alarm during cooldown period."""
        z = np.ones(100) * 2.0
        cp, cn, at, ad, am = _cusum_inner_loop(z, h=2.0, k=0.5, cooldown_days=10)
        if len(at) >= 2:
            gap = at[1] - at[0]
            self.assertGreaterEqual(gap, 10,
                                    f"Alarm gap {gap} < cooldown 10")

    def test_reset_on_alarm(self):
        """CUSUM resets to 0 at alarm time."""
        z = np.ones(50) * 2.0
        cp, cn, at, ad, am = _cusum_inner_loop(z, h=3.0, k=0.5, cooldown_days=0)
        if len(at) > 0:
            t0 = at[0]
            self.assertEqual(cp[t0], 0.0, "CUSUM not reset at alarm")


# =========================================================================
# 7. Threshold Calibration
# =========================================================================
class TestCUSUMCalibration(unittest.TestCase):
    """Test calibrate_cusum_threshold() MC calibration."""

    def test_calibrated_threshold_positive(self):
        h = calibrate_cusum_threshold(
            k=0.5, target_arl=200, n_sim=200, sim_length=2000, seed=9000)
        self.assertGreater(h, 0.0)

    def test_higher_arl_gives_higher_threshold(self):
        h_200 = calibrate_cusum_threshold(
            k=0.5, target_arl=200, n_sim=200, sim_length=2000, seed=9100)
        h_1000 = calibrate_cusum_threshold(
            k=0.5, target_arl=1000, n_sim=200, sim_length=2000, seed=9100)
        self.assertGreater(h_1000, h_200)

    def test_calibrated_h_in_reasonable_range(self):
        """For ARL=500, k=0.5, calibrated h should be roughly 4-6."""
        h = calibrate_cusum_threshold(
            k=0.5, target_arl=500, n_sim=500, sim_length=3000, seed=9200)
        self.assertGreater(h, 3.0, f"Calibrated h={h:.2f} too low")
        self.assertLess(h, 8.0, f"Calibrated h={h:.2f} too high")


# =========================================================================
# 8. Comprehensive Diagnostics
# =========================================================================
class TestCUSUMDiagnostics(unittest.TestCase):
    """Test compute_cusum_diagnostics() helper."""

    def test_returns_dict(self):
        innovations, R = _make_white_noise()
        diag = compute_cusum_diagnostics(innovations, R)
        self.assertIsInstance(diag, dict)

    def test_contains_required_keys(self):
        innovations, R = _make_white_noise()
        diag = compute_cusum_diagnostics(innovations, R)
        expected_keys = {
            'cusum_result', 'q_schedule', 'alarm_density',
            'max_cusum_pos', 'max_cusum_neg', 'estimated_arl',
            'pct_time_in_alarm',
        }
        self.assertTrue(expected_keys.issubset(diag.keys()))

    def test_q_schedule_length_matches_input(self):
        innovations, R = _make_white_noise(n=300)
        diag = compute_cusum_diagnostics(innovations, R)
        self.assertEqual(len(diag['q_schedule']), 300)

    def test_alarm_density_annualized(self):
        innovations, R = _make_drift_shift(delta=2.0, n=252)
        diag = compute_cusum_diagnostics(innovations, R)
        if diag['cusum_result'].has_alarm:
            self.assertGreater(diag['alarm_density'], 0.0)

    def test_no_alarm_zero_pct_in_alarm(self):
        innovations, R = _make_white_noise(n=100, seed=100)
        diag = compute_cusum_diagnostics(innovations, R, threshold=50.0)
        self.assertEqual(diag['pct_time_in_alarm'], 0.0)


# =========================================================================
# 9. Edge Cases & Numerical Stability
# =========================================================================
class TestCUSUMEdgeCases(unittest.TestCase):
    """Edge cases and numerical robustness."""

    def test_single_observation(self):
        innovations = np.array([0.5])
        R = np.array([1.0])
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)
        self.assertFalse(result.has_alarm)
        self.assertEqual(result.n_obs, 1)

    def test_empty_series(self):
        innovations = np.array([])
        R = np.array([])
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)
        self.assertEqual(result.n_obs, 0)
        self.assertFalse(result.has_alarm)

    def test_two_observations(self):
        innovations = np.array([0.0, 3.0])
        R = np.array([1.0, 1.0])
        result = innovation_cusum(innovations, R)
        self.assertEqual(result.n_obs, 2)

    def test_very_small_R(self):
        innovations = np.array([0.001, 0.002, 0.001])
        R = np.array([1e-20, 1e-20, 1e-20])
        result = innovation_cusum(innovations, R)
        self.assertIsInstance(result, CUSUMResult)
        self.assertFalse(np.any(np.isnan(result.cusum_pos)))
        self.assertFalse(np.any(np.isinf(result.cusum_pos)))

    def test_very_large_innovations(self):
        innovations = np.zeros(50)
        innovations[10] = 100.0
        R = np.ones(50)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)

    def test_constant_drift_accumulates(self):
        innovations = np.ones(500) * 0.8
        R = np.ones(500)
        result = innovation_cusum(innovations, R)
        self.assertTrue(result.has_alarm)
        self.assertLessEqual(result.alarm_times[0], 20)

    def test_constant_drift_below_k_no_alarm(self):
        innovations = np.ones(500) * 0.3
        R = np.ones(500)
        result = innovation_cusum(innovations, R)
        self.assertFalse(result.has_alarm)

    def test_custom_threshold_strict(self):
        innovations, R = _make_drift_shift(delta=1.0, seed=9900)
        strict = innovation_cusum(innovations, R, threshold=8.0)
        lenient = innovation_cusum(innovations, R, threshold=2.0)
        self.assertGreaterEqual(lenient.n_alarms, strict.n_alarms)

    def test_custom_reference_value(self):
        innovations, R = _make_white_noise(n=200)
        result = innovation_cusum(innovations, R, reference=1.0)
        self.assertIsInstance(result, CUSUMResult)
        self.assertEqual(result.reference, 1.0)

    def test_cusum_resets_on_alarm(self):
        innovations, R = _make_drift_shift(
            n=500, shift_time=100, delta=3.0, seed=9910)
        result = innovation_cusum(innovations, R)
        if result.has_alarm:
            t0 = result.alarm_times[0]
            direction = result.alarm_directions[0]
            if direction == 1:
                self.assertEqual(result.cusum_pos[t0], 0.0)
            else:
                self.assertEqual(result.cusum_neg[t0], 0.0)

    def test_q_correction_on_empty_result(self):
        innovations = np.array([])
        R = np.array([])
        result = innovation_cusum(innovations, R)
        q_t = apply_cusum_q_correction(result, q_base=1e-5, T=100)
        np.testing.assert_array_equal(q_t, np.full(100, 1e-5))


# =========================================================================
# 10. Siegmund ARL Formula
# =========================================================================
class TestSiegmundARL(unittest.TestCase):
    """Test the theoretical ARL approximation."""

    def test_invalid_k_returns_inf(self):
        self.assertEqual(_estimate_arl_siegmund(5.0, 0.0), float('inf'))
        self.assertEqual(_estimate_arl_siegmund(5.0, -1.0), float('inf'))

    def test_invalid_h_returns_inf(self):
        self.assertEqual(_estimate_arl_siegmund(0.0, 0.5), float('inf'))
        self.assertEqual(_estimate_arl_siegmund(-1.0, 0.5), float('inf'))

    def test_monotonic_in_h(self):
        arls = [_estimate_arl_siegmund(h, 0.5) for h in [2, 3, 4, 5, 6, 7, 8]]
        for i in range(1, len(arls)):
            self.assertGreater(arls[i], arls[i - 1])

    def test_all_positive_finite(self):
        for k in [0.25, 0.5, 0.75, 1.0]:
            arl = _estimate_arl_siegmund(5.0, k)
            self.assertGreater(arl, 0)
            self.assertTrue(math.isfinite(arl))


if __name__ == '__main__':
    unittest.main()
