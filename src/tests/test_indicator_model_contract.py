import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.indicator_state import (
    INDICATOR_CHANNELS,
    build_adx_dmi_bundle,
    build_atr_supertrend_bundle,
    build_bollinger_keltner_bundle,
    build_chandelier_exit_bundle,
    build_donchian_breakout_bundle,
    build_heikin_ashi_bundle,
    build_ichimoku_bundle,
    build_empty_indicator_state,
    build_kama_efficiency_bundle,
    build_macd_acceleration_bundle,
    build_oscillator_exhaustion_bundle,
    build_persistence_bundle,
    build_relative_strength_bundle,
    build_volume_flow_bundle,
    build_vwap_dislocation_bundle,
    channels_for_specs,
    compute_adx_dmi_state,
    compute_adx_q_tail_conditioner,
    compute_atr_supertrend_state,
    compute_bollinger_keltner_state,
    compute_bollinger_variance_conditioner,
    compute_chandelier_exit_state,
    compute_donchian_breakout_state,
    compute_heikin_ashi_drift_signal,
    compute_heikin_ashi_state,
    compute_ichimoku_equilibrium_signal,
    compute_ichimoku_state,
    compute_kama_efficiency_state,
    compute_kama_equilibrium_signal,
    compute_liquidity_variance_conditioner,
    compute_macd_acceleration_state,
    compute_oscillator_exhaustion_state,
    compute_persistence_q_regime_conditioner,
    compute_persistence_state,
    compute_relative_strength_state,
    compute_turtle_breakout_quality,
    compute_volume_flow_state,
    compute_vwap_confidence_conditioner,
    compute_vwap_dislocation_state,
    compute_wavelet_energy_tail_conditioner,
    compute_williams_failed_breakout_signal,
    normalize_indicator_feature_transport,
    orthogonalize_momentum_features,
    get_indicator_feature_spec,
    get_indicator_feature_specs,
    get_indicator_features_for_channel,
    output_names_for_specs,
    validate_indicator_state_bundle,
    validate_source_columns,
)
from models.model_registry import (
    create_indicator_integrated_spec,
    get_model_spec,
    make_indicator_integrated_model_name,
    make_student_t_improved_name,
    make_student_t_name,
)


class IndicatorModelContractTest(unittest.TestCase):
    def test_registered_indicator_specs_are_causal_and_model_mapped(self):
        specs = get_indicator_feature_specs()
        self.assertGreaterEqual(len(specs), 10)
        for name, spec in specs.items():
            self.assertEqual(name, spec.name)
            self.assertGreaterEqual(spec.lag, 1)
            self.assertGreaterEqual(spec.lookback, 1)
            self.assertTrue(spec.output_names)
            self.assertTrue(spec.channels)
            self.assertTrue(set(spec.channels).issubset(set(INDICATOR_CHANNELS)))

    def test_channel_lookup_and_source_column_validation(self):
        q_specs = get_indicator_features_for_channel("q")
        self.assertTrue(any(spec.name == "heikin_ashi_state" for spec in q_specs))

        validate_source_columns(["heikin_ashi_state"], ["Open", "High", "Low", "Close"])
        with self.assertRaises(ValueError):
            validate_source_columns(["heikin_ashi_state"], ["Close"])

    def test_empty_bundle_validates_and_builds_feature_matrix(self):
        bundle = build_empty_indicator_state(
            n_obs=5,
            spec_names=("heikin_ashi_state",),
            source_columns=("open", "high", "low", "close"),
        )
        validate_indicator_state_bundle(bundle)

        outputs = output_names_for_specs(("heikin_ashi_state",))
        matrix = bundle.feature_matrix(outputs[:2])
        self.assertEqual(matrix.shape, (5, 2))
        self.assertTrue(np.isnan(matrix).all())

    def test_bundle_rejects_unregistered_outputs_and_bad_lengths(self):
        bundle = build_empty_indicator_state(4, ("heikin_ashi_state",))
        bad_features = dict(bundle.features)
        bad_features["ghost_feature"] = np.zeros(4)
        bad_bundle = type(bundle)(
            n_obs=bundle.n_obs,
            features=bad_features,
            availability=bundle.availability,
            spec_names=bundle.spec_names,
            source_columns=bundle.source_columns,
        )
        with self.assertRaises(ValueError):
            validate_indicator_state_bundle(bad_bundle)

        short_features = dict(bundle.features)
        first_key = next(iter(short_features))
        short_features[first_key] = np.zeros(3)
        short_bundle = type(bundle)(
            n_obs=bundle.n_obs,
            features=short_features,
            availability=bundle.availability,
            spec_names=bundle.spec_names,
            source_columns=bundle.source_columns,
        )
        with self.assertRaises(ValueError):
            validate_indicator_state_bundle(short_bundle)

    def test_indicator_integrated_model_spec_keeps_base_control_contract(self):
        spec = create_indicator_integrated_spec(
            make_student_t_improved_name(8),
            "heikin_ashi",
            ("heikin_ashi_state",),
            extra_param_names=("ind_ha_drift_weight",),
        )
        self.assertTrue(spec.is_indicator_integrated)
        self.assertEqual(spec.base_model_name, make_student_t_improved_name(8))
        self.assertIn("heikin_ashi_state", spec.indicator_features)
        self.assertIn("mean", spec.indicator_channels)
        self.assertIn("ind_ha_drift_weight", spec.param_names)
        self.assertEqual(spec.default_params["ind_ha_drift_weight"], 0.0)

    def test_heikin_ashi_student_t_variants_are_registered_side_by_side(self):
        base_name = make_student_t_name(8)
        variant_name = make_indicator_integrated_model_name(base_name, "heikin_ashi")
        base_spec = get_model_spec(base_name)
        variant_spec = get_model_spec(variant_name)

        self.assertIsNotNone(base_spec)
        self.assertIsNotNone(variant_spec)
        self.assertTrue(variant_spec.is_indicator_integrated)
        self.assertEqual(variant_spec.base_model_name, base_name)
        self.assertIn("heikin_ashi_state", variant_spec.indicator_features)
        self.assertIn("ind_ha_drift_weight", variant_spec.param_names)

    def test_unknown_indicator_specs_fail_closed(self):
        self.assertIsNone(get_indicator_feature_spec("not_real"))
        with self.assertRaises(KeyError):
            channels_for_specs(("not_real",))

    def test_heikin_ashi_state_is_lagged_and_causal(self):
        open_ = np.array([10.0, 11.0, 12.0, 11.0])
        high = np.array([13.0, 13.0, 13.0, 12.0])
        low = np.array([9.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 12.0, 11.0, 10.5])

        state = compute_heikin_ashi_state(open_, high, low, close, lag=1, use_numba=False)

        self.assertTrue(np.isnan(state["ha_body_ratio"][0]))
        ha_close_0 = 0.25 * (open_[0] + high[0] + low[0] + close[0])
        ha_open_0 = 0.5 * (open_[0] + close[0])
        ha_high_0 = max(high[0], ha_open_0, ha_close_0)
        ha_low_0 = min(low[0], ha_open_0, ha_close_0)
        ha_range_0 = ha_high_0 - ha_low_0
        expected_body_0 = (ha_close_0 - ha_open_0) / ha_range_0

        self.assertAlmostEqual(state["ha_body_ratio"][1], expected_body_0)
        self.assertEqual(state["ha_color"][1], 1.0)
        self.assertEqual(state["ha_run_length"][1], 1.0)

    def test_heikin_ashi_numba_matches_reference(self):
        open_ = np.array([10.0, 10.5, 11.0, 11.7, 11.2, 10.8])
        high = np.array([10.8, 11.2, 12.0, 12.1, 11.8, 11.4])
        low = np.array([9.8, 10.1, 10.7, 11.0, 10.4, 10.2])
        close = np.array([10.6, 11.0, 11.8, 11.1, 10.7, 11.3])

        ref = compute_heikin_ashi_state(open_, high, low, close, lag=1, use_numba=False)
        fast = compute_heikin_ashi_state(open_, high, low, close, lag=1, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)

    def test_heikin_ashi_bundle_validates(self):
        open_ = np.array([10.0, 10.5, 11.0])
        high = np.array([11.0, 11.2, 11.4])
        low = np.array([9.8, 10.1, 10.7])
        close = np.array([10.6, 11.0, 11.2])
        bundle = build_heikin_ashi_bundle(open_, high, low, close, use_numba=False)
        validate_indicator_state_bundle(bundle)
        self.assertEqual(bundle.n_obs, 3)
        self.assertTrue(bundle.availability["heikin_ashi_state"])

    def test_heikin_ashi_drift_signal_is_bounded_and_lag_missing_is_neutral(self):
        open_ = np.array([10.0, 10.5, 11.0, 11.7, 11.2, 10.8])
        high = np.array([10.8, 11.2, 12.0, 12.1, 11.8, 11.4])
        low = np.array([9.8, 10.1, 10.7, 11.0, 10.4, 10.2])
        close = np.array([10.6, 11.0, 11.8, 11.1, 10.7, 11.3])

        state = compute_heikin_ashi_state(open_, high, low, close, lag=1, use_numba=False)
        signal = compute_heikin_ashi_drift_signal(state)

        self.assertEqual(signal.shape, close.shape)
        self.assertTrue(np.isfinite(signal).all())
        self.assertEqual(signal[0], 0.0)
        self.assertLessEqual(np.max(np.abs(signal)), 1.0)

    def test_atr_supertrend_numba_matches_reference_and_handles_gaps(self):
        high = np.array([10.0, 10.2, 15.0, 15.1, 14.9, 14.8, 16.0, 16.2])
        low = np.array([10.0, 10.0, 14.2, 14.8, 14.5, 14.0, 15.2, 15.8])
        close = np.array([10.0, 10.1, 14.7, 15.0, 14.6, 14.2, 15.9, 16.0])

        ref = compute_atr_supertrend_state(high, low, close, period=3, multiplier=2.0, use_numba=False)
        fast = compute_atr_supertrend_state(high, low, close, period=3, multiplier=2.0, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["atr_z"][0]))
        self.assertTrue(np.isfinite(ref["atr_z"][1:]).all())
        self.assertTrue(set(np.unique(ref["supertrend_side"][1:])).issubset({-1.0, 1.0}))
        self.assertTrue(set(np.unique(ref["supertrend_flip"][1:])).issubset({0.0, 1.0}))

    def test_atr_supertrend_bundle_validates_flat_bars(self):
        high = np.array([10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 10.0, 10.0, 10.0])
        bundle = build_atr_supertrend_bundle(high, low, close, period=3, multiplier=2.0, use_numba=False)
        validate_indicator_state_bundle(bundle)
        self.assertEqual(bundle.n_obs, 4)
        self.assertTrue(bundle.availability["atr_supertrend_state"])
        for values in bundle.features.values():
            self.assertTrue(np.isfinite(np.nan_to_num(values)).all())

    def test_chandelier_exit_numba_matches_reference_and_is_lagged(self):
        high = np.array([10.0, 10.7, 11.2, 10.9, 11.6, 12.0, 11.7, 12.2])
        low = np.array([9.7, 10.1, 10.6, 10.2, 10.9, 11.3, 11.1, 11.5])
        close = np.array([10.2, 10.5, 10.8, 10.4, 11.3, 11.6, 11.2, 12.0])

        ref = compute_chandelier_exit_state(high, low, close, period=3, multiplier=2.5, use_numba=False)
        fast = compute_chandelier_exit_state(high, low, close, period=3, multiplier=2.5, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)

        self.assertTrue(np.isnan(ref["chandelier_long_distance"][0]))
        self.assertTrue(np.isfinite(ref["chandelier_long_distance"][1:]).all())
        self.assertTrue(np.max(np.abs(ref["chandelier_crowding"][1:])) <= 1.0)

    def test_chandelier_exit_bundle_validates_flat_bars(self):
        high = np.array([10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 10.0, 10.0, 10.0])
        bundle = build_chandelier_exit_bundle(high, low, close, period=3, multiplier=2.0, use_numba=False)
        validate_indicator_state_bundle(bundle)
        self.assertEqual(bundle.n_obs, 4)
        self.assertTrue(bundle.availability["chandelier_exit_state"])
        for values in bundle.features.values():
            self.assertTrue(np.isfinite(np.nan_to_num(values)).all())

    def test_kama_efficiency_numba_matches_reference_and_is_bounded(self):
        close = np.array([10.0, 10.2, 10.7, 10.5, 10.9, 11.4, 11.2, 11.8, 12.1, 11.9])

        ref = compute_kama_efficiency_state(close, er_period=4, slow_period=8, use_numba=False)
        fast = compute_kama_efficiency_state(close, er_period=4, slow_period=8, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)

        self.assertTrue(np.isnan(ref["kama_efficiency"][0]))
        finite_eff = ref["kama_efficiency"][1:]
        self.assertTrue(np.all((finite_eff >= 0.0) & (finite_eff <= 1.0)))
        self.assertTrue(np.max(np.abs(ref["kama_slope"][1:])) <= 6.0)
        self.assertTrue(np.max(np.abs(ref["kama_distance"][1:])) <= 6.0)

    def test_kama_bundle_and_equilibrium_signal_validate(self):
        close = np.array([10.0, 10.1, 10.0, 9.9, 10.2, 10.3, 10.1, 10.0])
        bundle = build_kama_efficiency_bundle(close, er_period=3, slow_period=7, use_numba=False)
        validate_indicator_state_bundle(bundle)
        signal = compute_kama_equilibrium_signal(bundle.features)
        self.assertEqual(signal.shape, close.shape)
        self.assertTrue(np.isfinite(signal).all())
        self.assertEqual(signal[0], 0.0)
        self.assertLessEqual(np.max(np.abs(signal)), 1.0)

    def test_adx_dmi_numba_matches_reference_and_handles_gaps(self):
        high = np.array([10.0, 10.3, 10.1, 11.2, 11.4, 11.0, 11.8, 12.0, 11.7])
        low = np.array([9.8, 10.0, 9.7, 10.5, 10.8, 10.2, 11.1, 11.4, 11.0])
        close = np.array([9.9, 10.2, 9.9, 11.0, 11.1, 10.5, 11.6, 11.8, 11.2])

        ref = compute_adx_dmi_state(high, low, close, period=3, use_numba=False)
        fast = compute_adx_dmi_state(high, low, close, period=3, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)

        self.assertTrue(np.isnan(ref["adx_strength"][0]))
        self.assertTrue(np.all((ref["adx_strength"][1:] >= 0.0) & (ref["adx_strength"][1:] <= 1.0)))
        self.assertLessEqual(np.max(np.abs(ref["dmi_spread"][1:])), 1.0)

    def test_adx_dmi_bundle_validates_flat_bars(self):
        high = np.array([10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 10.0, 10.0, 10.0])
        bundle = build_adx_dmi_bundle(high, low, close, period=3, use_numba=False)
        validate_indicator_state_bundle(bundle)
        self.assertEqual(bundle.n_obs, 4)
        self.assertTrue(bundle.availability["adx_dmi_state"])
        for values in bundle.features.values():
            self.assertTrue(np.isfinite(np.nan_to_num(values)).all())

    def test_adx_q_tail_conditioner_is_bounded_and_conservative(self):
        state = {
            "adx_strength": np.array([np.nan, 0.0, 0.4, 0.9]),
            "dmi_abs_spread": np.array([np.nan, 0.0, 0.5, 0.8]),
        }
        q_mult, tail_mult = compute_adx_q_tail_conditioner(state)
        self.assertEqual(q_mult.shape, (4,))
        self.assertEqual(tail_mult.shape, (4,))
        self.assertTrue(np.all((q_mult >= 0.75) & (q_mult <= 1.35)))
        self.assertTrue(np.all((tail_mult >= 0.90) & (tail_mult <= 1.20)))
        self.assertLess(q_mult[-1], q_mult[1])
        self.assertGreater(tail_mult[-1], tail_mult[1])

    def test_ichimoku_numba_matches_reference_and_respects_lag(self):
        high = np.array([10.0, 10.4, 10.8, 11.1, 11.0, 11.5, 11.9, 12.1, 12.0, 12.4])
        low = np.array([9.6, 9.9, 10.2, 10.5, 10.3, 10.8, 11.0, 11.4, 11.2, 11.7])
        close = np.array([9.9, 10.2, 10.6, 10.9, 10.7, 11.3, 11.6, 11.9, 11.5, 12.2])

        ref = compute_ichimoku_state(
            high, low, close,
            tenkan_period=3, kijun_period=5, span_b_period=7, lag=2,
            use_numba=False,
        )
        fast = compute_ichimoku_state(
            high, low, close,
            tenkan_period=3, kijun_period=5, span_b_period=7, lag=2,
            use_numba=True,
        )
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["ichimoku_cloud_distance"][:2]).all())
        self.assertTrue(np.isfinite(ref["ichimoku_cloud_distance"][2:]).all())
        self.assertTrue(np.all(ref["ichimoku_cloud_thickness"][2:] >= 0.0))

    def test_ichimoku_bundle_and_equilibrium_signal_validate(self):
        high = np.linspace(10.2, 12.0, 12)
        low = high - 0.6
        close = high - 0.25
        bundle = build_ichimoku_bundle(
            high, low, close,
            tenkan_period=3, kijun_period=5, span_b_period=7, lag=2,
            use_numba=False,
        )
        validate_indicator_state_bundle(bundle)
        signal = compute_ichimoku_equilibrium_signal(bundle.features)
        self.assertEqual(signal.shape, close.shape)
        self.assertTrue(np.isfinite(signal).all())
        self.assertLessEqual(np.max(np.abs(signal)), 1.0)

    def test_donchian_numba_matches_reference_and_has_no_lookahead_breakout(self):
        high = np.array([10.0, 10.2, 10.4, 10.3, 10.5, 11.3, 11.4, 11.2])
        low = np.array([9.5, 9.7, 9.9, 9.8, 9.9, 10.8, 10.9, 10.6])
        close = np.array([9.8, 10.0, 10.1, 10.0, 10.4, 11.2, 11.1, 10.7])

        ref = compute_donchian_breakout_state(high, low, close, period=4, lag=1, use_numba=False)
        fast = compute_donchian_breakout_state(high, low, close, period=4, lag=1, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["donchian_position"][0]))
        self.assertTrue(np.isfinite(ref["donchian_position"][1:]).all())
        self.assertLessEqual(np.max(np.abs(ref["donchian_position"][1:])), 1.0)
        self.assertGreater(ref["donchian_breakout_age"][6], 0.0)

    def test_donchian_bundle_and_turtle_quality_validate(self):
        high = np.array([10.0, 10.1, 10.2, 10.4, 10.8, 10.9])
        low = np.array([9.8, 9.8, 9.9, 10.0, 10.3, 10.4])
        close = np.array([9.9, 10.0, 10.1, 10.3, 10.7, 10.5])
        bundle = build_donchian_breakout_bundle(high, low, close, period=3, use_numba=False)
        validate_indicator_state_bundle(bundle)
        quality = compute_turtle_breakout_quality(bundle.features)
        self.assertEqual(quality.shape, close.shape)
        self.assertTrue(np.isfinite(quality).all())
        self.assertLessEqual(np.max(np.abs(quality)), 1.0)

    def test_bollinger_keltner_numba_matches_reference_and_tracks_squeeze(self):
        high = np.array([10.2, 10.1, 10.2, 10.15, 10.25, 10.3, 10.9, 11.4, 11.8])
        low = np.array([9.8, 9.9, 9.85, 9.9, 9.95, 10.0, 10.2, 10.6, 11.0])
        close = np.array([10.0, 10.0, 10.05, 10.02, 10.1, 10.2, 10.7, 11.2, 11.5])

        ref = compute_bollinger_keltner_state(high, low, close, period=4, use_numba=False)
        fast = compute_bollinger_keltner_state(high, low, close, period=4, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["bb_percentile"][0]))
        self.assertTrue(np.all((ref["bb_percentile"][1:] >= 0.0) & (ref["bb_percentile"][1:] <= 1.0)))
        self.assertTrue(set(np.unique(ref["keltner_squeeze"][1:])).issubset({0.0, 1.0}))

    def test_bollinger_bundle_and_variance_conditioner_validate(self):
        high = np.array([10.2, 10.1, 10.2, 10.15, 10.25, 10.3, 10.9, 11.4])
        low = np.array([9.8, 9.9, 9.85, 9.9, 9.95, 10.0, 10.2, 10.6])
        close = np.array([10.0, 10.0, 10.05, 10.02, 10.1, 10.2, 10.7, 11.2])
        bundle = build_bollinger_keltner_bundle(high, low, close, period=4, use_numba=False)
        validate_indicator_state_bundle(bundle)
        mult = compute_bollinger_variance_conditioner(bundle.features)
        self.assertEqual(mult.shape, close.shape)
        self.assertTrue(np.all((mult >= 0.80) & (mult <= 1.45)))

    def test_oscillator_numba_matches_reference_and_is_bounded(self):
        high = np.array([10.0, 10.2, 10.5, 10.4, 10.1, 9.9, 10.3, 10.6, 10.4])
        low = np.array([9.6, 9.8, 10.0, 9.9, 9.7, 9.4, 9.8, 10.0, 9.9])
        close = np.array([9.8, 10.1, 10.3, 10.0, 9.8, 9.6, 10.1, 10.4, 10.0])

        ref = compute_oscillator_exhaustion_state(high, low, close, period=4, use_numba=False)
        fast = compute_oscillator_exhaustion_state(high, low, close, period=4, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["rsi_z"][0]))
        self.assertLessEqual(np.max(np.abs(ref["stoch_rsi_z"][1:])), 1.0)
        self.assertLessEqual(np.max(np.abs(ref["williams_r_z"][1:])), 1.0)

    def test_oscillator_bundle_and_failed_breakout_signal_validate(self):
        high = np.array([10.0, 10.2, 10.5, 10.4, 10.1, 9.9, 10.3, 10.6, 10.4])
        low = np.array([9.6, 9.8, 10.0, 9.9, 9.7, 9.4, 9.8, 10.0, 9.9])
        close = np.array([9.8, 10.1, 10.3, 10.0, 9.8, 9.6, 10.1, 10.4, 10.0])
        osc = build_oscillator_exhaustion_bundle(high, low, close, period=4, use_numba=False)
        don = build_donchian_breakout_bundle(high, low, close, period=4, use_numba=False)
        validate_indicator_state_bundle(osc)
        signal = compute_williams_failed_breakout_signal(osc.features, don.features)
        self.assertEqual(signal.shape, close.shape)
        self.assertTrue(np.isfinite(signal).all())
        self.assertLessEqual(np.max(np.abs(signal)), 1.0)

    def test_macd_acceleration_numba_matches_reference_and_is_lagged(self):
        close = np.array([10.0, 10.2, 10.4, 10.1, 10.6, 10.9, 10.7, 11.2, 11.5, 11.3])
        ref = compute_macd_acceleration_state(
            close, fast_period=3, slow_period=6, signal_period=3, use_numba=False
        )
        fast = compute_macd_acceleration_state(
            close, fast_period=3, slow_period=6, signal_period=3, use_numba=True
        )
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["macd_z"][0]))
        self.assertLessEqual(np.max(np.abs(ref["momentum_acceleration"][1:])), 6.0)

    def test_macd_bundle_and_orthogonal_momentum_basis_validate(self):
        close = np.array([10.0, 10.2, 10.4, 10.1, 10.6, 10.9, 10.7, 11.2])
        bundle = build_macd_acceleration_bundle(
            close, fast_period=3, slow_period=6, signal_period=3, use_numba=False
        )
        validate_indicator_state_bundle(bundle)
        matrix = bundle.feature_matrix(("macd_z", "ppo_z", "trix_z", "momentum_acceleration"))
        basis = orthogonalize_momentum_features(matrix)
        self.assertEqual(basis.shape, matrix.shape)
        gram = np.nan_to_num(basis).T @ np.nan_to_num(basis)
        off_diag = gram - np.diag(np.diag(gram))
        self.assertLessEqual(np.max(np.abs(off_diag)), 1e-10)

    def test_volume_flow_numba_matches_reference_and_is_lagged(self):
        high = np.array([10.2, 10.5, 10.3, 10.8, 11.0, 10.9, 11.3, 11.5])
        low = np.array([9.8, 10.0, 9.9, 10.2, 10.5, 10.3, 10.8, 11.0])
        close = np.array([10.0, 10.4, 10.1, 10.7, 10.8, 10.5, 11.2, 11.1])
        volume = np.array([1000.0, 1500.0, 900.0, 1800.0, 2200.0, 1200.0, 2500.0, 1600.0])
        ref = compute_volume_flow_state(high, low, close, volume, period=3, use_numba=False)
        fast = compute_volume_flow_state(high, low, close, volume, period=3, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["obv_z"][0]))
        self.assertLessEqual(np.max(np.abs(ref["obv_z"][1:])), 1.0)
        self.assertLessEqual(np.max(np.abs(ref["mfi_z"][1:])), 1.0)
        self.assertLessEqual(np.max(np.abs(ref["cmf_z"][1:])), 1.0)
        self.assertLessEqual(np.max(np.abs(ref["volume_z"][1:])), 6.0)

    def test_volume_flow_bundle_and_liquidity_conditioner_validate(self):
        high = np.array([10.2, 10.5, 10.3, 10.8, 11.0, 10.9, 11.3, 11.5])
        low = np.array([9.8, 10.0, 9.9, 10.2, 10.5, 10.3, 10.8, 11.0])
        close = np.array([10.0, 10.4, 10.1, 10.7, 10.8, 10.5, 11.2, 11.1])
        volume = np.array([1000.0, 1500.0, 900.0, 1800.0, 2200.0, 1200.0, 2500.0, 1600.0])
        bundle = build_volume_flow_bundle(high, low, close, volume, period=3, use_numba=False)
        validate_indicator_state_bundle(bundle)
        variance_mult, confidence_mult = compute_liquidity_variance_conditioner(
            close, volume, bundle.features, period=3
        )
        self.assertEqual(variance_mult.shape, close.shape)
        self.assertEqual(confidence_mult.shape, close.shape)
        self.assertTrue(np.all((variance_mult >= 0.85) & (variance_mult <= 1.60)))
        self.assertTrue(np.all((confidence_mult >= 0.70) & (confidence_mult <= 1.25)))
        self.assertEqual(variance_mult[0], 1.0)
        self.assertEqual(confidence_mult[0], 1.0)

    def test_vwap_dislocation_numba_matches_reference_and_is_lagged(self):
        high = np.array([10.2, 10.6, 10.4, 10.9, 11.2, 11.0, 11.4, 11.8, 11.7])
        low = np.array([9.8, 10.1, 9.9, 10.3, 10.7, 10.4, 10.8, 11.1, 11.0])
        close = np.array([10.0, 10.5, 10.2, 10.8, 11.0, 10.6, 11.2, 11.6, 11.3])
        volume = np.array([1000.0, 1600.0, 1100.0, 1900.0, 2400.0, 1300.0, 2600.0, 3100.0, 1700.0])
        ref = compute_vwap_dislocation_state(
            high, low, close, volume, period=4, anchor_period=5, use_numba=False
        )
        fast = compute_vwap_dislocation_state(
            high, low, close, volume, period=4, anchor_period=5, use_numba=True
        )
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["vwap_distance"][0]))
        self.assertLessEqual(np.max(np.abs(ref["vwap_distance"][1:])), 6.0)
        self.assertLessEqual(np.max(np.abs(ref["vwap_band_z"][1:])), 6.0)
        self.assertLessEqual(np.max(np.abs(ref["vwap_slope"][1:])), 6.0)

    def test_vwap_bundle_and_confidence_conditioner_validate(self):
        high = np.array([10.2, 10.6, 10.4, 10.9, 11.2, 11.0, 11.4, 11.8, 11.7])
        low = np.array([9.8, 10.1, 9.9, 10.3, 10.7, 10.4, 10.8, 11.1, 11.0])
        close = np.array([10.0, 10.5, 10.2, 10.8, 11.0, 10.6, 11.2, 11.6, 11.3])
        volume = np.array([1000.0, 1600.0, 1100.0, 1900.0, 2400.0, 1300.0, 2600.0, 3100.0, 1700.0])
        bundle = build_vwap_dislocation_bundle(
            high, low, close, volume, period=4, anchor_period=5, use_numba=False
        )
        validate_indicator_state_bundle(bundle)
        variance_mult, confidence_mult = compute_vwap_confidence_conditioner(bundle.features)
        self.assertEqual(variance_mult.shape, close.shape)
        self.assertEqual(confidence_mult.shape, close.shape)
        self.assertTrue(np.all((variance_mult >= 0.90) & (variance_mult <= 1.60)))
        self.assertTrue(np.all((confidence_mult >= 0.70) & (confidence_mult <= 1.20)))
        self.assertEqual(variance_mult[0], 1.0)
        self.assertEqual(confidence_mult[0], 1.0)

    def test_persistence_numba_matches_reference_and_is_bounded(self):
        close = np.array([
            10.0, 10.2, 10.1, 10.5, 10.7, 10.6, 11.0, 11.4,
            11.2, 11.8, 12.0, 11.9, 12.4, 12.9, 12.7, 13.2,
        ])
        ref = compute_persistence_state(close, period=8, use_numba=False)
        fast = compute_persistence_state(close, period=8, use_numba=True)
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(ref["hurst_proxy"][0]))
        self.assertTrue(np.all((ref["hurst_proxy"][1:] >= 0.0) & (ref["hurst_proxy"][1:] <= 1.0)))
        self.assertTrue(
            np.all((ref["fractal_dimension_proxy"][1:] >= 1.0) & (ref["fractal_dimension_proxy"][1:] <= 2.0))
        )
        self.assertLessEqual(np.max(np.abs(ref["wavelet_energy_z"][1:])), 6.0)

    def test_persistence_bundle_and_energy_conditioners_validate(self):
        close = np.array([
            10.0, 10.2, 10.1, 10.5, 10.7, 10.6, 11.0, 11.4,
            11.2, 11.8, 12.0, 11.9, 12.4, 12.9, 12.7, 13.2,
        ])
        bundle = build_persistence_bundle(close, period=8, use_numba=False)
        validate_indicator_state_bundle(bundle)
        q_mult, regime_fit = compute_persistence_q_regime_conditioner(bundle.features)
        variance_mult, tail_mult = compute_wavelet_energy_tail_conditioner(bundle.features)
        self.assertEqual(q_mult.shape, close.shape)
        self.assertEqual(regime_fit.shape, close.shape)
        self.assertEqual(variance_mult.shape, close.shape)
        self.assertEqual(tail_mult.shape, close.shape)
        self.assertTrue(np.all((q_mult >= 0.75) & (q_mult <= 1.55)))
        self.assertTrue(np.all((regime_fit >= 0.75) & (regime_fit <= 1.25)))
        self.assertTrue(np.all((variance_mult >= 0.90) & (variance_mult <= 1.70)))
        self.assertTrue(np.all((tail_mult >= 0.95) & (tail_mult <= 1.35)))
        self.assertEqual(q_mult[0], 1.0)
        self.assertEqual(variance_mult[0], 1.0)

    def test_relative_strength_numba_matches_reference_and_is_lagged(self):
        close = np.array([10.0, 10.4, 10.2, 10.8, 11.3, 11.0, 11.6, 12.1, 11.9])
        benchmark = np.array([20.0, 20.2, 20.1, 20.5, 20.8, 20.7, 21.0, 21.3, 21.2])
        peers = np.column_stack([
            np.array([9.8, 10.0, 10.1, 10.4, 10.5, 10.3, 10.7, 11.0, 10.9]),
            np.array([13.0, 13.3, 13.1, 13.6, 14.0, 13.8, 14.2, 14.5, 14.3]),
            np.array([7.0, 7.1, 7.0, 7.2, 7.5, 7.4, 7.6, 7.8, 7.7]),
        ])
        ref = compute_relative_strength_state(
            close, benchmark_close=benchmark, peer_close_matrix=peers, period=4, use_numba=False
        )
        fast = compute_relative_strength_state(
            close, benchmark_close=benchmark, peer_close_matrix=peers, period=4, use_numba=True
        )
        for key in ref:
            np.testing.assert_allclose(fast[key], ref[key], equal_nan=True, rtol=0, atol=1e-10)
        self.assertTrue(np.isnan(ref["relative_strength_z"][0]))
        self.assertLessEqual(np.max(np.abs(ref["relative_strength_z"][1:])), 6.0)
        self.assertLessEqual(np.max(np.abs(ref["beta_adjusted_momentum"][1:])), 6.0)
        self.assertTrue(np.all((ref["breadth_context"][1:] >= -1.0) & (ref["breadth_context"][1:] <= 1.0)))

    def test_relative_strength_bundle_and_beta_sector_normalization_validate(self):
        close = np.array([10.0, 10.4, 10.2, 10.8, 11.3, 11.0, 11.6, 12.1, 11.9])
        benchmark = np.array([20.0, 20.2, 20.1, 20.5, 20.8, 20.7, 21.0, 21.3, 21.2])
        peers = np.column_stack([
            np.array([9.8, 10.0, 10.1, 10.4, 10.5, 10.3, 10.7, 11.0, 10.9]),
            np.array([13.0, 13.3, 13.1, 13.6, 14.0, 13.8, 14.2, 14.5, 14.3]),
        ])
        bundle = build_relative_strength_bundle(
            close, benchmark_close=benchmark, peer_close_matrix=peers, period=4, use_numba=False
        )
        validate_indicator_state_bundle(bundle)
        matrix = bundle.feature_matrix(("relative_strength_z", "breadth_context", "beta_adjusted_momentum"))
        normalized = normalize_indicator_feature_transport(
            matrix,
            beta=np.linspace(0.8, 1.6, len(close)),
            sector_scale=np.linspace(0.9, 1.2, len(close)),
            rolling_window=4,
        )
        self.assertEqual(normalized.shape, matrix.shape)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertLessEqual(np.max(np.abs(normalized)), 6.0)


if __name__ == "__main__":
    unittest.main()
