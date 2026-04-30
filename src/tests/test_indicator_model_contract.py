import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.indicator_state import (
    INDICATOR_CHANNELS,
    build_atr_supertrend_bundle,
    build_heikin_ashi_bundle,
    build_empty_indicator_state,
    channels_for_specs,
    compute_atr_supertrend_state,
    compute_heikin_ashi_drift_signal,
    compute_heikin_ashi_state,
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


if __name__ == "__main__":
    unittest.main()
