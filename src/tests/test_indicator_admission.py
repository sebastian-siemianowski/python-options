import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.indicator_admission import (
    assert_candidate_controls_present,
    create_indicator_candidate_family,
    decide_indicator_admission,
    indicator_param_names_for_features,
    prune_correlated_indicator_columns,
    summarize_admission_decisions,
)
from models.model_registry import (
    make_gaussian_unified_name,
    make_student_t_improved_name,
    make_student_t_name,
    make_unified_student_t_improved_name,
)


class IndicatorAdmissionTest(unittest.TestCase):
    def test_param_names_follow_registered_channels(self):
        params = indicator_param_names_for_features(("heikin_ashi_state",))
        self.assertIn("ind_mean_weight", params)
        self.assertIn("ind_variance_weight", params)
        self.assertIn("ind_tail_weight", params)
        self.assertIn("ind_q_weight", params)

    def test_candidate_family_requires_controls_and_features(self):
        specs = create_indicator_candidate_family(
            (make_student_t_name(8),),
            "heikin_ashi",
            ("heikin_ashi_state",),
            extra_param_names=("ind_ha_drift_weight",),
        )
        self.assertEqual(len(specs), 1)
        self.assertTrue(specs[0].is_indicator_integrated)
        self.assertEqual(specs[0].base_model_name, make_student_t_name(8))
        assert_candidate_controls_present(specs, [make_student_t_name(8), specs[0].name])
        with self.assertRaises(AssertionError):
            assert_candidate_controls_present(specs, [specs[0].name])

    def test_student_t_improved_unified_and_gaussian_candidate_specs(self):
        specs = create_indicator_candidate_family(
            (
                make_student_t_improved_name(8),
                make_unified_student_t_improved_name(8),
                make_gaussian_unified_name(True),
            ),
            "momentum_volume",
            ("macd_acceleration_state", "volume_flow_state"),
        )
        self.assertEqual(len(specs), 3)
        for spec in specs:
            self.assertTrue(spec.is_indicator_integrated)
            self.assertIn("ind_mean_weight", spec.param_names)
            self.assertIn("ind_confidence_weight", spec.param_names)
            self.assertIn("ind_variance_weight", spec.param_names)

    def test_admission_requires_bic_and_lfo_support(self):
        accepted = decide_indicator_admission(
            "candidate",
            "control",
            candidate_bic=90.0,
            control_bic=100.0,
            candidate_lfo=-1.1,
            control_lfo=-1.2,
            max_bic_delta=-1.0,
            min_lfo_delta=0.0,
        )
        rejected = decide_indicator_admission(
            "candidate",
            "control",
            candidate_bic=101.0,
            control_bic=100.0,
            candidate_lfo=-1.0,
            control_lfo=-1.2,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.bic_delta, -10.0)
        self.assertFalse(rejected.accepted)
        self.assertEqual(summarize_admission_decisions([accepted, rejected])["accepted"], 1)

    def test_correlated_feature_pruning_keeps_ordered_low_redundancy_subset(self):
        base = np.linspace(-1.0, 1.0, 20)
        matrix = np.column_stack([base, base * 1.000001, np.sin(base * 3.0), np.ones_like(base)])
        kept = prune_correlated_indicator_columns(
            matrix,
            ("base", "duplicate", "nonlinear", "constant"),
            max_abs_corr=0.97,
        )
        self.assertEqual(kept, ("base", "nonlinear"))


if __name__ == "__main__":
    unittest.main()
