import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.model_registry import (
    extract_model_params_for_sampling,
    make_gaussian_unified_name,
    make_unified_student_t_improved_name,
)


class ModelRegistryParameterTransportTest(unittest.TestCase):
    def test_unified_improved_sampling_extraction_keeps_calibration_fields(self):
        params = {
            "q": np.float64(2e-6),
            "c": np.float64(1.2),
            "phi": np.float64(0.63),
            "nu": np.float64(7.5),
            "gamma_vov": np.float64(0.41),
            "alpha_asym": np.float64(-0.06),
            "variance_inflation": np.float64(1.19),
            "mu_drift": np.float64(0.00037),
            "risk_premium_sensitivity": np.float64(0.22),
            "skew_score_sensitivity": np.float64(0.018),
            "leverage_dynamic_decay": np.float64(0.28),
            "liq_stress_coeff": np.float64(0.09),
            "entropy_sigma_lambda": np.float64(0.07),
            "calibrated_nu_crps": np.float64(9.0),
        }
        extracted = extract_model_params_for_sampling(
            make_unified_student_t_improved_name(8), params)

        for key in (
            "variance_inflation",
            "mu_drift",
            "risk_premium_sensitivity",
            "skew_score_sensitivity",
            "leverage_dynamic_decay",
            "liq_stress_coeff",
            "entropy_sigma_lambda",
            "calibrated_nu_crps",
        ):
            self.assertIn(key, extracted)
            self.assertNotIsInstance(extracted[key], np.generic)
        self.assertAlmostEqual(extracted["nu"], 7.5)

    def test_gaussian_unified_sampling_extraction_keeps_pit_profile_fields(self):
        params = {
            "q": 4e-6,
            "c": 1.1,
            "phi": 0.91,
            "variance_inflation": 1.11,
            "chisq_ewm_lambda": 0.94,
            "pit_var_lambda": 0.92,
            "pit_var_dz_lo": 0.21,
            "pit_var_dz_hi": 0.46,
            "gas_q_alpha": 0.03,
            "gas_q_beta": 0.81,
        }
        extracted = extract_model_params_for_sampling(
            make_gaussian_unified_name(True), params)

        for key, expected in (
            ("chisq_ewm_lambda", 0.94),
            ("pit_var_lambda", 0.92),
            ("pit_var_dz_lo", 0.21),
            ("pit_var_dz_hi", 0.46),
            ("gas_q_alpha", 0.03),
            ("gas_q_beta", 0.81),
        ):
            self.assertIn(key, extracted)
            self.assertAlmostEqual(float(extracted[key]), expected)


if __name__ == "__main__":
    unittest.main()
