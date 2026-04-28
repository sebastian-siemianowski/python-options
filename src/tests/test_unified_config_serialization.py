import os
import sys
import unittest
import json
import tempfile
from dataclasses import fields

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.phi_student_t_unified import UnifiedStudentTConfig
from models.phi_student_t_unified_improved import UnifiedStudentTConfig as ImprovedUnifiedStudentTConfig
from decision.signal_modules.parameter_loading import _load_tuned_kalman_params


IMPORTANT_SIGNAL_FIELDS = (
    "variance_inflation",
    "mu_drift",
    "risk_premium_sensitivity",
    "skew_score_sensitivity",
    "skew_persistence",
    "crps_ewm_lambda",
    "rho_leverage",
    "kappa_mean_rev",
    "theta_long_var",
    "crps_sigma_shrinkage",
    "sigma_eta",
    "t_df_asym",
    "regime_switch_prob",
    "garch_kalman_weight",
    "q_vol_coupling",
    "loc_bias_var_coeff",
    "loc_bias_drift_coeff",
    "leverage_dynamic_decay",
    "liq_stress_coeff",
    "entropy_sigma_lambda",
    "calibrated_gw",
    "calibrated_nu_pit",
    "calibrated_nu_crps",
    "calibrated_beta_probit_corr",
    "calibrated_lambda_rho",
    "chisq_ewm_lambda",
    "pit_var_lambda",
    "pit_var_dz_lo",
    "pit_var_dz_hi",
)


def _rich_config(config_cls):
    return config_cls(
        q=np.float64(2.5e-6),
        c=np.float64(1.35),
        phi=np.float64(0.55),
        nu_base=np.float64(7.0),
        alpha_asym=np.float64(-0.08),
        k_asym=np.float64(1.35),
        q_stress_ratio=np.float64(8.0),
        ms_sensitivity=np.float64(2.6),
        ms_ewm_lambda=np.float64(0.96),
        gamma_vov=np.float64(0.42),
        vov_damping=np.float64(0.22),
        vov_window=17,
        risk_premium_sensitivity=np.float64(0.31),
        skew_score_sensitivity=np.float64(0.017),
        skew_persistence=np.float64(0.94),
        variance_inflation=np.float64(1.23),
        mu_drift=np.float64(0.00041),
        garch_omega=np.float64(1e-7),
        garch_alpha=np.float64(0.07),
        garch_beta=np.float64(0.88),
        garch_leverage=np.float64(0.04),
        garch_unconditional_var=np.float64(0.00021),
        rough_hurst=np.float64(0.18),
        jump_intensity=np.float64(0.035),
        jump_variance=np.float64(0.002),
        jump_sensitivity=np.float64(1.7),
        jump_mean=np.float64(-0.006),
        crps_ewm_lambda=np.float64(0.93),
        rho_leverage=np.float64(0.6),
        kappa_mean_rev=np.float64(0.08),
        theta_long_var=np.float64(0.00019),
        crps_sigma_shrinkage=np.float64(0.86),
        sigma_eta=np.float64(0.12),
        t_df_asym=np.float64(1.3),
        regime_switch_prob=np.float64(0.06),
        garch_kalman_weight=np.float64(0.38),
        q_vol_coupling=np.float64(0.24),
        loc_bias_var_coeff=np.float64(0.08),
        loc_bias_drift_coeff=np.float64(-0.05),
        leverage_dynamic_decay=np.float64(0.3),
        liq_stress_coeff=np.float64(0.11),
        entropy_sigma_lambda=np.float64(0.09),
        calibrated_gw=np.float64(0.63),
        calibrated_nu_pit=np.float64(5.5),
        calibrated_nu_crps=np.float64(8.5),
        calibrated_beta_probit_corr=np.float64(1.08),
        calibrated_lambda_rho=np.float64(0.977),
        chisq_ewm_lambda=np.float64(0.965),
        pit_var_lambda=np.float64(0.955),
        pit_var_dz_lo=np.float64(0.26),
        pit_var_dz_hi=np.float64(0.52),
        c_min=np.float64(0.02),
        c_max=np.float64(12.0),
        q_min=np.float64(1e-7),
        exogenous_input=np.ones(5),
    )


class UnifiedConfigSerializationTest(unittest.TestCase):
    def _assert_complete_scalar_serialization(self, config):
        serialized = config.to_dict()
        expected = {f.name for f in fields(config) if f.name != "exogenous_input"}

        self.assertEqual(expected, set(serialized))
        self.assertNotIn("exogenous_input", serialized)
        for key, value in serialized.items():
            self.assertIsInstance(value, (int, float, bool, str, type(None)), key)
            self.assertNotIsInstance(value, np.generic, key)

        for key in IMPORTANT_SIGNAL_FIELDS:
            self.assertIn(key, serialized)

        self.assertAlmostEqual(serialized["variance_inflation"], 1.23)
        self.assertAlmostEqual(serialized["mu_drift"], 0.00041)
        self.assertAlmostEqual(serialized["risk_premium_sensitivity"], 0.31)
        self.assertAlmostEqual(serialized["skew_score_sensitivity"], 0.017)
        self.assertEqual(serialized["vov_window"], 17)

    def test_unified_config_to_dict_keeps_all_signal_parameters(self):
        self._assert_complete_scalar_serialization(_rich_config(UnifiedStudentTConfig))

    def test_improved_unified_config_to_dict_keeps_all_signal_parameters(self):
        self._assert_complete_scalar_serialization(_rich_config(ImprovedUnifiedStudentTConfig))

    def test_signal_loader_preserves_unified_asset_adaptive_pit_profile(self):
        symbol = "ZZZUNIFIED"
        best_params = {
            "q": 2e-6,
            "c": 1.2,
            "phi": 0.63,
            "nu": 8.0,
            "fit_success": True,
            "unified_model": True,
            "model_type": "phi_student_t_unified",
            "chisq_ewm_lambda": 0.94,
            "pit_var_lambda": 0.92,
            "pit_var_dz_lo": 0.21,
            "pit_var_dz_hi": 0.46,
        }
        cache_payload = {
            symbol: {
                "global": {
                    "model_posterior": {"phi_student_t_unified_nu_8": 1.0},
                    "models": {"phi_student_t_unified_nu_8": best_params},
                },
                "regime": {},
                "meta": {},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f)
            loaded = _load_tuned_kalman_params(symbol, cache_path=cache_path)

        self.assertIsNotNone(loaded)
        for key, expected in (
            ("chisq_ewm_lambda", 0.94),
            ("pit_var_lambda", 0.92),
            ("pit_var_dz_lo", 0.21),
            ("pit_var_dz_hi", 0.46),
        ):
            self.assertAlmostEqual(float(loaded[key]), expected)


if __name__ == "__main__":
    unittest.main()
