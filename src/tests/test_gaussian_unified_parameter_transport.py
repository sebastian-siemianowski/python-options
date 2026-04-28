import json
import os
import sys
import tempfile
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from decision.signal_modules.parameter_loading import _load_tuned_kalman_params
from models.gaussian import GaussianUnifiedConfig


class GaussianUnifiedParameterTransportTest(unittest.TestCase):
    def test_config_to_dict_keeps_asset_adaptive_calibration_fields(self):
        cfg = GaussianUnifiedConfig(
            q=np.float64(3e-6),
            c=np.float64(1.4),
            phi=np.float64(0.72),
            variance_inflation=np.float64(1.18),
            mu_drift=np.float64(0.00033),
            chisq_ewm_lambda=np.float64(0.94),
            pit_var_lambda=np.float64(0.92),
            pit_var_dz_lo=np.float64(0.21),
            pit_var_dz_hi=np.float64(0.46),
            leverage_dynamic_decay=np.float64(0.35),
            liq_stress_coeff=np.float64(0.12),
            entropy_sigma_lambda=np.float64(0.08),
            gas_q_alpha=np.float64(0.05),
        )
        serialized = cfg.to_dict()

        for key in (
            "chisq_ewm_lambda",
            "pit_var_lambda",
            "pit_var_dz_lo",
            "pit_var_dz_hi",
            "leverage_dynamic_decay",
            "liq_stress_coeff",
            "entropy_sigma_lambda",
            "gas_q_alpha",
        ):
            self.assertIn(key, serialized)
            self.assertNotIsInstance(serialized[key], np.generic)

        self.assertNotIn("momentum_lookbacks", serialized)
        self.assertAlmostEqual(serialized["chisq_ewm_lambda"], 0.94)
        self.assertAlmostEqual(serialized["pit_var_dz_hi"], 0.46)

    def test_signal_parameter_loader_preserves_gaussian_calibration_fields(self):
        symbol = "ZZZTRANSPORT"
        best_params = {
            "q": 3e-6,
            "c": 1.4,
            "phi": 0.72,
            "fit_success": True,
            "gaussian_unified": True,
            "model_type": "gaussian_unified",
            "variance_inflation": 1.18,
            "mu_drift": 0.00033,
            "chisq_ewm_lambda": 0.94,
            "pit_var_lambda": 0.92,
            "pit_var_dz_lo": 0.21,
            "pit_var_dz_hi": 0.46,
            "leverage_dynamic_decay": 0.35,
            "liq_stress_coeff": 0.12,
            "entropy_sigma_lambda": 0.08,
            "gas_q_alpha": 0.05,
            "gas_q_beta": 0.8,
            "gas_q_omega": 1e-7,
        }
        cache_payload = {
            symbol: {
                "global": {
                    "model_posterior": {"kalman_phi_gaussian_unified": 1.0},
                    "models": {"kalman_phi_gaussian_unified": best_params},
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
            ("leverage_dynamic_decay", 0.35),
            ("liq_stress_coeff", 0.12),
            ("entropy_sigma_lambda", 0.08),
        ):
            self.assertAlmostEqual(float(loaded[key]), expected)


if __name__ == "__main__":
    unittest.main()
