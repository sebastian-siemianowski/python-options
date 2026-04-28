import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from decision.signals_calibration import (  # noqa: E402
    BETA_IDENTITY,
    EMOS_IDENTITY,
    HorizonData,
    _evaluate_metrics,
)


class SignalsCalibrationMetricsTest(unittest.TestCase):
    def test_calibrated_hit_rate_and_magnitude_use_calibrated_outputs(self):
        data = HorizonData(
            H=7,
            n_eval=4,
            predicted=np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            actual=np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64),
            p_ups=np.array([0.4, 0.4, 0.6, 0.6], dtype=np.float64),
            actual_ups=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
            sigma_pred=np.ones(4, dtype=np.float64),
            vol_regime=np.ones(4, dtype=np.float64),
            weights=np.ones(4, dtype=np.float64),
            nu_hat=np.full(4, 30.0, dtype=np.float64),
        )

        raw = _evaluate_metrics(
            data,
            {"type": "beta", **BETA_IDENTITY},
            {"type": "emos", **EMOS_IDENTITY},
            full=False,
        )
        calibrated = _evaluate_metrics(
            data,
            {"type": "isotonic", "x": [0.0, 1.0], "y": [1.0, 0.0]},
            {"type": "emos", "a": 0.0, "b": 0.5, "c": 0.0, "d": 1.0},
            full=False,
        )

        self.assertEqual(raw["hit_rate"], 0.0)
        self.assertEqual(calibrated["hit_rate"], 1.0)
        self.assertLess(calibrated["mag_ratio"], raw["mag_ratio"])


if __name__ == "__main__":
    unittest.main()
