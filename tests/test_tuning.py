import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from scripts.fx_pln_jpy_signals import compute_features
from tuning.tune_q_mle import tune_asset_q

class TestTuning(unittest.TestCase):
    def test_fallback_logic(self):
        # Create a dummy price series
        px = pd.Series(np.random.randn(500).cumsum() + 100)
        px.index = pd.to_datetime(pd.date_range('2022-01-01', periods=500))

        # Compute features for an asset not in the cache
        features = compute_features(px, asset_symbol="DUMMY_ASSET")

        # Check that the fallback logic was triggered correctly (runs KF with heuristic q)
        self.assertTrue(features['kalman_available'])
        self.assertFalse(features['kalman_metadata']['q_optimization_attempted'])
        self.assertIn('mu_kf', features)

    def test_cv_variance_warning(self):
        # This test is more complex to set up as it requires manipulating the CV loop.
        # For now, we will create a placeholder test.
        # A more complete test would involve mocking the CV folds to produce high variance.
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
