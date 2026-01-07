import json
import os
import unittest
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tuning.tune_q_mle import save_cache, load_cache


class TestCache(unittest.TestCase):
    def setUp(self):
        self.cache_json = 'test_cache.json'
        self.cache_csv = 'test_cache.csv'

    def tearDown(self):
        if os.path.exists(self.cache_json):
            os.remove(self.cache_json)
        if os.path.exists(self.cache_csv):
            os.remove(self.cache_csv)

    def test_save_and_load_cache(self):
        cache_data = {
            'asset1': {
                'q': 0.1,
                'c': 0.9,
                'nu': 5.0,
                'log_likelihood': -100.0,
                'aic': 206.0,
                'bic': 212.0,
                'pit_ks_pvalue': 0.5,
                'calibration_warning': False,
                'cv_variance_warning': False,
                'timestamp': '2024-01-01T00:00:00Z',
            },
            'asset2': {
                'q': 0.2,
                'c': 0.8,
                'nu': None,
                'log_likelihood': -200.0,
                'aic': 404.0,
                'bic': 408.0,
                'pit_ks_pvalue': 0.04,
                'calibration_warning': True,
                'cv_variance_warning': True,
                'timestamp': '2024-01-02T00:00:00Z',
            },
        }

        save_cache(cache_data, self.cache_json, self.cache_csv)
        loaded_cache = load_cache(self.cache_json)

        self.assertEqual(cache_data, loaded_cache)

        with open(self.cache_json, 'r') as f:
            json_content = f.read()
            self.assertIn('"c": 0.9,', json_content)
            self.assertIn('"q": 0.1,', json_content)

        self.assertTrue(os.path.exists(self.cache_csv))
        df = pd.read_csv(self.cache_csv)
        self.assertEqual(len(df), 2)


if __name__ == '__main__':
    unittest.main()
