import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tuning.tune_q_mle import kalman_filter_drift, compute_pit_ks_pvalue

class TestKalmanFilter(unittest.TestCase):
    def test_constant_drift_convergence(self):
        n_obs = 5000
        true_mu = 0.0005
        true_sigma = 0.01
        np.random.seed(42)
        returns = np.random.normal(loc=true_mu, scale=true_sigma, size=n_obs)
        vol = np.full(n_obs, true_sigma)

        q = 1e-8
        c = 1.0

        mu_filtered, _, _ = kalman_filter_drift(returns, vol, q, c)

        # Check if the drift converges to the true mean
        converged_drift = mu_filtered[-100:].mean()
        self.assertAlmostEqual(converged_drift, true_mu, delta=0.0002)

    def test_zero_process_noise(self):
        n_obs = 1000
        true_mu = 0.0005
        true_sigma = 0.01
        np.random.seed(42)
        returns = np.random.normal(loc=true_mu, scale=true_sigma, size=n_obs)
        vol = np.full(n_obs, true_sigma)

        q = 0.0
        c = 1.0

        mu_filtered, _, _ = kalman_filter_drift(returns, vol, q, c)

        # With q=0, the drift should converge to a constant
        converged_std = mu_filtered[-100:].std()
        self.assertAlmostEqual(converged_std, 0.0, delta=3e-5)

    def test_high_process_noise(self):
        n_obs = 1000
        true_mu = 0.0
        true_sigma = 0.01
        returns = np.random.normal(loc=true_mu, scale=true_sigma, size=n_obs)
        vol = np.full(n_obs, true_sigma)

        q = 1e-2  # High process noise
        c = 1.0

        mu_filtered, _, _ = kalman_filter_drift(returns, vol, q, c)

        # With high q, the filtered drift should track returns closely
        correlation = np.corrcoef(mu_filtered, returns)[0, 1]
        self.assertGreater(correlation, 0.8)

    def test_pit_uniformity(self):
        n_obs = 2000
        true_mu = 0.0
        true_sigma = 0.01
        returns = np.random.normal(loc=true_mu, scale=true_sigma, size=n_obs)
        vol = np.full(n_obs, true_sigma)

        q = 1e-8
        c = 1.0

        mu_filtered, P_filtered, _ = kalman_filter_drift(returns, vol, q, c)

        _, p_value = compute_pit_ks_pvalue(returns, mu_filtered, vol, P_filtered, c)

        self.assertGreater(p_value, 0.05)

if __name__ == '__main__':
    unittest.main()
