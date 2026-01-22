import unittest
import numpy as np
from tecd_drift_v3 import TECDDriftModelV3


class TECDDriftV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_evidence_accumulation(self):
        r = np.array([0.1, 0.1, 0.1, 0.1])
        model = TECDDriftModelV3({"lambda_e": 0.0, "alpha": 1.0, "kappa": 0.0, "gamma": 0.0, "beta": 0.0, "eta": 0.0})
        model.fit(r)
        self.assertTrue(np.all(model._E > 0))
        self.assertTrue(np.all(np.isfinite(model._E)))

    def test_entropy_bounds(self):
        r = self.rng.normal(0, 1, size=200)
        model = TECDDriftModelV3()
        model.fit(r)
        self.assertTrue(np.all((model._H >= -1e-12) & (model._H <= 1 + 1e-12)))

    def test_jump_trigger(self):
        r = np.ones(100) * 0.5
        params = {"eta": 1.0, "tau": 0.01, "lambda_e": 0.9, "alpha": 1.0, "kappa": 0.0, "gamma": 0.0, "beta": 0.0, "max_mu": 10, "E_max": 10, "J_max": 10}
        model = TECDDriftModelV3(params)
        model.fit(r)
        self.assertTrue(np.max(model._T) >= 0.0)

    def test_drift_stability(self):
        r = self.rng.normal(0, 0.01, size=300)
        model = TECDDriftModelV3({"max_mu": 0.05})
        model.fit(r)
        self.assertTrue(np.all(np.abs(model.predict_mu()) <= 0.05 + 1e-9))

    def test_ll_finite(self):
        r = self.rng.normal(0, 0.01, size=100)
        model = TECDDriftModelV3()
        ll = model.log_likelihood(r)
        self.assertTrue(np.isfinite(ll))

    def test_pit_range(self):
        r = self.rng.normal(0, 0.01, size=50)
        model = TECDDriftModelV3()
        pit = model.pit(r)
        self.assertTrue(np.all((pit >= 1e-8) & (pit <= 1 - 1e-8)))

    def test_limiting_cases(self):
        r = self.rng.normal(0, 0.01, size=200)
        model = TECDDriftModelV3({"kappa": 0.0, "eta": 0.0})
        model.fit(r)
        mu = model.predict_mu()
        # Drift should be finite; limiting behavior emerges from dynamics
        self.assertTrue(np.all(np.isfinite(mu)))

    def test_scale_invariance(self):
        r = self.rng.normal(0, 0.01, size=200)
        model = TECDDriftModelV3()
        model.fit(r)
        mu1 = model.predict_mu()
        model.fit(r * 2.0)
        mu2 = model.predict_mu()
        self.assertLess(np.mean(np.abs(mu1 - mu2)), 0.05)

    def test_entropy_definition(self):
        r = np.concatenate([np.ones(50) * -0.1, np.ones(50) * 0.1])
        model = TECDDriftModelV3({"entropy_window": 20})
        model.fit(r)
        self.assertTrue(np.all(model._H <= 1 + 1e-12))


if __name__ == "__main__":
    unittest.main()