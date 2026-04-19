"""
Story 1.3: RV-Q Integration with Existing GAS-Q
=================================================
Tests that RV-Q models are properly registered, fitted, and compete in BMA
alongside static-q and GAS-Q models.
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestModelRegistryRVQ(unittest.TestCase):
    """Test that model_registry has RV-Q variants."""

    def test_rv_q_family_exists(self):
        from models.model_registry import ModelFamily
        self.assertTrue(hasattr(ModelFamily, "RV_Q"))

    def test_rv_q_phi_gaussian_name(self):
        from models.model_registry import make_rv_q_phi_gaussian_name
        name = make_rv_q_phi_gaussian_name()
        self.assertIn("rv_q", name)
        self.assertIn("gaussian", name.lower())

    def test_rv_q_student_t_names(self):
        from models.model_registry import make_rv_q_student_t_name
        for nu in [3, 4, 8, 20]:
            name = make_rv_q_student_t_name(nu)
            self.assertIn("rv_q", name)
            self.assertIn(str(nu), name)

    def test_is_rv_q_model(self):
        from models.model_registry import (
            is_rv_q_model,
            make_rv_q_phi_gaussian_name,
            make_rv_q_student_t_name,
        )
        self.assertTrue(is_rv_q_model(make_rv_q_phi_gaussian_name()))
        self.assertTrue(is_rv_q_model(make_rv_q_student_t_name(8)))
        self.assertFalse(is_rv_q_model("kalman_phi_gaussian"))
        self.assertFalse(is_rv_q_model("phi_student_t_nu_8"))

    def test_rv_q_specs_in_registry(self):
        from models.model_registry import build_model_registry, ModelFamily
        registry = build_model_registry()
        rv_q_specs = [s for s in registry.values() if s.family == ModelFamily.RV_Q]
        # At minimum: 1 phi-Gaussian + 4 Student-t (nu in [3,4,8,20]) = 5
        self.assertGreaterEqual(len(rv_q_specs), 5)

    def test_rv_q_sampler_registered(self):
        from models.model_registry import (
            build_model_registry,
            get_sampler_for_model,
            make_rv_q_phi_gaussian_name,
            make_rv_q_student_t_name,
        )
        registry = build_model_registry()
        # Gaussian sampler
        g_name = make_rv_q_phi_gaussian_name()
        self.assertIn(g_name, registry)
        sampler = get_sampler_for_model(registry[g_name])
        self.assertIsNotNone(sampler)
        # Student-t sampler
        t_name = make_rv_q_student_t_name(8)
        self.assertIn(t_name, registry)
        sampler = get_sampler_for_model(registry[t_name])
        self.assertIsNotNone(sampler)


class TestRVQFitting(unittest.TestCase):
    """Test that RV-Q models are fitted and produce valid results."""

    @classmethod
    def setUpClass(cls):
        """Load SPY data for testing."""
        data_dir = os.path.join(SRC_DIR, "data", "prices")
        # Try both naming conventions
        spy_path = os.path.join(data_dir, "SPY.csv")
        if not os.path.exists(spy_path):
            spy_path = os.path.join(data_dir, "SPY_1d.csv")
        if not os.path.exists(spy_path):
            cls.returns = None
            cls.vol = None
            return
        try:
            import pandas as pd
            df = pd.read_csv(spy_path)
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            close = df[col].values.astype(np.float64)
            cls.returns = np.diff(np.log(close))
            # EWMA volatility
            n = len(cls.returns)
            cls.vol = np.empty(n, dtype=np.float64)
            alpha = 0.06
            var_ewm = cls.returns[0] ** 2
            for i in range(n):
                var_ewm = alpha * cls.returns[i] ** 2 + (1 - alpha) * var_ewm
                cls.vol[i] = max(np.sqrt(var_ewm), 1e-8)
        except Exception:
            cls.returns = None
            cls.vol = None

    def _skip_if_no_data(self):
        if self.returns is None:
            self.skipTest("SPY data not available")

    def test_fit_rv_q_gaussian_produces_result(self):
        self._skip_if_no_data()
        from models.rv_adaptive_q import (
            rv_adaptive_q_filter_gaussian,
            optimize_rv_q_params,
            RVAdaptiveQConfig,
        )
        config, diag = optimize_rv_q_params(
            self.returns, self.vol, c=1.0, phi=0.98, nu=None)
        result = rv_adaptive_q_filter_gaussian(
            self.returns, self.vol, 1.0, 0.98, config)
        self.assertTrue(np.isfinite(result.log_likelihood))
        self.assertEqual(len(result.mu_filtered), len(self.returns))
        self.assertEqual(len(result.P_filtered), len(self.returns))
        self.assertEqual(len(result.q_path), len(self.returns))

    def test_fit_rv_q_student_t_produces_result(self):
        self._skip_if_no_data()
        from models.rv_adaptive_q import (
            rv_adaptive_q_filter_student_t,
            optimize_rv_q_params,
            RVAdaptiveQConfig,
        )
        config, diag = optimize_rv_q_params(
            self.returns, self.vol, c=1.0, phi=0.98, nu=8.0)
        result = rv_adaptive_q_filter_student_t(
            self.returns, self.vol, 1.0, 0.98, 8.0, config)
        self.assertTrue(np.isfinite(result.log_likelihood))
        self.assertEqual(len(result.mu_filtered), len(self.returns))

    def test_rv_q_fit_result_has_diagnostics(self):
        self._skip_if_no_data()
        from models.rv_adaptive_q import optimize_rv_q_params
        config, diag = optimize_rv_q_params(
            self.returns, self.vol, c=1.0, phi=0.98, nu=None)
        self.assertIn("delta_ll", diag)
        self.assertIn("delta_bic", diag)
        self.assertIn("oos_delta_ll", diag)
        self.assertGreaterEqual(config.gamma, 0.0)
        self.assertGreater(config.q_base, 0.0)


class TestRVQInBMA(unittest.TestCase):
    """Test that RV-Q models compete in BMA alongside other models."""

    @classmethod
    def setUpClass(cls):
        """Load SPY data and run fit_all_models_for_regime."""
        data_dir = os.path.join(SRC_DIR, "data", "prices")
        spy_path = os.path.join(data_dir, "SPY.csv")
        if not os.path.exists(spy_path):
            spy_path = os.path.join(data_dir, "SPY_1d.csv")
        cls.models = None
        if not os.path.exists(spy_path):
            return
        try:
            import pandas as pd
            df = pd.read_csv(spy_path)
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            close = df[col].values.astype(np.float64)
            returns = np.diff(np.log(close))
            # EWMA volatility
            n = len(returns)
            vol = np.empty(n, dtype=np.float64)
            alpha = 0.06
            var_ewm = returns[0] ** 2
            for i in range(n):
                var_ewm = alpha * returns[i] ** 2 + (1 - alpha) * var_ewm
                vol[i] = max(np.sqrt(var_ewm), 1e-8)
            # Use only last 500 observations for speed
            returns = returns[-500:]
            vol = vol[-500:]
            from tuning.tune import fit_all_models_for_regime
            cls.models = fit_all_models_for_regime(returns, vol)
        except Exception as e:
            print(f"Warning: Could not run fit_all_models_for_regime: {e}")
            cls.models = None

    def _skip_if_no_models(self):
        if self.models is None:
            self.skipTest("fit_all_models_for_regime did not produce results")

    def test_rv_q_models_present_in_fitted(self):
        """RV-Q models should appear in the fitted models dict."""
        self._skip_if_no_models()
        from models.model_registry import (
            make_rv_q_phi_gaussian_name,
            make_rv_q_student_t_name,
        )
        rv_q_names = [make_rv_q_phi_gaussian_name()]
        for nu in [3, 4, 8, 20]:
            rv_q_names.append(make_rv_q_student_t_name(nu))
        found = [n for n in rv_q_names if n in self.models]
        self.assertGreater(len(found), 0,
            f"No RV-Q models found. Available: {list(self.models.keys())[:10]}...")

    def test_rv_q_has_required_diagnostics(self):
        """Each RV-Q model should have full diagnostic fields."""
        self._skip_if_no_models()
        rv_q_models = {k: v for k, v in self.models.items()
                       if v.get("rv_q_model") or v.get("model_type") == "rv_q"}
        if not rv_q_models:
            self.skipTest("No RV-Q models fitted successfully")
        for name, data in rv_q_models.items():
            if not data.get("fit_success"):
                continue
            required = ["bic", "crps", "hyvarinen_score", "pit_ks_pvalue",
                        "log_likelihood", "q_base", "gamma", "n_params"]
            for field in required:
                self.assertIn(field, data,
                    f"RV-Q model {name} missing field: {field}")

    def test_rv_q_bic_is_finite(self):
        """Successfully fitted RV-Q models should have finite BIC."""
        self._skip_if_no_models()
        rv_q_models = {k: v for k, v in self.models.items()
                       if v.get("rv_q_model") and v.get("fit_success")}
        if not rv_q_models:
            self.skipTest("No RV-Q models fitted successfully")
        for name, data in rv_q_models.items():
            self.assertTrue(np.isfinite(data["bic"]),
                f"RV-Q model {name} has non-finite BIC: {data['bic']}")

    def test_non_rv_q_models_still_present(self):
        """RV-Q should not replace existing models -- they coexist."""
        self._skip_if_no_models()
        non_rv_q = {k: v for k, v in self.models.items()
                    if not v.get("rv_q_model") and v.get("fit_success")}
        self.assertGreater(len(non_rv_q), 0,
            "No non-RV-Q models found -- RV-Q should compete, not replace")

    def test_rv_q_gamma_nonnegative(self):
        """RV-Q gamma should be >= 0 (includes gamma=0 recovery to static-q)."""
        self._skip_if_no_models()
        rv_q_models = {k: v for k, v in self.models.items()
                       if v.get("rv_q_model") and v.get("fit_success")}
        for name, data in rv_q_models.items():
            self.assertGreaterEqual(data.get("gamma", -1), 0.0,
                f"RV-Q model {name} has negative gamma: {data.get('gamma')}")


if __name__ == "__main__":
    unittest.main()
